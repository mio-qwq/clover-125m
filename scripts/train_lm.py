"""English 125M pretraining. Actual updates require --start-training explicitly."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import time
import torch

try:
    from . import lm_common as c
    from .lm_data import PackedTokenDataset, EpochBlockSampler
    from .lm_progress import TrainingProgress, write_event
except ImportError:
    import lm_common as c
    from lm_data import PackedTokenDataset, EpochBlockSampler
    from lm_progress import TrainingProgress, write_event


def append_event(path, value, progress=None):
    return write_event(path, value, progress)


def train(config, out, resume=None):
    c.require(torch.cuda.is_available(), 'CUDA is required; refuse a silent CPU training run')
    c.require(torch.cuda.is_bf16_supported(), 'This recipe requires native BF16 support')
    print('Checking accepted dataset hashes and training-readiness receipts...', flush=True)
    root, manifest, fingerprint = c.approved_data(config, verify_hashes=True)
    c.require_readiness(config, fingerprint)
    t = config['training']
    dataset = PackedTokenDataset(root, 'train', t['sequence_length'])
    validation = PackedTokenDataset(root, 'validation', t['sequence_length'])
    total_budget = min(t['max_tokens'], dataset.total_effective_targets*t['max_epochs'])
    c.require(total_budget == dataset.total_effective_targets*t['max_epochs'],
              'This recipe uses complete passes; token budget must match the configured passes')
    blocks_per_update = t['micro_batch_size'] * t['gradient_accumulation_steps']
    total_updates = math.ceil(len(dataset)/blocks_per_update)*t['max_epochs']
    c.seed_all(t['seed'])
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    print('Preparing the model, optimizer and optional resume checkpoint...', flush=True)
    model = c.build_model(config).to(device)
    optimizer = c.make_optimizer(model, config)
    state = {'epoch': 0, 'consumed_blocks': 0, 'tokens': 0, 'update': 0,
             'total_updates': total_updates, 'total_budget': total_budget,
             'best_validation_loss': None, 'completed': False}
    if resume:
        payload = c.load_checkpoint(resume, model, optimizer, config, fingerprint)
        state = payload['training_state']
        c.require(state['total_updates'] == total_updates and state['total_budget'] == total_budget,
                  'Resume schedule/budget differs')
        c.require(not state['completed'], 'Checkpoint already completed its training budget')
    out = c.safe_new_output(out)
    c.write_new_json(out/'config.json', config)
    c.write_new_json(out/'run.json', {'started_at_utc': c.now(), 'mode': 'actual_pretraining',
        'dataset_fingerprint': fingerprint, 'runtime_versions': c.runtime_versions(),
        'resume_checkpoint': str(Path(resume).resolve()) if resume else None,
        'total_updates': total_updates, 'total_valid_tokens': total_budget,
        'eta_scope': 'Remaining training tokens, using this session wall time; observed validation/save overhead included; final validation/export excluded',
        'raw_data_access': 'none; only independently accepted prepared tokens are read'})
    events = out/'metrics.jsonl'
    completed_boundary = dict(state)
    last_saved_update = state['update'] if resume else -1
    starting_update = state['update']
    progress = TrainingProgress(total_budget, total_updates,
        initial_tokens=state['tokens'], initial_update=state['update'])
    def stage(phase, message):
        progress.set_phase(phase)
        append_event(events, {'kind': 'stage', 'phase': phase, 'update': state['update'],
                              'message': message}, progress)

    def checkpoint(name=None):
        stage('Saving checkpoint', 'Writing and hashing an immutable recovery checkpoint...')
        started = time.perf_counter()
        path = c.save_checkpoint(out, model, optimizer, config, fingerprint, state, name=name)
        append_event(events, {'kind': 'checkpoint', 'update': state['update'],
            'checkpoint': str(path), 'duration_seconds': time.perf_counter()-started}, progress)
        return path

    model.train()
    try:
        append_event(events, {'kind': 'started', **state, **progress.snapshot(),
            'message': f'Training budget {total_budget:,} tokens / {total_updates:,} updates; '
                       f'resuming at step {state["update"]:,}. ETA is approximate and excludes final validation/export.'}, progress)
        while state['epoch'] < t['max_epochs']:
            sampler = EpochBlockSampler(len(dataset), seed=t['seed'], epoch=state['epoch'],
                                        start=state['consumed_blocks'])
            iterator = iter(c.make_loader(dataset, config, sampler=sampler))
            while True:
                batches = list(itertools.islice(iterator, t['gradient_accumulation_steps']))
                if not batches:
                    break
                valid_tokens = sum(int(b['valid_tokens'].sum()) for b in batches)
                c.require(valid_tokens > 0, 'Empty parameter update')
                learning_rate = c.learning_rate(config, state['update'], total_updates)
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate
                optimizer.zero_grad(set_to_none=True)
                summed_loss = 0.
                for batch in batches:
                    x = batch['input_ids'].to(device, non_blocking=True)
                    y = batch['targets'].to(device, non_blocking=True)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        value = c.forward_loss_sum(model, x, y)
                    c.require(bool(torch.isfinite(value)), 'Nonfinite loss; stop without stepping')
                    (value/valid_tokens).backward()
                    summed_loss += value.detach().float().item()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t['gradient_clip'], error_if_nonfinite=True)
                # This is the only actual model update in the project.
                optimizer.step()
                state['consumed_blocks'] += sum(len(b['block_id']) for b in batches)
                state['tokens'] += valid_tokens
                state['update'] += 1
                completed_boundary = dict(state)
                c.require(state['tokens'] <= total_budget, 'Training exceeded approved token budget')
                telemetry = progress.advance(state['update'], state['tokens'], summed_loss/valid_tokens)
                if (state['update'] == starting_update+1
                    or state['update'] % t['log_every_updates'] == 0
                    or state['update'] == total_updates):
                    append_event(events, {'kind': 'train', **state, 'loss': summed_loss/valid_tokens,
                        'learning_rate': learning_rate, 'gradient_norm': float(norm),
                        **telemetry, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}, progress)
                full_validation = state['update'] % t['full_validation_every_updates'] == 0
                if full_validation or state['update'] % t['mini_validation_every_updates'] == 0:
                    stage('Full validation' if full_validation else 'Mini validation',
                          'Evaluating held-out validation tokens; training counters stay fixed...')
                    result = c.evaluate(model, validation, config, device,
                        max_tokens=None if full_validation else t['mini_validation_tokens'])
                    append_event(events, {'kind': 'full_validation' if full_validation else 'mini_validation',
                                          'update': state['update'], **result}, progress)
                    if full_validation and (state['best_validation_loss'] is None or result['loss'] < state['best_validation_loss']):
                        state['best_validation_loss'] = result['loss']
                        checkpoint(name=f"best_{state['update']:07d}.pt")
                        c.write_new_json(out/f"best_{state['update']:07d}_validation.json", result)
                        last_saved_update = state['update']
                    progress.set_phase('Train')
                if state['update'] == 1 or state['update'] % t['checkpoint_every_updates'] == 0:
                    checkpoint()
                    last_saved_update = state['update']
                    c.prune_checkpoints(out, t['keep_latest_checkpoints'])
                    progress.set_phase('Train')
                completed_boundary = dict(state)
            c.require(state['consumed_blocks'] == len(dataset), 'Epoch skipped or repeated blocks')
            state['epoch'] += 1
            state['consumed_blocks'] = 0
            completed_boundary = dict(state)
        c.require(state['tokens'] == total_budget and state['update'] == total_updates,
                  'Final training coverage/schedule mismatch')
        stage('Final validation', 'Training tokens are 100% complete; final validation and model export are still pending...')
        result = c.evaluate(model, validation, config, device)
        append_event(events, {'kind': 'final_validation', 'update': state['update'], **result}, progress)
        state['completed'] = True
        checkpoint(name='final.pt')
        export = out/'model'
        stage('Exporting model', 'Saving local safetensors weights and the tokenizer...')
        model.save_pretrained(export, safe_serialization=True)
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(root/'tokenizer/tokenizer.json'),
            pad_token='<|pad|>', eos_token='<|endoftext|>', unk_token='<|unk|>', bos_token='<|beginoftext|>')
        tokenizer.model_max_length = t['sequence_length']
        tokenizer.save_pretrained(export)
        c.write_new_json(out/'completed.json', {'status': 'completed', 'state': state, 'validation': result,
                          'finished_at_utc': c.now(), 'test_set_used_for_tuning': False})
        progress.set_phase('Completed')
        append_event(events, {'kind': 'completed', **state, **progress.snapshot(), 'export': str(export)}, progress)
    except BaseException as error:
        # No save during a partial update: the previous immutable checkpoint is
        # the recovery point. Never label mixed weights/counters as resumable.
        progress.set_phase('Stopped')
        append_event(events, {'kind': 'stopped', 'error': type(error).__name__, 'message': str(error),
                              'last_saved_update': last_saved_update, 'last_complete_boundary': completed_boundary}, progress)
        raise
    finally:
        progress.close()
        dataset.close()
        validation.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=c.DEFAULT_CONFIG)
    parser.add_argument('--start-training', action='store_true', help='Explicitly enable actual optimizer updates')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    config = c.load_config(args.config)
    if not args.start_training:
        print(json.dumps({'mode': 'configuration_only', 'parameters': c.theoretical_parameters(config['model']),
                          'training_started': False, 'message': 'Actual training requires --start-training.'}, indent=2))
        return
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
    output = args.output or c.resolve_path(config['training']['output_root'])/timestamp
    train(config, output, args.resume)


if __name__ == '__main__':
    main()
