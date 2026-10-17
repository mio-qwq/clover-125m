"""Full SFT preparation: configuration only unless --start-training is explicit.

Fresh SFT loads only approved base safetensors weights. Its Adam state, schedule,
and counters start from zero. Resume accepts only this SFT stage's immutable
checkpoints with identical configuration, data, base weights and training code.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import shutil
import time

import torch

try:
    from . import lm_common as c
    from .lm_progress import TrainingProgress, write_event
except ImportError:
    import lm_common as c
    from lm_progress import TrainingProgress, write_event

STAGE = 'full_sft'
STATE_SCHEMA = 1
DEFAULT_CONFIG = c.ROOT / 'configs/english_125m_sft.json'


def common():
    # Lazy loading keeps pure scheduling/numerical utilities independently usable.
    if __package__:
        from . import sft_common
    else:
        import sft_common
    return sft_common


def learning_rate(config, update, total_updates):
    """Fresh, stateless linear warmup followed by cosine decay to the floor."""
    t = config['training']
    c.require(total_updates > 0 and 0 <= update < total_updates, 'Invalid SFT schedule position')
    c.require(0 <= t['warmup_ratio'] < 1, 'Invalid warmup ratio')
    warmup = min(math.ceil(total_updates * t['warmup_ratio']), max(0, total_updates - 1))
    if update < warmup:
        return t['learning_rate'] * (update + 1) / warmup
    if total_updates == 1:
        return t['learning_rate']
    fraction = min(1., (update - warmup + 1) / max(1, total_updates - warmup))
    return t['minimum_learning_rate'] + (t['learning_rate'] - t['minimum_learning_rate']) * \
        .5 * (1 + math.cos(math.pi * fraction))


def backward_accumulated(model, batches, device, loss_function=None):
    """Backpropagate token-weighted assistant loss; NEVER update parameters.

All microbatches share the same denominator: the number of effective assistant
targets in the entire accumulated batch. Non-assistant/padding targets are -100.
The caller owns zero_grad, clipping, and the explicitly enabled optimizer step.
    """
    loss_function = loss_function or common().forward_loss_sum
    counts = [int((batch['targets'] != -100).sum().item()) for batch in batches]
    total = sum(counts)
    c.require(total > 0 and all(count > 0 for count in counts), 'Empty assistant-target update')
    summed_loss = 0.
    examples = 0
    for batch in batches:
        c.require(batch['targets'].dtype == torch.long, 'SFT targets must be int64')
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            value = loss_function(model, batch, device)
        c.require(value.ndim == 0 and bool(torch.isfinite(value)), 'Nonfinite assistant loss; refuse update')
        (value / total).backward()
        summed_loss += float(value.detach().float())
        examples += len(batch['targets'])
    return {'loss': summed_loss / total, 'tokens': total, 'examples': examples, 'batches': len(batches)}


def checkpoint_fingerprint(approved_fingerprint, code_receipts):
    return {'training_stage': STAGE, 'approved_inputs': approved_fingerprint, 'code_receipts': code_receipts}


def initial_state(plans, target_tokens):
    total_updates = sum(math.ceil(len(plan) / accumulation) for plan, accumulation in plans)
    return {'training_stage': STAGE, 'sft_state_schema': STATE_SCHEMA,
            'epoch': 0, 'consumed_batches': 0, 'tokens': 0, 'examples': 0, 'update': 0,
            'total_updates': total_updates, 'total_budget': target_tokens,
            'best_validation_loss': None, 'completed': False}


def validate_resume_state(state, plans, dataset, config):
    c.require(state.get('training_stage') == STAGE and state.get('sft_state_schema') == STATE_SCHEMA,
              'Resume requires a Full SFT checkpoint; pretraining checkpoints are forbidden')
    epochs = len(plans)
    accumulation = config['training']['gradient_accumulation_steps']
    expected_updates = sum(math.ceil(len(plan) / accumulation) for plan in plans)
    c.require(state['total_updates'] == expected_updates
              and state['total_budget'] == dataset.total_target_tokens * epochs,
              'Resume SFT schedule or target budget changed')
    c.require(not state['completed'], 'SFT checkpoint has already completed')
    epoch, consumed = state['epoch'], state['consumed_batches']
    c.require(isinstance(epoch, int) and 0 <= epoch <= epochs, 'Invalid SFT resume epoch')
    c.require(isinstance(consumed, int) and consumed >= 0, 'Invalid consumed-batch counter')
    c.require((epoch < epochs and consumed <= len(plans[epoch])) or (epoch == epochs and consumed == 0),
              'SFT resume position is outside its deterministic batch plan')
    if epoch < epochs:
        c.require(consumed % accumulation == 0 or consumed == len(plans[epoch]),
                  'SFT checkpoint was not written at an optimizer-update boundary')
    prefix = plans[epoch][:consumed] if epoch < epochs else []
    expected_examples = epoch * len(dataset) + sum(len(batch) for batch in prefix)
    expected_tokens = epoch * dataset.total_target_tokens + sum(
        int(dataset[index]['valid_tokens']) for batch in prefix for index in batch)
    expected_done = sum(math.ceil(len(plan) / accumulation) for plan in plans[:epoch]) + \
        math.ceil(consumed / accumulation)
    c.require(state['examples'] == expected_examples and state['tokens'] == expected_tokens
              and state['update'] == expected_done,
              'SFT resume counters disagree with actually consumed deterministic batches')


def load_sft_checkpoint(path, config, fingerprint):
    # Validate everything before mutating a model, optimizer, or global RNG.
    payload = c.load_checkpoint(path, config=config, fingerprint=fingerprint, restore_random=False)
    state = payload.get('training_state', {})
    c.require(state.get('training_stage') == STAGE and state.get('sft_state_schema') == STATE_SCHEMA,
              'Resume requires a Full SFT checkpoint; pretraining checkpoints are forbidden')
    c.require(payload['dataset_fingerprint'].get('training_stage') == STAGE,
              'Checkpoint does not belong to the Full SFT stage')
    return payload


def evaluate_preserving_state(model, dataset, config, device, helper=None):
    helper = helper or common().evaluate
    state, was_training = c.rng_state(), model.training
    try:
        result = helper(model, dataset, config, device)
        c.require(result['tokens'] > 0 and math.isfinite(result['loss']), 'Invalid assistant-only validation')
        return result
    finally:
        c.restore_rng(state)
        model.train(was_training)


def loader_iterator(dataset, config, batches, helper=None):
    helper = helper or common().loader
    # Windows DataLoader's base_seed must not perturb model dropout RNG on resume.
    state = c.rng_state()
    try:
        return iter(helper(dataset, config, batches))
    finally:
        c.restore_rng(state)


def retention_check(model, dataset, config, device, helper=None):
    """Observe the fixed original VALIDATION subset; never use it for selection."""
    c.require(getattr(dataset, 'split', None) == 'validation',
              'Retention monitoring requires original validation, never Test')
    helper = helper or common().retention_evaluate
    return evaluate_preserving_state(model, dataset, config, device, helper)


def export_tokenizer(base_dir, export, config):
    """Copy unchanged tokenizer bytes; record the separate segmented SFT format."""
    if __package__:
        from . import sft_format
    else:
        import sft_format
    base_dir, export = Path(base_dir).resolve(), Path(export).resolve()
    c.require(export.is_dir(), 'Model export directory must already exist')
    names = ['tokenizer.json', 'tokenizer_config.json']
    if (base_dir / 'special_tokens_map.json').is_file():
        names.append('special_tokens_map.json')
    for name in names:
        c.require((base_dir / name).is_file() and not (export / name).exists(),
                  'Missing original tokenizer or existing export tokenizer: ' + name)
    c.require(not (export / 'sft_template.json').exists(), 'SFT template receipt already exists')
    receipts = {}
    for name in names:
        shutil.copy2(base_dir / name, export / name)
        receipts[name] = c.sha256(export / name)
        c.require(receipts[name] == c.sha256(base_dir / name), 'Tokenizer bytes changed during export')
    c.write_new_json(export / 'sft_template.json', {
        'template_version': sft_format.TEMPLATE_VERSION,
        'formatter_sha256': c.sha256(Path(sft_format.__file__)),
        'role_headers': dict(sft_format.HEADERS), 'separator': sft_format.SEPARATOR,
        'encoding_policy': 'Encode each role header, content and separator as separate segments; append existing EOS after assistant content',
        'assistant_eos_token_id': sft_format.EOS_ID,
        'inference_entrypoint': 'scripts/generate_sft.py',
        'generic_chat_template': False,
        'tokenizer_policy': 'Unchanged base tokenizer files and vocabulary; use the dedicated segmented chat formatter',
        'tokenizer_files_sha256': receipts,
        'model_context_length': config['model']['max_position_embeddings'],
        'sft_sequence_length': config['training']['sequence_length'],
    })
    return receipts


def finalize_export(model, base_dir, out, config, state, checkpoint, last_saved_update,
                    stage=lambda phase, message: None):
    """Keep a resumable full-budget boundary before any fallible export work.

If model/tokenizer export fails, completed remains False and the full-budget
recovery checkpoint can resume into a NEW run to repeat validation/export. The
training loop skips optimizer updates because its epoch/target budget is done.
Only a successful export permits final.pt to carry completed=True.
    """
    c.require(not state['completed'] and state['tokens'] == state['total_budget']
              and state['update'] == state['total_updates']
              and state['epoch'] == config['training']['max_epochs']
              and state['consumed_batches'] == 0,
              'SFT finalization requires a completed training budget at an unfinished export boundary')
    if last_saved_update != state['update']:
        checkpoint()
    export = Path(out) / 'model'
    stage('Exporting SFT model', 'Saving separate SFT safetensors weights and the unchanged base tokenizer...')
    model.save_pretrained(export, safe_serialization=True)
    tokenizer_receipts = export_tokenizer(base_dir, export, config)
    state['completed'] = True
    checkpoint(name='final.pt')
    return export, tokenizer_receipts


def _plans(dataset, config, helper):
    t = config['training']
    plans = [helper(dataset, t['micro_batch_size'], t['seed'], epoch, shuffle=True)
             for epoch in range(t['max_epochs'])]
    for plan in plans:
        c.require(plan and all(batch and len(batch) <= t['micro_batch_size'] for batch in plan),
                  'Empty or oversized SFT microbatch plan')
        flattened = [int(index) for batch in plan for index in batch]
        c.require(sorted(flattened) == list(range(len(dataset))), 'SFT batch plan skips or repeats examples')
    return plans


def train(config, out, resume=None):
    s = common()
    c.require(torch.cuda.is_available(), 'CUDA is required; no silent CPU SFT training')
    c.require(torch.cuda.is_bf16_supported(), 'Native BF16 support is required')
    _, base_dir, data_root, approved_fingerprint = s.approved_inputs(config, verify_hashes=True)
    s.require_readiness(config, approved_fingerprint)
    code = s.code_receipts()
    fingerprint = checkpoint_fingerprint(approved_fingerprint, code)
    t = config['training']
    c.require(t['precision'] == 'bf16', 'SFT requires BF16 AMP with FP32 master parameters')
    dataset, validation = s.SFTDataset(data_root, 'train'), s.SFTDataset(data_root, 'validation')
    retention = None
    progress = None
    try:
        retention = s.retention_dataset(config)
        plans = _plans(dataset, config, s.epoch_batches)
        state = initial_state([(plan, t['gradient_accumulation_steps']) for plan in plans],
                              dataset.total_target_tokens * t['max_epochs'])
        c.require(state['total_budget'] > 0 and state['total_updates'] > 0, 'Empty SFT training budget')
        payload = None
        if resume:
            payload = load_sft_checkpoint(resume, config, fingerprint)
            state = dict(payload['training_state'])
            validate_resume_state(state, plans, dataset, config)
        c.seed_all(t['seed'])
        torch.set_num_threads(4)
        device = torch.device('cuda:0')
        # Fresh SFT loads base safetensors only. Resume reconstructs the model
        # structure and then restores its independently verified SFT checkpoint.
        model = (c.build_model(config).to(device) if resume else s.build_base_model(config, device))
        c.require(all(parameter.dtype == torch.float32 for parameter in model.parameters()),
                  'Expected FP32 parameters with BF16 autocast, not a BF16 optimizer state')
        c.require(all(parameter.requires_grad for parameter in model.parameters()), 'Full SFT must train all parameters')
        optimizer = c.make_optimizer(model, config)
        c.require(not optimizer.state, 'Fresh SFT optimizer must have empty Adam state')
        if payload:
            model.load_state_dict(payload['model'], strict=True)
            optimizer.load_state_dict(payload['optimizer'])
            c.restore_rng(payload['rng'])
        del payload  # Release CPU checkpoint tensors/moments after device restore.
        out = c.safe_new_output(out)
        c.write_new_json(out / 'config.json', config)
        c.write_new_json(out / 'run.json', {'started_at_utc': c.now(), 'mode': STAGE,
            'dataset_fingerprint': fingerprint, 'code_receipts': code,
            'runtime_versions': c.runtime_versions(), 'base_model_dir': str(base_dir),
            'resume_checkpoint': str(Path(resume).resolve()) if resume else None,
            'optimizer_initialization': 'SFT checkpoint only' if resume else 'Fresh Adam moments and step 0',
            'schedule': 'Fresh SFT update counters; linear warmup then cosine decay',
            'raw_data_access': 'none; only independently approved SFT JSONL is read',
            'total_updates': state['total_updates'], 'total_assistant_targets': state['total_budget'],
            'retention_policy': {'split': 'original pretraining validation',
                'maximum_targets': t['retention_max_tokens'], 'every_updates': t['retention_every_updates'],
                'baseline': 'Fresh SFT update 0', 'model_selection': False, 'test_set_used': False},
            'eta_scope': 'Remaining assistant targets, session wall time including observed eval/save overhead; final validation/export excluded'})
        events = out / 'metrics.jsonl'
        starting_update = state['update']
        last_saved_update = state['update'] if resume else -1
        at_boundary = True
        completed_boundary = dict(state)
        progress = TrainingProgress(state['total_budget'], state['total_updates'],
                                    initial_tokens=state['tokens'], initial_update=state['update'])

        def stage(phase, message):
            progress.set_phase(phase)
            write_event(events, {'kind': 'stage', 'phase': phase, 'update': state['update'],
                                 'message': message}, progress)

        def checkpoint(name=None):
            nonlocal last_saved_update
            c.require(at_boundary, 'Refusing a checkpoint during a partial optimizer update')
            stage('Saving SFT checkpoint', 'Saving SFT-only weights, Adam moments, counters and RNG...')
            started = time.perf_counter()
            path = c.save_checkpoint(out, model, optimizer, config, fingerprint, state, name=name)
            last_saved_update = state['update']
            write_event(events, {'kind': 'checkpoint', 'update': state['update'],
                        'checkpoint': str(path), 'duration_seconds': time.perf_counter() - started}, progress)
            return path

        def monitor_retention(phase):
            stage('Retention validation', 'Monitoring the fixed original pretraining validation subset...')
            result = retention_check(model, retention, config, device, s.retention_evaluate)
            write_event(events, {'kind': 'retention_validation', 'phase': 'Retention validation',
                'retention_phase': phase, 'update': state['update'], 'dataset_split': 'validation',
                'used_for_model_selection': False,
                'message': f'Original validation loss {result["loss"]:.4f} | PPL {result["perplexity"]:.2f} | '
                           f'{result["tokens"]:,} targets; retention monitor only', **result}, progress)
            progress.set_phase('Train')
            return result

        model.train()
        try:
            write_event(events, {'kind': 'started', **state, **progress.snapshot(),
                'message': 'Full SFT: assistant-only targets; ' +
                           f'{state["total_budget"]:,} targets / {state["total_updates"]:,} updates; ' +
                           f'resuming at {state["update"]:,}; no pretraining optimizer state.'}, progress)
            if not resume:
                monitor_retention('baseline')
            while state['epoch'] < len(plans):
                plan = plans[state['epoch']]
                iterator = loader_iterator(dataset, config, plan[state['consumed_batches']:], s.loader)
                while True:
                    batches = list(itertools.islice(iterator, t['gradient_accumulation_steps']))
                    if not batches:
                        break
                    at_boundary = False
                    rate = learning_rate(config, state['update'], state['total_updates'])
                    for group in optimizer.param_groups:
                        group['lr'] = rate
                    optimizer.zero_grad(set_to_none=True)
                    measured = backward_accumulated(model, batches, device, s.forward_loss_sum)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t['gradient_clip'], error_if_nonfinite=True)
                    # This code path is reachable only through explicit --start-training.
                    optimizer.step()
                    state['consumed_batches'] += measured['batches']
                    state['examples'] += measured['examples']
                    state['tokens'] += measured['tokens']
                    state['update'] += 1
                    c.require(state['tokens'] <= state['total_budget'], 'SFT target budget exceeded')
                    at_boundary = True
                    completed_boundary = dict(state)
                    telemetry = progress.advance(state['update'], state['tokens'], measured['loss'])
                    if state['update'] == starting_update + 1 or state['update'] % t['log_every_updates'] == 0 \
                            or state['update'] == state['total_updates']:
                        write_event(events, {'kind': 'train', **state, **telemetry,
                            'loss': measured['loss'], 'learning_rate': rate, 'gradient_norm': float(norm),
                            'peak_allocated_bytes': torch.cuda.max_memory_allocated()}, progress)
                    if state['update'] % t['eval_every_updates'] == 0:
                        stage('SFT validation', 'Evaluating held-out assistant targets without optimizer updates...')
                        result = evaluate_preserving_state(model, validation, config, device, s.evaluate)
                        write_event(events, {'kind': 'full_validation', 'update': state['update'], **result}, progress)
                        if state['best_validation_loss'] is None or result['loss'] < state['best_validation_loss']:
                            state['best_validation_loss'] = result['loss']
                            checkpoint(name=f"best_{state['update']:07d}.pt")
                            c.write_new_json(out / f"best_{state['update']:07d}_validation.json", result)
                    if state['update'] % t['retention_every_updates'] == 0:
                        monitor_retention('periodic')
                    if state['update'] % t['checkpoint_every_updates'] == 0:
                        checkpoint()
                    # Retain two latest regular checkpoints and one best. core
                    # pruning only touches hashed files inside this new SFT run.
                    if (out / 'latest.json').exists():
                        c.prune_checkpoints(out, t['keep_latest_checkpoints'])
                    completed_boundary = dict(state)
                    progress.set_phase('Train')
                c.require(state['consumed_batches'] == len(plan), 'SFT loader skipped or repeated planned batches')
                state['epoch'] += 1
                state['consumed_batches'] = 0
                completed_boundary = dict(state)
            c.require(state['tokens'] == state['total_budget'] and state['update'] == state['total_updates'],
                      'Final SFT target coverage or schedule differs from the approved plan')
            stage('Final SFT validation', 'All SFT targets complete; final validation and export remain...')
            result = evaluate_preserving_state(model, validation, config, device, s.evaluate)
            write_event(events, {'kind': 'final_validation', 'update': state['update'], **result}, progress)
            if state['best_validation_loss'] is None or result['loss'] < state['best_validation_loss']:
                state['best_validation_loss'] = result['loss']
                best_name = f"best_{state['update']:07d}.pt"
                if not (out / best_name).exists():
                    checkpoint(name=best_name)
                    c.write_new_json(out / f"best_{state['update']:07d}_validation.json", result)
            final_retention = monitor_retention('final')
            export, tokenizer_receipts = finalize_export(
                model, base_dir, out, config, state, checkpoint, last_saved_update, stage)
            c.prune_checkpoints(out, t['keep_latest_checkpoints'])
            c.write_new_json(out / 'completed.json', {'status': 'completed', 'training_stage': STAGE,
                'state': state, 'validation': result, 'retention_validation': final_retention,
                'tokenizer_files_sha256': tokenizer_receipts, 'finished_at_utc': c.now(),
                'test_set_used_for_tuning': False, 'base_weights_preserved': True})
            progress.set_phase('Completed')
            write_event(events, {'kind': 'completed', **state, **progress.snapshot(), 'export': str(export)}, progress)
        except BaseException as error:
            # A partial forward/backward/step cannot produce a resumable receipt.
            # At a completed boundary, Ctrl+C can save the current coherent state.
            emergency_error = None
            if isinstance(error, (KeyboardInterrupt, SystemExit)) and at_boundary \
                    and state['update'] > starting_update and last_saved_update != state['update']:
                try:
                    checkpoint()
                    c.prune_checkpoints(out, t['keep_latest_checkpoints'])
                except BaseException as save_error:
                    emergency_error = type(save_error).__name__ + ': ' + str(save_error)
            progress.set_phase('Stopped')
            write_event(events, {'kind': 'stopped', 'error': type(error).__name__, 'message': str(error),
                'last_saved_update': last_saved_update, 'last_complete_boundary': completed_boundary,
                'partial_update_discarded': not at_boundary, 'emergency_checkpoint_error': emergency_error}, progress)
            raise
    finally:
        if progress is not None:
            progress.close()
        dataset.close()
        validation.close()
        if retention is not None:
            retention.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--start-training', action='store_true', help='Explicitly authorize actual SFT optimizer updates')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path, help='Only a verified checkpoint from this Full SFT stage')
    args = parser.parse_args(argv)
    s = common()
    config = s.load_config(args.config)
    if not args.start_training:
        print(json.dumps({'mode': 'configuration_only', 'training_stage': STAGE,
            'parameters': c.theoretical_parameters(config['model']), 'training': config['training'],
            'training_started': False, 'optimizer_updates': 0,
            'resume_checkpoint': str(args.resume) if args.resume else None,
            'message': 'Actual Full SFT updates require --start-training and matching readiness approval.'}, indent=2))
        return
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
    output = args.output or c.resolve_path(config['training']['output_root']) / timestamp
    train(config, output, args.resume)


if __name__ == '__main__':
    main()
