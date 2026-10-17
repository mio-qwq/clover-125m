"""Shared, local-only English LM configuration, validation and checkpoints."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('USE_TF', '0')

import numpy as np
import torch

DEFAULT_CONFIG = ROOT / 'configs/english_125m.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def resolve_path(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def load_config(path=DEFAULT_CONFIG):
    config = read_json(path)
    m, t = config['model'], config['training']
    require(config['schema_version'] == 1, 'Unsupported configuration schema')
    require(m['hidden_size'] % m['num_attention_heads'] == 0, 'Invalid attention dimensions')
    require(m['num_attention_heads'] == m['num_key_value_heads'], 'This recipe requires MHA')
    require(m['tie_word_embeddings'] and not m['attention_bias'] and not m['mlp_bias'], 'Unexpected model recipe')
    require(t['precision'] == 'bf16', 'This recipe requires BF16 AMP')
    for name in ('sequence_length', 'micro_batch_size', 'gradient_accumulation_steps', 'max_epochs', 'max_tokens',
                 'log_every_updates', 'mini_validation_every_updates', 'full_validation_every_updates',
                 'checkpoint_every_updates', 'keep_latest_checkpoints'):
        require(isinstance(t[name], int) and t[name] > 0, 'Invalid positive integer: ' + name)
    require(0 <= t['num_workers'] <= 8, 'Invalid data worker count')
    require(t['sequence_length'] <= m['max_position_embeddings'], 'Context exceeds model configuration')
    require(0 < t['minimum_learning_rate'] <= t['learning_rate'], 'Invalid learning rates')
    return config


def theoretical_parameters(model_config):
    h, f, layers, v = (model_config[k] for k in ('hidden_size', 'intermediate_size', 'num_hidden_layers', 'vocab_size'))
    return v*h + layers*(4*h*h + 3*h*f + 2*h) + h


def build_model(config, device='cpu'):
    from transformers import LlamaConfig, LlamaForCausalLM
    m = LlamaConfig(**config['model'])
    m._attn_implementation = 'sdpa'
    with torch.device(device):
        model = LlamaForCausalLM(m)
    require(sum(p.numel() for p in model.parameters()) == theoretical_parameters(config['model']),
            'Actual parameter count differs from the planned structure')
    require(model.get_input_embeddings().weight is model.get_output_embeddings().weight,
            'Input/output embeddings were not tied')
    model.config.use_cache = False
    if config['training']['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    return model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def approved_data(config, verify_hashes=True):
    pointer_path = resolve_path(config['dataset_pointer'])
    pointer = read_json(pointer_path)
    root = Path(pointer['prepared_dir']).resolve()
    manifest_path = root / 'manifest.json'
    manifest = read_json(manifest_path)
    release_path = root / 'release_validation.json'
    release = read_json(release_path)
    require(pointer['status'] == 'validated' and manifest['status'] == 'completed' and release['status'] == 'passed',
            'An independently accepted preparation is required')
    require(pointer['language'] == manifest['language'] == 'English', 'Expected an English corpus')
    require(pointer['vocab_size'] == config['model']['vocab_size'], 'Vocabulary size mismatch')
    special = manifest['tokenizer']['special_token_ids']
    require(special == {'<|pad|>': 0, '<|endoftext|>': 1, '<|unk|>': 2, '<|beginoftext|>': 3}, 'Special token mismatch')
    require(manifest['tokenizer']['training_set_only_verified'], 'Tokenizer holdout audit missing')
    require(release['parent_and_duplicate_components_no_cross_split'], 'Split audit missing')
    require(not pointer['language_model_training_started'], 'Unexpected preparation status')
    if verify_hashes:
        for entry in manifest['files']:
            path = Path(entry['path']).resolve()
            require(path.is_relative_to(root), 'Export resolves outside its preparation directory')
            require(path.stat().st_size == entry['bytes'] and sha256(path) == entry['sha256'],
                    'Dataset integrity failure: ' + str(path))
    fingerprint = {'pointer_sha256': sha256(pointer_path), 'manifest_sha256': sha256(manifest_path),
                   'release_sha256': sha256(release_path), 'tokenizer_sha256': sha256(root/'tokenizer/tokenizer.json')}
    return root, manifest, fingerprint


def safe_new_output(path):
    path = resolve_path(path)
    allowed = (ROOT / 'runs').resolve()
    require(path.is_relative_to(allowed) and path != allowed, 'Outputs must be inside a new project runs directory')
    require(not path.exists(), 'Output already exists; refuse replacement: ' + str(path))
    path.mkdir(parents=True)
    return path


def make_optimizer(model, config, fused=None):
    settings = config['training']
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{'params': [p for p in params if p.ndim >= 2], 'weight_decay': settings['weight_decay']},
              {'params': [p for p in params if p.ndim < 2], 'weight_decay': 0.0}]
    if fused is None:
        fused = next(model.parameters()).device.type == 'cuda'
    return torch.optim.AdamW(groups, lr=settings['learning_rate'], betas=tuple(settings['betas']),
                             eps=settings['adam_epsilon'], fused=fused)


def allocate_adam_buffers_without_update(optimizer):
    """Acceptance only: allocate realistic zero moments without optimizer.step."""
    for group in optimizer.param_groups:
        for param in group['params']:
            state = optimizer.state[param]
            state['step'] = torch.zeros((), dtype=torch.float32,
                device=param.device if group.get('fused') or group.get('capturable') else 'cpu')
            state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)


def learning_rate(config, update, total_updates):
    t = config['training']
    warmup = min(t['warmup_updates'], max(1, total_updates-1))
    if update < warmup:
        return t['learning_rate'] * (update+1) / warmup
    fraction = min(1., max(0., (update-warmup) / max(1, total_updates-warmup-1)))
    return t['minimum_learning_rate'] + (t['learning_rate']-t['minimum_learning_rate'])*.5*(1+math.cos(math.pi*fraction))


def forward_loss_sum(model, x, y):
    # Targets are ALREADY shifted by the data reader. Never pass HF `labels`.
    logits = model(input_ids=x, use_cache=False).logits
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                            ignore_index=-100, reduction='sum')
    return loss


def rng_state():
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (n[0], torch.tensor(n[1].astype(np.int64)), n[2], n[3], n[4]),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], n[1].cpu().numpy().astype(np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_checkpoint(out, model, optimizer, config, fingerprint, state, name=None):
    """Immutable checkpoint files; manifest/pointer replacement is atomic."""
    out = Path(out).resolve()
    require(out.is_relative_to((ROOT/'runs').resolve()), 'Checkpoint outside runs')
    filename = name or f"checkpoint_{state['update']:07d}.pt"
    require(Path(filename).name == filename and filename.endswith('.pt'), 'Invalid checkpoint name')
    path = out / filename
    require(not path.exists(), 'Refusing existing checkpoint')
    temporary = path.with_suffix('.tmp')
    require(not temporary.exists(), 'Interrupted checkpoint file exists')
    payload = {'schema_version': 1, 'config': config, 'config_sha256': config_digest(config),
               'dataset_fingerprint': fingerprint, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'training_state': dict(state), 'rng': rng_state(), 'saved_at_utc': now()}
    torch.save(payload, temporary)
    os.replace(temporary, path)
    receipt = {'checkpoint': filename, 'bytes': path.stat().st_size, 'sha256': sha256(path), 'state': dict(state)}
    write_new_json(path.with_suffix('.json'), receipt)
    latest_tmp = out / 'latest.tmp'
    latest_tmp.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    os.replace(latest_tmp, out/'latest.json')
    return path


def load_checkpoint(path, model=None, optimizer=None, config=None, fingerprint=None, restore_random=True):
    path = Path(path).resolve()
    receipt = read_json(path.with_suffix('.json'))
    require(path.stat().st_size == receipt['bytes'] and sha256(path) == receipt['sha256'], 'Checkpoint hash mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    require(payload['schema_version'] == 1, 'Unsupported checkpoint schema')
    if config is not None:
        require(payload['config_sha256'] == config_digest(config), 'Resume configuration mismatch')
    if fingerprint is not None:
        require(payload['dataset_fingerprint'] == fingerprint, 'Resume corpus/tokenizer mismatch')
    if model is not None:
        model.load_state_dict(payload['model'], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload['optimizer'])
    if restore_random:
        restore_rng(payload['rng'])
    return payload


def weights_digest(model):
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        digest.update(name.encode())
        digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def runtime_versions():
    import importlib.metadata
    return {name: importlib.metadata.version(name)
            for name in ('torch', 'transformers', 'tokenizers', 'safetensors', 'numpy', 'tqdm')}


def make_loader(dataset, config, sampler=None, start=0):
    t = config['training']
    workers = t['num_workers']
    generator = torch.Generator().manual_seed(t['seed']+12345)
    if sampler is None and start:
        sampler = range(start, len(dataset))
    options = dict(dataset=dataset, batch_size=t['micro_batch_size'], sampler=sampler,
                   shuffle=False, drop_last=False, num_workers=workers, pin_memory=True,
                   generator=generator)
    if workers:
        options.update(prefetch_factor=t['prefetch_factor'], persistent_workers=False,
                       multiprocessing_context='spawn')
    return torch.utils.data.DataLoader(**options)


def evaluate(model, dataset, config, device, max_tokens=None):
    """Token-weighted, deterministic source-stratified validation; never updates."""
    t = config['training']
    budgets = None
    if max_tokens is not None:
        require(isinstance(max_tokens, int) and max_tokens > 0, 'Evaluation token cap must be positive')
        totals = [v['effective_targets'] for v in dataset.source_metadata]
        total = sum(totals)
        limit = min(max_tokens, total)
        quotas = [limit*v/total for v in totals]
        budgets = [math.floor(v) for v in quotas]
        remainder = limit-sum(budgets)
        order = sorted(range(len(totals)), key=lambda i: quotas[i]-budgets[i], reverse=True)
        for i in order[:remainder]:
            budgets[i] += 1
    losses = [0.] * len(dataset.sources)
    counts = [0] * len(dataset.sources)
    was_training = model.training
    model.eval()
    indices = None
    if budgets is not None:
        indices = []
        generator = np.random.default_rng(20261014)
        for info, budget in zip(dataset.source_metadata, budgets):
            count = min(info['blocks'], math.ceil(budget/dataset.sequence_length)+1) if budget else 0
            if count:
                selected = generator.choice(info['blocks'], size=count, replace=False)
                indices.extend(int(info['first_block_id']+i) for i in selected)
    try:
        with torch.inference_mode():
            for batch in make_loader(dataset, config, sampler=indices):
                selected = [i for i, source in enumerate(batch['source_id'].tolist())
                            if budgets is None or counts[source] < budgets[source]]
                if not selected:
                    if budgets is not None and all(n >= b for n, b in zip(counts, budgets)):
                        break
                    continue
                x = batch['input_ids'][selected].to(device, non_blocking=True)
                y = batch['targets'][selected].to(device, non_blocking=True)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                    logits = model(input_ids=x, use_cache=False).logits
                    values = torch.nn.functional.cross_entropy(logits.transpose(1, 2), y,
                        ignore_index=-100, reduction='none').float()
                for row, original_index in enumerate(selected):
                    source = int(batch['source_id'][original_index])
                    valid = y[row] != -100
                    if budgets is not None:
                        remaining = max(0, budgets[source]-counts[source])
                        positions = valid.nonzero(as_tuple=False).flatten()[:remaining]
                        total_loss = values[row, positions].sum().item()
                        tokens = len(positions)
                    else:
                        total_loss = values[row, valid].sum().item()
                        tokens = int(valid.sum())
                    losses[source] += total_loss
                    counts[source] += tokens
    finally:
        model.train(was_training)
    total_tokens = sum(counts)
    require(total_tokens > 0, 'Evaluation contains no valid targets')
    average = sum(losses)/total_tokens
    require(math.isfinite(average), 'Nonfinite validation loss; refuse model acceptance')
    return {'loss': average, 'perplexity': math.exp(min(average, 80)), 'tokens': total_tokens,
            'sources': {s: {'loss': loss/n if n else None, 'tokens': n}
                        for s, loss, n in zip(dataset.sources, losses, counts)}}


def code_receipts():
    names = ('scripts/lm_common.py', 'scripts/lm_data.py', 'scripts/lm_progress.py', 'scripts/train_lm.py',
             'scripts/evaluate_lm.py', 'scripts/generate_lm.py', 'scripts/verify_lm_ready.py',
             'requirements-training.lock.txt', 'start_training.ps1')
    return {name: sha256(ROOT/name) for name in names}


def prune_checkpoints(out, keep):
    """Remove only verified checkpoint files created inside this particular run."""
    import re
    out = Path(out).resolve()
    require(out.is_relative_to((ROOT/'runs').resolve()) and out != (ROOT/'runs').resolve(),
            'Invalid retention directory')
    candidates = sorted(p for p in out.glob('checkpoint_*.pt')
                        if re.fullmatch(r'checkpoint_\d{7}\.pt', p.name))[:-keep]
    best = sorted(p for p in out.glob('best_*.pt') if re.fullmatch(r'best_\d{7}\.pt', p.name))[:-1]
    latest = read_json(out/'latest.json')['checkpoint']
    for path in candidates+best:
        require(path.resolve().parent == out and not path.is_symlink(), 'Unsafe checkpoint retention target')
        require(path.name != latest and path.name != 'final.pt', 'Cannot remove current checkpoint')
        receipt = read_json(path.with_suffix('.json'))
        require(path.stat().st_size == receipt['bytes'] and sha256(path) == receipt['sha256'],
                'Refusing to prune an unverified checkpoint')
        targets = [path, path.with_suffix('.json')]
        if path in best:
            validation = out/(path.stem+'_validation.json')
            if validation.exists():
                targets.append(validation)
        for target in targets:
            require(target.resolve().parent == out and not target.is_symlink(), 'Retention path escaped run directory')
            target.unlink()


def require_readiness(config, fingerprint):
    pointer = read_json(ROOT/'runs/CURRENT_TRAINING_READINESS.json')
    report = read_json(pointer['report'])
    require(report['status'] == 'passed' and report['formal_optimizer_updates'] == 0,
            'A successful pre-training acceptance is required')
    require(report['config_sha256'] == config_digest(config), 'Configuration changed after acceptance')
    require(report['dataset_fingerprint'] == fingerprint, 'Dataset changed after acceptance')
    require(report['code_sha256'] == code_receipts(), 'Scripts changed after acceptance; rerun acceptance')
    require(report['runtime_versions'] == runtime_versions(), 'Dependencies changed after acceptance')
    return report
