"""Local Full SFT inputs, masked losses and deterministic conversation batches."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

try:
    from . import lm_common as c
    from .sft_format import TEMPLATE_VERSION
except ImportError:
    import lm_common as c
    from sft_format import TEMPLATE_VERSION

DEFAULT_CONFIG = c.ROOT/'configs/english_125m_sft.json'


def load_config(path=DEFAULT_CONFIG):
    config = c.read_json(c.resolve_path(path))
    c.require(config['schema_version'] == 1 and config['stage'] == 'english_full_sft', 'Expected Full SFT config')
    t, m = config['training'], config['model']
    c.require(m == c.read_json(c.resolve_path(config['base_pretraining_config']))['model'], 'Base architecture changed')
    c.require(m['vocab_size'] == 32000 and m['tie_word_embeddings'], 'Keep the original tied 32k vocabulary')
    for key in ('sequence_length', 'micro_batch_size', 'gradient_accumulation_steps', 'max_epochs',
                'eval_every_updates', 'retention_every_updates', 'retention_max_tokens',
                'checkpoint_every_updates', 'log_every_updates', 'keep_latest_checkpoints'):
        c.require(type(t[key]) is int and t[key] > 0, 'Invalid positive integer: '+key)
    c.require(0 < t['sequence_length'] <= m['max_position_embeddings'], 'Invalid SFT context')
    c.require(0 <= t['num_workers'] <= 8 and t['prefetch_factor'] > 0, 'Invalid loader settings')
    c.require(t['precision'] == 'bf16' and 0 < t['minimum_learning_rate'] <= t['learning_rate'], 'Invalid precision/LR')
    c.require(0 <= t['warmup_ratio'] < 1 and t['gradient_clip'] > 0, 'Invalid warmup/clip')
    return config


def code_receipts():
    files = ('scripts/sft_format.py', 'scripts/sft_common.py', 'scripts/prepare_sft_data.py',
             'scripts/train_sft.py', 'scripts/verify_sft_ready.py', 'scripts/verify_sft_data.py',
             'scripts/verify_sft_source_rows.py', 'scripts/verify_sft_download.py',
             'scripts/archive_base_for_sft.py', 'scripts/generate_sft.py', 'requirements-sft-data.txt')
    return {**c.code_receipts(), **{name: c.sha256(c.ROOT/name) for name in files}}


def _verify_files(entries, allowed):
    for entry in entries:
        path = Path(entry['path']).resolve()
        c.require(path.is_relative_to(allowed) and path.is_file() and not path.is_symlink(), 'External/missing input file')
        c.require(path.stat().st_size == entry['bytes'] and c.sha256(path) == entry['sha256'],
                  'Input SHA-256 mismatch: '+str(path))


def approved_inputs(config, verify_hashes=True):
    base_manifest_path = c.resolve_path(config['base_manifest'])
    base = c.read_json(base_manifest_path)
    model_dir = c.resolve_path(config['base_model_dir'])
    c.require(model_dir.is_relative_to((c.ROOT/'runs/sft_inputs').resolve()), 'Use an independent SFT Base copy')
    c.require(base['status'] == 'verified' and base['kind'] == 'sft_base_copy', 'Unverified SFT Base copy')
    c.require(base['source_training_state']['completed'] is True, 'Incomplete original Base')
    c.require(base['base_pretraining_config_sha256'] == c.sha256(c.resolve_path(config['base_pretraining_config'])),
              'Base architecture receipt mismatch')
    archive_path = Path(base['archive_manifest'])
    c.require(c.sha256(archive_path) == base['archive_manifest_sha256'], 'Base archive receipt changed')
    pointer_path = c.resolve_path(config['dataset_pointer'])
    pointer = c.read_json(pointer_path)
    root = Path(pointer['prepared_dir']).resolve()
    c.require(root.is_relative_to((c.ROOT/'data/processed/sft_en_v1').resolve()), 'Unexpected SFT data location')
    manifest_path, release_path = root/'manifest.json', root/'release_validation.json'
    manifest, release = c.read_json(manifest_path), c.read_json(release_path)
    c.require(pointer['status'] == 'validated' and manifest['status'] == 'completed' and release['status'] == 'passed',
              'A passed independent SFT data acceptance is required')
    c.require(not manifest['sample_only'] and manifest['sequence_length'] <= config['training']['sequence_length'],
              'A full preparation within the configured SFT length is required')
    c.require(release['manifest_sha256'] == c.sha256(manifest_path), 'Stale SFT data acceptance')
    c.require(release['template_code_sha256'] == c.sha256(Path(__file__).with_name('sft_format.py')),
              'SFT formatter changed after data acceptance')
    for field in ('all_reencoded_equal', 'group_split_leakage_check', 'raw_metadata_unchanged', 'input_hashes_unchanged'):
        c.require(release[field] is True, 'SFT data acceptance missing: '+field)
    tokenizer_sha = c.sha256(model_dir/'tokenizer.json')
    c.require(release['base_tokenizer_sha256'] == tokenizer_sha and release['all_base_copy_reencoded_equal'] is True,
              'SFT Base tokenizer equivalence audit missing')
    c.require(release['prepared_tokenizer_sha256'] == manifest['tokenizer_sha256'], 'SFT preparation tokenizer mismatch')
    c.require(manifest.get('language', 'English') == 'English', 'English SFT only')
    c.require(not manifest['language_model_training_started'], 'Preparation must have zero training updates')
    if verify_hashes:
        _verify_files(base['files'], model_dir.parent)
        _verify_files(manifest['files'], root)
    base_config = c.read_json(c.resolve_path(config['base_pretraining_config']))
    _, _, retention_fingerprint = c.approved_data(base_config, verify_hashes=verify_hashes)
    fingerprint = {'base_manifest_sha256': c.sha256(base_manifest_path),
                   'base_checkpoint_sha256': base['source_checkpoint_sha256'],
                   'model_sha256': c.sha256(model_dir/'model.safetensors'),
                   'dataset_pointer_sha256': c.sha256(pointer_path),
                   'manifest_sha256': c.sha256(manifest_path), 'release_sha256': c.sha256(release_path),
                   'tokenizer_sha256': tokenizer_sha, 'template_version': TEMPLATE_VERSION,
                   'template_code_sha256': c.sha256(Path(__file__).with_name('sft_format.py')),
                   'retention_dataset_fingerprint': retention_fingerprint,
                   'retention_max_tokens': config['training']['retention_max_tokens']}
    return base_config, model_dir, root, fingerprint


def build_base_model(config, device='cpu'):
    """Load ONLY copied safetensors weights; no optimizer or pretraining RNG."""
    from safetensors.torch import load_model
    model = c.build_model(config).to(device)
    model_dir = c.resolve_path(config['base_model_dir'])
    load_model(model, str(model_dir/'model.safetensors'), strict=True, device='cpu')
    c.require(all(p.dtype == torch.float32 for p in model.parameters()), 'Expected FP32 master weights')
    c.require(model.get_input_embeddings().weight is model.get_output_embeddings().weight, 'Embedding tie lost')
    return model


class SFTDataset:
    """Compact read-only records; each item is already shifted exactly ONCE."""
    def __init__(self, root, split):
        c.require(split in ('train', 'validation', 'test'), 'Unknown SFT split')
        self.root, self.split = Path(root), split
        self.items = []
        self.total_target_tokens = self.input_tokens = 0
        with (self.root/'splits'/f'{split}.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                ids = np.asarray(row['input_ids'], dtype=np.int64)
                mask = np.asarray(row['assistant_mask'], dtype=np.bool_)
                c.require(len(ids) == len(mask) and len(ids) >= 2 and ids[-1] == 1 and bool(mask[-1]), 'Broken conversation/EOS')
                c.require(not mask[0] and np.all((ids >= 0) & (ids < 32000)), 'Invalid IDs or first-token supervision')
                targets = int(mask[1:].sum())
                c.require(targets > 0 and targets == row['assistant_targets'], 'Assistant target count mismatch')
                self.items.append({'ids': ids, 'mask': mask, 'valid_tokens': targets,
                                   'source': row.get('source', row['category']),
                                   'doc_id': row['doc_id'], 'category': row['category']})
                self.total_target_tokens += targets
                self.input_tokens += len(ids)-1
        c.require(self.items, 'Empty SFT split: '+split)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row = self.items[index]
        targets = row['ids'][1:].copy()
        targets[~row['mask'][1:]] = -100
        return {'input_ids': row['ids'][:-1].copy(), 'targets': targets,
                'attention_mask': np.ones(len(targets), dtype=np.int64),
                'valid_tokens': row['valid_tokens'], 'source': row['source'], 'doc_id': row['doc_id']}

    def close(self):
        pass


def collate(rows):
    length = ((max(len(row['input_ids']) for row in rows)+7)//8)*8
    x = torch.zeros((len(rows), length), dtype=torch.long)
    y = torch.full_like(x, -100)
    attention = torch.zeros_like(x)
    for i, row in enumerate(rows):
        n = len(row['input_ids'])
        x[i, :n] = torch.as_tensor(row['input_ids'])
        y[i, :n] = torch.as_tensor(row['targets'])
        attention[i, :n] = 1
    return {'input_ids': x, 'targets': y, 'attention_mask': attention,
            'valid_tokens': torch.tensor([r['valid_tokens'] for r in rows], dtype=torch.long),
            'source': [r['source'] for r in rows], 'doc_id': [r['doc_id'] for r in rows]}


def epoch_batches(dataset, batch_size, seed, epoch, shuffle=True):
    c.require(batch_size > 0 and seed >= 0 and epoch >= 0, 'Invalid batch plan')
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(seed+epoch)
    if shuffle:
        rng.shuffle(indices)
    batches = []
    pool_size = batch_size*128
    for start in range(0, len(indices), pool_size):
        pool = sorted(indices[start:start+pool_size].tolist(), key=lambda i: len(dataset.items[i]['ids']), reverse=True)
        batches.extend(pool[i:i+batch_size] for i in range(0, len(pool), batch_size))
    if shuffle:
        rng.shuffle(batches)
    return batches


def loader(dataset, config, batches):
    t = config['training']
    options = dict(dataset=dataset, batch_sampler=batches, collate_fn=collate,
                   num_workers=t['num_workers'], pin_memory=True,
                   generator=torch.Generator().manual_seed(t['seed']+12345))
    if t['num_workers']:
        options.update(prefetch_factor=t['prefetch_factor'], persistent_workers=False, multiprocessing_context='spawn')
    return torch.utils.data.DataLoader(**options)


def loss_values(model, batch, device):
    device = torch.device(device)
    x = batch['input_ids'].to(device, non_blocking=True)
    y = batch['targets'].to(device, non_blocking=True)
    attention = batch['attention_mask'].to(device, non_blocking=True)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        logits = model(input_ids=x, attention_mask=attention, use_cache=False).logits
        values = torch.nn.functional.cross_entropy(logits.transpose(1, 2), y, ignore_index=-100, reduction='none').float()
    return values


def forward_loss_sum(model, batch, device):
    return loss_values(model, batch, device).sum()


def evaluate(model, dataset, config, device):
    was_training = model.training
    random_before = c.rng_state()
    sources = {}
    model.eval()
    try:
        batches = epoch_batches(dataset, config['training']['micro_batch_size'], config['training']['seed'], 0, shuffle=False)
        with torch.inference_mode():
            for batch in loader(dataset, config, batches):
                rows = loss_values(model, batch, device).sum(1).tolist()
                for source, loss, count in zip(batch['source'], rows, batch['valid_tokens'].tolist()):
                    entry = sources.setdefault(source, {'loss_sum': 0., 'tokens': 0})
                    entry['loss_sum'] += loss
                    entry['tokens'] += count
    finally:
        model.train(was_training)
        c.restore_rng(random_before)
    count = sum(row['tokens'] for row in sources.values())
    c.require(count == dataset.total_target_tokens, 'SFT evaluation missed assistant targets')
    average = sum(row['loss_sum'] for row in sources.values())/count
    c.require(math.isfinite(average), 'Nonfinite SFT validation loss')
    return {'loss': average, 'perplexity': math.exp(min(average, 80)), 'tokens': count,
            'sources': {name: {'loss': row['loss_sum']/row['tokens'], 'tokens': row['tokens']}
                        for name, row in sources.items()}}


def require_readiness(config, fingerprint):
    pointer = c.read_json(c.ROOT/'runs/CURRENT_SFT_READINESS.json')
    report = c.read_json(pointer['report'])
    c.require(report['status'] == 'passed' and report['optimizer_updates'] == 0, 'Passed zero-update SFT acceptance required')
    c.require(report['config_sha256'] == c.config_digest(config), 'SFT config changed since acceptance')
    c.require(report['dataset_fingerprint'] == fingerprint, 'SFT inputs changed since acceptance')
    c.require(report['code_sha256'] == code_receipts(), 'SFT code changed since acceptance')
    c.require(report['runtime_versions'] == c.runtime_versions(), 'Training runtime changed since acceptance')
    return report


def retention_dataset(config):
    try:
        from .lm_data import PackedTokenDataset
    except ImportError:
        from lm_data import PackedTokenDataset
    base_config = c.read_json(c.resolve_path(config['base_pretraining_config']))
    root, _, _ = c.approved_data(base_config, verify_hashes=False)
    return PackedTokenDataset(root, 'validation', base_config['training']['sequence_length'])


def retention_evaluate(model, dataset, config, device):
    c.require(dataset.split == 'validation', 'Retention monitoring uses original validation ONLY')
    base_config = c.read_json(c.resolve_path(config['base_pretraining_config']))
    random_before, was_training = c.rng_state(), model.training
    try:
        return c.evaluate(model, dataset, base_config, torch.device(device),
                          max_tokens=config['training']['retention_max_tokens'])
    finally:
        c.restore_rng(random_before)
        model.train(was_training)
