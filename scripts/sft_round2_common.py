"""Round-two input/readiness adapter; frozen round-one training utilities reused."""
from pathlib import Path
try:
    from . import lm_common as c
    from . import sft_common as first
except ImportError:
    import lm_common as c
    import sft_common as first

DEFAULT_CONFIG = c.ROOT/'configs/english_125m_sft_round2.json'
SFTDataset = first.SFTDataset
collate = first.collate
epoch_batches = first.epoch_batches
loader = first.loader
forward_loss_sum = first.forward_loss_sum
evaluate = first.evaluate
retention_dataset = first.retention_dataset
retention_evaluate = first.retention_evaluate
build_base_model = first.build_base_model


def load_config(path=DEFAULT_CONFIG):
    config = first.load_config(path)
    c.require(config['sft_round'] == 2 and config['assistant_name'] == 'Clover', 'Expected Clover round two')
    c.require(c.read_json(c.resolve_path(config['identity_profile']))['name'] == config['assistant_name'], 'Identity profile mismatch')
    c.require(c.sha256(c.resolve_path(config['identity_profile'])) == config['identity_profile_sha256'], 'Identity profile hash mismatch')
    c.require(config['training']['output_root'] == 'runs/sft_125m_round2', 'Use separate round-two outputs')
    return config


def code_receipts():
    files = ['scripts/stage_sft_round2.py', 'scripts/download_sft_round2_expansion.py', 'scripts/curate_sft_round2_release.py', 'scripts/prepare_sft_round2.py',
        'scripts/verify_sft_round2_data.py', 'scripts/sft_round2_common.py',
        'scripts/train_sft_round2.py', 'scripts/verify_sft_round2_ready.py',
        'tests/test_sft_round2_data.py', 'tests/test_sft_round2_training.py']
    return {**first.code_receipts(), **{name: c.sha256(c.ROOT/name) for name in files}}


def approved_inputs(config, verify_hashes=True):
    model_dir = c.resolve_path(config['base_model_dir'])
    manifest_path = c.resolve_path(config['base_manifest'])
    parent = c.read_json(manifest_path)
    c.require(model_dir.is_relative_to((c.ROOT/'runs/sft_inputs').resolve()), 'Independent parent copy required')
    c.require(parent['status'] == 'verified' and parent['kind'] == 'sft_round_copy' and parent['parent_round'] == 1,
              'Verified round-one SFT copy required')
    c.require(parent['source_training_state']['completed'] and not parent['optimizer_state_reused'], 'Use completed weights with fresh optimizer')
    c.require(parent['base_pretraining_config_sha256'] == c.sha256(c.resolve_path(config['base_pretraining_config'])), 'Architecture receipt mismatch')
    archive = Path(parent['archive_manifest'])
    c.require(c.sha256(archive) == parent['archive_manifest_sha256'], 'Parent archive manifest changed')
    pointer_path = c.resolve_path(config['dataset_pointer'])
    pointer = c.read_json(pointer_path); root = Path(pointer['prepared_dir']).resolve()
    c.require(root.is_relative_to((c.ROOT/'data/processed/sft_en_round2').resolve()), 'Unexpected round-two data location')
    manifest = c.read_json(root/'manifest.json'); release = c.read_json(root/'release_validation.json')
    c.require(pointer['status'] == 'validated' and manifest['status'] == 'completed' and release['status'] == 'passed', 'Passed independent data acceptance required')
    c.require(not manifest['sample_only'] and manifest['sft_round'] == 2 and manifest['language'] == 'English', 'Full English round-two data required')
    c.require(manifest['sequence_length'] <= config['training']['sequence_length'], 'Prepared sequences exceed SFT context')
    c.require(manifest['assistant_name'] == config['assistant_name'], 'Data identity name mismatch')
    c.require(release['manifest_sha256'] == c.sha256(root/'manifest.json'), 'Stale round-two acceptance')
    c.require(release['template_code_sha256'] == c.sha256(c.ROOT/'scripts/sft_format.py'), 'Formatter changed')
    for key in ['all_reencoded_equal', 'all_selected_provenance_matched', 'group_split_leakage_check',
                'prior_holdouts_not_in_training', 'prior_train_not_new_holdout', 'raw_metadata_unchanged', 'input_hashes_unchanged']:
        c.require(release[key] is True, 'Data acceptance missing: '+key)
    c.require(manifest['profile_sha256'] == c.sha256(c.resolve_path(config['identity_profile'])), 'Profile changed after preparation')
    c.require(manifest['tokenizer_sha256'] == c.sha256(model_dir/'tokenizer.json'), 'Parent tokenizer changed')
    if verify_hashes:
        first._verify_files(parent['files'], model_dir.parent)
        first._verify_files(manifest['files'], root)
    base_config = c.read_json(c.resolve_path(config['base_pretraining_config']))
    _, _, retention = c.approved_data(base_config, verify_hashes=verify_hashes)
    fp = {'sft_round': 2, 'parent_manifest_sha256': c.sha256(manifest_path),
        'parent_checkpoint_sha256': parent['source_checkpoint_sha256'],
        'model_sha256': c.sha256(model_dir/'model.safetensors'), 'dataset_pointer_sha256': c.sha256(pointer_path),
        'manifest_sha256': c.sha256(root/'manifest.json'), 'release_sha256': c.sha256(root/'release_validation.json'),
        'tokenizer_sha256': manifest['tokenizer_sha256'], 'template_version': manifest['template_version'],
        'template_code_sha256': c.sha256(c.ROOT/'scripts/sft_format.py'),
        'identity_profile_sha256': manifest['profile_sha256'], 'assistant_name': config['assistant_name'],
        'retention_dataset_fingerprint': retention, 'retention_max_tokens': config['training']['retention_max_tokens']}
    return base_config, model_dir, root, fp


def require_readiness(config, fingerprint):
    pointer = c.read_json(c.ROOT/'runs/CURRENT_SFT_ROUND2_READINESS.json')
    report = c.read_json(pointer['report'])
    c.require(report['status'] == 'passed' and report['optimizer_updates'] == 0, 'Zero-update acceptance required')
    c.require(report['config_sha256'] == c.config_digest(config), 'Round-two config changed')
    c.require(report['dataset_fingerprint'] == fingerprint, 'Round-two inputs changed')
    c.require(report['code_sha256'] == code_receipts(), 'Round-two code changed')
    c.require(report['runtime_versions'] == c.runtime_versions(), 'Training runtime changed')
    return report
