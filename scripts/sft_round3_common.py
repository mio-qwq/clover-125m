"""Round-three inputs/readiness seam; frozen training and loss implementation."""
import random
try:
    from . import lm_common as c
    from . import sft_common as first
except ImportError:
    import lm_common as c
    import sft_common as first

DEFAULT_CONFIG=c.ROOT/'configs/english_125m_sft_round3.json'
SFTDataset=first.SFTDataset
collate=first.collate
loader=first.loader
forward_loss_sum=first.forward_loss_sum
evaluate=first.evaluate
retention_dataset=first.retention_dataset
retention_evaluate=first.retention_evaluate
build_base_model=first.build_base_model


def epoch_batches(dataset,batch_size,seed,epoch,shuffle=True):
    """Homogeneous full UPDATE blocks keep short targets from being drowned out.

    Every example appears once. Complete blocks are shuffled across task groups;
    leftovers form the final updates, preserving core resume boundaries and
    ceil(examples / effective batch) updates. No oversampling/duplication.
    """
    if not shuffle: return first.epoch_batches(dataset,batch_size,seed,epoch,shuffle=False)
    rng=random.Random(f'clover-r3:{seed}:{epoch}')
    buckets={'reading':[],'curriculum':[],'replay':[]}
    for index,row in enumerate(dataset.items):
        source=row['source']
        key='reading' if source.startswith('squad_') else 'curriculum' if source=='targeted_curriculum' else 'replay'
        buckets[key].append(index)
    block_size=batch_size*8;blocks=[];tail=[]
    for indices in buckets.values():
        rng.shuffle(indices);boundary=len(indices)//block_size*block_size
        blocks.extend(indices[start:start+block_size] for start in range(0,boundary,block_size))
        tail.extend(indices[boundary:])
    rng.shuffle(blocks);rng.shuffle(tail)
    while len(tail)>=block_size:
        blocks.append(tail[:block_size]);tail=tail[block_size:]
    if tail: blocks.append(tail)
    return [block[start:start+batch_size] for block in blocks for start in range(0,len(block),batch_size)]


def load_config(path=DEFAULT_CONFIG):
    config=first.load_config(path)
    c.require(config['sft_round']==3 and config['assistant_name']=='Clover','Expected Clover round three')
    c.require(config['training']['output_root']=='runs/sft_125m_round3','Separate round-three output required')
    c.require(config['training']['micro_batch_size']==4 and config['training']['gradient_accumulation_steps']==8,
              'Accepted task-block plan requires micro batch 4 and accumulation 8')
    c.require(c.sha256(c.resolve_path(config['identity_profile']))==config['identity_profile_sha256'],'Identity profile changed')
    return config


def code_receipts():
    files=['scripts/stage_sft_round3.py','scripts/stage_sft_round2.py','scripts/prepare_sft_round3.py',
        'scripts/prepare_sft_round2.py','scripts/verify_sft_round2_data.py','scripts/verify_sft_round3_data.py',
        'scripts/curate_sft_round3_release.py','configs/clover_round3_review.json',
        'scripts/sft_round3_common.py','scripts/train_sft_round3.py','scripts/verify_sft_round3_ready.py',
        'tests/test_sft_round3_data.py','tests/test_sft_round3_training.py']
    return {**first.code_receipts(),**{name:c.sha256(c.ROOT/name) for name in files}}


def approved_inputs(config,verify_hashes=True):
    model_dir=c.resolve_path(config['base_model_dir']);parent_path=c.resolve_path(config['base_manifest'])
    parent=c.read_json(parent_path)
    c.require(model_dir.is_relative_to((c.ROOT/'runs/sft_inputs').resolve()),'Independent parent copy required')
    c.require(parent['status']=='verified' and parent['kind']=='sft_round_copy' and parent['parent_round']==2,'Verified round-two parent required')
    c.require(parent['source_training_state']['completed'] and not parent['optimizer_state_reused'],'Completed parent and fresh optimizer required')
    c.require(parent['base_pretraining_config_sha256']==c.sha256(c.resolve_path(config['base_pretraining_config'])),'Parent architecture changed')
    c.require(c.sha256(parent['archive_manifest'])==parent['archive_manifest_sha256'],'Parent archive receipt changed')
    pointer_path=c.resolve_path(config['dataset_pointer']);pointer=c.read_json(pointer_path)
    root=c.resolve_path(pointer['prepared_dir']);manifest=c.read_json(root/'manifest.json');release=c.read_json(root/'release_validation.json')
    c.require(root.is_relative_to((c.ROOT/'data/processed/sft_en_round3').resolve()),'Round-three data only')
    c.require(pointer['status']=='validated' and manifest['status']=='completed' and release['status']=='passed','Passed data release required')
    c.require(manifest['sft_round']==3 and manifest['language']=='English' and not manifest['sample_only'],'Full English round three required')
    c.require(manifest['sequence_length']<=config['training']['sequence_length'],'Data exceeds configured context')
    c.require(not manifest['language_model_training_started'],'Preparation must not train')
    c.require(release['manifest_sha256']==c.sha256(root/'manifest.json'),'Stale data acceptance')
    c.require(release['template_code_sha256']==c.sha256(c.ROOT/'scripts/sft_format.py'),'Formatter changed')
    for key in ('all_reencoded_equal','all_selected_provenance_matched','group_split_leakage_check',
        'prior_holdouts_not_in_training','prior_train_not_new_holdout','input_hashes_unchanged',
        'raw_metadata_unchanged','training_runtime_reencoding_passed'):
        c.require(release[key] is True,'Missing data check: '+key)
    c.require(manifest['tokenizer_sha256']==c.sha256(model_dir/'tokenizer.json'),'Tokenizer changed')
    c.require(manifest['profile_sha256']==config['identity_profile_sha256'],'Data identity changed')
    if verify_hashes:
        first._verify_files(parent['files'],model_dir.parent)
        first._verify_files(c.read_json(parent['archive_manifest'])['files'],c.resolve_path(parent['archive_manifest']).parent)
        first._verify_files(manifest['files'],root)
    base=c.read_json(c.resolve_path(config['base_pretraining_config']))
    _,_,retention=c.approved_data(base,verify_hashes=verify_hashes)
    fingerprint=dict(sft_round=3,parent_manifest_sha256=c.sha256(parent_path),parent_checkpoint_sha256=parent['source_checkpoint_sha256'],
        model_sha256=c.sha256(model_dir/'model.safetensors'),dataset_pointer_sha256=c.sha256(pointer_path),
        manifest_sha256=c.sha256(root/'manifest.json'),release_sha256=c.sha256(root/'release_validation.json'),
        tokenizer_sha256=manifest['tokenizer_sha256'],template_version=manifest['template_version'],
        template_code_sha256=c.sha256(c.ROOT/'scripts/sft_format.py'),identity_profile_sha256=manifest['profile_sha256'],
        assistant_name='Clover',retention_dataset_fingerprint=retention,retention_max_tokens=config['training']['retention_max_tokens'])
    return base,model_dir,root,fingerprint


def require_readiness(config,fingerprint):
    pointer=c.read_json(c.ROOT/'runs/CURRENT_SFT_ROUND3_READINESS.json');report=c.read_json(pointer['report'])
    c.require(report['status']=='passed' and report['optimizer_updates']==0,'Passed zero-update acceptance required')
    c.require(report['config_sha256']==c.config_digest(config),'Round-three recipe changed')
    c.require(report['dataset_fingerprint']==fingerprint,'Round-three inputs changed')
    c.require(report['code_sha256']==code_receipts(),'Round-three code changed')
    c.require(report['runtime_versions']==c.runtime_versions(),'Training runtime changed')
    return report
