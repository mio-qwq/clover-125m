"""Clover round-two Full SFT, configuration only unless --start-training.

Uses the frozen training loop through its common-module seam. Weights start from
an independent completed round-one export; optimizer/scheduler/counters are new.
Round-one resume and data/readiness pointers remain untouched.
"""
from pathlib import Path
import sys
try:
    from . import train_sft as core
    from . import sft_round2_common as second
except ImportError:
    import train_sft as core
    import sft_round2_common as second


def export_tokenizer(base_dir, export, config, original):
    second.c.require(second.c.sha256(second.c.resolve_path(config['identity_profile'])) == config['identity_profile_sha256'], 'Identity profile changed before export')
    receipts = original(base_dir, export, config)
    profile = second.c.read_json(second.c.resolve_path(config['identity_profile']))
    second.c.write_new_json(Path(export)/'assistant_profile.json', profile)
    receipts['assistant_profile.json'] = second.c.sha256(Path(export)/'assistant_profile.json')
    return receipts


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(arg == '--config' or arg.startswith('--config=') for arg in argv):
        argv = ['--config', str(second.DEFAULT_CONFIG), *argv]
    previous_common, previous_export = core.common, core.export_tokenizer
    core.common = lambda: second
    core.export_tokenizer = lambda base, export, config: export_tokenizer(base, export, config, previous_export)
    try:
        return core.main(argv)
    finally:
        core.common, core.export_tokenizer = previous_common, previous_export


if __name__ == '__main__': main()
