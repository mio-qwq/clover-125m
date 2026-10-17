"""Clover round three. Default prints configuration; user explicitly starts training."""
from pathlib import Path
import sys
try:
    from . import train_sft as core
    from . import sft_round3_common as third
except ImportError:
    import train_sft as core
    import sft_round3_common as third


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if not any(arg=='--config' or arg.startswith('--config=') for arg in argv):
        argv=['--config',str(third.DEFAULT_CONFIG),*argv]
    previous_common,previous_export=core.common,core.export_tokenizer
    def export(base,out,config):
        third.c.require(third.c.sha256(third.c.resolve_path(config['identity_profile']))==config['identity_profile_sha256'],'Identity profile changed')
        receipts=previous_export(base,out,config)
        third.c.write_new_json(Path(out)/'assistant_profile.json',third.c.read_json(third.c.resolve_path(config['identity_profile'])))
        receipts['assistant_profile.json']=third.c.sha256(Path(out)/'assistant_profile.json')
        return receipts
    core.common=lambda:third;core.export_tokenizer=export
    try: return core.main(argv)
    finally: core.common,core.export_tokenizer=previous_common,previous_export


if __name__=='__main__': main()
