"""CPU streaming with the released FP32 GGUF and the exact Clover tokenizer."""
import argparse
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deployment.native_engine import ModelEngine
from scripts.serve_sft_web import DEFAULTS, parameters


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--greedy', action='store_true')
    p.add_argument('--temperature', type=float, default=.35)
    p.add_argument('--top-p', type=float, default=.85)
    p.add_argument('--top-k', type=int, default=30)
    p.add_argument('--repetition-penalty', type=float, default=1.1)
    p.add_argument('--no-repeat-ngram-size', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.threads < 1: p.error('--threads must be positive')
    options = {key: getattr(args, key) for key in DEFAULTS if hasattr(args, key)}
    options['do_sample'] = not args.greedy
    options = parameters(options)
    engine = ModelEngine(args.model_dir, threads=args.threads)
    plan = engine.prepare([{'role': 'user', 'content': args.prompt}], options)
    class Sink:
        def put(self, event):
            if event.get('type') == 'delta': print(event['text'], end='', flush=True)
    job = SimpleNamespace(cancel=threading.Event(), events=Sink())
    try: engine.generate(plan, options, job)
    finally: print(flush=True); engine.model.close()


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf8')
    try: main()
    except KeyboardInterrupt: raise SystemExit(130)
