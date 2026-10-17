"""Offline, streaming Clover-125M inference from a standalone HF export."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

try:
    from sft_format import encode_prompt
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.sft_format import encode_prompt


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, default=Path(__file__).resolve().parent)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt')
    source.add_argument('--messages-json', type=Path)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--greedy', action='store_true', help='Disable sampling')
    p.add_argument('--temperature', type=float, default=.35)
    p.add_argument('--top-p', type=float, default=.85)
    p.add_argument('--top-k', type=int, default=30)
    p.add_argument('--repetition-penalty', type=float, default=1.1)
    p.add_argument('--no-repeat-ngram-size', type=int, default=4)
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=2)
    return p


def validate(args):
    if not (1 <= args.max_new_tokens <= 512 and 0 <= args.seed <= 2147483647 and args.threads > 0):
        raise ValueError('Invalid output length, seed or CPU thread count')
    if not (math.isfinite(args.temperature) and args.temperature > 0 and
            math.isfinite(args.top_p) and 0 < args.top_p <= 1 and 0 <= args.top_k <= 32000 and
            math.isfinite(args.repetition_penalty) and args.repetition_penalty > 0 and
            0 <= args.no_repeat_ngram_size <= 32):
        raise ValueError('Invalid sampling or repetition controls')


def load(args):
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM
    validate(args)
    root = args.model_dir.resolve()
    template = json.loads((root / 'sft_template.json').read_text(encoding='utf8'))
    formatter = Path(sys.modules[encode_prompt.__module__].__file__)
    if hashlib.sha256(formatter.read_bytes()).hexdigest() != template['formatter_sha256']:
        raise ValueError('Role formatter does not match the released model')
    with (root / 'tokenizer.json').open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != template['tokenizer_files_sha256']['tokenizer.json']:
        raise ValueError('Tokenizer does not match the released model')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu')
    torch.set_num_threads(args.threads)
    # Keep the same seed-before-model-load order as the original inference CLI.
    torch.manual_seed(args.seed)
    if args.device == 'cuda': torch.cuda.manual_seed_all(args.seed)
    tokenizer = Tokenizer.from_file(str(root / 'tokenizer.json'))
    model = AutoModelForCausalLM.from_pretrained(
        root, local_files_only=True, dtype=torch.float32,
        attn_implementation='sdpa', trust_remote_code=False).to(args.device).eval()
    if model.num_parameters() != 124720896 or model.config.max_position_embeddings != 1024:
        raise ValueError('Unexpected Clover model architecture')
    return model, tokenizer


def run(args, model, tokenizer, messages, stream=None):
    import torch
    from transformers import TextStreamer
    validate(args)
    ids = encode_prompt(messages, tokenizer)
    if len(ids) + args.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError('Prompt + output exceeds 1024 tokens; shorten history or output')
    stream = sys.stdout if stream is None else stream
    class Streamer(TextStreamer):
        def on_finalized_text(self, text, stream_end=False):
            stream.write(text + ('\n' if stream_end else '')); stream.flush()
    options = dict(max_new_tokens=args.max_new_tokens, do_sample=not args.greedy,
                   repetition_penalty=args.repetition_penalty,
                   no_repeat_ngram_size=args.no_repeat_ngram_size,
                   suppress_tokens=[0, 2, 3], eos_token_id=1, pad_token_id=0,
                   use_cache=True, renormalize_logits=True)
    if not args.greedy:
        options.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    x = torch.tensor([ids], dtype=torch.long, device=args.device)
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if args.device == 'cuda' else nullcontext()
    with torch.inference_mode(), amp:
        output = model.generate(x, attention_mask=torch.ones_like(x),
                                streamer=Streamer(tokenizer, skip_prompt=True, skip_special_tokens=True),
                                **options)
    return output[0, len(ids):].tolist()


def main(argv=None):
    args = parser().parse_args(argv)
    messages = json.loads(args.messages_json.read_text(encoding='utf8')) if args.messages_json else [
        {'role': 'user', 'content': args.prompt}]
    model, tokenizer = load(args)
    run(args, model, tokenizer, messages)
    return 0


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf8')
    try: raise SystemExit(main())
    except KeyboardInterrupt: raise SystemExit(130)
