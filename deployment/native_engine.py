"""CPU FP32 inference using unchanged project tokenizer and role serialization."""
from pathlib import Path
import json
import time
import numpy as np
from tokenizers import Tokenizer, decoders
from llama_cpp import Llama, llama_cpp
from scripts.sft_format import encode_prompt
from scripts.serve_sft_web import fit_history


def sample(logits, history, params, rng):
    scores = np.asarray(logits, dtype=np.float64).copy()
    seen = np.unique(history)
    values = scores[seen]
    scores[seen] = np.where(values < 0, values * params['repetition_penalty'], values / params['repetition_penalty'])
    n = params['no_repeat_ngram_size']
    if n and len(history) + 1 >= n:
        prefix = tuple(history[-(n - 1):]) if n > 1 else ()
        banned = [history[i + n - 1] for i in range(len(history) - n + 1)
                  if tuple(history[i:i + n - 1]) == prefix]
        scores[banned] = -np.inf
    scores[[0, 2, 3]] = -np.inf
    if not params['do_sample']: return int(np.argmax(scores))
    scores /= params['temperature']
    k = params['top_k']
    if k: scores[scores < np.partition(scores, -k)[-k]] = -np.inf
    order = np.argsort(-scores, kind='stable')
    probs = np.exp(scores[order] - np.max(scores))
    probs /= probs.sum()
    remove = np.cumsum(probs) - probs >= params['top_p']
    probs[remove] = 0.; probs /= probs.sum()
    return int(rng.choice(order, p=probs))


class ModelEngine:
    def __init__(self, root, context=1024, threads=2):
        root = Path(root)
        self.context = context; self.device = 'cpu'
        self.release = json.loads((root / 'release.json').read_text(encoding='utf8'))
        self.label = 'Clover · v' + self.release['version']
        self.tokenizer = Tokenizer.from_file(str(root / 'tokenizer.json'))
        self.model = Llama(model_path=str(root / 'gguf/Clover-125M-v0.1.0-F32.gguf'), n_ctx=context,
            n_batch=32, n_ubatch=32, n_threads=threads, n_threads_batch=threads,
            n_gpu_layers=0, use_mmap=True, use_mlock=False, logits_all=False,
            type_k=llama_cpp.GGML_TYPE_F32, type_v=llama_cpp.GGML_TYPE_F32,
            flash_attn=False, offload_kqv=False, verbose=False)
        assert self.model.n_vocab() == 32000

    def prepare(self, messages, params):
        return fit_history(messages, lambda m: encode_prompt(m, self.tokenizer),
                           min(self.context, params['context_tokens']), params['max_new_tokens'])

    def logits(self):
        return np.ctypeslib.as_array(self.model._ctx.get_logits(), shape=(32000,)).copy()

    def generate(self, plan, params, job):
        self.model.reset()
        prompt = plan[1]
        deadline = time.monotonic() + 90
        for start in range(0, len(prompt), 32):
            if job.cancel.is_set() or time.monotonic() > deadline:
                return {'output_tokens': 0, 'finish_reason': 'stopped'}
            self.model.eval(prompt[start:start + 32])
        rng = np.random.default_rng(params['seed'])
        history = list(prompt); output = []
        decoder = decoders.DecodeStream(skip_special_tokens=True)
        reason = 'length'
        for _ in range(params['max_new_tokens']):
            if job.cancel.is_set(): reason = 'stopped'; break
            if time.monotonic() > deadline: reason = 'timeout'; break
            token = sample(self.logits(), history, params, rng)
            history.append(token); output.append(token)
            if token == 1: reason = 'eos'; break
            text = decoder.step(self.tokenizer, token)
            if text: job.events.put({'type': 'delta', 'text': text})
            self.model.eval([token])
        return {'output_tokens': len(output), 'finish_reason': reason}
