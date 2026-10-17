"""Lossless FP32 Llama safetensors -> GGUF, with inverse tensor verification.

Q/K layout permutation follows ggml-org/llama.cpp conversion/llama.py.
Native serving still uses the original tokenizer.json and sft_format.py, never
the native runtime's chat template or tokenizer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import gguf
import numpy as np
from safetensors import safe_open


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_name(name):
    fixed = {'model.embed_tokens.weight': 'token_embd.weight',
             'model.norm.weight': 'output_norm.weight', 'lm_head.weight': 'output.weight'}
    if name in fixed: return fixed[name]
    m = re.fullmatch(r'model.layers.(\d+).(.+).weight', name)
    if not m: raise ValueError('Unexpected tensor: ' + name)
    suffix = {'self_attn.q_proj': 'attn_q', 'self_attn.k_proj': 'attn_k',
              'self_attn.v_proj': 'attn_v', 'self_attn.o_proj': 'attn_output',
              'mlp.gate_proj': 'ffn_gate', 'mlp.up_proj': 'ffn_up',
              'mlp.down_proj': 'ffn_down', 'input_layernorm': 'attn_norm',
              'post_attention_layernorm': 'ffn_norm'}[m[2]]
    return f'blk.{m[1]}.{suffix}.weight'


def layout(array, heads, inverse=False):
    tail = array.shape[1:]
    shape = (heads, array.shape[0] // heads // 2, 2, *tail) if inverse else (heads, 2, array.shape[0] // heads // 2, *tail)
    return np.ascontiguousarray(array.reshape(shape).swapaxes(1, 2).reshape(array.shape))


def export(source, destination):
    destination.mkdir(parents=True, exist_ok=False)
    hf = destination / 'model'; shutil.copytree(source, hf)
    config = json.loads((source / 'config.json').read_text())
    assert config['model_type'] == 'llama' and config['tie_word_embeddings']
    tokenizer = json.loads((source / 'tokenizer.json').read_text(encoding='utf8'))
    vocab = tokenizer['model']['vocab']
    tokens = [None] * config['vocab_size']
    for token, index in vocab.items(): tokens[index] = token
    assert all(isinstance(t, str) for t in tokens)
    output = destination / 'clover-v0.1-f32.gguf'
    writer = gguf.GGUFWriter(output, 'llama')
    writer.add_name('Clover 125M SFTv3 v0.1 RC1')
    writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    writer.add_context_length(config['max_position_embeddings'])
    writer.add_embedding_length(config['hidden_size'])
    writer.add_block_count(config['num_hidden_layers'])
    writer.add_feed_forward_length(config['intermediate_size'])
    writer.add_head_count(config['num_attention_heads'])
    writer.add_head_count_kv(config['num_key_value_heads'])
    writer.add_layer_norm_rms_eps(config['rms_norm_eps'])
    writer.add_rope_dimension_count(config['head_dim'])
    writer.add_rope_freq_base(config['rope_parameters']['rope_theta'])
    writer.add_vocab_size(config['vocab_size'])
    writer.add_tokenizer_model('gpt2'); writer.add_tokenizer_pre('gpt-2')
    writer.add_token_list(tokens)
    writer.add_token_types([gguf.TokenType.CONTROL if i < 4 else gguf.TokenType.NORMAL for i in range(len(tokens))])
    merges = tokenizer['model']['merges']
    writer.add_token_merges([' '.join(m) if isinstance(m, list) else m for m in merges])
    writer.add_bos_token_id(3); writer.add_eos_token_id(1); writer.add_pad_token_id(0); writer.add_unk_token_id(2)
    writer.add_add_bos_token(False); writer.add_add_eos_token(False)
    receipts = []
    with safe_open(source / 'model.safetensors', framework='numpy') as weights:
        for name in weights.keys():
            array = weights.get_tensor(name)
            assert array.dtype == np.float32
            qk = name.endswith(('self_attn.q_proj.weight', 'self_attn.k_proj.weight'))
            heads = config['num_key_value_heads'] if name.endswith('k_proj.weight') else config['num_attention_heads']
            native = layout(array, heads) if qk else np.ascontiguousarray(array)
            writer.add_tensor(tensor_name(name), native)
            receipts.append({'source': name, 'gguf': tensor_name(name), 'shape': list(array.shape),
                             'permuted': qk, 'heads': heads, 'sha256': hashlib.sha256(array.tobytes()).hexdigest()})
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    reader = gguf.GGUFReader(output)
    native_tensors = {t.name: t for t in reader.tensors}
    assert len(native_tensors) == len(receipts)
    for receipt in receipts:
        array = np.asarray(native_tensors[receipt['gguf']].data).reshape(receipt['shape'])
        assert array.dtype == np.float32
        if receipt['permuted']: array = layout(array, receipt['heads'], inverse=True)
        assert hashlib.sha256(array.tobytes()).hexdigest() == receipt['sha256'], receipt['source']
    report = {'status': 'passed', 'precision': 'FP32 (no quantization)', 'all_tensors_bitwise_reversible': True,
              'tensor_count': len(receipts), 'parameters': sum(int(np.prod(r['shape'])) for r in receipts),
              'source_sha256': sha(source / 'model.safetensors'), 'gguf_sha256': sha(output),
              'tokenizer_sha256': sha(hf / 'tokenizer.json'), 'tensors': receipts,
              'native_tokenization_used': False,
              'upstream_layout_reference': 'https://github.com/ggml-org/llama.cpp/blob/master/conversion/llama.py'}
    (destination / 'conversion.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps({k: v for k, v in report.items() if k != 'tensors'}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path); p.add_argument('destination', type=Path)
    args = p.parse_args(); export(args.source, args.destination)
