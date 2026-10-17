# Clover-125M inference

Clover-125M can be tried directly at **[ai.mio.blue](https://ai.mio.blue/)**. For local inference, this repository provides both a GGUF-based CPU runtime and a PyTorch implementation for CPU or NVIDIA CUDA.

The examples below use **Python 3.12** and assume that the files from [Hugging Face](https://huggingface.co/mioqwq/Clover-125M) are stored in `models/Clover-125M/`, relative to the project root. The native CPU and PyTorch runtimes have different dependencies, so separate virtual environments are recommended.

## Native CPU inference

The GGUF runtime is the simplest way to run Clover on a CPU. It loads the unquantized FP32 GGUF weights through llama.cpp and does not require PyTorch or CUDA.

The model directory must include `tokenizer.json`, `release.json`, and `gguf/Clover-125M-v0.1.0-F32.gguf`. Install the native dependencies, then run a prompt from the project root:

```sh
python -m pip install -r requirements-native.txt
python -B inference/generate_gguf.py --model-dir models/Clover-125M --prompt "Hello! Who are you?"
```

The CLI uses sampling by default. Add `--greedy` to select the highest-probability token at each step.

The same native runtime can serve the local web interface:

```sh
python -B -m deployment.serve_public --root models/Clover-125M --origin http://127.0.0.1:8766
```

The interface is then available at **http://127.0.0.1:8766/**. It supports streaming replies, multi-turn history, sampling controls and cancellation. Server deployment details, including reverse proxy examples, are documented in [CPU deployment](CPU_2C1G.md).

The native engine uses mmap, FP32 weights and FP32 KV cache. By default, it runs with two CPU threads and a 1024-token context. Generation has an approximately 90-second deadline. Actual memory use depends on the runtime, context length and operating system, not just the GGUF file size.

## PyTorch inference

The PyTorch version uses the exported Safetensors weights and supports both CPU and CUDA. For CPU inference, install PyTorch from its official CPU index and then install the remaining dependencies:

```sh
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-inference.txt
python -B inference/generate.py --model-dir models/Clover-125M --prompt "Explain what Linux is in two short sentences." --device cpu
```

For NVIDIA CUDA, install the corresponding PyTorch build from `https://download.pytorch.org/whl/cu130` in a separate environment, then use `--device cuda`. The CPU path performs FP32 computation. On CUDA, the stored weights remain FP32 while BF16 autocast is used for computation. CPU inference was tested for this version; the CUDA path was not independently re-benchmarked for the release.

The PyTorch script also supports conversation input through `--messages-json`. For example, save the following as `conversation.json`:

```json
[
  {"role": "user", "content": "Hello! Who are you?"},
  {"role": "assistant", "content": "Clover is a small English language model."},
  {"role": "user", "content": "What is one simple way to relax?"}
]
```

Then run:

```sh
python -B inference/generate.py --model-dir models/Clover-125M --messages-json conversation.json --device cpu
```

An optional System message may appear first. After that, User and Assistant messages must alternate, with the last message coming from User.

## Sampling and context

By default, the CLI generates up to **128 new tokens** with temperature **0.35**, top-p **0.85**, and top-k **30**. Repetition penalty is **1.1**, no-repeat n-gram size is **4**, the default CLI seed is **42**, and the CPU thread count is **2**.

These values can be adjusted with flags such as `--max-new-tokens`, `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`, `--no-repeat-ngram-size`, `--seed`, and `--threads`. The `--greedy` flag disables sampling.

The model's context length is **1024 tokens**, including role markers, conversation history and the reserved generation space. If the input is too long, the PyTorch CLI rejects it. The web/native path instead discards the oldest complete conversation pairs while preserving the System message and latest User message when possible.

## Message formatting and compatibility

Clover uses a custom conversation format defined in `sft_format.py`. It is important to use the original `tokenizer.json` and to encode message headers, content and separators as separate segments. EOS token ID **1** ends Assistant turns.

The built-in chat templates used by other inference frameworks may not produce the same token sequence. Running the GGUF directly through another frontend therefore requires matching Clover's message formatting rather than assuming a standard Llama template.

The published GGUF contains the original FP32 weights without quantization or vocabulary changes. Its internal name may still contain `RC1/SFTv3`; the public filename and `release.json` identify v0.1.0. The original file bytes are retained so that the recorded artifact checksum remains valid.

The Hugging Face repository also provides standalone `generate.py` and `sft_format.py`, allowing PyTorch inference without the training repository. Neither inference path requires a serialized Python pickle checkpoint.
