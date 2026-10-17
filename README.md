# Clover-125M 🍀

**A 125-million-parameter English language model trained from scratch.**

**[Try Clover online →](https://ai.mio.blue/)** · [简体中文](README.zh-CN.md) · [Training notes](docs/TRAINING.md) · [Inference guide](docs/INFERENCE.md) · [Server deployment](docs/CPU_2C1G.md)

Clover-125M is a small language model trained from scratch. It first learned to predict the next token from English pretraining data, then went through three rounds of full-parameter supervised fine-tuning (SFT). The finished model has **124,720,896 parameters**.

This repository contains the training code, inference tools, and the source for its streaming chat website. Clover is a small model, so it's best suited to trying short conversations in English, exploring the training process, and experimenting with language models on everyday hardware.

**Project links**

- **Try Clover online:** [ai.mio.blue](https://ai.mio.blue/)
- **Model weights:** [Hugging Face · Clover-125M](https://huggingface.co/mioqwq/Clover-125M)
- **Source code:** [mio-qwq/clover-125m](https://github.com/mio-qwq/clover-125m)
- **Blog post:** [mio.blue/Clover-125M](https://mio.blue/Clover-125M/)

## Model details

| Item | Specification |
| --- | --- |
| Version | v0.1.0, final checkpoint from SFT round 3 |
| Architecture | Llama-style decoder, trained from random initialization |
| Parameters | 124,720,896 (~125M) |
| Layers / hidden size | 14 / 768 |
| Attention | 12 attention heads, 12 KV heads, head dimension 64 |
| FFN dimension | 2,080 |
| Tokenizer | English ByteLevel BPE, 32,000-token vocabulary |
| Context window | 1,024 tokens |
| Weight formats | FP32 `safetensors` / FP32 GGUF (unquantized) |
| Training GPU | NVIDIA RTX 5070 |

Clover uses the Llama architecture implemented in Transformers, but **does not use pretrained Meta Llama weights**. Its input and output embeddings share weights, and all three SFT rounds use the tokenizer from pretraining.

## Training

| Stage | Focus | Epochs | Parameter updates | Supervised target tokens |
| --- | --- | ---: | ---: | ---: |
| Pretraining | English text completion | 1 | 80,465 | 2,636,654,462 |
| SFT 1 | Short instructions and role formatting | 1 | 584 | 2,334,382 |
| SFT 2 | Everyday conversations and Clover's identity | 2 | 1,506 | 11,973,174 |
| SFT 3 | Reading provided text, recognizing missing information, and short tasks | 1 | 574 | 447,475 |

The release uses the **SFT 3 final** checkpoint. The model was not reinitialized between training stages. In the recorded evaluations, the reading test for round 3 had a Loss / PPL of **0.439808 / 1.552409**, while the dialogue test for round 2 had **2.152472 / 8.606110**.

For the data sources and evaluation details, see the [training notes](docs/TRAINING.md).

## Quick start

**[Chat with Clover online →](https://ai.mio.blue/)**

You can try the deployed web version directly, without downloading the model or setting up Python. Clover was trained primarily on English, so English prompts are recommended.

## Run locally

You'll need **Python 3.12**. Download the model files from [Hugging Face](https://huggingface.co/mioqwq/Clover-125M) and put them in `models/Clover-125M/` inside the source repository.

For native CPU inference, you'll need at least these files:

- `tokenizer.json`
- `release.json`
- `gguf/Clover-125M-v0.1.0-F32.gguf`

Create and activate a fresh Python virtual environment, then run the following commands from the repository root:

```sh
python -m pip install -r requirements-native.txt
python -B inference/generate_gguf.py --model-dir models/Clover-125M --prompt "Hello! Who are you?"
```

To start the local web interface, run:

```sh
python -B -m deployment.serve_public --root models/Clover-125M --origin http://127.0.0.1:8766
```

Then open **http://127.0.0.1:8766/**.

The web interface supports streaming responses, multi-turn conversations, sampling controls, stopping generation, and retrying when the server is busy. Since Clover was trained mainly on English, English prompts work best. For PyTorch inference on CPU or CUDA, see the [inference guide](docs/INFERENCE.md). Example nginx and systemd configurations are available in `deployment/examples/`; replace the example domains and paths before deploying.

## Can it run on very limited hardware?

Clover has run on a server with **2 vCPUs, 887 MiB of visible RAM, about 1 GiB of existing swap, and no GPU**. It used the **original FP32 GGUF** weights, without quantization, pruning, or additional training for deployment.

The setup uses mmap, two CPU threads, a single active generation, and a limited context window. One recorded snapshot showed process RSS at around **508 MiB**, with a high-water mark of around **585 MiB** and about **118 MiB** swapped out.

The memory calculations, observed performance, and deployment commands are in the [deployment notes](docs/CPU_2C1G.md).

## Repository contents and reproducibility

| Directory | Contents |
| --- | --- |
| `scripts/` | Pretraining, SFT, batch loading, and conversation formatting |
| `configs/` | Model architecture and training configurations |
| `inference/` | PyTorch and native GGUF inference |
| `deployment/` | FP32 conversion, server code, and deployment examples |
| `web/clover/` | Streaming chat interface |
| `docs/` | Training and deployment documentation |
| `provenance/` | Evaluation results, conversion checks, and source attribution |

The repository includes the core training implementation, but not the raw or processed training data, the collection and cleaning pipeline, or all private verification materials. **Cloning the code alone is not enough to reproduce the entire training run.** You can, however, run inference and use the web interface with the exported model weights.

**Clover can still misunderstand questions, repeat itself, or invent information. It's an experimental small language model, so please check any important output.**

**License:** Original Clover-125M project code and model weights are licensed under the **[Apache License 2.0](LICENSE)**, which permits use, modification, and redistribution under its terms. Third-party code and data retain their own licenses. See [License details](LICENSE_STATUS.md), [NOTICE](NOTICE), and [Third-party notices](THIRD_PARTY_NOTICES.md).
