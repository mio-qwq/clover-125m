# Clover-125M v0.1.0 🍀

First public source release of **Clover-125M**, a 124,720,896-parameter English language model trained from random initialization.

The model was pretrained on approximately 2.64 billion target tokens and went through three rounds of full-parameter supervised fine-tuning. This release uses the final checkpoint from SFT round 3.

- **Try it:** https://ai.mio.blue/
- **Model weights:** https://huggingface.co/mioqwq/Clover-125M
- **Source code:** https://github.com/mio-qwq/clover-125m
- **Training story:** https://mio.blue/Clover-125M/

## Model downloads and SHA-256

The model weights are hosted on Hugging Face rather than duplicated in GitHub Release assets. These are the **full 64-character SHA-256 digests** recorded in the model's release metadata and conversion checks:

```text
3fa7af023a29ac684c09f66dc40b80ad695681f43be6fa50759f79bed941c79c  model.safetensors
c8649696f1a89d73d59ae0ed7b9635ec7ce1b61db86ed5ddc9d3b64eb296da00  gguf/Clover-125M-v0.1.0-F32.gguf
43fb4fb8c8eedfef2bc57675c3dd980d505c3cd25126474a2f3269192baa2668  tokenizer.json
74094a77a5da6a99c43561f73625b17fefdf8d2e63cbe38ae8253ab2b791d834  sft_format.py
```

Verify the downloaded artifacts locally with `sha256sum` (Linux/WSL) or `Get-FileHash -Algorithm SHA256` (PowerShell). The digests above match `release.json` and the existing conversion/provenance records; they are **not a fresh independent re-download and re-hash of Hugging Face files**.

`SHA256SUMS` in this repository covers the Git-tracked source files (excluding itself). It is separate from the Hugging Face model weight checksums shown above.

## What's included

The public repository contains the model/training configurations, pretraining and SFT implementations, FP32 GGUF conversion, CPU/CUDA inference scripts, the native CPU web server, and the Clover streaming chat interface. It also includes training and deployment notes, conversion checks and relevant third-party attribution.

The published weights are **FP32**, with no quantization, pruning or additional training performed for the GGUF export. Recorded conversion checks compare all 128 tensors and reference inference outputs.

Raw/processed training data, data collection and cleaning tools, and private training inputs are not part of this source release. The repository documents the implementation, but it is not a one-command reproduction of the original training run.

## License

Original Clover-125M project code and model weights: **Apache License 2.0**. Third-party components and source materials retain their own terms; see [LICENSE](LICENSE), [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Clover is a small experimental language model. It may repeat, misunderstand requests, or confidently produce incorrect information.
