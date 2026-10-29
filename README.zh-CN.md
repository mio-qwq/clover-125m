# Clover-125M 🍀

**一个从零开始训练的 1.25 亿参数英文语言模型。**

> **在线体验自 2026 年 10 月 29 日起关闭。** 网站现仅保留项目介绍。
> 如需体验，请前往 [Hugging Face 下载模型](https://huggingface.co/mioqwq/Clover-125M)，并在本地运行。

**[Clover 项目介绍 →](https://ai.mio.blue/)** · [English](README.md) · [训练记录](docs/TRAINING.md) · [推理指南](docs/INFERENCE.md) · [服务器部署](docs/CPU_2C1G.md)

Clover-125M 是一个从零开始预训练的小型语言模型。它先通过预训练的英文语料学习预测下一个 token，再依次进行了三轮全参数监督微调（SFT）。最终模型有 **124,720,896 个参数**。

这个仓库整理了训练实现、推理程序和流式聊天网页的源码。Clover 的规模很小，更适合用来体验英文短对话、观察模型训练过程，以及研究如何在普通设备上运行语言模型。

**项目地址**

- **项目主页：**[ai.mio.blue](https://ai.mio.blue/)
- **模型权重：**[Hugging Face · Clover-125M](https://huggingface.co/mioqwq/Clover-125M)
- **源码仓库：**[mio-qwq/clover-125m](https://github.com/mio-qwq/clover-125m)
- **博客文章：**[mio.blue/Clover-125M](https://mio.blue/Clover-125M/)

## 模型信息

| 项目 | 参数 |
| --- | --- |
| 版本 | v0.1.0，第三轮 SFT 的最终权重 |
| 架构 | 从随机权重开始训练的 Llama 风格 Decoder |
| 参数量 | 124,720,896（约 125M） |
| 层数 / 隐藏维度 | 14 / 768 |
| 注意力 | 12 个 Attention Heads、12 个 KV Heads，Head Dim 64 |
| FFN 维度 | 2,080 |
| 分词器 | 32,000 词表的英文 ByteLevel BPE |
| 上下文窗口 | 1,024 tokens |
| 权重格式 | FP32 `safetensors` / FP32 GGUF（未量化） |
| 训练显卡 | NVIDIA RTX 5070 |

模型使用 Transformers 中的 Llama 结构实现，没有使用 Meta 预训练的 Llama 权重。输入和输出嵌入共享参数，三轮 SFT 沿用预训练时的分词器。

## 训练经历

| 阶段 | 主要内容 | Epoch | 参数更新次数 | 监督目标 token |
| --- | --- | ---: | ---: | ---: |
| 预训练 | 英文文本续写 | 1 | 80,465 | 2,636,654,462 |
| SFT 1 | 短指令、角色格式 | 1 | 584 | 2,334,382 |
| SFT 2 | 日常对话、Clover 的身份 | 2 | 1,506 | 11,973,174 |
| SFT 3 | 给定文本阅读、缺失信息判断、短任务 | 1 | 574 | 447,475 |

发布的是 **SFT 3 final**，且没有在不同轮次之间重新初始化模型。最终评估记录中，第三轮阅读 Test 的 Loss / PPL 为 **0.439808 / 1.552409**；第二轮对话 Test 为 **2.152472 / 8.606110**。

数据来源和评估说明见 [训练记录](docs/TRAINING.md)。

## 快速开始

在线体验已于 2026 年 10 月 29 日关闭。想体验 Clover，请前往 **[Hugging Face 下载模型](https://huggingface.co/mioqwq/Clover-125M)**，并按照下方“本地运行”说明配置。Clover 主要使用英文数据训练，建议先用英文和它聊天。

## 本地运行

需要 **Python 3.12**。先从 [Hugging Face](https://huggingface.co/mioqwq/Clover-125M) 下载模型文件，并放到源码仓库下的 `models/Clover-125M/` 目录。

使用原生 CPU 推理时，至少需要以下文件：

- `tokenizer.json`
- `release.json`
- `gguf/Clover-125M-v0.1.0-F32.gguf`

建议在新建的 Python 虚拟环境中，从项目根目录运行：

```sh
python -m pip install -r requirements-native.txt
python -B inference/generate_gguf.py --model-dir models/Clover-125M --prompt "Hello! Who are you?"
```

想体验网页聊天，可以继续运行：

```sh
python -B -m deployment.serve_public --root models/Clover-125M --origin http://127.0.0.1:8766
```

随后打开 **http://127.0.0.1:8766/**。

网页支持流式生成、多轮对话、采样参数调节、停止生成和忙时重试。Clover 主要使用英文数据训练，因此建议先用英文与它交互。PyTorch CPU / CUDA 推理见 [推理指南](docs/INFERENCE.md)；nginx 和 systemd 的服务器配置示例在 `deployment/examples/`，部署前需要替换示例域名与路径。

## 极低配置也能运行吗？

我曾在一台 **2 vCPU、887 MiB 可见物理内存、已有约 1 GiB swap** 的无显卡服务器上运行过 Clover。使用的是**原始 FP32 GGUF**，没有为了部署而量化、剪枝或继续训练。

这个部署采用 mmap 加载、两个 CPU 线程、单个生成任务和受限上下文。某次进程状态记录的 RSS 约 **508 MiB**，历史峰值约 **585 MiB**，同时有约 **118 MiB** 被换出到 swap。

内存计算、速度观察和部署命令见 [部署记录](docs/CPU_2C1G.md)。

## 仓库内容与复现范围

| 目录 | 内容 |
| --- | --- |
| `scripts/` | 预训练、SFT、数据批次加载和会话格式化 |
| `configs/` | 模型结构与各阶段训练配置 |
| `inference/` | PyTorch / 原生 GGUF 推理 |
| `deployment/` | FP32 格式转换、服务端与部署示例 |
| `web/clover/` | 流式聊天网页 |
| `docs/` | 训练和部署文档 |
| `provenance/` | 评估结果、转换校验及数据来源说明 |

这里公开的是训练核心实现，不包含原始/处理后语料、采集清洗流程和全部私有验收材料。因此，**仅克隆源码无法原样重跑整个训练过程**；不过下载导出的权重后，可以单独使用推理和网页功能。

**Clover 仍可能答非所问、重复生成或编造信息。这是一个正在探索的小模型项目，使用时请自行核对重要输出。**

**开源许可：**Clover-125M 的原创项目代码与模型权重采用 **[Apache License 2.0](LICENSE)**，允许在遵守许可证要求的前提下使用、修改和再分发。第三方代码与数据保留各自的授权条件。详情见 [许可证说明](LICENSE_STATUS.md)、[版权声明](NOTICE) 和 [第三方声明](THIRD_PARTY_NOTICES.md)。
