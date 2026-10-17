# Clover-125M 训练记录

Clover-125M 是一个从零开始训练的英文语言模型，参数量为 **124,720,896**。训练从随机初始化的权重开始，先进行英文文本预训练，再经过三轮全参数监督微调（SFT）。v0.1.0 使用第三轮 SFT 的最终权重。

模型的在线体验地址是 [ai.mio.blue](https://ai.mio.blue/)。推理与部署方式另见 [推理指南](INFERENCE.md) 和 [低配置服务器部署](CPU_2C1G.md)。

## 模型结构与训练环境

Clover 使用 Hugging Face Transformers 提供的 Llama 风格 Decoder 架构，但没有使用 Meta 的任何预训练权重。模型包含 14 层 Transformer、768 维隐藏状态、2080 维 FFN，以及 12 个 Attention Heads 和 12 个 KV Heads。Head Dimension 为 64，因此使用标准多头注意力（MHA）。

模型使用 SiLU、RMSNorm（ε = 1e-5）和 RoPE（theta = 10000）。上下文长度为 1024 tokens，词表大小为 32,000，输入 Embedding 与输出 LM Head 共享权重。Attention 和 MLP 均未启用 bias，也没有使用 dropout。

训练设备为 **NVIDIA RTX 5070**，使用 FP32 主权重和 BF16 混合精度。记录的环境包括 PyTorch 2.14.1+cu130、Transformers 5.19.0、Tokenizers 0.23.3、Safetensors 0.8.0 和 NumPy 2.5.4。各依赖版本可在 `requirements-training.lock.txt` 中找到。

## 预训练

预训练以英文文本续写为目标，使用的语料来源包括英文 Wikipedia、Project Gutenberg、网页/C4 材料和 arXiv 文本。数据划分为训练、验证和测试部分。32K 的 ByteLevel BPE 分词器只在训练集上训练，后续三轮 SFT 均沿用同一分词器。

训练采用标准的 Next-Token Prediction，序列长度为 1024，Micro Batch 为 4，梯度累积 8 次，名义有效 Batch 为 32 条序列。文档之间使用 EOS 标记，但 Attention 不会在 EOS 处重置。

优化器使用 AdamW：β = (0.9, 0.95)、ε = 1e-8、Weight Decay = 0.1，梯度裁剪阈值为 1.0。学习率从 3e-4 逐步降至 3e-5，前 1000 次参数更新用于 Warmup。

预训练共运行 **1 个 Epoch，80,465 次参数更新**，累计训练目标为 **2,636,654,462 tokens**。完整验证集覆盖 13,337,535 个目标 token，最终 Loss 为 **3.050373**，Perplexity 为 **21.123215**。

这一阶段得到的是能够进行英文文本续写的基础模型，还没有针对聊天和指令进行专门优化。

## 三轮监督微调

SFT 继续训练模型的全部参数，不使用 LoRA 等参数高效微调方式，也不改变模型结构或分词器。每一轮从上一轮的权重继续训练，同时重新初始化优化器。三轮均使用 512 tokens 的训练序列长度、Micro Batch 4、梯度累积 8、BF16 混合精度、0.05 Warmup Ratio 和 0.01 Weight Decay。

SFT 的监督目标只包含 Assistant 的回复正文及结尾 EOS；System、User 和角色标记不计入 Loss。不同会话分别编码，Padding 位置被 Mask，不把多段会话拼接成一条训练序列。消息格式由 `scripts/sft_format.py` 实现，名称为 `english_plain_roles_v1`。

### SFT 1：短指令与基本回复

第一轮主要使用 Smol Smoltalk 的英文指令和短回答，让模型适应指令形式及对话角色。训练集包含 **18,660 条会话**，训练 1 个 Epoch，完成 **584 次更新**。参与监督的 Assistant 目标 token 为 **2,334,382**，学习率从 2e-5 下降至 2e-6。

### SFT 2：日常对话

第二轮加入 UltraChat、OpenAssistant、Everyday Conversations，以及项目编写的 Clover 身份对话。这一阶段主要用于改善一般英文对话和多轮交互。

训练集包含 **24,066 条不同会话**，运行 2 个 Epoch，累计处理 48,132 次会话样本，完成 **1,506 次更新**。Assistant 目标 token 为 **11,973,174**，学习率从 1e-5 下降至 1e-6。

### SFT 3：给定文本阅读

第三轮重点处理阅读任务：根据用户提供的内容回答问题，并学习在材料缺少答案时避免随意补充信息。

训练集共 **18,343 条会话**，其中有 11,000 条 SQuAD 可回答问题、1,999 条缺失答案问题、1,374 条项目编写的模板练习，以及 3,970 条前一轮训练数据的回放。这些模板和回放样本不属于新收集的人类对话。

本轮运行 1 个 Epoch，完成 **574 次更新**；学习率从 8e-6 降至 8e-7。参与监督的 Assistant 目标 token 总计 **447,475**。该数字不包含所有输入 token，因此不能理解为第三轮总训练 token 数。

第三轮还按任务组织更新批次，避免长篇对话回放占据过多训练权重。每个训练样本在这一轮中出现一次。

## 评估结果

v0.1.0 采用 **SFT 3 Final**。在确定最终权重后，才对固定的模型进行正式测试；测试集没有用于事后挑选 Checkpoint 或调整生成参数。

第三轮阅读测试集包含 **650 条会话、8,386 个目标 token**，Loss 为 **0.439808**，Perplexity 为 **1.552409**。第二轮对话测试集包含 **388 条会话、90,855 个目标 token**，Loss 为 **2.152472**，Perplexity 为 **8.606110**。

作为补充，第三轮训练结束时的验证集覆盖 8,055 个目标 token，Loss 为 **0.443229**，Perplexity 为 **1.557729**。预训练保留验证子集覆盖 1,048,576 个目标 token，Loss 为 **3.282923**，Perplexity 为 **26.653567**。

这些测试衡量的是各自数据集上的 token 预测表现。任务不同，Loss 和 Perplexity 不能直接横向比较；阅读测试的低 Perplexity 也不代表通用问答正确率或推理能力。Clover 仍可能重复、忽略指令或产生不准确的信息。

## 权重导出与推理一致性

最终权重保留 FP32 Safetensors 格式，同时转换为未量化的 FP32 GGUF。转换过程对 128 个张量进行逐项校验，逆布局变换后的哈希比对一致。在三个参考 CPU 输入上，原 PyTorch 实现与原生推理实现给出的最高 Logit Token 相同，最大绝对误差约 **2.12e-5**。

由于推理内核和随机数生成器不同，即使设置相同的 Seed，也不能保证两个运行后端生成完全一样的文字。

两种权重格式都位于 [Hugging Face · Clover-125M](https://huggingface.co/mioqwq/Clover-125M)。

## 源码与复现

预训练入口为 `scripts/train_lm.py`，三轮微调入口分别为 `scripts/train_sft.py`、`scripts/train_sft_round2.py` 和 `scripts/train_sft_round3.py`。相关模型配置、数据读取和消息格式化实现也包含在仓库中。

仓库不包含原始或处理后的训练数据，也不包含完整的数据收集、清洗和准备流程。部分训练入口还保留了对输入数据、上一阶段模型和相关记录的检查。因此，**仅克隆本仓库无法完整复现原训练过程**；使用新数据训练时，需要另外准备输入并适配对应的检查流程。

不加 `--start-training` 参数运行以下命令时，只会输出训练配置，不会启动实际训练：

```sh
python -B -m scripts.train_lm
python -B -m scripts.train_sft
python -B -m scripts.train_sft_round2
python -B -m scripts.train_sft_round3
```

已经训练好的模型不依赖这些训练数据，下载 Hugging Face 导出的权重后即可使用 [推理程序](INFERENCE.md)。
