# 在 2 核、1 GB 左右的服务器上运行 Clover

Clover-125M 的参数量不大，但 **FP32 权重本身就接近 477 MiB**，再加上推理过程中需要的 KV Cache、临时缓冲区和程序本身的开销，小内存服务器仍然很容易吃紧。

我试过在一台没有显卡的低配置服务器上部署 Clover，并保留原始的 124,720,896 个参数。没有量化、剪枝或蒸馏，也没有为了部署而继续训练。这里记录实际用到的环境、内存占用和运行方式。

如果只是想体验聊天，可以直接去 **[ai.mio.blue](https://ai.mio.blue/)**，不必自己部署。

## 实际跑在什么机器上？

测试机器运行 Ubuntu 24.04 x86-64，有 **2 vCPU、887 MiB 可见物理内存和约 1 GiB 已配置的 swap**，没有 GPU。使用 Python 3.12、llama-cpp-python 0.3.36、Tokenizers 0.23.3 和 NumPy 2.5.4。

最重要的一点是：**这里的“1 GB 服务器”不是“不使用 swap，只靠 1 GB 物理内存就能保证运行”。** 实际机器已经启用了 swap，而且有部分进程内存被换出。

从保存的进程快照来看：

- FP32 GGUF 文件大小为 **500,008,288 bytes**，约 **476.85 MiB**。
- 一次初始短对话后，进程 RSS 约为 **551.43 MiB**；另一时刻的 RSS 约 **508.40 MiB**。
- 记录的 RSS 历史峰值（VmHWM）约 **584.59 MiB**。
- 同期进程约有 **117.59 MiB** 的内存位于 swap。
- systemd cgroup 的 MemoryPeak 约为 **591.62 MiB**。

这些是运行记录中的观测值，并不是稳定运行所需内存的下限，也不是新做的一轮性能测试。原始数值保存在 `provenance/cpu_2c1g.json` 中。

RSS、systemd 统计和操作系统的文件缓存不是同一种指标。特别是使用 mmap 后，被访问的权重文件页面可能同时显示在进程 RSS 与文件缓存中，不能简单把所有数字加起来当作总内存。swap 能缓解内存不足，但也会让部分请求变慢。

## 为什么能在这么小的机器上运行？

小服务器不需要安装整套训练环境。我选择了原生的 **llama.cpp 推理后端**，读取未量化的 FP32 GGUF；网页服务本身不需要导入 PyTorch、Transformers，也不会加载优化器状态。

几个细节对内存占用特别重要：

**权重通过 mmap 加载。** 使用 `use_mmap=True` 和 `use_mlock=False`，尽量避免在 Python 中再保存一份完整的权重张量，也不强制锁住所有映射页。当然，实际访问到的页面还是要占内存的。

**模型只加载一份。** 服务端一次只允许一个生成任务。忙时返回 HTTP 409，而不是同时再启动几个各自加载模型的 worker。网页会隔约 3 秒尝试重试，最多 5 次。正在生成的请求如果中途断开，不会自动把整段回复重新生成。

**线程和工作区尽量收紧。** 原生推理使用 2 个 CPU 线程，将 OpenBLAS 限制为 1 个线程；预填充批次使用 `n_batch=32` 和 `n_ubatch=32`。这样能减少小机器上的并发争抢，但长输入的处理速度会受到影响。

**保留共享 embedding。** 输入输出权重共享是在训练模型时就决定的，并不是部署时才做的压缩；因此也无需额外保存约 93.75 MiB 的输出嵌入矩阵。

服务端还对请求规模和运行时间设置上限：最多约 12 个 HTTP handler、JSON 请求体不超过 64 KiB、单次生成最多 256 tokens，并设置约 90 秒的生成截止时间。取消与超时通常在推理批次边界被检查，并非硬实时抢占。

## KV Cache 到底占多少？

Clover 有 14 层、12 个 KV Heads、每个 Head 64 维。K 和 V 都采用 FP32，主要 KV 数据量可以这样算：

```text
2 × 14 层 × 上下文 token 数 × 12 heads × 64 维 × 4 bytes
```

启动时如果设置 1,024 tokens，上述 KV 数组约占 **84 MiB**；512 tokens 约 **42 MiB**；256 tokens 则约 **21 MiB**。这些数字不包括图计算工作区、元数据、对齐和其他运行时开销。

需要区分两种“上下文长度”：网页里的 context 滑条只是限制本次请求使用多少历史，不会重新分配已经创建的 KV Cache。**想真正降低启动时的 KV 内存预算，必须在启动服务时改变 `--context-tokens`。**

例如，将启动上下文限制到 512 tokens：

```sh
python -B -m deployment.serve_public --root models/Clover-125M --origin http://127.0.0.1:8766 --context-tokens 512 --threads 2
```

上下文预算包含输入、角色标记、历史对话以及预留输出。设置得越小，能保留的对话也越少。运行时仍然使用 FP32 KV，并没有启用低精度 KV Cache。

## 从新环境开始部署

先从 [Hugging Face · Clover-125M](https://huggingface.co/mioqwq/Clover-125M) 下载所需模型文件，放到源码仓库内的 `models/Clover-125M/`。原生服务需要 `tokenizer.json`、`release.json` 和 `gguf/Clover-125M-v0.1.0-F32.gguf`。

最好在自己的电脑上先准备和检查这些文件，再传到服务器。小服务器只需要 GGUF、分词器及相关元数据，不需要 safetensors 或 PyTorch；也不建议在仅有 1 GB 左右内存的机器上临时做模型格式转换或编译大型推理依赖。

在 Linux 上可以这样启动：

```sh
python3 -m venv .venv-native
. .venv-native/bin/activate
python -m pip install -r requirements-native.txt
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
python -B -m deployment.serve_public --root models/Clover-125M --origin http://127.0.0.1:8766
```

`requirements-native.txt` 指向 llama-cpp-python 维护者提供的 CPU wheel 来源，并要求安装二进制 wheel。如果平台没有匹配的构建，安装会失败，不会自动在小服务器上编译。也可以参考 [llama-cpp-python 的安装说明](https://github.com/abetlen/llama-cpp-python#installation)。

## 用 systemd 和 nginx 对外提供服务

仓库的 `deployment/examples/clover.service` 是 systemd 的参考配置，其中 MemoryHigh 为 700M、MemoryMax 为 768M、MemorySwapMax 为 256M、TasksMax 为 48。这些是**资源限制**，不是把模型自动压缩到这么大；如果运行时超过硬限制，进程可能被终止。`MemorySwapMax` 也不会替服务器创建 swap，需要提前在系统中配置。

`deployment/examples/nginx.conf` 提供了反向代理和 HTTPS 的示例。模型进程监听本机 loopback，由 nginx 处理对外域名、TLS、请求限制和 SSE 流式转发，并关闭流响应的代理缓冲。部署时请替换示例域名、用户、工作目录与证书路径。

如果使用 Cloudflare 代理，建议使用 **Full (strict)**，确保源站 443 端口和证书配置正确，避免 Flexible 模式与源站 HTTPS 重定向造成循环。

## 实际速度怎么样？

这台机器的既有记录中，首次身份问答生成 11 个 tokens，总共约 **9.45 秒**，约 **3.36 秒** 后出现首段输出。之后一次短多轮回答生成 7 个 tokens，用时约 **1.00 秒**。一次接近 1,008 个 prompt tokens 的取消测试，首段输出大约等待了 **14.69 秒**。

这些只是不同请求的墙钟时间，受输入长度、预填充以及内存页面是否已加载等因素影响，不能简单当作稳定的 tokens/s 基准。开发电脑上的推理速度也不应直接套到这台小服务器上。

总之，这套配置比较适合**低并发的英文聊天体验**：一次处理一个生成任务，忙时让其他请求重试。它并不适合需要稳定低延迟或多人同时生成的服务，也不是所谓的“200 MB / 400 MB 就能装下全部运行时”的方案。

模型转换时的张量和 logits 比对结果见训练记录及 `provenance/` 下的文件。原生推理继续使用 Clover 自己的分词器和消息格式；不同后端即使采用相同 seed，也不一定给出逐字一致的采样结果。
