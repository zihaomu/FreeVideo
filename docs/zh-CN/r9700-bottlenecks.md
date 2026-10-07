# R9700 耗时分布与分阶段优化

同样的单张 R9700、1344×768、243 帧、24 FPS、8 步完整请求，本轮优化前的预热中位数为 559.7 秒。主要瓶颈在 GPU 采样，占约 82.2%；视频 VAE 解码占约 10.5%。提示词编码、模型加载和 MP4 封装的占比较小。

## 9 分钟基线的耗时分布

以下拆分来自上轮加速方案三次预热请求中的中位数请求。各项互不重叠，合计 559.6503 秒。

| 阶段 | 秒 | 占比 |
| --- | ---: | ---: |
| GPU 采样 | 459.80 | 82.16% |
| 视频 VAE 解码 | 58.91 | 10.53% |
| 音频 VAE 加载和解码 | 14.75 | 2.64% |
| 文本编码及编码模型加载 | 8.56 | 1.53% |
| 主模型加载 | 2.88 | 0.51% |
| 视频 VAE 加载 | 3.36 | 0.60% |
| MP4 封装 | 1.32 | 0.24% |
| 原始数据及 latent 保存 | 0.13 | 0.02% |
| 后处理及进程控制等剩余时间 | 9.94 | 1.78% |

剩余 9.94 秒按嵌套计时器差额拆为：解码器内后处理和控制 2.10 秒、引擎内其他工作 1.34 秒、请求级进程及控制 6.51 秒。它们不是独立采集的算子计时。`decode_save_seconds` 和 `work_seconds` 包含多个表内阶段，不能再次累加。

## 采样中的算子热点

使用当前空间卷积和窗口 Attention 内核，对首个真实 S2 block 的第一步输入做预热后分析。该 block 完整时间约 1.105 秒，GPU kernel 时间合计约 1.096 秒。

| GPU 运算 | 毫秒 | kernel 时间占比 |
| --- | ---: | ---: |
| 窗口 Attention | 373.15 | 34.03% |
| FP8 矩阵乘 | 263.27 | 24.01% |
| 其他矩阵乘 | 122.35 | 11.16% |
| 全局 Attention | 95.33 | 8.69% |
| 时间卷积 | 30.65 | 2.80% |
| Gather | 30.30 | 2.76% |
| 空间卷积 | 28.70 | 2.62% |
| 其他 kernel | 152.70 | 13.93% |

这是一个真实 block 的分析，不是整段 459.8 秒的逐算子测量。按占比粗略外推，窗口 Attention 对应约 156 秒，FP8 矩阵乘约 110 秒；这些数字只用于排序优化优先级。

窗口 Attention 的启动参数和 Q/K/V 布局测试均没有获得稳定收益。布局转换保留输出，但计入复制后所有候选更慢。FFN 的 8192 行切块把局部约 175–178 毫秒降到约 172 毫秒，整段收益预计只有约一两秒，暂未采用。

BF16 门控投影切换 hipBLASLt 在 S2 获得约 24.1 秒完整收益：三条预热请求 535.96、535.51、535.57 秒，中位数 535.57 秒，且四次重复原始输出完全一致。然而 S1 玩具车音频筛查中，音乐标签排第一，预期滚动声音退至第七；原方案滚动声音排第二。此方案已拒绝并从生产代码撤回。S2 单个场景通过不能代替多场景质量检查。

## 视频和音频解码优化

VAE Linear 权重缓存保留原 autocast 的 FP16 计算值，Norm、残差 scale、register token 和 post quantization 参数继续 FP32。同一真实 S2 片段的预热中位数从 3.77 秒降到 3.07 秒，浮点输出和 RGB 完全一致。14 个片段的局部外推收益约 9.8 秒，完整请求以最终重复测试为准。

空间 tile 合批继续使用原 tile、重叠边界和独立 Attention。2、4、7 个 tile 合批没有更快，14 个 tile 只有约 2%–3% 的局部收益，并使约 0.33% 的 RGB 通道值变化 1 个灰度级，首次新形状调用耗时约 19.7 秒，预热后约 3.0 秒。因此保留逐 tile 解码。

音频保留 GPU FP32 参数和计算，原生 PyTorch 卷积路径只在音频 VAE forward 期间关闭 MIOpen。单进程 MIOpen 首次解码约 14.57 秒，随后约 2.04 秒；独立新进程的原生卷积首次约 1.09 秒，三次预热中位数 0.917 秒。相对于保存的原始音频，相对 RMSE 为 1.22e-6，最大绝对差异约 1.12e-6。音频输入与采样保持原路径，正常和异常退出都会恢复后端设置。

缓存候选的 VAE trace 中，矩阵乘约 30.10 秒，Attention 约 6.85 秒。它使用 FP16 Linear 存储，不是原 FP32 存储路径的逐算子分析。原 59 秒视频解码阶段还包含首调用、CPU 拷贝及清理；不能用此 trace 的约 43.5 秒直接替代完整阶段计时。

## 完整请求验证

最终候选恢复原采样路径，只启用视频 Linear 计算缓存及原生 GPU FP32 音频卷积。四条独立完整请求均通过完整音视频解码检查：

| 请求 | 完整耗时 |
| --- | ---: |
| 首条编译缓存冷请求 | 554.94 秒 |
| 预热请求 1 | 535.67 秒 |
| 预热请求 2 | 535.37 秒 |
| 预热请求 3 | 535.83 秒 |
| 三条预热中位数 | **535.67 秒** |

与原 559.65 秒预热中位数相比，节省 **23.98 秒**，耗时减少 **4.28%**，加速 **1.045×**。这是一轮解码收益，采样仍是最大瓶颈。

以下对照分别使用新旧方案的完整请求中位数对应记录（新方案 run-01，旧方案 run-02），阶段不重叠；表内未列出的小项包含在完整时间中。

| 阶段 | 原中位数请求 | 新中位数请求 |
| --- | ---: | ---: |
| GPU 采样 | 459.80 秒 | 459.83 秒 |
| 视频 VAE 解码 | 58.91 秒 | 49.34 秒 |
| 音频 VAE 加载和解码 | 14.75 秒 | 1.60 秒 |
| 视频 VAE 加载 | 3.36 秒 | 2.32 秒 |
| 文本编码及模型加载 | 8.56 秒 | 8.28 秒 |
| 完整请求 | 559.65 秒 | **535.67 秒** |

四次请求的所有 243 帧原始 RGB 都与本轮 9 分钟基线逐像素完全一致，音频解码输入也完全一致。音频相对 RMSE 最大为 1.22e-06，最大绝对误差约 1.12e-6；原生 FP32 卷积的舍入差异没有改变本次 CLAP 标签排序，预期滚动声仍排第一。四次新方案的 RGB、音频和音频输入彼此完全一致。该 RGB 等价结论针对上轮 Triton 窗口方案，不扩大为与更早原生 Attention 方案等价。

四次都重新编码提示词，conditioning SHA256 与基线相同；实际 8 步、1600 次全局原后端调用、11200 次窗口 Triton 调用、4800 次包装器记录的 FP8 头投影调用及原 head/FF 切块均匹配，未减少帧数、步数或替换采样精度。4800 不是全部 FP8 GEMM 数量：首个 block 的 trace 实际包含 86 次，其中包装器覆盖 12 次，其余是 FFN 和 Attention 输出投影。15 个计算及运行文件在四次验证期间固定。原生音频探针也验证了正常和异常退出恢复后端设置。计时排除 Docker 启动，未并发其他 GPU 测试。


## 复现和回退

本轮沿用 [上轮测试](r9700-acceleration.md) 的模型、提示词、种子及硬件。单张 Radeon AI PRO R9700，`gfx1201`，GPU UUID `GPU-b11d6bcf3a61a551`；Torch `2.12.0+rocm10.0.0`、HIP `7.15.26333`、AMD Triton `3.8.0`；Docker 镜像 `sha256:55bf8baa2a513b1c05bd256119fbc57a6ca64170e6cf5fe519b1cdd0c458cfd9`。主模型继续 native FP8、head chunk 16、FF chunk 2048、projection chunk 1024、window batch 4、resident blocks 8；全局 Attention 继续使用原后端。提示词编码、采样及音视频模型推理均在 GPU，CLAP 筛查和 FFmpeg 封装使用 CPU。

当前工作区的新 profile 位于 `/dc1/zihaomu/free_token_mapping/experiments/freevideo-r9700/prepared/profiles/native-decoder-fast-v1.json`。从已有手动 profile 复现修改：

```python
import json
from pathlib import Path
profiles = Path('/dc1/zihaomu/free_token_mapping/experiments/freevideo-r9700/prepared/profiles')
profile = json.loads((profiles / 'native-final-v2.json').read_text())
profile['decoder']['linear_compute_cache'] = True
(profiles / 'native-decoder-fast-v1.json').write_text(json.dumps(profile, indent=2) + '\n')
```

在本工作区生成同样的 S2 视频：

`run_rocm_fast.sh` 已默认选择 `native` 音频卷积，可由 `FV_ROCM_AUDIO_CONV` 覆盖。视频 Linear 缓存通过下面的新 profile 开启；仅使用旧 profile 不会获得全部解码收益。

```bash
FV_ROCM_AUDIO_CONV=native bash scripts/amd/run_rocm_fast.sh -m freevideo_engine generate \
  --cache /data/models/prepared/edge-5e6ecc39f472f98c/cache \
  --base /data/models/vdn/h3-base --attention torch-flash \
  --width 1344 --height 768 --frames 243 --base-steps 8 \
  --no-two-pass --seed 2026090901 \
  --no-history-placement --no-tuning --resource-retries 0 \
  --profile /data/experiments/freevideo-r9700/prepared/profiles/native-decoder-fast-v1.json \
  --model-paths /data/experiments/freevideo-r9700/runtime/encoder-paths.yaml \
  --prompt-file /workspace/scripts/amd/prompt.txt \
  --out /data/experiments/freevideo-r9700/outputs/r9700-decoder-fast-video.mp4
```

完整重复测试使用 `scripts/amd/run_case.py`，设置 `FV_ROCM_SPATIAL_CONV=triton FV_ROCM_ATTENTION=triton-window FV_ROCM_VIDEO_BLAS=cublaslt FV_ROCM_AUDIO_CONV=native`，指定同一 profile、`--width 1344 --height 768 --frames 243 --repeats 4`，并使用新的 `--name` 保留旧结果。每次重新启动进程、编码提示词、加载和采样，首条使用独立编译缓存，其余仅复用磁盘和编译缓存；OS 文件缓存未清空。

回退本轮解码优化：设置 `FV_ROCM_AUDIO_CONV=miopen`，并使用原 `native-final-v2.json`（`decoder.linear_compute_cache=false`）。`run_rocm.sh` 的音频默认仍为 `miopen`；全局 `TORCH_BLAS_PREFER_HIPBLASLT` 和 `ROCBLAS_USE_HIPBLASLT` 继续关闭，视频 VAE 的 hipBLASLt 选择只在视频解码期间生效。原生音频路径仅验证了 gfx1201 和 FP32 输入，不扩大到其他架构。

## 实验记录与后续优先级

实验根目录为 `/dc1/zihaomu/free_token_mapping/experiments/freevideo-r9700`：

- 9 分钟基线：`outputs/pr-speed-window-s2-v1/run-00` 至 `run-03`。
- 最终解码候选：`outputs/bottleneck-decoders-s2-v1/run-00` 至 `run-03`。
- 本轮报告：`reports/bottleneck-20261007`，保留阶段分布、真实 block trace、参数及布局实验、VAE 缓存对照、音频首次调用探针、完整请求审计及原始输出差异。
- 拒绝的门控候选：`outputs/bottleneck-gates-s2-v1`、`outputs/bottleneck-gates-s1-v1`；拒绝原因、源码归档及补丁分别为 `gate-rejection.json`、`gate-rejected-source.zip`、`gate-rejected.patch`。没有把该候选的 535.57 秒作为最终方案收益。
- `decoder-source-snapshot.json` 固定四次最终验证期间的 15 个计算和运行文件 SHA256。`decoder-performance-summary.json` 与 `decoder-stage-breakdown.json` 保存最终中位数、阶段对照及四条原始输出数值结果。视频 VAE 诊断中的初始设备错误及缓存标记纠正记录在 `vae-profile-correction.json`，不用于最终收益计算。

下一轮应继续围绕窗口 Attention 和 FP8 矩阵乘优化采样。当前窗口启动参数、布局和 FFN 切块候选均没有足够稳定的完整收益，BF16 门控 BLAS 候选有质量回退，均未进入最终方案。视频解码矩阵乘仍值得分析，但最终优先级以完整采样的约 460 秒为主。上述收益仅覆盖固定版本、单张 R9700 和该 S2 请求，没有测量 LoRA 或二次采样。

后续 [采样优化候选评估](r9700-sampler-opportunities.md) 拆细 FP8 时间，并记录 Gather 融合、门控合并、K 切块和量化小算子的局部对照结果。
