# R9700 H3 加速结果

2026 年 10 月 7 日，在单张 Radeon AI PRO R9700（gfx1201）上，对 MiniMax H3 的空间卷积和窗口 Attention 加入可选 Triton 内核。同样的 1344×768、243 帧、24 FPS、8 步请求，三次预热后的完整生成中位数从 **691.0 秒降至 559.7 秒**，加速 **1.23×**，耗时减少 **19.0%**。

输出会有数值及构图差异。抽帧和音频标签检查通过本次筛查，不能据此声称逐像素等价或无损。原生路径继续作为默认；加速入口仅支持 gfx1201 推理。

后续的 [耗时分布与解码优化](r9700-bottlenecks.md) 以这里的 559.7 秒为基线，分析各阶段占比并验证视频 Linear 缓存和原生音频卷积。下表保留上轮测量记录。

## 测速结果

| 路径 | 首条完整请求 | 三次预热请求 | 预热中位数 |
| --- | ---: | --- | ---: |
| 原生卷积及 Attention，已有视频 VAE hipBLASLt 优化 | 706.7 秒；本轮复核 706.1 秒 | 691.0、693.3、690.9 秒 | 691.0 秒 |
| Triton 空间卷积及窗口 Attention，同样的视频 VAE 设置 | 579.1 秒 | 560.7、559.7、558.8 秒 | 559.7 秒 |

每次均是独立的完整请求：重新启动生成进程、编码提示词、加载模型、采样、视频和音频解码、封装 MP4。三次预热仅复用编译和磁盘缓存，不复用 conditioning 或驻留的生成引擎。表中时间排除 Docker 启动；首条请求使用独立的编译缓存目录。原生预热数据来自此前保存的三次测试，本轮使用当前源码关闭新内核跑了一次首条对照，完整时间和输入审计与历史基线一致。

采样阶段预热中位数为 **459.9 秒**，原生基线为 592.4 秒。新增收益主要发生在采样，视频和音频 VAE 沿用已有设置。没有并发 GPU 任务。

硬件和配置：单张 R9700，GPU UUID `GPU-b11d6bcf3a61a551`；Torch `2.12.0+rocm10.0.0`、HIP `7.15.26333`、AMD Triton `3.8.0`；Docker 镜像 `sha256:55bf8baa2a513b1c05bd256119fbc57a6ca64170e6cf5fe519b1cdd0c458cfd9`。原生 FP8 线性计算、head chunk 16、FF chunk 2048、projection chunk 1024、window batch 4、resident blocks 8。种子 `2026090901`，不启用二次采样，提示词使用 [`scripts/amd/prompt.txt`](../../scripts/amd/prompt.txt)。

## PR 中适用的思路

检查了用户提供的作者筛选结果，以及相关基础 PR。主要借鉴按实际 gfx 架构识别硬件、隔离 AMD 与 NVIDIA 后端，并按目标架构选择内核的思路。以下适用性判断结合了本次单卡、dense H3 工作负载；没有直接合并 FreeToken 的 MoE、通信或 GGUF 实现。

| FreeToken PR | 内容 | 本次适用性 |
| --- | --- | --- |
| [132](https://github.com/FlashML-org/FreeToken/pull/132)，已合并 | RDNA3 和 RDNA4 运行基础 | gfx 架构识别、AMD 路径隔离 |
| [133](https://github.com/FlashML-org/FreeToken/pull/133)，open | TVM FFI 索引及存储 JIT 的 HIP 移植 | 当前 H3 没有对应热点 |
| [134](https://github.com/FlashML-org/FreeToken/pull/134)，open | ROCm 上屏蔽 CUDA 专用可选后端 | 对 AMD 单独选择 Attention 内核 |
| [135](https://github.com/FlashML-org/FreeToken/pull/135)，open | 多卡通信使用 RCCL | 单卡测试不适用 |
| [136](https://github.com/FlashML-org/FreeToken/pull/136)，关闭未合并 | RDNA4 原生 GGUF 内核 | wave32 约束有参考价值；GGUF 算法不用于 H3 FP8 |
| [378](https://github.com/FlashML-org/FreeToken/pull/378)，open | CPU 和 Hybrid MoE 图重放安全 | dense GPU H3 不适用 |
| [491](https://github.com/FlashML-org/FreeToken/pull/491)，open | Hybrid MoE decode 调度重构 | dense GPU H3 不适用 |

首个真实 S2 block 的 GPU 时间约 42% 在 Attention，另有约 187 毫秒用于空间卷积。整个请求的 H2D 拷贝仅约 3.7 秒，与计算重叠，继续调整预取难以解释主要收益。

[`rocm_spatial.py`](../../freevideo_engine/rocm_spatial.py) 直接处理 H3 的帧内 BF16 5×5 depthwise 卷积，用 FP32 累加并输出连续布局，保留逐帧零填充和后续时间卷积。真实形状的局部对比约 33.0→4.2 毫秒，包含原实现的布局转换。

[`rocm_attention.py`](../../freevideo_engine/rocm_attention.py) 使用 BF16 WMMA 和 FP32 online softmax，保留所有传入的 key；窗口和 anchor 仍由原有 `WindowAttention` 构造。真实大窗口的局部对比约 43.4→33.4 毫秒。全局 Attention 保留 AOTriton。局部测试只解释热点，完整收益以表中独立请求为准。

## 质量与数值检查

空间卷积的 6 项检查覆盖帧边界、非整齐形状和真实 S1/S2 形状；Attention 的 7 项检查覆盖矩形窗口、独立 batch、非连续输入和 FP32 小形状参考。原有窗口及 anchor 布局探针的相对 RMSE 为 0.00161，低于 0.006 阈值。

玩具车、人物运动、物体运动、复杂市场场景的 S1 完整生成均通过音视频解码及数值筛查。CPU CLAP 的预期声音标签均排在前两位；S2 玩具车的轻声滚动标签排第一。CLAP 是指定标签间的相似度筛查，不是人工听感或质量评分。

抽帧对比未见明显新增画面破损，但生成内容改变。S2 加速结果从开头就是带小人物的敞篷车，基线从封闭车型在中途变成敞篷车；背景、照明和构图也不同。两者都存在偏离固定机位等原模型问题。人物开头裁切、球在前几帧缺失等问题在对应基线中也存在。抽帧检查不覆盖帧间所有细节。

曾尝试同时替换全局与窗口 Attention，数值探针通过，但 S1 音频的音乐标签排第一、滚动标签退至第七，已拒绝该方案；生产开关只提供窗口替换。CK SDPA 在当前 gfx1201 上不受支持，本次没有采用它。

四条新方案请求的原始 RGB、音频和音频解码输入完全一致。当前源码关闭新内核后的对照与历史原生对照也完全一致。新方案与原生对照之间的 RGB 相对 RMSE 为 0.504、音频为 0.551、音频解码输入为 0.305，说明生成内容并非数值等价。

## 使用加速入口

本工作区已准备好模型、固定镜像和手动 profile，可使用 [`run_rocm_fast.sh`](../../scripts/amd/run_rocm_fast.sh) 运行相同规格。以下启动参数已更新为后续解码优化方案，使用 Linear 缓存和原生音频卷积；上表仍保留上一轮测量结果：

```bash
bash scripts/amd/run_rocm_fast.sh -m freevideo_engine generate \
  --cache /data/models/prepared/edge-5e6ecc39f472f98c/cache \
  --base /data/models/vdn/h3-base --attention torch-flash \
  --width 1344 --height 768 --frames 243 --base-steps 8 \
  --no-two-pass --seed 2026090901 \
  --no-history-placement --no-tuning --resource-retries 0 \
  --profile /data/experiments/freevideo-r9700/prepared/profiles/native-decoder-fast-v1.json \
  --model-paths /data/experiments/freevideo-r9700/runtime/encoder-paths.yaml \
  --prompt-file /workspace/scripts/amd/prompt.txt \
  --out /data/experiments/freevideo-r9700/outputs/r9700-fast-video.mp4
```

脚本默认启用下列主机环境变量，允许调用者覆盖：

| `run_rocm.sh` 主机变量 | 引擎环境变量 | 加速入口值 | 原生值 |
| --- | --- | --- | --- |
| `FV_ROCM_SPATIAL_CONV` | `FREEVIDEO_ROCM_SPATIAL_CONV` | `triton` | `miopen` |
| `FV_ROCM_ATTENTION` | `FREEVIDEO_ROCM_ATTENTION` | `triton-window` | `aotriton` |
| `FV_ROCM_VIDEO_BLAS` | `FREEVIDEO_ROCM_VIDEO_BLAS` | `cublaslt` | `default` |
| `FV_ROCM_AUDIO_CONV` | `FREEVIDEO_ROCM_AUDIO_CONV` | `native` | `miopen` |

`cublaslt` 是 PyTorch 在 HIP 上选择 hipBLASLt 的枚举名，作用域只限视频 VAE。采样及音频沿用原来的 BLAS；不要用全局 BLAS 开关代替它。

批量复现用 `run_case.py`，将四个主机变量设为加速入口值，并提供同一手动 profile 及 `--no-two-pass`。每次采用新的 `--name`，以保留历史结果。`run_rocm.sh` 自身继续默认 `miopen` / `aotriton`；复现表中原生基线时，设置 `FV_ROCM_SPATIAL_CONV=miopen FV_ROCM_ATTENTION=aotriton FV_ROCM_VIDEO_BLAS=cublaslt FV_ROCM_AUDIO_CONV=miopen`，并使用原 `native-final-v2.json`。

## 实验记录

原始请求、MP4、RGB、音频和逐阶段记录保存在 `/dc1/zihaomu/free_token_mapping/experiments/freevideo-r9700`：

- 新方案：`outputs/pr-speed-window-s2-v1/run-00` 至 `run-03`；本轮原生对照：`outputs/pr-speed-control-s2-v1/run-00`。
- 历史原生预热基线：`outputs/native-p4-v4-video-hipblaslt-s2/run-01` 至 `run-03`。
- 本轮分析：`reports/pr-acceleration-20261007`，包含 `performance-summary.json`、输入审计、原始输出差异、数值探针、CLAP 结果和视觉检查范围。
- `window-source-snapshot.json` 保留整个验证期间固定的 12 个计算及运行文件 SHA256；失败的全局 Attention 方案单独封存。

收益只在上述 R9700、固定版本和 S2 配置上完成三次预热验证；本轮没有测量其他 GPU、LoRA 或二次采样的完整性能。
