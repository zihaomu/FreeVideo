# R9700 BF16 算子替换与端到端复测

2026 年 10 月 7 日，将上轮确认低效的七类 BF16 操作接入局部 hipBLASLt 路径，完成四次独立完整生成。**预热请求中位数从 535.67 秒降至 508.86 秒，减少 26.81 秒，耗时降低 5.00%，加速 1.053×。** 较短视频的音频语义筛查出现回退，因此替换作为实验入口提供，默认启动路径继续使用已验证方案。

## 替换的算子

[`rocm_bf16.py`](../../freevideo_engine/rocm_bf16.py) 为门控和状态操作分别提供局部后端选择。下面的局部耗时来自[上轮同卡算子对照](r9700-cuda-parity.md)，不是把局部倍数直接套到整段视频。

| BF16 操作 | 原路径 | hipBLASLt | 本轮接入位置 |
| --- | ---: | ---: | --- |
| Softmax gate | 5.15 ms | 1.61 ms | 原来的逐头组投影 |
| 视频 Beta gate | 5.16 ms | 1.56 ms | 原来的逐头组投影 |
| 文本 Beta gate | 0.417 ms | 0.0146 ms | 49 行文本投影也切换 |
| 输出门控 down | 10.06 ms | 1.50 ms | 每个 block 的低秩投影 |
| 输出门控 up | 4.51 ms | 2.13 ms | 原来的逐头组投影 |
| 线性 Attention 状态统计 B | 3.98 ms | 1.20 ms | BF16 状态矩阵乘 |
| 线性 Attention 状态读出 | 3.99 ms | 1.55 ms | 查询与状态矩阵乘 |

每次完整请求记录到 **6800 次门控投影、3200 次状态矩阵乘**走新路径。FP8 继续使用原生路径；FP32 状态统计 A、Cholesky、三角求解和前后向扫描继续调用原来的实现。head chunk 16、尾组 8、FF chunk 2048、projection chunk 1024 和窗口批量 4 均匹配基线。窗口 Attention 使用此前已验证的 Triton 路径，全局 Attention 使用原后端。

后端选择只包围具体的 BF16 调用，提交后立即恢复原设置，异常也会恢复，不在每个算子前后增加 GPU 同步。状态读出绑定到当前模型实例，未替换全局 `torch.matmul` 或修改 vendor 源码。环境选择、源文件哈希及调用计数写入运行记录。GPU 探针检查正常恢复、异常恢复和 CPU 回退；真实首个 block 的返回对照输出完全一致。

## 完整请求耗时

同一张 Radeon AI PRO R9700，GPU UUID `GPU-b11d6bcf3a61a551`，`gfx1201`；Torch 2.12.0+rocm10.0.0、HIP 7.15.26333、Triton 3.8.0。沿用 `native-decoder-fast-v1.json`，1344×768、243 帧、24 fps、全分辨率单次 8 步，提示词和种子 2026090901 相同。

每次启动新 worker 并重新编码提示词。请求时间包括编码、模型加载、采样、音视频解码和封装，排除 Docker 启动和事后的媒体校验。使用新的 JIT 缓存组，首轮冷启动，随后三轮缓存预热的独立请求；未清除 OS 文件缓存。基线是上一轮同设备同工作量的三次预热请求。

| 请求 | 当前基线 | 全部 BF16 替换 |
| --- | ---: | ---: |
| 冷启动 | 554.94 秒 | 526.41 秒 |
| 预热 1 | 535.67 秒 | 508.86 秒 |
| 预热 2 | 535.37 秒 | 513.11 秒 |
| 预热 3 | 535.83 秒 | 507.51 秒 |
| 预热中位数 | **535.67 秒** | **508.86 秒** |

新方案预热范围为 507.51–513.11 秒；第二轮后段耗时有所上升，同时记录到 GPU 频率下降。传感器均值按 worker 时间分段，不能据此认定单一原因。报告保留全部结果，不只采用最快的一轮。

下面分别取两组中位数对应的完整请求；其余时间为请求时间减去列出的非重叠阶段。

| 阶段 | 当前基线 | 全部 BF16 替换 |
| --- | ---: | ---: |
| 采样 | 459.83 秒 | 432.57 秒 |
| 视频 VAE 解码 | 49.34 秒 | 49.33 秒 |
| 音频 VAE 加载及解码 | 1.60 秒 | 1.60 秒 |
| 文本编码及加载 | 8.28 秒 | 8.31 秒 |
| 视频模型加载 | 2.89 秒 | 2.99 秒 |
| 视频 VAE 加载 | 2.32 秒 | 2.21 秒 |
| 封装 | 1.28 秒 | 1.37 秒 |
| 其余请求阶段 | 10.12 秒 | 10.50 秒 |
| 完整请求 | **535.67 秒** | **508.86 秒** |

采样减少 27.26 秒，解码时间基本一致。原首个 block 中这七类 BF16 操作占 GPU kernel 总时间约 8.98%，因此局部 2–7 倍的改善最终对应约 5% 的完整请求收益。此前的解码优化已包含在 535.67 秒基线内。

四轮都匹配模型 manifest、提示词、种子、实际几何、8 步和 GPU；重新编码的 conditioning SHA256 也匹配基线。全局原后端 1600 次、窗口 Triton 11200 次、包装器计数的 FP8 头投影 4800 次均匹配。4800 不是全部 FP8 GEMM 数量。20 个计算与运行文件在验证期间固定，所有新方案请求的原始 RGB、音频及音频解码输入彼此完全一致。

## 数值差异与质量回退

真实首个 S2 block 的原路径为 1.110–1.124 秒，全部替换为 1.053 秒。候选 block 输出相对 RMSE 为 0.001542，约 29.12% 元素改变；关闭替换的前后对照均完全一致。相同 BF16 输入输出类型不代表不同库路径会得到相同的舍入结果。

误差经过整段采样后明显放大。与基线比较，S2 原始 RGB 的 RMSE 为 56.71（像素范围 0–255）、PSNR 13.06 dB，约 98.81% 元素改变；音频解码输入相对 RMSE 为 0.3065。五个时间点的抽检仍是红色小车在木桌上运动，但车辆外观、背景和轨迹发生变化。该候选没有逐像素等价结论。S1 与上轮窗口方案比较时，音频解码输入相对 RMSE 已达 0.5172，差异在采样输出中就已存在。

| 音频筛查 | 基线预期声音最佳排名 | 替换后排名 | 替换后首位标签 |
| --- | ---: | ---: | --- |
| S2，243 帧，四轮 | 1 | 四轮均为 1 | 小车在木桌上安静滚动 |
| S1，124 帧 | 2 | **4** | **空调嗡声** |

所有媒体有效性、黑帧、重复帧、有限值和饱和检查通过；S1 的预期声音未达到此前使用的前三名筛查条件，因此**没有将该候选启用为默认加速路径**。CLAP 是固定模型、固定标签集的 CPU 辅助筛查，不等同于人工听感评价。S1 完整请求耗时 108.48 秒，作为质量检查记录，不纳入 S2 的速度中位数。

## 运行与回退

实验入口 [`run_rocm_operators.sh`](../../scripts/amd/run_rocm_operators.sh) 开启两组 BF16 替换。当前只支持 gfx1201、inference kernels、head chunk 16 和串行头组。

```bash
bash scripts/amd/run_rocm_operators.sh -m freevideo_engine generate \
  --cache /data/models/prepared/edge-5e6ecc39f472f98c/cache \
  --base /data/models/vdn/h3-base --attention torch-flash \
  --width 1344 --height 768 --frames 243 --base-steps 8 \
  --no-two-pass --seed 2026090901 \
  --no-history-placement --no-tuning --resource-retries 0 \
  --profile /data/experiments/freevideo-r9700/prepared/profiles/native-decoder-fast-v1.json \
  --model-paths /data/experiments/freevideo-r9700/runtime/encoder-paths.yaml \
  --prompt-file /workspace/scripts/amd/prompt.txt \
  --out /data/experiments/freevideo-r9700/outputs/operator-repro/video.mp4
```

`FV_ROCM_GATE_BLAS=default` 可关闭门控替换，`FV_ROCM_STATE_BLAS=default` 可关闭状态替换。两者关闭即返回已验证路径；原来的 `run_rocm_fast.sh` 默认也关闭两者。全局 `TORCH_BLAS_PREFER_HIPBLASLT` 与 `ROCBLAS_USE_HIPBLASLT` 继续为 0，新后端选择只在具体 BF16 操作中生效。

复测四次完整请求使用 [`run_case.py`](../../scripts/amd/run_case.py)，需要显式设置两个新开关和此前四个快路径开关，并使用新实验名：

```bash
FV_ROCM_GATE_BLAS=cublaslt FV_ROCM_STATE_BLAS=cublaslt \
FV_ROCM_SPATIAL_CONV=triton FV_ROCM_ATTENTION=triton-window \
FV_ROCM_VIDEO_BLAS=cublaslt FV_ROCM_AUDIO_CONV=native \
python3 scripts/amd/run_case.py --name operator-repro-s2 \
  --width 1344 --height 768 --frames 243 --repeats 4 \
  --profile /dc1/zihaomu/free_token_mapping/experiments/freevideo-r9700/prepared/profiles/native-decoder-fast-v1.json
```

## 验证记录

数据根目录为 `/dc1/zihaomu/free_token_mapping/experiments/freevideo-r9700`。完整输出在 `outputs/operator-fast-all-s2-v1` 和 `outputs/operator-fast-all-s1-v1`；报告在 `reports/operator-replacement-20261007`。

`performance-summary.json` 记录四轮完整耗时、调用计数和源文件固定性；`s2-run-*-raw.json`、`s2-run-*-repeat.json`、`s1-raw.json` 记录原始输出比较；`real-block-v2.json`、`backend-probe.json` 记录真实 block 和后端恢复检查；`audio-semantics-v2.json` 与 `final-validation.json` 记录质量回退。初始音频审计引用了不存在的历史 S1 `run-01`，失败日志保留在 `audio-semantics.log`，修正为实际 `run-00` 后完整重跑，未覆盖失败记录。

源码与 profile 冻结归档为 `reports/operator-replacement-20261007-frozen-worktree-files.zip` 及对应 code receipt、patch；20 个运行文件另有 `source-snapshot.json`。诊断、完整 GPU 请求和较短视频 GPU 请求串行执行，使用设备租约。音频筛查使用 CPU。
