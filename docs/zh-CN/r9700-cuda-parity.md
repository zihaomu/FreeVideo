# R9700 与 CUDA 的性能对齐及低效算子定位

2026 年 10 月 7 日，沿用已验证的 R9700 加速方案，继续定位为何采样仍需约 460 秒。**已实测确认 BF16 门控、输出门控和线性 Attention 的 BF16 状态矩阵乘存在明显的后端选择差距：同一张卡、相同形状，另一条库路径快约 2–7 倍。** FP8 大矩阵已使用原生矩阵内核，不是 CPU 或 BF16 回退。

目前没有可访问的 NVIDIA GPU 或同条件 CUDA trace。已检查本机设备及现有 `nvidia-4090` SSH 别名，后者无法解析。因此这里不提供虚构的“CUDA 某算子快几倍”，而是给出 AMD 实测、源码路径差异和可直接在 CUDA 上执行的对齐工具。生产采样代码未修改，完整请求的已验证预热中位数仍为 **535.67 秒**。

## 先对齐比较对象

[项目公开结果](https://github.com/FlashML-org/FreeVideo/blob/main/docs/zh-CN/execution-planning.md)中的 RTX 5090 为 Windows、1344×768、10 秒、**二次采样，122 秒**。当前 AMD 为 Linux、1344×768、243 帧、**全分辨率单次 8 步**。GPU 型号、采样流程、Attention 后端和软件版本尚未逐项对齐，535.67/122 不能作为算子速度比。

当前源码的默认二次采样是 672×384 的 8 步，加上 1344×768 的 3 步。按当前几何规划：

| 视频工作量代理 | 单次全分辨率 | 当前默认二次采样 | 二次 / 单次 |
| --- | ---: | ---: | ---: |
| 视频 token × 步数 | 580608 | 362880 | 62.5% |
| 视频 token² × 步数 | 42138206208 | 18435465216 | 43.75% |

第二行只是视频 Attention 的二次复杂度近似。窗口边界、全局音频/文本 token、上采样、不同精化时间表及解码成本都没有包含，不能据此预测整段时间；公开的 122 秒也没有提供足够日志来确认其精化步数。

硬件也有差异。若比较对象是 RTX 5090，其标称显存带宽为 1792 GB/s，R9700 为 640 GB/s，相差 2.8 倍。这是硬件规格，不能直接套成每个算子的倍数。[NVIDIA 规格说明](https://www.nvidia.com/en-us/geforce/news/rtx-50-series-graphics-cards-gpu-laptop-announcements/)、[AMD 规格](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html)。不能混用稀疏 INT4/FP4 TOPS 和稠密 BF16/FP8 TFLOPS。

## 哪些算子已确认没有达到合理速度

下面是 11 次预热计时的中位数。使用真实 H3 的形状和 stride、确定性合成输入，在同一张 gfx1201 R9700 上对照原 rocBLAS 路径和局部 hipBLASLt 路径。FP8 不切换全局 BLAS。这里的预期来自同卡可达到的库路径，不要求每个算子达到标称峰值。

| 算子 | 真实形状摘要 | 当前路径 | hipBLASLt 对照 | 对照加速 |
| --- | --- | ---: | ---: | ---: |
| Softmax gate | M73435、K5376、N16，BF16，有 bias | 5.15 ms | 1.61 ms | 3.20× |
| Beta gate | M72576、K5376、N16，BF16 | 5.16 ms | 1.56 ms | 3.30× |
| 输出门控 down | M72576、K5376、N128，BF16 | 10.06 ms | 1.50 ms | 6.73× |
| 输出门控 up | M72576、K128、N2048，BF16，有 bias | 4.51 ms | 2.13 ms | 2.12× |
| 线性 Attention 状态统计 B | B1120、M128、K1008、N128，BF16 | 3.98 ms | 1.20 ms | 3.30× |
| 线性 Attention 状态读出 | B1120、M1008、K128、N128，BF16 | 3.99 ms | 1.55 ms | 2.58× |
| 文本 Beta | M49、K5376、N16，BF16 | 0.417 ms | 0.0146 ms | 28.6× |

默认 BF16 内核的名称包含 `MT32x128x16` / `MT64x32x8`，对照的不同内核带有 `MI16x16x1`。这与矩阵内核选择不理想的判断一致；本轮没有硬件计数器或 ISA 占用率报告。不能将“GPU 忙”解释成矩阵核被充分利用。

例如窄门控默认路径的最小输入/输出流量除以耗时约为 152–154 GB/s，对照路径约为 492–501 GB/s；大 BF16 拷贝探针约为 480 GB/s。这是最小流量估计，不是硬件测得的 DRAM 事务量，但结合相同设备、相同输入的后端对照，足以确认存在软件路径差距。

文本 Beta 的局部倍数很大，但实际首个 block 的四次文本 Beta 合计仅约 1.82 ms。应先处理大矩阵及重复扫描大输入的门控，不按倍数单独排序。

## 真实输入验证：速度成立，数值还不能直接采用

合成输入取 CPU 固定种子的 int8 值，再转换并除以 16，便于跨设备核对 SHA256。这些值上的后端输出一致不能代替真实模型验证。

对首个真实 S2 block 的实际 BF16 状态操作补测：

| 算子 | 原路径 | 对照路径 | 真实输入相对 RMSE |
| --- | ---: | ---: | ---: |
| 状态统计 B | 3.955 ms | 1.189 ms | 6.87e-5 |
| 状态读出 | 3.992 ms | 1.550 ms | 4.05e-5 |

只路由这两种 BF16 状态操作，不改变门控、FP8、FP32 状态统计、求解及扫描，完成三轮原路径→候选→候选→原路径交替测试。首个 block 分别节省 15.78、16.49、16.96 ms，耗时降低约 1.41%–1.50%。但是 block 输出的相对 RMSE 为 **1.03e-3**，约 **17.94%** 元素改变；原路径返回对照均完全一致。因此只保留诊断原型，没有启用默认路径，也没有宣称完整视频质量通过。

此前全体 BF16 门控切换 hipBLASLt 的完整质量筛查已有回退，见[耗时与解码报告](r9700-bottlenecks.md)。不能为了速度把旧方案重新打开，也不能用全局 BLAS 开关代替局部路由：当前环境的全局 hipBLASLt FP8 路径有已记录的失败。

这些已确认低效的 BF16 操作在真实首个 block 中合计约 **98.41 ms，占 GPU kernel 时间 8.98%**。它们值得修复，但不能解释全部数倍端到端差距。

## 两个大热点与 CUDA 路径的差异

**窗口 Attention：34.03%。** 当前 AMD 使用 BF16 Q/K/V、FP32 online softmax 和 RDNA4 WMMA，主形状为 B4、Q5040、K17995、H16、D128。同卡合成输入探针：AOTriton 40.99 ms，可选窗口内核 31.04 ms；当前生产已采用窗口内核。CUDA 可选 SageAttention 2，其 QK/PV 使用量化运算，不等同于当前 BF16 计算。[SageAttention 官方实现](https://github.com/thu-ml/SageAttention)列出了量化方式。尚不能确认公开 CUDA 请求具体使用了哪个后端，也不能把项目宣传的其他模型收益直接套到 H3。后续 CUDA 应同时测 BF16 同精度对照和实际所选后端，再分别报告差距。

**FP8 GEMM：24.01%。** 当前 E4M3FN、per-tensor FP32 scale、BF16 输出、原生 GPU 矩阵内核。真实 block 的有效吞吐为 FFN up 234.72、FFN down 214.38、QKV 242.70、Attention 输出 255.39 TFLOPS；隔离算子探针约为 231–261 TFLOPS。相对 R9700 标称稠密 FP8 峰值 383 TFLOPS，这些是吞吐/峰值比例，不是硬件占用率或可直接获得的收益。它们仍有调优空间，但目前没有 BF16 小矩阵那种已验证的数倍同卡差距。FP8 时间中 56.7% 在 FFN，应优先对齐 FFN 两种形状的 CUDA GEMM。不要把包装器的 4800 次头投影计数当成全部 FP8 调用数。

**FP32 状态统计与 TF32。** 上游 `linear_attention/scan.py` 在状态统计 A 的矩阵乘期间请求 TF32，然后恢复设置。CUDA 可利用支持 TF32 的硬件；本机 `torch.cuda.is_tf32_supported()` 明确返回 false。相同 B1120、M128、K1008、N128 探针打开 TF32 标志后仍约 2.33 ms、15.9 TFLOPS，未测出加速。当前 [PyTorch HIP 文档](https://docs.pytorch.org/docs/main/notes/hip.html#tensorfloat-32-tf32-on-rocm)描述的是 MI300 的 TF32 路径，不能套到 R9700。也不能直接将 A 改成 BF16：源码保留 FP32 是为保持后续矩阵的正定性和求解稳定性。

现有首个 block trace 的 GPU kernel 活跃区间为 1096.45 ms，首尾跨度为 1109.29 ms，间隙约 12.83 ms。这里是内核时间线的覆盖率，不是算力利用率。`hipStreamSynchronize` 的 CPU 等待与 GPU 运算重叠，不能把约 825 ms 等待再加成一个 CPU 瓶颈。当前证据仍指向 GPU 运算路径；完整采样的其他 block 和 H2D 重叠须另行覆盖。

## 优先处理顺序

1. 修复已确认低效的 BF16 门控、输出门控、状态 BMM 路径，独立控制作用域，并检查真实输入的舍入及多场景质量。优先解决当前路径与更快矩阵内核之间的差距。
2. 将窗口与全局 Attention 的 CUDA BF16、CUDA 实际后端、AMD BF16 分别测量。当前 Gather 融合有输出一致的局部收益，可作为保守候选；单纯扩大 K 切块已经测出更慢。
3. 对齐 FFN up/down 的原生 FP8 GEMM，确认 CUDA 与 AMD 的 scale、accumulation、输出精度、尾块及软件版本，再按 TFLOPS 定位差距。
4. 最后处理状态求解、560 次扫描更新及零散数据搬运；它们是小热点，不能凭 CPU 等待或调用数解释整段耗时。

## 可复现对齐工具与记录

[`benchmark_operators.py`](../../scripts/amd/benchmark_operators.py) 不需要模型权重，使用真实形状、固定 CPU 种子、输入 SHA256 和原 stride，记录预热 GPU event 计时、同步墙钟、有效吞吐及逐项 profiler trace。量化及输入构造在计时外，FP8 测的是 GEMM 本身。CUDA/ROCm 均可运行；生产采样没有导入此工具。

AMD：

```bash
FV_CACHE_GROUP=cuda-parity-repro bash scripts/amd/run_rocm_fast.sh \
  /workspace/scripts/amd/benchmark_operators.py \
  --device-uuid GPU-b11d6bcf3a61a551 \
  --out /data/experiments/freevideo-r9700/reports/cuda-parity-repro
```

在已配置 CUDA Torch 的 NVIDIA 机器上，尽量对齐 Torch 主次版本，先运行同精度比较：

```bash
python scripts/amd/benchmark_operators.py --blas default \
  --attention torch-flash --out /path/to/new-cuda-report
```

若已有 SageAttention，再单独选择 `--attention torch-flash,sage2` 测实际后端；FP32 统计的 TF32 对照用 `--cases fp32_state_gram --tf32`，与 IEEE FP32 结果分开。GPU 使用对应机器的可用设备；可提供 `--device` 和 `--device-uuid` 加设备租约。

[`compare_operators.py`](../../scripts/amd/compare_operators.py) 校验形状、dtype/stride、输入 SHA256、scale、TF32 与 Attention 精度；不匹配就拒绝给速度比。它标记 Torch 主次版本是否一致，不把微基准当成完整质量验证。

```bash
python scripts/amd/compare_operators.py \
  --baseline /path/to/new-cuda-report/operators.json \
  --candidate /path/to/amd-report/operators.json \
  --candidate-attention rocm-triton --out /path/to/new-comparison.json
```

本轮报告保存在 `/dc1/zihaomu/free_token_mapping/experiments/freevideo-r9700/reports/cuda-parity-20261007`：`amd-operators-v1` 为 30 项算子/后端测量，`amd-state-video-v2` 为完整视频状态 batch 和 TF32 补测，`real-bmm-v1` 为真实输入及三轮交替 block 测试，另有 `real-operator-efficiency.json`、`tf32-capability.json`、`cuda-access.json`。v1 的 `state_solve` 是上三角小 batch 探针，不当作生产下三角视频求解的对照；当前工具及 v2 已按真实下三角、70×16 batch 对齐。

原型和冻结的 v1 工具位于同实验根目录的 `prepared/cuda-parity-20261007`。合成探针、真实 block 对照及设备查询均串行运行并使用设备租约。源码和报告 SHA256 另行归档；比较器的六项 CPU 测试覆盖不匹配形状、输入、布局、精度及失败输出，防止发布无效速度比。
