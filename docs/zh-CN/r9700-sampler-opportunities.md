# R9700 采样优化候选评估

2026 年 10 月 7 日，对同一张 gfx1201 R9700、1344×768、243 帧、8 步请求继续测试采样热点。当前完整请求预热中位数仍为 **535.67 秒**，采样约 **459.83 秒**。本轮评估的是首个真实 S2 block，复用已保存的 conditioning，并没有执行新候选的完整视频生成。以下局部结果和外推数字不能作为完整请求收益。

目前优先验证 **窗口 Gather 融合**，其次是 **仅合并 Softmax gate 投影**。两者在首个 block 上输出完全一致，但仍需覆盖其他 block、采样步骤并完成独立完整请求对照。默认生成代码和加速入口沿用已验证的原方案。

## 各方向的判断

| 方向 | 局部证据 | 下一步 |
| --- | --- | --- |
| 窗口 Gather 与 Attention 融合 | 14 组真实窗口及完整首个 block 输出完全一致；三轮交替测试 block 耗时降低约 1.4% | 优先做完整采样验证；按 400 个 block 粗估约 6.4 秒空间 |
| 仅合并 Softmax gate 投影 | Sigmoid 输出和完整首个 block 完全一致；投影约 20.5→9.6 ms | 单独验证其他 block 和完整请求；局部外推约 4 秒 |
| 窗口 Attention 内核 | 占 block GPU 时间 34.0%；扩大 K 切块使主要形状约 32.5→41.5 ms，反而更慢 | 大收益仍依赖内核的寄存器、访存和 softmax/WMMA 调度优化；继续保留全部 key |
| FP8 GEMM | 占 24.0%；FFN 两个大矩阵占 FP8 时间 56.7% | 优先分析 FFN 升维和降维的 GEMM 内核，保持原量化尺度及舍入点 |
| 合并 Beta gate 投影 | Sigmoid 相对误差 1.49e-5；合并两种门控后 block 相对误差扩大至 8.23e-4 | 不与 Softmax gate 一起采用，需要独立完整质量验证 |
| Q/K/V 一次投影 | 每个投影原分组约 23.7–24.6 ms，一次投影约 24.1 ms；有少量数值差异 | 没有稳定的明显收益，暂不采用 |
| 量化、SwiGLU 小算子 | 新方案输出一致，但量化约 3.7 ms、SwiGLU 约 0.35 ms，计时没有改善 | 本轮候选不采用 |

这些方向的收益不能直接相加：它们可能竞争同一段访存开销，时钟、内核布局及编译成本也会影响完整请求。

全局 Attention 占约 8.7%，此前更换后端出现音频标签回退，暂时保留原路径。空间和时间卷积合计约 5.4%，空间卷积已有优化，继续投入的优先级低于窗口 Attention 和大矩阵乘。

## FP8 时间具体花在哪里

按已有真实 block trace 的 `aten::_scaled_mm` 与 GPU kernel 的 External id 对应关系拆分，避免把父算子和子 kernel 重复累计：

| FP8 运算 | 每 block 调用数 | GPU 毫秒 | FP8 时间占比 | 全部 block kernel 时间占比 |
| --- | ---: | ---: | ---: | ---: |
| FFN 升维 | 36 | 96.45 | 36.64% | 8.80% |
| FFN 降维 | 36 | 52.80 | 20.06% | 4.82% |
| Q/K/V 投影 | 12 | 69.96 | 26.57% | 6.38% |
| Attention 两个输出投影 | 2 | 44.06 | 16.74% | 4.02% |
| 合计 | **86** | **263.27** | 100% | **24.01%** |

FFN 主切块的升维 GEMM 为 `M=2048, K=5376, N=28672`，降维为 `M=2048, K=14336, N=5376`，另有 1755 行尾块。Q/K/V 每个投影按 56 个头分为 16、16、16、8，即三个 2048 列块和一个 1024 列块。已有 FF chunk 实验只有约 2% 的 FFN 局部改善，不足以替代 GEMM 内核优化。

此前运行报告中的 `fp8_kernel_calls.torch_scaled_mm=4800` 是包装器计数，覆盖每 block 的 12 次头投影，不能解读为整段全部 FP8 GEMM 调用数。86 是本轮分析的首个 block 实测值，不是对全采样逐个 kernel 的计数。

## Gather 融合的验证范围

原路径先用 `q[rows]`、`k[keys]`、`v[keys]` 生成连续临时窗口，再调用 Attention。候选在 Attention 内按原索引读取，保留原窗口、anchor、全部 key、BF16 输入、FP32 online softmax 和 `BM/BN` 切块。

14 项对照覆盖 16 头和最后 8 头的全部 7 个实际窗口 batch。完整首个 block 对照进一步覆盖四个头组及原全局 Attention。原始浮点输出均完全一致。

为减少时钟漂移的影响，进行了三轮“原路径→候选→候选→原路径”交替测试，每个位置预热后测五次，再取中位数；每轮对两次原路径及两次候选的中位数分别取均值：

| 轮次 | 原路径 | 融合候选 | 节省 | 耗时降低 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1124.62 ms | 1109.06 ms | 15.56 ms | 1.38% |
| 2 | 1131.32 ms | 1115.34 ms | 15.98 ms | 1.41% |
| 3 | 1133.62 ms | 1117.48 ms | 16.14 ms | 1.42% |

用 8 步×50 block 外推，中位数约为 6.39 秒；实际只测了首个 block 的固定输入。没有完成新候选的整段采样、冷编译成本或其他分辨率性能验证。

## 门控合并为何要分开验证

一次计算 56 个头，再按原头组切片，可以减少窄矩阵乘对大输入的重复读取。它仍是 BF16 计算、原 rocBLAS 后端，没有切换到此前被拒绝的 hipBLASLt 门控方案；矩阵形状变化仍可能改变浮点舍入。

Softmax gate 的线性结果有极少差异，但这份真实输入经过 Sigmoid 后完全一致。仅合并它的完整首个 block 输出也完全一致：原路径约 1123 ms，候选约 1116 ms，回到原路径约 1131 ms。投影本身约节省 11 ms，按 400 block 粗估约 4 秒；此候选尚未做三轮交替测试或完整请求验证。

Beta gate 的 Sigmoid 输出有约 0.00145% 元素改变。把两种门控同时合并后，完整 block 约 11.34% 元素改变，相对 RMSE 为 8.23e-4，说明微小的 Beta 差异会在后续分支中传播。这里没有据此宣称视频或音频必然退化，也没有把约 20 ms 的局部提速采用到默认路径；完整质量仍需另测。

## 记录和复现

固定环境和工作负载沿用 [耗时分布与解码优化](r9700-bottlenecks.md)。计算源码基于 commit `25f15da`，本轮生产计算文件未修改。GPU 实验串行执行，并使用设备对应的运行锁。

实验目录为 `/dc1/zihaomu/free_token_mapping/experiments/freevideo-r9700`：

- 报告：`reports/sampler-opportunities-20261007`，包含环境信息、`fp8-stage-breakdown.json`、`indexed-local-summary.json` 和各项原始重复计时。
- 原型：`prepared/sampler-opportunities-20261007`。`indexed_attention.py` 是融合内核，`merged_softmax_head_chunk.py` 是单独合并 Softmax gate 的原型；没有导入默认生产路径。
- 复现驱动：`study_indexed_windows.py`、`study_indexed_windows_alternating.py`、`study_merged_softmax.py`；其余候选由 `study_full_projections_v2.py`、`study_window_key_tiles.py`、`study_quantization.py`、`study_merged_gates.py` 记录。
- 局部输入：`outputs/bottleneck-decoders-s2-v1/run-01/video.artifacts/conditioning.pt`；profile 为 `prepared/profiles/native-decoder-fast-v1.json`。驱动在首个 block 测量后停止，不生成完整 MP4。

通过 `run_rocm_fast.sh` 调用相应 `/data/experiments/freevideo-r9700/prepared/sampler-opportunities-20261007/` 驱动，提供 `--profile`、`--conditioning`、`--out`。复现使用新的报告目录和缓存组，保留本轮数据。原型及 SHA256 归档为 `sampler-candidate-source.zip` 和 `sampler-source-sha256.json`；初始投影驱动的迭代器错误单独保存，不参与收益计算。

方法参考：[Triton 官方 Fused Attention 示例](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) 提供切块搜索思路；[AMD hipBLASLt 离线调优文档](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to/how-to-use-hipblaslt-offline-tuning.html) 描述按具体 GEMM 搜索算法，结果不能跨库版本或架构复用。这里的性能和误差结论均来自本地实测，没有把其他 GPU 的收益套用到 R9700。
