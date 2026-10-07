# R9700 借鉴 h3-vdn.c 的算子优化路径

当前最值得实施的是**按分辨率选择索引融合与 Softmax gate 合并，然后独立评估全局 Attention，最后调优现有 FP8 FFN**。在真实首块交叉测试中，全分辨率的前两项组合把 GPU event 耗时从 1129.45 ms 降至 1101.79 ms，降低 **2.45%**，输出逐元素一致。低分辨率应只合并 gate，索引融合没有收益。

这些是局部验证，尚未构成新的端到端加速或完整音视频质量结果。全分辨率首块使用固定种子的第一步激活，**不是原始第二轮精化的输入**；672×384 测试则使用与基线相同的第一轮形状和种子。原始 385.77 秒及全步热点见[逐算子报告](r9700-official-operators.md)。

## 当前热点与可实施路径

| 当前 GPU 热点 | 实现方向 | 当前证据及适用范围 |
| --- | --- | --- |
| 窗口 Attention 69.02 秒，索引相关 11.28 秒 | 将 Q/K/V 索引读取并入现有 WMMA Attention，避免先构造窗口副本；保留 key 顺序与 BF16/FP32 运算 | 全分辨率真实首块单独降低 1.57%，输出一致；低分辨率无收益，需按形状选择 |
| BF16 门控及相关投影 19.98 秒中的 Softmax gate | 一次计算全部 Softmax gate，再按原 head group 使用；保留 Beta 和其他投影的原路径 | 首块降低约 1.01%／1.28%；与全分辨率索引融合组合降低 2.45%，输出一致 |
| 全局 Attention 20.81 秒 | 针对矩形 Q/K 长度采用当前 BF16 Triton WMMA 路径，比较流水线策略 | 合成形状探针最高 1.47 倍；较保守的一阶流水线在真实全分辨率首块降低 2.40%，但改变结果，需要多步音视频验证 |
| FP8 FFN up/down 39.48 秒 | 保留已有 E4M3FN 权重、量化 scale 和 BF16 输出，比较 2048／4096／8192 行 chunk，以及相同数学合同的库算法 | 8192 行归一化工作量中，up 约 1.03 倍、down 约 1.13 倍；仅为 GEMM 探针，需计入 SwiGLU、全局 activation scale、真实尾块和峰值显存 |
| 视频 VAE FP16 GEMM 31.64 秒 | 保留现有 hipBLASLt FP16 路径，研究 bias epilogue 和 addmm 内部复制 | 更换回 rocBLAS 会明显退化；addmm 内部复制约 2.84 秒是后续可研究目标，融合收益尚未验证 |
| 状态求解、扫描和时间／空间卷积 | 保留 FP32 求解与现有加速内核，最后研究减少启动和搬运 | 优先级低于前述热点；不能直接套用 native H3 的版本相关 solution index |

表中 GPU 热点来自同输入的完整 8＋3 步 trace，局部加速来自本轮独立探针；两种计时不能直接相加。门控合并只覆盖其中一个投影，不能将其收益套到整类 19.98 秒。

## h3-vdn.c 中哪些实现可借鉴

固定检查 [`vdn-h3-rocm` 的提交 6391dd0](https://github.com/zihaomu/h3-vdn.c/tree/6391dd0fe5fb4ca4611f2bf836b80beb14ce158d)，同时检查 HIP 源码、优化 ledger 和后续工作区集成记录。

**窗口索引直接读取和按形状选择 kernel 可借鉴。** Native H3 的窗口实现根据 frame、chunk、anchor 和 global 范围直接访问原张量，可用于减少 gather 中间结果；当前 FreeVideo 已有精确的分解窗口计划，可以把该思路用于 WMMA 路径。[HIP 实现](https://github.com/zihaomu/h3-vdn.c/blob/6391dd0fe5fb4ca4611f2bf836b80beb14ce158d/h3_gpu_hip.cpp#L2244)的 wave32 BF16 kernel 逐键执行标量归约，并非当前 FreeVideo 的 WMMA Flash Attention，因此应迁移索引和调度思路，而不是整段替换。

**FP8 的显式算法搜索方法可借鉴，旧倍率不能套用。** H3 的 probe 对照是 BF16 rocBLAS 与新量化的 FP8；FreeVideo 已使用 native FP8。后续应沿用现有 FP8 bytes 和 scale，按真实 M/N/K 搜索算法、workspace 与输出精度，不重新量化有效权重。[H3 FP8 探针](https://github.com/zihaomu/h3-vdn.c/blob/6391dd0fe5fb4ca4611f2bf836b80beb14ce158d/tests/bench_vdn_fp8_gemm.hip)提供了相同 row-major 数据的等价列主序视图和 heuristic 测量方法。

**Sage 的任务拆分与 gfx12 WMMA 布局可供进一步设计参考，但已拒绝的数学路径不进入当前默认配置。** H3 的 E27 INT8 QK 路径没有通过三提示词音频筛查；后续 E33 补偿路径在八步运行中仍出现音频回退。这些结果说明算子误差小并不保证多步音频质量。H3 还拒绝了逐行 FP8 权重 roundtrip 和 block-scaled weight-only fallback；当前应保持已验证的模型量化方式。[优化记录](https://github.com/zihaomu/h3-vdn.c/blob/6391dd0fe5fb4ca4611f2bf836b80beb14ce158d/doc/VDN_ROCM_OPTIMIZATION.md)

**VAE F32 split-score 不适用于当前 FP16 Flash 路径。** H3 后续对 S1797 的 split-score 做了三轮完整媒体验证，确实改善了其 F32 wave32 基线；该验证范围见[工作区集成记录](https://github.com/zihaomu/h3-vdn.c/blob/6391dd0fe5fb4ca4611f2bf836b80beb14ce158d/doc/KERNEL_WORKSPACE_INTEGRATION_ASSESSMENT.md)。但 FreeVideo 相同 S1797/H32/D64 形状已经使用 FP16 Flash，本轮直接编译原始两个 F32 kernel 对照后明显更慢。

H3 的 GEMM solution index 还限定了 gfx1201、rocBLAS 5.2.0 和特定 F32 形状；本机为 Torch 2.12.0＋ROCm 10.0 wheel、HIP 7.15。不能把旧索引当成通用快速算子。

## 真实首块验证

同一 R9700，固定提示词条件和种子；每个候选做两轮 baseline→candidate→candidate→baseline，共四个预热 event 计时／每臂。每项都独立回到原路径，以下耗时已经包含该 block 的其他计算，不能将多个行的降低比例简单相加。

| 分辨率 | 候选 | 基线 ms | 候选 ms | 耗时降低 | 相对 RMSE | 逐元素一致 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 672×384 | 索引融合 | 240.95 | 241.27 | −0.13% | 0 | 是 |
| 672×384 | Softmax gate 合并 | 241.43 | 238.35 | **1.28%** | 0 | 是 |
| 672×384 | 索引＋gate | 241.44 | 238.75 | 1.11% | 0 | 是 |
| 1344×768 | 索引融合 | 1122.10 | 1104.43 | **1.57%** | 0 | 是 |
| 1344×768 | Softmax gate 合并 | 1125.87 | 1114.45 | 1.01% | 0 | 是 |
| 1344×768 | 索引＋gate | 1129.45 | 1101.79 | **2.45%** | 0 | 是 |
| 1344×768 | 全局 Triton，stages 1 | 1129.34 | 1102.26 | 2.40% | 0.001744 | 否 |
| 1344×768 | 窗口 stages 2 | 1125.95 | 1106.85 | 1.70% | 0.000385 | 否 |

这里的“输出一致”是所测首块，不代表全部 50 blocks、全部 11 步和最终解码已经通过。非一致候选必须先做真实精化输入、多 block、多步及视频／音频对照。

## 算子探针结果

窗口和全局探针使用 BF16 Q/K/V、FP32 online softmax，保留全部 key；FP8 探针使用固定 E4M3FN operands 和 unit per-tensor scales，量化及输出拼接不计时。每项七次交替 baseline／candidate，汇总取中位数。

| 探针 | 基线 ms | 最佳候选 ms | 局部倍数 | 数值说明 |
| --- | ---: | ---: | ---: | --- |
| base window，B4/Q1260/K5143/H16/D128 | 2.515 | 2.255 | 1.115 | BM128/BN64/warps8/stages2；非逐元素一致 |
| refine window，B4/Q5040/K17995/H16/D128 | 32.479 | 30.593 | 1.062 | BM128/BN64/warps8/stages2；非逐元素一致 |
| full global，B1/Q2875/K73435/H16/D128 | 26.617 | 18.076 | 1.473 | 同上；非逐元素一致，真实 block 采用更保守的 stages1 |
| FP8 FF up，8192 行总工作量 | 11.792 | 11.406 | 1.034 | 2048→8192 chunk；本合成输入一致 |
| FP8 FF down，8192 行总工作量 | 6.381 | 5.672 | 1.125 | 2048→8192 chunk；本合成输入一致 |

更大的 BN、更多的流水级并非普遍更快。低分辨率真实 block 没有采用上述 BM128/stages2 策略；归约变化需要质量验证。

QKV／Attention 输出另有人工切分控制探针，但**没有用于当前生产收益排序**：该 profile 的 QKV 和 native Attention 输出已经按整段行数执行，`projection_chunk=1024` 主要约束模型输入／最终输出打包等路径，不表示 QKV 有 1024 行切块。将人工 1024 行对照的约 1.21 倍误认成现成生产收益会高估空间。原始结果保留，并以 `operator-scope-audit.json` 明确排除；最终工具将这类控制设为可选。

VAE 的对照结果进一步限定了适配方向：

| VAE 算子 | 当前路径 ms | 对照路径 ms | 结论 |
| --- | ---: | ---: | --- |
| S1797/H32/D64 Attention | FP16 Flash 0.458 | 原始 H3 F32 split＋输入输出 cast 31.093 | 对照约慢 67.8 倍，不移植 |
| M1797/K2048/N2048 FP16 Linear | hipBLASLt 0.131 | rocBLAS 1.515 | 保留当前路径 |
| M1797/K2048/N16384 FP16 FF up | hipBLASLt 0.998 | rocBLAS 11.817 | 保留当前路径 |
| M1797/K8192/N2048 FP16 FF down | hipBLASLt 0.573 | rocBLAS 7.943 | 保留当前路径 |

H3 F32 Attention 的 kernel 保持原源码及归约顺序，只有外部 stream-aware C ABI。分数 workspace 约 0.385 GiB，计时排除了编译和 workspace 分配，包含必要的 dtype 转换。这一对照是候选路径适用性测试，数值合同与现有 FP16 Flash 不同。

## 实施顺序

1. 将已验证的索引 kernel 和 Softmax gate 合并整理为独立候选。全分辨率启用二者，低分辨率只合并 gate；窗口尾块和 H8 尾 head group 保留原语义。先验证完整官方 8＋3 步逐元素结果，再交叉测量独立请求的预热中位数。
2. 单独评估全局 Triton 与窗口流水线候选。先覆盖真实精化 latent、边界窗口、原 stride 和 H8／H16，再检查全部步骤的音视频误差。不能因首块相对误差小就默认通过。
3. 对 FFN 4096／8192 chunk 做真实模块测试。保持整段 activation scale，核对 up→SwiGLU→量化→down 的完整时间与峰值显存；若库算法搜索更有收益，则固定实际版本、形状和 scale 合同。
4. 最后研究 VAE FP16 bias epilogue／布局复制，以及状态扫描启动次数。现有 VAE 库路径和 FP32 求解继续作为验证基线。

当前局部数据支持小幅累计改进，还不足以解释或解决全部 385 秒与官方 NVIDIA 结果的差距。候选没有切换生产默认值，采样步数、分辨率和时长沿用官方配置。

## 复现与记录

[`probe_h3_paths.py`](../../scripts/amd/probe_h3_paths.py) 和 [`probe_h3_real_block.py`](../../scripts/amd/probe_h3_real_block.py) 分别提供算子对照和真实首块交叉测试。使用现有隔离 R9700 容器和设备租约，所有数据写入新目录：

```bash
FV_CACHE_GROUP=h3-paths-20261007 bash scripts/amd/run_rocm_fast.sh \
  /workspace/scripts/amd/probe_h3_paths.py \
  --h3-source /data/experiments/freevideo-r9700/references/h3-vdn-20261007 \
  --device-uuid GPU-b11d6bcf3a61a551 \
  --out /data/experiments/freevideo-r9700/reports/new-h3-operator-probes
```

完整记录位于 `/dc1/zihaomu/free_token_mapping/experiments/freevideo-r9700/reports/h3-paths-20261007`。本轮保存 51 项算子对照、两个分辨率的 8 项真实 block 候选、固定源码与 SHA256、native 编译日志及原始计时。`operators-v1/v2` 的编译失败记录保留；`operators-v3` 为完成的数据。`capture-source.json` 对应实际运行版本，最终工具补充了人工投影控制的范围标记。

真实 block 原型固定在该目录的 `prototypes`，源于本仓库先前保留的索引及 gate 合并实验。核验每项输出有限值、合成算子重复一致和两轮交叉计时，完整媒体验证留在候选正式集成阶段。
