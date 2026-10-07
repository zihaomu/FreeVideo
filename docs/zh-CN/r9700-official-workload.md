# R9700 官方工作负载对齐

从 2026-10-07 开始，AMD 后续端到端测速默认使用 **1344×768、请求时长 10 秒、8＋3 步二次采样**。参数固化在 [`workload.json`](../../scripts/amd/workload.json)，[`run_case.py`](../../scripts/amd/run_case.py) 显式传给生成器并校验实际请求回执。

| 参数 | 默认值 |
| --- | --- |
| 输出分辨率 | 1344×768 |
| 请求时长 | 10 秒 |
| 帧率与实际帧数 | 24 fps，243 帧，10.125 秒 |
| 第一轮 | 672×384，8 步 |
| 潜空间上采样 | 1344×768 |
| 第二轮 | 1344×768，3 步独立精化调度 |
| 总采样步数 | 11 步 |

10 秒先换算为 240 帧，再按 H3 VAE 的 `17*n+5` 约束向上对齐至 243 帧。音频沿用官方二次采样流程，保留第一轮音频。

## 官方依据

固定参考上游 commit `7c3536a999e23ba0e6404b085cbdf42718d435d5`。[官方端到端记录](https://github.com/FlashML-org/FreeVideo/blob/7c3536a999e23ba0e6404b085cbdf42718d435d5/docs/zh-CN/execution-planning.md#端到端) 是 Windows、1344×768、10 秒、二次采样，RTX 5090 为 122 秒。[当前官方 CLI](https://github.com/FlashML-org/FreeVideo/blob/7c3536a999e23ba0e6404b085cbdf42718d435d5/freevideo_engine/cli.py) 默认开启二次采样，第一轮默认 8 步，第二轮默认 3 步。

历史 122 秒记录未注明当时的步数，本配置以当前官方源码的 8＋3 为准。此次对齐采样流程、分辨率和时长；AMD 的计算后端与硬件仍按本机运行。此前全分辨率单轮 8 步的 535.67 秒和实验性 BF16 替换的 508.86 秒属于旧工作负载。

## 后续测速入口

```bash
bash scripts/amd/run_official_case.sh \
  --name official-repro-v1 \
  --profile /dc1/zihaomu/free_token_mapping/experiments/freevideo-r9700/prepared/profiles/native-decoder-fast-v1.json \
  --repeats 4
```

该入口采用已验证的 Triton 空间卷积、窗口 Attention、视频 Linear 缓存和原生音频卷积配置。实验性 BF16 门控和状态矩阵替换默认关闭。每次使用新的 `--name` 保留原始结果。默认独立请求四次，首条为该实验独立 JIT 缓存的冷启动，其余三条用于预热中位数；每条重新编码提示词、加载模型、完成采样和解码。

只查看计划可加 `--dry-run`。可显式覆盖 `--seconds`、`--width`、`--height`、`--base-steps` 或 `--refine-steps`；`--frames` 与 `--seconds` 互斥。复现旧单轮结果需显式提供 `--no-two-pass`。旧矩阵、短视频质量筛查及历史文档复现命令已补齐该参数。

## 本轮验证

参数检查已覆盖官方默认、243 帧对齐、历史 124/243 帧单轮模式、显式 8＋2 调度以及帧数与时长互斥。生成计划与固定上游源码的官方计划完全相同。

单张 R9700、已验证 profile、固定提示词与种子 `2026090901`，完成一次独立 JIT 冷启动请求：

| 测量项 | 耗时 |
| --- | ---: |
| 第一轮 672×384，8 步 | 117.99 秒 |
| 潜空间上采样阶段 | 4.33 秒 |
| 第二轮 1344×768，3 步 | 185.71 秒 |
| 视频 VAE 解码 | 49.86 秒 |
| 音频加载与解码 | 1.66 秒 |
| 端到端请求（含编码、加载、采样、解码与封装） | **385.77 秒，约 6 分 26 秒** |

这是一次冷启动验证，尚未测量新工作负载的预热中位数。回执确认实际执行两轮 8＋3 步，输出 1344×768、243 帧、24 fps。全部帧和音频完整解码通过，无黑帧或相邻完全重复帧；潜变量、提示词条件与音频均有限值，原始音频饱和比例为 0。已检查视频抽帧图；本轮未重新进行音频语义对照。

原始报告位于数据盘 `experiments/freevideo-r9700/reports/official-workload-20261007`，含固定的官方源码、运行源码哈希、profile、提示词及上采样权重 SHA-256 验证。`configuration-validation.json` 记录参数检查；`performance-summary.json` 记录测量；`final-validation.json` 确认 234 个运行源文件和 profile 在测试期间未变化。完整请求使用独立案例 `official-8plus3-s2-v1`，输出位于 `outputs/official-8plus3-s2-v1/run-00`。

后续[逐算子耗时调研](r9700-official-operators.md)完整采集同配置的全部 8＋3 步、上采样和解码，定位具体 GPU 热点，并验证输出与本次基线逐元素一致。
