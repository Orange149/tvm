# 第三创新点相关工作定位与可迁移规律

检索更新：2026-09-12。Cheng 等论文以用户提供的 9 页正式全文为首要来源；其他工作优先采用
论文、出版社页面、官方文档和本地源码。

## 1. 相关工作已经做了什么

| 工作 | 已有能力 | 本文不能冒充的新意 | 本文可做的增量 |
|---|---|---|---|
| TVM/AutoTVM | 模板化 schedule 空间、真实设备测量、学习型 cost model | XGBoost、随机/遗传搜索不是本文提出 | 在 `MeasureInput` 产生前加入 VTA 驻留与硬件证书；错误/未知候选不进入性能搜索空间 |
| Ansor | sketch 生成、随机补全、进化搜索和学习型模型 | 分层搜索不是本文提出 | 将 VTA 特有的 SRAM、DMA、UOP、cthread 条件做成可判定后处理/候选证书 |
| MetaSchedule | schedule rule、mutator、postprocessor；已有 GPU-code、VTCM-limit verifier | “搜索前验证硬件限制”不是本文首次提出 | 增加显式 DMA 加速器所需的双路径 lowering、全张量 DMA Pareto 和真实 FPGA correctness 证书 |
| Timeloop/MAESTRO | 用架构容量、映射、数据流和复用建立 mapspace/代价模型与 Pareto 点 | 容量约束、数据流分析、Pareto 搜索不是本文首次提出 | 从 TVM 最终 Lowered TIR/命令流提取真实编译结果，并接到 VTA AutoTune 派发和硬件 canary |
| VTA 原论文 | 参数化硬件、任务 ISA/微指令 ISA、显式 access--execute、operator autotuning | VTA 软硬件协同调优不是本文首次提出 | 固定 bitstream 后，系统化生成驻留变换并以精确硬件指纹缓存正确性证书 |
| Cheng 等 minimum-access 工作 | 构造 output/input-priority × 原始/权重复用四方案；修改 TIR/runtime 实现不覆盖式权重驻留；按 SRAM 适用性回退；三网络 pure-instruction 改善约 10%--25% | 输入优先、权重驻留、四方案、SRAM 条件和 fallback 均不是本文首次提出 | 论文未定义 ConfigEntity 候选搜索、trial budget、time-to-target 或等预算 tuner 对照；本文研究驻留×tile 联合空间中，以最终共享内存/DMA 先验减少找到 FPGA-correct 优解所需搜索成本 |

这里的“分层正确性门、实现哈希、强 incumbent、u-dma-buf 布局和命令定容”属于可信实验与部署
支撑，不再作为相对 Cheng 工作的核心学术差异。核心差异只落在搜索问题与搜索算法上。

## 2. 为什么这个结论不依赖“本板参数恰好等于 1”

本文不使用 `oc_nthread=1` 之类常数作为规律，而使用硬件参数化条件。对任意显式 DMA 张量加速器
硬件 `H` 和映射 `x`：

### 2.1 容量合法性

```math
F_m(x,H)\le C_m(H),\qquad m\in\{input,weight,acc,uop\}.
```

这里 `C_m` 来自目标硬件描述，`F_m` 来自变换后而非变换前的 buffer lifetime。换一块 FPGA，只需
替换容量和阵列参数，不需改判据。

### 2.2 虚拟 context 复制后的有效复用

TVM 的 `InjectVirtualThread` 对被 virtual-thread 触及或写入的 allocation 将 extent 乘以
`num_threads`，并重写访问下标。本地源码直接说明“每个 virtual thread 一份 buffer copy”。因此：

```math
F_m^{physical}=d_{ctx}(t,x)F_m^{logical},
\qquad
R_{effective}=\frac{R_{logical}}{d_{ctx}(t,x)}.
```

线程数 `t` 只有在复制后的 footprint 仍装入 SRAM、`R_effective>1` 且总 DMA 不恶化时才值得保留。
所以某实验中 `t=1` 较好，是因为该映射的复用不足以偿还 context 复制，并非“所有 FPGA 都选 1”。

### 2.3 全张量传输而不是单张量最小化

```math
T_{dma}\approx N_{dma}L_{req}+B_{dma}/BW_{eff}.
```

驻留 input 可能增加 weight，驻留 weight 也可能增加 input/同步。无平台常数时先采用保守 Pareto
条件：目标张量下降，并且所有 tensor 合计 bytes/calls 不增。若已标定 `L_req` 与不同 request shape
的 `BW_eff`，再使用上式排序，而不是拟合一个固定板卡阈值。

### 2.4 分析模型不能代替实现正确性

Timeloop/MAESTRO 说明架构约束与复用模型适合构建合法 mapspace；MetaSchedule 的 postprocessor
说明候选可在测量前拒绝。但 E03 显示本地 lowering 和 FSim 仍可能漏掉实际 FPGA 错误。因此对
模拟器没有覆盖的约束，通用处理不是猜测一个新的 tile 常数，而是：

```text
unknown exact implementation identity -> hardware correctness canary -> pass/fail certificate
```

证书键包含硬件/bitstream/源码/workload/完整映射/TIR；改变任一项后重新成为 unknown，避免把单板
经验错误外推。

## 3. 与当前 TVM 实现的直接对应

- `src/meta_schedule/utils.h`：任一 postprocessor 返回 false，候选 trace 立即成为 `NullOpt`；证明
  “硬件 verifier 位于搜索测量前”与 TVM 现有架构一致。
- `src/meta_schedule/postproc/verify_gpu_code.cc` 与 `verify_vtcm_limit.cc`：已有 GPU 代码和片上容量
  verifier；本文实现的是 VTA 显式 DMA/命令/FPGA 版本。
- `python/tvm/autotvm/tuner/model_based_tuner.py`：失败测量会以 `ys=0.0` 更新模型。因此只在 runner
  内发现 wrong answer 仍会影响模型；本文必须在产生 `MeasureInput` 前用证书 allowlist 裁剪空间。
- `vta/python/vta/build_module.py`：input/weight/acc 的 `MemoryInfo.max_num_bits` 直接取当前 VTA
  环境容量，支持以配置文件参数化而不是硬编码 AXU5EVB 数值。
- `src/tir/transforms/inject_virtual_thread.cc`：被 context 触及的 allocation extent 乘线程数，构成
  `d_ctx` 复制公式的源码依据。

## 4. 当前差异化结论

最稳妥的论文定位不是“提出新的 AutoTune”或“提出 input-stationary”，而是：

> 面向软件显式管理片上存储与 DMA 的张量加速器，提出一种以强 incumbent 为锚点的驻留变换和
> 分层硬件证书调优方法。该方法将 context 复制后的有效复用、变换后 SRAM/命令容量、全张量 DMA
> Pareto 和精确实现身份的硬件正确性证书用于测量前裁剪，只允许通过证书的候选进入学习型 cost
> model，并在未知或无收益时回退 incumbent。

该表述的普适对象是“编译器显式生成 DMA/命令、片上存储容量固定、模拟器与实际实现可能有差距”
的加速器；VTA/AXU5EVB 是实现与验证平台，不是规则中的常数来源。

## 5. 一手来源

1. Chen et al., TVM, OSDI 2018: https://www.usenix.org/conference/osdi18/presentation/chen
2. Zheng et al., Ansor, OSDI 2020: https://www.usenix.org/system/files/osdi20-zheng.pdf
3. Apache TVM MetaSchedule RFC 0005: https://github.com/apache/tvm-rfcs/blob/main/rfcs/0005-meta-schedule-autotensorir.md
4. Apache TVM MetaSchedule postprocessor API: https://tvm.apache.org/docs/reference/api/python/meta_schedule.html
5. Parashar et al., Timeloop, ISPASS 2019: https://research.nvidia.com/publication/2019-03_timeloop-systematic-approach-dnn-accelerator-evaluation
6. Kwon et al., MAESTRO: https://arxiv.org/abs/1805.02566
7. Moreau et al., VTA hardware--software blueprint: https://arxiv.org/abs/1807.04188
8. Cheng et al., VTA minimum-access schedule: https://doi.org/10.1016/j.future.2025.108165
