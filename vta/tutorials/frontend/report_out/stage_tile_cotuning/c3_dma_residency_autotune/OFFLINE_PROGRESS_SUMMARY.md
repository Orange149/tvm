# C3 离线阶段进展总结：面向 VTA 显式 DMA 的驻留策略与请求形态协同调优

> **板端状态更新（2026-09-11）：** 本文档原有正文是冻结的离线阶段快照；其中“待上板”的表述已由同日 P6c/P6d/P6e 实验推进。最新板端结论见本文末尾“八、RAM-only 板端推进结果”以及[P6e 状态](06_prospective_board/20260911_p6e_same_tile_paired_ablation_run01/STATUS.md)。

更新日期：2026-09-11  
证据范围：P2 run02、P2b、P3--P3e、P4--P4j、P5/P5b/P5c、P6a/P6b；本总结不访问开发板。

离线阶段综合回归为 **138 passed**，135 个 JSON/JSONL 文件、1825 条记录全部可解析，构建与 `git diff --check` 均通过；完整记录见[离线最终验证](00_governance/OFFLINE_FINAL_VALIDATION.md)。

## 一、论文创新点应当怎样讲

这一创新点不应讲成“看到 input、weight、DMA 都能改，于是分别做几个小优化”，而应讲成一条完整的问题求解链：

> 在 FPGA 硬件结构、VTA bitstream 和共享内存数据通路固定后，首先由算力与带宽模型确定单算子的理想性能边界；然后从最终 Lowered TIR 提取真实的逻辑 DMA 请求形态，识别调度把算子从 compute-bound 推向 DMA-bound 的风险；再把驻留模式、tile 实体、片上容量、VTA 队列拓扑和请求碎片作为同一候选的编译期约束与特征，在保留 TopHub 强 incumbent 的前提下剪去非法和明显退化候选，减少有限板端预算的浪费。最后只用前瞻、分组留出的板端试验判断该方法能否在不损失强基线性能的情况下减少派发，或把单算子收益传递到 stage/流水。

因此，候选的决策链是：

```text
硬件指纹与卷积几何
  → 驻留模式和完整 tile 实体
  → Lowered TIR
  → SRAM/DMA形态/VTA依赖合法性
  → 请求签名（bytes、calls、size histogram、stride/pad、reload）
  → incumbent保护下的shortlist
  → 正确性与板端测量
  → stage服务时间和整网流水验证
```

创新增量不是首次提出 input-stationary、weight-stationary 或一般 AutoTVM，而是针对固定 VTA 显式 DMA 共享内存路径，把“按 workload 选择驻留模式、最终 TIR 请求形态、FPGA 约束证书、稳定候选身份、强 incumbent 保护和有限预算验证”连接成一个安全调优流程。2026 年相关文章在本项目中只能称为 `paper_inspired_hybrid` 的方法来源/对照，现有公开证据不足以声称 exact reproduction；详见[论文复现状态](01_paper_reproduction/STATUS.md)和[主控计划的创新边界](../C3_DMA_RESIDENCY_AUTOTUNING_MASTER_PLAN.md)。

截至当前，这一表述仍是“已形成可验证的方法框架，局部机制与约束工具已有证据”，尚不能写成“独立创新点的性能有效性已经完成”。最终能否作为独立创新点，取决于 grouped holdout 上的派发效率和上板后的性能/正确性结果。

## 二、理论上限：先回答还能提升多少

冻结配置为 `BLOCK_IN=BLOCK_OUT=16`、100 MHz、128-bit AXI。对 packed convolution：

```text
MAC = N × OH × OW × Cout × Cin × KH × KW
Peak = BLOCK_IN × BLOCK_OUT × FREQ = 25.6 GMAC/s
Tcompute = MAC / Peak
Bdirection = AXI_DATA_BITS × FREQ / 8 = 1.6 GB/s
```

最乐观 full-duplex 情景必须分开读写：

```text
Tread  = read_bytes / Bdirection
Twrite = write_bytes / Bdirection
Tperfect-overlap = max(Tcompute, Tread, Twrite)
Tno-overlap      = Tcompute + Tread + Twrite
```

`(read_bytes+write_bytes)/Bshared` 只作为读写共享同一串行瓶颈的另一种情景，不能冒充 full-duplex 的最乐观下界。1.6 GB/s 假定每周期都是有效载荷，未包含 AXI 协议、竞争、短突发、DMA 启动和流水线气泡，因此也是 optimistic 值，而不是实测带宽。

对冻结 10 类卷积，当前静态 config 在 full-duplex 理想模型中均为 compute-bound；W03 最接近平衡，`Tcompute/Tdma=1.0064`。历史 80 点中可可靠关联 32 个正确候选，其不可约下界效率中位数为 **75.45%**，观测延迟相对下界的中位差距为 **1.325×**；最高有效算力为 **22.874 GMAC/s**，即 25.6 GMAC/s 理论峰值的 **89.35%**。这些数字来自历史单算子 latency 的离线关联，不是本轮新测量，更不是整网 FPS。公式、逐 workload 数值和历史关联见[P2b 公式](02_baselines/20260910_p2b_theoretical_bound_run01/FORMULAS.md)、[P2b 状态](02_baselines/20260910_p2b_theoretical_bound_run01/STATUS.md)和[P2b 汇总](02_baselines/20260910_p2b_theoretical_bound_run01/summary.json)。

在系统层面，既有静态资源模型给出的 II 下界为 **70.807 ms**，其倒数为 **14.123**；自然 Top-20 池内实测最好为 **11.511 FPS**。二者相除约为 **81.5%**，只能帮助直观理解“真实系统离乐观资源边界仍有距离”。`14.123` 是静态资源下界的倒数和排序量，不是准确的绝对 FPS 预测；11.511 FPS 则只是在冻结 Top-20 测量池内的最好结果。完整口径和表格见[硕士论文草稿第 7 章对应段落](../../resource_aware_maxplus/MASTER_THESIS_DRAFT.md)和[C1 收口证据](../C1_CLOSEOUT_EVIDENCE.md)。

## 三、为什么不能只按 DMA 总字节优化

P2b 给出了一个看似矛盾、实际上很关键的结论：冻结 10 类 workload 的当前静态 DMA 相对不可约 input+weight+output 流量仍有 **1.010–2.371×** 放大，中位数 **1.613×**；但是在“满带宽且完美重叠”的模型里，即使把当前流量全部降到不可约流量，10/10 workload 的理论 `max(Tcompute,Tread,Twrite)` 都不下降，因为算力项仍然最大。

这说明“减少 bytes”是必要特征，却不是充分的性能模型：

1. 相同总 bytes 可以被组织成少量长 burst，也可以被拆成大量小请求；固定启动成本不同。
2. stride、padding、compact-2D 限制会同时改变请求有效载荷率和可编译性。
3. cache lifetime 不当会形成 input/weight reload；历史正确候选中已有 **4/32** 在测后 runtime traffic 下呈 DMA-bound，而其余 28 个呈 compute-bound。
4. compute、LOAD、STORE 并非总能完美重叠，VTA 依赖 token 和单物理 VTA 的串行服务会改变关键路径。
5. 在整网层面，CPU stage、VTA service、边界 owner 和共享 DDR 是 `max` 型资源约束，不能把从同一 VTA service 派生的 DMA 时间再无条件相加，否则会重复计费。

已有整网消融同样支持这一谨慎结论：M2 的共享 DDR 下界在 **972,528** 个执行配置中 **0 个**成为预测瓶颈，最接近者也只有 M1 下界的 **17.27%**，M1/M2 Top-20 为 **20/20** 相同。这是“当前带宽下界未改变排名”的负结果，不是“物理 DDR 不影响吞吐”的证明，见[内存模型消融](../stage_memory_experiments/MEMORY_MODEL_ABLATION.md)。

另一方面，历史退化案例显示请求形态有很强的诊断价值：`tile_w` 邻居的 LOAD calls、payload、runtime 中位数分别为 incumbent 的 **4.5×、4.173×、3.374×**；一次弱新构建相对旧 topology B 的 LOAD calls、payload 和 device wait 分别放大 **14.91×、3.78×、3.39×**。这不能直接证明某个请求特征具有普适因果系数，但足以说明 shortlist 不能只看总 bytes，且必须保留历史强 incumbent。证据见[主控计划已有证据](../C3_DMA_RESIDENCY_AUTOTUNING_MASTER_PLAN.md)和[旧强基线恢复](../legacy_baseline_recovery/README.md)。

## 四、驻留模式、请求形态和 FPGA 约束如何统一

驻留模式改变的是“数据在 SRAM 活多久、跨哪些循环复用”，最终效果必须落到 Lowered TIR 的 LOAD/STORE 上。候选不应直接按 mode 名称加分，而应经过两层判断：

### L0：硬约束与安全证书

- input/weight/ACC/UOP 等容量是否越界；
- DMA 是否满足 compact 2-D、padding 和 innermost 约束；
- VTA LOAD(1)、COMPUTE(2)、STORE(3) 的依赖边是否能由物理 token 通路表示；
- lower、交叉编译、FSim/板端正确性是否通过；
- TopHub incumbent 是否始终保留；
- candidate ID 是否绑定硬件指纹、模板、schedule 版本、workload、mode 和完整 ConfigEntity，而不依赖易变的 `config.index`。

### L1：在合法候选中比较请求形态

- input、weight、output 分类 bytes 及 reload ratio；
- LOAD/STORE calls、small request ratio；
- request-size histogram 和最大请求；
- stride/padding 比例；
- 后续补齐的 instruction/UOP/FINISH/replay 足迹；
- workload 是否位于流水关键 stage。

这使 FPGA 信息不是泛泛地“告诉 AutoTVM SRAM 有多大”，而是变成确定的候选拒绝理由和 Pareto 维度。P4 已实现规范 JSON SHA-256 身份：10/10 冻结 workload 产生 10 个互异 ID，7/7 单测通过；历史适配器保留了分类 bytes、reload、最大请求和精确 request-size histogram，但 command footprint 仍明确为 `not_measured`，见[P4 状态](04_dma_command_signatures/20260910_p4_local_identity_adapter_run01/STATUS.md)。

P4b 又把历史 80 个来源映射为 **60 个唯一强制 tile 实体**，去重 **20** 个重复来源，保护 **10** 个 original incumbent，并形成 **180** 个驻留实验候选。这个去重本身就是可审计的搜索空间压缩。250 个总候选中有 165 个 lower 成功、85 个被本地合法性检查淘汰；165 个成功项随后全部通过三 seed FSim，并全部完成 AXU5EVB 交叉编译/导出资格，仍不包含板端性能。见[P4b 状态](04_dma_command_signatures/20260910_p4b_local_residency_pool_run01/STATUS.md)、[统一本地资格](04_dma_command_signatures/20260911_p4e_unified_local_qualification_run01/STATUS.md)和[交叉编译资格](04_dma_command_signatures/20260911_p4f_axu_cross_compile_run01/STATUS.md)。

## 五、当前证据必须分三层陈述

### 5.1 已证实：本地、静态或模拟器层面的事实

1. **原路径可重复。** P2 run02 的 10-workload 静态 DMA 与冻结文件逐字节相同，SHA-256 为 `ff154c...a3a5`；3 seeds × 10 workloads 的 FSim 为 **30/30** 正确；AXU5EVB 交叉编译 **10/10** 成功；10/10 TIR-IR 与对照一致。它证明本地原路径没有回归，不证明当前 boot 的板端性能，见[P2 run02 状态](02_baselines/20260910_p2_local_baseline_run02/STATUS.md)。
2. **input-stationary 存在局部正机制。** W00/W02/W09 的 input DMA bytes 分别下降 **50%、75%、50%**，四种安全模式在这三个 workload 上均能 lower 并通过 FSim。该结果是静态逻辑 DMA 和模拟正确性，不是 AXI transaction 或 latency，见[P3 结果](03_residency_schedules/20260910_p3_local_residency_run01/results.json)。
3. **原先的安全 weight 模式没有减重。** W00/W02/W09 的 safe weight-stationary 静态 weight bytes 与 original 完全相同；safe hybrid 的收益全部来自 input。P4b 扩展到 60 个 tile 后也得到 weight-stationary lower 成功 **43/60**、但 weight reduction **0/43**；paper-inspired hybrid lower 成功 **35/60**、target DMA 正下降 **28/35**，其 weight reduction仍为 **0/35**。因此这两个旧模式不能冒充已经实现的片上权重驻留。
4. **失败根因已定位，并形成了编译器级修复。** 首次把 weight cache 提升到外层后形成跨循环的 `STORE(3)→LOAD(1)`；VTA 只有 `1↔2` 和 `2↔3` token 通路，没有 `1↔3` FIFO，删除 runtime assertion 并不安全。P3d 证明仅放一个旧式 sync pragma 仍然不可执行；P3e 将显式 full-sync 改为注册的 `tir.vta.coproc_sync`，并让 `CoProcInstDepDetector` 在该屏障处封闭前段 token、清空依赖状态。这个编译器依赖分段增强没有修改 runtime/RTL/ISA，见[P3b 根因审计](03_residency_schedules/20260910_p3b_weight_reuse_audit_run01/WEIGHT_REUSE_FEASIBILITY.md)、[P3d 负结果](03_residency_schedules/20260910_p3d_weight_sync_probe_run01/STATUS.md)和[P3e 正结果](03_residency_schedules/20260910_p3e_weight_barrier_legalizer_run01/STATUS.md)。
5. **显式屏障使权重驻留在本地变为可执行机制。** P3e 的 mode 4 在 W00/W02/W09 上均 lower、build 成功，依赖 push/pop 平衡且无 `1↔3`，9/9 个 FSim seed 与独立 NumPy 参考逐元素一致。W00、W02 的 weight bytes 分别下降 **85.71%** 和 **75%**；W09 的 bytes 不变，但 weight LOAD calls 从 **32 降到 2**，说明“总字节相同而请求形态不同”确实可以由 schedule/编译器机制产生。代价是每个驻留组增加 full-sync；其性能净收益必须上板测。mode 4 当前仍是 probe，尚未自动升格为正式候选赢家。
6. **权重屏障机制已经扩展到完整 60-tile 池。** P4g 为 60 个 mapped ConfigEntity 生成与原 250 ID 零重叠的新 stable ID，并把 schedule version 绑定到 schedule、VTA transform 和通用 CoProcSync 源码哈希。32/60 通过 lower 与依赖审计，28 个失败分为 allocation 13、compact 7、pad 5、2-D pattern 3；合法项 **32/32** 无 `1↔3` 且全部通过三 seed FSim（96 次检查）。其中 **22/32** 减少 weight bytes，降幅中位数 **50%**、最大 **87.5%**，10 个 bytes 不变。每候选有 2 个静态 sync callsite；展开后总 sync 为 2--9 次、中位数 3，新增 residency drain 为 1--8 次、中位数 2。P4i 又对这 32 个合法候选完成独立 AXU5EVB build/export，**32/32** 通过且未保留二进制。这一分布量化了“少 LOAD 与多同步”的待测权衡；交叉编译只证明本地可部署性，不证明当前 boot 可加载、正确或加速，见[P4g 状态](04_dma_command_signatures/20260911_p4g_weight_barrier_pool_run01/STATUS.md)和[P4i 状态](04_dma_command_signatures/20260911_p4i_weight_barrier_cross_compile_run01/STATUS.md)。
7. **依赖根因已有自动拒绝工具。** P3c 依赖审计器测试 **10/10** 通过；人工合法 case exit 0，禁用的 `3→1` case exit 2。P3e 又以纯 TIR barrier 测试和完整 VTA 回归验证依赖分段，最终合并回归 **50/50** 通过。静态审计不展开所有动态路径，仍不能替代 FSim/runtime。
8. **约束能在不上板时暴露大量不可行候选。** P4b 中 input、weight、paper-inspired hybrid 分别只有 **32/60、43/60、35/60** 能 lower；85 个总失败已分为 allocation capacity 24、DMA 2-D pattern 22、compact buffer 14 和 pad-innermost 25。input-stationary 在成功 lower 的 32 个中有 **25** 个减少 input bytes，中位下降 **50%**。
9. **所有既有 lower-success 候选均已完成本地资格。** 110 个驻留实验候选完成 330/330 个 seed 检查；45 个 same-tile original control 完成 135/135；10 个 protected incumbent 由 P2 的 30/30 检查提供证据。P4e 按 stable candidate ID/TIR hash 合并为 **165/165** 三 seed FSim 通过，P4f 又得到 **165/165** AXU5EVB build/export 通过（其中 155 个本轮新编译、10 个引用 P2）。这些证明本地正确性与可部署性，不证明板端正确性和速度。
10. **无标签 shortlist 和两类未来派发合同已冻结。** P5b 不读取 latency/correctness label，只用 lower 成功、静态 bytes/calls、small/stride/pad、reload、最大请求和直方图形成 B4--B7、预算 4/8/16；250 个 ID 精确关联，165 个 eligible、85 个过滤，所有前缀保护 incumbent。P6a 冻结 W00/W02/W09 的通用 B7/budget-4 探索合同，共 12 个候选。P6b 则另外冻结机制 canary：每个 workload 恰好选择 original、input-byte 降幅最大的已资格 input-stationary 和 weight-byte 降幅最大的已资格 mode-4，共 9 个候选；9/9 均绑定既有 FSim 与 AXU 证书，并预注册 3 correctness seeds、3 warmup、30 轮 complete-crossover、runtime counters、±2% 等价和 ≥5% 稳定改善口径。两个合同均 `board_executed=false`；B7 模式覆盖和机制静态降幅都不是性能证据，G5/G6 仍未评价，见[P6b 状态](06_prospective_board/20260911_p6b_offline_mechanism_canary_run01/STATUS.md)。
11. **FSim dry-run 已量化权重驻留的命令代价。** P4h 对 W00/W02/W09 的 original 与代表 mode-4 各做一次 seed-0 正确性和队列诊断，6/6 正确，共解析 14 条队列记录。original 均为 1 次提交；mode-4 分别为 3/5/3 次提交，新增 residency drain 为 2/4/2，与 P3e 一致。mode-4 的单次提交 instruction 峰值均低于 original，但累计 UOP bytes 从 236/208/68 增至 1624/5536/976；W00/W02 的累计 LOAD bytes 下降，W09 不变。这一结果把“少 LOAD、低单次峰值”和“多同步、多提交、更多累计 UOP”的权衡显式化，支持 command-aware 选择；它仍只是 `fsim_dry_run`，FINISH 数量来自 runtime 源码推导，replay 在日志中不可观测，不能作为板端容量、语义等价或性能证据，见[P4h 状态](04_dma_command_signatures/20260911_p4h_fsim_command_signature_run01/STATUS.md)。
12. **结构命令画像已覆盖全部 197 个本地合格候选。** P4j 合规 run02 联合 P4e 的 165 项与 P4g/P4i 的 32 项 mode-4，197 个 stable ID 无重复；197/197 经无 RPC 的本地模块直调重新 lower/build、seed-0 逐元素正确且 TIR/source 绑定一致，mode-4 的既有 drain 证据 32/32 匹配。共记录 286 次提交；mode 0--3 均为每候选 1 次，mode-4 为 2--9 次、中位数 3。全池单次 instruction 峰值中位/最大为 **2,080/23,008 B**，UOP 峰值中位/最大为 **428/10,820 B**；mode-4 累计 UOP 最大 **43,552 B**。所有 `*_us` 已丢弃，FINISH 286 次均为源码推导，replay 不可观测。这使 command-aware shortlist 在板前有完整结构特征，但仍不等于板端安全 backing 证书，见[P4j 合规状态](04_dma_command_signatures/20260911_p4j_full_pool_fsim_command_run02/STATUS.md)。首版 run01 虽未联网或访问开发板，但调用了 local-only `rpc.LocalSession()`，因违反严格 no-RPC 预注册协议而保留为无效审计轨迹，禁止下游消费。
13. **全池 command-aware shortlist 已在无标签条件下冻结。** P5c 精确联结 P4e/P4f 的 165 项和 P4g/P4i 的 32 项，仅消费 P4j 合规 run02，形成 10 workload、197 个唯一候选的统一资格池。排序只允许预注册的静态 DMA 与无 timing 命令字段，使用全目标等权 Pareto front、front 内无量纲 rank-sum 和同 front 模式分层；禁止 latency/correctness 标签及学习模型，并通过 label-poison invariance 测试。budget 4/8/16 的 10/10 workload 均以 incumbent 为首；相对 P5b，候选池新增 32 个 mode-4、删除 0 个，所有 workload 的三个前缀均发生变化。这只证明规则确实利用新增模式与命令结构改变派发顺序，尚不能证明新顺序更快或更省搜索，见[P5c 状态](05_grouped_replay/20260911_p5c_full_pool_command_shortlist_run01/STATUS.md)。

### 5.2 历史证据：可用于提出假设，但不是本轮前瞻验证

1. 自然 Top-20 实测为 **10.220–11.511 FPS**，前 10 项包含该池内实测最好候选，但池内 Spearman 只有 **0.155**；说明 shortlist 覆盖了高性能区域，却不能声称静态分数能准确预测绝对 FPS 或精确细排。
2. TopHub 单轴邻域共 **80** 次：**32** 次通过、**41** 次编译失败、**7** 次数值失败，安全替换 **0**。这只说明 TopHub 在该邻域内是局部强 incumbent，不证明全局最优。
3. P5 将 80 点按 10 workload × 8 位置离线重放，预算按所有派发位置而不是成功候选计；B2 每 workload/预算使用 1000 个随机排列。预算 4 时，包含两个无 oracle workload 的总体 within-2% 命中率为 **42.29%**。只有 B2 可作为历史随机基线；B4 使用测后 correctness，是 `diagnostic_oracle_upper_bound_leaky`；B5/B6 的 DMA 特征只存在于 32 个正确候选，状态为 `not_evaluable_missing_prefeatures`。任何 B4/B5/B6 的零 regret 都不能写成方法效果，见[P5 状态](05_grouped_replay/20260910_p5_local_historical_replay_run01/STATUS.md)。

### 5.3 待上板：决定创新点是否成立的缺口

- 当前 boot 的 bitstream、u-dma-buf、runtime、driver、runner、频率/governor 和 TopHub 尚未重新资格化；SSH 新 host key 必须由用户确认，不能绕过校验。
- 既有 165 个 lower-success 候选以及新增的 32 个显式屏障合法候选均已通过各自的多 seed FSim 和 AXU5EVB 交叉编译资格，但仍缺真实 runtime DMA 对齐、板端正确性和 latency 测量。
- P4j/P5c 已补齐 197 个本地合格候选的 FSim 结构命令画像并生成无标签派发顺序，但其 instruction/UOP 峰值、源码推导 FINISH 和不可观测 replay 尚不是板端安全 backing 证书；P5 历史 80 点的候选级板前特征仍不完整，原历史回放继续存在缺失和标签泄漏。
- input-stationary 与显式屏障 weight-residency 的静态正结果尚未证明能减少设备等待或单算子 latency；safe weight/hybrid 仍无权重收益，不能包装成性能赢家。
- 尚未在预注册 grouped holdout 上证明：达到 strongest incumbent ±2% 所需派发减半、固定预算 regret 改善，或慢候选识别改善且不误删 near-oracle。
- 尚未把被接受的单算子 overlay 传递到至少两个 stage 或一条完整流水并得到可重复收益。

## 六、开发板恢复后的最小闭环实验

最小实验应先验证“局部机制是否有真实价值”，再决定是否扩大搜索，不直接把 180 个候选全部派发。

### 第 0 步：恢复 G2 强基线

1. 用户确认当前 SSH host key 后，记录 boot ID 和板端 bitstream/ko/runtime/driver/runner 哈希；不得使用关闭 host-key 校验的方式绕过。
2. 运行 original/TopHub canary，验证输出；重放 topology B。相对历史 **10.724 FPS** 原则上不得下降超过 5%，即应不低于约 **10.188 FPS**，否则先诊断环境，不测新模式。

### 第 1 步：三个开发 workload 的机制 canary

只测已有 FSim 正证据的 W00、W02、W09，各保留 original TopHub，并各选 1 个静态 input-DMA 降幅最大的 input-stationary 候选及 1 个已经通过 P4g/P4i 的显式屏障 weight-residency。safe weight 和 paper-inspired hybrid 不作为“预期胜者”。这 9 个具体 ID、完整 ConfigEntity、证据哈希和 30 轮平衡顺序已经由 P6b 冻结，恢复开发板后不得看结果换点。

- original/input 采用交叉或随机顺序；每个候选先做多 seed 逐元素正确性，再 warmup 后至少 5 组重复计时。
- 同时采集 runtime LOAD/STORE calls、分类 bytes、small/stride/pad 和 device wait；先检查 static/runtime 描述能否对应，再解释 latency。
- 单算子 overlay 的等价/保护门槛为 strongest incumbent 的 ±2%；只有稳定达到至少 5% 才写“强性能提升”。2%–5% 只能称小幅改善或等价候选。

这一阶段最多 6 个唯一 candidate，目的是回答 input bytes 的 50%–75% 静态下降是否真正转化为请求或等待下降，而不是评价完整搜索方法。

### 第 2 步：最小无泄漏 shortlist 验证

只有第 1 步至少一个 input 候选正确且不劣于 incumbent，才继续：

1. 在读取新板端结果前，为所有待测候选补齐同版本的 lower/TIR hash、SRAM、完整静态请求签名和失败类别；缺任一板前字段的 B5/B6 不进入效果比较。
2. 使用预注册 grouped holdout W01/W04/W07/W08，保持同 workload 的所有 config 不跨 development/holdout；每 workload 先用预算 4，必要时扩到 8/16。
3. 所有方法使用同一去重测量池，TopHub B0 不可删除；`paper_inspired_hybrid` 只有通过同一正确性门槛后才能作为 B1，不能称 exact 2026 baseline。
4. 主指标是达到 strongest valid incumbent ±2% 所需的**总派发数**、墙钟、Regret@budget、near-oracle recall 和 valid-board-trial ratio。失败也占预算。

若 proposed full-signature 方法在多数 holdout 上以不超过 B2/B3 一半派发达到同等 best，且最终性能位于 strongest incumbent 2% 内，才能主张“FPGA 约束与请求形态减少调优开销”；若 2–3 个 workload 相对强基线稳定提高约 5%，才有资格进一步主张性能收益。

### 第 3 步：最小系统传递

只把第 2 步冻结接受的 overlay 写回包含相应 workload 的 stage，先做完整 stage correctness 和服务时间，再固定其他变量重放对应 pipeline。至少两个 stage 或一条流水出现可重复收益，才说明单算子改善传递到了整网；单算子 operator rate、一次单 boot 观察或静态资源下界倒数都不能替代整网 FPS 证据。

## 七、当前最稳妥的论文结论

目前可以写：固定 FPGA 上，驻留调度会同时受到片上容量、DMA 形态和 VTA 队列拓扑约束；input-stationary 已减少代表 workload 的 50%–75% 静态 input DMA；显式 full-sync 的编译器依赖分段使外层权重驻留在三个代表 workload 上正确执行，其中两项 weight bytes 下降 75%–85.71%，另一项在 bytes 不变时把 weight LOAD calls 从 32 降到 2；候选身份、去重池、Lowered-TIR 请求签名、本地资格证书和不可覆盖派发合同已经形成离线工具链。理论和实验共同表明，总 DMA bytes 不是充分指标，请求碎片、reload、同步代价、依赖及是否落在关键资源上必须联合考虑。

目前不能写：safe weight/hybrid 已有权重收益；显式屏障 weight-residency 已经加速；已精确复现 2026 方法；B5/B6 已证明优于随机/XGB；静态 DMA 下降已经带来板端 FPS 提升；单算子 22.874 GMAC/s 或系统静态排序量 14.123 等于整网性能上限。创新点最终应以“安全减少板端搜索预算”或“强 incumbent 上稳定的 stage/流水收益”至少完成一条前瞻证据链后再定稿。

## 八、RAM-only 板端推进结果（2026-09-11）

损坏且曾 100% 占满的 `/dev/mmcblk1p2` 先以只读方式绕行，随后已在当前 boot 完成备份、离线 `e2fsck`、读写挂载和写入资格检查。bitstream、`u-dma-buf.ko`、TVM/VTA runtime、RPC 和上传模块仍部署在 `/var/volatile` 的 tmpfs，以减少 SD 写入与计时干扰。当前 boot ID 为 `a68a7983-719f-47bf-94d7-41f974c5342c`，192 MiB u-dma-buf、FPGA 和 RPC 在全部实验后仍正常，4057 秒存储错误水位之后没有新 EXT4/MMC 错误。当前 boot 已修复并可实验，但冷启动资格仍待用户以后重启确认；RAM 部署重启后仍需重建。

板端证据分三步：

1. G2 RAM canary：冻结的 10 个 VTA 卷积 workload 为 **10/10** 精确正确；单次 timing 仅用于恢复诊断，不作性能结论。
2. P6c/P6d composite canary：9 个 original/input/barrier 候选完成 **27/27** 正确性和 **270/270** 平衡计时。相对各 workload 的 TopHub incumbent，五个机制候选退化，一个 W09 input 候选位于 ±2% 等价区。该结果否定的是“按 DMA 降幅挑出的整套 config 能击败 incumbent”，不能把回退归因给 residency 本身，因为 tile/thread 配置不同。
3. P6e same-tile causal ablation：5 个 mode-0 control 与 6 个 mechanism 组成同 ConfigEntity 配对，完成 **33/33** 正确性和 **360/360** 平衡样本。六组机制均在 30/30 round 中更快；input 的 W00/W02/W09 中位改善为 **6.40%/15.67%/37.96%**，barrier 的 W00/W02/W09 为 **4.33%/7.41%/60.84%**。按冻结阈值五组为 improved，W00 barrier 为 indeterminate。

因此当前最准确的故事是：**驻留能够在固定 tile 上真实减少 runtime DMA 并带来局部算子收益，但只按 DMA 选 tile 会牺牲并行度/计算效率，导致最终候选仍输给 TopHub。** 下一阶段应把 residency mode、SRAM/DMA bytes、request calls、同步数与 tile/thread/compute-utilization 联合作为硬件感知特征，并保留 incumbent；在 grouped holdout 和相同板端派发预算下验证 shortlist，而不是继续手工找单点。当前仍不能声称 generic autotune 搜索效率、跨 boot 稳定性、stage 或整网 FPS 提升。

## 九、P7Q grouped-holdout 上板结果（2026-09-11）

P7 冻结池的真实 FPGA 正确性为 79/80 候选通过、237/240 seed 检查正确。唯一失败项是 W04 `input_stationary` ConfigSpace 139；其三个 seed 和多轮复查均稳定错误，而原始哨兵保持正确。该候选没有被替换或计时，并在搜索重放中继续消耗 gross dispatch。由于原 P7 要求 80/80 通过后才能确认性计时，后续结果严格标为 P7Q post-qualification salvage，而不是回写为原确认终点。

P7Q 对 79 个正确候选完成五组随机完整区组，共 395 个真实 FPGA 候选计时样本。W01/W04/W07/W08 的冻结池 oracle 分别为 3.048571、2.772418、2.653857、5.027001 ms，四项均为原始 TopHub incumbent。B7/B8 通过 incumbent 首位保护在 budget 4 即 4/4 命中 oracle；Random 的平均 regret@4/8/16/24 为 39.43%/25.98%/9.38%/0%，ConfigEntity-only XGB 为 33.80%/33.68%/0%/0%。但是 full-request B6 没有优于 bytes-only B5，B8 也没有显示超出 B7 的增量，因而 G7 判为 NO-GO，不进入 P8 stage/FPS。

同一批标签的次分析仍给出清晰机制证据：相同 workload 和相同七个 ConfigEntity 参数的 56 组比较中，42 组驻留版本更快，中位改善 3.31%，22 组改善至少 5%。input-stationary 和 paper-inspired hybrid 的中位改善分别为 5.56% 和 6.01%；没有真实减重的旧 safe weight-stationary 中位仅 0.03%。这说明驻留机制有效，但没有与最强计算配置组合。

关键限制是候选生成器沿用了机制隔离阶段的 `oc_nthread=h_nthread=1` 强制约束，而四个 TopHub oracle 均使用 `oc_nthread=2`。因此本轮没有真正覆盖“强 tile/thread × 驻留模式”的联合候选，不能据此否定联合调优本身，也不能用同一 holdout 看过标签后反调规则并声称成功。若继续，必须新建日期隔离协议，让驻留 schedule 支持 virtual thread，并改用未见 workload/几何做确认。完整结果见 [P7Q 结果](07_grouped_holdout/20260911_p7q5_search_analysis_run01/RESULTS.md)。

## 十、P7R 虚拟线程联合开发结果（2026-09-11）

P7R 已建立新协议并仅开放 `input_stationary × oc_nthread=2`，原 TopHub 模板行为和冻结 TIR
回归保持不变。W04 完整配置空间中有 320 个满足该虚拟线程条件的配置：275 个同 tile original
已被完整 lowering 拒绝，9 个驻留后 accumulator SRAM 超限，9 个虽合法但 input DMA 不下降，
只剩 27 个进入 FSim；27/27 候选的三个固定 seed 均逐元素正确。因此硬件证书在任何板端派发前
淘汰了 293/320 个候选。

真实 FPGA 再暴露了 FSim 没发现的边界：静态排名第一的 config330 连 original 都在三个 seed
分别出现 334/336/334 个错误，因此整对永久排除且没有计时。新恢复合同只保留此前已选但未执行的
config455/461；其 original、input-stationary 和前后 TopHub 哨兵完成 18/18 seed 精确正确。

7 个确定性随机完整区组的配对计时显示：config455 的输入 DMA bytes 减半、LOAD calls 从
448 降到 224，并在 7/7 区组中加速，成对中位提升 8.93%；config461 有完全相同的聚合
bytes/calls 变化，却在 0/7 区组中获胜，成对中位下降 12.64%。前者 input descriptor 是
`x_size=4/5,y_size=28,x_stride=28` 的短行跨距请求，后者是
`x_size=28,y_size=4/5,x_stride=28` 的连续整行请求。这给出新的、可检验的硬件规则：驻留选择器
必须区分二维请求连续性、逐行/启动代价和访问--执行 overlap loss，不能只看总 bytes/calls。

两种候选的 7.693685/6.212186 ms 中位延迟仍远慢于受保护 TopHub config463 的 2.778751 ms，
所以当前仍没有新 incumbent、stage 或 FPS 收益。W04 从此停止继续计时；下一阶段只允许先冻结
request-shape-aware 规则，再在未见卷积几何上比较 Random、knob-only XGB 与证书引导策略的
派发数、regret@budget 和 near-oracle recall。完整方法和结果见
[P7R 方法](00_governance/P7R_VTHREAD_JOINT_METHOD.md)及
[P7R W04 结果](07_grouped_holdout/20260911_p7r9_w04_joint_timing_run01/RESULTS.md)。

## 十一、P7R 未见验证与硬件合法性转向（2026-09-11）

E00/E01/E02 的 2368 个 `input_stationary × oc_nthread=2` 配置经解析 lowering、三类 SRAM、
复用收益与 DMA 证书后保留 209 个（8.83%）。冻结的六个 residency 点均通过三 seed FSim，
但真实 FPGA 只有 E02 完成 15/15 正确性并进入计时。E02 request-shape 与连续对照相对同 tile
original 分别下降 14.80% 和 13.18%，因此 W04 上拟合的 request-shape 符号规则前瞻失败，
不得继续作为通用排序器。

正确性反而给出更重要的 autotune 约束：E00 config1081 的 original 在命令生成阶段触发连续 UOP
`dst_idx` 冲突；补测原/驻留双路径后可在本地 FSim 提前拒绝。另有 E00 config1061 及 E01
951/755/957/877/301 在 FSim 正确而 FPGA 错误；关闭虚线程或缩小 `tile_ci` 均未消除 E01 错误，
因此当前证据不足以发明一个静态正确性公式，必须保留真实 FPGA canary 和 TopHub 回退。

第三创新点由此修正为四级硬件感知搜索门：解析资源/DMA 证书、原/驻留双路径命令与 FSim、
每个新几何少量 FPGA 正确性 canary、受保护 TopHub 下的性能搜索。该结果支持“减少无效板端派发并
安全回退”的方法设计，但当前尚未通过 near-oracle/regret 的创新等级 B 门槛。完整记录见
[P7R 未见验证结果](00_governance/P7R_UNSEEN_VALIDATION_RESULTS.md)。
