# 共享内存相关研究、创新边界与实验决策

日期：2026-09-07。本文档服务于论文《面向嵌入式 CPU-FPGA 共享内存平台的深度神经网络流水线运行时优化研究》的创新点收敛，不把尚未完成的实验写成结论。

> **决策更新（2026-09-08）：本文件保留为文献与候选路线档案，不再作为执行计划。** 下文 `T3-A/C3-A`“APM 标定后反馈 shortlist”的主候选、以 APM 失败自动触发 T3-B 的分支及其 F0--F3 门槛均已归档；当前第三点改为“共享内存路径自争用感知的 stage 准入与选择性重叠”，APM 只是可选离线物理标签，C3-B 只在独立容量 gate 通过时重新立项。唯一当前执行口径见 [`ONE_MONTH_EXECUTION_PLAN.md`](ONE_MONTH_EXECUTION_PLAN.md) 的 C3-G0--G3；引用本文件时只使用相关工作、平台事实和被否决方案，不沿用旧推荐。

## 1. 结论先行

第一创新点可以而且应该与共享内存紧密关联，但不能只是在原成本模型中增加一个 `boundary_bytes` 权重。建议最终表述为：

> **面向共享 DDR CPU--VTA 流水的编译上下文等价类驱动分层图划分与调度搜索。** 方法从切图后的 fusion/layout、共享边可实现性和 incumbent tile 的 lowered TIR 中，提取框架物化、边界 live tensor、DMA 请求数/粒度/重复载入、VTA island 串行需求及 CPU/VTA 并发访存窗口；这些信息用于候选去重、支配剪枝、资源下界排序，并只对编译上下文发生变化的 workload 做有界局部重调。

这个表述的关键不是“本文首先发现共享 DDR 会竞争”，而是把三层过去割裂的信息接起来：

```text
Relay 切图与融合上下文
        ↓
VTA tile/lowering 后的 DDR↔SRAM 逻辑请求
        ↓
跨 Executor 边界的物化或同址共享实现
        ↓
单 VTA、共享 CPU 核心和共享 DDR 下的流水搜索
```

**历史决策记录：**2026-09-07 曾把“编译期逻辑 DMA 与 PS DDR_APM 对齐的共享 DDR 请求形态标定与反馈优化”作为条件第三点，并把逐边 handoff/可变槽规划作为 APM 不可用时的备选。2026-09-08 复核后已废止这套主次与自动 fallback 关系；下文 T3-A/T3-B 仅保留为被评估过的历史方案，不能作为当前任务或论文贡献口径。

## 2. “共享内存”至少包含六层状态

CPU 和 FPGA 能访问同一片 DDR，只回答了物理可达性，不能自动推出零复制、cache 一致、无 DDR 搬运或无竞争。本文需要分开记录：

| 层次 | 本平台中的问题 | 当前可观察量 |
|---|---|---|
| 地址可达性 | CPU view 与 `ext_dev` view 是否指向同一 u-dma-buf 物理范围 | 虚拟/物理地址、range、对齐 |
| 一致性与所有权 | HPC/CCI、一致性属性、producer/consumer 完成顺序 | 端口配置、owner FSM、generation、必要的 sync |
| 框架物化 | 相邻 GraphExecutor 是否先 `get_output` 再 `set_input` 复制 | copy calls、payload、API service time |
| 图边转换 | dtype/layout/quantization/contiguity 是否允许直接别名 | Graph JSON、Relay/TIR、shape/dtype/layout、adapter |
| VTA 片上搬运 | tile 如何使输入/权重从 DDR 重复 LOAD、输出如何 STORE | lowered-TIR 和 runtime 逻辑 DMA calls/payload |
| 共享 DDR 服务 | CPU 与 VTA 的请求何时重叠、AXI burst/latency 如何变化 | 需要 PS DDR_APM 或等价硬件 counter 校验 |

因此，本文应避免把“边界 tensor 大小”“VTA runtime DMA payload”和“DDR 控制器物理流量”混为一个数字。

## 3. 相关工作的覆盖范围

| 工作 | 已覆盖内容 | 对本文创新边界的约束 |
|---|---|---|
| HaX-CoNN, PPoPP 2024 | 逐层异构映射、设备 transition、共享内存 contention、并发 DNN 调度 | 不能泛称“首个共享内存争用感知 DNN 切图”；本文需限定 CPU--VTA、编译 lowering、tile-sensitive DMA 和单 VTA 复用 |
| AxoNN, DAC 2022 | shared-memory SoC 上的多 DSA layer mapping 与 transition cost | “切图不能只线性相加算子时间”不是单独的新意 |
| CoDL, MobiSys 2022 | 统一内存 CPU--GPU 协同中的数据转换、mapping、同步和共享开销 | 共享物理内存不等于交接免费；本文区别在跨 stage/Executor 和跨帧流水 |
| Unexpected Diversity, FPT 2019 | ZynqMP AXI 端口、burst、访问模式、QoS、并发 master 的定量影响 | 支持请求数、平均 payload 和小请求比例；也说明仅用总 bytes 不足 |
| PCCS, MICRO 2021 | 异构 SoC memory-interference/slowdown 的通用校准模型 | 不能用四个 topology 包装成新的通用 contention model |
| LAG-Guided Runtime Framework, TODAES 2026 | 多 DNN 嵌入式 GPU 上的 block conflict table、动态执行顺序和选择性串行化 | conflict-table 与“有冲突就延后 block”本身不是新意；当前 C3 只能强调同一 DNN 跨帧 CPU--VTA、TVM 编译内存签名、单 VTA/K2 约束与前瞻验证的组合差异 |
| SVM/IOMMU 与 Zynq CNN unified memory 工作 | 连续内存、mmap、地址转换、一致性和 zero-copy | 共享虚拟地址与一般 zero-copy 不是本文的新意 |
| FPL 2019 I/O coherency study | HP、显式 cache sync、HPC、ACP 的策略比较 | coherence policy selection 本身已有系统研究；本文固定 HPC 并研究其上的图/运行时语义 |
| DNNFusion、ShortcutFusion、LCMM | 算子融合、片上数据/shortcut 复用、tensor lifetime 与 weight prefetch | 跨算子 SRAM 常驻和一般预取不是低风险空白，而且通常需要改编译器/硬件存储规划 |
| RIMMS, TECS 2025 | 动态 task-to-PE 下的位置/一致性跟踪、内存复用和冗余传输消除 | owner/location tracking 概念已有；本文只能强调固定 TVM 拓扑、跨 Executor、多 tensor 原子发布和 frame generation |

本文不应使用“首个”来概括其中任一单项。更可靠的 novelty 是这些约束在当前 TVM/VTA 跨帧流水中的具体交集，以及完整的编译、搜索、运行时与板端证据链。

## 4. 第一创新点如何真正成为共享内存感知搜索

### 4.1 两类签名

每条异构边建立：

```text
SharedEdgeSignature(e) = {
  producer_device, consumer_device,
  shape, dtype, layout, alignment, contiguous,
  physical_reachability, coherence_mode,
  adapter_or_requantize,
  zero_copy_eligibility,
  materialized_copy_bytes, copy_service,
  slot_count, generation, live_interval,
  adjacent_fusion_context
}
```

每个已确定编译上下文和 tile 的 VTA segment 建立：

```text
VtaMemorySignature(s,c) = {
  load_calls, load_bytes, store_calls, store_bytes,
  small_load_ratio, average_load_payload,
  input_reload_ratio, weight_reload_ratio,
  SRAM_working_set_legality,
  island_reentry, serialized_VTA_service
}
```

其中 `c` 不是无条件只按卷积 shape 定义，而先使用：

```text
compile_context = workload
                + fused-op signature
                + boundary layout
                + quantization state
```

只有完整 H5/H6 审计表明后面三项不改变 TIR 或 tile 排名，才把等价类退化为单纯 workload。这正是“切图与 AutoTVM 的关系”中尚缺的证据。

### 4.2 分层搜索而不是笛卡尔积

推荐算法流程为：

1. 外层枚举 compiler-valid 的连续 CPU/VTA segment、VTA island 和各 CPU stage 线程数；
2. 编译期生成 edge/fusion/context 签名，先排除 layout、量化、连续性、地址可达或 slot 容量不合法的方案；
3. 对 context 等价类查询 TopHub/history incumbent，并从 lowered TIR 提取逻辑 DMA；相同签名只查一次；
4. 只有出现新的或性能敏感的 context，才触发小型局部 tile 重放；
5. 将 stage service、shared edge、单 VTA 串行、CPU 核心池和共享 DDR 下界回填 k-best DP；
6. 对 dominated 的切点或 tile 组合做安全剪枝，最后只把 topology-diverse shortlist 上板。

这解决的是 AutoTVM 不负责的外层问题。AutoTVM 在给定 TE workload/schedule template 后搜索 `tile_b/h/w/ci/co` 与 virtual-thread 参数；它不选择整网 CPU/VTA stage、不决定跨 GraphExecutor 是否物化，也不管理跨帧 slot 所有权。第一创新点不需要“打败 AutoTVM”，而是决定在哪个编译上下文复用它、何时局部重调，以及结果如何反向改变全局切图。

### 4.3 成本所有权必须唯一

同一个数据活动不能在多个项里重复收费：

- GraphExecutor 的 memcpy 属于 `SharedEdgeSignature`；
- VTA 从共享 DDR 到片上 SRAM 的 LOAD/STORE 属于 `VtaMemorySignature`；
- dtype/layout adapter 和 cache maintenance 属于边界 host work；
- stage 实测 service time 若已经包含内部 DMA，就不能再把 `DMA_bytes/bandwidth` 无条件相加；
- PS APM 的 controller bytes 用于验证/校准，不能与软件 payload 直接相加。

流水下界继续使用资源最大值而不是总和：

```math
T(P)=\max(T_{CPU-pool},T_{single-VTA},T_{shared-DDR},T_{longest-stage})+T_{edge}.
```

## 5. 现有证据能说明什么

| 已有证据 | 可以说明 | 不能说明 |
|---|---|---|
| 10 个 workload 跨 39 个 placement 命中同一 TopHub config | dispatch/config 可按 workload 去重查询 | 融合/TIR 相同、该 tile 在所有完整 context 中都最优 |
| 单 workload TIR 与 runtime DMA 对齐 | tile 确定后，逻辑请求数、payload 和 reload 可在编译期获得 | AXI burst、物理 DDR bytes、DDR latency、compute stall |
| `03..15/16/17` endpoint delta | 相邻切点可在上板前比较内部 VTA DMA 与边界 bytes 的权衡 | 该向量必然准确预测 FPS |
| Experiment C | block-aligned 分裂没有改变该例 VTA DMA；共享 slot 消除一次框架 copy | 所有切点都不影响融合/量化/TIR |
| M0/M1/M2 全空间消融 | M1 的边界/资源约束在四个实测代表上纠正 Top-1；M2 可用于 delta/剪枝 | M2 提升了完整搜索排名；当前 DDR bound 曾成为瓶颈 |
| 双 slot B2 | 框架边界物化为 0，交接协议在多帧下正确 | 峰值内存下降，或自然 Top-20 FPS 必然提高 |

尤其要保留两个负结果：M1/M2 Top-20 完全相同，且 972528 个配置中 `T_shared-DDR` 从未成为最大下界。这意味着 tile-DMA 当前更适合做机制解释、endpoint delta、退化排除和支配剪枝，不能包装成已经改善吞吐预测的自由 DDR 模型。

## 6. 老师提出的四个问题应如何回答

### 6.1 有没有冗余拷贝

现有 runner counter 已能精确识别跨 Executor 的 framework copy，B2 已把该层从 `1,806,336 B/frame` 降为 0。后续 E6 应对 4623 个 topology 静态列出每条边的物化、shared-slot 资格和 pinned slot 容量。

但 runtime copy counter 不能自动发现 DDR controller 内部的全部重复事务；VTA 的重复输入/权重 LOAD 由 lowered TIR 另行统计，物理 AXI 层再由 E7 的 APM 校验。

### 6.2 有没有不合适的频繁回落

需要先明确“回落”是哪一层：

- CPU/VTA 设备回落或 island 重入：由切图 topology 直接知道；
- DDR↔VTA SRAM 的重复 LOAD：tile 确定后从 TIR 得到 `R_input/R_weight`；
- 融合被切断后中间值 STORE 到 DDR 再 LOAD：必须通过 E1/E2 比较完整 segment 的 fusion/TIR/DMA，不能只看 workload key；
- GraphExecutor 内部 pool 与边界 slot 重复占用：由 Graph JSON 和 driver high-water 验证。

### 6.3 能否做预取和预驱逐

VTA 本身用显式 LOAD/compute/STORE task、依赖队列和 virtual thread 组织访问—执行重叠；当前 VTA TOPI schedule 固定 cache/DMA/tensorize 结构，AutoTVM 的 tile/virtual-thread 只会间接改变请求形态和潜在重叠。它不搜索“让下一个独立算子的张量跨 Executor 预先驻留 SRAM”。

真正的跨算子预取、预驱逐或 SRAM residency 需要联合修改 FuseOps、TE/TIR schedule、片上 buffer lifetime、依赖 token，必要时还要改硬件计数/控制；DNNFusion、ShortcutFusion 和 LCMM 等已有直接先例。以一个月周期看，这一方向不应实现，只作为未来工作。更实际的“软件预取”是提前获得一个空 shared slot 或 adapter direct-write，但它优化 DDR 中的跨 Executor 交接，不是 VTA SRAM 预取，命名时必须区分。

### 6.4 compute 等待传输多久

当前 `driver_poll_wait_us` 是 host 等整次 VTA command 完成，不等于 compute pipeline 因 LOAD/STORE 阻塞的时间。软件 profile 能给请求数量和 payload，PS APM 能给 AXI bytes/transactions/latency，但两者都不能单独分解出 compute-stall cycle。没有 VTA 内部 stall counter 时，论文只能报告：

- 完整 stage/device service time；
- LOAD/STORE 逻辑请求与 payload；
- APM 可行时的 AXI transaction/bytes/latency；
- 通过受控 tile/context 或 CPU pressure 的 matched experiment 推断性能敏感性。

“compute 等待 LOAD 为 X ms”必须留到增加硬件 stall counter 后才能直接声称。

## 7. 历史 T3-A/T3-B 实验方案（已归档，不执行）

> 本节以下编号、优先级和“转向”规则只记录 2026-09-07 的旧方案；当前 G0--G3、E6-S/E6-R、E7/E8 规则只以 `ONE_MONTH_EXECUTION_PLAN.md` 为准。

1. **E0--E2：融合/TIR/context compile-only 审计。** 这是证明 stage 与 AutoTVM 关系的必要证据，不能跳过。
2. **E6：全 4623 topology shared-edge/slot realization audit。** 不需要板卡，直接把第一点与共享内存执行语义接起来。
3. **T3-A0：两天 PS DDR_APM feasibility gate。** 先不改 bitstream；验证安全访问、counter save/restore、DDR slot 归因，再做 idle、CPU-only、VTA-only、CPU+VTA。能得到稳定 matched delta 则继续，不能归因就记录负结果并转向 T3-B 或保持两主创新。
4. **T3-A1--A3：请求形态×CPU 压力校准、分组 holdout 和 shortlist 反馈。** 只在 T3-A0 通过后执行；标定最小版 8--11 个工作日，含三 boot 端到端反馈约 12--16 个工作日，在线节流只是加分项。
5. **E8：driver allocation high-water 与流水 K=1/K=2。** 验证实际 pool，并把“双槽是流水所需”与“串行模式”分离；也是 T3-B 的先验 gate。
6. **E3--E5/E9：只由前述结果触发的局部因果实验。** 不恢复全空间 AutoTune，也不把通用 DAG/arena 作为毕业门槛。

PS APM 工具本身不是创新；能够作为增量第三点的是“编译特征—物理计数对齐—校准模型—搜索反馈”的完整方法链。其直接价值是填补“runtime 逻辑 DMA ≠ physical AXI traffic”的缺口。

## 8. 历史第三创新点候选评估（已归档）

> 表中“主候选/备选”是旧决策，不表示当前推荐；当前主候选已经改为共享内存路径自争用感知的 stage 启动门控。

| 候选 | 新意风险 | 一个月工作量 | 当前证据 | 决策 |
|---|---:|---:|---|---|
| 编译—APM 对齐的请求形态×压力校准与搜索反馈 | 中；单项技术已有，但 CPU--VTA/TIR/APM/shortlist 闭环有场景差异 | 中；标定 8--11 日，完整反馈 12--16 日 | 已有 TIR extractor/runtime profile/P7C；缺 APM collector 和 holdout | **T3-A 主候选** |
| GraphExecutor pool 复用与逐边 handoff/`K_e` 规划 | 中；arena、lifetime、zero-copy 各自已有，差异在跨 Executor 线性流水与单 VTA 约束 | 中高，14--20 日 | runner 有固定 external 双槽/FSM；缺 mixed policy、pool-anchor 审计和 planner | **T3-B 备选** |
| 在线 DDR contention-aware 节流 | 高；HaX-CoNN/PCCS 已有 | 高 | 当前自然 workload 的争用收益弱、噪声大 | 只作 T3-A 可选增强 |
| cache/coherence policy 自动选择 | 高；已有 FPL 2019 系统研究 | 中高 | 主实验固定 HPC，flush/invalidate 已接近零 | 不做 |
| 跨算子 SRAM 预取/预驱逐/常驻 | 高；融合和 FPGA memory planning 已有 | 很高 | 需要改 compiler/lowering/可能改硬件 | 不做 |
| 动态 data-location manager | 高；RIMMS 等已有 | 很高 | 本文映射静态，通用动态机制没有必要 | 不做 |
| adapter direct-write 到 consumer slot | 中 | 中 | 尚未证明现有边界有显著 adapter 物化 | 只在 E1 找到真实转换热点时作为第二点增强 |

### T3-A 成立门槛

1. 两天内完成 PS DDR_APM 安全访问和 DDR port/slot 归因；matched delta 在 VTA-only、CPU-only 与 concurrent 三类运行中可重复，目标 CV `<=5%`、受限时 `<=10%`，且 CPU/VTA 增量显著高于 idle；
2. 至少 6--10 个正确 tile/config、3 对近似等 bytes 不同请求形态、3 档实测 CPU pressure，最终子集跨 3 boot；
3. 按 workload/context 分组 holdout，对比 bytes-only、logical-shape 和 APM-calibrated 三层模型；
4. 若请求形态与 APM `bytes/transaction` 或 `latency/transaction` 的关系可复现，且 grouped holdout 相对 bytes-only 的误差明显下降，则可称为“共享 DDR 物理标定方法与规律”，并入第一点；
5. 只有 held-out 场景至少改变一个 shortlist/thread/overlap 决策，并取得 `>=2%` FPS、`>=5%` P95 II 改善或 `regret@1 <=2%`，才采用独立第三点的强名称“反馈优化”。

冻结的最小矩阵为：F0 六类 APM gate（idle、CPU streaming、两类 VTA、两类并发）；F1 `2 shape × 3 tile × 3 CPU pressure=18` cells；F2 workload/context grouped holdout；F3 topology B/D 的自然与受控干扰端到端。F0 每类 5 warmup + 20 计分，F1 每 cell 5 warmup + 30 计分，最终 F3 为 62 frames × 3 boot，并使用 ABBA 或随机交错次序。

### T3-B 成立门槛

1. runner 支持逐 edge `copy/shared`、至少 `K=1/2` 及 `pool-anchor/external` 来源，并能生成静态 offset/alias/release 计划；
2. 范围明确限制为当前线性 CPU--VTA stage 链；至少两个 topology 通过慢消费者、多 tensor 原子交接和 generation reorder correctness，不把尚未实现的通用 fan-out/DAG 写入贡献；
3. 对比 all-copy、all-shared-fixed-K、现有固定双槽三个强基线；
4. 跨三个独立 boot 得到稳定 II/FPS 收益，或相对固定 external-K2 在 II/FPS 的 95% 置信上界退化不超过 2% 时至少降低 20% slot bytes/high-water；小空间实测 oracle 上 planner regret `<3%`；
5. 论文明确把新意限定在 TVM 跨 Executor、跨帧线性流水和编译生成的逐边 slot，而非一般 buffer reuse。

当前四个冻结 package 的 Graph JSON 静态初算为：

| Topology | VTA Graph storage pool + 双 slot | 占 192 MiB |
|---|---:|---:|
| A | 10,845,184 B | 5.39% |
| B | 11,088,384 B | 5.51% |
| C | 11,020,800 B | 5.47% |
| D | 15,490,048 B | 7.69% |

这些数字尚需固化脚本和 driver high-water 复核，但已经说明当前 ResNet18 很可能没有容量压力。若 T3-A0 失败且 K=1/2 也不产生可用 memory--II Pareto，第三点就停止，保留两个已经扎实的主要创新。

## 9. 一个重要的实现口径修正

当前 AXU5EVB driver 在 `3rdparty/vta-hw/src/axu5evb/axu5evb_driver.cc:285-388` 中使用 256 B 对齐的 bump allocator，`Free` 明确为 no-op。`src/runtime/graph_executor/graph_executor.cc:609-712` 在 `SetupStorage` 时分配 storage pool，而 `SetInputZeroCopy/SetOutputZeroCopy` 只更新算子 DLTensor 的 data pointer。

因此：

- 可以说 B2 消除了 steady-state 每帧 framework materialization；
- 可以报告额外双 slot 的静态 bytes；
- 不能说重绑后 GraphExecutor 原 pool 已释放；
- 在 driver high-water 实测前不能说 B2 降低了峰值 u-dma-buf 占用。

这一负面实现事实不是失败，反而说明搜索中的内存容量必须按真实 allocator/storage-pool 语义计算，而不能只按 live tensor 理想值估计。

它同时给 T3-B 留出一个具体、可测的微调点：通过 storage-id 独占性、物理范围和生命周期审计后，可将既有 GraphExecutor 边界 pool 借为 shared edge 的 `slot0`，K=2 时只新增一个 external slot。当前 rank-1 静态初算可把额外 slot 从 `1,806,336 B` 降至 `1,003,520 B`（`-44.44%`）；三 VTA-island 拓扑可从 `9,031,680 B` 降至 `7,024,640 B`（`-22.22%`）。VTA input storage-id 当前看起来适合借用，output storage-id 有内部复用风险；正式结论必须由 Graph JSON 审计、range check、generation 压力测试和 driver high-water 一起确认。

## 10. 历史推荐结构（已废止）

> 下列 T3-A 结构只为保留决策演进，不得复制到当前论文或答辩；最新贡献组合见 `ONE_MONTH_EXECUTION_PLAN.md`。

1. **共享内存实现与编译上下文感知的 CPU--VTA 流水线图划分。** 负责回答切在哪里、CPU stage 用几线程、需要几个 VTA island、哪些 context 可复用 incumbent、每条边是否可同址共享，以及逻辑 DMA/边界/单 VTA/CPU pool 如何进入搜索和剪枝。
2. **固定拓扑下跨 GraphExecutor 的双 slot 零拷贝运行时。** 负责把规划落实为 u-dma-buf 双视图、所有权状态机、frame generation、多 tensor 原子交接和完成后释放，保证多帧流水正确。
3. **编译—硬件计数器对齐的共享 DDR 请求形态标定与反馈优化（T3-A，条件项）。** 负责把 TIR/runtime 逻辑 DMA 对齐到 PS DDR_APM 物理事务和 CPU 并发压力，用分组 holdout 验证轻量模型，再反馈 shortlist、线程或 overlap 策略。T3-A0 失败时不保留空标题，可改做通过 gate 的 T3-B，或正式采用两个创新点。

一句话关系是：

> 第一项生成候选和共享内存执行计划；第二项安全执行共享边；条件第三项把板级物理 DDR 反馈给第一项做二次校准。

## 11. 一手资料

- Moreau et al., *A Hardware--Software Blueprint for Flexible Deep Learning Specialization*: <https://arxiv.org/abs/1807.04188>
- Dagli and Belviranli, *HaX-CoNN*: <https://doi.org/10.1145/3627535.3638502>；开放稿 <https://arxiv.org/abs/2308.05869>
- Dagli et al., *AxoNN*: <https://doi.org/10.1145/3489517.3530572>
- Jia et al., *CoDL*: <https://chrisplus.me/assets/pdf/mobisys22-CoDL.pdf>
- Manev et al., *Unexpected Diversity*: <https://research.manchester.ac.uk/en/publications/unexpected-diversity-quantitative-memory-analysis-for-zynq-ultras/>
- AMD Zynq UltraScale+ PS--PL interface, PG201: <https://docs.amd.com/r/en-US/pg201-zynq-ultrascale-plus-processing-system/Slave-Interface>
- AMD Zynq UltraScale+ TRM hardened PS APM, UG1085: <https://docs.amd.com/v/u/en-US/ug1085-zynq-ultrascale-trm>
- AMD System Performance Analysis, UG1145: <https://docs.amd.com/r/2024.2-English/ug1145-sdk-system-performance/Evaluating-High-Performance-Ports>
- AMD AXI Performance Monitor, PG037: <https://docs.amd.com/v/u/en-US/pg037_axi_perf_mon>
- Vogel et al., *Exploring Shared Virtual Memory for FPGA Accelerators with a Configurable IOMMU*: <https://doi.org/10.1109/TC.2018.2879080>
- Min et al., *Analysis and Optimization of I/O Cache Coherency Strategies for SoC-FPGA Device*: <https://sitaohuang.com/publications/2019_fpl_io_cache_coherence.pdf>
- Wei et al., *Layer Conscious Memory Management*: <https://vast.cs.ucla.edu/sites/default/files/publications/_DAC_2019___Final_Version__LCMM.pdf>
- Niu et al., *DNNFusion*: <https://arxiv.org/abs/2108.13342>
- Nguyen et al., *ShortcutFusion*: <https://doi.org/10.1109/TCSI.2022.3153288>
- Gener et al., *RIMMS*: <https://doi.org/10.1145/3760257>
- Kim et al., *LAG-Guided Runtime Framework: Block-Level Scheduling and Dynamic Compression for Multi-DNN Environments*: <https://doi.org/10.1145/3798107>
- Linux dma-buf documentation: <https://www.kernel.org/doc/html/latest/driver-api/dma-buf.html>
