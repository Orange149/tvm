# 历史计划归档（2026-09-08，非当前执行指令）

旧计划与实验状态原样保留；唯一当前执行入口为
[ONE_MONTH_EXECUTION_PLAN.md](ONE_MONTH_EXECUTION_PLAN.md)。

# Stage–tile 与论文一个月收敛计划

实验冻结基准日：2026-09-07；计划修订日：2026-09-08。目标不是展开完整切图×AutoTVM 笛卡尔积，而是在已有两项工作上形成可复现、可答辩的最小闭环。

## 当前执行优先级：C1 收尾，资源转向第三点（按用户优先级调整）

本节及 [C1 收尾与 C3 聚焦执行单](C1_CLOSEOUT_C3_FOCUS.md) 优先于下文旧阶段的扩展任务、预算和候选优先级；已完成实验和未完成标记原样保留，不把“停止扩展”改写成“全部完成”。本轮只调整计划，未启动新板端实验。

- **C1 转入收尾。** 主线是资源与边界代价感知的 CPU/VTA 划分、island 和 CPU 线程搜索；不再以 stage–tile 反复迭代为主创新。FuseOps、量化、packing 和 AutoTVM 属于复用的编译基础，DMA/编译上下文审计属于工具可信性与适用范围证据。停止扩展 E4/E5b、新 tile 搜索、全域 E6-R build 及为增加工作量而继续堆叠局部对照；未完成部分缩小论文主张，不虚报完成。必要的数值政策、反例回归与 score/运行模式一致性不能取消。
- **C2 保持既有机制和协议。** 只补论文承诺范围内的正确性/证据缺口；不把第三点实验混算为原零拷贝收益。
- **旧 C3-G0 门控路线暂缓。** 57 个复现条件未发现等待收益，原成立门槛仍未通过；不进入 G1/G2/G3、不反复调整 offset 寻找正例。这是资源止损，不是证明所有争用门控都无效。
- **第三点优先评估 C3-S：编译存储别名感知的跨 Executor 共享缓冲复用。** 这是从旧 C3-B 中抽取的窄范围候选：固定 topology/tile/线程、all-shared K2 和所有权协议，只决定每条边可否安全借用既有 GraphExecutor pool 作为 slot0，否则保持全外部分配。不启动旧 C3-B 的 mixed mode、K1、通用 arena 或关键路径 planner。
- **先做 1–2 个工作日的可行性检查。** 冻结 A/B/C/D；固化逐 entry storage-id/alias 审计及 graph-load/slot-allocation high-water 基线。至少发现一个安全可借用类别且预测分配减少可落到驱动计数，才进入实现；无真实节省或生命周期不安全则停止。该工期是预算，不是收益或完成保证。C3-S 尚未成立；只加一条绑定捷径不足以独立成第三贡献。

后续顺序：C1 最小收尾清单 → C3-S 静态别名与实际分配基线 → 条件式 K2 pool-anchor 实现 → 固定二进制的容量/正确性/性能保护对照。最后 4–5 个完整工作日仍留给论文与证据冻结。

## 历史实验进度与原分支协议

以下保留实验时序、原任务门槛与备选路线细节供追溯。与当前优先级冲突的未执行支路不再自动排入日程；其复用或重启需按新的聚焦执行单判断。

**最新 C1 进度（2026-09-08，E2 上板完成）：** E0、E1 两级编译审计及 E2 已完成。23/23 冻结代表全部上板，与静态 census 相同的 Graph/TIR 下，69 组输入、207 次采样、3,726 项 LOAD/STORE/ACC/host ALU/GEMM 字段比较、27,998,208 个输出元素比较均无差异。boot 仍为 `a5220e22`，未更换硬件/runtime；RPC 已关闭。见 [E2 上板报告](compile_context_audit/E2_ON_BOARD_QUALIFICATION.md)。现在可称这 23 段的精确静态逻辑 DMA 已经运行时资格化，不能外推为 87 段全部上板或物理 DDR/stall 预测。

编译阶段原结论保持：87 段的 693 次卷积映射到 19 个语义卷积，23 个代表覆盖 72 类 primitive；共同卷积 TIR 未变，完整 LOAD 残差由 fused ACC 解释。layer2/3/4 的 `main+projection → +tail` 保持 VTA DMA，而两个 float32 输出变成一个；stage 内逻辑 CPU 访问减少 196/98/49 KiB。另已建立 6 条完整 CPU→VTA→CPU 路径的边界 ledger，明确 tail 从 CPU suffix 移到 VTA 内部 CPU helper；这只是 contract 对账，不是六条流水性能/量化等价。**本轮没有新 AutoTVM 或流水 FPS，不声称已改善排序/吞吐。**

**同日 E6-S 完成：** 全部 4,623 topology / 972,528 含线程配置、25,410 次边界出现完成 contract 级审计，得到 15 类 contract、739 个保守 edge-context key；后者不是最小融合等价类。两次全域遍历结果相同，M1 Top-20 重算不变，M0/M1/M2 各 20 个旧候选已对齐。所有理论 K2 slot 恰为 payload 两倍；静态 Pareto 的 4 个点不是安全剪枝结论。23 段使用直接 TIR/上板证据，另 64 段仅由 primitive 类组合，物理绑定资格化仍属 E6-R。见 [E6-S 报告](shared_edge_audit/E6_STATIC_DOMAIN_AUDIT.md)。

**同日 E3 数值 pilot：** 复用 E2 的三组冻结输入，六份 LLVM stage reference 做三对 tail 内/外对照，526,848 个元素中有 43 处不等价（layer3 normal 1 处、layer4 normal 42 处）。Relay 终端 `int32 add/ReLU → int8 cast` 无 clip，回绕预测精确解释全部差异。E2 的逐 stage 自参考正确性仍成立，但不能充当跨切图等价证明。下一步先冻结共同量化/数值参考，再做 E3 四臂性能对照及 E6-R；暂不改量化规则、重调 tile 或启用剪枝。见 [E3 pilot 报告](compile_context_audit/E3_TAIL_QUANTIZATION_PILOT.md)。完整 E3、E5a、E6-R 仍未完成。

**同日 E3 后续（主机一次量化资格化完成）：** 从三份冻结的局部 joined 量化图派生 A/B、双 Executor int8 shared/copy、精确 float32 shared/copy 六臂，18 个 LLVM build、2,634,240 个元素比较全部一致；保留原回绕，**不是修复了 43 个反例，也不是整网一次量化**。三个 int8 切点本就有 stop_fusion，A/B 及原生切分前后 primitive 多重集合不变。另做 3 个主机去屏障反事实 build，primitive 均 6→5，526,848 个元素比较无差异，说明融合机会确实受屏障位置约束，但尚无 VTA 合法性/性能证据。抽出的 float32 producer 与旧独立量化 producer body 三组均结构相同。下一步优先复用旧 VTA producer，资格化保留共同参考语义的板端 CPU tail/shared/copy；补真实激活和 block-aligned 对照。见 [新报告](compile_context_audit/E3_ONCE_QUANTIZED_HOST_QUALIFICATION.md)。本轮未上板、未改正式量化政策/TopHub，无新 FPS，完整 E3 仍不勾。

**同日 E3 板端后续完成：** 六个冻结 VTA binary/params 原样复用，只新增三个共同参考 CPU tail；mono/shared/copy 三臂共 81 次正式 VTA 运行、18 次 warmup，最终输出 4,741,632 个元素和 producer 输出 6,322,176 个元素比较全部一致。三臂逻辑 LOAD/STORE 请求数与 payload 相同，CPU tail 无新增 VTA DMA；shared 的边界 API copy 为 0，copy 每次分别为 802,816/401,408/200,704 B，不称物理 DDR 流量。六个共享 tensor 范围通过 256 B 对齐、池内、不重叠及稳定性检查。独立 RPC 适配/只读查询模块未替换冻结 runtime 或 bitstream，RPC 已关闭。见 [板端报告](compile_context_audit/E3_VTA_CPU_TAIL_QUALIFICATION.md)。仍保留 int8 回绕；本次是单帧资格化，不是 K2 多帧安全或 FPS 实验。下一步补真实流水激活、block-aligned 对照及最终数值政策，再测限定路径无 profiler 的配对 service；不能复用旧无窄化 CPU tail 的时间作为新成本。

**最新执行进度（2026-09-08，boot `a5220e22`）：** 已按重启后环境恢复 192 MiB u-dma-buf 与冻结 HPC bitstream，完成 A/B/C/D 旧 runner 基线及新增被动 readiness 时间戳的上板重放，共 3296 帧输出校验通过。shared K2 的 302 帧运行实测 FPS 为 `10.944/10.160/10.857/10.599`；自然 CPU--VTA graph-run 重叠并集占墙钟约 `76.9%/71.2%/76.5%/76.3%`。已观察到可达方向及单 VTA owner 限制，允许继续 G0 受控 pair discovery；**没有测出 protect/allow 标签或可恢复收益，完整 G0 和第三创新点仍未成立**。详见 [G0 readiness 实验报告](g0_discovery/EXPERIMENT_G0_READINESS.md)。本轮不替代 C1 的 E0/E1，也没有重新进行 AutoTVM。

**同日后续进度：** 已实现 binding 后的 native 双 worker 配对 hook，并用 A 的 CPU0/VTA1 两个方向完成 100 个 pair 样本（每方向 5 次/策略 warmup＋10 个 ABBA block），输出与同步时序通过。两个方向 allow/wait makespan 中位分别为 `93.943/120.757 ms`、`70.887/120.570 ms`，本次 wait 均更慢；这说明单 stage slowdown 不能代替 pair makespan，也不代表全部 cell 没有保护收益。状态为 **harness qualification 完成、完整 G0 discovery/confirmation 未完成**；动作未分类，不据此实现 G2。详见 [G0 pair 资格化报告](g0_discovery/EXPERIMENT_G0_PAIR_QUALIFICATION.md)。

**扩展 discovery（本轮完成）：** A/B/C/D 全部 owner-eligible 自然方向枚举已完成：22 个方向、60 个时龄条件、3000 个 pair 样本、6000 次被测 stage invocation，输出与协议审计通过。57 个条件成功复现 active/overlap，均在 10/10 ABBA block 中 wait 更慢，未发现保护候选；D CPU0→VTA3 的三个时龄条件未复现，不能作为负结果；另有 2 个方向在原 trace 中未见。详见 [完整普查报告](g0_discovery/EXPERIMENT_G0_NATURAL_SWEEP.md)及 [预注册](g0_discovery/sweep_a5220e22_natural60/preregistered.json)。这是自然可达域的配对普查，不替代 isolated 内存强度 3×3、compute-only 负对照或五 boot confirmation；正式标签和 opportunity ceiling 保持未知。**当前不进入 G1/G2/G3，优先收口 C1/C2；完整 G0 未通过，也未被普遍否证。** 若重启 C3，须先复现失败状态或事前提出新的机制对照，不以事后调整 offset 寻找正例。

**实验状态（2026-09-07）：原定内存实验已经完成，最终冻结前新增编译耦合收敛和第三创新点机制 gate。** H1--H4、单 workload 静态 DMA 提取、相邻切点 delta、三个独立 boot、完整 972528 配置的 M0/M1/M2 消融均已有产物；但现有 H1 只证明同一卷积 workload 命中相同 TopHub config，尚未完成冻结域的 FuseOps/lowered-TIR 编译签名审计，也没有验证变化 context 中已测 tile 邻域的排名。最终冻结前必须补 H5 的 compile-only 融合上下文审计；H6 板端局部重调仅在 H5 发现融合/TIR 差异时触发。第三点先审计自然流水是否存在可调度的异质 CPU--VTA 干扰组合，再决定是否实现运行时控制器；不新增 VTA 硬件，不恢复大规模切图×AutoTVM 搜索。

## 最终贡献边界

1. 第一项贡献暂定为“面向共享内存 CPU–VTA 平台的编译上下文与内存资源感知分层搜索”：外层搜索编译合法的 CPU/VTA stage、CPU 线程和 VTA island，内层按 `workload + fusion/layout context` 等价类复用 TopHub/history incumbent；将地址可达性、cache/coherence 方式、dtype/layout adapter、CPU–VTA 边界物化、共享 slot、DDR–VTA DMA 碎片与重复载入、共享内存路径的逻辑 demand、经板端资格化的 service proxy 和单物理 VTA 串行反馈到切图排序。只有 H5 显示冻结域编译签名等价，或 H6 在已测候选邻域未观察到 `>=2%` 的 context-dependent 排名翻转后，才在对应范围内把等价类简化为 workload；不外推到完整 AutoTVM 空间。
2. 第二项贡献保持为“跨 Executor 双 slot 零拷贝机制”：共享物理 slot、双视图、所有权状态机、frame generation 和多张量原子交接。
3. 第三项增量贡献主候选改为“面向单 DNN 多帧 CPU--VTA 流水的共享内存路径自争用感知 stage 启动门控与选择性重叠调度”：固定第一项选出的 topology/tile/名义 CPU 线程和第二项的双 slot 交接，优先用编译期 stage 内存签名、真实 CPU--VTA stage-cell 离线标定和运行时 active/pending 状态，只延后被证明有害的跨帧重叠。只有 G1 证明签名相对 bytes-only/pair-independent baseline 有额外预测力时才称 `compiler-informed`；若仅 pair-ID 表有效，降名为 profile-guided、workload-specific stage-gating case study。PS DDR_APM 是可选物理标定层，不是创新本身；AFIFM QoS/issue 只保留为主结论完成后的附录增强。只有自然流水存在异质的 `protect/allow` cell、G1 证明静态签名的额外预测力，且冻结策略在两个前瞻 held-out topology 上相对 calibration 选出的 `best-global-fixed` 改善预注册的稳态 II、同时守住 submit-to-complete frame-latency P95 不退化保护门槛，才独立使用第三创新点名称；若只在人工 streaming 压力下有效，则降为鲁棒性增强。逐边可变 slot 规划保留为 C3-B 研究备选，不自动与本路线同时展开。
4. 不把 AutoTVM 本身、完整联合搜索、APM 工具本身、硬件 compute stall、硬件预取或预驱逐作为独立贡献；论文只声称已实际测得和通过对照验证的部分。

第一项贡献的目标不是超过官方 AutoTVM，而是解决 AutoTVM 给定 workload 后的局部调优没有覆盖的全局问题：不同切图如何改变共享内存边界、VTA island 重入、完整 stage 的 DDR–SRAM 搬运、CPU/VTA 并发 DDR 需求和流水瓶颈。当前 TopHub 是由同 VTA 微架构的 PYNQ 调优记录迁移到 AXU5EVB 的强 incumbent，不是针对本板共享 DDR 与流水切图的联合优化结果。

## 创新点、实现工作与验证矩阵

以下矩阵是后续实施的统一索引。`C1+C2` 是必须收口的两项主创新；`C3` 是条件验证的增量第三点，先做 3--4 天自然重叠、真实 stage-cell 和 trace opportunity upper-bound gate。APM/AFIFM 失败不等于调度问题必然不存在，反之人工压力有效也不能代替自然流水证据。`C3-GQ/C3-B` 均是截止顺延后的增强/研究备选，本月默认不启动。增强项可以增加工作量和论文完整性，但不能把同一项工作重复拆成多个创新点。

| 编号 | 创新点及状态 | 与已有工作的具体差异 | 必须完成的实现 | 关键实验 | 成立门槛、失败归宿与剩余工作量 |
|---|---|---|---|---|---|
| **C1：主创新，已建立待收口** | 面向共享 DDR CPU–VTA 流水的编译上下文与内存资源感知 stage–tile 分层搜索 | AutoTVM 优化给定 workload 的局部 schedule；C1 外层联合 stage/device、CPU 线程、VTA island 和共享边界，内层按 `workload+fusion/layout/quant context` 复用 incumbent 或局部重调 tile，并考虑逻辑 DMA 请求形态与单 VTA 串行 | 87/87 合法 segment 的 pre-codegen context 签名；对唯一签名代表、全部差异段和负对照做完整 build/TIR/DMA；4623 topology 的静态 `SharedEdgeSignature` 与编译代表的运行时资格化；context cache、局部重调触发器及经验性风险筛选 | 已完成 H1–H4、A–D、M0/M1/M2、E0/E1/E2、E6-S；待完成 E5a、E6-R；E3 pilot 已发现 tail 窄化差异，先做共同数值资格化；E4 仍服从 context 差异 gate，仅稳定 `>=2%` 排名翻转才做 E5b | 冻结域 context 可审计、shared-edge 静态全域审计、代表性运行时资格化和统一消融必须完成。H5 编译上下文等价时只复用冻结 incumbent 记录；若 H5 触发 H6，tile 排名翻转则局部重调，未翻转则只在已测候选邻域受限复用。两者都不声称完整 AutoTVM 空间的最优 config 不变。原必做预算约 5–7 日，条件实验再加 2–4 日；已完成项不重复计入剩余工期 |
| **C1+：C1 条件增强** | 融合上下文感知的边界局部 tile 重调与 DMA 请求形态支配剪枝 | 处理同一 AutoTVM workload key 在完整 fused stage 中可能出现不同 TIR、DMA 或 tile 排名的问题；只调变化 context，不做 stage×tile 笛卡尔积 | `compile_context_signature`、变化 context 枚举、正确 tile 交叉重放、局部 history 覆盖和不误删 near-oracle 的 dominance audit | 条件 E3/E4/E5b；2 个变化 context＋1 个不变对照；每个重放 incumbent＋2–4 个正确候选；20–30 次随机/交错测量 | 只有出现方向稳定且 `>=2%` 的跨 context 排名翻转才启用重调；否则只报告“冻结域已测邻域未观察到 `>=2%` 排名翻转”的复用边界证据。约 2–4 日，不单列创新点 |
| **C2：主创新，已建立待收口** | 固定 CPU–VTA 流水的跨 GraphExecutor 双 slot 零拷贝机制 | 差异不在“双缓冲”概念，而在 u-dma-buf CPU/ext_dev 双视图、跨 Executor 直接绑定、四态所有权、frame generation、多张量原子发布和单物理 VTA 下的跨帧安全 | 冻结 slot 分配/绑定和异常协议；补物理 range、owner/generation、slot wait 与 allocator high-water；固化二进制、配置和日志哈希 | materialized-copy vs shared-slot；single-frame serial vs Eager-K2；单/多边界和三 VTA-island；慢消费者、多 tensor、generation reorder、长帧和 3 boot；pipeline-runner 单在途仅作条件诊断 | 正确性、无死锁/过早复用和每帧 framework materialization 为 0 是成立条件；吞吐无需必然提高，但必须报告自然 Top-20 负结果。未测 high-water 前不声称峰值内存下降。收尾约 2–4 日 |
| **C3：条件主候选** | 面向单 DNN 多帧 CPU–VTA 流水的共享内存路径自争用感知 stage 启动门控与选择性重叠调度 | 已有工作多保护加速器或调度多个独立 DNN；只有 G1 证明静态签名的额外预测力时，本文才定位为冻结 AXU5EVB/VTA、单 ResNet18 跨帧场景中的 compiler-informed runtime integration；pair-ID-only 时降为 profile-guided case study，不声称通用 admission 或 contention model | 自然 overlap/readiness trace；真实 reachable stage-cell 表与 `overlap_action_class` manifest；事件式 `StageOverlapController`；`off/serialize-all/class-aware` 策略；active-instance/pending、scheduler wait、overlap 与回退日志；冻结策略 deterministic replay | G0 真实 cell 与自然机会/乐观收益上限 gate；G1 签名分类及 grouped internal validation；G2 控制器安全/开销；G3 两个新 topology 的自然流水前瞻验证；人工 streaming 只作因果放大 | G0 至少找到 2 个 `protect` 与 2 个 `allow` 可达 cell，效应超过冻结噪声门槛，且 opportunity ceiling `>=3%`；G1 的静态签名胜过 bytes-only/pair-independent baseline；G3 在 E/F 上胜过 calibration 选定的 `best-global-fixed`，满足预注册 II 改善与 submit-to-complete P95 不退化保护门槛，三者同时通过才独立列为第三点。stage-gating-only 全链路约 11–16 日 |
| **C3-GQ：答辩后/延期增强** | 端口 provenance 与安全读写资格化后的 AFIFM issue/profile 选择 | 仅当冻结设计确认 load/store→HPC0、compute data→HPC1、instruction/uop→ACP 的端口结构后，才根据 stage/context 与 CPU 重叠状态选择有限安全 profile；不是把 QoS 寄存器本身当创新 | 先做 read-only、snapshot/restore 和 whole-run static sweep；per-stage switching 另计实现与 quiescence 证明 | GQ：default、HPC0/HPC1 issue 单轴、有限 QoS profile；VTA-only、真实 CPU stage 并发、streaming control；比较 fixed-best、C3 gating 与二者组合 | 至少两个 context 各自在 `>=4/5` boot 稳定选择不同最佳 profile，且冻结 selector 优于 calibration 选出的最佳单一 profile，才作为 C3 附录增强。静态 gate 2–4 日，per-stage switching/多 boot 另加 2–3 日；本月默认不启动 |
| **C3-B：延期研究备选** | 面向单物理 VTA 线性多帧流水的 GraphExecutor storage-pool 复用与逐边共享缓冲规划 | C2 提供固定 all-shared K2 原语；C3-B 按边生命周期、关键路径、slot wait 和容量预算选择 `copy/shared`、`K_e`、`pool-anchor/external` | per-edge mode/K、pipeline K1、mixed runner、storage-id 独占性/lifetime 审计、pool-anchor slot0、静态 alias/release plan、exhaustive oracle 与 planner | high-water 基线；K1/K2；all-copy/all-shared/mixed；external-K2 vs pool-anchor；至少四个 topology、5 boot 和正确性压力 | external-K2 的 II 退化不超过 2% 时 slot/high-water 降 `>=20%`，或同容量下 II 改善 `>=3%`，且固定实测池 planner regret `<3%`。只做全局 K 仍属于 C2；两个 topology/3 boot 只能称 case study。14–20 日，本月默认不启动 |

### W1：C1 编译上下文与共享内存搜索闭环

- [x] 冻结正式 pipeline 的量化、`graph_pack`、`FuseOps`、build、TopHub 和哈希记录路径。
- [x] 为 87/87 合法 VTA segment 生成 occurrence 级 pre-codegen manifest，保存量化、`graph_pack`、`FuseOps`、workload/config 和边界签名；只对每个唯一签名代表、全部差异段及分层负对照执行完整 `relay.build`、Graph JSON、归一化 TIR 和直接 DMA 提取。若要改为 87 段全量 build，必须显式废止旧 manifest 的 `full_segment_compile_sweep_forbidden=true` 并重新估算工期。
- [x] 对直接 segment DMA 与单 workload occurrence 聚合做 E2 校验，区分精确量与估计量。
- [ ] 为 4623 个 topology 生成 contract 级 shared-edge 必要条件、理论物化/slot 字节、逻辑 DMA 和单 VTA 资源记录；storage-id、真实 pool、实际物理范围、对齐与 allocator high-water 只在已编译代表上资格化。
  - [x] E6-S 全域静态子项完成，见 `shared_edge_audit/domain_run2`；保留 E6-R 运行时子项未勾。
- [ ] 提供 `task-only`、`context-aware`、`boundary/logical-DMA-aware` 三种统一搜索模式和消融脚本。
- [ ] 签名变化先进入 E4，重放 incumbent 和已有正确邻居；只有 E4 出现方向稳定且 `>=2%` 的排名翻转，才采用 E4 已测最佳配置更新 context history 并反馈重排，不循环到收敛。额外生成 tile 候选只属于有余量时的 C-P2，不能混入 E5b 的最低闭环。
- [ ] 冻结搜索预算、Top-K、regret、被剪枝数量及 oracle/near-oracle 未被误删的证明表。

### W2：C2 零拷贝机制证据冻结

现有基线证据为：框架边界物化 `1,806,336 B/frame → 0`，边界 API 服务时间下降 `97.11%`；自然 Top-20 的稳态 FPS 没有显著提高，三 VTA-island 串行压力方案单 boot 观察到时延下降 `2.83%`。前两项是 C2 的确定机制结果；`2.83%` 在完成多 boot 配对重复前只能称观察值。

- [ ] 固化 copy/shared 两种 runner、配置、二进制、runtime、bitstream 和原始日志哈希。
- [ ] 统一输出每条边的 slot 状态迁移、generation/frame id、物理 range、acquire/release wait、copy calls/bytes 和 API service。
- [ ] 把协议不变量写成可执行断言：同一 slot 任一时刻只有一个 owner；发布前 producer 写入完成；消费完成前不得复用；一组边界 tensor 按同一 frame generation 原子交接；异常退出必须唤醒所有等待者并回收逻辑所有权。用逐帧唯一输入/校验哨兵、随机 producer/consumer 延迟和 generation 扰动覆盖这些不变量，不能只用重复同一张输入验证。
- [ ] 增加 u-dma-buf 在 graph load、slot allocation 和 steady state 的 offset/high-water，区分原 GraphExecutor pool 与新增 slot。
- [ ] 完成慢 producer/consumer、多张量原子交接、错误 generation、长帧及异常退出测试。
- [ ] 完成 single-frame serial 与 Eager-K2、all-copy/all-shared、自然和多 island 压力拓扑的 3 boot 对照，形成 correctness、FPS/II、P95 wait、物化流量和 peak 总表。pipeline-runner 单在途只作条件诊断：只有在论文冻结前已经具备安全的 `max_inflight=1` 且允许 `K_e=1` 的校验/分配协议时，才比较 `(max_inflight=1,K_e=1)` 与 `(max_inflight>=2,K_e=2)`；若仍分配双 slot，只能称 `global-inflight=1/K_e=2`，不能声称 K1 slot/high-water 下降，也不得为这一诊断临时扩展主线开发。
- [ ] 增加“copy service 是否落在关键路径”的受控实验：在固定 stage service 下逐级增加下游 slack/延迟，比较 materialized-copy 与 shared-slot 的 II、边界 service 和 overlap；用它解释“API 时间和物化字节显著下降但自然 FPS 不变”，并区分 copy 消除、并行隐藏和 backpressure 三种情况。

### W3：C3 共享内存路径自争用感知 stage 启动门控与选择性重叠

- [x] 完成首轮 A/B/C/D 自然 overlap/readiness 审计：新增 `queue_pop/slots_acquired/binding_done/run_start/run_end` 时间戳，shared K2 每配置 302 帧，输出 CPU--CPU/CPU--VTA overlap、mutex/slot wait 和 directed encounter，并排除 newly-ready VTA 已有其他 owner 的立即启动机会。旧 runner 保留；尚未做同帧数 ABBA instrumentation-overhead 资格化，未新增 queue occupancy，不能称 queue empty/full 已测。细节与单 boot 限制见 [G0 报告](g0_discovery/EXPERIMENT_G0_READINESS.md)。
- [ ] 完成 G0：仅按 isolated/static 特征预先冻结真实 CPU low/medium/high-memory stage 与 VTA compute/mixed/DMA-heavy segment，不按并发结果挑样本；对自然 trace 中可同时 ready 的方向做 isolated/sequential/allow-now/predefined-wait 配对。固定 governor、温度窗口、CPU/VTA host 绑核，加入 CPU compute-only 负对照；无 APM 时只报告逻辑/有效触达带宽，不称物理 DDR bandwidth。
- [ ] 为 G0 扩展同步 pair harness；不能只改 `run_cpu_vta_pipeline_v1_p7_profile.py`。先在 native stage/component runner 增加逐样本 pre-run hook：CPU 侧在工作集/输入准备完成后、VTA 侧在 `RunStage` 的 `set_input`/view binding 完成后且取得 `vta_run_mutex` 与实际 graph invocation 之前，分别发出 ready 并等待各自 start/release token；每个样本结束再发 done token。Python 只编排两个方向的 start offset、“active done 后 release newly-ready”的非抢占 wait，并保存 actual ready/release/start/end 与 schedule manifest/hash。pair component 本身没有 managed slot 时不得虚构 slot 事件；gate 前 adapter/cache/binding work 单列，不计入可恢复 penalty。
  - [x] 首个真实 pair 的双方向工具资格化：新增独立 `vta_stage_pair_runner.cc`，复用 RunStage 的 optional binding/start/done hooks。token 以同进程 condition-variable 状态实现，Python 预先冻结日程，避免逐样本 SSH/file polling；仅一个 VTA，未实现生产多 island admission。两边结束后才做 get-output 拷贝；100 个样本通过。仍需扩展完整矩阵、isolated/负对照、故障注入与五 boot confirmation，因此父项不标完成。
  - [x] A/B/C/D 自然可达域扩展普查：60/60 预注册时龄条件、3000 个 pair 样本完成；57 个状态匹配条件无等待收益候选，3 个不匹配条件及 2 个未观测方向保留。独立审计输出全部 CSV、真实 overlap 和温度，不能称 3×3/multi-boot 完成；当前不启动控制器。
- [ ] 为每个可达 cell 固化离线 `overlap_action_class`、isolated service、逻辑 DMA/request shape 和已确认动作差异；APM 只作训练组离线标定/事后验证标签，运行时不得读取当前候选自己的 APM 后再声称静态预测或节省上板测量。
- [ ] 在 native runner 实现 `StageOverlapController` 和 `off/serialize-all/class-aware` 三种可重放策略。MVP 决策状态只含 active/pending 的 stage/frame/device、active age 与实际 VTA owner；动作类别不是 stage 固有属性，而是由 `(active,newly-ready,direction,可选 active-age bucket)` 查询冻结 manifest。queue/slot wait 只作诊断。先画 wait-for graph，再实现 VTA pending→无锁等待→取得 `vta_run_mutex`→controller 下原子 revalidate/grant；revalidate 不通过就释放 mutex 重试。任何可能阻塞的 admission wait 不得持有 controller mutex 或 `vta_run_mutex`，不得把等待者记为 active；使用 FIFO ticket/有界优先级、RAII release 及 error `abort+notify_all` 防止死锁和饥饿。若 worker 内 gate 无法证明安全，本月停止独立 C3；central dispatcher 只记为未来工作并重新估时。
- [ ] 输出 active/pending stage、策略状态、scheduler wait/reason、真实 overlap、stage service、VTA mutex/slot wait 和 policy-off overhead；性能 run 的事件先写内存缓冲，禁止逐事件 flush。若另加 queue 诊断，必须显式记录 `push_wait/pop_wait/size_before/after/event_ms`；`queue_depth` 只是逐边容量，不得称全局 frames-in-flight。
- [ ] G2 核心只实现 stage gating，不改变 C1 的线程/topology/tile，也不改变 C2 的 K/handoff/pool。临时 CPU thread cap、global credit 与 AFIFM 均移到主结论完成后的附录/延期增强，不计入 C3 成立条件；在线 APM 和强化学习不做。
- [ ] 若 G0 没有同时观察到自然 `protect/allow` cell、trace opportunity ceiling 不足 3%，或 memory-path 归因 gate 不通过，立即停止 C3；保留负结果作为平台机制表征，不靠人工 pressure 拼第三点。只有把代理实际接入 C1、重新执行消融并证明排序或候选效率改善后，才可另称 C1 的校准层。C3-B/C3-GQ 本月默认不启动。

### W4：C3-B 研究备选缓冲规划

- [ ] 仅在论文截止顺延、C3 的 G0 机制 gate 失败且 E8 证明存在实际内存优化空间时重新评估；至少预留 14--20 个工作日。它不是 APM 失败后的自动 fallback，本月默认不启动。
- [ ] 在 runner 中实现逐边 `copy/shared`、`K_e=1/2`、pipeline K1 和四种输入/输出 mixed path。
- [ ] 审计 Graph JSON storage-id 独占性、物理范围和生命周期，安全时借用既有 pool 作为 slot0，否则回退 external slot。
- [ ] 输出每边 mode、K、来源、live time、物化字节、slot bytes、P95 wait 与 allocator high-water。
- [ ] 先在一个两边小拓扑上穷举 `copy/shared1/shared2` 的全部 9 种组合，建立实测 oracle，再验证 planner；两个 topology/3 boot 只够 case study，若要作为独立贡献须扩到至少四个 topology、5 boot，并包含多 island 与不同容量/关键路径结构。

### 可选微优化：只能增强主创新，不能单独凑数

| 微优化 | 从属项 | 触发条件 | 需要做的工作与实验 | 成功门槛 |
|---|---|---|---|---|
| DMA 请求形态支配剪枝 | C1 | TIR 静态特征和正确候选记录稳定 | 对 bytes、calls、small ratio、reload、SRAM legality 做 Pareto/dominance，并与已测候选 oracle 核对 | 明显减少候选，且不删除实测最优或 2% 内 near-optimal |
| adapter/requantize direct-write 到 consumer slot | C2 | E1/E6-R 找到真实且显著的 dtype/layout 物化热点 | 绑定 adapter 输出到 slot；对比 materialized/shared，测正确性、bytes、边界 service | 消除一笔真实物化，边界 service 改善 `>=5%`；FPS 仅作加分 |
| GraphExecutor input pool 借作 slot0 | C2 或 C3-B | storage-id 独占、range 和 lifetime 审计通过 | K2 只额外分配一个 slot；做 range/generation 压测和 high-water 对照 | 额外 slot/high-water 降 `>=20%`，II 退化上界 `<=2%` |
| 临时 CPU thread cap 或全局 frame credit | C3 附录，不计成立条件 | C3-G3 已经成立、论文截止顺延，且 trace 明确指出单纯 gate 的剩余损失 | 两者只选一个；加入可重放动作、实际线程/credit 日志，与 best fixed 和 stage-gating-only 对照 | 在新的附录 held-out 上优于已成立的 stage gating；否则删除，不回改 C1/C2 主结论 |
| AFIFM QoS/issue profile | C3-GQ，延期增强 | C3-G3 已成立，且 bitstream 端口映射、只读资格化、安全 save/restore 和 whole-run static sweep 均通过 | default、有限单轴 profile、`best-global-fixed`、context profile 与 C3 组合 | 至少两个 context 各自在 `>=4/5` boot 稳定选择不同最佳 profile，且 selector 优于 `best-global-fixed`；否则只报告固定敏感性 |
| 提前 acquire 空 shared slot | C2 | slot wait 被证明位于关键路径 | producer 提前 acquire；与慢消费者及原协议配对比较 | P95 wait 或 II 改善 `>=3%`，且没有额外错误或不可接受内存 |
| K3/K4 | C2/C3-B | K2 仍有可复现 backpressure 且队列深度允许 | 补 K2/K3/K4 memory–II 曲线 | 至少一个额外 K 点形成非支配 Pareto；否则停止 |

提前 acquire 只能称为 shared-slot 调度优化，不是 VTA SRAM 预取。硬件预取/预驱逐、跨算子 SRAM 常驻、VTA stall counter、IOMMU/SVM 和通用 DAG allocator 继续列为未来工作。

### 贡献组合与时间规则

1. **最低、优先保证：`C1+C2`。** 先补 W1/W2 的审计、正确性、高水位和消融，不能因为追逐第三点使已有两项证据不完整。
2. **条件上限目标：`C1+C2+C3`。** 本月先用 3--4 天完成自然 overlap 审计、G0 和 trace opportunity upper-bound audit；只有真实、可达的 `protect/allow` cell 共存且乐观机会上限达到 3%，才考虑 G1--G3。该上限只负责止损，不是反事实 II 或性能结论；最后 4--5 个完整工作日冻结论文、脚本和原始证据。
3. **C3 的降级归宿：**若请求形态/APM 表征成立但 G3 没有端到端收益，默认只称平台机制表征；只有代理实际接入 C1 排序、重跑统一消融并证明排序或候选效率改善，才可称 C1 的板级校准层。若只有人工 streaming 压力有效，只称鲁棒性增强；若 G0 没有异质性、自然机会或收益上限，则保留严谨负结果并停止实现。
4. **延期研究备选：`C1+C2+C3-B`。** 只有论文截止整体顺延、C3 的机制 gate 失败、E8 又证明真实容量/slot 热点并可另留 14--20 日时才重新立项；不能把“APM 不可用”直接当成转向理由，也不得同时全面实现 C3 与 C3-B。C3-GQ 同样默认不在本月启动。
5. **边界不重叠：**C1 只在预先固定的 Eager-K2 执行语义下选择 topology、tile、VTA island 与名义 CPU 线程并输出静态内存签名，不学习或控制运行时 pair overlap；C2 独占物理绑定、handoff 协议与固定 K2；C3 只消费冻结签名并决定 ready stage 的启动时机，不改变 topology/tile/thread/K/handoff/pool。thread cap、global credit、AFIFM 均为主结论完成后的附录/延期增强，不计入 C3 成立条件。

工期按单人串行预算而不是乐观并行预算排期。无 H6 条件支路时，W1 约 5--7 日、W2 约 2--4 日、G0 含 discovery 与触发后的 5-boot confirmation 约 3--4 日，最终证据/论文冻结另留 4--5 日，总计约 14--20 个工作日；编译任务与板端等待可以交错，但不预先从预算中扣除。若 E1 触发 H6，E3/E4/E5b 再加 2--4 日，最坏变为 16--24 日：此时必须先砍 E7、G0 confirmation 和全部 C3 后续，不能删掉 C1 条件闭环。只有 G0 通过后仍剩 G1--G3 所需的 8--12 个执行日，且另有最终 4--5 个冻结日，才启动 G1；G1 通过且仍保有 G2+G3 的 7--10 日与冻结窗口，才启动 G2。否则第三点止于 G0/G1 的候选、case study 或负结果。C3 全链路现实预算为 11--16 日（G0 3--4、G1 1--2、G2 4--6、E/F build 与 G3 3--4，板端失败恢复另计），不能再叠加 C3-GQ、C3-B 或额外执行器。因而本月不可动摇的毕业硬承诺是 `C1+C2+最终冻结`；G0 discovery 是第三点的优先 go/no-go，完整 confirmation 服从上述预算 gate。

## 核心研究问题与可证伪假设

必须依次回答以下问题，不能预设“stage 一定导致非线性 DMA”：

1. **H1：workload/config 不变性。** 同一卷积置于不同 stage 后，其 AutoTVM workload signature 和 TopHub config 是否相同？若相同且进一步通过 H5/H6 gate，则在冻结搜索域和已测候选邻域内按同一 workload 受限复用调优记录，避免 stage×tile 笛卡尔积；若不同，则定位 shape、layout、fusion 或量化变化。
2. **H2：VTA DMA 可加性。** 完整 stage 的 LOAD/STORE 请求与 payload 是否等于所含已调优 workload 的逐算子之和？用“完整 stage profile − 单算子聚合预测”定义残差，不人为要求残差非零。
3. **H3：stage 分裂效应。** 在算子集合和 TopHub 配置固定时，一个连续 VTA Executor、两个普通交接的 VTA Executor、两个共享 slot 交接的 VTA Executor，在内部 DMA、边界物化、串行时延和流水 FPS 上是否不同？
4. **H4：共享内存流水系统效应。** 即使单算子 DMA 可加，不同切图是否仍通过 CPU–VTA 边界字节、CPU/VTA 并发访存窗口、VTA island 重入和单物理 VTA mutex wait 改变实际流水吞吐？其中只有经过 E7 或等价计数器归因的部分才进一步称为物理 DDR 效应。
5. **H5：fusion/TIR 上下文不变性。** 同一个语义卷积 occurrence 位于不同合法 VTA segment 时，虽然 workload key/config 相同，其 FuseOps primitive、额外残差输入、量化与 layout 边界、归一化 lowered TIR 和完整 segment DMA 是否仍相同？该问题不能由 H1 的 dispatch audit 代替。
6. **H6：上下文相关 tile 排名。** 若 H5 发现 fusion/TIR 变化，在完整 fused primitive/stage 中重放同一组正确 tile 时，TopHub incumbent 的排名是否发生稳定且具有实际意义的变化？只有出现至少 2% 且配对方向稳定的翻转，才进入边界邻近 workload 的局部重调；否则停止 stage×tile 迭代。
7. **H7：自然共享路径干扰的异质性。** 在真实多帧流水中，不同 `(active stage, newly-ready stage, direction)` cell 的 `allow-now` 与预定非抢占 `wait` 是否产生超过噪声、且方向稳定的差异，并同时存在 `protect` 与 `allow` 类？若所有可达 cell 近似相同，简单固定策略已经足够，C3 不成立。
8. **H8：编译签名的动作可预测性。** `SharedEdgeSignature + VtaMemorySignature + isolated service + direction` 能否在按 stage-cell/context 整组留出时预测 `allow/protect` 动作？APM 若可用只作训练组离线物理监督和冻结后机制验证，不把待预测样本的测量泄漏进输入。
9. **H9：运行时 stage 启动门控的端到端价值。** 冻结的 class-aware 策略能否在未用于定阈值的两个新 topology 和新 boot 上，优于 `off`、`serialize-all` 与 calibration 选定的 `best-global-fixed`，改善预注册的稳态 II 并守住 submit-to-complete frame-latency P95 不退化保护门槛？只有 cell 微基准预测准确而无流水收益时，不独立成点。
10. **H10：AFIFM 动作的互补性。** 若延期启用 C3-GQ，不同 context 是否各自在 `>=4/5` boot 稳定选择不同的最佳安全 QoS/issue profile，且 context selector 是否优于 calibration 选出的最佳单一固定 profile、进一步改善 C3 的 Pareto？否则 AFIFM 只作平台敏感性实验。

H2、H5、H6 的正负结果都允许形成贡献：若 H5 中 fusion/TIR/DMA 编译签名等价，则支持在冻结域 occurrence 间复用同一 incumbent 记录，但不证明完整 AutoTVM 空间的全局最优 config 不变；若 fusion/TIR 变化而 H6 在已测候选邻域未观察到 `>=2%` 的 tile 排名翻转，则只加入 fusion/boundary correction，并把复用限制在该已测邻域；若 tile 排名也变化，则只对变化签名的边界邻近 workload 局部重调。不得为了强化创新而隐瞒线性或不敏感结果。

假设与实验编号固定映射为：`H5 → E1/E2/E5a`，需要因果拆分时再触发 E3；`H6 → E4/E5b`；`H7 → G0`；`H8 → G1`；`H9 → G2/G3`；`H10 → C3-GQ`。H1 的 workload dispatch 表不能代替 H5，E3 的融合/量化拆分也不能代替 H6 的 tile 交叉排名；人工 streaming 不能代替 H7/H9 的自然流水证据。

## 编译期信息与 Profile 边界

| 信息 | 切图后、tile 前 | TopHub/tile 确定并 lowering 后 | 必须板端 profile |
|---|---:|---:|---:|
| CPU↔VTA 边界张量 shape、方向和字节 | 精确 | 精确 | 否 |
| VTA island 数量、重入次数、workload multiset | 精确 | 精确 | 否 |
| 量化、graph_pack 与 FuseOps primitive 分组 | 可在各候选 Relay module 上审计 | 精确 | 否 |
| AutoTVM workload key 与命中 config | 可提取 key | 精确 | 否 |
| 融合上下文下的归一化 lowered TIR/Graph JSON | 不确定 | 可提取并做结构比较 | 否 |
| 输入/权重/输出唯一必要字节 | 可计算 | 精确 | 否 |
| tile 对 SRAM 的工作集和容量合法性 | 候选级估计 | 可确定 | 否 |
| 逻辑 DMA 指令/请求数及 payload | 不确定 | 静态 shape 下原则上可从 lowered TIR/指令循环精确统计 | 用 runtime counter 验证 |
| 小 DMA 请求比例和平均 payload | 不确定 | 可由请求尺寸静态统计 | 用 runtime counter 验证 |
| 输入/权重重复载入倍数 | 不确定 | 可计算 | 用 runtime counter 验证 |
| DMA 时间、有效带宽、CPU/VTA DDR 争用 | 不可得 | 不可得 | 是 |
| compute 等待 LOAD/STORE 的 cycle | 不可得 | 不可得 | 当前硬件无 stall counter，不作结论 |

定义三个核心静态特征：`R_weight = DMA_weight_bytes / unique_weight_bytes`、`R_input = DMA_input_bytes / unique_input_bytes`、`S_avg = DMA_load_bytes / DMA_load_calls`。runtime profile 的 DMA API 计数只验证逻辑搬运；不得表述为 DDR 控制器实际 AXI burst 或 compute stall。

## 第一周：修复可比基线

- [x] 显式恢复 TopHub dispatch，避免审计 context 关闭 Relay 的自动 TopHub。
- [x] 在 stage manifest 与 cache key 中记录 TopHub 文件绝对路径、版本、字节数和 SHA-256。
- [x] 重启板卡后重新部署 192 MiB u-dma-buf、固定 HPC bitstream 与匹配 RPC runtime，并保存 boot/runtime/bitstream 哈希。
- [x] 从源码重新编译 topology B，逐 stage 检查 TopHub workload 覆盖和数值正确性。
- [x] 运行 22 帧串行与 queue-depth=2 流水，丢弃前 2 帧；第二次流水为 10.430 FPS，距旧 B 的 10.724 FPS 为 -2.74%。

退出门槛：新编译 B 的所有 packed convolution 均命中 TopHub，输出稳定，FPS 与旧基线处于可解释误差范围；不得再用 1.746 FPS 的无 TopHub 构建作为正式 baseline。

## 第二周：incumbent-aware 有界 tile 搜索

- [x] 将 Top-20 去重 workload 的 TopHub/history 配置全部写入 incumbent manifest；10/10 命中，0 fallback。
- [x] 冻结第一批候选生成规则：每类 workload 最多 8 个，先覆盖 `tile_h/tile_w/tile_ci/tile_co/oc_nthread/h_nthread` 单轴最近邻，再补第二空间/通道邻居；总预算不超过 80 次板端测量。
- [x] 每个候选先过独立数值正确性，再比较时延；80 次测量中 32 次通过、41 次编译失败、7 次数值失败，失败记录未进入覆盖日志。
- [x] 记录 LOAD/STORE 请求数、payload、小请求、输入/权重 payload 和请求粒度；按单轴邻居生成候选级对照表。
- [x] incumbent 永不被删除；实验日志改为 TopHub 上的稀疏安全覆盖，只有 incumbent 同 runner 通过且候选至少快 2% 才能覆盖。
- [x] 将“AutoTVM 候选测量后重载固定 bitstream，再做完整 Relay stage 正确性”加入协议；非法候选运行可能污染 PL/RPC 状态。

退出门槛已满足：10/10 workload 都有可重放 incumbent；首轮安全覆盖为 0，最终配置不慢于 incumbent，并已形成 DMA 碎片与重复载入对照表。该结论仅表示 TopHub 在所测单轴邻域内未被击败，不表示全局最优。

## 第三周：stage 反馈和原生流水线验证

- [x] 完成首轮“TopHub 固定 tile—单轴有界候选—安全覆盖”比较；没有候选越过 2% 接受门槛，因此没有生成冒进的 tuned 配置。
- [x] 对自然 Top-20 的 4 种 topology、5 个唯一 VTA stage 重建成本并重排；因安全覆盖为 0，排序按预期不变。
- [x] **实验 A：workload/config 不变性表。** 4 种 Top-20 topology 的 10 类 workload 跨 stage 均命中同一 TopHub tile。该结果只建立 dispatch/config 不变性；在 H5/H6 完成前，只能说明可按唯一 workload 去重 TopHub 查询，不能单独证明不同融合上下文中的性能最优 config 可无条件复用。
- [x] **编译期 DMA 提取器（搬运部分）。** 已在 TopHub 配置确定后遍历 lowered TIR 并静态展开循环，输出 10 类 workload 的 LOAD/STORE 请求、输入/权重/输出 payload、小请求、stride、平均请求大小、`R_input` 和 `R_weight`。8 个 direct-template 正确 workload 的六项静态指标与 runtime profile 全部精确一致；两个 Relay projection 的卷积 input/weight/STORE 精确一致，额外差异为图级 ACC 请求。SRAM 峰值工作集仍待从 buffer/index 范围补充，不阻塞 DMA 聚合模型。
- [x] **相邻切点内存 delta 表。** 固定 VTA 起点 03，对终点 15/16/17 联合计算 incumbent tile 的静态 DMA 与图级 live-tensor 边界。`15->16` 增加 64 次/217600 B 静态卷积 LOAD（实测 66 次/219648 B，残差为 ACC）并减少 100352 B 边界；`16->17` 的静态及实测 VTA LOAD/STORE 增量均为 0，边界再减少 100352 B。由此把“unit 交给 CPU 还是 VTA”转化为可在部署前取得的向量 delta，而不是按算子时间线性相加。
- [x] **实验 B（runtime 与单 workload TIR 交叉验证）：DMA 可加性表。** 已按 occurrence 聚合 10 类 workload，并用两个数值正确的 isolated Relay projection profile 补齐裸模板失败项；5 个完整 stage 的 input、weight 和 STORE bytes 均精确可加，LOAD 总字节误差为 -0.12% 至 -0.16%，LOAD 请求误差为 -1.92% 至 -6.34%，剩余残差来自 conv-only 聚合未覆盖的 ACC/ALU 与图级工作。这里仍是“单 workload TIR 聚合预测 vs 完整 stage runtime”，不是 87 个完整 segment 的逐段 TIR 提取。
- [x] **实验 C（串行与内存计数）：受控 stage 分裂。** 固定 layer4 的 5 个单元及 TopHub config `203/243/203`，对比单一 VTA Executor、两个 VTA Executor 向独立 ext_dev DDR buffer 物化一次、两个 VTA Executor 共享同一 buffer。30 次交错测量中三者中位为 20.327/22.929/22.065 ms；物化路径明确增加 100352 B、1 次 framework copy，共享路径输出与其逐元素相同。三者 VTA LOAD/STORE、payload 和 driver 指令完全一致，说明该单一 block-aligned cut 不产生 DMA 碎片；共享路径相对物化的配对节省中位 1.290 ms，但相对单 Executor 仍有 1.407 ms 调度/交接代价。独立切分量化相对整体量化产生 30/25088（0.120%）元素差异，因此 C 不能证明融合/量化上下文不变；流水 FPS 与跨 boot 重复并入实验 D。
- [x] **实验 D：共享内存流水系统效应。** 在一个月时限内冻结 Top-20 的 4 个代表 topology，覆盖长/短单 island、双 island、大/小边界和不同 DMA 强度；比较边界字节、VTA mutex wait、逻辑 DMA 描述和实际 FPS，并完成 3 个独立 boot。样本不足以安全拟合物理 DDR 时间系数，因此实验退出条件收敛为机制复现与剪枝规则，不再扩展到 6--10 个混杂 topology。
  - [x] 首轮四 topology 机制表已完成：A/B/C/D 的实测周期为 91.977/96.990/90.962/95.233 ms；旧版把不同 owner 的逻辑 demand 直接求和后得到 14.158/14.114/14.259/17.750 MB/帧，该标量只保留为被否定的 bytes-only descriptor，不代表物理 DDR traffic。A/C 的 VTA DMA 完全相同但周期相差 -1.015 ms；D 比 B 的 LOAD payload 高 29.30%、权重 payload 高 49.61%、mutex wait 多 20.336 ms，周期反而低 1.757 ms。该结果只是否定“总字节简单相加即可排序”，促使模型保留 CPU、VTA、边界 work 与逻辑内存 service proxy 的分资源向量；它不构成物理 DDR 瓶颈证据。当前仅 4 个混杂样本，不拟合 M2 系数；仍需扩展样本和跨 boot。
  - [x] 第二独立 boot 已完成四 topology 的冻结二进制重放：每类先做串行正确性与 22 帧流水，再补一次 22 帧和一次反向顺序 62 帧流水。所有输出哈希及 LOAD/STORE calls/payload 跨 boot 完全一致；第二 boot 62 帧 FPS 为 A/B/C/D=`10.955/10.020/10.818/10.840`。第一 boot 的细粒度顺序 `C>A>D>B` 在第二 boot 变为 `A>D>C>B`，周期排序 Spearman 仅 0.400；B 的 62 帧 completion interval P95 为 136.753 ms。机制规律已复现，但细粒度 FPS 排序受 CPU/runtime 抖动影响，不能据两 boot 拟合 DDR 系数。
  - [x] 第三 boot 完成 4×4 Latin-square：`A-B-C-D / B-C-D-A / C-D-A-B / D-A-B-C`，每格 62 帧、共 992 帧。A/B/C/D 四次中位 FPS=`10.896/10.091/10.756/10.489`，范围=`10.567--10.928/9.973--10.404/10.383--10.851/10.463--10.588`。D 在 4/4 轮快于 B（中位 +0.444 FPS）；A 在 3/4 轮快于 C（中位仅 +0.125 FPS），四轮排序仍不完全相同。三 boot 输出哈希及 LOAD/STORE calls/payload 完全一致。正式结论限制为内存机制、端点 delta 和支配剪枝，不拟合 DDR 系数，不报告亚 1 FPS topology 优势。
- [x] 已报告切图排序变化与负结果边界：M0 在四个实测代表上的顺序为 `B>C>A>D`，M1 加入边界/单 VTA 资源后变为 `A>B>C>D`，命中第三 boot 中位实际 Top-1 A；M2 的单-workload 精确、segment 聚合 tile-DMA service-proxy 下界未继续改变顺序。按 workload key 去重可把相同 8-trial 邻域从 39 个 stage-placement 对应的 312 次降为 10 个唯一 workload 的 80 次，减少 232 次（74.36%）；这首先是查询/候选预算上界，只有通过 H5/H6 gate 后才可称“冻结搜索域与已测候选邻域内、由编译审计支持的受限复用”，不能外推为完整 AutoTVM 空间的最优配置不变。TopHub incumbent 在已测邻域内无合格替换，不包装成计算加速。

原 H1--H4 退出门槛已经满足；最终退出还要求 H5 给出冻结搜索域内的 fusion/TIR 上下文矩阵，并按触发规则完成或明确跳过 H6。最终必须明确 stage 影响来自算子集合、融合/量化上下文、边界物化还是并发资源竞争，不能用相同 config 代替相同编译结果。

## 内存感知切图成本模型与消融

切图 `P` 的 VTA 逻辑搬运先按已调优 workload 聚合，再加入由实验 B/C 资格化的 stage 修正：

```math
D^{logical}_{VTA}(P)=\sum_{w\in W(P)} n_w D(w,c_w)+D_{stage-cut}(P),
```

不同 owner 的量不能再直接求和成“共享 DDR 字节”。内存描述保持为分量向量：

```math
\Phi_{mem}(P)=\left(D^{logical}_{VTA},D^{logical}_{CPU},B^{copy}_{boundary},M_{slot},Q_{request}\right).
```

流水周期模型保持资源下界形式：

```math
T_{model}(P)=\max\left(\widetilde T_{CPU-pool},\widetilde T_{single-VTA},\widetilde T_{mem-proxy},\widetilde T_{longest-stage}\right).
```

其中 `c_w` 是 TopHub/history incumbent，`D(w,c_w)` 是 lowering 后的单 workload DMA 描述，`Q_request` 保存 calls/payload/small-ratio/reload 等请求形态；`D_stage-cut` 只有在实验测得显著残差时才启用。所有边界 work 在进入带波浪号的资源项前只归属一个 owner：framework memcpy/adapter/cache work 归 CPU 或明确的 boundary host owner，VTA LOAD/STORE 归 VTA stage，shared-slot 容量只进入 `M_slot`，因此公式外不再另加 `T_boundary`。`T_mem-proxy` 仅表示由独立 component qualification 得到的逻辑内存 service proxy，不是物理共享 DDR 时间。三 boot 的 4 个 topology 不足以从逻辑请求/字节稳定回归该项，因此本月不拟合自由 DDR 系数：DMA 特征只用于端点增量、经验性资源下界和支配剪枝；实测 VTA service 已包含其内部传输，不能再重复收费。

- [x] 已在完整 972528 个执行配置、4623 个 topology 上比较 M0/M1/M2。M0 Top-20 只有 1 种 topology；M1/M2 Top-20 均有 4 种。当前 M2 对 87 个 VTA segment 使用的是 10 类单 workload 的精确 incumbent-tile TIR DMA 按 occurrence 聚合后的估计，再结合独立资格化组件带宽；不是 87 个完整 segment 的逐段精确 TIR 提取，也不使用完整 topology FPS 拟合参数。
- [x] 已在三 boot 的四个冻结实测代表上报告描述性验证：M0/M1/M2 Spearman=`-0.400/0.400/0.400`，regret@1=`7.39%/0%/0%`，达到 95% 实测 oracle 所需上板候选数=`2/1/1`。样本仅 4 个，结果不宣称泛化。
- [x] 校准与验证口径已分离：CPU 原子成本通过 23 个 grouped holdout；LOAD/STORE 带宽来自独立 DMA component qualification；tile-DMA 来自 lowered TIR；A/B/C/D 完整流水 FPS 只用于冻结后的描述性评价，没有反向拟合任何系数。
- [x] M2 没有改善排名，按预注册规则回退到 M1 作为最终排序模型。M1/M2 Top-20 为 20/20 完全相同；在全部 972528 个配置中，M2 的逻辑内存 service-proxy 下界成为瓶颈的数量为 0，最接近的配置也只达到 M1 资源下界的 17.27%。精确 DMA 保留用于 endpoint delta、tile 退化排除和支配特征，不强行增加自由参数，也不把该 proxy 改称物理 DDR 下界。

第二轮双轴 tile 搜索降为可选项：仅在实验 A/B/C 完成后，最多追加 40--80 个保持 SRAM 工作集近似不变的 `tile_h/tile_w ↔ tile_co` 联动候选；停止继续测试已经显示退化的纯空间缩小和当前 `tile_ci` 邻域。是否超过 TopHub 不作为退出门槛。

## 补充收敛阶段：切图—融合—AutoTVM 关系

该阶段只补现有证据链缺口，预算为 2--3 天本地 compile-only 审计和至多 2--4 天条件式板端实验。它不重复实验 A/B/C：A 只看 workload/config，B 只验证 5 个 stage 的 DMA 可加性，C 同时改变 Executor、buffer 和独立量化；本阶段分别观察 FuseOps/TIR，并在同一量化图和单 Executor 内隔离局部 fusion barrier。

### C-P0：必须完成的 compile-only 审计

- [x] **实验 E0：编译职责与搜索空间审计。** 用源码位置和 SHA-256 冻结表明确：切图限定 Relay module 范围，`quantize/graph_pack/FuseOps` 决定量化、布局与 primitive 分组；在本项目冻结 commit 的 VTA `conv2d_packed` schedule template 中，TOPI schedule 固定 cache/DMA/tensorize 结构，AutoTVM 搜索 `tile_b/h/w/ci/co` 与 `oc_nthread/h_nthread`。AutoTVM 的空间由 template 定义，不能把上述参数外推为 AutoTVM 的普遍能力边界，也不得写成“AutoTVM 自动搜索 Relay 算子融合、跨算子预取或驱逐”。2026-09-08 已完成职责表与来源哈希冻结；三个正式 build 的 FuseOps 前后结构与恢复的 workload/config 均和审计入口一致。见 [E0/E1 报告](compile_context_audit/E0_E1_COMPILE_CONTEXT_AUDIT.md)。
- [x] **实验 E1：87 个合法 VTA segment 的两级编译上下文与融合签名审计。** 第一级复用正式准备/优化路径并显式恢复查询 workload，三个完整 build 已资格化入口；87/87 段保存量化、pack、primitive、contract 与 TopHub 信息。第二级对冻结的 23 个代表执行正式 build，覆盖全部 72 类 primitive、层级端点及负对照，保存 Graph JSON、lowered TIR、逻辑 DMA 和来源哈希。跨段卷积由完整 packed weight 映射到 unit/ordinal，再通过 compiler hash 关联实际生成函数，不只按 shape 合并。完成的是“manifest all; compile grouped representatives”，不是 87 次完整 build。
- [x] **E1 完整性门槛。** 87/87 第一级与 23/23 第二级完成；72/72 类覆盖。代表集中同一语义卷积有 84 次出现、19 个身份、65 次非自 TIR 比较，全部相同；primitive 类有 110 次非自比较全部相同，另有 37 类只有单实例，不能将其计为跨实例不变实验。逻辑 DMA 的段级对账覆盖 23 段，物理 DDR/运行时性能不在此结论内。层级端点和各级分母见 [新报告](compile_context_audit/E1_REPRESENTATIVES_AND_MEMORY.md)。不变结论针对冻结 incumbent 和已编译代表，不能外推完整 tile 空间。
- [x] **E1 第一级子门槛（历史先完成项）。** 87/87 成功，19 个语义卷积的 693 次 placement 全映射，workload/config/fusion context 在各自 occurrence 内均不变。完整图 1,267 次 primitive 调用归为 72 类；[23 段代表清单](compile_context_audit/precodegen87_run1/coverage_and_representatives.json)现已全部执行。
- [x] **实验 E2：直接 segment DMA 与 occurrence 聚合的校验。** 23/23 E1 代表完成完整图静态 LOAD/STORE/ACC 与 occurrence 对账，LOAD 残差全部由 fused ACC 解释；全体代表再完成上板输出及 LOAD/STORE/ACC、host ALU/GEMM push 计数资格化，207 次采样无差异。只对这些段称“segment 级精确静态逻辑 DMA（运行时资格化）”，不把 ALU push 次数当作硬件操作数或 stall，也不称未完整捕获/上板的其他 64 段已逐段精确验证。见 [合并证据](compile_context_audit/board_e2_all23_summary.json)。

- [x] **E2 静态子项。** 23 段完整 LOAD/STORE/ACC 与裸卷积聚合对账，输入/权重/STORE 相同，全部 LOAD 残差由 fused ACC 解释；另保存 host ALU/GEMM push-scope 静态次数和 CPU helper 逻辑访问。CPU 计数包含向量 lane、静态条件及 GraphExecutor 重复调用。后续已完成全体 23 段的 DMA/push counter 和数值检查；CPU helper 的物理 DDR traffic 仍未测量。

E2 已通过层级端点、block-aligned 对照及全部剩余代表；E6-S 后续全域审计已完成。下一轮优先解决 E3 pilot 揭示的跨切点 int8 回绕差异，冻结共同数值参考，再执行完整路径数值/量化对照。已有 [6 路径 ledger](compile_context_audit/endpoint_full_path_contracts.json)记录 CPU prefix 不变、suffix 的 tail 归属变化、CPU 线程控制和 float32 边界；不把内部 helper 再收费为 edge copy，也不丢掉外部 tail。三个配对的 stage 内逻辑 CPU 访问减少 `2B`、输出 contract 减少 `4B`（B 为 packed int8 单输出大小），不等于整条流水 DDR 或实际峰值内存变化，也尚不构成数值等价的性能对照。当前没有出现共同卷积 context 差异，不立即新增 tile 搜索；量化/边界因果检查仍需按 E3 执行，H5/H6 总门槛不在本轮自动全勾。

E1 的预注册解释分支如下：

1. `fusion/TIR/DMA` 全部不变：当前冻结域的 incumbent 编译结果没有观察到卷积调度上下文差异，可在这些 occurrence 间复用同一 incumbent 记录；不外推成 stage-specific AutoTVM 在完整配置空间绝对没有必要，也不声称未测配置的排名不变。
2. workload/config 不变但 fusion/TIR 局部变化：AutoTVM key 对上下文失明；先加入 fusion-cut/完整 segment correction，不能直接推导必须重调 tile。
3. workload key 或 layout 变化：按 `workload + fusion/layout/quant context` 等价类去重，而不是只按 workload。
4. 差异扩散到远离切点的内部 occurrence：推翻 workload-only reuse，缩小可复用范围并触发 H6；不隐瞒反例。

### C-P1：由 E1 触发的小型因果与 tile 实验

- [ ] **实验 E3：融合、Executor、物化和量化的因果拆分。** 至少选一个融合敏感候选切点（优先 layer4 `skip_proj | add_relu_tail`）和一个 block-aligned 负对照（复用 block0|block1）。尽可能从同一份一次量化的 Relay 派生四个版本：A=`单 Executor + 默认融合`；B=`单 Executor + 仅目标边 stop_fusion`；C=`双 Executor + shared slot`；D=`双 Executor + materialized copy`。同时做“全图一次量化后切分/冻结 qparams”与当前“各 stage 独立量化”的最小正确性对照，以解释 Experiment C 的 0.120% 差异。记录 primitive、边界 dtype/layout/storage、完整 LOAD/STORE/ACC/ALU、driver、copy API/bytes、配对时延和输出。
  - [x] 三对层级 tail 的 CPU-only 数值 pilot，复用冻结 E2 输入并验证模型参数身份；已发现并用 Relay 链及逐元素回绕预测解释 43 处差异。
  - [x] 局部 once-quantized 主机共同参考及六臂数值资格化：三组切分后原生表达式重组与原图结构相同；A/B 与 int8 split primitive 多重集合一致；float32 编码/解码对 256 个 int8 值精确往返；九组冻结输入全通过。producer 与旧独立量化版本 body 结构相同。本项保留回绕，不是整网量化政策修复或 VTA 物理绑定资格化。
  - [x] 既有屏障的主机机制对照：只去掉两个 pre-tail stop_fusion，三组 primitive 均 6→5，输出保持一致。原先设想的 tail “加屏障”在这些冻结图上实际是已有屏障的负对照；不再把其零变化写成“切图永不干扰融合”。去屏障尚未 VTA 编译，不据此启动 E4 或修改正式 schedule。
  - [x] 三组局部板端共同参考资格化：旧 float32 VTA producer 加保持量化语义的 CPU tail，mono/shared/copy 各 27 次正式运行，总计 81 次；输出及各自 census 计数全通过。共享 CPU view 物理范围、对齐和交接通过，未重调 producer。仅 single-inflight，不是完整流水/coherence 结构归因或 E6-R 全域通过。
  - [ ] 真实流水激活、block-aligned 对照、最终数值政策及四臂性能仍需完成。不得将补 clip 默认为等价修复，不得把当前机制直接当成旧 Experiment C 的 0.120% 原因。若量化政策改变，单列精度资格化与新基线，保留旧结果；新 CPU tail 含显式转换，service 必须重测，不回填旧 profile 冒充成本不变。
- [ ] **实验 E4：条件式融合上下文 tile 交叉验证。** 只有 E1/E3 显示 fusion 或 TIR 签名变化才执行。最多选择 2 个变化 context 和 1 个不变负对照，每个 context 先重放 `TopHub incumbent + 2--4 个已有正确邻居`，必须在完整 fused primitive/stage 中交叉编译，不能用 bare conv template 代替。若已有正确记录不足 3 个，允许只为凑足 3 个正确 config 生成最少量 qualification-only 候选，所有成功/失败尝试均计入跨 context 合计不超过约 16 个 stage build 的上限；到达上限仍不足 3 个时，将该 context 标为证据不足并停止 H6 强结论。正确配置在 3 次 warmup 后做 20--30 次随机/交错重复，报告 Top-1、rank correlation、incumbent regret、DMA 和配对区间；未出现 `>=2%` 且方向稳定的 context-dependent tile 翻转即停止。
- [ ] **实验 E5a（必做）：复用预算和静态搜索消融。** 定义 `compile_context_signature=(workload, fused-op signature, boundary layout, quant state)`；无论 E4 是否触发，都比较 task-only reuse、context-signature reuse 和 boundary/logical-DMA-aware ranking，报告 `10 个 unique workload` 对应的实际 `K 个 unique context`、覆盖率、测量预算、剪枝数和现有 Top-20/Top-1 是否变化。
- [ ] **实验 E5b（条件）：最低配置反馈和重排。** 只有 E4 出现方向稳定且 `>=2%` 的跨 context tile 排名翻转才执行；最低闭环只采用 E4 已测且通过正确性的最佳配置更新 context-keyed history，并重排已有 972528 配置，不在 E5b 内再生成候选，也不重新全空间 AutoTune。E4 未过 gate 或证据不足时明确跳过 E5b，但 E5a 仍须完成。

### C-P2：可选而非毕业门槛

只有 E4 通过 2% gate、E5b 已用 E4 实测最佳配置完成最低反馈且仍有余量，才允许对被 E1 标记的边界邻近 context 额外生成并测量合计不超过 16 个搜索候选；这一扩展与 E4 的 qualification-only build 分开记账。全局禁用 FuseOps、87 段逐段全量 AutoTune、继续扩大纯 tile 邻域均不做；全局 `FuseOps=off` 只能作为编译诊断，因为它会同时改变大量 primitive，不适合作为切点因果对照。

## 共享内存研究对齐与第三创新点决策

相关工作的边界必须先写清。AxoNN 已覆盖共享内存 SoC 上的单 DNN layer mapping 与 device transition，HaX-CoNN 进一步覆盖并发多 DNN 的 shared-memory contention；CoDL 已说明统一内存中的 layout transform、mapping、同步和交接不能忽略；CPU--FPGA 的 SVM/IOMMU、zero-copy、cache-coherence policy、FPGA tensor lifetime/arena、weight prefetch 和片上残差复用也都有直接先例。因此本文不得泛称“首个共享内存感知切图”“首次 zero-copy”“首次缓冲生命周期规划”或“首次预取”。最可防守的交集是：**面向单物理 VTA 的 TVM 流水，把切图产生的 fusion/layout 编译上下文、lowered-TIR 的 tile-sensitive 逻辑 DMA 请求形态，以及跨 Executor shared-edge 实现语义共同反馈到切图去重、支配剪枝和有界局部重调。** 详细文献与被比较过的候选路线见 `SHARED_MEMORY_RESEARCH_AND_INNOVATION_OPTIONS.md`；其中旧 `T3-A/C3-A` 段已标为历史档案，当前执行口径只以本文 C3-G0--G3 为准。

共享内存不能只等价为“边界 tensor bytes”。第一项方法对每条 CPU/VTA edge 和每个 VTA segment 分别建立签名：

```text
SharedEdgeSignature(e) = {
  producer/consumer device,
  shape/dtype/layout/alignment,
  physical reachability and coherence/cache mode,
  adapter or requantize requirement,
  materialized-copy versus shared-slot legality,
  framework-copy bytes and service,
  slot count, generation and live interval,
  adjacent fusion context
}

VtaMemorySignature(s, c) = {
  LOAD/STORE calls and payload,
  small-request ratio and average payload,
  R_input and R_weight,
  SRAM working-set legality,
  VTA island/re-entry and serialized service
}
```

外层切图不只改变算子集合，还改变 edge 数量、方向、融合可达范围、共享 buffer 生命周期和 CPU/VTA 对同一 DDR 的并发访问窗口。成本模型只能计一次 owner：framework materialization 属于 edge，VTA LOAD/STORE 属于 VTA stage，CPU adapter/cache maintenance 属于边界 host work，不能把 boundary bytes 再无条件加到 DDR 总量中重复收费。runtime 的 VTA DMA 描述是编译器生成的逻辑请求；在 PS APM 校验前不得称为物理 AXI burst、DDR 实际 bytes 或 compute-stall cycle。

### M-P0/M-P1：把“共享内存感知”落到可复现实验

- [x] **实验 E6-S（M-P0，完整 4623 topology 的 contract 级静态审计）。** 对每条候选边输出 live tensor bytes `B_e`、方向、shape/dtype/layout、理论交接 payload、理论 slot 数与 live interval，以及由编译 contract 能确定的 zero-copy 必要条件；计算 `M_slot(P,K)=sum_e K_e sum_t align256(B_et)`。`sum B_e` 是每边一次交接规范下的 payload，不直接等于具体 runner 的 API copy 字节；只有每边恰好一次完整 memcpy 时，`2 sum B_e` 才是相应 CPU 逻辑读写量，多次 get/set copy 须另计，不能称为普遍上界或物理 AXI traffic。4,623 个 topology 的 `预测 II--边界 payload--理论 pinned slot` Pareto 与 M0/M1/M2 各 Top-20 已对齐；全域 K2 容量正比于 payload，4 个静态前沿点不能作为安全剪枝。151 个输入/脚本来源哈希、两遍相同结果和 5 项单元测试见 [报告](shared_edge_audit/E6_STATIC_DOMAIN_AUDIT.md)。未 build 的 topology 无 Graph JSON `storage_id`、物理地址、allocator high-water 或“已经可零拷贝”的结论。
- [ ] **实验 E6-R（M-P1，编译代表的实现资格化）。** E6-S 先输出唯一 shared-edge/context 签名数 `N`。本地 build/Graph JSON/contract 检查覆盖各去重签名代表和全部差异个例；真实开发板物理 range/slot binding 只覆盖 A/B/C/D 加按风险分层预注册的代表，总量上限 8 个 package。检查 storage-id 独占性、连续性、256 B 对齐、u-dma-buf 物理可达性和 adapter/requantize；cache/coherence/端口结构只有在 XSA/HWH 或等价 provenance 可得时才作结构确认，否则只保存既有 P8 正确性证据并明确“物理路径未完全归因”。若 `N` 使本地 build 超出预算，则优先全部差异类和每层负对照并报告覆盖率；unique representative build 覆盖不足 100% 时，必须缩小并哈希 C1 冻结搜索域到已覆盖类，或把全域“编译审计支持的受限复用”主张降为采样证据，不能仍判定全域 C1 强出口，也不得外推成 4623 个二进制均已验证。
- [ ] **E6-R 初步容量判断及复核门槛。** 按四个冻结 package 的 Graph JSON `storage_id/shape/dtype` 做的只读初算中，VTA Graph storage pool 加双 slot 分别为 A/B/C/D=`10845184/11088384/11020800/15490048 B`，仅占 192 MiB u-dma-buf 的 `5.39%/5.51%/5.47%/7.69%`。正式使用前必须固化计算脚本、逐 pool/slot 明细和哈希，并用 E8 driver high-water 校验；当前结果说明 ResNet18 容量大概率不是瓶颈，不能预设“容量规划必然提高性能”。
- [ ] **实验 E7（M-P1，可选的两天 PS DDR_APM 资格化与逻辑—物理对齐）。** 先核对冻结 bitstream/设计导出的路由、计数器权限、清零/采样边界和 CPU/VTA 可归因性；DDR_APM 的 6 个 slot 对应 DDR controller 的 6 个 AXI/XPI slave data port，不等于 HPC master ID，不能假设 HPC0/1 与 slot 一一对应。必须验证 HPC/ACP 请求经 PS interconnect 后的路由、聚合与可用 ID filtering；未证明过滤时，并发测量只能称端口聚合 traffic。首轮只读或严格 snapshot/restore，不修改 VTA RTL/bitstream。按 `idle → CPU compute-only → CPU streaming → VTA-only → 真实 CPU stage+VTA segment` 做 matched run，并加入固定 tile 的相邻切点 `03..15/16/17` 与 copy/shared 对照，将 APM 观察值和软件 VTA 逻辑 DMA 计数对齐；固定 governor、绑核、温度窗口和输入。APM 若可稳定归因，只作为 C1/C3 的离线校准标签；若不可用或噪声过大，记录 feasibility negative 后停止 E7，但不自动否定 C3，也不自动转 C3-B。端口归因完成前只称“APM 观察到的共享 AXI/DDR 路径计数”，不能称“某个 VTA stage 的实际 DDR bytes”；APM latency 更不等于 compute 等待 LOAD/STORE 的 cycle。
- [ ] **实验 E8（M-P1，真实分配 high-water 与条件 slot-depth 诊断）。** 当前 AXU5EVB `VTAMemFree` 是 bump-allocator no-op，GraphExecutor zero-copy 只重绑 op 输入/输出而不释放 `SetupStorage` 已分配的 `storage_pool_`，所以现有 B2 只能声称每帧物化归零，不能声称峰值内存下降。必做项只是在现有 Eager-K2 下分别冻结 graph load、slot allocation 和 steady-state offset/high-water，并做 copy/shared 对照。pipeline-runner 单在途只有在论文冻结前已具备安全实现时才追加：必须同时设置 `max_inflight=1` 与 `K_e=1`，并修改当前 pipeline `p8_managed_slots>=2` 的校验后，才能与 `(max_inflight>=2,K_e=2)` 比 slot/high-water；若只加 credit 而仍分配双 slot，只能称 `global-inflight=1/K_e=2`，不能声称 K1 容量对照，也不得为该条件诊断挤占 W1/W2/G0。`K=3/4` 本月不做；high-water 日志有扰动，性能 run 与内存诊断分开采集。
- [ ] **实验 E9（M-P1，shared-edge 模型消融；仅 C3-B 由此触发）。** 在冻结 topology 上比较“只按 tensor bytes”“加入 `SharedEdgeSignature`”“实际 handoff mode”三种模型；若 runner 支持，再将每条合法边的 `handoff_mode ∈ {materialized_copy, shared_slot}` 作为变量，比较 all-copy、all-shared 和 mixed policy。若不支持逐 edge mixed mode，只做离线可实现性/oracle 审计，不声称该运行时功能已经完成。C3 的调度机制由 G0 触发，与 E7 是否成功及 E9 无关。

### 第三创新点：共享内存路径自争用感知 stage 启动门控，C3 条件优先

本论文不要求每个基础概念从未出现，可以在已有 shared-memory mapping、contention calibration、admission control 和异构 DAG runtime 上做面向本体系结构的组合与微调，但必须准确声明差异。最接近的先例包括保护 GPU/加速器带宽的 [BWLOCK++](https://doi.org/10.4230/LIPIcs.ECRTS.2018.19) 与 [动态内存带宽分配](https://doi.org/10.1109/TCAD.2020.3012210)、异构流水调度 [DART](https://doi.org/10.1109/RTSS46320.2019.00042)、动态异构运行时 [CEDR](https://doi.org/10.1145/3529257)、共享内存多 DNN 映射 [HaX-CoNN](https://doi.org/10.1145/3627535.3638502)、同平台内存干扰策略 [MemPol](https://doi.org/10.1109/RTAS58335.2023.00026)、ZynqMP 上的 [hardware QoS 先例](https://doi.org/10.4230/LIPIcs.ECRTS.2021.3)，以及 2026 年已经用 block conflict table 在多 DNN 嵌入式 GPU 上动态保留并行或选择性串行化的 [LAG-Guided Runtime Framework](https://doi.org/10.1145/3798107)。因此 conflict-table、选择性串行化、争用感知调度、动态线程或 QoS 控制本身都不是本文创新。本文可防守的组合差异只保留为：**同一 DNN 不同帧上的 CPU stage 与 VTA segment 都是流水关键路径上的有用工作；在四核 CPU、单物理 VTA 多 island 串行和跨 Executor K2 shared-slot 约束下，用 TVM 编译内存签名和少量真实 cell 标定决定哪些跨帧重叠应该保留、哪些刚 ready 的 stage 应该延后，并以整条流水 II 为主终点、submit-to-complete frame-latency P95 为保护指标。** 只有 G1 证明静态签名比 bytes-only/pair-independent baseline 更能预测动作，才称 compiler-informed integration；若仅 pair-ID 表有效，则降为 profile-guided、workload-specific case study。两者都不声称通用 admission 算法、通用 contention model 或跨 DNN 泛化。

暂定论文名称为：**“面向单 DNN 多帧 CPU--VTA 流水的共享内存路径自争用感知 stage 启动门控与选择性重叠调度”**。只有 E7 或等价物理证据能完成端口归因时，结果章节才进一步使用“DDR 控制器/AXI 事务”措辞。如果最终只实现离线阈值/查表，正文称“标定驱动的运行时策略选择”，不称“在线自适应”。

方法链冻结为：

```text
C1: compile_context + SharedEdgeSignature + VtaMemorySignature
                         |
isolated/forced-cell calibration + natural readiness/overlap trace
                         |
      overlap-action class / cell allow-wait table（离线冻结）
                         |
runtime: active/pending action class + actual VTA owner
                         |
  off / serialize-all / best-global-fixed / class-aware gating
                         |
      whole-pipeline II（主终点；FPS 仅报倒数）+ frame-latency P95（保护指标）
```

C1、C2、C3 的变量必须分开：C1 只在预先固定的 Eager-K2 语义下选择 topology、tile、VTA island 和名义 CPU 线程，并输出静态内存签名；C2 独占 shared-slot 的物理绑定、handoff 协议和固定 K2；C3 只消费冻结签名并决定刚 ready 的 stage 是否立即启动，不改变 topology、tile、线程、K、handoff 或 pool。thread cap、global credit 和 AFIFM 都是主结论完成后的附录/延期增强，不计入 C3 成立条件。这样第三点不是把第一点重新命名，也不是把第二点的双缓冲重复计算一次。

G0 开始前的先验信号来自 P7C 单 boot 受控实验：CPU streaming 与 DMA-heavy VTA 并发时观察到约 `1.067×` slowdown（约 `1.378 ms`），但增量大部分不在 VTA driver run 内。后续已补自然 readiness 和首个真实 pair 资格化，进度见文首；这些结果仍不能证明纯 DDR 原因或自然 K2 流水存在可恢复收益，所有第三点措辞仍由下述 gate 决定。

#### C3-G0：3--4 天机制、可达性与收益上限 gate

- [ ] **现有日志审计＋最小 readiness 时间戳（部分完成）。** boot `a5220e22` 已完成被动时间戳、shared K2 encounter 比例、overlap union/VTA graph-run、mutex/slot wait 与物理 range/generation 审计；旧 runner 仍无精确 readiness，两种 runner 均无 queue occupancy。这里 encounter 定义为“某 stage 已 active 时另一冲突 stage 到达 binding-done/ready 点”，还须按 actual VTA owner 区分是否能够立即启动。尚需同帧数 ABBA 开销资格化，以及预注册口径的 `candidate-ready critical-path duration / 全流水墙钟`；不能拿当前总 overlap 比例代替 critical-path 或可恢复收益上限。标签冻结后再单列 confirmed-protect 部分。若没有自然可达 cell，forced pair 只作机制对照，不能推动 C3。
- [ ] **预先冻结 cell，而非事后挑 pair。** 受控实验条件定义为 `(active stage, newly-ready stage, delay direction, controlled ready-state/active-age)`，但运行时动作 key 只允许保留当下可观测字段。`wait` 只能非抢占地延后刚 ready 的 stage，直到已 active stage 退出，不能利用未来信息暂停已经运行的一侧。3 个真实 CPU stage（low/mid/high static memory intensity）与 3 个真实 VTA segment（compute/mixed/DMA-heavy）及方向只依据 isolated/static 特征和自然可达性预先选择并哈希，不依据 concurrent slowdown 挑样本。每个条件比较相同输入、到达顺序和 ready-state 下的 `allow-now` 与这个唯一预定 `wait`；主终点是 two-job critical completion/makespan，单 stage slowdown 只作机制指标。
- [ ] **先实现可执行的 G0 harness。** Python 编排层与 native component 必须同时改：为 native CPU component 和 stage runner 增加逐样本 pre-run ready/start hook，尤其把 VTA hook 放进 `RunStage` 的 `set_input`/view binding 之后、`vta_run_mutex` 与实际 graph invocation 之前，并在样本完成时发 done token；只在 `run_cpu_vta_pipeline_v1_p7_profile.py` 增加文件 token 无法满足这个边界。CPU/VTA 各有独立 start/release token，支持冻结 start offset 和“active done 后 release pending”的 wait，记录 actual ready/release/start/end 与 schedule hash。cell key 只能使用部署时也能观测的字段；首先验证 `(active-stage-id,newly-ready-stage-id,direction)` 的标签在自然 active-age 分布内是否翻转，若翻转则预先加入 `elapsed-active-time bucket` 并重新 grouped validation，否则停止 pair-only MVP。pair component 没有 managed slot 时不得伪造；grant 前的 binding/adapter/cache 流量和时间单列，不能计入 opportunity ceiling。
  - 2026-09-08 已完成首个 pair 资格化；同步 token 改以 native 同进程 condition-variable 实现，Python 只在运行前冻结日程，语义仍为 binding→ready→release→run→done。本次人为延后 pending eligibility，双方 binding 在 active 启动前完成；不声称复现自然 slot/binding 压力。后续保留旧结果，扩展预注册矩阵和自然 active-age 桶，不能按此次性能重新挑 offset。
- [ ] **完整报告与固定测量协议。** 9 个预定义 pair 的可达方向均报告 discovery 结果；confirmation 集按静态特征分层预注册，不只留下正例。discovery 可用 1 boot，confirmation 做 5 个独立 boot，每个 cell 先 5 次 warmup，再至少 10 个 ABBA/随机交错 block；固定 governor、CPU/VTA host affinity、输入、bitstream/runtime 和温度窗口。无 APM 时 CPU 只报告逻辑/有效触达带宽，streaming generator 另测 achieved bandwidth；boot 是统计独立单位，帧不能伪装成独立样本。
- [ ] **三分类与机会 gate。** 标签揭盲前先用同策略 ABBA 相邻重复的绝对相对差 95th percentile 冻结 paired repeatability half-width `h_rep`。`protect` 定义为预定 wait 使 pair makespan 改善超过 `max(5%,h_rep)` 且至少 4/5 boot 同方向；`allow` 要求 boot-paired `log(T_allow/T_wait)` 的单侧 95% paired-t 上界不超过 `log(1.02)`，并明确该小样本结论依赖近似正态假设。其余均为 `unknown/abstain`，不得因“未证明 protect”而强行归入 allow。至少要有 2 个 `protect` 和 2 个 `allow` 自然可达 cell；unknown 在运行时采用预先冻结的 `best-global-fixed` 回退。
- [ ] **只计算乐观机会上限，不伪造反事实 replay。** 在冻结 off trace 上，先把 confirmed-protect encounter 的 controllable-overlap 合并为互不相交的 union `I_j`，每个 encounter 的 calibrated avoidable penalty 只归属一个 union；定义 `S_j=min(|I_j|, sum_{k in j} penalty_k)`，再令 `U=min(sum_j S_j, B_ready)/observed_wall_time`，其中 `B_ready` 是 confirmed-protect candidate-ready critical-path 区间的全局并集长度。gate boundary 之前的 slot/binding/adapter/cache work 全部扣除。该定义同时避免 overlap 与 penalty 双计，仍假设可避免 penalty 能完全转成墙钟收益，因此只是 opportunity ceiling：`U<3%` 立即止损，`U>=3%` 只允许进入 G1；只有 G1 再通过签名/查表稳定性 gate 后才允许进入 G2，最终收益只能由 G3 实测。除非未来离散事件模型在未参与拟合的 off/serialize trace 上把 II、slot wait 和 mutex wait 都复现到预注册误差内，否则不得称 trace replay/oracle。
- [ ] **内存归因 gate。** `protect/allow` 差异还必须随 CPU memory-intensity、VTA 逻辑 DMA 强度或受控 bandwidth 档呈方向一致的 dose response，而 compute-only/cache-resident 负对照不出现同量级效应。若只看到温度、host 线程、mutex 或一般计算争用，控制器即使有效也只能称通用资源重叠调度，不能使用“共享内存路径自争用感知”名称；E7/APM 可把归因进一步细化到物理 AXI/DDR 路径，但不是证明 memory-path sensitivity 的唯一手段。

#### C3-G1：离线签名、分类与无泄漏冻结

- [ ] 为每个 CPU/VTA stage 固化 occurrence id、编译内存签名、isolated service、名义线程、边界方向、逻辑 DMA bytes/calls/平均 payload/reload、可达方向和 `overlap_action_class`。比较 `(a) pair-independent majority/best-fixed`、`(b) bytes-only`、`(c) request-shape+isolated service`、`(d) 仅用训练组 APM 目标校准参数的模型`；部署和 validation 输入始终只有静态签名、方向、可选 active-age bucket 与 active/pending 状态，候选自身 APM 不能作为输入。目标是动作分类与错误代价，不为好看的 MAPE 拟合自由 DDR 时间系数。
- [ ] 分组单位是语义 CPU-stage × VTA-segment × 可达方向/完整静态签名，不是帧、tile 重复或单个 run。G0 discovery 已看过方向，因此 G1 准确称 grouped internal validation，不称完全盲测；同一 cell 的重复不能跨组。只有独立 group `>=8` 且 validation group `>=3` 才讨论特征分类泛化；不足时只能称本板 pair-table calibration。模型/阈值和两个新 topology 的选择 manifest 必须在任何 E/F 并发性能揭盲前哈希冻结。
- [ ] 只有静态签名相对 bytes-only/pair-independent baseline 在 grouped validation 中有额外预测力，才保留 `compiler-informed` 主张。若只靠 pair ID 可复现，只能继续 profile-guided、workload-specific case study，E/F 仅验证已标定 cell 在新 topology 组合中的前瞻复现；若 E/F 主要由 unseen/unknown cell 构成则停止 C3。若签名和查表都不能稳定区分 `allow/protect`，停止 G2。
- [ ] E/F 性能运行前冻结每个可达 cell 的预测、覆盖率和 fallback；unseen/unknown 一律执行 calibration 上选定的 `best-global-fixed`，不得用 E/F 结果补标或重训。最终报告 class-aware 与 best-fixed 不同的 encounter 比例及其可控关键路径时长；没有实际 treatment exposure 时，任何性能差都不能归因于门控。

#### C3-G2：最小运行时控制器与并发安全

- [ ] 在 `vta_stage_pipeline_runner.cc` 增加事件驱动 `StageOverlapController`，复用当前每 stage 常驻 worker、逐边 bounded queue 和 `vta_run_mutex`；MVP 只支持 `off`、`serialize-all`、`class-aware`。`off` 完全保持当前 eager 行为；`serialize-all` 只禁止 CPU--VTA 重叠，不禁止 CPU--CPU，单物理 VTA 仍由原 mutex 串行；`class-aware` 对刚 ready 的请求检查所有 opposite-device active instance，只要任一有向 cell=`protect` 就执行 `wait-until-no-conflicting-active`，全部为 allow 才启动，unknown/unseen 走冻结的 `best-global-fixed`。active 状态是 `{stage_id,frame_id,device_class,start_ts}` multiset，VTA 另有 actual owner；每个 active/newly-ready 组合分别查询 pair action，动作类别不缓存为 active stage 的固有字段。若 G0 要求 active-age bucket，则由 `now-start_ts` 当场计算。queue/slot wait 只作诊断，不在 MVP 决策环内。
- [ ] 先画 wait-for graph，再实现并用模型检查/压力测试验证状态机。CPU 在拿齐 slot、完成 view binding 后登记 pending，CV wait 时释放 controller mutex，grant 时原子执行 `pending--` 并插入 active instance；VTA 先登记 pending 并无锁等待候选 grant，再取得 `vta_run_mutex`，随后在 controller lock 下原子 revalidate，允许才更新 pending/active 并运行，不允许则释放 VTA mutex 后重试。任何可能阻塞的 admission wait 都不得持有 controller mutex 或 VTA mutex；mutex/admission wait 都不得计为 active。等待期间已持有的 incoming/outgoing slot 必须进入 wait-for graph 并单独记时，证明 active stage 完成路径不再等待该资源。每个 device class 使用严格 FIFO ticket；head request 因 protect 被拒后启动 drain rule，禁止新的冲突 opposite-device request 继续旁路，直到 blocking active 清空。所有退出用 RAII 删除 active/`notify_all`，worker 错误同时 abort queue、slot manager 和 controller。若 G2 第 2 日仍不能证明 worker gate 安全，停止独立 C3；central dispatcher 记为未来工作并重新估时，不能暗中扩大 4--6 日预算。
- [ ] 输出每帧 active/pending、决策/reason、scheduler wait、真实 cell overlap、stage service、VTA mutex wait、slot wait 和实际 CPU 线程；若新增 queue 诊断，显式记录 `push_wait/pop_wait/size_before/after/event_ms`。事件先写内存缓冲，性能 run 禁止逐事件 flush。`queue_depth` 是逐边容量，不得冒充 global frames-in-flight。
- [ ] 新控制器必须补多 VTA island、交替 `protect/allow`、随机 stage delay、异常退出和至少 1000 帧逐帧唯一输入压力测试；断言结果/顺序与 `off` 一致、无死锁/乱序/永久 pending，报告最大 scheduler wait 和公平性。控制器开销是单独 sanity check：用“新 binary + policy-off”与旧未修改 binary 做 5-boot 配对，`|ΔII|` 中位目标 `<1%` 且每 boot `<2%`，逐 boot 公开；旧 binary 不进入 G3 主推断，未指定更多样本的 TOST/bootstrap 前不称“统计上界”。
- [ ] G2 核心不实现动态线程、global credit 或 AFIFM；它们只在 G3 已成立且截止顺延时作为附录增强，不能拿来补救 core gate 失败。

#### C3-G3：冻结策略的自然流水 held-out 验证

- [ ] 基线固定为：`Single-frame-serial`、现有 `Eager-K2/off`、`serialize-all` 和本文 `class-aware`。`best-global-fixed` 不是额外策略，而是在冻结 calibration 集 A/B/C/D 上按 topology 等权的相对饱和 II 指标，从 `off/serialize-all` 选一次的预声明主对照；不能按 pair makespan 或 opportunity ceiling 选择，且原样用于所有 held-out。E/F 中仍完整报告 off/serialize-all 两条及 post-hoc fixed regret，但不得重选主对照。小型冻结动作池的 measurement oracle 只作事后上界/regret，不参与主推断。G3 的 off/serialize-all/class-aware 及 `d_b` 全部使用同一份新控制器 binary、同一 topology/tile、输入、C2 K2、线程和频率配置；旧 binary 只用于 G2 policy-off 开销 sanity check。
- [ ] A/B/C/D 已用于机制发现，不能称完全前瞻。必须在测任何并发性能前只依据静态 manifest 选定并哈希两个新 topology E、F；二者都不得参与 cell 阈值、best-fixed 或 policy 选择，各保留一个饱和主 cell。若最终只有一个新 topology，只能称 held-out context case study，不声称 topology 泛化；即使 E/F 均通过，也只声称 ResNet18 不同切图间的前瞻复现，不声称跨 DNN 泛化。
- [ ] 每个 policy×topology×boot 至少 300 个 steady frames、5 个独立 boot，顺序随机或 Latin-square；同一个 `boot_id` 内必须同时完成 E/F 与 fixed/C3 的配对 block，才能进入 `d_b`。帧只用于估计该 boot 的 II 与分位数，不能当独立样本。报告 boot-level paired effect、描述性区间和全部失败/回退；若资源只够 3 boot，只能写探索性 case study。
- [ ] 本月预注册唯一主分析为 E、F 两个 topology 等权的相对饱和 II 效应。对 boot `b` 定义 `d_b=0.5*[log(II_fixed,E/II_C3,E)+log(II_fixed,F/II_C3,F)]`，正值表示 C3 更快；成立门槛为 `exp(median_b(d_b))-1 >=3%`、至少 4/5 boot 的 `d_b>0`，且 E/F 任一 topology 的 boot-median II 不回退超过 2%。5 boot 的区间只作描述，不声称统计显著；若论文坚持正式 95% CI/非劣推断，至少扩到 10 个独立 boot，并在揭盲前冻结 cluster-at-boot 方法。
- [ ] 安全指标固定为 `P95 submit-to-complete frame latency = final_complete_ts - submit_ts`。`submit_ts` 使用现有 queue0 `Push` 前记录的 `enqueue_ms`，因此明确包含 queue0 push/admission wait；它不是 P95 completion interval，后者只能另列为输出 jitter。若另报 admitted-to-complete latency，必须新增并单独命名 `queue0_push_success_ts`，不能用 `admit/enqueue` 斜杠混写。对每 boot 先算 E/F 等权的 P95 latency log-ratio，5-boot 配对中位退化不得超过揭盲前由 baseline ABBA 冻结的 `max(2%,repeatability margin)`，逐 boot 公开。arrival/deadline 留作未来工作。`class-aware` 还必须在 E/F 暴露到非零 treatment，实际使用 allow/protect 两种动作而不退化为常量；不能在 FPS、II、P95 间事后择优，也不能用训练 cell 准确率或 synthetic streaming 代替。

#### C3-GQ：延期 AFIFM QoS/issue 执行器

本设计脚本显示 load/store 接到 HPC0、compute data 接到 HPC1、fetch/uop 接到 ACP，但冻结 bitstream 的真实端口 provenance 仍须由 XSA/HWH 或等价构建产物确认。[AMD AFIFM 文档](https://docs.amd.com/r/en-US/ug1087-zynq-ultrascale-registers/AFIFM-Module)提供的是平台机制，不构成论文创新。

- [ ] 先做 read-only inventory，再对 AFIFM0/1 使用 snapshot、masked write、readback、restore 和异常恢复。相应读/写通道的 `RDCTRL/WRCTRL.FABRIC_QOS_EN=0` 时使用 `RDQoS/WRQoS` 的静态 APB 配置值；为 1 时静态值被忽略，QoS 来自 fabric 的 `axds_*QoS`，该位是逐通道选择而非“全局开关”。若读到 1，本月跳过对应静态 QoS sweep，不修改选择位。`RDISSUE/WRISSUE` 只限制 outstanding command，不是带宽配额、隔离或预留。下游 DDR traffic-class 映射也会影响结果，禁止触碰全局 DDRC throttle/urgent。AFIFM0/1 不控制走 ACP 的 fetch/uop。开发板必须独占并预先验证恢复路径。
- [ ] 2--4 日 gate 只承诺整次 benchmark 固定 profile sweep：default、HPC0/HPC1 issue 单轴和极少数 QoS profile。只有 G3 已完成、截止顺延，并额外预留 2--3 日证明端口 quiescence、切换开销和并发安全，才允许 per-stage/context switching；VTA 路径顺序固定为 slots/binding→pending/candidate grant→`vta_run_mutex`→controller revalidate/final grant→AFIFM set/readback→run→quiescence→restore→active release。
- [ ] C3-GQ 的成立条件不是“寄存器可写”，而是至少两个真实 context 各自在 `>=4/5` boot 稳定选择不同的最佳安全 profile，且冻结 selector 优于 calibration 选出的最佳单一全局 profile、形成 CPU/VTA 非支配 Pareto；否则只报告 whole-run 固定 AFIFM 敏感性。它不单列创新点，也不能挤占 G3。

#### C3 结果解释与工作量上限

2026-09-08 执行决策：自然配对普查 57 个成功复现条件均无保护候选，3 个条件未复现，故**当前不进入 G1/G2/G3**，先收口 C1/C2。本月预算不因“需要第三点”而自动投入完整控制器；若重启 C3，应先为未复现的并发背景或新机制对照写出有依据、事前冻结的恢复条件。该决策不将未测项标完成、不将未知 opportunity ceiling 写成 0，也不声称所有自然共享内存调度均无效。

- G0 没有自然可达 cell、没有 `protect/allow` 异质性、opportunity ceiling `<3%` 或只有不可归因噪声：不实现控制器，第三点不成立；把负结果用于解释固定 Eager-K2 为何足够。
- 只在 synthetic streaming 下成立：写成共享内存压力鲁棒性增强，不作为自然工作负载第三创新点。
- G0/G3 通过但 G1 只有 pair-ID 查表、静态签名不优于 bytes-only/pair-independent baseline：只能写成 profile-guided、workload-specific stage-gating case study，不使用独立 compiler-informed C3 名称。
- G1 模型改善但 G3 不改变决策/端到端终点：默认只称平台机制表征；只有代理实际接入 C1、重跑 M1/M2/E5a 统一消融（E5b 仍只由既定 tile-rank gate 触发）并改善排序或候选效率后，才称 C1 的板级校准层。
- G1 静态签名 gate 与 G3 预注册 held-out 终点同时通过：才使用上述独立 C3 名称；若延期后再通过 C3-GQ，只把 QoS 作为执行器增强，不拆第四点。

Stage-gating-only 的独立增量现实预算为 11--16 个工作日：G0 3--4 日、G1 1--2 日、G2 4--6 日、E/F build 与 G3 3--4 日，板端失败恢复另计。C3-GQ 静态 gate 另需 2--4 日、per-stage switching 再需 2--3 日，因此本月默认不启动。在线连续读取 APM、强化学习、动态线程/credit/QoS 和跨网络泛化均不在本月范围。APM latency 与 stage slowdown 最多说明共享内存/AXI 服务相关变化，没有 VTA stall counter 时绝不能写成精确 compute-stall cycle。

#### C3-B（备选）：线性流水的逐边 handoff 与槽深规划

只有论文截止整体顺延、C3-G0 因缺乏自然干扰异质性/重叠机会而停止、E8 又证明存在真实容量或 slot 热点，并且能另留 14--20 个工作日，才重新评估把 E8/E9 扩成 C3-B；APM 不可用本身不是触发器。范围只承诺当前线性、相邻 CPU/VTA 交替的 stage 链：由边界字节、stage/copy service、单 VTA 冲突、slot wait 和内存预算选择每条 edge 的 `copy/shared`、`K_e∈{1,2}` 与 `pool-anchor/external` 分配来源，`K=3` 只作可选对照；不承诺通用 DAG/fan-out arena。当前 runner 已有统一 all-shared K2、FSM、generation、bundle 和 wait counter，但真正 pipeline K1、逐边 mode/K、mixed path、planner、storage-id 独占性审计和 high-water 仍需实现或资格化，不能把已有串行 B1 写成 pipeline K1。

C3-B 最值得实现的微调不是再造一般内存池，而是安全借用 GraphExecutor 已经分配的 input/output storage pool 作为 `slot0`：若某个边界 storage-id 独占、物理范围与生命周期均通过审计，双槽时只额外分配一个 external slot；若 storage-id 被内部临时张量复用，则保持全外部分配。当前 rank-1 静态初算中，借用 CPU→VTA input pool 可使两条边的额外双槽由 `1,806,336 B` 降到 `1,003,520 B`（`-44.44%`）；三 VTA-island 拓扑可由 `9,031,680 B` 降到 `7,024,640 B`（`-22.22%`）。这些数字只是假设审计通过后的静态上限，必须用 driver high-water 与正确性压力测试复核。

C3-B 只有在至少四个 topology、5 个 boot 上实现逐边计划，并相对固定 external-K2 的 II 退化不超过 2% 时再降低至少 20% slot bytes/high-water，或在相同容量预算下使 II 改善至少 3%，才能独立写成“**面向单物理 VTA 多帧线性流水的 storage-pool 复用与关键路径感知共享边缓冲规划**”。小空间全部组合的 planner regret 目标为 `<3%`。若只完成两个 topology/3 boot，只能称 case study，不声称一般 planner 精度。只实现全局 K 切换仍归入第二创新点。自然 Top-20 的 all-shared FPS 尚无正收益，且当前 ResNet18 pool 占用低，因此 C3-B 是延期研究备选而不是默认主线；若采用它，第一创新点不再声称联合决定每边 handoff/K，避免贡献重叠。

- [ ] **不立项方向。** 不把共享虚拟地址本身、TVM zero-copy API、双缓冲概念、HPC 端口、u-dma-buf 驱动、cache flush API、一般 buffer lifetime 或一般 DDR contention model 当成新创新；不在一个月内实现硬件预取/预驱逐、跨卷积 SRAM 常驻、IOMMU/SVM 或新 cache-coherent interconnect。VTA 已有显式 LOAD/compute/STORE task 和依赖队列；在本项目冻结 commit 的 `conv2d_packed` template 中，TOPI schedule 决定 DMA/cache/tensorize 结构，AutoTVM 通过该 template 暴露的 tile/virtual-thread knob 间接改变请求与重叠，但这不是 AutoTVM 的普遍能力上限。真正跨算子 SRAM 常驻需要改 fusion/lowering/存储规划，且已有直接先例，本月只做文献边界和未来工作。

## 剩余周期末：冻结与答辩材料

- [ ] 完成 E0--E2、E5a、E6-S/E6-R；按 E1 结果执行或书面跳过 E3/E4，只有 E4 出现稳定 `>=2%` 排名翻转才执行 E5b。C3 本月优先完成 G0 discovery；完整 G0 confirmation 及 G1--G3 服从前述剩余日 gate。E7/APM 是最多两天的可选物理标定，不决定 C3 生死，只在 E3/E4 未触发且有余量时执行；二者冲突时 E7 无条件让位，不能替代 H6。E8 必做现有 Eager-K2 的 copy/shared high-water；pipeline `(max_inflight=1,K_e=1)` 只在冻结前已有安全实现时作条件诊断，不临时立项。C3-GQ/C3-B 本月默认不启动，最后 4--5 个工作日不得再增加功能。
- [ ] 停止增加功能，冻结代码、配置、bitstream、runtime 和原始日志哈希。
- [ ] 更新论文相关工作、方法、实验、局限性和威胁；加入 HaX-CoNN/AxoNN/CoDL/FPGA memory-management 对照，统一“共享物理地址、framework copy、runtime DMA 请求、物理 DDR/AXI traffic、compute stall”的五层口径。
- [ ] 生成 stage–tile 流程图、`SharedEdgeSignature + VtaMemorySignature` 图、DMA 碎片对照图、TopHub incumbent 消融表、双 slot 状态机图；若 C3 成立，再增加“自然 overlap→cell calibration→stage gating→E/F held-out”图，未成立则改为机制否证图。
- [ ] 准备答辩问题：为何不做完整联合搜索、为何零拷贝未提高自然 Top-20 FPS、为何 runtime profile 不能等同 compute stall。
- [ ] 准备 H2 的双分支解释：若 DMA 可加，为何 stage 仍会通过共享边界、共享内存路径并发窗口和流水平衡产生系统效应；只有 E7/等价证据通过后才把其中一部分进一步归因为物理 DDR 竞争。若不可加，则说明 stage correction 的可观察来源和适用范围，不臆测未测硬件机制。

## 明确停止项

本月不修改 VTA 硬件，不增加 stall counter，不实现跨算子 SRAM 常驻、硬件级预取/预驱逐、IOMMU/SVM 或新一致性互连，不扩展完整 97 万切图×AutoTVM 空间，也不以超过 TopHub 或多网络普适 tile 定律作为毕业门槛。不实现在线连续 APM、强化学习、动态线程、global credit、动态 QoS 或 arrival/deadline generator。C3-GQ、C3-B、E8 的 K3/K4 与真正 K1 改造、C-P2 扩展 tile 搜索和额外网络均默认停止。E1 差异触发的 E3/E4 属于 C1 必须闭合的条件支路，E5b 只在 E4 稳定翻转 gate 通过后加入；E7 只能在 E3/E4 未触发且有余量时执行，不能替代 H6。必须保留 E0--E2、E6-S/E6-R、原实验 A--D、内存感知模型消融、强基线多 boot 验证和已有双 slot 证据，以及至少 C3-G0 discovery 的明确 go/no-go；若预算允许进入 confirmation，则按冻结协议做完，不能只挑正例。若 G0 通过且已经正式开始 C3-G2，则必须完成 policy-off 安全/开销、1000 帧/异常压力测试和 G3 的预注册 E/F 结果，不能只展示 calibration 正例；如果来不及完成 G3，应停止声称独立第三创新点。
