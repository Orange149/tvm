# 第三创新点定稿方向：面向固定 FPGA 的共享内存访问规律驱动 AutoTVM 搜索优化

状态：`MECHANISM_SEARCH_DEPLOYMENT_CHAIN_COMPLETE; LITERATURE_BASELINES_AND_R18_CANDIDATES_FROZEN; NODE_C_LOCAL_QUALIFICATION_PENDING`

> P7R469--P7R470 已完成文献对齐节点 B：实现六策略统一 runner、HW-Aware 四级初始化消融、
> ML²Tuner P/V/A 与 A+DMA 消融，以及 Cheng 四方案。新增 Scheme 4 功能级实现已在三个历史代表
> 配置上通过三 seed FSim；三个新 ResNet18 几何的完整 original 域和 24 tile×4 mode proposal
> identity 已在任何目标 lowering/FPGA/latency 前冻结。真实 ConfigSpace 并非计划假设的统一 1280，
> 而是 H1/H2/H3 分别 2304/1600/480；完整空间实验将使用这些真实基数。当前没有新 R18 性能标签，
> 不能提前写任何基线收益。详见 `20260914_P7R469_P7R470_LITERATURE_BASELINES_FREEZE.md`。

> P7R462--P7R468 已完成 Y10 同空间在线选择器归因消融。DMA prior 与 mode-aware XGB 各运行
> 三个 seed，均使用相同 12 点 `tile×mode` 池、budget=6、完整 YOLO 图构建/正确性/七轮计时路径。
> 两者三次均在第 1 次派发进入完整池 oracle+2%；DMA prior 三次均于第 3 次命中 exact oracle，
> XGB 为第 4/6/6 次。exact time-to-quality 中位数为 212.564 s 对 396.797 s，DMA prior 减少
> 46.43%、提前 1.87×。由于两边均执行满六点，总墙钟中位数 400.933 s 对 398.963 s，不能声称
> 固定终止预算的总成本下降。Y10 标签在实验前已存在，因此这是选择期间不读标签的同空间真实成本
> 重演与归因消融，不是 prospective holdout。详见
> `20260914_P7R462_P7R468_Y10_SAME_SPACE_ATTRIBUTION.md`。

> P7R452--P7R460 已补上 P7R445 发现的主缺口：在 Y10 上完成空 history AutoTVM-XGB 与本文流程
> 三对三 clean-start T0→T1 实验。三轮中位总时间为 696.427 s 对 244.844 s（-64.84%，2.84×），
> 本文最早在 120.279 s 获得落入 XGB 最终整图 +2% 带的已验证完整图。最终整图中位为
> 1054.773 ms 对 1057.774 ms，本文仅慢 0.285%；两边均每轮三输入/八输出正确、7/7 配对获胜。
> 这是搜索空间和测量粒度不同的系统流程比较，Y10 也是身份精确的已暴露 workload 成本重演；
> 同空间算法归因现已由 P7R462--P7R468 单独闭合；另一网络 T0→T1 仍未完成，因此 64.84% 仍只能
> 作为不同空间的完整系统流程收益，不能全归因于搜索器或外推泛化。
> 协议与偏差说明见 `20260913_P7R445_FROM_SCRATCH_AUTOTUNE_AB_PROTOCOL.md`。

> P7R444 更新：对 Y10 完整池最终 oracle 的 exact YOLO-320 图实测得到单次提交 instruction/UOP
> 峰值 1,065,280/1,320 B，页对齐定容为 1,069,056+4,096 B。三 seed、全部 8 个输出在缩容后
> 精确匹配；instruction 再少一页时，前 5 次合法提交完成，下一条超限命令在送入 FPGA 前被拒绝。
> 命令 backing 相对默认 64 MiB 减少 98.4009%，随后默认 RPC 与完整图健康推理恢复。该数值只对
> 绑定的静态图/二进制/bitstream 成立，也直接证明旧 Y00 的 155,648+4,096 B 不能跨 allowlist
> 套用。详见 `20260913_P7R444_Y10_SELECTED_FULLGRAPH_CAPACITY.md`。

> P7R425--P7R443 更新：首次将同一 cheap-proxy/peeling 在线流程和三条独立控制同步扩展到
> 合同绑定的 Darknet YOLOv3-tiny-320 完整图。Y09 conv8 的 24 个身份仅 2 个通过本地资格、没有
> 完整三模式 family，按预注册规则不上板；Y10 conv18 形成 12 点、4 个完整 family，12/12 均通过
> 三 seed、8 输出 FPGA 正确性和七轮交错计时。peeling 以 3/12 命中 exact oracle，候选 build、
> FPGA invocation 与逻辑 DMA 相对穷举约减少 75%--76%；但 calls-lazy 和本次单个 frozen Random
> 均第 1 次 exact，peeling/bytes 到第 3 次，故本轮支持跨模型完整图闭环，不支持全面基线优势。
> 四个 input 驻留全图配对均略快；weight barrier 虽将算子 bytes 降 34.13%--92.92%，仍出现一组
> 全图慢 0.115%，再次证明最少访问不等于最快。详见
> `20260913_P7R425_P7R443_YOLO320_HOLDOUT.md`。

> P7R420--P7R424 更新：R50I 的 calls/bytes/Random 冻结顺序已各用独立进程执行两候选预算，
> 不再只依靠全池 replay 估计成本。peeling、calls、bytes、Random 的完整外层墙钟分别为
> 213.527/212.812/214.571/199.165 s；前两者命中 exact oracle，后两者未进入 oracle+2%。
> calls 第一点即命中，time-to-oracle 146.179 s，明确优于 peeling 的 213.311 s；Random 更短是因
> 第二点 FPGA 错误后跳过计时，不能作为同质量优势。详见
> `20260913_P7R409_P7R419_R50I_AND_THREE_HOLDOUT_AGGREGATE.md`（已扩展至 P7R424）。

> P7R409--P7R419 更新：第二个源模型一致的 stride-2 层 R50I
> `stage3_unit1_conv1`（CI512→CO256、28x28）在标签前冻结 5 点合法池和两点前沿；两点均正确，
> 在线按规则在 213.527 s 停止，事后完整池为 4/5 FPGA-correct，且在线第二点就是 exact oracle。
> 相对穷举实测组件，dispatch -60%、完整候选 build action -63.41%、逻辑 DMA bytes -52.22%。
> R50G/H/I 三个前瞻 cheap-proxy/peeling 留出共 25 点、在线派发 7 点（-72%），3/3 达到
> oracle+2%、2/3 exact；只有 R50H 初始前沿含错误支配点并真实展开。详见
> `20260913_P7R409_P7R419_R50I_AND_THREE_HOLDOUT_AGGREGATE.md`。

> P7R399--P7R408 更新：在模型源自动核验后，新冻结 R50H
> `stage4_unit1_conv1`（CI1024→CO512、14x14、1x1/stride-2）形成 15 点 lowering/FSim 合法池。
> 首波三轴算子代理前沿的最低 DMA weight-barrier 在第二 seed 出现整网输出错误；在线搜索器将其
> fail closed 后重新计算前沿，第二波唯一新增点正是 14 点 FPGA-correct 完整池的精确 oracle。
> 在线全过程构建并派发 4/15 点、外层墙钟 312.942 s；相对穷举的实测组件，candidate dispatch、
> FPGA kernel invocation、逻辑 DMA bytes/calls 分别减少 73.33%、77.46%、78.43%、78.12%。
> 但冻结的 bytes-lazy 顺序只需 2 次便命中同一 oracle，故本轮证明的是错误支配点回退与可靠性闭环，
> 不是全面优于简单字节排序。详见 `20260913_P7R399_P7R408_R50H_ONLINE_PEELING.md`。

> P7R385--P7R398 更新：首次 R50F 冻结把 `relay.testing.resnet` 的
> `stage2_unit1_conv2` 错写成 3x3/stride-2；P7R388 在上板前通过 Relay InferType 发现该层实际为
> 28x28/stride-1 且撞上历史 W02，故原合同及本地资格全部撤销，不产生板端标签。重新以真实未见的
> `stage2_unit1_conv1`（CI256→CO128、56x56、1x1/stride-2）冻结为 R50G 后，5 个合法候选全部
> FPGA-correct。预注册单点 cheap proxy/invalid-dominator peeling 首测已进入完整池 oracle+2%，
> 但漏掉精确 oracle，paired-ratio regret 为 0.714%；本池没有 invalid 点，因此没有验证 peeling
> 展开本身。详见 `20260913_P7R385_P7R398_R50G_PEELING_HOLDOUT.md`。

> P7R367 离线更新：驻留生命周期感知的 weight/ACC/input 容量预筛已实现。12个历史池
> 288个身份回放识别30个容量失败、保留149个static-ok；Y08/Y07/R50B真实lowering对照
> 分别将尝试24→13/14/18，成功TIR集合保持一致。这仍是开发池编译成本证据，不是新
> FPGA或完整AutoTune加速结果。Y08六候选与正式搜索顺序已冻结，板端网络恢复待完成。
> 详见 `20260913_P7R367_RESIDENCY_PRESCREEN.md`。

> P7R352 已完成一次准备、两个候选逐个构建/上板的真实执行链，211.937 s 内两个候选均通过。
> 序列化模型往返、参数语义与最终程序 DMA 均验证；此项是主机调优执行优化，尚非新的搜索
> holdout 或 u-dma-buf 复用机制。下一未见选择必须补上后续合同的几何查重，旧固定清单不充分。
> 详见 `20260913_P7R352_SHARED_PREPARATION.md`。

> P7R351 已用已知 R50E 身份完成真实预算执行链：重新构建并三 seed/七轮上板验证，共 144.169 s。
> 内部构建与调用计时仅合计 70.474 s，外层另覆盖 73.694 s，直接证明不能把阶段和当完整墙钟。
> 这是执行器集成验证；既有算子资格在预算外，不增加搜索 holdout。下一步是一次性共同准备和
> 未见多候选预算实验。见 `20260913_P7R351_REAL_BUDGET_EXECUTOR.md`。

> P7R349 成本审计补充：R50E 的 77.39% 是模型构建阶段节省，73.60% 是 static/FSim/build
> 已计时三阶段和节省，尚非完整进程墙钟。按相同 pool-oracle 目标回放，原前沿为 134.684 s，
> bytes 按需构建 151.012 s，calls 按需构建 99.155 s；首点命中受哈希顺序影响。前沿保留 oracle
> 的主张仍成立，但尚不能主张全面优于简单搜索。详见 `20260913_P7R349_EQUAL_TARGET_COST_AUDIT.md`。

状态：`FIFTEEN_LABEL_ISOLATED_FPGA_POOLS_COMPLETE; R50I_THREE_INDEPENDENT_FROZEN_CONTROL_PROCESSES_COMPLETE; THREE_PROSPECTIVE_CHEAP_PROXY_PEELING_HOLDOUTS_3_OF_3_ORACLE_2PCT_2_OF_3_EXACT; R50H_ONLINE_INVALID_DOMINATOR_PEELING_EXACT_ORACLE_COMPLETE; R50G_INVALID_PEELING_ORACLE_2PCT_PASS_EXACT_ORACLE_MISS; Y08_YOLOV3_TINY_CONV12_FULL_POOL_COMPLETE; FIVE_STRICT_ADAPTIVE_HOLDOUTS_COMPLETE; R50C_OPERATOR_TO_FULLGRAPH_HOLDOUT_CHAIN_CROSS_BOOT_VERIFIED; PROSPECTIVE_FUSED_SERVICE_PROXY_FAILURE_AUDITED; R50D_PARETO_ORACLE_2PCT_CROSS_BOOT_STABLE_EXACT_ORACLE_NOT_STABLE; R50E_PROSPECTIVE_OPERATOR_PROXY_BUILD_AND_FPGA_COST_REDUCTION_CONFIRMED; EXACT_RELAY_RESIDENCY_DISPATCH_COMPLETE; RESNET50_AND_YOLO_FULL_GRAPH_TRANSFER_COMPLETE; GRAPH_NODE_ROUTE_ISOLATION_COMPLETE; FUSED_TIR_GRAPH_DMA_MODEL_COMPLETE; GROUPED_ADMISSION_TWO_CALLSITE_LATENCY_HOLDOUTS_COMPLETE; FULL_GRAPH_TILE_DMA_FRAGMENTATION_BOUNDARY_CONFIRMED; YOLO_TWO_ROUTE_FACTORIAL_COMPLETE; INCUMBENT_PROTECTED_ROUTE_PLANNER_COMPLETE; EXACT_GRAPH_ROUTE_MANIFEST_COMPLETE; UDMABUF_ALLOCATOR_AUDITED; WHOLE_GRAPH_DEPLOYMENT_FAIL_CLOSED_COMPLETE`  
目标：硕士论文主创新，证据质量以 CCF-C 水平为目标；研究对象是搜索成本与搜索可靠性，
不是超过 TopHub 的最终 latency。

## 1. 一句话故事

> 通用 AutoTVM 面对固定 FPGA 时，仍会把大量违反硬件容量、张量化约束或产生高重复 DMA 的
> 配置送去编译和上板。本文先复现输入优先、权重片上复用和最小访存选择三类规律，再把 tile
> 对 u-dma-buf 数据流量、DMA 请求数、片上容量和真实 FPGA 正确性的影响编码为逐级可见的证据；
> 生存率自适应搜索器同时决定“下一个候选”和“下一种验证保真度”；最终融合程序若在 bytes/calls/
> submissions 上联合支配则淘汰被支配者，目标冲突则升级到 FPGA 测量；廉价算子级前沿先决定
> 哪些候选值得构建完整图，并始终以多 seed 正确性和真实 latency 关门。选中 route 再绑定图节点、物理布局与
> exact-allowlist 命令容量。目标是在保持最终配置质量的前提下，减少达到正确优解所需的候选、
> 编译/仿真/上板墙钟和搜索期间的共享内存访问，并保证搜索结果能够安全部署。

这里 u-dma-buf 是共享物理内存载体；tile、loop order 和驻留不是偏题，而是决定 u-dma-buf 被访问
多少次、每次多大、命令描述符多少的上游因素。TopHub 只承担两个角色：对已有精确硬件记录提供
事后质量参照，以及在部署时作为经过当前硬件正确性资格的回退配置。它不属于在线搜索器的输入，
也不是本文必须击败的对手；若精确匹配的 TopHub 已经可直接使用，本来就没有重新调优的必要。

## 2. 统一模型

对候选 `x`，外部共享内存访问量不是张量唯一大小，而是 tile 引起的重复加载：

```math
B_{data}(x)=\sum_{t\in\{inp,wgt,out\}} B_{unique,t}\,R_t(x),
```

其中 `R_t` 是由循环顺序、tile 和片上生命周期共同决定的 reload ratio。请求时间近似为：

```math
T_{shared}(x)\approx N_{dma}(x)L_{req}
 +\sum_s\frac{B_{dma,s}(x)}{BW_{eff}(s)}+T_{cache}(x),
```

`s` 表示连续、跨距、padding 等请求形态。因此“bytes 少”并不必然更快；请求数、请求形态、同步和
计算并行度都可能抵消收益。端到端算子时间写为：

```math
T(x)\approx \max\{T_{shared}(x),T_{compute}(x),T_{store}(x)\}
 +T_{sync}(x)+T_{launch}(x).
```

候选还必须满足片上容量：

```math
F_m(x,H)\le C_m(H),\qquad m\in\{input,weight,acc,uop\}.
```

若各区域由 runtime 独立保留，最终部署的 u-dma-buf **reserved backing** 分解为：

```math
M_{reserved}=M_{tensor/slot,reserved}
 +C_{insn}(A)+C_{uop}(A)+M_{other,reserved},
```

```math
C_q(A)=align\left(\max_{x\in A,submit} B_q(x,submit)\right),
```

其中 `A` 是调优结束后的 exact allowlist。第二创新点已经减少
`M_tensor/slot`；第三创新点研究 `B_data/N_dma` 和 `C_insn/C_uop`，两者在同一 u-dma-buf 中但
不混为同一指标。上式不是实际同时存活峰值；在缺少完整 birth/death、FINISH 与 replay 冲突图时，
各局部 peak 的求和只能报告为 naive-sum 上界。

## 3. 方法主次关系

### 3.1 机制复现：扩展可搜索的共享内存访问方式

依据 Cheng 2026 全文核对并实现三种数据复用规律：

- input-prioritized：扩大 input tile 的片上生命周期，减少跨 output-channel tile 的重复读取；
- on-chip weight reuse：论文通过不覆盖的 SRAM 地址、条件 LOAD WGT、TIR passes 和 runtime
  `PushGEMMOp` 联合实现；本地当前用显式 drain/barrier 扩大 weight tile 生命周期，是相同目标的
  替代机制，并非论文源码级复现；
- minimum-access selection：在 original、input-prioritized 和 weight-resident-barrier 之间选择，按
  完整 input/weight/output DMA 与同步代价排序，而不是把两种驻留强行写成一个“同时驻留”模式。

这三条规律用于构造比原始 VTA 模板更有意义的搜索空间。它们属于机制复现基线，不能仅凭“课题于
2025 年开始”而把 2026 年已经公开的结论写成首次提出。现在已有完整论文，可引用 Algorithm 1/2、
Fig. 1--3 和 Table 1/2；但作者源码和 TVM commit 仍未公开，因此本地实现只能称功能级独立重实现，
不能写逐行或 code-level exact reproduction。

### 3.2 核心创新：固定硬件知识驱动的分层调优

```text
ConfigEntity + residence mode（时刻 0 只见便宜的 knob/SRAM 粗特征）
 -> 按需付费 lowering，成功后才揭示精确 DMA bytes/calls/request-shape
 -> 共享内存服务代价 S=B+65536*N_dma+131072*max(N_submit-1,0)
 -> 按需付费三 seed FSim，成功后才揭示命令峰值/submission
 -> 按需交叉编译与 FPGA correctness（首错即停，通过必须三 seed）
 -> 只有 FPGA-correct 候选才测 latency
 -> 根据已观察的候选存活率，自适应选择固定前沿或 same-tile family wave
```

其中两个系数是由开发池冻结的“等效字节”搜索权重，用来表达请求启动和额外提交的服务代价；它们
不是物理 AXI 字节，也不是直接测得的带宽或延迟常数。Y04 已经否定单独按 `B` 排序的跨几何泛化，
而 P7R324--P7R327 又以前瞻冲突对否定固定 `64 KiB/request` 系数的跨 tile 泛化。因此正式规则保留
`B_dma/N_dma/N_submit` 的多目标关系：联合支配时才静态淘汰，发生权衡时升级保真度，同时继续由
FPGA correctness gate 隔离数值错误。

早期 Y00/Y03 实验先对整个冻结池完成公共 lowering/FSim，再离线排序，能够验证 DMA 先验，但不能
证明在线减少资格成本。P7R173 以后将候选改成逐阶段部分可见状态：未 lowering 时不得偷看精确 DMA，
未 FSim 时不得偷看命令签名，未上板时不得偷看正确性和 latency。冻结的生存率自适应策略先探测
一个硬件多样 family 的三个 mode；若 lowering 通过数不超过 1，则判断当前空间稀疏，逐 family
立即晋级最小服务代价候选，否则保留 4→2 前沿。ML²Tuner 启发的 Model V 作为对照保留；它在开发
实验中未稳定占优，因此不强行进入主策略。

算子级 lowering 仍可能遗漏 Relay fusion 和同一 workload 在图中的真实出现次数，因此少数晋级 tile
在首次整图 FPGA 调用前增加一个更高保真动作：构建最终 fused TIR，按 Graph JSON occurrence 聚合
`B_dma/N_dma`。P7R314--P7R315 没有重拟合 P7R166 的 64 KiB/request 系数；在标签已暴露的
R50AF00/F01 上，它把 bytes-only 的 0/2 修正为 2/2，四程序两两次序由 2/6 改善到 5/6。随后
P7R316--P7R324 在不知道 R50C 两个新 tile 整图 latency 的条件下冻结冲突对：F01 为
`38.10 MB/980 calls`，F02 为 `6.52 MB/2900 calls`，固定服务系数选择 F01。P7R326 的 6 次
correctness 与 14 次 timing 全等非零、十字段 DMA 精确命中，但真实 latency 为
`375.140/350.550 ms`，固定代理以 7.015% regret、0/7 胜失败，bytes-only 在本对反而选对。
因此最终方法不再把固定标量当作 latency 预言器；final fusion 是获得精确多目标坐标的高保真动作，
bytes/calls 冲突点必须保留到更高保真度测量。

失败候选仍消耗一次 gross candidate dispatch，但不会污染性能回归标签。对同一候选的多 seed 硬件
认证采用 fail-fast：任何 seed 错误立即拒绝，只有全部 seed 正确才能通过。这样不减少失败候选的
gross 计数，却能减少其重复 DMA、命令执行和板端墙钟。

算子搜索结束后还增加 **incumbent-protected route composition**。设已经部署的 route 集合为 `S`，
待加入的资格化 route 为 `i`，它不能因为 same-tile 改善或 DMA 减少就自动写入最终 manifest，而要
在目标图上下文中检查边际量：

```math
\Delta_i(S)=T_{graph}(S\cup\{i\})-T_{graph}(S),
\qquad M_{live}(S\cup\{i\})\le M_{udmabuf}.
```

只有全图正确性通过、共享内存空间合同满足且 `Delta_i(S)` 在配对测量中低于准入阈值时才接受该
route；否则保留当前 incumbent。P7R249 的双 route 因子实验说明这不是形式化补丁：Y02 barrier
相对 same-tile original 确实更快且大幅减重，但相对 TopHub tile 仍使整网变慢约 23%，必须被部署
门拒绝。P7R250--P7R251 已把该规则实现为证据哈希绑定、基线集合连续性检查和五条件 fail-closed
的贪心规划器；它接受 Y00、拒绝 Y02。当前实现不提供 `2^n` route 子集的全局最优保证。

论文的主要评价对象不是最末端 latency 是否超过 TopHub，而是达到同等候选质量所付出的搜索成本。
对 workload `w`，先在冻结并实际测完的正确候选池中定义 oracle：

```math
T_w^*=\min_{x\in\mathcal X_{w,FPGA\text{-}correct}}T(x),
\qquad
C_{\pi,w}(\epsilon)=\min\{b:T_{best}^{\pi}(b)\le(1+\epsilon)T_w^*\}.
```

比较 Random、原始 AutoTVM/XGB、仅静态合法性、仅 minimum-access 规则和本文完整方法在
`C_pi,w`、time-to-target、编译失败数、FPGA 调用数及搜索期 DMA 字节上的差异。TopHub 只在
硬件指纹匹配且通过正确性资格时作为额外参考线；其实体、位置和 latency 在选择完成前不可见。
因此正确的结论形式是“用少多少搜索代价达到 pool oracle/TopHub 的 2% 或 5% 等价带”，不是
“比 TopHub 快多少”。

评价同时保留两种不能混淆的成本口径：`gross candidate` 表示算法依次考虑的配置数，`FPGA
dispatch` 表示真正送入板端的配置/seed 次数。静态和 FSim 预资格不是免费操作，因此端到端成本为：

```math
T_{search}=T_{common\_qualification}
+\sum_{x\in D_\pi}(T_{lower,x}+T_{FSim,x}+T_{cross,x}+T_{FPGA,x}+T_{measure,x}),
```

```math
B_{search}=\sum_{x\in D_\pi}\sum_{r\in runs(x)}
\left(B_{load,x,r}+B_{store,x,r}\right).
```

其中 `T_common_qualification` 单独报告，不能藏在搜索器外面；`B_search` 来自 runtime 的逻辑 VTA
LOAD/STORE 计数，不冒充物理 AXI burst。当前 Y00 主实验在已经冻结的 11 点 FSim-pass 池内公平比较
排序策略，24→11 的公共预资格成本和淘汰率作为独立前置成本报告，不能写成某个排序策略减少了
13 次 gross candidate。

### 3.3 工程闭环：exact-allowlist 安全定容（非核心创新）

只对最终允许部署的候选集合计算 instruction/UOP 峰值，并由 runtime manifest 核对配置、源码、
TIR、bitstream/u-dma-buf 指纹。它不单独承担创新点，只作为“搜索结果能够安全部署”的附属闭环，
避免固定 64 MiB 命令预留。

定容本身不是要包装成一个大创新。它仍有价值，因为显式 residency barrier 一方面延长 weight tile 的片上生命周期，
减少从 u-dma-buf 重复 LOAD；另一方面把长命令流切成若干可证明安全的提交段，从而降低单段
instruction/UOP 峰值。方法直接研究三者的 Pareto 关系：

```math
\min_x\left(T(x),\;B_{data}(x),\;C_{insn}(x)+C_{uop}(x)\right),
\qquad x\in\mathcal{X}_{FPGA\text{-}correct}.
```

更多 barrier 可能增加同步/提交时间，所以最终仍以真实 FPGA latency 为主、共享内存占用为约束
或次目标，而不是看到 bytes 下降就默认更快。

### 3.4 相关工作留下的问题与本文解决的问题

这四项工作形成了一条自然递进路线：2022 年先解决“初始样本中非法点太多”，2024/2025 年进一步
解决“整个在线调优过程中持续预测有效性和性能”，2026 年转向“从调度机制上减少卷积重复访存”。
但是对固定的显式 DMA FPGA，仍缺少一条把 **共享内存访问机制、合法候选排序、真实硬件数值正确性
和安全部署** 连起来的闭环。本文解决的是这个尚未贯通的问题，而不是重复宣称上述单项能力。

| 工作 | 已经解决的问题 | 解决后仍存在的问题 | 本文怎样解决该问题 | 当前证据 |
|---|---|---|---|---|
| HW-Aware Initialization，2022 | 利用 VTA 合法配置的邻域聚集，替换随机初始样本，降低 AutoTVM 冷启动的不稳定性 | `valid/invalid` 只回答能否编译，不能回答合法候选会产生多少 u-dma-buf 重复访问、谁应优先上板；编译合法也不能隔离真实 FPGA 数值错误 | 对真实 lowering 后的合法候选提取 input/weight/output DMA、请求数和提交数；FSim 后再做三 seed FPGA correctness，错误候选不进入性能池 | Y01 干净启动留出池中 6 个 FSim-pass 身份仍有 3 个 FPGA-invalid；首错即停只执行 12/18 次 seed 检查且不产生错误 latency 标签 |
| ML²Tuner，2024 | 用独立 Model V 持续过滤无效配置，并用 Model A 的编译隐藏特征改善性能预测 | 模型仍需要在线样本；通用 loop/branch/partial-tile 特征没有直接回答“哪一种 tile/驻留方式减少了多少共享内存搬运”，也不负责精确 bitstream/程序身份的板端数值认证 | 将最终 VTA DMA 语义量作为无性能标签的冷启动先验，显式搜索 input/weight residency；确定性资格和板端认证不让预测结果直接获得“正确”身份 | Y04 否定 bytes-only 后，冻结的 service proxy 在未见 latency 的 Y01 池用 2 次 gross dispatch 命中 oracle，bytes-first 需 3 次 |
| Multi-level ML-Guided，2025 | 完整展开 ML²Tuner，并用 ResNet18、InceptionV3 验证跨卷积的有效性预测与性能预测 | 其主要闭环仍止于“更少样本得到相当性能”；没有把搜索期间的共享内存流量、VTA submission/命令峰值以及 u-dma-buf 物理布局作为统一研究目标 | 在样本数之外累计逻辑 LOAD/STORE、DMA calls、FPGA invocations 和五阶段墙钟；最终候选再绑定缓冲复用/布局及 instruction/UOP 容量合同 | aggregate budget=8 相对 stock knob-XGB：LOAD -42.13%、DMA calls -49.24%、墙钟 -15.34%；同时如实报告 FPGA invocations +25% |
| Data Access and Schedule Optimization，2026 | 用输入优先和片上权重复用减少卷积冗余访存，并形成 minimum-data-access 调度 | “访存最少的调度”放入更大的 AutoTVM tile 空间后，仍需解决哪些组合能真实 lowering、访存下降是否抵消同步/submission、如何用较少板端试验选出正确优解，以及如何安全部署 | 将 original/input/weight residency 与 tile 联合搜索；以 real lowering/FSim/FPGA correctness 分层资格；跨 tile 按全张量 DMA 排序；以完整正确池做等预算评价并连接部署合同 | Y00 五个输入驻留对和 Y03 五个驻留对全部加速；Y02 权重驻留两次独立启动分别改善 25.28% 和 23.55% |

#### 3.4.1 2022 工作之后还缺什么，本文补了什么

Rieber 等已经证明随机初始化会被大量非法 VTA 配置干扰，并通过编译器 validity check、邻域预采样、
合法/非法平衡的 `E0` 和模拟退火偏置改善收敛，报告平均只需要基线 41.6% 的 trial 找到最优点。
因此本文不能再把“无效候选很多”或“先检查能否编译”写成创新。

但该方法把候选压缩为一个二值问题：`valid` 或 `invalid`。它没有继续回答合法点之间谁会反复读取更多
共享内存，也不能处理本项目已经观察到的“lowering 和 FSim 均通过、真实 FPGA 数值却错误”。本文
补上的两个环节是：

```text
编译合法候选
 -> 从最终 lowered TIR 计算全张量 DMA，决定优先级
 -> 三 seed FSim
 -> 三 seed FPGA 数值认证，失败则 fail closed
```

所以本文解决的不是 Rieber 已解决的初始化问题，而是其后的“合法候选内部如何按共享内存代价排序，
以及编译合法性如何升级为真实硬件可信性”。两种方法仍可组合：前者构造更好的初始覆盖，本文负责
合法点内部排序和硬件认证。

#### 3.4.2 2024/2025 工作之后还缺什么，本文补了什么

2024 workshop 版和 2025 LCTES 正式版属于同一 ML²Tuner 方法演进。它们用 Model P 预测性能、Model V
预测有效性，再编译 `(alpha+1)N` 个候选提取 loop、branch、partial tile 等隐藏特征，由 Model A 选出
最终 N 个候选。正式版报告只用 TVM-like 方法 12.3% 的样本达到相当性能、无效 profiling 平均下降
60.8%，并增加 InceptionV3 的 5x5、1x7、7x1 等卷积实验。

这已经解决了“调优过程中怎样持续学习有效性和性能”。剩余问题是，模型特征主要承担统计预测作用，
没有把 VTA 的显式 LOAD/STORE 转化为一个可审计的共享内存目标，也没有从搜索空间本身增加明确的
input/weight 驻留机制；预测为 valid 也不能代替特定 bitstream、TIR 和物理缓冲布局下的数值认证。

本文对应地解决三件事：

1. 把 original/input-stationary/weight-resident-barrier 变成受 tile 约束的搜索维度，直接改变重复搬运；
2. 用最终 VTA 程序的逻辑 DMA 字节、请求数和提交数构造无 latency 标签的服务代价。Y00/Y03 中
   bytes-only 首次命中 oracle，但 Y04 将其反例化；修正规则随后在未见 latency 的 Y01 池以第 2 次
   gross dispatch 命中 oracle，而 bytes-first 需第 3 次；
3. 用精确实现身份、FSim/FPGA 多 seed 正确性及 u-dma-buf 布局合同封闭“预测有效但真实硬件错误”
   的风险。

本文没有解决 ML²Tuner 已经擅长的跨 workload 学习问题，且 exact lowering/FSim 的公共成本较高：
Y03 为 103.236 s + 12.222 s，另有 57.321 s lowerability 重冻结扫描。因此正确主张不是“取代或全面
优于 ML²Tuner”，而是“补上共享内存语义与硬件可信性层”。后续可将 Model V 作为更廉价的粗筛，
再用本文共享内存服务代价作 Model A 特征或最终 acquisition/tie-breaker。

#### 3.4.3 2026 工作之后还缺什么，本文补了什么

完整论文确认 Cheng 等已经较完整地解决“怎样生成低访存 VTA 程序”：Algorithm 1/2 给出 output/input
prioritized 循环，Fig. 2/3 给出权重不覆盖地址以及 StorageFlatten、InjectVirtualThread、
SchedulePostprocToPrimFunc、InjectCopyIntrin、runtime `PushGEMMOp` 的联合修改；还按 ACC/weight SRAM
判断适用范围并在资源不足时回退 naive TVM。YOLOv3、YOLOv5m、ResNet50 的 pure-instruction time
分别约改善 10%、23%、25%。所以这些机制和资源回退都不能作为本文原创。

但全文同时明确留下了 **搜索问题**：它只定义四种方案，在少量层上全部做硬件计时并按层选择最快者；
没有给出 AutoTVM ConfigEntity 空间、Random/XGB/SA tuner、trial/sample budget、regret、time-to-target、
停止条件或相同搜索预算比较。它所说的 tuning strategy 是 input-prioritized 下的块参数/virtual-thread
配置和资源适用策略，不是一套经过搜索效率评价的 AutoTVM tuner。

而且 Table 2 已经证明“minimum data access”不等于“每层最快”：c6/c9 都是 Scheme 3 快于组合后的
Scheme 4，c2/c8 两者持平，单独 input-prioritized 在 c6/c9 反而显著变慢。这使后续问题自然成立：

> 当四种驻留语义与更大的 tile ConfigEntity 空间组合后，如何在未知 latency 前决定先测谁，并以
> 最少真实搜索代价找到 FPGA-correct 的最快方案？

本文针对这个问题构造“驻留×tile 联合候选—real lowering/FSim 资格—跨 tile 全张量 DMA 排序—FPGA
correctness—完整池等预算评价”。真正的方法增量是 **跨 tile 搜索次序和 time-to-target**；FPGA
correctness、u-dma-buf 布局及命令容量是保证该搜索可落地的支撑层，不再充当相对论文的核心区别。

当前 `paper minimum-access` 仍只是本地 same-tile mode-only 对照，因为本地 barrier weight reuse 与
论文的不覆盖地址/runtime 实现不同，而且尚未实现论文 Scheme 4 的精确组合机制。不能写成全面优于
Cheng 2026；对方的三网络整网 pure-instruction 结果也强于本文当前的两个算子候选池。

#### 3.4.4 最终研究缺口

综上，已有工作已经分别解决：

```text
随机初始化不稳
 -> 在线预测无效配置和性能
 -> 生成更少输入/权重访存的调度
```

本文解决它们之间尚未贯通的最后一段：

```text
在固定显式 DMA FPGA 上，
怎样把数据驻留机制纳入 tile 搜索，
怎样用最终编译程序的共享内存流量在冷启动时决定先测谁，
怎样阻止仿真正确但板端错误的候选进入性能标签，
以及怎样把搜索结果按真实缓冲布局和命令容量安全部署。
```

#### 3.4.5 创新性审计：哪些只是工程落地

相对 Cheng 2026，以下内容虽然必要，但单独都只能算工程落地或系统加固，不能承担核心算法创新：

- 将论文描述的 input-prioritized/weight reuse 移植到当前 VTA；
- 增加 lowering、FSim、交叉编译和 FPGA correctness gate；
- 修复 u-dma-buf 分配复用与物理布局；
- 按最终候选缩小 instruction/UOP backing；
- 保存哈希、manifest、日志和失败证据。

当前唯一能够承担独立方法贡献的是“以最终 VTA 共享内存流量为无标签先验，优化达到同一 FPGA-
correct pool oracle 所需的搜索代价”。全文核对已确认 Cheng 论文没有给出 AutoTVM 搜索算法、预算或
time-to-target 实验，因此 **研究问题层面的差异已经成立**。Y04 已否定 `sort(total_dma_bytes)` 的
跨几何泛化，当前方法已修正为显式服务代价与在线存活率自适应，并在 Y05/Y06 完成两个严格冻结池
验证，其中 Y06 是规则冻结后的前瞻确认。
当前应区分两种结论：

1. **硕士论文系统型创新：已形成。** 机制重实现、共享内存感知搜索、真实硬件认证和部署合同组成
   完整系统，并已有五个标签隔离候选池与一个 latency-unseen recovery 池支持。
2. **相对 Cheng 2026 的基础搜索方法增量：已形成并有严格前瞻证据。** 论文没有解决跨 tile 等预算
   搜索；本文先建立 DMA 先验、用 Y04 反例修正服务模型，再把 lowering/FSim/FPGA 作为按需付费的
   保真度选择。Y06 的冻结自适应策略以 6/24 gross 候选命中完整池 oracle。该结果仍只是一个严格
   自适应 holdout，不能夸大为成熟通用 tuner。

第三点现在已经由系统工程推进为 **搜索成本感知的多保真调优**：搜索器根据便宜解析界、exact DMA、
FSim 和 FPGA 四级成本逐步决定“下一个候选、下一种验证保真度以及是否停止”，直接减少达到 oracle
等价带的端到端墙钟。后续强化应在新网络上继续与 Rieber 初始化、ML²Tuner 式 validity/performance
模型、minimum-access 和 stock-XGB 做同预算比较，而不是继续在 Y06 上修改规则。

一手来源：Rieber 2022 [arXiv 2205.15568](https://arxiv.org/abs/2205.15568)；ML²Tuner 2024
[NeurIPS ML for Systems workshop PDF](https://mlforsystems.org/assets/papers/neurips2024/paper6.pdf)；
ML²Tuner 2025 [LCTES DOI](https://doi.org/10.1145/3735452.3735538)；Cheng 等
[FGCS DOI](https://doi.org/10.1016/j.future.2025.108165)。

## 4. tile 需要探索清楚的四类影响

| tile/调度变化 | 共享内存影响 | 可能收益 | 可能代价 | 必测量 |
|---|---|---|---|---|
| 增大 `tile_co` | input reload 可能下降 | input bytes/calls 少 | weight/acc SRAM 增大，计算并行性变化 | 三 tensor bytes/calls、SRAM、latency |
| 增大空间 tile | weight reload 可能下降 | weight bytes/calls 少 | input halo/padding、连续性改变 | request `x/y/stride/pad`、latency |
| 改 `tile_ci` | reduction 分块次数变化 | LOAD/同步可能下降 | weight/input SRAM 增大，UOP 依赖风险 | reload、UOP、FSim/FPGA correctness |
| virtual thread | 计算/访问重叠可能提高 | 隐藏 DMA | buffer context 复制、容量和依赖错误 | 物理 footprint、同步、device wait |
| residence mode | 改变 input/weight 生命周期 | 消除重复 DMA | 另一张量流量上升、compact-buffer 失败 | same-tile 原/新配对 |

因此实验必须以 same-tile 配对分离“驻留机制收益”和“tile 本身的计算效率”，再以统一候选池评价
搜索。只看一个最优点或只看 input bytes 都不够。

## 5. 当前已有证据

1. 旧 ResNet same-tile 板端配对中，56 组有 42 组驻留更快，中位改善 3.31%；证明减少共享内存
   重复访问在真实 VTA 上可能转化为时间收益。
2. W05 中 400 个 bounded-hybrid 配置经容量/全张量 DMA 规则缩到 19 个，冻结测量 2 个即找到距
   TopHub 0.31% 的候选；这是单 workload 正证据，不冒充跨网络结论。
3. exact allowlist 已在真实 runtime 把两条命令队列的 requested backing 从 64 MiB 缩到 8 KiB，
   6/6 seed 正确；这是共享命令内存部署结果，不解释为 FPS 收益。
4. P7R115 在目标 latency 前冻结三个 YOLO 真层。完整域 3920 个 ConfigEntity，96 点扩展池与
   36 点 pilot 均已提交哈希，三层各有精确但封存的 TopHub reference。该 v1 池仍含不会真实减重
   的 feasibility-only mode2，因此只作为候选合法性开发证据；正式性能池必须以显式 barrier
   权重复用替换它，并生成新的不可变 v2 合同。
5. P7R117 pilot 的 static lowering 为 20/36，三 seed FSim 为 15/20；失败包括 compact-buffer、
   padding、SRAM 和 UOP dependency，证明 tile 的共享内存合法性不能由简单容量公式代替。15 个
   FSim 正确身份的单候选 4 KiB 对齐 instruction+UOP backing 为 8,192--335,872 B，中位
   36,864 B；直接否定“8 KiB 是 VTA 通用容量”，并把命令空间需求与 tile 联系起来。
6. 96 点静态审计为 52/96 成功：Y00 20/32、Y01 32/32、Y02 0/32。Y02 全失败是当前候选规则的
   真实缺口，必须修正规则或换成另一未见 YOLO 几何后重新冻结，不能上板后偷偷补点。
7. no-leak 搜索器已比较 Random、stock-XGB、rules、validity V 和三档 `V+DeltaT`。旧 P7Q 开发
   replay 中复杂模型未稳定优于 rules/stock-XGB，故最终方法优先保持简单、可解释。
8. P7R119 已把正式机制语义改为 original/input/`weight_resident_barrier`。Y02 三个 same-tile
   family 共 9 项，8 项 lowering 与三 seed FSim 全部通过；`tile_co=8` 的 barrier 因真实 allocation
   超限失败。`tile_co=1/2` 时 weight 从 11,501,568 B、4,992/2,496 calls 降到 884,736 B、
   16/8 calls，但分别增加 16/8 次 residency drain。minimum-access 因此是访问量与同步代价的选择，
   不是一个名义 hybrid。
9. P7R120--P7R121 证明“TopHub 命中”不能替代当前硬件正确性：Y02 精确 TopHub entity 的 source
   template 与 mode0 adapter 生成完全相同的 lowered TIR 和交叉二进制，本地 FSim 均 3/3 正确，
   但板端均 0/3，且重复调用 actual hash 不同。因此该记录不能作为当前 Y02 的部署 incumbent，
   也没有产生任何候选 latency 标签。
10. P7R122 先以历史 W05 config575 完成当前会话 3/3 健康门，再检查 Y02 两个 family 的六个身份。
    只有 Y02B00 original 与 `weight_resident_barrier` 各 3/3 正确；input-stationary 和整个
    `tile_co=2` family 被 FPGA correctness 拒绝。这说明 lowering/FSim 门必要但不充分，真实 FPGA
    canary 必须计入 gross search cost。
11. P7R123 给出新的同 tile 板端正结果：original 为 140.989 ms，weight-resident-barrier 为
    105.349 ms，改善 25.28%，7/7 配对轮获胜。按单推理归一化，总 LOAD 从 19,488,768 B 降至
    8,871,936 B（-54.48%），weight LOAD 从 11,501,568 B 降至 884,736 B（-92.31%），LOAD calls
    从 9,984 降至 5,008；同步从 1 增至 17，仍获得净加速。全部 28 次 timed call（含健康 canary）
    均逐元素正确。该结果仅证明 Y02B00 same-tile 机制，不声称 TopHub 等价或整网 FPS。
12. 同一 P7R123 身份把命令峰值与时间结果连在了一起：original 的 instruction/UOP peak 为
    259,760/328 B，barrier 的单提交 peak 为 11,792/7,504 B。按 4 KiB 分页，所需命令 backing
    由 266,240 B 降为 20,480 B（-92.31%）。这是“驻留减少数据重放 + barrier 限制命令峰值”的
    联合时间/空间证据，不等同于释放整块 192 MiB u-dma-buf。
13. P7R124 的 Y01 8 family × 3 模式共 24 个身份全部通过 lowering 和三 seed FSim；P7R125 的
    首个结构覆盖板端池却 0/6 身份正确。P7R126 在剩余 18 个身份中完成 17 个，完成部分为 0/17
    正确；最后一个身份因 RPC broken pipe 尚未分类。由于该 family 的 original/input 已失败，可以
    确定八个 family 的“三模式全正确”数为 0/8，但不能写成 24/24 身份均已测失败。Y01 当前只能
    作为强负向正确性边界，不能产生 latency/oracle 结论。
14. P7R127 在不连接板端、不读取性能标签的前提下冻结 Y00 第三几何 24 身份：18/24 static-pass，
    11/24 通过三 seed FSim 与一致命令签名，但 0/8 family 三模式全合法。六个 barrier 因 weight
    residency region 无法证明 compact 在 lowering 阶段失败；另有七个身份在 FSim warmup 被
    duplicate `dst_idx` UOP 检查拒绝。Y00 因而适合在板端恢复后先做 original/input 的机制配对，
    不能强行构造三模式完整池。
15. P7R129 在 29 个具有完整三 seed 板端记录的 Y01/Y02 身份上回放顺序 fail-fast correctness gate。
    完整策略需要 87 次 FPGA 调用；首错即停、通过仍需三 seed的策略只需 33 次（-62.07%），且
    29/29 最终分类一致。被避免的逻辑 LOAD 为 1,176,345,600→448,836,608 B（-61.84%），LOAD
    calls 为 625,926→238,626（-61.88%）。这是搜索验证成本和共享内存访问的回顾性正结果；未来
    候选仍须前瞻报告晚失败率，不能将 29/29 一致外推成零漏判保证。
16. P7R130 把 Y00/Y01/Y02 的 57 个 v2 身份统一为 tile—SRAM—DMA—命令—资格图，得到 38 个
    same-tile 驻留对和 40 个仅改变一个 knob 的受控关联对。input-stationary 的 LOAD bytes 中位变化
    分别为 Y00 -30.51%、Y01 -5.06%、Y02 -22.55%；weight-resident-barrier 分别为 -40.16%、
    -83.80%、-61.50%。这些结果证明收益强烈依赖几何和 tile，不能只写一条固定复用规则；其中只有
    Y02 一对完成板端计时，其他数值仍是本地结构证据。
17. P7R131 在已有完整性能标签的 P7Q 开发池上增加两项严格消融：只在 same-tile family 内选择
    最小访问模式的 `paper_minimum_access`，以及跨 tile 按总 LOAD bytes、LOAD calls、padding 和
    submissions 排序的 `shared_memory_lexicographic`。4 个 workload、76 个非 TopHub 候选、20 seeds
    的回放中，budget=1 进入 pool oracle 2% 带的成功率分别为 Random 20%、stock knob-XGB 12.5%、
    paper minimum-access 33.75%、bytes-only rules 83.75%、本文 lexicographic 100%；本文方法的
    median trials-to-pool-2% 为 1。该规则是在已暴露开发集上选定，且池中 75/76 候选可计时，故只能
    冻结为下一轮确认算法，不能提前主张未见泛化或无效候选节省。
18. P7R132 已在读取任何 Y00 板端正确性或性能标签之前冻结前瞻确认合同。原始 24 个身份经静态门
    得到 18 个、三 seed FSim 门得到 11 个待上板身份；确认阶段将对 11 个身份全部做 FPGA 正确性，
    并对全部正确身份计时以构造完整 pool oracle。比较 Random、stock knob-XGB、公开 minimum-access、
    bytes-only 与本文 `bytes -> calls -> padding -> submissions` 五类搜索顺序，预算固定为 1/2/4/8。
    TopHub entity、位置和 latency 均不参与候选生成、排序或 oracle 定义。
19. P7R134 已离线交叉编译健康探针和 11 个冻结候选，12/12 成功；11/11 候选 lowered-TIR 哈希与
    P7R132 预注册身份一致。产物明确记录 `board_contacted=false` 和
    `performance_labels_collected=false`，因此不会把开发规则选择伪装成前瞻结果。板端恢复后只能按
    已冻结合同依次执行健康门、全池正确性、全池平衡计时和 20-seed 等预算回放，不能再改排序规则。
20. P7R137/P7R138 将搜索评价升级为五阶段成本模型。P7R137 在原 P7Q 720 条轨迹上保持全部候选
    顺序和核心结果不变，并对缺失的历史成本显示 `n/a`，不把零当作零开销。P7R138 对新版 Y00
    执行器重新封存：24 身份公共静态资格耗时 11.937 s、FSim 耗时 12.650 s；11 个候选交叉编译
    合计约 1.70 s（中位 152.24 ms/候选）。板端运行后，每个策略还将累计 correctness/measurement
    的 host wall time、逻辑 LOAD/STORE bytes、DMA calls 和 FPGA kernel invocations。P7R138 同时
    绑定执行器和搜索器两个源码哈希，仍未连接板端或读取 Y00 标签。
21. P7R144 完成了上述冻结前瞻协议：11/11 候选通过三 seed FPGA correctness，全部 77 次计时调用
    逐元素正确，完整池 oracle 为 17.621 ms。budget=1 时，按静态 DMA 总字节排序进入 oracle+2%
    带的成功率为 100%，Random 为 15%，stock knob-XGB 为 5%，公开 same-tile minimum-access 为
    20%。但 bytes-only 与 `bytes -> calls -> padding -> submissions` 在这个池中完全同序，因此正式
    方法应收敛为更简单的 DMA-bytes 主规则；request/command 增量只能作为消融和诊断，不能再声称
    带来额外预测能力。
22. P7R144 同时形成第二个真实 FPGA 机制几何。Y00 的五个 original/input-stationary 同 tile 对全部
    加速，分别为 18.29%、12.60%、7.97%、25.00%、28.80%；逻辑 LOAD 下降 13.52%--43.45%，
    DMA calls 均下降 33.33%。这比“只看一个最优候选”更能说明输入 tile 生命周期与共享内存访问
    的关系，但仍只是一个 Y00 候选池，不能外推到整网。
23. P7R139--P7R144 暴露出一个直接属于共享内存执行面的边界：逐调用新分配在完成 11/11 正确性后，
    于计时第 3 轮耗尽 192 MiB 的非释放分配路径；直接复用缓冲但把布局从原来的
    output-data-weight 改成 data-weight-output 时只有 1/11 正确；保持原布局并复用后恢复 11/11
    正确并完成 77 次计时而不再耗尽。因此“缓冲区复用”不能只按张量大小证明，还必须把物理分配
    顺序/地址布局写入运行合同。当前三次运行尚不是同启动 A/B/A，底层原因仍需地址级诊断。
24. P7R145 在新的 boot `1d260256-086b-4bf5-a834-4092e773acea` 上复测 Y02B00：权重驻留从
    143.445 ms 降为 109.657 ms，改善 23.55%，7/7 配对轮全部获胜，逻辑 LOAD 与同步结构和
    P7R123 一致。结合上一启动的 25.28%、7/7，这一单 family 结果已具备跨启动方向与量级稳定性；
    仍不能把两个启动当作两个 workload 或声称 YOLO 整网收益。
25. P7R146--P7R151 暴露并修正了 Y03 的“公式合法、编译失败”边界。完整 1280 个 ConfigEntity 中
    185 个通过手写容量/复用谓词，但首批按无标签哈希冻结的 8 个 family、24 个模式全部 lowering
    失败；改为只用真实 original lowering 合法性重冻结时，扫描 73 个 family 才得到 8 个可用 family，
    65 个因 padding、allocation 或 compact 2D DMA 约束失败，扫描耗时 57.321 s。随后三模式展开的
    24 个身份中 17 个 static-pass、13 个三 seed FSim-pass。P7R149/P7R150 是 FSim 环境缺失导致的
    无效运行，不计入候选失败。这说明固定 FPGA 的搜索空间不能只靠容量公式剪枝，必须落到编译器
    真实 lowering 和仿真合同。
26. P7R152 在任何 Y03 板端标签前冻结了 13 候选和五类搜索规则；P7R154 随后完成 13/13 三 seed
    FPGA correctness 与 91/91 正确计时，完整池 oracle 为 11.528 ms。budget=1 时，DMA-bytes 与
    复杂 lexicographic 均为 100% 命中 oracle+2%，Random 为 0%、stock knob-XGB 为 15%、paper
    minimum-access 为 10%。复杂规则与 bytes-only 第二次完全同序，当时得到“真实合法性门 + 静态
    DMA 总字节优先 + FPGA 正确性门”的中间结论；该结论后来被 Y04 反例推翻。Y03 的四个 input-stationary 对和一个 weight-
    barrier 对也全部加速，改善为 12.69%--80.61%，证明同一种驻留模式的收益仍受 tile/几何控制。
27. 合并 Y00/Y03 两个标签隔离完整池，共有 24 个 FPGA-correct 候选和 40 个 replay seeds。budget=1
    时 DMA-bytes 40/40 进入池 oracle+2%，Random、stock knob-XGB、paper minimum-access 分别为
    7.5%、10%、15%。aggregate budget=8 下，相对 stock knob-XGB，DMA-bytes 将逻辑 LOAD 降低
    42.13%、DMA calls 降低 49.24%、五阶段墙钟降低 15.34%；但 FPGA kernel invocation 增加 25%，
    因部分驻留候选包含多次 submission。公共资格成本不能隐藏：Y03 的 24 模式 static/FSim 另需
    103.236/12.222 s，且之前还有 57.321 s lowerability 重冻结扫描。
28. P7R155--P7R158 完成最终 selected/fallback exact allowlist 的命令容量闭环。Y00 selected、Y00
    same-tile original fallback 和 Y03 selected 的静态峰值分别为 `7,664/460`、`13,488/208`、
    `4,016/1,364 B`（instruction/UOP），因此按 4 KiB 页对齐得到 `16 KiB + 4 KiB`。P7R155 正向
    阶段在该容量下完成 3 身份 × 3 seed = 9/9 正确，观测峰值为 `13,488/1,364 B`、9 次提交。
    缩小一页至 `12 KiB + 4 KiB` 后，13,488 B 身份以 `submissions=0`、`driver_run_calls=0` 在设备
    提交前被拒绝，完成不足容量 fail-closed。但 P7R158 同时发现更强的边界：峰值只有 7,648 B 的
    Y00 selected 在 12 KiB 下虽提交一次仍产生 684,302 个错误元素；随后不重载 bitstream 切回
    16 KiB 仍错误（684,101 个），重载同一哈希 bitstream 后才恢复正确。A1 与重载后的 A3 均正确，
    W05 前后健康门均 3/3，且实验已用 4 KiB 哑元分配补偿队列缩小导致的预期张量地址偏移。
    因而“`peak <= capacity`”只是不越界必要条件，最终部署合同必须绑定**精确容量、物理布局、
    bitstream 状态和板端数值资格**；本结果不把尚未定位的底层原因武断归为 u-dma-buf 驱动缺陷。
29. P7R159--P7R165 增加第三个标签隔离完整池 Y04（YOLOv3-tiny conv18，`1x1, CI=256, CO=128,
    H=W=13`）。24/24 身份通过 static 与三 seed FSim。第一次板端运行 P7R163 在复用过的 RPC/bitstream
    状态下 0/24 正确；P7R164 重载相同 bitstream 后同一预注册 canary 恢复 3/3，P7R165 随后把
    “停止 RPC—重载 bitstream—新 RPC—健康门”加入执行合同，最终 24/24 三 seed FPGA-correct、
    168/168 计时调用正确，pool oracle 为 3.137311 ms。因此 P7R163 只能记为执行状态污染，不能当
    候选无效；干净启动是此平台搜索评价的必要前置条件。
30. Y04 同时给出关键反例：旧 DMA-bytes 规则在 budget=1 的 pool-oracle+2% 命中率降为 0%，
    所选点比 oracle 慢 75.35%；复杂 bytes-first lexicographic 的 regret 也有 37.96%。Y04 内 latency 对 DMA
    calls 的 Spearman 约 0.949，而对 bytes 约 0.375；最低字节的 barrier 方案因请求/提交代价而慢。
    这正式否定“只要总搬运字节少就优先”的跨几何泛化。
31. P7R166 在已暴露的 P7Q/Y00/Y03/Y04 共 7 个 workload 上冻结共享内存服务代价：
    `S=B+65536*N_dma+131072*max(N_submit-1,0)`。81 个离散系数组合中有 24 个让七个 workload 的
    首派发都进入 oracle+2%；按最小系数和等固定规则选出 `64 KiB/request + 128 KiB/extra-submit`，
    开发集最大/平均 regret 为 0.3855%/0.0662%。Y04 参与了拟合，所以该结果只负责参数冻结，不能
    冒充确认；两个权重也只能解释为等效字节 acquisition penalty。
32. P7R169 在任何正式 Y01 latency 出现前，冻结旧 P7R125 已预注册且 timing sample 为 0 的六点池
    `Y01F02/Y01F07 × 三模式`；P7R170 用干净启动恢复执行。六个 FSim-pass 身份中 3 个三 seed
    FPGA-correct、3 个首 seed 失败，fail-fast 实际执行 12/18 次 correctness 检查；三个正确候选完成
    21/21 平衡计时，oracle 为 Y01F02 input-stationary 的 13.675876 ms。同 tile original 为
    19.806108 ms，input-stationary 加速 30.95%，逻辑 LOAD 降 4.61% 而 LOAD calls 降 50%。冻结的
    service proxy 第 1 次碰到 FPGA-invalid barrier、第 2 次命中 oracle；bytes-only 与 bytes-first
    均需 3 次，paper minimum-access 的 20-seed 中位数为 4.5 次。达到 oracle+2% 时，proxy 相对
    bytes-first，达到目标所需 gross dispatch 为 **2 对 3（少 33.3%）**，
    五阶段已知墙钟 1.084 对 6.864 s、逻辑 DMA calls 314 对 24,666、FPGA kernel invocation 8 对 25。
    该小池支持修正规则，但首个低代价点仍然错误，证明服务代价排序与 correctness 模型必须分工。
33. P7R173--P7R174 首次把搜索改成 candidate×phase 在线状态机，并严格限制信息可见时间。在
    Y00/Y01/Y03/Y04 的已暴露开发池上，硬件多样 4→2 前沿相对“全池资格后再排序”，达到同一
    oracle+2% 时总墙钟由 184.589 s 降到 80.394 s（-56.45%），lowering 78→26、FSim 65→8；
    FPGA invocation 与逻辑 LOAD 保持相同。该结果只用于算法选择，不是未见确认。
34. P7R175--P7R183 完成第一个严格在线 Y05 holdout（YOLOv3-tiny conv6，3×3，CI64/CO128，
    52×52）。24 个身份在任何目标结果前冻结；完整结局为 18 个 lower-invalid、2 个 FSim-invalid、
    4 个 FPGA-correct。冻结 4→2 策略的第一次 FPGA/measure 身份就是 16.305 ms 完整池 oracle，
    独立首波值为 16.299 ms。与全池资格基线一致回放时墙钟 49.468→39.339 s（-20.48%）。但首波
    收集器实际同时认证了两个晋级点，故不能把额外收集工作冒充字面上的在线早停节省。
35. P7R184--P7R185 只用已暴露开发标签冻结“生存率自适应”规则：首个 same-tile family 三模式
    lowering 通过数不超过 1 时走 sparse family-wave，否则走固定 4→2。在五个开发 workload 上均
    达到 oracle+2%；Y05 相对固定前沿墙钟减少 95.23%，但 Y04 增加 4.21%，说明它是稀疏/稠密折衷，
    不是无条件占优。跨 workload Model V 在开发池未稳定胜出，因此只保留为负向消融。
36. P7R186--P7R194 在完全未见的 Y06（YOLOv3-tiny conv4，3×3，CI32/CO64，104×104）严格确认
    自适应规则。24 个身份中 22 个 lower-invalid、2 个通过三 seed FSim 且均三 seed FPGA-correct；
    14/14 计时调用正确。在线前缀观察首 family 为 0/3 后走 sparse path，仅 6 次 lowering、1 次
    FSim、1 次编译、1 次 FPGA gate 和 1 次测量便命中 30.103 ms 完整池 oracle；首波独立值为
    30.074 ms。统一成本回放相对全池资格的 4.822 s 为 3.522 s（-26.95%），相对固定 4→2 的
    4.972 s 为 -29.16%，gross candidates 由 24 降到 6。该首波只有一个候选，因此不存在 Y05 的
    收集器偏差。实验后 SD 两分区仍读写挂载，dmesg 的 ext4/mmc/I/O error 计数为 0，192 MiB
    u-dma-buf 与新 RPC 正常；这只证明本 boot 的运行环境健康。
37. P7R195--P7R203 用第二个未见 YOLO 几何 Y07（conv14，3×3，CI256/CO512，13×13）重复 sparse
    path。完整 480 个 ConfigEntity 中 44 个过容量粗筛，冻结 24 身份后却只有 2 个能 lowering/FSim，
    两者均三 seed FPGA-correct、14/14 计时正确。策略以 6 次 lowering、1 次 FSim 和第一次测量命中
    322.949 ms oracle；相对全池资格墙钟 -12.91%，相对固定 4→2 为 -14.42%。最初 run01 把层误写
    为 1×1，在任何目标结果前已废弃并以正确 3×3 描述重冻 run02，不混入正式证据。
38. P7R205--P7R213 增加跨模型严格留出 R50A：仓库 ResNet50 bottleneck 的 stride-1 1×1 conv1，
    CI512/CO128/28×28。3456 个 ConfigEntity 中 510 个过容量粗筛；冻结 24 身份后首 family 2/3
    lowering 通过，触发未曾在 holdout 验证的 dense 4→2 分支。完整池 10 个候选通过 lowering/FSim、
    10/10 三 seed FPGA-correct、70/70 计时正确；第一次测量候选 5.907 ms 即完整池 oracle，其余为
    8.335--192.917 ms。adaptive 相对全池资格墙钟 12.707→8.850 s（-30.35%），gross 24→10；与
    固定 4→2 相同，符合 dense 分支定义。纯 family-wave 更慢且多测一个候选，验证了分支选择价值。
39. P7R214 不重新调参地汇总 Y06/Y07/R50A 三个严格 holdout、72 个预注册身份。20-seed 配对结果中
    自适应策略三池全部命中率为 100%，中位 gross 22，对固定 4→2 的 58 和全池资格的 72；统一墙钟
    为 19.735/22.426/25.983 s。相对全池资格，自适应 gross -69.44%、墙钟 -24.05%；相对固定 4→2
    墙钟 -12.00%。三种方法达到目标时 FPGA invocations 与目标逻辑 DMA 相同，收益来自少做无必要的
    lowering/FSim/compile，不能写成物理 AXI 流量下降。
40. P7R215 在 R50A 完整正确池中构造五个严格 same-tile 的 original/input-stationary 配对，五对
    input-stationary 全部更快，改善为 29.13%--79.08%，中位 62.80%；输入逻辑 DMA 字节中位下降
    75.00%，总逻辑 DMA 字节下降 40.25%，DMA calls 下降 72.73%。这把该 ResNet50 真层的搜索命中
    与“延长输入 tile 生命周期—减少 u-dma-buf 重复 LOAD—真实 FPGA 加速”机制链直接连起来。
    但五个 family 共享同一几何和 boot，精确双侧 sign test 为 `p=0.0625`；bootstrap 区间仅是 family
    cluster 的描述性区间，不是跨 workload 的总体置信区间。
41. P7R216--P7R224 在标签前冻结 ResNet50 另一真层 R50B：stride-1 bottleneck conv3 的 1×1
    扩张卷积，CI128/CO512/28×28。完整空间 3456 个 ConfigEntity、容量粗筛 581 个，冻结 24 身份；
    首 family 0/3 lowering 存活，按未改动阈值走 sparse path。完整结果仅 4/24 lowering/FSim-pass，
    但 4/4 均三 seed FPGA-correct、28/28 次计时正确。第一次测量的 barrier 候选在首波为
    25.275 ms，独立全池中为 25.151 ms，仍是 oracle。统一回放中 adaptive 为 1.352 s，全池资格
    为 2.839 s（-52.36%），固定 4→2 为 2.668 s（-49.32%），gross 为 6/24。
42. R50B 的机制边界与 R50A 不同：八个 input-stationary 身份全部在 lowering 被拒绝，20 个失败按
    `dma_2d_pattern/allocation_capacity/dma_compact_buffer` 分为 10/6/4；唯一可配对的 weight barrier
    相对相同 tile original 为 25.746→25.151 ms（+2.31%），七个计时轮 7/7 更快，总逻辑 DMA 字节
    -15.95%、DMA calls -42.86%、instruction peak -94.43%。这只是一对同 boot 机制证据，不能写成
    ResNet50 普遍权重驻留增益。
43. P7R226 在不改阈值和 service proxy 的情况下汇总 Y06/Y07/R50A/R50B 四个严格 holdout、96 个
    预注册身份。20-seed 配对结果全部命中完整池 oracle；adaptive 的中位 gross 为 28，对固定 4→2
    的 82 和全池资格的 96，统一墙钟为 21.088/25.094/28.822 s。adaptive 相对全池资格 gross
    -70.83%、墙钟 -26.83%，相对固定 4→2 墙钟 -15.96%。四个 target 都在第一次测量命中 oracle，
    是强正结果也可能存在选择运气，必须继续保留完整池审计和外部有效性限制。
44. P7R227--P7R228 实现并验证 `ExplicitResidencyDispatch`：以 exact workload 和 complete
    ConfigEntity 把已资格化候选送入普通 `conv2d_packed.vta` 的 Relay schedule 入口，未命中项继续
    委托 TopHub，配置不一致则 fail closed。同 workload/config 的最小 Relay 图保持 graph/params
    完全相同而 VTA 二进制与 TIR 结构不同；clean-start 板端 6/6 correctness、14/14 timing 正确，
    weight barrier 为 25.674→25.083 ms（+2.36%）、7/7 更快。单次 weight LOAD 为
    458,752→65,536 B、calls 448→16，submission 1→17，与独立函数 P7R225 的结构和量级一致。
45. P7R229--P7R230 首次完成 ResNet50-v2 整图 exact dispatch 和板端计时，但确定性随机参数使最终
    logits 全零。该运行仍证明 route 命中、整图可运行和 7/7 性能方向，只能作为开发记录，不能
    承担整网数值正确性主张；随后以预训练参数重新冻结并复测，未沿用其正确性结论。
46. P7R231--P7R232 使用官方预训练 ResNet50-v2 参数完成强复测。A/B graph 相同，107 个 lowered
    参数逐字节语义一致；exact workload 在整图对应四个算子实例。三输入 seed 的 1,000 个非零
    logits 全部逐元素相同，14/14 计时调用配对相同。整网中位 `run` 为
    406.701→405.362 ms（+0.330%），7/7 轮驻留版更快；每次推理 weight LOAD 减少 1,572,864 B、
    weight calls 减少 1,728、submission 增加 64，恰为单算子差值四倍。总逻辑 DMA 字节下降 1.84%、
    calls 下降 12.35%。这是 E5 的首次完整 ResNet50 graph 传递，不是 ImageNet accuracy 或物理 AXI
    实验，也不能把 0.33% 包装成显著 FPS 提升。
47. P7R234--P7R235 将 P7R144 的 Y00F06 original/input-stationary same-tile pair 放回完整
    YOLOv3-tiny 图。A/B graph 和参数语义一致、exact route 各命中一次；person 图片与两组随机输入
    共 6/6 graph calls 的 8 个输出张量全部非零且逐元素相同，14/14 timing calls 配对相同。整图
    266.250→264.247 ms（+0.758%），7/7 轮更快；单次推理 input LOAD 减少 794,368 B，恰等于
    P7R144 独立 pair 的差值，总逻辑 DMA 字节下降 1.73%。这证明单个 YOLO `conv2` 的 input
    residency 收益可传递到真实检测图。
48. P7R237--P7R238 增加未安装 route 的 stock TopHub 整图保护线。相同 selected 配置保持全部
    输出逐元素相同，并以 267.503→264.220 ms（+1.243%）、7/7 胜通过参考线；单次总逻辑 DMA
    46,513,152→45,031,040 B（-3.19%），DMA calls 21,392→20,300（-5.10%）。该差值同时包含 tile
    和 residency，不能冒充纯驻留机制效应；正确口径只是当前一个 workload/boot 的整图部署结果，
    不是“全面超过 TopHub”、COCO mAP 或物理 AXI 结果。
49. P7R240--P7R241 首次把 Y00 input 与 Y02 weight 两条独立资格化 route 同时放入 YOLOv3-tiny。
    组合版两条 schedule 均精确命中，三类输入的八个非零输出与 Y00-only 版全等；但整图
    264.467→342.878 ms，慢 22.87%、0/7 获胜。单次 weight LOAD 虽减少 2,654,208 B，input LOAD
    却增加 3,170,304 B，LOAD 总量反增 525,312 B，并增加 16 次 submission。这是“局部正候选不可
    无条件叠加”的部署反例，不是候选正确性失败。
50. P7R242--P7R243 用固定 Y00 route 的三方对照拆开 Y02 的 tile 与 mode。Y02 same-tile original
    为 377.554 ms，weight barrier 为 343.089 ms，barrier 加速 10.05%、6/6 获胜；单次恰好减少
    10,616,832 B weight LOAD 和 4,976 次 LOAD，与孤立 Y02 pair 的结构差值完全一致。但 TopHub
    Y02 基线为 265.090 ms，barrier 仍慢 22.73%。所以 weight residency 机制仍有效，失败来自该
    ConfigEntity 的绝对质量，而非 route 组合破坏了复用语义。
51. P7R245--P7R249 完成 Y00/Y02 两 route 的 2×2 边际审计。Y00 在 Y02 关闭/开启时分别使整网
    配对中位变化 -3.413/-4.346 ms，均 7/7 获胜；Y02 在 Y00 关闭/开启时分别增加
    79.473/78.355 ms，均 0/7。两种求法的 latency interaction 约为 -0.93/-1.12 ms；更关键的是
    两条边得到的逻辑 DMA bytes/calls interaction 严格为 0，证明访问变化可加、Y02 退化是其自身
    相对 incumbent 不合格。由此把最终部署规则修正为全图正确性、共享内存容量和边际 latency
    三重保护，而不是“same-tile 更快就接受”。
52. P7R246 暴露测量器自身的空间边界：同一 clean-start RPC 同时构造四个完整 YOLO graph
    executor 后，首次 VTA 调用因 `fpga_buff_ == nullptr` 失败；同 boot 三 executor 的 P7R243
    可以完整运行。该失败保留为 192 MiB u-dma-buf 下的替代版本并存上界证据，但未记录精确分配
    峰值，不能归因驱动缺陷或计作候选 FPGA-invalid。正式因子实验因此改为每次仅保留两个 executor
    的四条 clean-start 边。
53. P7R250--P7R251 实现 incumbent-protected 贪心 route 规划器。每个 proposal 必须以当前已接受
    route 集为 baseline，只允许增加一个 exact candidate，并核对证据 SHA256、全图输出一致、
    live-memory fit、最少 7 轮、全胜和负中位 latency delta。正式回放中 Y00 的五项检查全过而被
    接受；Y02 虽正确且可两图并存，但因 0/7 和 +78.355 ms 被拒绝，最终 manifest 只保留 Y00。
    三项单测另验证接受后拒绝不污染 incumbent、陈旧 baseline 拒绝和证据篡改拒绝。
54. P7R252 将 P7R251 选择结果绑定为 exact YOLO graph route manifest：最终只含 Y00 candidate，
    绑定其完整 identity、候选源合同、graph/params/AArch64 binary、P7R241 板端执行与 clean-start
    bitstream/u-dma-buf 证据；目标 binary SHA256 为 `3fbecf15...55cb`，manifest ID 为
    `0ad7d0b7...11111b`。Y02 及失败检查仍留在清单中。该阶段明确不复用 W05 单 workload 的命令容量
    作为 YOLO 整图上界，并把整图 command peak 留给后续独立采集；该缺口由 P7R259--P7R261 闭合。
55. P7R253--P7R259 为 AXU u-dma-buf bump allocator 增加只读快照，并只部署在隔离诊断 runtime。
    每个 YOLO graph+params 精确增加 34,300,928 B、38 次分配；四图 dry-load 为 137,203,712 B。
    默认首次运行先分配 32 MiB UOP 队列到 170,758,144 B，剩余 30,568,448 B，随后 32 MiB
    instruction 请求因此差 2,985,984 B。这与 P7R246 失败时的 `used=170758144` 完全吻合，排除了
    SD 卡或候选数值错误，确认根因是四份图 backing 与默认命令队列的空间叠加。
56. P7R259 同时得到完整图各变体的命令峰值：stock/Y00/Y02/both 的 instruction 为
    151,744/151,760/151,744/151,760 B，UOP 为 1,384/1,384/7,508/7,508 B；Y02 barrier 增加
    submission，因此提高 UOP 峰值。按四变体 allowlist 的最大值与 4 KiB 对齐，所需容量为
    155,648 B instruction + 8,192 B UOP，而非默认 64 MiB。
57. P7R260 用上述四变体容量重跑原先 OOM 的四 Executor 2×2 因子实验，12/12 correctness、32/32
    timing 全部逐元素正确并完成八轮。stock/Y00/Y02/both 中位分别为 272.758/268.461/350.584/
    347.737 ms；Y00 仍有益，Y02 仍有害，逻辑 DMA interaction 仍为零。P7R258 先用 Y00-only 的
    4 KiB UOP 容量运行四变体时在 Y02 上 fail closed，证明容量合同必须按 allowlist 最大峰值生成。
58. P7R261 最终把 P7R252 exact selected graph 与整图容量绑定：155,648 B instruction + 4,096 B
    UOP 下 person 与两组随机输入的输出集合 3/3 逐哈希匹配，36 次 submission 的实测峰值为
    151,760/1,384 B；instruction 少一页至 151,552 B 时，首个合法 submission 后在下一段提交前以
    `queue backing capacity exceeded before submission` 拒绝，距离实测峰值仅 208 B。命令 backing
    从 67,108,864 B 降为 159,744 B（-99.762%）；这是单 boot、静态图、replay-disabled 的精确合同，
    不是 VTA 通用常数，也不等同释放整块 192 MiB。
59. P7R262 验证了不能用 TOPI 查询序号冒充 Relay call-site：同一 R50B workload 在 Graph JSON 中
    只有节点 57/72/85/98 四个执行实例，但编译期间出现 7 次目标查询；前两版 route 还未进入实际
    schedule。该原型已完整回退，失败目录不计候选或板端失败。由此将 call-site 身份约束为稳定的
    图节点/函数绑定，而不是易受 probe/cache 影响的编译器内部时序。
60. P7R264 将 P7R231 已资格化的 original/barrier 两份图模块组合成 exact graph-node 变体：Graph
    JSON 与 120 个参数语义保持不变，只把节点 57 的函数名改为别名，并由 AArch64 wrapper 精确转发
    到 barrier DSO；另外三个同 workload 节点仍调用 incumbent。构建器、wrapper 和 8 个产物均由
    SHA256 绑定，三项单测验证只改一个节点、实例数变化拒绝和 occurrence 越界拒绝。
61. P7R265 run02 在真实 FPGA 上完成该单节点组合的 6 次正确性和 14 次计时，全部输出逐元素相同且
    非零。单推理 weight LOAD 恰好减少 393,216 B/432 calls、同步和 driver runs 各增加 16，等于
    孤立 R50B 的一份差值及 P7R232 四节点广播差值的四分之一，证明没有误改另外三个实例。延迟
    406.375→406.017 ms，仅名义改善 0.088%、4/7 轮获胜，故只支持 call-site 流量隔离与正确性，
    不支持整图加速结论。当前实现是 Graph JSON/二进制部署组合，尚非 Relay/AutoTVM 原生 call-site
    搜索。
62. P7R266--P7R267 把 R50B 的四个相同 workload 实例冻结为 `k=0..4` 的前缀驻留剂量。四组独立
    clean-start A/B 共 24 次 correctness、56 次 timing 全部正确；每增加一个实例都严格带来 weight
    LOAD `-393,216 B/-432 calls` 与 `+16` 次同步，证明共享内存变化线性可加。延迟却不线性：
    `k=1/2/3/4` 的中位差分别为 `-0.036/-1.428/-1.314/-0.791 ms`，说明 bytes 可用于形成候选组，
    不能直接替代真实整图性能门。
63. P7R268--P7R269 又测量四个节点在“其余三点全 original”和“其余三点全 resident”两种上下文中的
    八条边。每条边的 DMA 与同步差值完全相同，但延迟边际明显依赖图位置和当前上下文：节点 57 的
    两条边为 `-0.898/-0.176 ms`，节点 85 为 `-0.124/-0.419 ms`，节点 98 的 full-context 边甚至为
    `+0.008 ms`。因此部署选择必须相对当前 incumbent 测边际，不能把孤立算子收益机械广播。
64. P7R270--P7R271 在全部 16 个 R50B 子集上执行严格逐点准入。四个单点虽然中位数都名义更快，
    但仅获 4/7、6/7、5/7、4/7，因未达到 7/7 门而全部拒绝；最终三方复核却显示全四点 bundle
    相对 all-original 快 `1.000 ms`、9/9 获胜。该反例证明保守单点 greedy 会在低信噪比下出现
    false negative，也直接给出“利用 DMA 可加性先测整组、失败再拆分”的新搜索动作。
65. P7R272--P7R273 首次把 R50A input-stationary 广播到预训练 ResNet50 的三个真实实例。输出 6/6
    正确，但预注册的“孤立模板差值×3”把总 LOAD calls 预测为 `-192`，板端却是 `-204`，执行器按
    合同在计时前停止。额外 12 次来自融合图中的 ACC LOAD，说明完整图预测必须基于最终 fused TIR，
    不能只乘未融合算子特征；这不是候选数值失败。
66. P7R274--P7R275 捕获两种模式各 68 个 `CPUAccessRewrite` 后的 fused TIR 模块，并按 Graph JSON
    节点 67/80/93 的出现次数聚合。每个实例的精确差值为总 LOAD `-401,408 B/-68 calls`、input
    `-401,408 B/-32 calls`、weight `0 B/-32 calls`、ACC `0 B/-4 calls`；三实例合计与 P7R273
    板端十项 LOAD/STORE 字段逐项完全相等，补上了“编译结果—整图共享内存请求”的可审计模型。
67. P7R276 在修复后的 fused-TIR 合同下重新 clean start：6 次 correctness 与 14 次 timing 全部
    输出相同且非零，静态预测在 correctness 和 timed profile 中均逐项精确命中。三个 R50A 实例
    驻留使整网中位延迟从 `337.273` 降到 `330.125 ms`，提升 `2.165%`、7/7 获胜。这是第二种
    ResNet50 驻留机制的整图正结果，但仍只是一种 workload、一个 boot，并非 ImageNet accuracy。
68. P7R277--P7R278 将 R50A 三个节点展开为八个精确子集，比较相同严格门下的逐点 greedy 与整组
    准入。逐点法三点均接受并选到 `111`，需要三对、60 次整网执行；整组 `000→111` 一对即通过
    三 seed、精确 DMA 和 7/7 性能门，只需 20 次整网执行，验证成本下降 66.67%，整组中位差
    `-6.977 ms`。P7R279 将它与 R50B 反例统一为 **DMA 可加性引导的层次化 bundle admission**：
    先测同驻留语义、资源合同兼容且编译期 DMA 差值可加的整组，失败的非单点组才递归拆分。当前
    两个域均为标签已暴露的开发证据，且尚未实际触发“整组失败后递归拆分”，不能声称未见泛化、
    全局最优或最坏成本优于逐点法。
69. P7R280 用既有 YOLO 四变体因子数据做强制异构 bundle 负向控制：`stock→Y00+Y02` 为
    `+75.012 ms、0/8`，拆分后 `stock→Y00` 为 `-4.196 ms、8/8` 而
    `Y00→Y00+Y02` 为 `+79.285 ms、0/8`，最终选择与既有顺序 planner 完全一致。它证明失败后
    “相对更新 incumbent 递归拆分”的决策语义可恢复安全子集；同时三次 pair test 比两次逐点多
    50%，明确保留最坏成本边界。Y00/Y02 属于不同 workload/mode，本来不会由正常 same-mode
    call-site 分组规则合并，因此这只是回顾性对抗控制，不是自然 bundle 失败或新上板实验。
70. P7R281--P7R288 在不读取目标完整图或 partial-mask latency 的条件下，按注册表顺序确定性冻结
    新 tile `R50AF00` 及节点 67/80/93，并以最终 fused TIR 预注册整组三点 LOAD 差
    `-3,612,672 B/-9,072 calls`。P7R286 四个独立 clean start 中，三条单点边分别为
    `-15.088/-16.099/-15.993 ms`、均 7/7，逐点 greedy 选择 `111`；整组也选择 `111`，中位延迟
    `422.292→374.452 ms`、提升 12.776%、7/7，但只需 20 而非 60 次整网执行，验证成本下降
    66.67%。所有 correctness/timing 输出非零且相等，板端十项 LOAD/STORE 均逐项命中预注册预测。
    这是规则冻结后的首个 **新 tile 调用点/完整图 latency holdout**，不是新网络、新 workload、
    独立算子 correctness holdout 或自然失败 bundle；历史算子级标签存在但没有用于选择。P7R286
    原始摘要沿用了旧 runner 的硬编码 claim boundary，P7R287 已在不改原始证据的前提下校正解释。
71. P7R289--P7R296 用相同冻结规则继续选择下一个正确性合格 tile `R50AF01`
    (`tile_h=14,tile_w=1,tile_ci=1,tile_co=1`)。最终 fused TIR 预注册三节点 LOAD
    `-8,429,568 B/-76,440 calls`；四个 clean start、24 次 correctness 和 56 次 timing 全部输出
    非零且相等，十项 DMA 差逐项命中。三个 singleton 均 7/7 接受，whole group
    `898.821→435.519 ms`、吞吐提升 106.379%、7/7，逐点和整组仍都选 `111`，验证执行仍由
    60 降至 20。更重要的是，与 F00 的受控跨 tile 对比形成“bytes-only 会排错”的完整图反例：
    F01 original 的 LOAD bytes 比 F00 少 23.25%，但 LOAD calls 多 622.22%、延迟高 112.84%；
    驻留后 bytes 少 47.43%，calls 仍多 261.11%、延迟高 16.31%。F00/F01 的 input reload ratio
    分别为 4/8，平均 LOAD 请求分别从 2,232.9/237.3 B 提升到 7,736.9/1,126.4 B。由此可主张
    最终 fused program 的 bytes 与 calls 必须联合进入共享内存服务代价，不能以最少字节替代真实
    latency。P7R294 run01 只因未导出 SSH askpass 在板状态读取前失败，未重载 FPGA 或执行候选；
    run02 才是有效板端证据。F01 仍与 F00 共享 workload、模型和 boot，不是独立域，也未触发自然
    bundle 失败。
72. P7R297--P7R305 将冻结的生存率自适应多保真方法带到新的 ResNet50 bottleneck `R50C`
    (`CI=1024,CO=256,H=W=14,1x1`)。完整 ConfigEntity 域 2240 点，按无标签规则冻结 8 family、
    24 模式；首 family 0/3 lowering-pass 后走 `sparse_family_wave`，只考虑 6/24 个候选，并以
    lower/FSim/compile/FPGA/measure=`6/1/1/1/1` 的动作找到第一测量点。完整池审计得到 11 个
    lowering/FSim-pass、10 个 FPGA-correct，首个前瞻测量的 input-stationary 候选就是
    `4.442324 ms` pool oracle。可比 replay wall 为 `822.451 ms`，相对 exhaustive 的
    `8559.153 ms` 减少 90.39%；实际执行器另有 `1713.486 ms` 本地编排开销，必须单列。
73. R50C 同时提供机制正证据和硬件资格反例。五个 FPGA-correct original/input same-tile 对全部
    加速 `24.336%--79.853%`，中位 44.694%，input LOAD bytes 中位 -75%、总 DMA calls 中位
    -66.667%；但唯一 lower/FSim-pass 的 weight barrier 在前两 seed 正确、第三 seed 出现
    17,920 个错误元素并被拒绝。它证明三 seed FPGA gate 不能由 lowering/FSim 或单 seed 取代，
    也证明 P7R129 顺序 fail-fast 的平均节省不适用于“最后一个 seed 才失败”的候选。
74. P7R306 合并 Y06/Y07/R50A/R50B/R50C 五个规则冻结后的严格 workload、120 个预注册身份。
    20 个 replay seed 中自适应方法均到达五个 pool oracle；中位 gross 为 34，对 fixed 94、
    exhaustive 120；中位墙钟为 `21910.022/26580.804/37380.841 ms`。相对 exhaustive 的 gross
    与墙钟分别减少 71.667%/41.387%，相对 fixed 墙钟减少 17.572%。这些 seed 是搜索顺序消融，
    不是 100 次独立开发板实验，不能写成任意网络 100% 首测最优。
75. P7R308--P7R313 在任何 R50C full-graph/partial-mask latency 前冻结 R50CF00 pair，再从两侧
    各 68 个最终 fused TIR 模块确认其在预训练 ResNet50 的节点 118/131/144/157/170 出现五次，
    预注册整图 LOAD `-3,010,560 B/-270 calls`。上板 6 次 correctness、14 次 timing 的输出全等
    且非零，十项 LOAD/STORE 差逐项命中；完整图中位 `346.687→330.277 ms`，吞吐提升 4.968%、
    7/7 获胜。P7R313 对 302 个链上文件和身份/哈希/统计完成独立审计。这是新的独立 workload
    整图 latency 留出，但仍共享 ResNet50 模型和一个 boot，未做 ImageNet accuracy 或物理 AXI。
76. P7R314--P7R315 将“最终 fused-program calls”真正写入第二级重排，而不是停留在 P7R296 的
    相关性描述。测试只使用 P7R166 早已冻结的 `65536 B/request`，不以 F00/F01 重拟合；original
    与 input-stationary 两项跨 tile 选择中，bytes-only 均错误，regret 分别 112.844%/16.308%，
    fused service 均选择 F00 并命中 oracle。四个程序的两两排序由 2/6 提高到 5/6，但仍错排
    F00 original 与 F01 residency。因此该层可作为少量晋级 tile 的 final-fusion 高保真 acquisition，
    不能替代真实 FPGA latency。目标标签在测试规范前已暴露，所以这是 no-refit post-hoc 审计，
    不是第三个 prospective call-site holdout。
77. P7R316--P7R327 完成 fixed service proxy 的前瞻可证伪检查。F01/F02 两个新 R50C tile
    只按既有算子三 seed correctness 入选，operator/full-graph latency 均未参与选择；最终 fused TIR
    分别给出 input-stationary `38,097,920 B/980 calls` 与 `6,517,760 B/2,900 calls`。冻结的
    `B+65536N` 选择 F01；上板 6 次 correctness、14 次交错 timing 全部输出相等非零，十项逻辑
    DMA 差精确命中，但 F01/F02 中位为 `375.140/350.550 ms`，选择 regret 7.015%、0/7 获胜。
    P7R325 因计时 profiler 双执行未除二而在发布前 fail closed；P7R327 证明全部 raw delta 恰为
    2 倍并保留该失败，P7R326 仅修复计数归一化且未改变候选或系数。这一负结果要求将正式策略改为
    多目标联合支配与冲突升级，不能再宣称 64 KiB 请求系数跨 tile 泛化。
78. P7R328 将修正后的 Pareto 动作实现为可执行审计。在 R50A original、R50A residency、R50C
    三 tile input 与 R50C F01/F02 冲突对四个比较组中，真实 oracle 均保留在 `(bytes,calls)` 前沿；
    前三个两 tile 冲突组中有三个需要升级测量，R50C 三 tile 组则由 F00 同时支配 F01/F02，前沿
    从 3 缩为 1 且保留 oracle。bytes-only/service scalar 分别只命中 2/4、3/4。四组相互重叠且标签
    已暴露，所以这是 post-hoc 规则设计证据，不是四次独立 holdout，也尚未证明在线搜索成本收益。
79. P7R329--P7R335 在全新 R50D (`CI=2048,CO=512,H=W=7,1x1`) 上完成首次 final-fused Pareto
    前瞻留出。768 点完整域冻结 8 family 后，24 个正式身份有 19 个 lowering-pass、18 个三 seed
    FSim-pass；对这 18 点及 stock 各做一次完整预训练 ResNet50 构建，656.866 s 后按最终图中两个
    真实出现点的 `(LOAD+STORE bytes,calls,extra submissions)` 冻结 4 点前沿。前瞻上板中 2/4
    正确、两个 weight barrier 首 seed 错误；标签暴露后补测被支配 14 点，13/14 正确。完整 15 点
    FPGA-correct oracle 为前沿中的 R50DF07 input-stationary，配对 stock ratio 1.096984，front regret
    为 0。相对完整穷举，前瞻波次 candidate dispatch `4/18` (-77.78%)、graph API `44/306`
    (-85.62%)、driver invocations -86.59%、逻辑 DMA bytes -87.99%、calls -92.29%。但 static/FSim
    `42.924/3.509 s` 与全候选 fused build `656.866 s` 都是公共成本；bytes+fail-fast 与 Pareto 达到
    oracle+2% 都需 2 次派发，只有 exact oracle 为 6 对 3。因此可主张“高成本硬件候选安全缩减”，
    不能主张端到端搜索已经更快或全面优于简单 bytes 规则。R50D 还显示 original 8/8、input 5/5、
    weight barrier 仅 2/5 FPGA-correct，进一步证明 FSim 合法和最小访存都不能替代最终硬件认证。
80. P7R336 用标签已暴露的 R50D 开发一个廉价前置代理：仅以算子 lowered-TIR 的 expanded
    bytes/calls 和 weight-barrier 标志构造三轴前沿。代理也是 4/18 点，与 exact fused 前沿重合
    3 点；漏掉的是 FPGA-invalid weight 点，换入 FPGA-correct weight 点，并保留 pool oracle。按已
    暴露的实际构建耗时回放，stock+候选构建 `656.866→169.592 s` (-74.18%)，候选 full-graph build
    `18→4`。这是 post-hoc 规则开发，不是实际前瞻节省；必须在下一未见 workload 中先冻结代理前沿、
    只构建该前沿，再于标签暴露后补齐完整 oracle，才可升级主张。
81. P7R337--P7R348 在新 R50E (`CI=256,CO=1024,H=W=14,1x1`) 上完成上述廉价代理的首个独立
    前瞻留出。24 个正式身份经真实 lowering/FSim 得到 12 个合法候选；在任何目标完整图、FPGA 或
    latency 标签出现前，算子 expanded bytes/calls/barrier 三轴前沿冻结为 2/12。实际只构建 stock
    与这两个前沿点，耗时 100.120 s；前沿上板 2/2 正确。标签封存后才构建和测量剩余 10 点，完整
    池 12/12 FPGA-correct，oracle 仍是前沿首点 R50EF03 input-stationary，配对 stock ratio
    1.128623、front regret 0。相对 stock+全部候选 442.808 s 的反事实实际构建总和，前瞻构建墙钟
    减少 77.39%；加回 20.567 s static 与 2.263 s FSim 公共成本后，选前本地墙钟仍由
    465.638 s 降到 122.950 s (-73.60%)。candidate build/dispatch 均由 12 降到 2，逻辑 DMA bytes
    减少 90.62%、calls 减少 84.56%。该结果关闭了 R50D
    “必须先构建全池”的主要端到端缺口，但只是一项 ResNet50 stage3 1x1、单 boot、12 点池验证，
    且所有候选仍慢于 stock。五个 input same-tile 对全部加速；一个 weight 对即使 DMA bytes/calls
    下降 29.98%/37.14% 仍慢 1.09%，继续证明代理是搜索先验而非 latency 定理。
82. P7R354--P7R378 完成第三个网络上的冻结 Y08 验证：YOLOv3-tiny conv12 的 24 个预注册身份中
    6 个通过真实 lowering 与三 seed FSim；P7R357 在任何 Y08 FPGA/latency 标签前冻结顺序，
    P7R377 在新 boot 上得到 6/6 三 seed FPGA-correct、42/42 交错计时正确，完整池 oracle 为
    Y08F06 input-stationary 的 `132.920140 ms`。三个 same-tile input-stationary 配对全部加速
    27.71%、36.22% 和 3.70%。bytes/calls/Pareto 顺序均首测命中精确 oracle；但唯一冻结 Random
    顺序首点也已进入 oracle+2%，所以该池只支持 exact-oracle time-to-target 与第三网络机制结论，
    不支持 success@2% 优于随机。Y08F01 的 bytes 仅 -1.30%、calls -92.31%、latency -27.71%，
    且两个最优 input 候选总 bytes 相同而 latency 不同，再次确认 bytes 是冷启动先验而非性能定理。
83. P7R379--P7R380 在独立新 boot 上原样复用 R50CF00 的预训练 ResNet50 图、参数、A/B DSO 与
    fused-TIR 合同。新启动得到 `346.912→330.290 ms`、吞吐提升 5.033%、7/7；旧启动为
    `346.687→330.277 ms`、4.968%、7/7。两启动合计 12 次 correctness 和 28 次 timing 均输出
    相等非零，十项 LOAD/STORE 差均精确命中。因此可主张该 frozen pair 的整图收益跨启动方向与
    量级稳定，但不能把 14 个配对轮当作 14 个独立启动或外推为 ResNet50 通用加速率。
84. P7R381--P7R384 对 R50D 同一 18 点最终融合池做第二启动完整复验，得到与旧启动相同的
    15/18 FPGA-correct 分类，但暴露出固定 Pareto 前沿的边界：旧启动前沿保留精确 oracle，当前
    启动精确 oracle 变成前沿外 R50DF04 weight-barrier。前沿最佳按绝对 latency 仅慢 0.0327%，
    按预注册 paired-ratio 指标 regret 0.1706%，所以 2/2 启动仍保留 oracle+2%，却只能写 1/2
    保留 exact oracle。根因是当前 oracle 只被一个前沿内 FPGA-invalid weight 候选静态支配；移除
    两个失败前沿点并重算时，唯一新增点恰为当前 oracle。这提出 invalid-dominator peeling 的后续
    多保真动作，但它来自标签已暴露的开发审计，必须在新 workload 先冻结后才能成为方法贡献。
85. P7R385--P7R388 保留了一次必要的目标撤销。P7R385 把仓库 ResNet50-v2 的
    `stage2_unit1_conv2` 人工声明为 56x56、3x3/stride-2；P7R387 只做了本地 lowering/FSim，尚未
    接触板端。P7R388 用同一 Relay 模型的 InferType 证明该层实际输入为 28x28、stride-1，而且
    几何与已有 W02 标签重合，因此将 P7R385--P7R387 标记为“合成算子本地负例、禁止作为
    ResNet50 holdout”。候选冻结器随后增加模型源几何自动核验，声明不一致时直接 fail closed。
86. P7R389--P7R392 重新冻结真正未见的 R50G：`stage2_unit1_conv1`，CI256→CO128、输入 56x56、
    1x1/stride-2/pad0。完整域 2880 点、解析谓词通过 435 点，固定哈希选 8 个 family/24 个生产
    身份；真实 lowering 和三 seed FSim 仅保留 5 点。算子 `(bytes,calls,barrier)` 前沿在任何整图、
    FPGA 或 latency 标签前冻结为一个 input-stationary 候选，其余四点的支配关系和 bytes/calls/
    Random 控制顺序同时冻结。P7R393 还揭示模型源身份边界：使用 MXNet ResNet50 整图时 exact
    route 不命中；该失败无板端标签且不进入结果。P7R394 改为与几何推导相同的
    `relay.testing.resnet` 后，exact route 只落到 Graph node 48 的一个融合函数，图结构和参数合同
    通过；随机参数模型仅用于调度正确性与相对时延，不冒充 ImageNet 精度。
87. P7R395--P7R398 完成 R50G 前瞻波次和事后完整 oracle。在线单点通过 3 seed 正确性与 7 轮
    配对计时，paired ratio 0.837109；因该轮无 FPGA-invalid 点，预注册 peeling 正常停止。标签锁定
    后补测其余四点，5/5 全部正确；完整池 oracle 为另一个被支配 input 候选，ratio 0.831173，故
    在线点 exact-oracle miss，但仅有 0.714% regret，仍在 oracle+2%。在线只构建/派发 1/5 候选，
    候选 build 数与 board dispatch 均 -80%，model-build 阶段墙钟相对 stock+全池实际总和
    -68.60%；但现有板端执行器尚未把完整外层进程墙钟写入原始结果，不能把这些阶段数升级为完整
    `T_search` 结论。冻结 Random 达到 +2% 需 3 次，bytes/calls 需 1 次、精确 oracle 需 2 次；
    peeling/fixed-front 需 1 次达到 +2%，但永远不发现 exact oracle。本池没有自然 invalid
    dominator，因此它确认的是近优质量—成本交换和停止边界，不是 peeling fallback 的正向验证。
    两个 same-tile input 对分别以 fused DMA bytes -82.52%/-82.26% 获得整图 4.44%/2.49% 改善；
    weight barrier 仅改善 0.18%，继续说明访问量下降不是等比例 latency 定理。
88. P7R399--P7R408 在新的源模型一致 R50H 上第一次真正触发预注册 invalid-dominator peeling。
    672 个 ConfigEntity 中解析谓词保留 125 个，固定哈希冻结 8 个 family/24 个生产身份，真实
    lowering 和三 seed FSim 得到 15 个合法点。初始三点代理 Pareto 前沿中，R50HF05 weight-barrier
    虽以 750,080 B 成为最低 DMA 点，却在第二 seed 出现 1000/1000 logits 不一致；搜索器不读取
    latency，删除该失败支配点后重算前沿，唯一新增的 R50HF01 weight-barrier 在第二波通过并最终被
    事后 14/15 正确完整池确认是 exact oracle（588.975 ms，paired ratio 0.774019）。在线 4 次
    dispatch 的完整外层墙钟为 312.942 s；相对 15 点穷举，候选 dispatch -73.33%、完整候选构建
    动作 -76.12%、FPGA kernel invocation -77.46%、逻辑 DMA bytes -78.43%、calls -78.12%。首波
    第二个正确点已经处于 oracle+0.1%，第四次找到 exact；冻结控制中 bytes-lazy 第 2 次找到 exact，
    calls-lazy 第 10 次、Random 第 13 次，固定首波永远漏掉 exact。故本轮建立了“仿真正确但板端错误
    的代理支配点—首错拒绝—自动暴露被遮蔽 oracle”的前瞻闭环；但不能写成该算法在本池全面优于
    bytes-only，控制结果仍是完整池上的冻结顺序 replay，不是独立在线运行。五个 same-tile input
    候选全部正确并加速 0.17%--6.29%；四个正确 weight 候选加速 0.65%--5.34%，另一个就是上述
    FPGA-invalid 点，进一步把共享内存收益与正确性风险同时保留。
89. P7R409--P7R418 将同一规则原样带到第二个源模型一致、标签隔离的 R50I：
    `stage3_unit1_conv1`，CI512→CO256、输入 28x28、1x1/stride-2。1920 个完整 ConfigEntity 中
    解析资格保留 328 个，固定哈希冻结 8 个 family/24 个生产身份；真实 lowering 与三 seed FSim
    最终只留下 5 点。两点 wave0 在任何目标 FPGA/latency 标签前冻结，在线 2/2 通过后按预注册规则
    停止，完整外层墙钟 213.527 s。标签后补齐全池得到 4/5 FPGA-correct，在线第二点 R50IF07
    input-stationary 正是 exact oracle（681.140 ms，paired ratio 0.895435）。相对五点穷举实测组件，
    dispatch -60%、完整候选 build action -63.41%、FPGA invocation -51.22%、逻辑 DMA bytes/calls
    -52.22%/-51.98%。冻结控制的 exact-oracle 次数为 calls 1、online/fixed-front 2、bytes 3、Random
    5，因此本池支持在线规则的成本—质量结果，但仍不支持全面优于所有单轴规则。R50IF05 的
    same-tile input 模式在 bytes 仅下降 1.14% 时因 calls 下降 72.79% 获得 7.49% 整图加速；
    R50IF07 original 则是 FPGA-invalid，故该 family 不能伪造纯机制 latency 配对。
90. P7R419 将 R50G/R50H/R50I 三个按相同时间顺序冻结的 cheap-proxy/peeling 留出合并审计：
    完整合法池共 25 点，其中 23 点 FPGA-correct；在线共派发 7 点，较穷举候选数减少 72%。三池
    均进入各自 oracle+2% 带，二池命中 exact oracle。三个不同结果分别是 R50G 的近优停止、R50H
    的错误支配点剥离后 exact 回退、R50I 的无错误前沿停止且 exact 保留。只有 R50H 真正发生第二波
    expansion，因此该汇总支持方法具有 fail-closed 分支与可复现停止行为，但不能写成“已在两个
    自然错误支配 workload 上重复剥离”；R50G 还保留 0.714% exact regret，冻结控制仍是完整池上的
    顺序 replay。
91. P7R420--P7R424 把 R50I 的三条简单控制从 replay 升级为独立完整进程。三条控制严格读取
    P7R412 在标签前冻结的 candidate order，各自固定派发两点并重新执行模型准备、stock/candidate
    构建、clean-start FPGA 正确性与计时；运行中不读取 pool oracle。peeling/calls/bytes/Random 的
    完整外层墙钟分别为 213.527/212.812/214.571/199.165 s。peeling 与 calls 在两点预算内命中
    exact，calls 的首次命中为 146.179 s、早于 peeling 的 213.311 s；bytes 与 Random 均未进入
    oracle+2%。Random 第二点复现 FPGA-invalid，因此只有 8 次 correctness、14 次 timing，而前三者
    各为 12/28；其墙钟更短不能解释为同质量优势。本实验关闭的是独立 complete-wall 对照缺口，
    同时进一步限制主张为“在多种静态启发式互有胜负时提供 fail-closed 多目标搜索”，而非普遍领先。
92. P7R444 将最终部署定容从旧 Y00 图更新到当前 Y10 完整池 oracle。exact YOLO-320 图在三个输入
    的 36 次提交中实测 instruction/UOP 单提交峰值为 1,065,280/1,320 B，页对齐容量为
    1,069,056+4,096 B，共 1,073,152 B，较默认 64 MiB 命令 backing 减少 98.4009%。缩容后三个
    输入的全部 8 个图输出精确匹配；instruction 再少一页时，前 5 次合法提交完成，下一条超限
    命令在送入 FPGA 前拒绝，之后默认 RPC 与整图健康推理恢复。该结果仅适用于绑定的静态图、
    二进制、runtime 与 bitstream；它不释放整块 192 MiB，也不是动态 shape 或物理 AXI 结论。
93. P7R452--P7R460 完成 Y10 空历史 AutoTVM-XGB 与本文方法的三对三 clean-start 系统 A/B。
    XGB 三轮在 600 s 搜索预算内提出 173/173/157 个配置，均选中 original config543；从共同 T0
    到完整图三输入/八输出正确性与七轮计时 T1 的时间为 696.427/695.965/699.170 s。本文三轮重演
    同一 1280 点 tile 域，生成身份逐行匹配原冻结合同，24 个正式模式中 12 个 lowering/FSim-pass，
    仅派发三个完整图候选；T0→T1 为 244.844/244.835/245.926 s。按中位数，端到端墙钟降低
    64.84%，候选程序派发由 115 降至 3（-97.39%），在 120.279 s 已得到 XGB 最终图 +2% 内的
    可部署整图。最终 latency 为 XGB 1054.773 ms、本文 1057.774 ms，差 0.285%；两边三轮均
    3/3 输入、全部 8 输出等价且 7/7 配对获胜。负向成本是本文完整图安全门的逻辑 DMA 每轮约
    14.486 GB，高于 XGB 可审计下界 5.052 GB；粒度不同且不是物理 AXI。该结果支持单 Y10 系统
    time-to-quality，不是新 prospective holdout、同空间算法归因或跨网络泛化。“原生 XGB”只指
    官方空历史搜索器与原始空间；因 P7R447 已证明 fatal 配置会污染共享 RPC，正式基线使用三 seed
    正确性和逐候选 bitstream/RPC clean-start 安全适配器，64.84% 也包含这一部署可靠性代价。
94. P7R462--P7R468 在相同 Y10 12 点 `tile×mode` 候选池和相同 budget=6 上完成 DMA prior 与
    mode-aware XGB 各三次真实完整图重执行。36/36 个候选派发全部通过正确性门。两种方法均在
    第一次派发进入 pool oracle+2%；DMA prior 三次均在第 3 次命中 exact oracle，XGB 为第
    4/6/6 次。exact time-to-quality 中位数由 396.797 s 降至 212.564 s（-46.43%，提前 1.87×）。
    两边强制执行满六点后的总墙钟中位数分别为 400.933/398.963 s，故不能声称固定终止预算的总
    成本更低。该实验消除了候选空间、候选预算和完整图执行路径差异，支持 Y10 上的搜索次序归因；
    但完整池标签在协议前已存在，只能称已暴露 workload 的独立成本重演，不能称新前瞻留出。
95. P7R469--P7R470 完成文献对齐实验的本地实现与预冻结。统一 runner 覆盖 Random、mode-aware
    XGB、Rieber 初始化、ML²Tuner P/V/A、Cheng minimum-access 和本文多保真方法，并以单向结果
    ledger 隔离未来标签。新增 `input_weight_resident_barrier` 在三个可容纳整层权重的历史配置上
    同时降低 input/weight DMA，并通过 9 次 FSim 数值检查；资源超限明确 not_applicable。三个新
    ResNet18 几何的真实完整 original ConfigSpace 为 2304/1600/480，不是统一 1280；各自以无标签
    max-min 冻结 24 tile×4 mode=96 proposal，总计 288。相关回归 62/62 通过。该节点没有执行目标
    288 身份的 lowering/FSim 或任何上板计时，只证明方法、身份和协议已冻结，不产生性能主张。

## 6. CCF-C 级必要实验

### E1：tile—共享内存机制图（算子级已完成）

在至少三个卷积几何上报告 tile 与 `input/weight/output bytes`、DMA calls、request shape、SRAM、
instruction/UOP 的关系；包含正确点和失败点。主消融采用 same-tile 的 original、input-stationary
和 `weight_resident_barrier`，回答什么几何适合 input reuse、什么几何适合 weight reuse，以及
同步代价何时抵消访问下降。旧 safe-weight/hybrid 只作补充反例。

### E2：等预算搜索（旧系统 A/B 已完成；文献基线与新 ResNet18 协议已冻结）

现有 Y00/Y03/Y04 完整池与 Y01 latency-unseen recovery 已比较 Random、原始 XGB、minimum-access
rules 和本文“合法性 + 共享内存服务代价”。P7R173--P7R194 又实现逐阶段部分可见的在线调度，先以
Y05 开发前沿，再在 Y06/Y07 与 ResNet50 R50A/R50B/R50C 新冻结池完成完整 oracle 审计；
Y06/Y07/R50B/R50C 确认 sparse 分支，R50A 以 10 个 FPGA-correct 候选确认 dense 分支。P7R306
给出五个规则冻结后严格 holdout 的配对汇总。继续报告
success@pool-oracle+2%/+5%、regret、gross candidate、
compiler/FSim/FPGA correctness/timing 调用数、搜索期逻辑 DMA 与墙钟；公共24→11预资格开销单列，
不能只报告上板次数。TopHub 只作为通过当前硬件
正确性资格后的附加质量线，不作为搜索器必须超过的目标。CCF-C 目标不是必须把所有成本减半：
多数 workload 上达到同等质量所需上板测量或墙钟显著减少，且最终质量不劣于原始 XGB，即可形成
主结果；若复杂模型不优于简单规则，则以规则型方法定稿。继续增加网络、几何和独立启动属于外部
有效性增强，不再是证明“在线多保真方法确实执行过”的缺口。P7R452--P7R460 已按 P7R445 从空
history 执行三轮原生 AutoTVM-XGB 与三轮本文方法，并把选中配置部署到同一 YOLO 完整图。共同
T0→T1 中位由 696.427 s 降至 244.844 s，最终图只慢 0.285%，因而“从头 tune 代价与最终整图
质量”的系统级主结果已经获得。P7R462--P7R468 又让两边在完全相同的 12 点 `tile×mode` 宇宙、
相同 budget=6 和相同完整图执行路径中独立在线运行：DMA prior 的 exact time-to-quality 中位数
比 mode-aware XGB 低 46.43%，但两者第 1 次均进入 +2% 带，执行满预算后的总墙钟也近似相等。
因此可以把“更早找到 exact oracle”归因于本池的搜索次序；P7R460 的 64.84% 仍是候选生成、资格、
搜索空间与部署门共同作用的系统流程收益，不能整体改写为搜索算法加速。

### E3：真实 FPGA 正确性与性能（算子级已完成）

所有进入性能池的候选先三 seed correctness，再做不少于五轮平衡交错计时。至少两个未见几何得到
可用完整池 oracle；至少一类复用规律出现方向一致的 DMA 与 latency 改善。失败候选完整保留。

### E4：共享内存部署闭环（支撑实验，已完成）

对最终 selected/TopHub fallback allowlist 生成 instruction/UOP 容量，报告张量 slot、命令 backing
和整个 192 MiB u-dma-buf 预留之间的区别。至少做一次 runtime manifest attestation 和不足容量
拒绝测试。

早期 16 KiB instruction + 4 KiB UOP 结果只对其精确单 workload allowlist 成立；P7R252 因此正确
地把 YOLO 整图容量标为未推导。P7R259 随后在 exact stock/Y00/Y02/both 图上实测峰值，P7R260 用
四变体 allowlist 容量 155,648+8,192 B 消除了四 Executor OOM，P7R261 再对最终 selected Y00 图
以 155,648+4,096 B 完成 3/3 正向输出匹配和 instruction 少一页的负向拒绝。至此“选谁—哪个二进制
—命令多大—不足时怎样拒绝”已经贯通。P7R444 又对当前 Y10 完整池最终 oracle 重新发证：整网
实测峰值为 1,065,280/1,320 B，页对齐容量 1,069,056+4,096 B，三 seed 的 8 输出全部精确，
instruction 少一页在第六次提交前 fail closed。旧 Y00 容量比 Y10 小约七倍，进一步证明形状、图、
route、源码或硬件身份改变时必须重新推导，不能发布一个 VTA 通用小容量。

### E5：整网或 stage 传递（ResNet50、YOLO 及图节点组合/成组准入均已完成）

P7R231--P7R232 已将 R50B 资格化的 exact workload/config 通过显式 dispatch 放回预训练 ResNet50-v2
完整 Relay graph；四个同 workload 实例统一替换后，三 seed 非零 logits 逐元素一致，整网 `run`
中位提升 0.330%、7/7 配对轮更快，DMA 差值严格对应四个实例。它完成“算子候选能进入真实模型图”
的首次证据，但不是 ImageNet accuracy 或物理 AXI/FPS 结果。P7R264--P7R265 进一步只替换四个相同
workload 节点中的节点 57，得到恰好一份孤立算子的 DMA 差值并保持全部输出相同，证明部署期可以
按图节点独立控制；其 0.088%、4/7 不构成加速。该原型依赖精确 Graph JSON 和二进制组合，尚未把
call-site 身份原生送入 Relay/AutoTVM 搜索键。

P7R266--P7R296 进一步把单节点控制推进到多调用点搜索。R50B 的四点 DMA 剂量严格线性，但延迟
非单调；严格逐点 7/7 准入把四点全拒绝，而全四点 bundle 反而稳定快 1.000 ms、9/9。R50A 则先用
最终 fused TIR 修复孤立模板遗漏的 ACC 请求，再得到三点整图 2.165%、7/7 正收益；同一严格门下，
逐点与整组都选到三点，但整组只需 20 而不是 60 次整网执行。由此形成的层次化 bundle admission
以静态 DMA 可加性决定“先把谁一起测”，以真实 FPGA 正确性和边际 latency 决定“是否接受”，并在
整组失败时递归拆分。P7R280 已用既有 YOLO 四变体作强制异构 bundle 的回顾性负向控制，跑通“组失败—接受
Y00 子组—拒绝更新 incumbent 后的 Y02 子组”，但该组合不满足正常 same-mode call-site 分组规则，
不能代替未见自然失败域。P7R281--P7R288 又在规则和门槛冻结后，对确定性选出的新 tile R50AF00
完成首次调用点/完整图 latency holdout：逐点与整组都选择三个节点，整组验证把整网执行从 60 降至
20，且获得 12.776%、7/7 的净收益。该确认仍与开发域共享 ResNet50、workload 和 boot，且历史
算子标签存在，因此只能称新 tile 的调用点延迟留出，不能称独立模型/工作负载泛化。P7R289--P7R296
又用相同协议完成第二个 tile R50AF01，整组同样以 20 次执行复现 `111` 选择，并形成一个更关键的
完整图排序边界：F01 虽在 original/residency 两种模式下都搬更少 LOAD bytes，却因请求碎片远多而
始终慢于 F00。因而 bundle 形成可以利用 DMA 可加性，但跨 tile 排序必须同时看 bytes/calls 并保留
真实 FPGA latency 门。

P7R234--P7R238 又在 YOLOv3-tiny 上完成第二网络验证：same-tile input-stationary 相对 original 的
整图收益为 0.758%，相对 stock TopHub 图为 1.243%，两项均 7/7 更快；三类输入的 8 个非零输出
张量全部 A/B 相同。P7R240--P7R249 随后完成 Y00/Y02 双 workload 的联合替换和 2×2 边际审计：
Y00 在有无 Y02 时都保留约 3.4--4.3 ms 收益；Y02 在有无 Y00 时都增加约 78--79 ms，逻辑 DMA
interaction 为零。它证明多个 route 的访问变化能够组合，但局部 same-tile 正收益不保证相对当前
incumbent 的绝对部署收益。仍未做 COCO mAP、跨 boot、三个以上 route 或组合搜索的全局最优保证。

P7R308--P7R313 又在独立于 R50A/R50B 的新 `R50C` workload 上闭合整图留出：冻结的多保真搜索
首个测量候选先在 10 点正确池中确认为算子 oracle，再在不知道整图 latency 的条件下进入最终 fused
TIR；五个真实图节点聚合预测 LOAD `-3,010,560 B/-270 calls`，板端十字段逐项命中，预训练
ResNet50 完整图 `346.687→330.277 ms`、吞吐提升 4.968%、7/7。这第一次在同一新 workload 上把
“搜索到谁—为什么减少共享内存访问—融合后实际改了多少—整图是否获益”连成一条证据链；但模型、
boot 与 tile 仍各只有一个，不能外推为 ResNet50 平均收益。

## 7. 停止条件与论文口径

- 若一个几何的冻结池没有任何 lowering/FSim 合法候选，先做编译合法性归因；允许在没有看目标
  latency 的前提下修正规则并发布新版本合同，但必须保留旧失败合同。
- 若 request/command 特征不优于 bytes/calls，删去它们的性能预测主张，只保留诊断和部署用途。
- 若新模式始终不接近 TopHub，仍可报告复用规律、失败边界和无效试验减少；但不能写性能搜索成功。
- 若 TopHub exact hit 未通过当前硬件正确性，则不得为了保留基线而跳过 canary；改用已资格化
  incumbent 或冻结池的正确候选 oracle，并单独报告 TopHub portability failure。
- u-dma-buf 驱动本身不需要强行修改。创新落点是“如何使用这块共享内存以及如何由编译结果决定
  它的访问和安全容量”，而不是给 ko 模块添加缺少需求依据的功能。

达到 E1--E4 后，这个创新点才具备一篇 CCF-C 风格 case-study/system paper 所需的完整问题、方法、
基线、消融和真实硬件证据。其中 E2 是决定论文高度的主实验，E1/E3 证明搜索依据真实，E4 只负责
部署闭环；不能用定容结果替代搜索效率结果。

## 8. 当前水平（P7R468 后）

现在有十五个在目标标签前冻结并完整上板测完的候选池：Y00、Y03、Y04 用来建立并反驳静态排序
规律，Y05 用来开发多保真前沿，Y06/Y07/R50A/R50B/R50C 是生存率自适应规则冻结后的严格确认；
R50D/R50E 则分别确认最终融合 Pareto 和廉价算子代理的按需构建路径；Y08/R50G 增加跨模型机制
与 cheap-proxy/peeling 停止边界，R50H 首次确认错误支配点剥离与第二波 exact-oracle 回退，R50I
则作为第二次完整在线运行确认无错误前沿的停止分支仍能保留 exact oracle。Y04 仍是必须保留的
关键反例：bytes-only 首派发
regret 为 75.35%，因此最终方法不是“字节最少必然最快”，而是服务代价、候选存活率和真实硬件
正确性共同决定下一步付费动作。

P7R173--P7R226 已经补上上一阶段最关键的算法缺口：搜索器不再把全池 lowering/FSim 当作免费公共
前置，而是在候选和保真度两维上选择下一动作。Y05 的固定 4→2 策略第一次测量即命中 16.305 ms
oracle，但需要 23 次 lowering，暴露稀疏空间浪费；据此只在开发集冻结生存率阈值后，严格未见 Y06
观察首 family 0/3 存活并切换 sparse family-wave，以 6 次 lowering、1 次 FSim、1 次编译、1 次
FPGA gate 和 1 次测量命中 30.103 ms 完整池 oracle。统一成本相对全池资格降低 26.95%，相对固定
4→2 降低 29.16%，gross candidates 由 24 降到 6。Y07 在第二个稀疏 YOLO 3×3 池重复该路径；
R50A 则在真实 ResNet50 bottleneck 1×1 的 10 点正确池触发并确认 dense 4→2 分支，第一次测量仍为
5.907 ms oracle，相对全池资格墙钟降低 30.35%。R50B 随后在 1×1 扩张层上再次触发 sparse 路径，
第一次测量的 weight barrier 仍是 25.151 ms 完整池 oracle，相对全池资格墙钟降低 52.36%。四个严格
holdout 的早期配对汇总为 96 个预注册身份。R50C 随后在新 `CI1024→CO256,14x14,1x1` workload
上再次触发 sparse 路径，以 6 次 lowering 和一次后续各级付费命中 4.442 ms oracle。五个严格
holdout 的 P7R306 汇总为 120 个预注册身份，自适应中位只考虑 34 个，五池全部到达各自 oracle；
相对全池资格 gross -71.67%、墙钟 -41.39%，相对固定策略墙钟 -17.57%。

机制与部署证据仍作为完整系统的两侧支撑：Y00/Y03/Y01 的多个 input pair、Y02 跨两启动的 weight
barrier 均有同 tile 正收益；P7R215 又在 R50A 的五个 same-tile input pair 上得到 5/5 加速、中位
62.80%，且输入字节和 DMA calls 同向下降。R50B 的唯一合法 weight pair 则只加速 2.31%，但仍以
总 DMA 字节 -15.95% 和 calls -42.86% 给出另一种访问规律。Y04 和若干 FPGA-invalid 身份给出失败
边界。P7R227--P7R232 又补上此前最后一个主要集成缺口：exact workload/config 已能 fail-closed 地
选择 residency schedule；最小 Relay 图复现 R50B 的 2.36% 局部收益，预训练 ResNet50 整图中四个
同 workload 实例保持三 seed 非零 logits 逐元素一致，整网 `run` 提升 0.330%、7/7 胜，逻辑 DMA
差值与四个实例严格对应。P7R234--P7R238 随后在 YOLOv3-tiny 上重复整图传递：单个 `conv2`
input-stationary 的 input LOAD 差值与独立候选严格相等，same-tile 整图提升 0.758%，并以 1.243%
通过 stock TopHub 整图参考线，两项均 7/7 更快且八个非零输出张量全等。单 workload exact allowlist
的 `16 KiB instruction + 4 KiB UOP` 和最终 YOLO selected graph 的 `155,648 B instruction +
4,096 B UOP` 均已完成正向与不足容量 fail-closed，且容量、物理布局、bitstream 状态和 clean start
被共同写入执行合同。这些保证搜索结果能安全落到 u-dma-buf，但不冒充搜索算法，也不把两个不同
作用域的容量相互复用。P7R452--P7R460 又首次用三对三完整进程把这些环节共同计入同一 T0→T1：
Y10 上本文中位 244.844 s、空历史 XGB 中位 696.427 s，端到端减少 64.84%，最终整图只慢
0.285%。这使“调优总时间与最终整图质量”的主数据不再依赖分项求和或冻结池 replay。

P7R240--P7R249 又把整图证据从单 route 推进到多 workload 组合。Y02 barrier 相对同一 tile 的
original 在整图仍加速 10.05%，其 10,616,832 B weight LOAD 降幅和孤立算子完全一致；然而它相对
TopHub Y02 tile 仍使整网慢约 23%。完整 2×2 边际审计进一步显示 Y00 在有无 Y02 时均有收益、Y02
在有无 Y00 时均有代价，且逻辑 DMA 交互严格为零。因此多 route 本身没有破坏访问语义，真正缺少
的是相对受保护 incumbent 的部署准入。P7R250--P7R251 已将“全图正确性 + live memory fit + 配对
边际 latency”三重 route gate 落成证据哈希绑定、基线连续性检查的规划器，按顺序接受 Y00、拒绝
Y02；P7R252 又把结果绑定到实际上板二进制和硬件状态的 exact graph manifest。当前仅有两个 route
的贪心证据，不声称任意多 route 的全局组合最优。
P7R259 已进一步精确解释四个完整图 executor 为何耗尽 192 MiB：四份 graph backing 占
137,203,712 B，默认首个 32 MiB 队列把水位推到 170,758,144 B，第二个 32 MiB 队列还差
2,985,984 B。P7R260 将四变体命令 backing 定容为 163,840 B 后，原先 OOM 的完整 2×2 因子实验
全部跑通；P7R261 又把最终 Y00-only 图缩到 159,744 B，并用少一页负例证明 runtime 在越界 submission
前拒绝。因而替代版本/工作区的空间预算不仅已纳入合同，也有了可审计的正反实验。

P7R262--P7R296 又关闭了 ResNet50 的 workload 广播限制，并找到一个新的组合搜索问题。编译器查询
ordinal 被证明不是稳定 call-site；最终实现绑定 Graph JSON 节点、原/别名函数与两份已资格化 DSO。
R50B 的四点实验说明逻辑 DMA 严格可加而 latency 不可加，保守逐点 greedy 会把四个低信噪比正边
全部拒绝，却漏掉 9/9 更快的整组。R50A 的三个 input-stationary 节点则用最终 fused TIR 精确预测
板端十项 LOAD/STORE 差值，整图提升 2.165%；成组准入与逐点得到相同选择，但把验证整网执行从
60 降为 20。因此第三点现在不仅有“候选×保真度”的算子级搜索，还有“相同驻留 route 应逐点还是
成组付费”的整图级动作。P7R280 还用既有 YOLO 四变体完成强制失败 bundle 的递归拆分回放，既
证明能恢复安全子集，也如实显示 pair tests 可能比逐点法多 50%。P7R281--P7R296 随后把冻结规则
带到两个新 tile 的调用点/完整图 latency holdout：R50AF00/F01 的逐点与整组均选择 `111`，整组都
只用 20 而非 60 次整网执行，完整图分别提升 12.776% 和 106.379%、均 7/7。F00/F01 还给出直接
属于共享内存搜索的反例：F01 的逻辑 LOAD bytes 更少却因 calls 多 2.61--6.22 倍而更慢，证实
`B_data` 不能替代 `N_dma L_req`。这些结果消除了“所有 bundle 结果均为 latency-exposed 开发数据”
的旧缺口，但两个留出仍共享同一 ResNet50 workload、模型和 boot；失败拆分仍只有回顾性异构负控，
节点身份也未原生进入 Relay/AutoTVM，重新融合或重编译必须重新发现并发证。

P7R297--P7R313 进一步缓解了“只有 R50A 新 tile、没有独立 workload”的限制。R50C 的完整池不仅
以前瞻首测命中算子 oracle，还在五个 same-tile input 对上形成方向一致的 DMA/latency 关系，并用
第三 seed 才出错的 weight 候选给出 correctness gate 的必要反例；同一选中 tile 最终在五个真实
ResNet50 调用点上精确实现融合程序的 DMA 预测和 4.968% 整图收益。至此算法、机制和整图传递不再
只来自彼此分离的 workload。不过它仍是单模型/单 boot 的一条纵向闭环，不能取代跨模型和跨启动验证。

P7R314--P7R315 又把 P7R296 暴露的 calls 因素接回搜索动作：晋级 tile 可以在上板前付费得到最终
fused program，再按图出现次数和冻结服务代价二次排序。在两个已有 full-graph tile 对照中，它以
不重拟合的旧系数修复 bytes-only 的两次选择错误，并把全部两两排序从 33.33% 提高到 83.33%；但
剩余的一次错序再次说明最终 FPGA gate 不可删除。该证据是 post-hoc，无资格扩大“严格 holdout=5”
的计数；其意义是把一个观察转成了下一新 workload 可前瞻执行的明确多保真动作。

P7R316--P7R327 已完成这项前瞻检查，并推翻了“固定请求等效字节可以直接重排”的强版本。两个
R50C 新 tile 的最终程序恰好形成 bytes/calls 冲突；旧系数选择的低 calls 程序比低 bytes 程序慢
7.015%，0/7 轮获胜，而所有正确性与十字段 DMA 预测均成立。故这不是编译画像错误，而是静态标量
到真实 latency 的校准失效。由此得到更稳健、也更符合多保真故事的规则：final fused TIR 负责给出
精确 `(B,N)`；若一候选同时不劣且至少一项更优，允许以 Pareto dominance 静态淘汰；若 bytes 与
calls 相互冲突，则不得用全局常数强行裁决，必须让两者进入 FPGA 测量前沿。P7R315 保留为说明
calls 不能删除的事后证据，P7R327 保留为固定系数不能万能化的前瞻反例。

P7R328 已把该规则实现并在现有完整融合程序上做事后审计：四个比较组的 oracle 均未被 Pareto
淘汰，其中 R50C 三 tile 前沿从 3 缩为 1，另三个冲突组则诚实保留两个点继续测。这证明规则能表达
“什么时候可以省测量、什么时候必须为可靠性付费”，但比较组重叠且标签已暴露，不能用 4/4 当作
新的泛化成功率；下一轮仍需在标签未见的新 workload 中冻结前沿再做等预算评价。

P7R329--P7R335 已关闭上述“只有事后 Pareto”的缺口：新 R50D 的 18 点最终融合程序池在目标板端
标签前冻结 4 点三轴前沿，完整补测后确认前沿保留 15 点正确池 oracle，并把 candidate dispatch 与
实际逻辑 DMA 分别减少 77.78% 和 87.99%。这个正结果同时暴露两条新边界：三个 FSim-pass weight
barrier 在完整图 FPGA 上错误；而简单 bytes+fail-fast 达到 oracle+2% 与 Pareto 同为 2 次。更大的
限制是 exact 前沿需要先为所有候选支付 656.866 s 完整图构建，所以本轮证明的是“最后一级硬件测量
缩减”，不是整条在线搜索墙钟下降。下一算法步骤应让便宜的算子 TIR/解析特征预测可能支配关系，
只对边界候选按需构建 final fused program，并以相同端到端预算与现有 adaptive/bytes 基线比较。
P7R336 已在暴露标签的 R50D 上得到第一步开发信号：算子 expanded bytes/calls/barrier 三轴代理也
用 4/18 点保留 oracle，历史构建耗时回放下降 74.18%；但这必须到下一未见 workload 才能变成
前瞻端到端证据。

P7R337--P7R348 已完成这一步未见验证。R50E 的代理前沿在目标完整图和板端标签前冻结为 2/12，
实际只为 stock 与两个候选支付 100.120 s 构建；随后补齐的 12 点 FPGA-correct 完整池确认前沿首点
就是 oracle。相对真实补齐得到的 stock+全候选构建总和 442.808 s，前瞻构建墙钟减少 77.39%；
候选派发减少 83.33%，搜索期逻辑 DMA bytes 减少 90.62%；加回 22.830 s 公共 static/FSim 后，
选前本地墙钟仍减少 73.60%。这使“便宜算子语义筛选—按需构建最终
融合程序—真实 FPGA 认证”第一次具有严格时间顺序和实际端到端成本证据，不再只是 R50D 的事后
回放。公共 static/FSim 22.830 s 已明确计入上述严格口径，不冒充免费资格。

P7R354--P7R378 又在 YOLOv3-tiny conv12 上关闭了停电前冻结池的板端断点。六个预注册合法候选
全部通过三 seed FPGA 正确性和七轮交错计时，三个 input-stationary 同 tile 配对均加速，完整池
oracle 为 132.920140 ms。冻结的 bytes/calls/Pareto 顺序均首测得到精确 oracle，但唯一 Random 顺序
首点也已进入 2% 等价带，因此该池增强的是跨模型机制证据和 exact-oracle time-to-target，不增加
“优于随机的 success@2%”计数。bytes 只降 1.30% 而 calls 降 92.31% 仍加速 27.71% 的 F01，以及
等 bytes 不等 latency 的 F02/F06，共同强化了“字节先验用于廉价筛选、冲突点交给高保真测量”的
方法边界。

P7R379--P7R384 利用本次新启动完成两项跨启动审计。R50C 同一冻结整图 pair 在两个 boot 上分别
提升 4.968%/5.033%、均 7/7，关闭了其正向整图结果仅有单启动的限制。R50D 则给出更重要的
可证伪边界：15/18 正确性分类跨启动不变，静态前沿也在两个 boot 都保留 2% 等价质量，但只在
1/2 boot 保留精确 oracle；第二启动的前沿外 oracle 仅被一个 FPGA-invalid 前沿候选支配。正式
口径因此必须把 R50D 的“精确 oracle 保留”限制在首 boot，并将 invalid-dominator peeling 作为
下一未见 workload 的待冻结算法，而不能用本次已暴露标签追认新方法成功。

P7R385--P7R398 随后第一次前瞻执行 invalid-dominator peeling，并同时补上“模型源必须与层几何
推导一致”的治理约束。错误 R50F 在上板前被撤销；真实 R50G 的 5 点正确池中，单点在线前沿用
1/5 候选和 50.494 s stock+候选 model-build 达到 oracle+2%，相对完整 160.815 s 构建总和减少
68.60%，板端 candidate dispatch 减少 80%。不过完整 oracle 位于静态支配点之外，在线 regret
0.714%；因为初始点正确，peeling 没有展开，因而不能宣称已验证“删除 invalid dominator 后恢复
oracle”。这项结果把方法边界进一步明确为：静态共享内存前沿可以作为节省成本的近优资格层，
FPGA correctness 负责失败展开，但既不能保证精确 latency oracle，也不能替代受保护 incumbent。
R50G 使用 Relay testing 的确定性随机参数，只支持完整图调度等价与相对时延，不支持 ImageNet 精度。

P7R399--P7R408 已关闭 R50G 留下的“peeling 从未真实展开”缺口。在任何 R50H 完整图、FPGA 和
latency 标签前，三点 wave0、全部支配关系及 bytes/calls/Random 控制顺序已经封存；最低 DMA
weight-barrier 通过算子 lowering/FSim 却在整图 FPGA 第二 seed 出错。删除它以后，被它单独遮蔽的
R50HF01 weight-barrier 成为 wave1 唯一新点，最终也是 14 点正确池的精确 oracle。在线执行器真实
记录从模型准备、stock build、逐点 final-graph build 到 clean-start FPGA 的 312.942 s 外层墙钟，
不再用局部阶段和冒充完整 `T_search`。这使 cheap operator proxy 的贡献从一次固定前沿筛选推进为
可运行的错误反馈搜索动作：正确的前沿点继续作为支配者，只有错误点被删除，且更新完全不使用
latency。与此同时，bytes-lazy 在相同冻结池只需两次就找到 exact oracle，而在线 Pareto peeling
需四次；因此算法价值是覆盖 bytes/calls/synchronization 冲突并对板端错误自适应回退，不是每个
workload 都比单轴启发式更省。控制仍是标签后的冻结顺序 replay，后续若追求投稿强度，应在第二个
自然 invalid-dominator workload 上重复在线展开，或把对照也作为独立在线完整墙钟运行。

P7R409--P7R419 又把完全相同的冻结规则带到第二个源模型一致的在线留出 R50I。它没有为展示算法
而制造错误：两点首波均通过，搜索按规则停止，并以 2/5 dispatch 在 213.527 s 内命中 4 点正确池
的 exact oracle；补齐阶段另发现一个前沿外 original 在 FPGA 上错误。三留出合并后为 25 点合法池、
23 点正确池、7 次在线派发，候选 dispatch 相对穷举减少 72%，3/3 达到 oracle+2%、2/3 exact。
这比只报告 R50H 更完整：它同时保留近优但漏 exact、错误反馈后恢复 exact、无需展开即保留 exact
三种运行轨迹。限制也必须一起保留——真正的 invalid-dominator expansion 仍只有 R50H 一例；
R50I 上 frozen calls 顺序一次就找到 exact，优于在线两次；R50G 的 0.714% regret 证明方法没有
exact-oracle 保证。因而正式贡献应表述为“在有限硬件试验预算下，以共享内存语义构造候选前沿，
并用真实 FPGA 错误反馈安全展开”，而不是“新规则稳定战胜所有 AutoTVM 基线”。

P7R420--P7R424 随后把 R50I 中最关键的简单控制真正作为独立进程执行，而不是继续把 replay 成本
当成实测在线成本。同为两候选预算，calls-lazy 首次测量即命中 exact，且 time-to-oracle 比 peeling
少 67.133 s；bytes-lazy 和 frozen Random 均未进入 2% 带。四条进程的总墙钟处在 199--215 s 的
相近范围，但 Random 因错误候选 fail-fast 少做一半 timing，不能直接按总墙钟评为更优。这项对照
使方法评价更可信，也说明论文不应追求“所有池都赢”：真正优势是把 bytes、calls、submission 的
冲突保留在前沿，并能在仿真正确、板端错误时继续安全搜索；若某一硬件上 calls 单轴已经恰好命中，
本文方法允许诚实承认其额外多目标保险成本。

因此当前水平可定为 **较强的硕士论文独立创新点；已经形成完整的“机制扩展—在线多保真搜索—真实
FPGA 认证—安全部署”闭环，并具备 CCF-C 风格 system/case-study 的核心证据**。与 Cheng 2026 的
核心区别不再只是工程移植，而是对驻留×tile 大空间的 search action、信息可见时间和 time-to-oracle
进行优化。Y10 已关闭“在线流程仅有 Relay-testing ResNet50”和“新 workload 没有同步独立控制”两项
重要限制：真实 YOLO 拓扑/权重、12 点完整池、8 输出等价、四条独立完整墙钟均已取得，且 peeling
以 3/12 保留 exact oracle。

边界也比此前更清楚：R50H 仍是唯一真正触发 invalid-dominator 删除与下一前沿恢复的 workload；
Y10 没有错误候选，calls 和单个 frozen Random 反而比 peeling 更早 exact；R50G 仍有 0.714% regret。
这说明方法应定位为“以共享内存语义构造小型可靠前沿，并对真实硬件错误安全展开”，而不是保证
战胜每个单轴规则。P7R444 已用当前 Y10 最终 oracle 关闭最后一个必做部署项：从 exact graph 实测
峰值推导页对齐容量，缩容后保持三 seed/八输出正确，并以少一页负控在违规提交前拒绝。至此 E1--E4
和完整图 E5 均已有分项证据，第三创新点作为硕士论文系统链路已经形成。P7R452--P7R460 又补齐
三对三从 T0 调优到完整图 T1 的直接系统 A/B：本文端到端墙钟中位降低 64.84%，以 0.285% 的最终
整图 latency 差进入 XGB 2% 等价带。因而现在可以在 Y10 case study 范围内明确回答“从头 tune
需要多少代价、最后整图性能怎样”，不再只依赖 phase sum 或冻结池 replay。

尚未关闭的限制包括：第二个自然错误展开、YOLO 跨 boot、论文式 code-exact weight reuse、原生
Relay/AutoTVM call-site 搜索、ImageNet/COCO accuracy 与物理 AXI 计数。这些现在都属于提升投稿
外部有效性的增强项。P7R462--P7R468 已完成相同候选空间的独立在线搜索器消融：DMA prior 的
exact time-to-quality 中位数低 46.43%，但双方均在第 1 次进入 +2% 带且满六点总墙钟近似相同。
P7R469--P7R470 已按冻结协议实现 HW-Aware、ML²Tuner P/V/A 与 Cheng 四方案基线，并预注册三个
ResNet18 候选域。下一优先级是节点 C 的真实 lowering/FSim/交叉编译与完整 FPGA 池，随后才能在
这个未暴露网络上重复 T0→T1。Y10 本轮是严格身份匹配且不读取旧性能标签的成本重演，不能重新包装
成 prospective holdout；也不应继续利用其已暴露标签调整方法或追逐 TopHub。
