# C3 CCF-C 共享内存主线：论文主张独立审计

> P7R349 最新校正：H90 的前瞻前沿保留和模型构建节省有效；“端到端”措辞收窄为已计时阶段和，
> 因为模型准备、筛选、RPC 重载/上传/分配等外层成本未完整计时。同目标回放中 calls 逐点构建
> 99.155 s 优于原先前沿全部预构建的 134.684 s；哈希顺序反转后首点命中消失。后续必须同时报告
> 前沿筛选、内部排序与付费时机的消融。详见 `20260913_P7R349_EQUAL_TARGET_COST_AUDIT.md`。

> 更新（P7R348）：本文件早期正文冻结在 P7R126，仅保留为历史审计。其“Y01 完成部分 0/23”和
> “搜索闭环未成立”均已失效。当前口径以 `C3_CCF_C_SHARED_MEMORY_INNOVATION.md`、
> `CLAIM_LEDGER.md` 的 H81--H90 以及最新 session 文档为准：五个严格生存率多保真 holdout 已完成；
> R50D 的 final-fused 前沿保留 oracle；R50E 又前瞻确认 cheap operator proxy 用 2/12 个候选保留
> 完整正确池 oracle，并实际减少 77.39% 模型构建墙钟和 83.33% 候选派发。

日期：2026-09-13  
审计范围：`C3_CCF_C_SHARED_MEMORY_INNOVATION.md`、P7R117--P7R126，以及既有 C1/C2 证据边界。  
结论状态：`SINGLE_GEOMETRY_BOARD_MECHANISM_CONFIRMED; GENERALITY_AND_SEARCH_NOT_CLOSED; EXECUTION_PLANE_RECOVERY_REQUIRED`

## 1. 总结判定

C3 可以形成独立于 C1/C2 的第三项贡献，但当前最稳妥的对象不是“又减少一块内存”，而是：

> 面向固定 VTA 硬件，提出一条共享内存访问感知的分层调优与部署链：把 residence mode、
> lowered-TIR 的 DMA/请求/命令签名、编译与 FSim 合法性、等 gross-budget 的无泄漏派发，以及
> exact-allowlist 命令容量证书统一起来，使 tile 选择同时回答“共享内存怎样被访问、候选是否安全、
> 最终需要多少命令 backing”。

这条主线与双 slot 的对象不同：C2 优化的是 **Executor 边界上的张量所有权和物理交接**；C3 优化的
是 **固定 VTA 算子内部由 tile/驻留产生的逻辑 LOAD/STORE 与命令流，并把选择结果带到安全部署**。
P7R123 已将这条主线从“只有本地签名”推进到一个真实 FPGA 正例：在 Y02B00 的同 tile 配对中，
`weight_resident_barrier` 相对 original 的中位 latency 从 140.988870 ms 降至 105.349023 ms，改善
25.28%，7/7 个平衡配对轮次均获胜，同时 weight LOAD bytes 下降 92.31%、总 LOAD bytes 下降
54.48%、LOAD calls 下降 49.84%。这是强的单几何机制证据，但仍然只是 **一个 family 的一对候选**，
不是搜索算法、跨几何泛化、TopHub 等价或整网 FPS 证据。

同时，P7R120--126 暴露了比性能更优先的正确性缺口：当前 Y02 TopHub 在本地 FSim 正确、lowered TIR
一致的两种入口下均在板端错误；Y01 本地 24/24 FSim 正确，但已完成板端检查的 23/24 个身份全部
错误；P7R126 最后因 RPC broken pipe fail-closed，余下一个身份未分类。因而当前 C3 的等级应判为
“机制正例成立、方法和负证据可写、CCF-C 完整搜索闭环未成立”。

## 2. P7R120--126 新证据等级

| 证据 | 审计判定 | 对 CCF-C 创新点的作用 |
|---|---|---|
| P7R120：Y02 六身份合同在 candidate dispatch 前被 sealed TopHub wrong-answer canary 阻断 | 正确的 fail-closed 行为；0 个候选、0 条 latency | 支撑实验治理，不支撑候选性能或搜索效率 |
| P7R121：source TopHub 与 mode-0 adapter lowered TIR 一致；run02 本地 FSim 均通过、板端三 seed 均失败 | 排除了“adapter 单独改坏 TIR”的简单解释；证明当前 TopHub 在该板端环境不是合格 reference | 当前禁止计算 Y02 `regret-to-TopHub`、success@TopHub+2%/5% 或称 TopHub fallback |
| P7R122：W05 health gate 3/3 正确；Y02 六身份中仅 Y02B00 original 与 barrier 通过，2/6 | 证明板端不是全局失活，也证明 local/cross qualification 不能替代 FPGA correctness | 强化“逐候选硬件正确性门”的必要性，但暴露现有 validity gate 精度不足 |
| P7R123：Y02B00 同 tile original/barrier，7 轮均正确且 barrier 全胜，median 改善 25.28% | **强单机制/单几何证据**；同步从 2 增至 34 次后仍显著获益，访问下降与 latency 方向一致 | 可作为 C3 的核心机制图和 thesis 正例；样本单位仍为 1 个 family，不能作平均收益或一般规律 |
| P7R124：Y01 v2 本地 8 family×3 mode，24/24 static/FSim 通过 | 合格的无板端标签冻结池 | 只支撑池构造；后续板端结果表明它不能作为硬件合法池 |
| P7R125：预选 2 family×3 mode，6/6 身份、18/18 seed checks 全部 wrong answer | 健康 W05 canary 同时通过，故不能归为板卡整体失活 | 禁止 Y01 timing；original 也错误，不能将失败特指 input/weight residence |
| P7R126 run01：剩余 18 身份完成 17 个，51/51 seed checks 全 wrong；第 18 个执行时 broken pipe | 与 P7R125 合并为 **完成 23/24，完成部分 0/23 pass**；最后一个为 unknown，不是 failed | Y01 没有可用性能池；broken pipe 是执行面故障，不能混入 candidate-invalid 统计 |
| P7R126 run02：只有重新冻结/资格材料，没有观测结果 | 未执行结果，不得与 run01 拼成完成实验 | 必须等执行面恢复并重新 health-attest 后才能续测 |

### 当前等级结论

- **机制可信度：强。** P7R123 以 same-tile、三 seed 正确性、7 轮配对、W05 前后 bracket 和一致的
  DMA/latency 方向，证明显式 barrier 权重驻留在 Y02B00 上真实有效。
- **跨几何外部有效性：未通过。** Y02 只有一个可配对 family；Y01 的已观测身份全部错误。
- **搜索算法证据：未开始。** 当前没有两个完整 FPGA-correct pool，也没有合格的 Y02/Y01 TopHub
  reference，所以不能生成确认性 regret 或 trials-to-threshold。
- **硬件 validity 层：必要但尚不充分。** local FSim 的 Y01 24/24 与板端已完成身份 0/23 形成强烈
  domain gap；这可以写成设计动机和负结果，不能写成本文已经准确预测 FPGA validity。
- **整体 CCF-C readiness：未闭合。** 若现在投稿，最强结果仍是一对微基准，难以单独支撑“分层
  AutoTune”主张；完成第二几何和完整池搜索后才达到目标证据形态。

## 3. 当前可以写进论文的主张

| 等级 | 可写主张 | 证据与严格边界 |
|---|---|---|
| 已完成的方法贡献 | 实现了 `ConfigEntity + residence mode -> 静态合法性 -> lowered-TIR 访问签名 -> FSim/命令签名 -> no-leak 派发 -> allowlist 容量证书` 的闭环接口 | 可以写“本文设计并实现”；不能写成已经在新网络上证明搜索收益 |
| 本地实证 | tile 和 residence mode 会共同改变逻辑 input/weight/output DMA bytes、calls、请求形态以及 instruction/UOP 需求 | P7R117/P7R119 的 36 点 pilot：36 gross、20 static-pass、15 FSim-pass；这些是编译/模拟签名，不是物理 AXI 计数或 latency 分解 |
| 本地实证 | VTA 合法性不能只靠 SRAM 容量判断，还受 padding、compact-buffer、DMA 形态和 UOP/运行时结构约束 | P7R117 与 P7R118；P7R118 在 Y02 硬件容量合格域中，original lowering 29/343、四模式同时合法 12/343 |
| 本地实证 | 命令 backing 必须按最终候选身份/allowlist 定容，不能把 8 KiB、36 KiB 或 SRAM 容量写成板级常数 | P7R119 的 15 个 FSim-pass 身份需要 8,192--335,872 B、median 36,864 B；旧 W05 的 8 KiB 只对其 exact allowlist 成立 |
| 负向结果 | P7R119 的旧 mode-2 `weight_stationary` 没有实现真实 weight DMA 减少，只能作为 negative control | P7R120 以后已换成显式 drain/barrier；P7R123 只证明新实现的 Y02B00，不可回填为旧 mode-2 有效 |
| 板端机制正例 | 在精确 Y02B00 same-tile pair 上，`weight_resident_barrier` 的中位算子 latency 改善 25.28%，7/7 配对轮次获胜 | 仅适用于该 workload/family/板端会话；必须同时报告 original 140.988870 ms、barrier 105.349023 ms 和正确性门，不能写成 YOLO 平均加速 |
| 访问—时间同向证据 | Y02B00 的 weight LOAD bytes 下降 92.31%、总 LOAD bytes 下降 54.48%、LOAD calls 下降 49.84%，尽管 synchronize calls 从 2 增至 34，latency 仍下降 | runtime profile 是本实现的逻辑请求/指令统计；没有 APM 时不称物理 AXI 总线 bytes 或因果时间分解 |
| 板端负向证据 | Y02 6 个身份仅 2 个正确；Y01 已完成 23/24 个身份且完成部分 0/23 正确 | 支撑 FPGA correctness gate 必须位于 timing 前；不证明已有 gate 能预测这些失败，也不把未执行的第 24 个身份算失败 |
| 既有板端先验证据 | 在旧 ResNet same-tile 数据中，56 对有 42 对更快，中位改善 3.31%；W05 冻结测量 2 点得到距 TopHub 0.31% 的候选 | 只能作为机制可行性/开发证据；不是新 YOLO 的前瞻确认，也不能替代等预算完整池实验 |
| 部署子模块 | W05 exact allowlist 的 instruction/UOP requested backing 已由 64 MiB 降到 8 KiB，并完成 6/6 seed 的真实 runtime 资格化 | 可写为“候选身份驱动的部署闭环”；数值不可外推到其他 workload、stage 或整网，也不代表释放系统固定 192 MiB 预留 |

P7R118 的规则在当前 Y02 全扫描数据上达到 100% 匹配，只能称为“对同一源码、硬件配置和枚举域的
精确描述”。它不是 held-out 泛化准确率，也不是通用 VTA 合法性定理。

P7R117 的 5 个 FSim 失败发生在 `correctness_warmup` 前后的 runtime 结构检查，日志指向重复 UOP
`dst_idx`。结果文件将其汇总为 `environment` 不够准确；论文应称“runtime/UOP 结构安全失败”，
不能称“三 seed 数值错误”，因为这些候选没有取得三 seed 结果。

## 4. 需要后续结果才能写的主张

以下句子当前必须保持“待验证”：

1. `weight_resident_barrier` 在多个 YOLO 几何、多个 tile family 上稳定改善 latency；当前只能说
   Y02B00 这一对改善 25.28%。input-prioritized 当前没有新 YOLO 板端性能正例。
2. validity 或共享内存增量特征在相同 gross dispatch/wall-clock 预算下，优于 Random、stock-knob
   XGB 或 rules-only。
3. 新池能以更少 trials 进入 sealed reference 的 2%/5% 等价带。当前 Y02 TopHub 板端错误，不能
   作为 reference；W05 health canary 是另一几何，也不能替代。
4. 新 selected/fallback allowlist 的缩容 manifest 在真实 runtime 上通过，并能 fail-closed 拒绝
   不足容量或身份不匹配。
5. C3 改善 stage latency、整网 FPS 或多帧吞吐。

即使板端结果为阴性，仍可保留“合法性门控减少无效派发”和“候选特定安全定容”两项，但不能把它们
改写为性能搜索成功。

当前尤其禁止以下表述：

- “YOLO/多个卷积平均加速 25.28%”：25.28% 只有一个 Y02B00 same-tile pair。
- “本文优于或达到 TopHub”：当前 Y02 TopHub 不正确，没有可比较 latency；wrong answer 更不能被
  当作性能劣势。
- “Y01 24/24 板端失败”：准确记录是 23/24 身份完成，完成的 23/23 均失败，1 个因 broken pipe
  未观测。
- “驻留优化导致 Y01 错误”：original、input、barrier 都错误，现有证据只定位到该编译/执行路径，
  还不能归因具体机制。
- “FSim/静态规则已经准确过滤硬件非法候选”：Y01 的 24/24 FSim pass 与 0/23 FPGA pass 直接否定
  这一强主张。
- “P7R126 最后一个候选非法”或“板卡已永久损坏”：broken pipe 只证明该 RPC 执行面在当时中断，
  必须由新 session health attestation 重新分类。

## 5. 与 C1/C2 的不可重复计数边界

| 层次 | 已有贡献对象 | C3 可引用但不得重复算作新贡献的内容 | C3 的独立对象 |
|---|---|---|---|
| C1 | CPU/VTA stage 切分、island/topology、CPU threads、边界成本下的 DP 搜索 | 4623 topology/972528 配置、历史 copy-path 边界模型和 C1 排名 | 固定 C1 选定的 topology/stage，只优化其中 VTA workload 的 tile/驻留/命令；当前不能声称已经把新算子成本反馈给 C1 DP |
| C2 | 跨 Executor 的 K2 双 slot 零拷贝、edge/slot/generation、所有权状态机、输入池复用为 slot0 | 1,806,336 B/frame 物化消除、边界 API 下降、额外 slot 分配下降 36.36%--44.44%、多帧覆盖安全 | VTA 内部 tile 导致的逻辑 DDR/u-dma-buf LOAD/STORE 重放及命令流，不处理跨 Executor handoff |
| C2 空间增强 | 四 topology 的通用队列缩容及 pool high-water 实验 | 64 MiB→16 KiB、T2656、总池 high-water 下降 82.23%--86.96% 等旧数值 | tuner 产出 candidate-specific command signature，并随 exact allowlist 生成、失效和校验 deployment manifest；不能再次把旧节省比例当 C3 的独立收益 |

因此，C3 不应使用“零拷贝”“少一个 slot”“覆盖多帧缓冲”“重新规划图内 tensor lifetime”作为核心
新意。这些属于 C2 或尚未具备生命周期 trace 的 USMP 范畴。C3 可以报告 command backing，但其
新意必须落在“由候选/allowlist 自动导出和 fail-closed 验证”，而不是再次宣称发现固定队列过大。

## 6. 主文档需要保持的公式边界

`T_shared` 和 `T=max(...)` 是待板端标定的物理代理，不是理论速度上界，也不是已测量的时间分解。
在没有 AXI performance monitor/stall counter 时，只能将 lowered-TIR bytes/calls 称为逻辑请求签名。

当前还没有统一的 tensor/slot、command、replay/FINISH 生命周期冲突图，所以不能直接写：

```math
M_{udmabuf}^{peak}=M_{tensor/slot}^{peak}+C_{insn}+C_{uop}+M_{other}^{peak}.
```

各分量的峰值可能不同时发生。若它们是独立保留的 backing，应改称：

```math
M_{reserved}=M_{tensor/slot,reserved}+C_{insn}(A)+C_{uop}(A)+M_{other,reserved}.
```

若讨论实际 peak，只能先给 naive-sum 上界；要给精确值，必须补齐 birth/death、queue ownership、
replay 和 FINISH trace 后做冲突图 packing。该限制不影响 exact command capacity 的单独证书。

## 7. 最小后续实验

### B0：先恢复并重新资格化执行面

- P7R126 的 broken pipe 之后不得继续解释任何缺失观测。先建立新 RPC session，记录 runtime、
  bitstream、hardware/u-dma-buf 与源码指纹，并重新运行独立健康 canary；不要将 broken pipe 记为
  candidate invalid。
- P7R123 可作为其原会话内的不可变结果保留；若将新会话数据与它做总体统计，至少把 session/boot
  作为 block，最好重跑该 pair 作为跨会话稳定性检查。
- 当前 Y02 TopHub 不合格。先诊断并恢复其板端正确性；若无法恢复，则将 reference 明确改为
  `long_budget_stock_xgb` 或仅在完整池测完后使用 `full_pool_oracle`。不正确的候选永远不能作为
  latency baseline、fallback 或阈值分母。

### B1：最小正确性定位，不立即扩大 timing

- Y01 的 original、input、barrier 都失败，说明应先选一个最小 original 身份与已知正确原生模板做
  TIR/参数布局/运行时逐级对照；在 original 三 seed 正确前，不再测 Y01 residence latency。
- 对未完成的第 24 个 Y01 身份只补“unknown→observed”，结果无论正负都原样保留；它不能改变
  已完成 23 个全部错误的事实。
- 修复只能基于 correctness/结构证据，并发布新版本合同；不能读取 latency 后替换候选。旧 24 点
  合同和失败结果永久保留。

### B2：补一个独立几何的 same-tile 正例或诚实反例

- 在 Y01 修复后的池或另一个预冻结 YOLO 几何中，取得至少两个 family 的 original 与一种 residence
  mode 三 seed 正确配对，并做不少于 5 轮平衡交错 timing。
- 报告访问下降但 latency 不降的反例同样有价值。最低充分证据不是再重复 Y02B00，而是证明机制
  适用条件能跨一个独立几何解释。

### B3：建立完整池后再做等预算 replay

- 至少两个 workload 各有可用的完整 FPGA-correct/invalid 标签池；冻结池中所有 FPGA-correct 候选
  都取得 latency oracle，不能只测方法推荐的 shortlist。lower/compile/FPGA-invalid 仍计入 gross
  dispatch 和阶段墙钟。
- 20 seeds；budget 4/8/12/24；比较 Random、stock-knob XGB、rules-only、validity V，以及冻结的
  `V+DeltaT(bytes_calls / +request / +command)` 三档特征。
- 报告 median/IQR 的 regret、success@2%/5%、gross trials-to-threshold、invalid 分类以及
  lower/compile/FPGA/timing wall-clock 分项。reference 必须板端正确；pool oracle 只能在完整顺序结束
  后计算。

### B4：最终部署负向测试

- 对 selected + sealed fallback 的 exact allowlist 生成容量，运行时核对 candidate/source/TIR、
  bitstream/u-dma-buf、queue instance、FINISH/replay 和容量峰值。
- 做 3-seed 正确性与一次精确容量通过；再用小于证书值一页或缺失必要身份的 manifest 验证
  fail-closed。容量必须从记录推导，不在脚本或论文中固化为 8 KiB/36 KiB。

B0--B4 足以支持 C3 的算子级 CCF-C 风格主张。stage/整网实验和 AXI APM 计数是增强项；若不做，
论文只需明确“不主张整网 FPS”和“不把逻辑 DMA 签名等同物理总线流量”。

## 8. 最小图表集合

1. **职责边界图**：C1 选择 topology/stage，C2 完成跨 Executor K2 handoff，C3 在固定 VTA workload
   内选择 tile/驻留并输出访问签名与部署证书。该图直接消除“三项创新重复”的质疑。
2. **same-tile 机制图**：P7R123 已可画第一组：7 个 paired latency 点及 original/barrier 的 weight、
   total bytes、calls、instruction 和 sync 对照。第二几何取得前，图注明 `Y02B00 case study`，不用
   “平均提升 25.28%”。把其他 lowering/FSim/FPGA-invalid 作为旁侧堆叠条。
3. **等 gross-budget 搜索图**：横轴 dispatch budget 或累计 wall-clock，纵轴 regret/success@2%/5%，
   展示 20 seeds median/IQR，并附各阶段失败数。它是“调优有效”的主图。
4. **资格漏斗图**：local static/FSim、cross、FPGA correctness、timing 各阶段计数；明确 Y01
   `24/24 FSim -> 23 observed -> 0 correct`，另 1 个 unknown，并把 RPC broken pipe 单列为
   infrastructure failure。这张图比当前没有数据的搜索曲线更应先进入论文。
5. **部署容量表/瀑布图**：列 selected/fallback 的 instruction、UOP peak、对齐后 certificate、实际
   runtime peak 与不足容量拒绝；将 C2 tensor-slot high-water 作为既有背景单列，明确 192 MiB 固定
   reserve 未释放，不把互不同时发生的分量强行相加为实际 peak。

若篇幅紧，当前先保留图 1、P7R123 case-study 图、资格漏斗和一张部署表。完整池形成前不要绘制确认性
搜索曲线；否则空缺 reference 和大量 correctness failure 会使曲线本身失真。

## 9. 审计结论

C3 当前已经具备明确问题、独立作用层次、可运行方法、可复核的负向证据，以及一个效果量很强的
真实 FPGA same-tile 正例；与双 slot 的
重复可以通过“跨 Executor tensor ownership”与“VTA 内部 shared-memory access/command”二分彻底
切开。P7R123 的 25.28% 足以证明“这项机制值得研究”，但不足以证明“本文的分层 AutoTune 已经
有效”。当前不可替代的缺口是：恢复执行面、解释 local/FSim 与 FPGA correctness 的巨大断层、取得
第二几何的配对证据、建立带正确 reference 的完整池，再做等预算搜索和最终 allowlist 部署。
在此之前，C3 可作为硕士论文中已实现且有单点硬件证明的创新方向，但尚未达到 CCF-C 风格完整
evaluation 的等级。

## 10. 2026-09-13 状态校正（P7R313）

上面的第 7--9 节是 P7R126 时点的历史审计，不能继续当作当前待办。其关键缺口已经按更严格合同
关闭：

- clean start 已成为每个目标池的强制执行合同，旧 Y01 `0/23` 也已被明确归因为状态污染而非几何
  全错；
- 已形成 Y00/Y03/Y04/Y05/Y06/Y07/R50A/R50B/R50C/R50D/R50E 十一个标签隔离完整池，其中
  Y06/Y07/R50A/R50B/R50C 是规则冻结后的五个严格生存率多保真 holdout，R50D/R50E 是后续
  final-fused/cheap-proxy 前沿的两个独立前瞻 holdout；
- 五池 120 个预注册身份的配对汇总中，自适应方法均到达 pool oracle，中位 gross `34→120`
  （相对 exhaustive -71.67%）、可比墙钟 -41.39%，并相对 fixed 策略墙钟 -17.57%；
- same-tile 机制正证据已跨 input/weight、多种 YOLO/ResNet50 几何及 Y02 两个启动；Y04、Y02 和
  R50C 第三 seed late failure 保留为合法性/性能反例；
- exact allowlist 的命令容量已经完成运行时 attestation、完整 YOLO graph 正向运行以及少一页
  fail-closed，四 Executor OOM 也已由 u-dma-buf graph/queue backing 精确解释并通过定容消除；
- R50C 首次把同一独立 workload 的严格算子搜索、五个 same-tile 机制对、最终 fused TIR 五调用点
  DMA 预测和预训练 ResNet50 整图结果贯通：LOAD `-3,010,560 B/-270 calls` 逐项命中，整图
  `346.687→330.277 ms`、7/7。
- P7R314--P7R315 又把 final fused TIR 聚合变成上板前的高保真重排动作；P7R166 的旧请求系数在
  不重拟合时将 F00/F01 的两项 bytes-only 错选修复为 2/2，但四程序仍有 1/6 错序。它是 post-hoc
  方法消融，不能计作新的严格 holdout，也不能删除 FPGA latency gate。
- P7R316--P7R327 对该固定系数进行了真正的前瞻冲突对检查。F01/F02 的最终融合程序形成
  `38.10 MB/980 calls` 对 `6.52 MB/2900 calls`；旧系数选择 F01，但 FPGA 中位 latency 为
  `375.140/350.550 ms`，regret 7.015%、0/7。6 次 correctness、14 次 timing 和十字段 DMA
  对账全部通过，因此结论是标量校准失败，而不是程序或静态画像错误。P7R325 的 timed profiler
  两倍计数失败已由 P7R327 原样哈希并归因为执行器漏除二，P7R326 只修正归一化、未重选候选。
- 由此，论文可保留的强结论不是“calls 系数解决 bytes 错排”，而是“final fused TIR 提供精确
  bytes/calls 多目标证据；联合支配可静态淘汰，目标冲突必须升级到真实 FPGA”。P7R315 和
  P7R327 分别证明 bytes-only 与固定 scalar 都存在边界。
- P7R328 已将上述 Pareto 规则实现为 post-hoc 审计：四个重叠比较组全部保留 oracle；R50C 三 tile
  可由 3 静态缩到 1，另三个冲突组各保留 2 点。因此它提供的是“安全筛选与升级成本”的设计证据，
  不是四个独立成功 trial，也不允许宣称冲突升级仍然减少测量。
- P7R329--P7R335 随后在未见 R50D 上完成真正的 prospective final-fused Pareto 检查。18 个
  lowering/FSim-pass 程序在板端标签前按 bytes/calls/extra-submissions 缩为 4 点；补测全部被支配
  点后，4 点前沿保留 15 点 FPGA-correct pool oracle，candidate dispatch 与实际逻辑 DMA bytes
  分别少 77.78% 和 87.99%。因此“只有事后 Pareto”的缺口已关闭。
- 该结果没有关闭端到端成本缺口：42.924 s static、3.509 s FSim 和 656.866 s 全候选最终融合构建
  都已在选择前支付；bytes+fail-fast 与 Pareto 到 oracle+2% 均为 2 次派发。三个 FSim-pass weight
  barrier 还在整图 FPGA 上错误。故 H88 只支持最后一级硬件候选/流量缩减，不支持全面优于
  bytes-only、Pareto 是 latency 定理、端到端搜索已经更快或超过 TopHub。
- P7R336 在暴露标签后以廉价算子 TIR bytes/calls/barrier indicator 重放 R50D，4 点代理前沿仍
  保留 oracle，并把历史 stock+候选构建耗时回放为 169.592 s 而非 656.866 s。它只负责冻结下一
  未见 workload 的规则；在新 workload 真正先筛后构建之前，不能写成已取得构建或端到端节省。
- P7R337--P7R348 已在新 R50E 上完成这项严格验证。12 个真实 lowering/FSim 合法候选在任何目标
  完整图、FPGA 或 latency 标签前被廉价算子代理缩成 2 点；实际只构建 stock+2 点 100.120 s，标签
  后补齐全池才得到 stock+12 点 442.808 s 的同平台反事实。2 点前沿保留 12/12 FPGA-correct 池
  oracle，实际构建墙钟 -77.39%；加回公共 static/FSim 后选前本地墙钟仍 -73.60%；candidate
  dispatch -83.33%、逻辑 DMA bytes -90.62%。因此 H90
  可以支持“先筛后构建”的端到端成本主张；但只覆盖一个 ResNet50 1x1 workload/boot，全部候选
  都比 stock 慢，logical DMA 也不是物理 AXI，不能写成整网加速或通用不漏点保证。

因此第 9 节“尚未达到 CCF-C 风格完整 evaluation”的旧结论已失效。当前权威结论以
`C3_CCF_C_SHARED_MEMORY_INNOVATION.md` 的 P7R348 版本和 `CLAIM_LEDGER.md` H81--H90 为准：已经
形成较强硕士论文独立创新及 CCF-C 风格 system/case-study 核心骨架，但仍不能保证投稿录用。未关闭
的外部有效性问题是 cheap proxy 的跨模型/跨 boot 重复、跨 boot 整图复验、自然 same-mode bundle 失败的前瞻递归、原生编译期 call-site
身份、论文式 code-exact weight reuse、正式 ImageNet/COCO accuracy 和物理 AXI 计数。
