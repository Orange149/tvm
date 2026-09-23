# 输入—权重复用补充实验：双启动机制与时间—空间权衡定稿

日期：2026-09-15。状态：`A_B_C1_C2_D_COMPLETE`。本轮停止，不追加上板、整图或在线调优。

## 1. 最终解决了什么问题

此前主要有输入单项配对、一个权重驻留family，以及包含回退/退化的四方案记录，不能证明
“输入复用与权重复用在同一程序中同时实际生效”。现在已补齐三个固定机制几何、11个same-tile
四元组：每组四种不同TIR/ARM二进制，IW实际同时减少input与weight搬运，两个启动均通过正确性，
每启动八轮平衡计时全部保留。另有4个独立reload控制，区分驻留收益与直接改循环/同步的关联。

新结论是**固定容量下加载作用域、驻留跨度和同步结构的具体权衡**，不是新的共享内存驱动、
USMP算法或首次提出Cheng的复用机制；也不把本轮机制测量冒充新tuner的搜索效率实验。

## 2. 稳定收益与组合边界

按预先写入计划的门槛，稳定加速要求：两个启动分别至少改善2%，且各至少6/8同轮配对获胜。
这只是工程复测门槛，不是统计显著性或零错误概率保证。

| 模式 | 稳定快于original的family | 输入/权重加载解释 |
|---|---:|---|
| I：输入优先 | 11/11 | 移除跨output-channel重复输入加载；权重请求可能合并但字节不必减少 |
| W：权重驻留barrier | 10/11 | 保留对应output-channel tile的完整reduction权重，减少跨空间重复LOAD；增加drain/submission |
| IW：组合 | 11/11 | 同时移除两类重复LOAD，但需要更大weight/ACC作用域且保留完成边界 |

IW相对O改善范围：C1为20.99%--85.99%，C2为19.26%--86.00%。这些范围来自机制筛选后的11组，
不是自然ConfigSpace平均收益。组合在两个启动均10/11组最快；剩余M1D0034是I略快，差距只有
0.82%/1.17%，不声称组合显著更差。

加法交互 `T(IW)-T(I)-T(W)+T(O)` 在5组中两个启动均为负，其余6组均为正。
负交互不意味着IW必然最快：M1D0034交互为负但I最快。正交互也不意味着组合无收益：例如
M2D0140组合仍大幅快于O/I/W。不能用一个交互符号替代四方案实际对比。

## 3. 空间到底换了什么：M1D0139

同一ConfigEntity：`tile_h=2,tile_w=28,tile_ci=1,tile_co=2`，virtual-thread knobs固定为1。
单完整算子的静态/FSim数据与两个启动实际runtime LOAD/STORE核对一致。

| 模式 | input B/calls | weight B/calls | input arena B | weight arena B | ACC arena B | submission/drain |
|---|---:|---:|---:|---:|---:|---:|
| O | 193536/112 | 516096/112 | 1920 | 4608 | 7168 | 1/0 |
| I | 96768/56 | 516096/56 | 1920 | 9216 | 14336 | 1/0 |
| W | 193536/112 | 36864/2 | 1920 | 18432 | 7168 | 3/2 |
| IW | 96768/56 | 36864/1 | 1920 | 36864 | 14336 | 2/1 |

输出字节始终50176 B；input SRAM并没有因输入复用而变大。I扩大ACC中共同消费同一input tile的
output-channel范围，同时扩大临时权重范围；W扩大权重的reduction覆盖，IW再同时覆盖全部CO/CI。
这解释了为什么“驻留模式”不是一个DMA数字：它改变哪些消费者可以在覆盖前共同使用同一份数据，
并改变下一次加载发生的位置和同步边界。加载循环、条件、源矩形、SRAM目标和UOP发射上下文
均保留；还没有形成逐个GEMM消费者的统一地址—异步覆盖证明，不能升级声称完整冲突图已实现。

USMP对**既定程序**的buffer大小、生命周期冲突与偏移进行规划，不会自动把O/I/W/IW互相转换；
因此这里的驻留作用域生成与USMP是互补的。USMP不必然“牺牲执行时间换空间”，Cheng也不必然
“始终增加空间且始终加速”。本轮用分域资源约束和实际时间判断可行权衡，不声称理论全局均衡最优。

当前W/IW的full-layer gate还会保守拒绝整层294912 B、超过256 KiB的边界；不代表分tile权重复用
原则上不可能。两个original合法边界family及全部失败/not_applicable记录在独立表中保留，不上板。

## 4. 独立reload控制：驻留收益不是只靠改提交次数

对照恢复reduction-scope权重LOAD，匹配compute上下文、输入/输出DMA与同步次数。
resident与reload在**同一个独立八轮Latin块**配对，不把主块时间混入。

| family/mode | resident/reload weight B | C1改善 | C2改善 | 每启动胜轮 |
|---|---:|---:|---:|---:|
| M1D0139 W | 36864/516096 | 24.63% | 24.83% | 8/8 |
| M1D0139 IW | 36864/516096 | 26.46% | 26.35% | 8/8 |
| M2D0140 W | 147456/2064384 | 49.47% | 49.63% | 8/8 |
| M2D0140 IW | 147456/2064384 | 54.92% | 55.03% | 8/8 |

四对均达到稳定门槛，支持权重驻留的时间收益。但SRAM索引/allocation仍可能不同，不能称完全
隔离物理地址和所有硬件因素的因果实验。

## 5. 必须保留的负结果：bytes不是完整性能模型

### 同tile反例

M1D0034的W把weight从73728降至36864 B，总数据DMA从338944降至302080 B，但submission
从1增至5，两个启动仍慢7.34%/3.23%，只赢0/8与1/8轮。同步、计算/UOP和地址变化未完全隔离，
只能说访存减少未抵消综合执行代价，不能只归因于某一个因素。

IW的数据DMA最少140800 B，却略慢于I的177664 B。这直接反证“同tile最小访问方案一定最快”。

### 跨tile反例：M3

| IW配置 | 总数据DMA B | input LOAD calls | submission | C1 ms | C2 ms |
|---|---:|---:|---:|---:|---:|
| M3D0056，空间tile 1×1 | 51584 | 98 | 2 | 1.737 | 1.741 |
| M3D0046，空间tile 1×7 | 56960 | 7 | 2 | 0.544 | 0.549 |

在已暴露机制池回顾性按bytes/calls排序，M1/M2首点进入pool oracle+2%，M3首点却慢219.40%/
217.25%（约3.19/3.17倍）。M3的较少源搬运来自不同请求覆盖；更小tile带来更碎的请求，且
`tile_ci/tile_co`与计算命令粒度同时变化。这里不能用唯一张量大小或source union直接预测时间。

本轮因此**不修改正式tuner、也不发布新的搜索效率正贡献**。已有Y00/Y03的前瞻DMA先验正结果
保持原范围，不能再泛化为固定FPGA上所有卷积都适用。calls、请求形态、计算/UOP粒度和同步是
值得独立确认的候选特征，但本池标签已暴露，不能拟合后重新包装为前瞻；将来若改策略须另冻新池。

## 6. 原始与扩展空间：局部最优结果

| 几何 | C1扩展最优相对original最优改善 | C2改善 |
|---|---:|---:|
| M1 | 21.63% | 20.20% |
| M2 | 28.32% | 28.25% |
| M3 | 49.71% | 47.79% |

均在同一已实测tile集合中比较；O属于扩展空间，所以扩展最小值不会更差，不意味着搜索自动更容易。
本地缓存TopHub逐workload核查为M1/M2/M3均0个精确workload命中，未新增板端reference测试或
TopHub参考线；不声称超过官方全空间最优。逐点latency/data/command Pareto保留非支配解，
不是仅展示最快IW点；SRAM按分域容量约束检查，不混入一个跨物理域的虚构总峰值。

## 7. 正确性、计时与空间账本

- 两启动分别659次完整算子调用、416次正确计时；共1318/832，全部dispatch、seed、显式warmup、
  三seed正确性、主块/控制块Latin顺序与逻辑DMA/submission逐项核对。
- C1连续W0/T0为61.635/36.001 s，C2为51.180/35.623 s；分别报告，不把两个boot拼成一条时间线。
- B实际468次compiler、480条lower提案、58个FSim worker；失败、200次完成FSim与8次未完成尝试、
  48次SDK setup失败和48次独立交叉编译恢复都保留。只读审计没有重做大空间扫描。
- 各阶段及旧失败/诊断/恢复的已记录非派生墙钟合计下界550.763 s；缺失overlay/早preflight完整
  时间为unknown，全部含失败总投入仍unknown。不是连续time-to-quality，也不把未知时间填零。
- instruction/UOP分别按4 KiB对齐其FSim观测峰值需求；本轮不追加不足容量部署认证。
  张量SRAM跨度、命令backing和整个192 MiB u-dma-buf预留分账，不能混称释放192 MiB。
- 完成握手新driver使本次结果可消费；0x1c valid位置仍缺生成映射认证。C1/C2共同使用同一实验库，
  不混比旧无效会话的绝对latency。全部历史失败只计成本/边界，不获性能标签。

## 8. 图表、代码与验收定位

主要分析目录：`../07_grouped_holdout/20260915_c3_reuse_node_d_dual_boot_run01/`。
补充核验目录：`../07_grouped_holdout/20260915_c3_reuse_node_d_retrospective_audit_run01/`。

| 交付 | 文件 |
|---|---|
| 图1：准确LOAD作用域与UOP消费者上下文 | 主目录 `01_load_consumer_scopes.svg`、`scope_evidence.json` |
| 图2：逐张量bytes/calls与全部八轮双启动延迟 | 主目录 `02_four_mode_dma_calls_latency.svg` |
| 图3：SRAM占用、同步、命令与反例 | 主目录 `03_space_sync_latency_tradeoff.svg`、`boundary_rejections.json` |
| 图4：同tile池best-original/best-extended | 主目录 `04_best_original_extended.svg` |
| 旧证据/新证据覆盖与回退表 | 补充目录 `legacy_and_new_coverage.csv` |
| 四元组资格与实际复用表 | 补充目录 `four_mode_correctness_reuse.csv` |
| 两启动成本表及全部失败/恢复分账 | 补充目录 `two_boot_benefit_and_cost.csv`；主目录 `boot_phase_costs.csv`、`all_attempt_cost_ledger.csv` |
| 稳定配对、reload和加法交互 | 主目录 `stable_same_tile_pairs.csv`、`reload_control_pairs.json`、`family_classification.json` |
| 暴露池排序负结果与Pareto | 补充目录 `retrospective_bytes_rule.csv`、`measured_pool_pareto.csv`、`M3_bytes_counterexample.png` |

新增 `analyze_vta_c3_reuse_dual_boot.py` 和 `audit_vta_c3_reuse_node_d.py`；只读核验旧文件，不修改
冻结源码或覆盖旧artifact目录。新测试覆盖显式round对齐、seed/缺失/NaN/错误拒绝、双启动稳定门槛、
比值中位数区别、分队列对齐、交互与最快分离、完整真实池顺序及DMA、Pareto和M3负结果不隐藏。
合并已有资格/身份/回退/区域/padding/条件LOAD/布局/中断/只读测试共88项通过，四模式真实FSim
测试全部执行，没有跳过；只验证本计划范围，不等于ImageNet准确率或通用硬件安全证明。

最终逐项完成审计及测试输出另存 `20260915_c3_reuse_supplement_completion_audit_run01/`。
本补充计划完成并停止；完整生成寄存器认证、统一覆盖证明及新在线搜索属于明确保留的非本轮工作。
