# P7R297--P7R328：R50C 独立 workload、多保真、整图留出与 fused 冲突升级

日期：2026-09-13  
开发板 boot：`aa7a3e5c-021d-4d6b-88ae-7f696faa567c`

## 目标与隔离边界

R50C 是此前未进入 C3 性能标签的 ResNet50-v2 stage3 bottleneck `1x1` 卷积：
`CI=1024, CO=256, H=W=14`。本轮不再增加 R50A 的 tile，而是在新 workload 上同时检查：

1. P7R185 已冻结的生存率自适应多保真规则能否以较低代价找到完整池 oracle；
2. input/weight 驻留在相同 tile 下怎样改变逻辑 DMA、正确性和 latency；
3. 选中 tile 放回预训练 ResNet50 的全部真实调用点后，融合 TIR 预测与整图收益是否成立。

候选、策略和首轮上板顺序在 R50C latency 前冻结。整图部分在任何 R50C full-graph/partial-mask
latency 前另行冻结；它使用已经存在的算子级 correctness，但不使用算子 latency 来选 pair。因此
算子搜索是独立 workload 的严格留出，整图是同一 workload 的后续调用点/整图 latency 留出。

## P7R297--P7R305：独立 workload 在线多保真搜索

P7R297 枚举完整 2240 个 ConfigEntity，纯硬件谓词保留 324 个，并按无标签哈希冻结 8 个 family。
P7R298 将 original/input-stationary/weight-resident-barrier 展开为 24 个公开候选，沿用既有服务代价：

```text
B_dma + 65536 * N_dma + 131072 * max(submissions - 1, 0)
```

没有用 R50C 标签重拟合系数。P7R299 的第一个 family 为 0/3 lowering-pass，因此冻结策略按原规则
进入 `sparse_family_wave`。前瞻路径只考虑 6/24 个候选，执行 6 次 lowering、1 次 FSim、1 次交叉
编译、1 次 FPGA correctness 和 1 次性能测量；首个测量候选为 R50CF00 input-stationary：

- candidate：`cbeec5d3c92de320929553a96df2989dd6e2d577ec27d14a8e72d9580b72ae61`
- knobs：`tile_h=14,tile_w=7,tile_ci=16,tile_co=4`
- 前瞻中位：4.445534 ms；独立完整池复测：4.442324 ms。

P7R301/P7R302 随后为审计而测完整池：24 个候选中 11 个 lowering/FSim-pass，10 个 FPGA-correct，
1 个 FPGA-invalid；70 次性能调用全部正确。首个前瞻测量候选就是 10 点正确池 oracle。P7R305 的
可比成本回放如下：

| 策略 | gross | replay wall | 相对 exhaustive |
|---|---:|---:|---:|
| 生存率自适应 | 6 | 822.451 ms | gross -75.00%，wall -90.39% |
| 固定 hardware-diverse | 12 | 1,487.061 ms | wall -82.63% |
| exhaustive qualification | 24 | 8,559.153 ms | 基线 |

前瞻执行器实际记录的 canonical phase wall 为 980.704 ms；另有 1,713.486 ms 本地编排开销，不能
藏进算法节省。Random 的 20-seed 中位为 gross 15、wall 6,215 ms；结果只对该冻结池成立。

## P7R302/P7R307：机制正证据与正确性反例

五个 FPGA-correct 的 original/input same-tile 对全部由 input-stationary 加速，范围
24.336%--79.853%，中位 44.694%；input LOAD bytes 中位下降 75%，总 DMA bytes 中位下降
26.380%，DMA calls 中位下降 66.667%。五对共享同一几何和 boot，精确双侧符号检验
`p=0.0625`，只能作为该几何的受控机制描述，不能当五个独立 workload。

唯一 lower/FSim-pass 的 weight barrier 是一个重要负例。R50CF05 barrier 在 seed 0 和 20250901
正确，却在 seed 20260910 出现 17,920 个错误元素；三 seed gate 因此拒绝它，不生成 latency。
它证明：

- lowering/FSim 不能替代真实 FPGA 数值认证；
- 单 seed canary 会错误接纳该候选；
- 顺序 fail-fast 对“晚失败”没有节省，P7R129 的平均节省不能外推为每个失败点都省调用。

## P7R306：五个严格留出汇总

将 Y06、Y07、R50A、R50B、R50C 五个规则冻结后的严格 workload 合并，共 120 个预注册身份。
20 个 replay seed 下自适应策略全部到达五个 pool oracle：

- adaptive 中位 gross 34，fixed 94，exhaustive 120；
- adaptive 中位 wall 21,910.022 ms，fixed 26,580.804 ms，exhaustive 37,380.841 ms；
- 相对 exhaustive：gross -71.667%，wall -41.387%；相对 fixed：wall -17.572%；
- 自适应阶段动作中位为 lower 34、FSim 6、compile 6、FPGA 5、measure 5。

这是 workload 配对汇总，不把 20 个确定性 replay seed 写成 100 个独立硬件实验，也不主张任意新
网络必然首测命中 oracle。

## P7R308--P7R313：预训练 ResNet50 整图留出

P7R308 在没有 R50C 整图或 partial-mask latency 时，按冻结注册表选定 R50CF00 original/input pair。
P7R309 构建两份预训练 ResNet50，Graph JSON 相同且目标 route 均精确命中。P7R310 捕获两侧各
68 个 `CPUAccessRewrite` 后的最终融合 TIR；P7R311 在上板前确认 R50C 实际出现于图节点
118/131/144/157/170，并冻结十字段预测：

| 范围 | LOAD bytes | LOAD calls | input calls | weight calls | ACC calls |
|---|---:|---:|---:|---:|---:|
| 每调用点 | -602,112 | -54 | -24 | -24 | -6 |
| 五调用点 | -3,010,560 | -270 | -120 | -120 | -30 |

weight/ACC LOAD bytes 与 STORE 均不变。P7R312 clean start 后完成 3 seed、6 次完整图 correctness 和
7 轮、14 次交错 timing，所有 20 次输出均相同且非零；correctness/timed profiler 的十个字段都与
P7R311 逐项相等。完整图中位：

```text
original 346.686947 ms -> input-stationary 330.277203 ms
吞吐提升 4.968%，7/7 配对轮获胜
```

P7R313 重新验证冻结链的 302 个上游/板端文件、candidate/mode/graph/hash 链、seed/round 覆盖、
DMA 差和延迟统计。P7R312 contract 中有一段从通用化前继承的 P7R273/R50A 旧说明；原始文件不改，
P7R313 明确校正为 R50CF00、五调用点的新 workload 整图留出。代码中的后续文案已修复。

这是第二个 ResNet50 workload 的整图正结果，也是第一个在算子级严格多保真留出之后继续闭合整图
证据的 workload。仍只覆盖一个模型、一个 boot、固定 tile；逻辑 DMA 不是物理 AXI，未做 ImageNet
accuracy，不能写成通用 ResNet50 加速率。

## 对第三创新点的增量

R50C 把原来分散的证据第一次串成同一个新 workload 的完整闭环：

```text
完整 ConfigEntity 域
 -> 生存率自适应地选择 lowering/FSim/FPGA/measure 付费动作
 -> 第一次测量命中完整正确池 oracle
 -> same-tile 驻留解释 DMA 与 latency
 -> 三 seed 拒绝 late-failure weight 候选
 -> 最终 fused TIR 预测五个真实调用点
 -> 预训练 ResNet50 整图 7/7 加速
```

它显著缓解“只有四个严格 holdout”和“R50A 两个 tile 不是独立 workload”的限制。尚未解决的是自然
same-mode bundle 失败后的前瞻递归、跨 boot 整图稳定性、原生 Relay/AutoTVM call-site key，以及
物理 AXI/正式精度评价。

## 完整性检查

- P7R297--P7R315 共 19 个目录、370 个 manifest 条目全部重新计算哈希通过；
- adaptive/multifidelity、same-tile、holdout、fused-TIR、服务重排与 ResNet50 route 的 23 项相关
  测试通过；
- 实验后板端 boot 未变化，FPGA=`operating`，u-dma-buf=`201326592`，默认 runtime RPC 存活，
  本次启动的 `dmesg` 未出现 EXT4/I/O/mmc error；
- 一次扩大测试命令因误覆盖 `PYTHONPATH` 在 pytest 收集阶段失败，未执行测试或接触开发板；恢复
  原环境后的测试才是有效结果。

## 关键产物

- `20260912_p7r297_r50c_bottleneck_adaptive_holdout_contract_run01`
- `20260912_p7r298_r50c_adaptive_multifidelity_registry_run01`
- `20260912_p7r299_r50c_adaptive_live_local_prefix_run01`
- `20260912_p7r300_r50c_adaptive_board_wave01_run01`
- `20260912_p7r301_r50c_full_local_oracle_completion_run01`
- `20260912_p7r302_r50c_full_board_oracle_completion_run01`
- `20260913_p7r303_r50c_canonical_completed_pool_run01`
- `20260913_p7r304_r50c_multifidelity_completed_pool_run01`
- `20260913_p7r305_r50c_adaptive_multifidelity_holdout_analysis_run01`
- `20260913_p7r306_five_strict_adaptive_holdout_aggregate_run01`
- `20260913_p7r307_r50c_same_tile_memory_effect_run01`
- `20260913_p7r308_r50c_callsite_latency_holdout_contract_run01`
- `20260913_p7r309_r50cf00_resnet50_generic_pair_build_run01`
- `20260913_p7r310_r50cf00_resnet50_fused_tir_audit_run01`
- `20260913_p7r311_r50cf00_fused_tir_occurrence_contract_run01`
- `20260913_p7r312_r50cf00_fullgraph_fused_tir_pair_board_run01`
- `20260913_p7r313_r50cf00_fullgraph_latency_holdout_audit_run01`

## P7R314--P7R315 后续：最终 fused program 服务代价不重拟合审计

R50C 完成后，按既定下一步把 P7R296 的“calls 应进入最终程序排序”从观察落实为可执行重排。
P7R314 先冻结测试规范，明确目标 latency 已暴露，不能冒充新 holdout；唯一允许使用的系数是
P7R166 在更早开发池冻结的 `65536 B/request` 与 `131072 B/extra submission`，禁止用 F00/F01
重新拟合。F00/F01 都是 input-stationary、没有 barrier 额外 submission，因此本消融只检查：

```math
S_{fused}=B_{LOAD+STORE}+65536N_{LOAD+STORE}.
```

P7R315 直接读取最终 fused-TIR 的目标图出现量和已经审计的整图 latency：

| 模式 | bytes-only 选择/相对 oracle regret | fused service 选择/相对 oracle regret |
|---|---|---|
| original | F01 / 112.844% | F00 / 0% |
| input-stationary | F01 / 16.308% | F00 / 0% |

两种 mode 的选择准确率从 `0/2` 变为 `2/2`。四个 exact program 的所有六个两两次序中，bytes-only
只排对 `2/6`，fused service 排对 `5/6`；剩余反例是 F00 original 实测仍快于 F01 residency，服务
代价却相反。因此这一层的准确定位是：

```text
算子级便宜特征/按需 lowering
 -> 少数晋级 tile 做完整模型 final fused-TIR 重排
 -> FPGA correctness + latency 最终关门
```

它把融合和图出现次数变成一种更高保真但更昂贵的搜索动作，修复已知 bytes-only 错排，却不取代
FPGA 标签。由于测试规范在目标标签暴露后才冻结，证据只能称“系数早于目标冻结、无重拟合的
post-hoc 外部检查”；下一新 workload 才能承担 prospective fused-reranker 确认。

新增产物：

- `20260913_p7r314_fused_program_service_proxy_norefit_contract_run01`
- `20260913_p7r315_fused_program_service_proxy_norefit_analysis_run01`

## P7R316--P7R327：固定服务系数的前瞻反例

为避免 P7R315 停留在标签暴露后的正向解释，P7R316/P7R317 在不读取 operator latency 或新的
full-graph latency 时，从 R50C 已通过三 seed correctness 的候选中冻结两个不同 tile：
F01=`h1,w7,ci64,co4`，F02=`h7,w7,ci1,co1`。P7R318--P7R323 分别完成 exact Relay build、两侧
各 68 个 fused TIR snapshot 和五个真实图节点的 occurrence 合同。两个 input-stationary 最终程序为：

| program | 最终 LOAD+STORE bytes | 最终 LOAD+STORE calls | P7R166 固定服务分数 |
|---|---:|---:|---:|
| F01 | 38,097,920 | 980 | 102,323,200 |
| F02 | 6,517,760 | 2,900 | 196,572,160 |

P7R324 因而在接触板端前锁定选择 F01；这与 bytes-only 选择 F02 相反，构成一次真正的冲突对测试。
P7R325 完成 3-seed 与 7 轮计时后，因为新执行器漏掉 time-evaluator 的双执行 profiler `/2`
归一化而拒绝发布。P7R327 对原始文件逐项哈希并证明 correctness delta 等于单次预测、timed raw
delta 恰为两倍；P7R326 只修正该归一化，没有改变候选、系数或顺序。

P7R326 的有效上板结果为：6 次 correctness 与 14 次 timing 全部 logits 相等且非零，十项
LOAD/STORE 差在正确性和计时两条路径上都精确命中；但 F01/F02 中位 latency 为
`375.139952/350.550236 ms`。因此冻结代理选错，regret 7.0146%，0/7 轮获胜；bytes-only 在本对
反而命中 oracle。该负结果排除了“compiler DMA 画像错误”解释，说明失败来自把 bytes/calls 权衡
压成一个跨 tile 固定常数。

正式方法由此收敛为：final fused TIR 是精确多目标高保真动作；候选在 bytes/calls/submissions 上
联合支配时可静态淘汰，被支配关系不成立时保留到更高保真度，而不是继续重拟合一个万能请求系数。
P7R315 说明 calls 不能删除，P7R327 说明固定 scalar 也不能代替 FPGA latency。

新增产物：

- `20260913_p7r316_r50c_fused_reranker_tile_a_holdout_contract_run01`
- `20260913_p7r317_r50c_fused_reranker_tile_b_holdout_contract_run01`
- `20260913_p7r318_r50cf01_resnet50_generic_pair_build_run01`
- `20260913_p7r319_r50cf02_resnet50_generic_pair_build_run01`
- `20260913_p7r320_r50cf01_resnet50_fused_tir_audit_run01`
- `20260913_p7r321_r50cf02_resnet50_fused_tir_audit_run01`
- `20260913_p7r322_r50cf01_fused_tir_occurrence_contract_run01`
- `20260913_p7r323_r50cf02_fused_tir_occurrence_contract_run01`
- `20260913_p7r324_r50c_fused_program_reranker_holdout_run01`
- `20260913_p7r325_r50c_fused_program_reranker_board_run01`（fail-closed，无成功 manifest）
- `20260913_p7r326_r50c_fused_program_reranker_board_run01`
- `20260913_p7r327_r50c_fused_program_reranker_audit_run01`

本轮最终完整性复核覆盖 P7R297--P7R327 的 30 个成功 manifest 目录、973 个条目，全部重算哈希
通过；P7R325 的四个原始文件由 P7R327 单独绑定。纵向链相关测试 33/33 通过。实验后板端仍为
boot `aa7a3e5c-021d-4d6b-88ae-7f696faa567c`、FPGA=`operating`、u-dma-buf=`201326592`，默认
runtime RPC 存活且无 EXT4/I/O/mmc error。

## P7R328：Pareto-safe 规则实现

P7R328 不再尝试拟合第二个标量，而是实现二维联合支配：只有当另一程序在最终融合
LOAD+STORE bytes 与 calls 上均不差、且至少一项严格更好时才静态淘汰；否则把所有非支配点升级。
在标签已暴露且相互重叠的四组上，oracle 4/4 留在前沿：R50A original/residency 与 R50C F01/F02
三个冲突组的前沿均为 2；R50C F00/F01/F02 三 tile 中 F00 同时支配另外两点，前沿由 3 缩为 1，
且 F00 是 oracle。bytes-only 与旧 service scalar 分别命中 2/4、3/4。

这一分析的价值是把负结果转成一个 fail-safe 搜索动作，并明确承认冲突升级需要测量两个点；它不是
新的 holdout，也不能用 4/4 宣称泛化。产物为
`20260913_p7r328_fused_program_pareto_escalation_posthoc_run01`。最终完整性检查覆盖 31 个成功
manifest 目录、975 个条目并全部重算通过；纵向链相关测试 34/34 通过。
