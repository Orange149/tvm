# 第三创新点：论文方法—实现—实验缺口审计

> 更新（P7R194）：正文是 P7R114 时点的历史缺口审计。后续已完成五个标签隔离完整池、
> exact-allowlist 容量正向/不足一页 fail-closed，以及 candidate×phase 在线多保真调度。Y04 否定
> DMA-bytes-only 泛化；Y05/Y06 验证冻结的共享内存服务代价与存活率自适应，其中 Y06 以 6/24
> gross 候选命中完整池 oracle。最新结论和剩余缺口以
> `C3_CCF_C_SHARED_MEMORY_INNOVATION.md` 为准。

日期：2026-09-11  
结论：`RESOURCE_SUBSYSTEM_VALID; SEARCH_METHOD_NOT_YET_CLOSED; G7_NO_GO`

## 1. 纠偏结论

此前工作的真实完成边界是：驻留 schedule 机制、静态容量/DMA/TIR 过滤、FSim 与真实 FPGA
correctness canary、硬件证书、强 incumbent 回退，以及 exact allowlist 的 instruction/UOP backing
证书和 runtime manifest attestation。它不是 Rieber、ML²Tuner、Timeloop、USMP 或 Cheng 方法的
完整复现，也没有证明未见 workload 上的 fixed-budget 搜索收益。

因此命令内存 8 KiB 闭环保留为“空间与部署安全子模块”，但撤回“第三创新点整体已经 B 级闭环”
的表述。总 Gate G7 仍为 NO-GO。

## 2. 把论文方法组织成一条故事，而不是零散堆点

```text
固定 FPGA 硬件合同
  -> 生成满足张量化、SRAM、依赖约束的数据驻留候选
     （Banerjee / Cheng / ROLLER / Timeloop 思想）
  -> 独立预测编译合法性，昂贵层级逐级晋升
     （Rieber / ML²Tuner / MetaSchedule 思想）
  -> 预测 same-tile 驻留带来的增量延迟，而非只预测绝对延迟
     （解析 DMA 模型 + AutoTVM cost model）
  -> 在性能等价带内最小化部署命令 backing，并由 manifest 强制执行
     （AEx 的资源权衡 + USMP/DORY 的静态内存规划思想）
  -> 未通过证书或没有收益时回退 TopHub/original
```

统一研究问题是：**硬件已经定型后，如何用硬件知识减少无效搜索和外存访问，同时把最终部署所需
共享内存压到 exact allowlist 的安全下界。**

## 3. 逐类审计

| 论文/系统 | 已借鉴并实现 | 尚未实现或不能声称 | 最小下一步 |
|---|---|---|---|
| AutoTVM | ConfigSpace、XGB/Random、RPC measurement；新增 pre-measure semantic allowlist | 新模式与 TopHub 不是 tuner 内统一候选池；incumbent 是部署级回退 | 新 holdout 统一候选池做等 gross-budget 对照 |
| Banerjee 2021 | 驻留/tiling、exact DRAM bytes、分层 FSim→FPGA；W05 hybrid 有 same-tile 正收益 | 未忠实移植其 TPS 和消除冗余 LOAD 的 UOP reorder；无 TSim/作者 fork 基线 | 作者 fork/忠实 TPS 与本文候选在相同配置和预算下对照 |
| Rieber 2022 | 邻域、静态合法性 gate、valid/invalid/unknown 分流的部分思想 | 未复现 Algorithm 1 的 presampling、balanced/max-distance E0 与 SA bias | 20 seeds 严格复现三级消融，比较 compiler calls、valid yield 与 IQR |
| ML²Tuner 2025 | 已有 visible knobs、lowered-TIR/DMA 特征与分层门禁 | 没有独立 P/V/A 三模型；没有 hidden compiler feature shortlist | grouped holdout 先做 V，再做 P→compile→A 两级 performance |
| Ansor | 只有“按 stage 价值分预算”的计划 | 无 sketch/evolution/task scheduler 实现 | 冻结日志回放 uniform 与 occurrence×criticality 预算 |
| MetaSchedule | 外围实现了 verifier/postprocessor 式 fail-closed 门禁 | 未实现 MetaSchedule rule/postproc/database | 先抽象统一 `verify(candidate)`，暂不移植全栈 |
| Timeloop/MAESTRO/ZigZag | SRAM/traffic/reload 的窄版解析特征 | 无架构 YAML、NoC/energy/latency mapper | 三者只选一个，对 frozen mappings 做 traffic/latency 排序相关性 |
| CoSA | 静态约束与目标已分离，当前以 brute scan/Pareto 求解 | 无 MIP、未证明解析 filter 完备 | 小空间整数约束与 exact lowering feasible set 对照 |
| ROLLER | DMA Pareto 后按 TopHub tile 距离缩短 W05 shortlist | 不是从硬件原语推导的 rTile，也未在 unseen workload 成立 | 从 VTA cfg 推导 alignment/working-set distance 并做 leave-one-workload-out |
| DORY | 有 activation/weight working-set 与命令资源证书的组成部分 | 无完整多级 tensor lifetime/CP planner | 同时报告 data/control memory breakdown；必要时只做一个 conv 的 CP baseline |
| USMP | exact allowlist 定容、双 slot 与池复用思想 | 未建立跨 stage/submit/FINISH/replay 的统一 lifetime conflict graph | 比较 naive sum、per-queue max、lifetime packing 峰值并 fail closed |
| AEx 2023 | 显式研究性能—命令资源关系 | AEx 是架构 expand/prune 和 HLS/DSE；本文没有重综合，host backing 也不是片上 instruction SRAM | 在固定 VTA 上做 latency–backing Pareto，只称部署资源优化 |
| Cheng 2026 | 全文已核对 output/input-priority、片上权重复用、四方案、资源回退与三网络实验；W05/Y00/Y02/Y03 有本地机制数据 | 本地 barrier 不是论文式“不覆盖权重地址 + 条件 LOAD WGT + PushGEMMOp/UOP”实现；论文也没有研究 ConfigEntity 等预算搜索效率 | 将四方案作为机制基线；另做驻留×tile 在线搜索，以 time-to-target、总搜索 DMA/编译/板端成本对照 Random/XGB/论文规则 |

相关一手来源与正式题名已记录在
`FPGA_HARDWARE_AWARE_AUTOTUNING_LITERATURE_REVIEW.md`。其中 Banerjee 2021 的正式题名是
*A Highly Configurable Hardware/Software Stack for DNN Inference Acceleration*；Rieber 2022 是
*HW-Aware Initialization of DNN Auto-Tuning to Improve Exploration Time and Robustness*；
ML²Tuner 的 A 是带 compiler-hidden features 的 advanced performance model，不是 accuracy 模型。

## 4. 本轮已经补做的两个零上板实验

### 4.1 P7R109：论文启发式合法性与增量性能 replay

输入为 W00--W09 的 310 个 frozen local candidates（197 lowering valid、113 invalid），以及
W01/W04/W07/W08 的 56 个 same-tile FPGA 配对。严格按 workload 留一。

| 合法性方法 | budget=4 合法数 | budget=8 | budget=16 |
|---|---:|---:|---:|
| Random（1000 次） | 2.54 | 5.10 | 10.18 |
| Rieber-inspired 邻域 | 2.80 | 6.00 | 11.80 |
| ML²-inspired 基础 V 模型 | 3.70 | 7.50 | 14.20 |
| ML²-inspired 硬件 V 模型 | 4.00 | 7.60 | 14.50 |

这说明“先学合法性、再花编译/板端预算”有明显开发信号；但 label 只是本地 lowering status，不能
代替 FPGA correctness。当前 Rieber 项只是邻域思想，不是 Algorithm 1 完整复现。

| same-tile 增量模型 | 符号准确率 | regret@1 | regret@2 | regret@4 |
|---|---:|---:|---:|---:|
| 模式+计算上下文 | 80.5% | 11.38 | 3.87 | 1.94 |
| + DMA bytes | 75.5% | 0.02 | 0.00 | 0.00 |
| +完整 request signature | 75.5% | 0.02 | 0.00 | 0.00 |
| +命令/硬件特征 | 73.4% | 0.02 | 0.00 | 0.00 |

DMA bytes 对 top-1 排序有用；full request 与 command/hardware 没有新增收益。不能继续把“请求形态
一定优于 bytes”写成创新结论。产物位于
`07_grouped_holdout/20260911_p7r109_literature_guided_offline_replay_run02/`。

### 4.2 P7R110：性能—命令 backing Pareto

将 P7Q 的 79 个 latency 标签与 P4J run02 的 exact command peaks 相连，每个候选按 instruction、
UOP 各至少一页并分别向 4 KiB 对齐。四个 workload 的 latency oracle 都已经使用最小 8 KiB
backing，Pareto front 各只有一个点；在 0%、2%、5%、10% 性能等价带内也没有进一步节省。

这是明确的否定结果：当前小池不能证明 command-aware 联合目标改善最终选择。8 KiB 的价值仍是
部署定容，而不是搜索排序收益。产物位于
`07_grouped_holdout/20260911_p7r110_latency_command_pareto_run01/analysis.json`。

### 4.3 P7R113：显式标定的 same-tile `ΔT` 物理代理

对相同 56 个配对改用 `ΔT_ms=T_mechanism-T_original`；相同 workload 和 ConfigEntity 使 MAC、tile
与 compute lower bound 在配对内相消。模型只在三个 workload 上标定，物理代价系数约束非负，
mode intercept 允许正负。它是解析结构的线性代理，不是理论上下界。

| 模型 | MAE (ms) | Spearman | nDCG@4 | regret@1 (ms) |
|---|---:|---:|---:|---:|
| constant mean | 0.410725 | -0.0958 | 0.4406 | 0.895892 |
| mode mean | 0.303263 | 0.4922 | 0.6164 | 0.417852 |
| mode + bytes/calls | **0.119648** | **0.8356** | 0.9487 | 0.327558 |
| + request shape | 0.155524 | 0.8075 | 0.9593 | **0** |
| + command | 0.191115 | 0.7663 | 0.9616 | 0.000915 |

结论不能简化成“request 一定更好”：request 使四个留出 workload 的 `regret@1=0`，但 MAE 和
Spearman 比 bytes/calls 下降；command 也没有稳定新增收益。最稳妥的新信号是“same-tile 增量
建模 + bytes/calls 物理特征”明显优于模式均值，request 排序信号需要新网络前瞻确认。产物位于
`07_grouped_holdout/20260911_p7r113_analytic_delta_model_run01/`。

### 4.4 P7R112：Rieber 2022 严格方法审计与稀疏池降级复现

当前 310 行并非完整 ConfigSpace：每个 workload 只有 31 行、7 个独特 ConfigEntity，并分属 5 个
residency task；observed-domain 的邻居闭合率只有 29.55%。因此无法忠实复现论文的 E0=50、最多
1000 点 presampling、750 次测量与真实 SA，只能在 observed-knob-rank 域中做降级复现。

20 seeds、每 workload 固定 16 次不重复 compiler calls 的结果如下：

| 方法 | valid@16 中位数 [IQR] | invalid ratio | 首 4 valid 所需 calls |
|---|---:|---:|---:|
| Random | 10.0 [2.0] | 0.375 [0.125] | 6.0 [2.25] |
| 冻结静态顺序 | **11.5 [5.0]** | **0.281 [0.312]** | **5.0 [4.0]** |
| R1 locality | 10.0 [3.0] | 0.375 [0.188] | 6.0 [2.0] |

R2 能构造 4 valid + 4 invalid 的 balanced/max-distance E0；R3 bias 把已知样本的 valid@12 从 7
提高到 8，但 @16 仍为 10。这只是 label 重排，不是性能或 SA 收敛。结论是：在当前稀疏池上
停止把 Rieber 邻域当主改进；若要严格复现，必须重新生成完整单 task ConfigSpace 和 latency
ground truth。产物位于 `07_grouped_holdout/20260911_p7r112_rieber2022_sparse_reproduction_run01/`。

### 4.5 P7R114：USMP 式生命周期图可行性审计

状态为 `insufficient_evidence_fail_closed`，没有生成虚构的 packing 节省数字。A/B/C/D 现有 32 帧
trace 各只观察到一个 queue，局部峰值均为 instruction 5312 B、UOP 2792 B；但缺少逐资源
alloc/free、物理 byte range、static↔runtime resource ID、frame/stage↔submit、独立 device_done、
逐批 FINISH 解码与 replay retained lifetime。

因此 naive sum 与 unified lifetime packing 均不可计算，per-queue max 只在已有单队列范围内成立。
最小事件字段、完整性条件和 fail-closed 规则已冻结在
`07_grouped_holdout/20260911_p7r114_usmp_lifetime_gap_audit_run01/instrumentation_contract.json`。

## 5. 实现边界与待修缺陷

1. 本轮已把 hardware correctness ledger 升级为 semantic v2：完整 `ConfigEntity` 进入 key，
   `config_index` 只保留为审计信息；旧 v1、缺实体、实体/TIR/硬件变化均 fail closed。28 个相关
   回归测试通过。旧冻结证据不会被静默升级，后续合同必须显式重冻。
2. `tune_resnet18_vta.py` 只验证并记录 command certificate，不会自行给远端 runtime 注入缩容环境；
   8 KiB enforcement 属于后续 manifest/runtime 部署阶段。
3. 每点 direct RPC correctness 默认单 seed；正式通过必须来自独立的多 seed FPGA canary。
4. incumbent protection 是 deployment-level fallback，不是 experimental tuner 内的公平共搜。
5. 分层 certificate 已实现，但自适应 promotion 和等成本对照未实现，故不称 multi-fidelity optimizer。

## 6. 不重复旧实验的执行顺序

1. `P7R111`：semantic hardware-ledger v2 与重编号/篡改/fail-closed 回归，已完成（28 tests）。
2. `P7R112`：已完成降级复现并得到负结果；严格复现受完整单 task ConfigSpace 缺失阻断，停止该分支。
3. `P7R113`：显式标定的解析 `ΔT` 模型，已完成；bytes/calls 有正信号，request 混合，command 阴性。
4. `P7R114`：已完成可行性审计；按冻结 instrumentation contract 补生命周期观测后才能算图。
5. 若继续性能路线，冻结 3 个全新网络几何、每个最多 12 点的板端合同；统一比较
   Random/XGB、hardware initialization、validity model、DMA/request/command 消融，budget 为
   4/8/12。所有正确候选都计时以得到真实 pool oracle。

旧 P7Q/W05/E00--E03 仅作为 development，不能再次充当 confirmation。P8 stage/FPS 与三次独立
boot 仍由 G7 阻断。
