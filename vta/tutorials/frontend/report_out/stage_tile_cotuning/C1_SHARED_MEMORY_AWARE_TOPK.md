# 第一创新点：共享资源、共享内存实现与通信代价感知的 CPU--VTA 流水线 Top-K 划分

更新日期：2026-09-15。

状态：`E0_E1_CLOSED; E2_COMPILE_DMA_CLOSED; E2_SERVICE_3_OF_3_CLOSED; E3_SHARED_TOPK_OFFLINE_FROZEN; UNIFIED_STRUCTURED_ORDER_DIAGNOSTIC_CLOSED; E3_CONTENTION_ONE_BOOT_ABSTAIN_CLOSED; E3_BOARD_3_OF_3_CLOSED; E3_T0_CONDITIONAL_SINGLE_CAMPAIGN_CLOSED; E4_COMPILER_MECHANISM_11_OF_12; E4_EXISTING_NATURAL_REUSE_0_OF_12_FAIL_CLOSED`

本文件是第一创新点的当前治理入口。第一创新点研究“切在哪里、使用哪种已认证边界实现、先测谁”；
第二创新点负责跨 Executor 双 slot、输入池复用和所有权协议；第三创新点负责 VTA 内部 DDR--SRAM
驻留、DMA 请求和命令资源。C1 消费 C2/C3 的证书，但不重复声明其运行时贡献。

## 1. 研究问题与最终表述

在冻结的 ResNet-18、VTA、TVM 和 TopHub 环境中，对每个卷积 workload 固定经审计的 tile，并联合
选择：

```math
x=(P,\boldsymbol\Theta,\mathbf t,\boldsymbol\rho),
```

- `P`：连续 CPU/VTA segment 与 VTA island；
- `Theta={theta_w}`：VTA workload 的 Tile/schedule；冻结域内按 workload 精确复用，域外局部重编译或重调优；
- `t`：各 CPU stage 线程参数；
- `rho`：各边界的 `ordinary_copy`、`external_k2_shared` 或 `pool_anchor_k2_shared` 实现。

未知策略、容量不足或环境不匹配必须回退 copy。目标不是证明“零拷贝必然更快”，而是把切图造成的
融合变化、边界合同、共享 slot、同步和共享 DDR 压力纳入同一可验证 Top-K 过程。

> 针对嵌入式 CPU--FPGA 共享内存推理中物理计算资源复用、切图通信与边界内存实现相互耦合的问题，
> 本文提出共享资源、共享内存实现与通信代价感知的 CPU--VTA 流水线 Top-K 划分方法。该方法通过
> 编译上下文审计建立固定 Tile 的条件化复用基础，并将 segment 传输分解为卷积基础 DMA、融合残差、
> 边界合同和 CPU 辅助服务；随后把切图、CPU 线程与经认证的 copy/双 slot/输入池复用策略组成候选，
> 以单物理 VTA、
> CPU 核心池、共享 slot 容量、同步和共享 DDR 压力构造结构化资源向量，并用 k-best DP 与带内
> Pareto 重排生成拓扑多样的板端测量序列。未知策略严格回退 copy，从而在不把逻辑字节误当真实
> DDR 流量的前提下减少重复完整编译，并提高有限板端预算的可解释性与利用效率。

## 2. 为什么共享内存不能只写成 boundary bytes

同一个边界至少有六层语义：逻辑 tensor contract、框架 set/get 物化、CPU/VTA 同址可达性、
slot/generation 所有权与同步、graph/slot/queue 空间高水位、cache/coherency/VTA DMA/DDR 争用。
所以共享内存不等于 `B_edge=0`。零拷贝只令特定框架物化项为零，不能自动消除 CPU 访问、VTA
LOAD/STORE、cache/coherence 服务或同步等待。

现有三启动证据正好支持这一边界：框架物化从 1,806,336 B/frame 降为 0，边界 API service 从
1.446899 ms 降为 0.041830 ms；但稳态 II 改善的 95% 区间为 `[-2.477096, 1.363220]` ms，跨 0。
copy 位于流水 slack 时，局部服务下降不等于 FPS 提升。

## 3. 固定 Tile 与融合感知模型

### 3.1 复用条件

对 workload `w` 固定 `theta_w*`，只在 workload key、normalized primitive/TIR、target、VTA 配置、
TVM/编译器哈希一致，AutoTVM 无 fallback 且片上容量合法时复用。新 workload、新融合类、动态 DMA
或环境漂移必须局部完整 lowering。固定 tile 是冻结域内的经验解耦，不是跨模型定理。

### 3.2 切点--AutoTune--DMA 的因果关系

切点 `P`、Tile `Theta` 和边界实现 `rho` 是三个不同的因果变量，不能写成“切图决定张量大小，张量
大小直接决定 DMA 次数”。对卷积 occurrence `o`，应先由切点确定设备归属、融合上下文 `g_o` 和
调用次数，再由精确 schedule 决定 lowered loop 中请求的重复次数：

```math
\Phi_{conv}(P,\Theta)=\sum_o n_o(P)\,\phi(w_o,\theta_{w_o},g_o),
```

```math
N_{k,d}(w,\theta,g)=\sum_{q\in Q_{k,d}}
\operatorname{mult}(q;\operatorname{loops}(\theta,g)),
```

```math
B_{k,d}(w,\theta,g)=\sum_{q\in Q_{k,d}}
\operatorname{mult}(q)\,x_q\,y_q\,s_d,
```

其中 `k∈{LOAD,STORE}`，`d∈{inp,wgt,acc,out}`。张量 shape/dtype 给出逻辑 payload 和传输下界；
Tile 决定循环分块、片上复用、padding、stride、请求重复次数和平均请求粒度；融合上下文决定是否出现
ACC/ewise/中间结果传输。因此 AutoTune 不是可另加到通信模型上的一项成本，而是 `theta` 的产生器；
只有把选出的 ConfigEntity 放入最终融合上下文并 lowering 后，才得到该程序的精确 DMA signature。

已有证据通过三个控制轴把关系拆开：

| 反事实轴 | 保持不变 | 已观测结果 | 模型含义 |
|---|---|---|---|
| 改切点 `P` | 固定审计 Tile | `03..15→03..16` 新增 projection：LOAD `+66` 次/`+219648 B`；`03..16→03..17` 仅新增 CPU tail，VTA DMA 增量为 0 | 切点通过归属与融合改变 DMA，不是仅由边界 tensor 大小决定 |
| 改 Tile `theta` | 固定 workload/placement | `tile_w` 合法样本 LOAD calls/bytes/time 中位为 `4.5×/4.173×/3.374×`；`oc_nthread` DMA 不变而 time 为 `1.302×` | schedule 同时改变 DMA 和非 DMA 执行；相同字节也不保证相同时间 |
| 改策略 `rho` | 固定切点/Tile | Experiment C 的 materialized `100352 B/1 copy` 变为 shared `0`，但均为 `522 LOAD/8736256 B`、`10 STORE/125440 B` | CPU--VTA/VTA--VTA 共享边界物化与 VTA 内部 DDR--SRAM DMA 分层 |

另一个同 workload、同 graph、同 boot 的 R50A Tile 对照给出更强的请求碎片反例：右侧 Tile 的 LOAD
bytes 少 `23.25%`，但 LOAD calls 多 `622.22%`，时延反而高 `112.84%`。因此近优候选必须联合保留
分 memory 的 bytes、calls、small/strided calls 和平均请求大小，禁止使用总 DMA bytes 的标量排序。

完整反事实证据已由只读汇总器冻结到
`compile_context_audit/c1_cut_tile_dma_causal_synthesis_run3/analysis.json`，artifact SHA-256 为
`b52546686cb751bcc23cef428f948702312c8acae7a5a7d6e549a8ea79a979b6`。该汇总同时锁定分析器与所有
源文件 SHA-256，只复用旧实验，没有
重新执行 AutoTune、完整编译或板端测量。

若需要显式检查切点与 Tile 的交互，可定义：

```math
I=C(P_1,\theta_1)-C(P_1,\theta_0)-C(P_0,\theta_1)+C(P_0,\theta_0).
```

但当前不为所有切点重复完整笛卡尔搜索：冻结 ResNet-18 域内，693 个 placement record 对应的 19/19
语义卷积均保持一个 workload/config，故主流程按 workload 复用 `theta_w*`；只对新 workload、新融合
类、环境漂移、fallback、动态 DMA 或容量失败做局部完整 lowering/调优。该策略减少重复成本，同时
保留交互项的 fail-closed 出口。

### 3.3 传输分解

```math
\Phi(P,\Theta,\rho)=\Phi_{base}(P,\Theta)+\Delta\Phi_{fuse}(P,\Theta)
+\Phi_{edge}(P,\rho)+\Phi_{host}(P,\rho),
```

```math
\Delta\phi_{fuse}(g)=\phi_{full}(g)-
\sum_{i\in Conv(g)}\phi_{conv}(w_i,\theta_{w_i}^{*}).
```

融合残差允许为负，表示融合消除了中间 LOAD/STORE。统一保留：

```math
[B_{inp},B_{wgt},B_{acc},B_{out},N_{load},N_{store},
B_{edge}^{logical},B_{framework}^{materialized},B_{slot},
B_{cpu},N_{convert},T_{sync}].
```

不同资源和单位不直接相加。逻辑合同与策略无关；普通 runner 下框架物化为 `m_e B_e`，共享策略下
为 0；物理 DDR 流量只有计数器或合格物理模型才能给出，不能由 payload 替代。

融合类键包含 normalized DAG、shape/dtype/layout、固定 tile、调用次数、CPU helper 类型及环境哈希。
状态为 `exact_reuse/new_fusion_class/new_workload/environment_mismatch/unsupported_dynamic_dma`，只有
`exact_reuse` 可免完整构建。

## 4. 共享边界策略证书

`SharedEdgePolicyCertificate` 必须绑定：

- exact candidate/topology 和完整 edge-id 集；
- tensor contract、arity、layout、dtype、连续性；
- graph/lib/params/runtime/driver/bitstream 哈希；
- virtual/physical 范围、256 B 对齐和 u-dma-buf 池内性；
- storage-id 独占、参数排除、读写者和跨边冲突；
- K2、slot 来源/字节、owner 与 generation 状态机；
- coherency 模式、同步协议、正确性 boot 数和性能统计资格。

证书只对完全相同的 candidate/edge/policy 生效。edge 集不一致、功能门失败、容量超限或缺证书时，
代码一律回退 `ordinary_copy`。单启动证书只允许使用已验证的功能和空间特征；吞吐置信区间跨 0 时，
即使三启动正确，也不能修改 M1 分数。

三项创新的接口为：

```text
C1 compile audit: workload/tile + fusion residual + logical edge
                              |
                              v
C1 Top-K resolver <--- C2 shared-edge certificate
  partition/thread/policy     binding, slot, ownership, sync, correctness
                              |
                              +--- C3 workload certificate
                                   internal DMA/request/command signature
```

## 5. Top-K 算法与计费边界

M1 已把普通 copy 的 CPU owner service 和 VTA mutex service归属到相应资源，并包含 CPU 核心池、
单物理 VTA 和共享 DDR 下界，因此不能再次把所有 DMA/边界时间线性加到 `M1_II`。

```math
\widehat{II}(P,\Theta,\mathbf t,\rho)\ge
\max\{D_{CPU-pool},D_{single-VTA},D_{shared-DDR},\max_j S_j,D_{sync}\}.
```

当前策略吞吐差量尚未通过统计资格，故 M1 保持主排序；共享策略通过硬可行性和结构化向量影响 2%
近优带内的测量顺序。流程为：

1. M1 k-best DP 生成合法 `(P,t)`；
2. 按 topology 去重，只保留各拓扑最优线程实现；
3. 查询 exact policy certificate，未知项回退 copy；
4. 始终保留 M1 第一；
5. 对最优值 2% 内的其余候选，以
   `[framework materialization, slot bytes, per-memory VTA DMA bytes/calls, small/strided calls,
   average request size, islands, CPU helper, sync]`
   做确定性 Pareto peeling；
6. 带外继续按 M1，完全相同时按 topology-id；
7. 只重排、不删除，不称为安全剪枝或全局最优保证。

这里必须区分两个已经冻结的版本，不能事后混写。最初的融合感知 F2 确实使用
`[VTA service, CPU helper bytes, boundary materialization, fragmented DMA calls, convert calls]`，但自然
Top-20 被同一 topology 的线程变体占据，F0/F1/F2 Top-5 完全相同，因此按预注册规则没有上板。随后
用于当前 boot0--2 的共享感知 S2 先做 topology 去重并引入 exact shared-policy certificate，实际冻结
维度为 `[framework materialization, additional slot, total VTA DMA calls, VTA islands]`；它产生了可检验
的不同顺序，但当时尚未把 per-memory 和请求密度接回同一接口。

当前实现已补齐统一细粒度特征：inp/wgt/acc/out bytes 与 calls、small/strided calls、LOAD/STORE 平均
请求大小及其 request-density、CPU helper bytes/calls、conversion calls、slot 和边界物化。只读诊断在
4623/4623 个 topology 上全部 `exact`；统一 Top-5 与冻结 S2 的候选/策略集合相同，仅交换第 4、5 位。
由于该统一顺序在已有两个 boot 标签后才密封，它不能取代 S2 的前瞻合同，已有 boot 也不被事后改称
统一顺序的前瞻证据。这个限制保留了实验有效性，同时说明下一版无需增加候选集合或重复测量。

按 topology 去重是重要修正：旧 P8 Top-20 有 20 个 candidate-id，却只有 4 种拓扑，其余是 CPU
线程变体。直接用它做 Top-K 会把预算集中在同一切图，无法检验边界数、slot 容量和 VTA 重入。

## 6. 实现与结果

### 6.1 固定 Tile / 融合传输

- `fusion_aware_transfer.py`：模板、环境合同、五类状态和结构化向量；
- `run_vta_fusion_aware_transfer.py`：23 代表校准和 64 段预测冻结；
- `validate_vta_fusion_transfer.py`：盲测真值门控；
- `analyze_vta_fusion_transfer_endpoints.py`：三组机制差分；
- `measure_vta_fusion_endpoint_service_boot.py`、`summarize_vta_fusion_endpoint_service.py`：E2 工具。
- `prepare_vta_c1_cross_model_e4.py`、`freeze_vta_c1_e4_fixed_tiles.py`：跨模型代表与结果前
  label-free Tile 冻结；
- `build_vta_c1_e4_cross_contexts.py`、`analyze_vta_c1_e4_cross_contexts.py`：两类融合上下文的
  exact dispatch、容量守卫和静态 DMA 差分。
- `audit_vta_c1_e4_natural_placement_reuse.py`：只读扫描既有 fullgraph 构建，只有 workload、完整
  ConfigEntity、schedule mode 和 normalized primitive 均精确匹配时才允许复用为自然 placement 证据。
- `synthesize_vta_c1_cut_tile_dma_evidence.py`：只读复用既有 placement、Tile、切点、共享边界和
  多 boot 证据，生成带源文件哈希的三轴反事实汇总；不调用编译器、AutoTune 或开发板。

87/87 segment、693/693 卷积 occurrence 已映射；19/19 语义卷积跨 placement 只有一种
workload/config。23 个完整代表建立 10 个 workload record 和 72 个融合类。64 段预测先冻结、后完整
lowering，64/64 逐项一致，总 DMA byte MAPE 为 0。冻结域在线完整模板构建由 87 降到 23，即减少
73.56%；64 次盲测构建是论文验证 gross cost。

三组端点 `05..05/06/07`、`10..10/11/12`、`15..15/16/17` 已证明新增 projection 时基础
DMA/ACC 按模型变化，新增 CPU tail 时 VTA DMA 不变而 host 工作变化。E2 service 的首个独立 boot
已经完成：boot ID `e274207e-e345-448f-b263-c615ff83c512`，81/81 样本输出正确且 DMA 计数逐项
命中；`samples.jsonl` SHA-256 为
`4b04128dc457c711210a82acecef937f88d316098512572eec4066655e3ea2c4`，冻结 contract SHA-256 为
`81aeabc9248053014ad9803a35a8a371c3d890a8fd4f994e48e5d255cd35f683`。

第二个独立 boot `e36394f4-7dd9-4276-8229-867911b31dce` 已完成，产物为
`compile_context_audit/c1_fusion_endpoint_service_boot2_run1/summary.json`，文件 SHA-256 为
`5fdde8b4af03eff746818f5404821e4a1ad523c0d3360f128f62a434d532df5d`；同样 81/81 样本通过完整输出、
精确 DMA 与 service 分解门。两份去除启动快照后的 contract identity 均为
`12989c64fe6387dc0e57f341492f576833030d99fb62963b60ef1e6a99d262c0`，两启动全部完整输出哈希一致。
专用 RPC 已在完成后关闭。

第三个独立 boot `17cec1fe-4bb5-4f9e-90fb-eaf0e5f1b2b8` 已完成，产物为
`compile_context_audit/c1_fusion_endpoint_service_boot3_run1/summary.json`，文件 SHA-256 为
`65127da785176eb7b146f0c016221192695757ac0ed2318e51bebf6306efa88c`。三启动权威汇总为
`compile_context_audit/c1_fusion_endpoint_service_three_boot_summary_run1/service_summary.json`，
artifact seal 为 `ec5f3439931ee4805ec63b33555974efa95b35d5444b20741c0ebe0b97b0b35e`，文件
SHA-256 为 `8ebaa2b4d12ef1262db6b18b9643bd2fb04fd63cce6c58745385680acf72c67d`。
243/243 样本均正确、DMA 计数精确，三个 protocol identity 与全部完整输出一致。第三次专用 RPC
同样已关闭，不再增加 boot 或重测端点。

| 端点组 | 新增 projection：driver Δms | 新增 CPU tail：driver Δms | projection：完整 stage Δms |
|---|---:|---:|---:|
| 05..05/06/07 | +1.224650 | +0.002500 | -0.346504 |
| 10..10/11/12 | +0.933550 | -0.015450 | +0.435035 |
| 15..15/16/17 | +0.917250 | -0.001600 | +0.928059 |

这里先在各独立 boot 内取中位数，再跨 boot 取中位数。projection 的 driver 服务均增加，而 CPU
tail 的 driver 中位差异绝对值不超过 0.015450 ms；这与前者增加 VTA 工作、后者内部 DMA 不变的
机制预测一致。但没有对尾部变化做统计等价检验，不能称为“driver 时间严格相等”。host/graph
remainder 在增加 CPU tail 后下降，而非必然增加；该 proxy 也包含转换、graph dispatch 和融合变化，
不是纯 CPU helper 时间。三组 projection 的完整 stage 增量符号并不一致，故必须保留分资源解释。
每个原始样本满足 `stage=driver+remainder`；各字段分别取中位数后不保证仍可直接相加，表中聚合
差异不能当作可加的确定性服务公式。

三 boot 汇总器已加固为 fail-closed：必须恰好三个不同 boot ID；去除 boot snapshot 后的 protocol、
runner、9 段二进制/参数、三输入 LLVM 参考和采样参数必须同一 identity；每 boot 的 81 个样本必须按
预注册旋转顺序连续编号，DMA difference 为空且 `stage=driver+host/graph remainder`；跨 boot 完整输出
哈希必须一致。最终 `service_summary.json` 已密封分析器及三份 summary/contract/samples 哈希。本次
contract 内 runner SHA-256 与当前测量脚本均为
`081488e4a91e3f0ab13fd76565dcdf846e62da596b31244ada14923f080d9b6e`，三个 boot 保持同一 lineage。

### 6.2 共享内存感知 Top-K

- `shared_memory_topk.py`：证书校验、copy/shared realization、容量回退、拓扑去重和 Pareto；
- `freeze_vta_c1_unified_structured_order.py`：把融合 helper/conversion 与分 memory DMA、请求密度、
  shared slot/物化统一到同一 Pareto 接口；结果只作 post-two-boot 实现诊断，不改写冻结 S2；
- `run_vta_shared_memory_topk.py`：组合全域审计和已有共享实测，冻结 S0/S1/S2；
- `prepare_vta_shared_memory_topk_board.py`：把 9 个冻结配置解析为 5 个精确 package、4 份
  pool-anchor plan 和三组预注册 boot 内顺序；
- `run_vta_shared_memory_topk_board_boot.py`：恢复并逐项校验去重内容仓，先做三输入完整输出字节门，
  再采集 II、物化、slot、高水位和 VTA 内部 DMA；
- `analyze_vta_shared_memory_topk_board.py`：三 boot、union oracle、Regret@K、首次命中与机制门；
- `run_cpu_vta_pipeline_v1_p7_profile.py` 的 `p7c-ddr/p7c-multiboot`：分别生成单 boot
  CPU--VTA matched control 和严格的三 boot 配对验收；
- 三个对应测试文件覆盖功能、部署冻结和阴性结论路径。

权威冻结目录为 `compile_context_audit/c1_shared_memory_topk_run5/`；其
`selection_contract.json` SHA-256 为
`66f846e473e45f37ce26ba9e995016e46810035ec2ca089988fc7bafd764f5a4`。离线结果：

- 4623 个拓扑、25,410 个边 occurrence、739 个保守 edge context；
- 边界 payload 200,704--6,823,936 B，external-K2 必要容量 401,408--13,647,872 B；
- 8 张 exact policy certificate，覆盖 4 个拓扑的 external-K2 与 pool-anchor-K2；
- 1 张 external-K2 有三启动功能证据，0 张有合格吞吐收益；
- 未覆盖候选全部回退 copy；
- 1/2/3/4/8/16 MiB external-K2 容量上限仅按必要条件分别容纳
  17/98/350/846/4018/4623 个拓扑；这不是零拷贝资格证明；
- S0/M1-copy、S1/M1+payload、S2/M1+certified-shared 的顺序不同；冻结并集为 5 个拓扑、9 个
  `candidate+policy` 配置。

统一结构化诊断位于
`compile_context_audit/c1_unified_structured_order_diagnostic_run1/analysis.json`，artifact SHA-256 为
`a717d18d95a47fb5e6953b86609aa634f00e656e622b40963b5587084d05967a`。它未读取任何板端产物，锁定
脚本、库、融合估计、共享证书和旧 selection contract 的哈希；4623/4623 行完整，Top-5 与 S2 同集合、
不同顺序。其状态是 `post_two_boot_diagnostic_freeze`，明确不进入当前 E3-SM2 的正式比较。

现有 pool-anchor-K2 在四拓扑将额外 slot 降低 36.36%--44.44%，总池高水位降低
1.027%--1.215%；这是单启动分配/正确性资格，不是吞吐证据。

### 6.3 前瞻板端 boot 0--2（三启动闭合）

部署冻结目录为 `compile_context_audit/c1_shared_memory_topk_board_prepare_run8/`，其
`deployment_manifest.json` SHA-256 为
`23d4d0cf241cc7cce2c7c22afe2b3448d98c3636fdbec66db1ffec1608e95f99`。它在产生新性能标签前固定了
9 个配置、5 个拓扑、4 份 pool-anchor plan 和三种独立 boot 内顺序。第五个拓扑复用相同 ordered
stage DAG 和编译产物，仅将 CPU runtime thread map 从 `[1,1,2,1,2]` 改为 `[2,1,3,1,4]`；VTA
stage 线程均保持 1。

boot 0 产物为 `compile_context_audit/c1_shared_memory_topk_board_boot0_run6/boot.json`，boot ID
`5cc28376-3132-49bd-88d2-92353d45f2bc`，文件 SHA-256
`e2e38869162a4b22f20701e9fbe3af662688a7e9b3bc84a48e890baf955a9594`。9/9 配置均先通过三个确定性
输入的串行参考--流水完整输出逐字节相等，再完成 302 帧测量。四组同拓扑 copy/anchor 对照均满足：

- framework materialization 从每帧 1,806,336--2,609,152 B 降为 0；
- VTA 内部 LOAD/STORE 字节及请求数逐项完全相等；
- pool-anchor 的额外 slot 为 1,003,520--1,605,632 B；
- 未认证的第五拓扑严格使用 ordinary-copy。

该 boot 的 union 最小 II 为第五拓扑 ordinary-copy 的 `83.521695 ms`。按本 boot 描述值，S0/S1 在
各自第 5 个候选首次进入 oracle+2%，S2 在第 4 个候选进入；三者 Regret@5 均为 0。这只是一个
experimental unit，不能用于通过 S2 主张门。五次失败尝试均发生在模型加载和计时前，依次暴露并
修正了重启挂载点、FPGA manager 不导出 firmware 回读属性、已弃用 package-local `libvta.so` 不应
作为远端依赖，以及旧 package 去重内容仓断链等部署问题；它们不计入性能样本。内容仓已由主机冻结
产物按 SHA-256 去重恢复。

最终分析器还会从 `boot.json` 反向验证 deployment/selection hash、boot-index、预注册运行顺序、每个
性能 JSONL、VTA profile、正确性输出 tar 及串行参考的完整字节；boot 0 的这条原始证据链已通过，
但分析器仍拒绝少于三个不同 boot ID 的正式结论。

物理重启后的 boot 1 使用同一 selection/deployment hash 和预注册 `boot_index=1` 顺序。产物为
`compile_context_audit/c1_shared_memory_topk_board_boot1_run9/boot.json`，boot ID
`e274207e-e345-448f-b263-c615ff83c512`，文件 SHA-256
`25193c4b7c39e1da811245cbf0eacda54f7661133a61ff4657db0c67b5e1ebe5`。9/9 配置同样先通过三个
确定性输入的完整输出门，再各完成 302 帧；最低描述性 II 为未认证第五拓扑 ordinary-copy 的
`84.972147 ms`。四组同拓扑 copy/anchor 再次满足 framework materialization 由
`1,806,336--2,609,152 B/frame` 降为 0，且 VTA 内部 DMA calls/bytes 不变。两次独立 boot 的机制结果
一致。以下 boot 2 和正式三启动分析补齐了这条证据链。

boot 2 产物为 `compile_context_audit/c1_shared_memory_topk_board_boot2_run1/boot.json`，boot ID
`e36394f4-7dd9-4276-8229-867911b31dce`，文件 SHA-256
`9369d23dd1481bd621e033e62b12d3b8b8cccc6c8418379256d853cd2bd13028`。同一冻结合同与
`boot_index=2` 下，9/9 配置先通过完整输出门，再各完成 302 帧。连接时通过用户串口公钥核对新的
SSH 主机身份，使用独立固定密钥记录和显式 RSA 兼容选项；未改变冻结测量源码或任何候选。身份与
环境恢复记录位于 `compile_context_audit/c1_boot_e363_transport_attestation_run1/README.md`。

权威三启动结果为
`compile_context_audit/c1_shared_memory_topk_board_analysis_three_boot_run1/analysis.json`，文件
SHA-256 `2d0797c3a9054025d64e271470b5d0aacfbebb8da734fae9908491f0100de725`。分析器逐项反向验证
原始输出、性能 JSONL、profile 和冻结身份，得到：

| 方法 | 首次命中 union oracle+2% | Regret@5 |
|---|---:|---:|
| S0：M1-copy | 5 | 0 |
| S1：M1+payload | 5 | 0 |
| S2：M1+certified-shared | 4 | 0 |

union oracle 的跨启动中位 II 为 `83.935836 ms`。27/27 个配置--启动正确；四组同拓扑
copy/shared 在三启动中均满足物化降至 0、VTA 内部 DMA bytes/calls 完全不变。预注册主张门通过，
故可声明冻结候选并集、三启动中位评价与既定预算下，S2 首次合格命中少一个候选（5→4）。这不是
硬件全局最优、共享策略吞吐收益或统一 post-label diagnostic 的前瞻验证。

原三启动完整 T0→合格候选墙钟没有在冻结 runner 中单独封存；每配置的 correctness/performance subprocess
墙钟已保留，但不包含完整搜索、部署及数据回收成本，不能冒充端到端 T0 指标。因此不声明端到端
墙钟节省；三启动成功也不自动补齐这一次指标的缺失。用户随后授权的独立成本补测见6.4，
不回填本节三启动的T0，也不混入原排名分析。

为避免仅报告阳性计数指标，新增只读成本会计
`account_vta_c1_topk_observed_budget.py`，从已验证的原始 profile 与 boot 记录按冻结顺序和跨启动
中位首次命中位置进行前缀统计，不调用编译器或开发板。权威结果为
`compile_context_audit/c1_topk_observed_budget_accounting_run1/accounting.json`，artifact seal
`8780141c79a09bf1ff4c318be820c5ac4fc5fae24f942f0fcdcd755ad3328788`：

| 方法 | 达标前缀配置数 | 性能阶段 driver_run_calls | correctness+performance 子进程成本中位秒 |
|---|---:|---:|---:|
| S0 | 5 | 22348 | 155.524713 |
| S1 | 5 | 22348 | 155.524713 |
| S2 | 4 | 17818 | 156.757045 |

driver_run_calls 在三个启动中完全一致，计数仅涵盖性能进程，不含串行参考/正确性 dispatch 或 FPGA
重新编程次数。S2 子进程成本描述值反而略高，故候选数和 dispatch 减少不能换算为时间收益。
这些是按策略顺序重放的成本前缀，不是实际交错并集顺序的经过时间，更不是完整 T0。当前 E3
复用所有已冻结编译产物，板端协议中新 VTA 完整编译数为0；23代表＋64盲测的历史 gross 编译成本
必须另报，不能用这个0代替。

### 6.4 唯一一次专门成本补测：条件化在线 T0

用户于2026-09-15明确允许“仅为补齐 T0 进行一次专门成本补测”。协议在新成本标签前冻结于
`compile_context_audit/c1_dedicated_t0_cost_run1/protocol.json`，文件SHA256
`9fd3ebcf9d92731d3bd7b5f51f80dff5c3d5856f414da2ab4c821eb21423bb52`。
仅运行一个campaign，使用既有boot `17cec1fe-4bb5-4f9e-90fb-eaf0e5f1b2b8`，
固定S0→S1→S2顺序、每臂最多5候选、首次合格即停止，无失败、无重试。

这里的T0以**冻结拓扑库存、资源profile、模板及编译包已经可用**为条件。从本地源/库存哈希审核前
的主机monotonic_ns开始，包含原顺序复算、预检、诊断/计划/输入部署、远端package/runtime校验、
按需三个输入串行参考、完整输出正确性、302帧性能计时、profile/allocation回收、漂移/boot检查与
合格决策持久化。每臂新进程、新临时目录；已有SD包和OS页缓存不清除。

它**不包含**从原始Relay开始的切图枚举、上下文重新提取、M1精确DP/资源profiling、离线AutoTune、
23模板校准、64盲测构建和板卡初始化，不能称为“从零到部署”的全部墙钟。
原selector/library的诊断字段版本增补已在协议中单列当前hash与旧hash；原数据源保持不变，
冻结前及每臂计时内都复算并要求全部Top-5顺序与原合同完全相同，原板端runner源码保持原hash。

合格阈值预先固定为旧三启动union oracle×1.02，即 `85.61455276775919 ms`，
不由新性能标签调整。任何候选错误禁止计时；五候选皆不合格则报告右删失，不补跑。
权威分析为 `compile_context_audit/c1_dedicated_t0_cost_analysis_run1/analysis.json`，
artifact seal `16bf65df27ae1df19eec831f1bb8520b97567d830dc5d45082fd5e826bc90d83`：

| 策略 | 本次首次合格候选 | 实测T0→首次合格秒 | 性能阶段driver_run_calls |
|---|---:|---:|---:|
| S0 | 5 | 202.880903 | 22348 |
| S1 | 5 | 201.046925 | 22348 |
| S2 | 4 | 200.683615 | 17818 |

14/14候选通过三个输入的完整输出门，各完成302帧；新完整编译数为0。分析器反向复核全部
原始输出档案、JSONL、profile、冻结顺序、阈值、单调时钟事件及文件hash。这里的T0是真实经过
时间，不是旧子进程前缀的相加，也不是原三启动完整T0的恢复。

S2性能dispatch少20.27%，但对最佳基线S1的墙钟只低0.18%。本次参考/正确性/性能子进程累计
分别为S1的7.097/8.077/147.202秒和S2的6.125/23.043/132.781秒；正确性子进程开销增大，
抵消了大部分性能测量时间减少。各项是含进程启动/远程调用的墙钟，不能解释为纯CPU helper服务。
单boot、固定臂序、共享页缓存带来时间/温度混杂，**不宣称显著或可归因的墙钟优势**。
本次结论是“候选与性能dispatch预算减少，但条件化在线墙钟基本持平”；保留旧成本阴性结果。
新标签仅用于独立成本终点，不进入6.3的三启动ranking/Regret，也不重开DMA/驻留/精度实验。

最终本地回归覆盖19个相关测试文件，79/79通过；包含成本计时4项和时钟复核3项新增测试。
本轮允许的板端补测已经全部结束，不增加boot或重复任何成功协议。

## 7. 实验收尾与结论门

### E3-SM2：前瞻 Top-K 板端比较

已完成冻结的 9 个配置与三个不同独立 boot。每个配置先用三个确定性输入逐字节检查完整输出，
失败者禁止计时；以 boot 为统计单位，boot 内交错顺序。结果与结论边界见 6.3，不再重跑。对相同
candidate 的 copy/shared 核对 VTA
LOAD/STORE bytes/calls 不变，同时报告 framework materialization、boundary API、slot wait、额外 slot、
pool high-water 和 II。

主指标为首次命中 union oracle+2% 的序号；次指标为 Regret@5、正确率、完整编译/FPGA dispatch 数和
T0 到首次合格候选的墙钟。仅当 S2 命中不晚于 S0/S1 且 Regret@5 不高于最佳基线时宣称筛选效率
改善；否则保留传输分解和证书贡献，报告排序阴性结果。

### E2-S：端点 stage service 三启动闭合

已按冻结的 9 个端点 × 3 个 case × 3 repeat 协议完成三个不同独立 boot，243/243 样本正确与
DMA 精确，协议身份及全部输出跨 boot 一致。汇总与机制限制见 6.1；不再重跑。该结果解释
projection、CPU tail 及转换的资源归属，不把 CPU/graph remainder 视为纯 CPU 时间，不要求
“DMA 少者 FPS 必然更高”。

### E3-SM3：共享 DDR 争用机制（单 boot 阴性收口）

已有 P7C runner 完成了一个 boot 的 4 组 matched control：CPU cache-resident/streaming × VTA
compute-heavy/DMA-heavy，每组均比较 CPU-only、VTA-only、CPU+VTA concurrent，并用独立 CPU affinity
和 start barrier 控制用户态核竞争。streaming 压力下 VTA slowdown 为 1.052x/1.066x，bootstrap 区间
下界均大于 1；但额外延迟大多位于 host runtime 的 run-minus-driver，且只有一个 boot，因此只能称为
CPU--VTA interference 信号，不能拟合纯 DDR contention 系数。当前 Top-K 实现读取这份冻结 summary，
保留 bytes/calls 特征但拒绝加入争用系数。

原 P7D 评审已决定：该信号没有解释冻结 Top-20 的排序主误差，且额外延迟不能唯一归因于 DDR，因而
不为拟合参数重复补 boot。已实现的 `p7c-multiboot` 合并器保留为未来协议工具，但现在增加了逐 boot
stage/memory runner 哈希门；旧 boot 1 没有封存 memory 二进制身份，所以不能被静默升级为
exact-protocol coefficient。资源 slowdown 仍定义为：

冻结 stage runner 已从 Git blob `493bac16c3a2a704dbf2e4bed42e6eb74cfbdfbd` 精确重建，固化二进制
SHA-256 `dbc0caa3...aec` 与 boot 1 session 完全一致；当前主线 runner 不能替代它。boot 1 未记录
memory microbench 的二进制哈希，因此即使后续统计门通过，也必须披露该 provenance 限制。P7C 只作
精确压力协议的机制证据，不阻塞 Top-K/E2，也不外推为候选级 DDR 曲面。

```math
s_r(q)=T_r^{concurrent}(q)/T_r^{isolated}(q).
```

为隔离“只是同时运行”与 streaming DDR 压力，正式的精确协议参数使用 cache-resident control 做
配对差分：

```math
\kappa_{DDR,c}=\operatorname{median}_{boot}
\frac{s_{VTA}(CPU_{stream},c)}{s_{VTA}(CPU_{cache},c)},\qquad
T_{VTA}^{contended}=T_{VTA}^{isolated}\kappa_{DDR,c}.
```

其中 `c` 仅为已测的 `compute_heavy/dma_heavy` 两类。现有结果只报告单 boot service slowdown 与
profiler proxy，对“真实 DDR 争用量” abstain。只有未来新的前瞻候选先证明争用是排序主误差，才启动
三个全新 boot 的协议；届时还必须补同总字节不同 request/stride 以及
copy/external-K2/pool-anchor-K2，并按 workload/contract class 做 grouped holdout。若 AXI/APM 计数器
可用，再报告 transaction/stall。即使统计门通过，也不能在候选之间连续插值。

Top-K 冻结器保留可选的 `--contention-multiboot-summary` 消费接口。当前没有可准入的三 boot
summary，因此该接口不改变 M1 或 Pareto 排序；这避免把弱干扰信号包装为共享 DDR 常数。

### E4：跨模型轻量验证

`prepare_vta_c1_cross_model_e4.py` 已从既有 label-free 算子合同冻结 ResNet-50 和 YOLOv3-tiny 各 6 个
几何多样代表。权威 readiness 产物为
`compile_context_audit/c1_e4_cross_model_readiness_run3/readiness.json`，artifact SHA-256 为
`9d2bea192d8c69fe7a19d74da6795dedc3b06551ef3138e07bb7b6c52e3a915b`。该产物同时锁定 preparer 自身
SHA-256；12/12 来源合同和候选哈希通过，
且 12/12 均被 C1 模型判为必须完整构建新上下文。

随后在查看跨上下文结果前，按既有 label-free `full_structural_order` 与 static+三 seed FSim 资格，为
12 个 workload 各冻结一个模型内 Tile。冻结产物为
`compile_context_audit/c1_e4_fixed_tiles_run1/fixed_tiles.json`，artifact SHA-256 为
`4a35a564d68185af090c8b9c57b1708d0eac00e52be2ce1a8a245770ce97e168`；选择过程不读取性能标签或后续
融合构建结果。

编译机制检查对每个固定 Tile 构造 `conv+quantize` 与 `conv+residual+quantize` 两个 normalized
Relay/TIR 上下文。权威 lowering 与分析分别位于
`compile_context_audit/c1_e4_cross_context_run1/summary.json` 和
`compile_context_audit/c1_e4_cross_context_analysis_run2/analysis.json`，artifact SHA-256 分别为
`f5bf16577ce2197e8500edc0eac1d6fd4d898032fd438ac1939922d69fdf1c26` 与
`89d42f03eceb642fc46965d7a2a7a8156a1c2e5c21aa025ddb0b9aeddaa6eb8d`。结果为：

- 11/12 workload 在两类上下文均 exact 命中事前冻结 ConfigEntity，且 normalized primitive 签名不同；
- 这 11/11 的 input、weight 与 output 基础 DMA 字节完全不变，残差融合均产生正的 ACC LOAD 残差；
- Y03 的基础上下文成功，但残差上下文需要 1,384,448 bit `local.acc_buffer`，超过 1,048,576 bit
  容量，因而按规则 fail-closed，没有事后更换 Tile。

这组结果验证的是“固定 Tile + 融合残差 + 容量守卫”的编译机制，不是跨模型定理：两个上下文是
synthetic paired context，而非真实 ResNet-50/YOLO 切图 placement；强制 exact dispatch 成功也不等于
自然 AutoTVM 查询会选择同一 Tile。既有 C3 合同同样不能替代 C1 的真实 placement、normalized fused
primitive 和 shared-edge certificate。故 E4 仍需补真实模型 placement 完整编译与边界策略覆盖；跨模型
板端加速不是 C1 成立条件。

为避免重复已有整图实验，新增只读复用审计扫描了 16 份既有 `fullgraph` build：12 个冻结 workload
中只有 R50G 出现 workload/ConfigEntity 匹配，但其 mode 为 `input_stationary` 而非冻结的 `original`，
且未保留 normalized primitive inventory。因此可复用 exact natural context 为 0，具备至少两个自然
placement 的 workload 为 0/12；没有把旧整图结果改名为跨 placement 证据，也没有因此重跑编译。权威
产物为 `compile_context_audit/c1_e4_natural_placement_reuse_audit_run1/analysis.json`，artifact SHA-256
为 `085d663c7d5643e93d458a30c11b28b738b9ac21f64c0141c7d3e22bc81c1b19`。自然 placement 扩展仍是
后续外部有效性实验，不是 ResNet-18 第一创新点成立门槛。

## 8. 相关工作与本方案的差异

| 工作 | 借鉴 | 本文差异 |
|---|---|---|
| [VTA](https://arxiv.org/abs/1807.04188) | task ISA 显式协调 memory/compute | 固定 VTA 上研究跨 CPU/VTA Executor 的切图、边界实现与单 VTA 复用 |
| [TVM](https://www.usenix.org/conference/osdi18/presentation/chen) | graph fusion、硬件 intrinsic 与低层 schedule 搜索共同决定程序 | 本文先审计 placement 下 workload/Tile 稳定性，再对最终融合 TIR 提取分 memory 请求签名 |
| [Ansor](https://www.usenix.org/conference/osdi20/presentation/zheng) | 分层生成 schedule、学习代价模型并跨 subgraph 分配调优预算 | 本文不重做通用 schedule 搜索；保留 TopHub incumbent，仅对新 workload/上下文局部调优并把结果作为 `theta` 输入 |
| [MAESTRO](https://arxiv.org/abs/1805.02566) | data-centric mapping 用 action count 分析数据复用、访问和性能 | 本文把思想落到 VTA lowered DMA request，区分 inp/wgt/acc/out bytes、calls 与碎片，而非只计 tensor payload |
| [Timeloop](https://research.nvidia.com/publication/2019-03_timeloop-systematic-approach-dnn-accelerator-evaluation) | workload、architecture、mapping 三元组共同定义存储层次访问 | 本文硬件固定，额外把 CPU/VTA 切点、融合 primitive 和经认证共享边界纳入外层 Top-K |
| [Interstellar](https://arxiv.org/abs/1809.04070) | loop blocking/ordering/parallelism 决定片上驻留和数据搬运 | 本文用既有单轴及同 workload Tile 对照证明请求数与字节不可互代，并采用 fail-closed 的精确 lowering |
| [Practical Performance Guarantees for Pipelined DNN Inference](https://proceedings.mlr.press/v235/archer24a.html) | 以含通信的瓶颈 stage 为目标 | 本文是固定链、异构交替和共享资源约束的 exact/k-best DP，不声称通用图近似保证 |
| [CoDL](https://www.microsoft.com/en-us/research/publication/codl-efficient-cpu-gpu-co-execution-for-deep-learning-inference-on-mobile-devices-2/) | 统一内存仍有 mapping、同步和变换开销 | 本文把融合残差、跨 Executor slot 与 VTA 显式 DMA 分层计费 |
| [HaX-CoNN](https://arxiv.org/abs/2308.05869) | 联合 layer 特征、共享内存争用和设备 transition | 本文聚焦单 DNN 多帧流水，先证书 fail-closed，再实测 slowdown |
| [Zynq I/O coherency study](https://research.ibm.com/publications/analysis-and-optimization-of-io-cache-coherency-strategies-for-soc-fpga-device) | coherency 效果依赖端口、访问模式和 cache 状态 | 证书必须包含 coherency/端口环境，禁止把同址访问视为免费 |
| [Linux dma-buf](https://kernel.org/doc/html/next/driver-api/dma-buf.html) | 地址共享、CPU coherency bracket 与 fence 同步职责不同 | 分别认证地址、coherency 和 generation/slot 同步 |
| [TVM USMP RFC](https://discuss.tvm.apache.org/t/rfc-unified-static-memory-planning/10099) | 编译期 workspace 生命周期和 offset | pool-anchor 不做通用图内重规划，只复用已审计输入池并处理跨 Executor/帧所有权 |

上述工作的共同启示是“mapping/Tile 决定内层数据动作，通信必须进入瓶颈目标、共享地址不等于零
成本、争用必须按访问模式实测”。本文不把 MAESTRO/Timeloop 式分析冒充 VTA 物理 AXI 计数，也不把
Ansor/AutoTVM 的搜索成本混成 DMA 成本；AutoTune 只产生 schedule，最终请求签名以融合后 lowering
和运行时资格为准。本创新点的可辨识增量不是一般的 zero-copy 或 pipeline partition，而是把以下三件
事放入同一可执行闭环：

1. 固定 Tile 条件下由 lowered TIR 精确重构融合相关逻辑传输；
2. 以 candidate/edge/environment 精确绑定的策略证书，把 copy、external-K2 和 pool-anchor-K2 变成
   切图搜索的受约束变量；
3. 在单物理 VTA、CPU 核心池和共享 DDR 的 max-plus 资源目标下，以 topology-diverse Top-K 做前瞻
   板端验证，并对未合格性能系数主动 abstain。

这三部分缺一不可：只有第 1 项是编译审计，只有第 2 项会退化为零拷贝运行时，只有第 3 项而没有
证书则会把不可部署候选错误纳入搜索。

## 9. 论文口径

允许宣称：冻结域 workload--tile 经验解耦；segment 逻辑传输精确重构；23 代表使在线模板构建减少
73.56%；C1 已将 exact copy/shared/anchor、slot 容量和同步资格纳入 Top-K；共享实验证明局部物化和
API service 改善；已有控制实验支持“切点、Tile、边界实现是三个不同因果轴”，并证明 DMA bytes
不能单独决定请求数或时延。E3-SM2 三启动主张门已通过，可增加“冻结候选池内首次合格命中
少一个候选（5→4），Regret@5 不劣”的有限预算结论；端点 service 三启动及243个原始样本已闭合。

禁止宣称：切图只影响融合；tile 普遍恒定；logical bytes 等于真实 DDR；零拷贝消除 VTA 内部 DMA；
最少 DMA/共享内存必然更快；单启动证书可外推吞吐；Pareto 保证全局最优；C1 重新发明 C2/C3 机制；
完整 T0 墙钟节省；统一 post-label 顺序已取得前瞻收益。

## 10. 复现

```bash
python vta/tutorials/frontend/run_vta_fusion_aware_transfer.py --output-dir <freeze-dir>
python vta/tutorials/frontend/run_vta_shared_memory_topk.py \
  --output-dir <shared-topk-freeze-dir>
python vta/tutorials/frontend/prepare_vta_c1_cross_model_e4.py \
  --output-dir <e4-readiness-freeze-dir>
python vta/tutorials/frontend/freeze_vta_c1_e4_fixed_tiles.py \
  --output-dir <e4-fixed-tile-dir>
python vta/tutorials/frontend/build_vta_c1_e4_cross_contexts.py \
  --fixed-tiles <e4-fixed-tile-dir>/fixed_tiles.json \
  --output-dir <e4-context-lowering-dir>
python vta/tutorials/frontend/analyze_vta_c1_e4_cross_contexts.py \
  --source <e4-context-lowering-dir>/summary.json \
  --output-dir <e4-context-analysis-dir>
python vta/tutorials/frontend/synthesize_vta_c1_cut_tile_dma_evidence.py \
  --output-dir <cut-tile-dma-causal-synthesis-dir>
python vta/tutorials/frontend/freeze_vta_c1_unified_structured_order.py \
  --output-dir <post-two-boot-unified-order-diagnostic-dir>
python vta/tutorials/frontend/prepare_vta_shared_memory_topk_board.py \
  --output-dir <board-deployment-freeze-dir>
python vta/tutorials/frontend/run_vta_shared_memory_topk_board_boot.py \
  --deployment-manifest <board-deployment-freeze-dir>/deployment_manifest.json \
  --boot-index <0-or-1-or-2> --output-dir <one-boot-dir>
python vta/tutorials/frontend/analyze_vta_shared_memory_topk_board.py \
  --contract <shared-topk-freeze-dir>/selection_contract.json \
  --deployment-manifest <board-deployment-freeze-dir>/deployment_manifest.json \
  --boots <boot1.json> <boot2.json> <boot3.json> \
  --output-dir <board-analysis-dir>

# P7C 已按单 boot 阴性结果收口，不在剩余冷启动中重复执行。
```

所有输出目录必须不存在。共享策略必须匹配 graph/lib/params/runtime/driver/bitstream 哈希，否则回退
copy，不能在当前证书上追加结果。
