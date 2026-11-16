# C3 selector v3：通用性、创新边界与验证路线

状态：`GENERALIZED_IMPLEMENTATION_AVAILABLE; COMPLETION_AWARE_MODEL_CONTRACT_AVAILABLE; DATA_GATE_FALLBACK; EXTERNAL_VALIDITY_PENDING`

## 1. 收敛后的第三创新点

第三创新点不再表述为“提出输入/权重驻留”或“最小化 DMA”，而收敛为：

> **面向显式 DMA、软件管理片上存储和解耦任务流水的加速器，提出组合安全的驻留—Tile 联合选择
> 方法。方法把驻留方式与完成策略共同作为程序身份，通过单候选、静态候选集合和精确图上下文三层
> 证书约束候选组合，再利用编译后数据请求、μop、指令、提交、同步和 arena 特征安排有限硬件测量；
> 只有完整图正确且跨启动稳定改善的精确候选集合才允许部署，否则回退 incumbent。**

这里的选择单位不是孤立算子，也不是若干算子分数的简单相加，而是：

\[
S=\{x_i\mapsto(w_i,n_i,f_i,p_i)\},
\]

其中 `x_i` 是完整 Tile/线程/驻留程序，`w_i` 是执行任务，`n_i` 是精确图节点，`f_i` 是生成函数，
`p_i` 是完成策略。候选集合必须满足：每个 workload 至多一个实现、图节点不重叠、局部完成作用域不
重叠、策略可用于生产，并绑定精确 graph/params/binary 身份。

## 2. 为什么它不是 mode10 特例

实现已经拆成四个正交层次：

1. `c3_completion_policy.py`：版本化完成策略注册表；mode10 只是
   `scoped_force_serial_completion_v1` 的首个实现。未知策略和诊断用全局串行均 fail-closed。
2. `c3_composability.py`：从策略合同生成本地、静态集合和图上下文证书，不再以数字模式作为选择器
   接口。
3. `c3_graph_candidate_set.py`：候选到 workload、图节点、函数和证书的精确身份；支持有界精确组合
   枚举，超出声明上限时拒绝，绝不把启发式截断冒充完整池。
4. `c3_selector_v3.py`：以精确图候选集合生成短名单和部署门禁；静态数量只安排测量优先级，不保证
   时延顺序。
5. `c3_selector_v3_model.py`：以精确图集合相对incumbent为训练单位，加入完成策略/作用域特征，按
   执行任务留一并以macro simple regret优先选择正则；训练和推理合同不足时fail-closed。
6. `c3_selector_v3_evaluation.py`：在完整参考池上以相同额外候选预算回放selector、bytes、calls、
   command和固定随机顺序；invalid尝试消耗预算，oracle严格限定为已完整测量的声明池。

因此，换用新的局部完成机制时，需注册新策略并提供相应证书生成规则，不需要重写图集合选择算法。
换网络时，图节点和函数由精确 workload 输入/权重形状映射并要求唯一命中，不依赖 YOLO 节点编号。

## 3. 与相关工作的边界

| 代表工作 | 已有范围 | 本文不能主张 | 当前具体增量 |
|---|---|---|---|
| [VTA](https://arxiv.org/abs/1807.04188) | task ISA 显式编排并发访存/计算任务，微码 ISA 与 JIT 编译 | 解耦任务流水、虚拟线程或 VTA 编译栈 | 驻留变换改变完成条件后，将完成策略纳入候选身份并验证图组合 |
| [Ansor](https://www.usenix.org/system/files/osdi20-zheng.pdf) | 自动生成张量程序；任务调度器在多个子图间分配调优资源 | 跨任务预算分配或学习 cost model | 被组合程序之间的完成策略兼容约束，以及精确候选集合图门禁 |
| [MetaSchedule](https://arxiv.org/abs/2205.13603) / [TensorIR](https://arxiv.org/abs/2207.04296) | 模块化搜索空间、程序变换和张量化程序优化 | 把新 schedule rule 加入搜索空间 | 完成策略证书作为硬语义门，并将图集合而非单程序作为测量/部署单位 |
| [Timeloop](https://research.nvidia.com/publication/2019-03_timeloop-systematic-approach-dnn-accelerator-evaluation)、[LoopTree](https://arxiv.org/abs/2409.13625)、[CoSA](https://arxiv.org/abs/2105.01898)、[COSMA](https://arxiv.org/abs/2311.18246) | 映射、Tile、驻留、容量、数据搬运、跨层保留或图级调度/分配 | 数据驻留、容量约束、联合映射或图级组合的一般概念 | 从真实 VTA lowering/命令/二进制提取完成作用域，并以板端整图正确性验证可组合性 |
| [Cheng 等](https://doi.org/10.1016/j.future.2025.108165) | VTA output/input priority × 原始/权重复用四方案和容量回退 | O/I/W/IW、全层权重驻留或 minimum-access | 论文四方案作为候选机制；新增的是完成策略、组合证书、图集合选择与安全回退 |
| [DAE 编译研究](https://doi.org/10.1145/3708493.3712695) | 分析 access/execute 解耦中的控制/数据依赖与同步代价 | 解耦执行、同步问题或保持并行的一般目标 | 在 VTA 驻留调优中将完成方式显式化，并用真实生成程序及图上下文结果驱动集合准入 |

基于上述一手来源的研究范围，当前没有证据支持“首次提出图级选择”“首次提出完成合同”这类宽泛
表述。可以防守的是这组约束在固定 VTA 上的具体结合、实现与实证：驻留—Tile—完成策略联合身份、
真实命令作用域证书、精确图候选集合和受正确性/跨启动性能门约束的选择。

## 4. 当前真实证据

YOLOv3-tiny 608 的两个 mode10 route 已迁移到通用图集合身份：

- candidate set：`0eaab41212c2ebb1da667080d278300994d48a9db02fafce2adae144c2333377`；
- `1024→256` route 唯一映射到节点38 / compute函数6；
- `512→256` route 唯一映射到节点61 / compute函数10；
- 聚合静态请求为1,132,544 B / 401 calls，UOP累计260 B，指令10,576 B，4次提交、2个局部
  completion scope；这些是两个被替换程序的数量聚合，不是完整图总流量，也不假设时延可加；
- 旧图正确性证书和 boot-1 计时通过 graph/params、候选成员和函数作用域的精确相等完成身份富化
  迁移，任何字段变化都会拒绝；迁移没有修改时延值；
- boot-1 六轮中位改善13.413%、6/6获胜，但部署门仍因缺少第二个不同 boot ID 而拒绝。

ResNet18 H3 提供负对照：mode10 恢复整图正确性，但相对 stock 为-0.250%、1/9获胜，因此 selector
回退 incumbent。这证明三层证书解决的是“能否安全组合”，不是把所有正确候选强行部署。

## 5. 还缺什么才能主张通用方法

当前只能主张“通用接口已实现、两个网络验证组合安全、一个真实两-route集合取得单boot正信号”。
以下证据仍缺失：

1. 同一 YOLO 集合在不同 boot ID 下原样通过正式门；
2. 至少一个新的 latency-unseen 网络/分辨率，在性能标签出现前由集合级 selector 冻结选择；
3. 至少一个三 route 以上集合，验证有界枚举、兼容冲突和图正确性门不是两点特例；
4. completion-aware模型接口已经实现，但准备度审计将现有v2的121条孤立算子行全部拒绝为图级
   训练标签；还需达到8任务/40比较、3个scoped任务、3个dependency-only任务，并提供匹配
   incumbent聚合特征后才能从静态回退切换到模型；
5. 在相同池和预算下，与 bytes-only、calls-only、固定随机和强 incumbent 比较选择质量。

第二启动当前不可用不妨碍第2--4项的本地准备。严禁用同一 boot 的更多 session 代替第1项，也不能
用当前已暴露 YOLO/ResNet18 重新调规则后再称 holdout。

## 6. 论文表述建议

当前可用表述：

> 本文发现，VTA 驻留候选的孤立算子正确性不能保证其在解耦命令流水中的图级可组合性。为此，本文
> 将完成策略加入驻留—Tile 程序身份，建立候选局部、静态集合和精确图上下文三级资格，并以受兼容
> 约束的图候选集合为单位组织编译后代价排序、有限板端测量和 incumbent 回退。该实现已在
> ResNet18 与 YOLOv3-tiny 上恢复历史错误组合的整图正确性，并在 YOLO-608 的一个正式启动中取得
> 13.413%的完整图中位改善；跨启动和未见任务结果仍待完成。

只有完成第5节的验证后，才能把最后一句提升为“在多个未见任务上以相同预算更可靠地选择可部署
程序”。

## 7. 关键产物

- `20260924_c3_yolo608_graph_candidate_set_run01/`：真实节点/函数映射、通用集合身份与图证书迁移；
- `20260924_c3_selector_v3_model_readiness_run02/`：121条v2行和现有图证据的准入审计，冻结
  `static_fallback`合同并绑定模型源码哈希；
- `20260924_c3_selector_v3_graph_set_plan_run04/`：显式读取审计合同后的集合级短名单、boot-1身份
  迁移和部署门禁，记录`model_contract_not_completion_aware_ridge`；
- `20260924_c3_selector_v3_evaluation_protocol_run02/`：未读取新时延前冻结的B1/B2/B3、1000随机
  顺序、regret/oracle命中和invalid预算规则；
- `c3_completion_policy.py`、`c3_graph_candidate_set.py`、`c3_selector_v3.py`、
  `c3_selector_v3_model.py`、`c3_selector_v3_evaluation.py`：通用实现。

本次真实池只有一个新集合且没有图级预测分数，因此它验证的是身份、正确性和部署门禁闭环，不是
selector v3 已学会预测图性能。completion-aware模型算法与数据门已经实现，但当前没有合格训练合同；
同预算基线比较仍属于后续未完成工作。
