# E6-S：全域共享边界静态审计

日期：2026-09-08。**E6-S contract 级审计完成；E6-R 编译/物理绑定资格化未完成。** 本轮未上板、未 AutoTVM、未新增流水 FPS。

## 做了什么

`audit_shared_edge_domain.py` 从冻结 unit schema/profile/local cost 与四种 CPU 线程的 atomic profile 重新枚举全域；每条路径检查 unit 顺序完整、不重复、不遗漏，以及相邻输出/输入 shape、dtype、张量顺序一致。边界方向、每 tensor payload、理论 K2 分配和生命周期均落盘。外部 NCHW 是正式 stage 接口的语义约定，不是内部 packed layout 或物理连续性的实测。

VTA 侧由 72 类已编译 primitive 的 DMA/host push/CPU helper 计数组合出 87 段账本，23 段与完整 TIR census 精确回归；其余 64 段明确标记为类组合估计，未冒充完整 build 或上板验证。内部 CPU helper 属于 VTA stage 执行，不能再次记作跨 Executor copy。

正式产物为 [domain_run2](domain_run2/summary.json)，保存 151 个输入/脚本来源哈希及 atomic CPU 原始测量来源；M1 完整重枚举，M0/M2 使用原冻结排名作对齐而非重新拟合。[第一次遍历](domain_run1/summary.json)和补齐 provenance 后的第二次遍历，其 4,623 条 topology 结果完全一致。5 项单元测试通过，包括带并列的 Pareto 对照暴力求解、重复 primitive 计数、非整齐对齐和不匹配 contract 拒绝。

## 结果

| 项目 | 结果 |
|---|---:|
| topology / 含线程配置 | 4,623 / 972,528 |
| 1 / 2 / 3 个 VTA island 的 topology | 87 / 990 / 3,546 |
| 全部路径累计 edge 出现次数 | 25,410 |
| contract 规格类 | 15 |
| 保守 edge-context key | 739 |
| 每帧单次交接规范下的边界 payload | 200,704–6,823,936 B |
| 理论全外部分配 K2 slot | 401,408–13,647,872 B |

739 是保留 producer/consumer segment 身份及 VTA context 的保守 key 数；CPU 融合上下文尚未编译去重，**不是 739 个已证明最小等价类，也不是只编译 15 个 contract 就足够**。物理可达性、实际对齐、storage-id 独占性和 cache/coherence 均保留未知，数值等价也不能由 contract 匹配推出。

每条边 `B_e` 是一次交接的 tensor payload；`sum B_e` 不是对具体 runner 已测的 copy API 字节。如果每边恰好一次完整 memcpy，CPU 逻辑读写为 `2 sum B_e`；若 get_output/set_input 各有一次拷贝，需按实际次数追加，不能把该式称为普遍上界或物理 DDR traffic。

所有本域 tensor 大小都满足 256 B 对齐，因此理论 K2 slot **恰好为 payload 两倍**。在固定 K2、全外部分配、不复用边间存储的模型下，两者不是独立排序信号；不推导 allocator high-water、实际峰值内存或容量瓶颈。

## 静态前沿与旧排名

以下 II 是既有 M1 成本代理，不是实测帧间隔，不能换算成新测 FPS。

| VTA 区间（其余 unit 为两端 CPU） | 最佳线程 M1 II 代理/ms | 边界 payload/B | 理论 K2 slot/B |
|---|---:|---:|---:|
| 03..17 | 70.8074 | 903,168 | 1,806,336 |
| 08..19 | 97.6509 | 501,760 | 1,003,520 |
| 13..19 | 136.7417 | 301,056 | 602,112 |
| 18..19 | 176.9719 | 200,704 | 401,408 |

完整散点见 [II–payload 图](domain_run2/ii_payload_pareto.png)。这是模型坐标下的 4 个非支配点，**不构成把 4,623 个切图安全剪成 4 个的证明**：预测误差、量化资格化、运行时共享路径和候选实测排名均未被该前沿覆盖。

[M0/M1/M2 对齐](domain_run2/legacy_model_alignment.json)覆盖各 20 个旧候选：M0 对应 1 种 topology，payload 为 3,512,320 B；M1/M2 各对应 4 种 topology，payload 范围 903,168–1,304,576 B。M1 的 20 个 candidate ID、顺序和分数重新计算后不变；本轮没有新的收益、排序改进或模型系数。

## 对 C1 的意义与下一步

已补上“切图变量→共享边界 contract→payload/slot/DMA 分账”的全域可复现实现，而非只有三个局部例子。这是内存感知搜索的输入层证据，还不是额外性能收益证据。

E6-R 下一步先在已有冻结 package 上核验 Graph storage、adapter 和绑定必要条件，再做风险分层覆盖。739 个保守 key 不能直接转成 739 次完整 build 的承诺；先编译 CPU 端点上下文去重，再报告真实类数与覆盖率，按原计划的预算/缩域规则执行。

同轮 [E3 数值 pilot](../compile_context_audit/E3_TAIL_QUANTIZATION_PILOT.md)发现独立量化切点的 int8 回绕差异。故 E3 数值政策/共同参考资格化先于把相邻切点作为等价性能对照；目前不修改 incumbent、不启用 tile 重调或安全剪枝。

复现（输出目录必须不存在）：

```sh
TVM_THREAD_POOL_SPIN_COUNT=0 MPLCONFIGDIR=/tmp/mpl /home/orange/miniconda3/envs/vta-resnet/bin/python vta/tutorials/frontend/audit_shared_edge_domain.py --output /tmp/vta_e6s_reproduction
TVM_THREAD_POOL_SPIN_COUNT=0 /home/orange/miniconda3/envs/vta-resnet/bin/python -m unittest discover -s vta/tutorials/frontend -p test_shared_edge_domain.py -v
```
