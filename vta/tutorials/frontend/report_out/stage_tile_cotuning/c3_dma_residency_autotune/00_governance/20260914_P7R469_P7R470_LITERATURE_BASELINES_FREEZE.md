# P7R469--P7R470：文献对齐基线实现与 ResNet18 候选冻结

状态：`NODE_B_COMPLETE; LOCAL_ONLY; PAUSED_BEFORE_NODE_C`

## 1. 完成内容

新增统一实验 runner：

```text
random
stock_mode_aware_xgb
rieber_hw_init_xgb
ml2tuner_pva
cheng_minimum_access
ours_dma_multifidelity
```

统一命令接口已经提供 `--policy`、`--workload-contract`、`--candidate-budget`、`--seed`、`--phase`
和 `--output-dir`；每次有效运行生成不可覆盖的 `contract.json`、`timeline.jsonl`、`results.jsonl`、
`summary.json` 和 `artifact_hashes.json`。目标结果由单向 outcome ledger 在付费动作后揭示，候选合同
若包含 latency、validity、correctness 或 oracle 字段会直接拒绝。

### HW-Aware Initialization

已实现四级消融接口：random initialization、neighbor presampling、balanced E0、balanced E0 +
validity bias。E0 固定最多 25 valid + 25 invalid，presampling 为
`min(1000, |S_w|)`，20 个 seed 冻结为 57001--57020；初始化后统一接 mode-aware XGB。邻域扩展只在
当前 compiler batch 返回后使用 validity，不能预读未来合法性。

### ML²Tuner

已分离 Model P、Model V 和 Model A：P/V 仅使用 workload、tile 和 mode 等可见特征；A 在真实
lowering 后增加 loop、extent、branch、partial tile、allocation 和 tensorize 特征。固定
`N=10, alpha=1`，P/V 选择 20 个编译候选，A 选择最多 10 个进入 FPGA 层；另有 `--ml2-include-dma`
消融。编译、FSim、设备或数值错误候选只进入 validity 数据，不能进入 P/A 性能回归。

### Cheng 四方案

正式区分 original、input-stationary、weight-resident-barrier 和
`input_weight_resident_barrier`。新增组合模式采用输入优先循环顺序，并在整层权重能放入 SRAM 时
把完整 kernel 放到 batch 作用域、末尾显式同步；这是当前 TVM/VTA 上的功能级独立重实现，不是
作者未公开的“不覆盖地址 + 条件 LOAD WGT + PushGEMMOp”源码复现。

三个历史代表配置的组合模式均同时达到 input-stationary 的 input bytes 和 weight-resident 的
weight bytes，并通过 3 seeds FSim 逐元素等价；整层权重超过 SRAM 时抛出 `not_applicable`，实验
策略记录并回退同 family original。

## 2. 冻结的 ResNet18 候选

原计划假定每个几何的完整 original ConfigSpace 都是 1280 点。真实枚举否定了该假设：ConfigSpace
基数随轴 extent 的因子分解变化，不能为了对齐计划人为截断。冻结结果是：

| workload | 真实完整 original 空间 | max-min tile | 四模式身份 | Scheme 4 解析适用 |
|---|---:|---:|---:|---:|
| R18-H1 | 2304 | 24 | 96 | 24/24 |
| R18-H2 | 1600 | 24 | 96 | 0/24 |
| R18-H3 | 480 | 24 | 96 | 24/24 |

R18-H2 的整层权重为 589,824 B，超过当前 262,144 B weight SRAM，因此 weight reuse/组合方案按
Cheng 的资源原则记为 `not_applicable`，后续必须回退 original，不能把本地 barrier 能分块执行
伪装成论文式整层权重复用。

每个几何同时保存完整 original 空间供 HW-Aware 扫描，以及 24 个标签不可见 max-min tile 展开的
96 个四模式身份。此时 `lowered_tir_sha256` 明确为空并标为 pending；节点 C 只有在真实 lowering
成功后才能生成绑定 TIR 哈希的 implementation identity。失败身份仍保留 proposal identity 并计入
gross，不会伪造 TIR 哈希。

## 3. 冻结产物

- P7R469：`07_grouped_holdout/20260914_p7r469_literature_baseline_protocol_run01/`
- P7R470：`07_grouped_holdout/20260914_p7r470_resnet18_literature_candidates_run01/`

两份 manifest 已逐文件复核。P7R469 冻结 6 个策略、20 个 seed、三个 workload、预算、W0/T0/T1
边界、失败计费和论文复现等级；P7R470 冻结实际 ConfigSpace、max-min 顺序、288 个四模式 proposal
identity 与源码/硬件哈希。二者均记录 `board_contacted=false`、`performance_labels_read=false`。

## 4. 测试与边界

使用恢复后的本地 FSim 配置运行相关回归：`62 passed`。覆盖：未来标签泄漏、E0 平衡与邻域反馈、
P/V/A 数据隔离、invalid 不进入性能模型、四模式与 fallback、失败计入 gross/墙钟、RPC/断电 session
不可拼接、oracle 仅在完整池结束后连接、输入证据不可覆盖，以及 Scheme 4 三 seed 数值正确性。

本节点没有执行 R18-H1/H2/H3 的 288 身份 lowering/FSim，没有交叉编译、没有接触 FPGA、没有产生
算子或整图 latency。因此现在只能主张“基线、组合机制和候选协议已实现并冻结”，不能主张 HW-Aware
提高 valid yield、ML²Tuner 减少 profiling、Cheng 方案在新几何加速，或本文优于这些方法。

节点 C 的下一动作是严格消费 P7R470：先对 288 个身份做静态检查和真实 lowering，提取 hidden/DMA/
command 特征，再做三 seed FSim；若任一几何存活少于 12，只能按冻结 max-min 顺序扩展，且不得读取
FPGA latency。按计划，本节点完成后暂停，不自动开始该阶段。
