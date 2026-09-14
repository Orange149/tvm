# P7R399--P7R408：R50H 在线 invalid-dominator peeling 前瞻留出

## 结论

R50H 是第一次在目标标签完全隔离的条件下真实触发 invalid-dominator peeling 的留出实验。初始
三点算子代理 Pareto 前沿中，最低 DMA 的 weight-resident 候选通过真实 lowering 和三 seed FSim，
却在完整 ResNet50-v2 图的 FPGA 第二 seed 上产生 1000/1000 logits 不一致。在线搜索器按预注册
规则只删除这个错误支配点，不读取 latency，重新计算后暴露唯一第二波候选；该候选随后通过板端
认证，并被事后完整 15 点池确认是精确 oracle。

这关闭了 R50D 事后发现、R50G 未能触发的算法缺口：FPGA correctness 不再只是搜索末端的被动
检查，而是能改变下一候选集合的在线反馈保真度。

## 冻结对象与时间顺序

- P7R399：从 `relay.testing.resnet.get_workload(num_layers=50)` 自动提取并核验
  `stage4_unit1_conv1`，几何为 CI1024→CO512、14x14、1x1/stride-2/pad0；672 个 ConfigEntity
  中解析谓词保留 125 个，固定哈希冻结 8 family × 3 生产模式。
- P7R400：在目标 lowering、FSim、完整图、FPGA 与 latency 标签前冻结三轴支配、错误点剥离、
  600 s 外层预算及 bytes/calls/Random/fixed-front 控制规则。
- P7R401：24 个身份中 15 个通过真实 lowering 和三 seed FSim，构成五个完整三模式 family。
- P7R402：冻结 15 点的全部支配关系、三点 wave0 与四种控制顺序；没有目标完整图或板端标签。
- P7R403：一次真实在线进程完成模型准备、stock 构建、逐候选完整图构建、clean-start FPGA
  correctness/计时、失败剥离和第二波选择。
- P7R404/P7R406：在线结果封存后才构建、上板其余 11 点，仅用于完整 oracle 审计。
- P7R405：因主机命令漏传空密码 askpass，在第一次 SSH preflight 前 exit 255；无板端候选标签，
  保留为基础设施失败记录。
- P7R407：首版分析的 `terminal_best` 展示字段实现有误；原始实验数据未变。
- P7R408：修正展示字段后重新封存的最终分析。

## 在线波次

| 波次 | 候选 | 代理语义 | FPGA 结果 | 作用 |
|---|---|---|---|---|
| wave0 | R50HF05 weight | 最低 bytes 的支配点 | 第二 seed 输出不一致，fail closed | 删除错误支配者 |
| wave0 | R50HF04 input | 低 calls 前沿点 | 通过，paired ratio 0.774094 | 第 2 次已进 oracle+0.1% |
| wave0 | R50HF04 weight | bytes/calls/barrier 冲突前沿点 | 通过，paired ratio 0.777932 | 保留多目标覆盖 |
| wave1 | R50HF01 weight | 只被错误 R50HF05 weight 支配 | 通过，paired ratio 0.774019 | 完整池 exact oracle |

在线搜索共派发 4 个候选，外层进程墙钟 312.9417 s。三个正确候选均完成三 seed correctness 和
7 轮交错计时；错误候选在第二 seed 首错即停。实验结束后 boot ID 始终为
`4d232b6f-6e2a-4395-aafe-853a021403e8`，FPGA 为 `operating`，u-dma-buf 为 201,326,592 B，
RPC 位于 RAM runtime，未见新增存储错误。

## 完整池与搜索成本

P7R404 新构建在线未触及的 11 个候选，P7R406 的 11/11 三 seed correctness 和 154 次计时调用
全部通过。与在线结果合并后，15 个 FSim-pass 候选中 14 个 FPGA-correct、1 个 FPGA-invalid；
oracle 为 R50HF01 weight-resident，候选中位 588.975369 ms，paired ratio 0.774018869。

在线 4 点相对完整 15 点的实测组件：

- candidate dispatch：减少 73.33%；
- complete candidate build action：减少 76.12%；
- FPGA kernel invocation：减少 77.46%；
- 逻辑 LOAD+STORE DMA bytes：减少 78.43%；
- 逻辑 LOAD+STORE calls：减少 78.12%；
- 仅调用阶段 host wall：减少 78.14%。

这些比例不含共同的算子 lowering/FSim 资格成本。只有 P7R403 的 312.9417 s 可称直接测得的完整
在线外层墙钟；控制顺序没有各自重新运行独立在线进程，不能把组件回放包装成完整墙钟 A/B。

## 冻结控制与负面结果

| 顺序 | oracle+2% | oracle+0.1% | exact oracle |
|---|---:|---:|---:|
| online peeling | 2 | 2 | 4 |
| bytes-lazy | 2 | 2 | 2 |
| calls-lazy | 1 | 1 | 10 |
| Random（单冻结顺序） | 1 | 7 | 13 |
| fixed wave0 | 2 | 2 | 未命中 |

因此不能写“peeling 全面优于 bytes-only”。本池中，bytes-lazy 恰好在错误最低字节点之后立即遇到
oracle，只需 2 次；peeling 为了完整测完三点 Pareto wave0 并按正确性更新，需 4 次。它相对
fixed-front 的优势是恢复 exact oracle，相对 calls/Random 的优势是更早 exact，但不是单轴搜索的
普遍支配者。真正可主张的方法贡献是：当廉价代理的支配关系被真实 FPGA 错误破坏时，搜索状态能
fail-closed 更新并继续，而不是永久漏掉被遮蔽候选。

## 机制观察

五个 same-tile input-stationary 候选均 FPGA-correct，整图相对 original 改善 0.17%--6.29%；四个
正确 weight-resident 候选改善 0.65%--5.34%。R50HF05 weight 虽将算子 DMA bytes 降低 80.75%，
却数值错误，说明低访存收益与硬件正确性必须是两个独立门，不能用前者替代后者。

## 论文口径

可主张：第一次严格前瞻地跑通“廉价算子代理前沿—按需完整图构建—多 seed FPGA correctness—
错误支配点删除—下一 Pareto 层—精确 oracle”的在线闭环，并直接记录完整在线外层墙钟。

不可主张：跨 workload 泛化、全面优于 bytes/calls、Pareto 必保 oracle、物理 AXI 计数、ImageNet
accuracy、TopHub 优势。R50H 使用确定性随机参数，只验证完整图调度等价、共享内存运行语义和相对
时延。下一增强实验应是第二个自然 invalid-dominator 的独立在线复验，或让冻结控制也独立运行以
获得完整端到端墙钟分布。
