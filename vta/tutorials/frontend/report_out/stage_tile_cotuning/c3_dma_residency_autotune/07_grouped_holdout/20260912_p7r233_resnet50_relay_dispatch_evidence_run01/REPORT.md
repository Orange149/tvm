# P7R227--P7R232：驻留候选进入 Relay 与 ResNet50 的证据

状态：`PASS_WITH_EXPLICIT_BOUNDARIES`

## 结论

此前的 `conv2d_packed_residency.vta` 只用于独立算子候选，普通 Relay build 不会仅凭 AutoTVM
历史记录选择另一套 residency schedule。P7R227 新增 exact-workload、complete-ConfigEntity 绑定的
`ExplicitResidencyDispatch`，并在注册的 `conv2d_packed.vta` schedule 入口选择 original 或
`weight_resident_barrier`；未命中的 workload 继续委托外围 TopHub context，配置不一致则 fail closed。

该路径先在最小 packed Relay 图上验证，再放入完整 ResNet50-v2 图。两级结果都说明：驻留机制的
DMA 与 latency 收益没有在 Relay/graph-executor 集成时消失。

## 最小 Relay 算子（P7R227/P7R228）

- 原始与驻留构建使用同一 workload 和 complete ConfigEntity；graph 与 params 哈希相同，生成的
  `graphlib.so` 哈希不同；CPUAccessRewrite 后 TIR 哈希不同，`coproc_sync` 为 1 对 2。
- clean-start 板端三 seed 共 6/6 正确，7 轮交错计时共 14/14 正确，驻留模式 7/7 更快。
- 中位 latency：25.674497 -> 25.083010 ms，提升 2.3581%。
- 按单次推理归一化，weight LOAD 为 458,752 -> 65,536 B（-85.71%），weight LOAD calls 为
  448 -> 16（-96.43%）；input/store 不变。
- 总逻辑 DMA 为 2,465,792 -> 2,072,576 B（-15.95%），总 DMA calls 为 1,008 -> 576
  （-42.86%）；submission/synchronize 为 1 -> 17。该结构与 P7R225 独立函数结果一致。

## 预训练 ResNet50 整图（P7R231/P7R232）

- 使用官方 `resnet50_v2` 预训练参数，原始/驻留两次 build 的 graph 哈希完全相同；107 个 lowered
  参数逐名、逐 shape/dtype、逐字节一致，语义哈希均为
  `778eb4136dbd80dbba51a67f065c9dfe5a18988fae17584ffcfe790d5ff7e76a`。
  `save_param_dict` 文件哈希因字典序列化顺序不同而不同，不能误判为参数内容变化。
- exact dispatch 在整图中命中一次已编译 schedule；相同 workload 被四个算子实例复用。因此本
  A/B 是“同一 exact workload 的四个实例统一替换”，不是单层替换。
- clean-start 板端三输入 seed 共 6 次执行，原始/驻留的 1,000 个非零 logits 均逐元素相同；
  7 轮交错计时共 14 次调用也全部配对相同。
- 中位整网 `run` latency：406.700897 -> 405.361804 ms，提升 0.3303%，驻留版 7/7 轮更快。
- 按单次推理归一化，整网 weight LOAD 减少 1,572,864 B，weight LOAD calls 减少 1,728，
  submission/synchronize 增加 64。三项差值分别是最小算子单次差值的四倍，和四个 workload
  实例严格对应，没有观察到 input LOAD 或 store 差值。
- 整网总逻辑 DMA 为 85,323,264 -> 83,750,400 B（-1.84%）；整网总 DMA calls 为
  13,996 -> 12,268（-12.35%）。单算子收益被整网其他 48 次 VTA submission 和 CPU 尾部摊薄，
  因而整网只有 0.33% 是合理结果，不应包装成显著 FPS 提升。

## 证据边界

1. P7R232 完成的是完整 ResNet50 graph 的 schedule 集成、配对正确性和端到端 `run` 时间，不是
   ImageNet accuracy 测试；输入为三组确定性随机张量。
2. exact workload 可能对应多个同形算子。若论文需要逐层不同策略，manifest 还必须加入 Relay
   call-site/层 ID，而不能只用 AutoTVM workload key。
3. 该结果复用了已在完整 R50B 候选池中资格化的 ConfigEntity，没有重新调参，也没有使用整网
   latency 反向选择候选。
4. runtime 数据是逻辑 VTA LOAD/STORE 和 submission 计数，不冒充物理 AXI burst/带宽计数。

## 对 E5 的影响

E5 可由“加分项未完成”更新为“ResNet50 整图首次传递完成”。这消除了“驻留候选只能在独立测试
模板运行”的主要工程缺口，但不会改变核心创新仍是按需付费的共享内存感知多保真搜索；0.33% 的
整网收益只作为机制能够落地的外部有效性证据。

## 绑定产物

- P7R227 最小 Relay 交叉构建：`../20260912_p7r227_r50b_relay_residency_dispatch_cross_build_run02`
- P7R228 最小 Relay 板端：`../20260912_p7r228_r50b_relay_residency_dispatch_board_run01`
- P7R231 预训练 ResNet50 交叉构建：`../20260912_p7r231_resnet50_pretrained_relay_residency_dispatch_cross_build_run01`
- P7R232 预训练 ResNet50 板端：`../20260912_p7r232_resnet50_pretrained_relay_residency_dispatch_board_run01`

