# P7R516E--P7R516G：ResNet18 整图正确性回退与性能闭环

状态：`LABEL_FREE_FALLBACK_FROZEN; THREE_INPUT_FULLGRAPH_CORRECTNESS_PASS; SEVEN_ROUND_TIMING_COMPLETE`

## 1. 为什么需要修订

P7R516/P7R516D 已证明 H3 D0098 `input_weight_resident_barrier` 虽然在孤立算子中通过三 seed，
但放入完整 ResNet18 后，两个包含它的程序均为 0/6 正确，并且每次有 1000 个分类输出不一致。
旧 P7R516 在任何计时之前 fail closed，不能继续使用该 route，也不能从失败 session 拼接计时。

P7R516E 因此只读取已有的正确性证据，不读取任何 selected-fullgraph latency，在标签不可见条件下
冻结以下安全修订：拒绝精确 H3 combined 候选
`05a48341e4db17c245227e5bedbb1b2897b3f0317e8d5c4aad6f1fbf2fe25eb8`，统一回退到既有候选池中
已经通过孤立算子 3/3 和整图诊断 6/6 的 H3 D0203 original
`b26df2d47b578e1b48224fef680644c7cca98dce286f2b5d2dd6608c13dd7d87`。D0098 family 在冻结池中
没有 same-tile original，故不能事后创造一个未预注册的同 tile 回退点。

## 2. 回退后的策略与构建

| 策略 | 回退后程序 | H1 | H2 | H3 |
|---|---|---|---|---|
| Cheng minimum-access | `selected_e53bbdbb71050431` | D1213 original | D0575 original | D0203 original |
| ML²Tuner P/V/A | `selected_e53bbdbb71050431` | D1213 original | D0575 original | D0203 original |
| 本文 DMA 多保真 | `selected_e53bbdbb71050431` | D1213 original | D0575 original | D0203 original |
| stock mode-aware XGB | `fallback_8ff96c46487a92f8` | D0637 original | D0575 original | D0203 original |

P7R516F 将四个策略去重为两份程序。Cheng 的程序与 P7R515 精确同签名，因此连同 stock reference
直接复用并逐文件验 hash；只新构建 stock-XGB 回退程序，新构建墙钟为 11.722 s。冻结与构建摘要
均明确记录 `fullgraph_performance_labels_read=false`。这不是根据性能结果换配置，而是正确性门拒绝
不可部署 route 后的 fail-closed fallback。

## 3. P7R516G 整图结果

P7R516G 在 boot `bf5bc78c-115b-40e0-8853-6684f04ecf3d` 上重新加载冻结 bitstream，并启动新的
默认 RPC。stock reference 加两份唯一策略程序共三份图，在同一不可拼接 session 中完成：

- 三个确定性输入、全部图输出的 9/9 次正确性调用；
- 七轮平衡交错的 21/21 次计时调用；
- 所有输出逐元素与 stock 完全相等且非全零；
- W0→T0 7.959 s，T0→T1 12.919 s，W0→T1 20.878 s；
- 运行期累计 1,020 次 FPGA kernel invocation、996,033,536 B 逻辑 LOAD、95,961,600 B 逻辑
  STORE 和 116,620 次逻辑 DMA 调用。

| 程序/策略 | 中位整图延迟 | FPS | 相对 stock 延迟 |
|---|---:|---:|---:|
| stock reference | 116.464 ms | 8.586 | 基线 |
| stock-XGB fallback | 117.093 ms | 8.540 | +0.540% |
| Cheng / ML²Tuner / 本文共同 fallback | 117.722 ms | 8.495 | +1.080% |

stock-XGB 的七轮 IQR 为 117.000--117.223 ms，共同 fallback 为 117.098--117.830 ms，stock 为
116.220--116.721 ms。四个策略最终都进入“相对 stock 不劣于 2%”带，但没有一个比 stock 更快。
ML²Tuner、Cheng 与本文在正确性回退后是同一二进制，不能把同一组数值当作三次独立实验，也不能
比较三者的整图性能高低。

## 4. 成本与结论边界

如果只看修订后的可部署闭环，新增可测构建为 11.722 s，板端 W0→T1 为 20.878 s。若审计从最初
selected build 开始且保留失败代价，已计时阶段的最低小计为：P7R515 构建 46.541 s、P7R516 失败
14.441 s、P7R516D 诊断 18.699 s、P7R516F 新构建 11.722 s、P7R516G 成功 20.878 s，合计约
112.281 s；未计时的选择/冻结/人工分析不能填成零。这说明整图正确性失败本身就是部署成本，不能
只报告最后一次成功运行。

该结果支持两条结论：

1. 最终图上下文正确性门确实能把孤立算子正确但整图错误的驻留 route 拒绝，并在不看整图性能的
   条件下恢复到可部署程序；
2. 算子级 DMA/latency 收益不保证整图加速。安全回退后的本文程序比 stock 慢 1.080%，只能主张
   满足预注册的 2% 非劣门槛，不能主张提高 ResNet18 FPS。

正确性仅覆盖三个确定性随机输入与全部图输出等价，不是 ImageNet accuracy；运行期计数是逻辑 VTA
LOAD/STORE，不是物理 AXI burst 或带宽。

## 5. 审计信息

- P7R516E ledger：`c5e75f6bb30a5919ed42cbc64dc41d9c4b1ca06e73563c22bde3fe45a3c7aee1`
- P7R516F ledger：`153d405888b3c81e7eaaaacd2cf9758b969512540c7d3a9818b8ac30718ee913`
- P7R516G ledger：`16e4a594b298c5fcf2d720b20a9abcc4471bacf24e2c58404c31cef1b0445e96`
- 三份 artifact ledger 已完成逐文件校验。
- P7R516G 后仍为同一 boot，FPGA `operating`，u-dma-buf 201,326,592 B，默认 RPC cwd 与四项
  runtime hash 正确，未观察到新的 EXT4/MMC 错误。

本节点至此暂停。P7R517--P7R522 的 H1 三对三 clean-start 属于下一节点，不能把 P7R516G 的
fullgraph 标签反馈进其候选顺序或模型。
