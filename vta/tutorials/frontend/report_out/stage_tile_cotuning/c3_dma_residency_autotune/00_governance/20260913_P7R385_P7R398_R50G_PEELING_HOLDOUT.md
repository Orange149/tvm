# P7R385--P7R398：R50G invalid-dominator peeling 前瞻留出

## 结论

R50G 在一个新的、与源模型一致的 ResNet50-v2 stride-2 1x1 层上完成了标签前冻结、算子资格、
按需完整图构建、真实 FPGA 正确性/计时和标签后完整 oracle。在线单点达到 oracle+2%，但没有命中
精确 oracle；本池没有 FPGA-invalid 前沿点，因此没有触发 peeling 展开。

## 被撤销的 R50F

- P7R385 将 `stage2_unit1_conv2` 声明为 CI128/CO128/56x56/3x3/stride2/pad1。
- P7R387 只完成本地 lowering/FSim，没有接触开发板或读取性能标签。
- P7R388 通过 `relay.testing.resnet` 的 InferType 确认真实层为 CI128/CO128/28x28/3x3/stride1/pad1；
  该几何与历史 W02 重合。
- P7R385--P7R387 保留为不可变失败证据，但不得计作 ResNet50 holdout。冻结器现会自动核对精确
  ResNet50 层名和 Relay 几何，不一致即拒绝。

## R50G 时间线

1. P7R389：冻结真实 `stage2_unit1_conv1`，CI256→CO128、56x56、1x1/stride2/pad0；完整域
   2880，解析谓词通过 435，固定选择 8 family/32 legacy 身份。
2. P7R390：在目标 lowering/FSim/full-graph/FPGA/latency 前冻结 invalid-dominator peeling 与
   fixed-front、bytes、calls、Random 控制。
3. P7R391：24 个生产身份中 5 个通过真实 lowering 和三 seed FSim：两个 original、两个 input、
   一个 weight barrier。
4. P7R392：算子 `(bytes,calls,barrier)` wave0 冻结为一个 input-stationary；四个被支配点和全部
   控制顺序同时冻结。
5. P7R393：误用 MXNet ResNet50 完整图导致 exact route 未命中；没有接触板端，无有效 manifest，
   只作为模型实现身份不一致的失败尝试。
6. P7R394：改用与推导同源的 `relay.testing.resnet`；exact workload 只对应 Graph node 48 的一个
   融合函数，route、Graph JSON 和参数合同通过。
7. P7R395：在线 wave0 通过 3-seed 正确性、profile 对账与 7 轮配对计时；因无 invalid 点，按冻结
   规则停止。
8. P7R396/P7R397：在线结果固定后才构建、测量余下四点；5/5 全部 FPGA-correct。
9. P7R398：冻结后分析，不再改变候选顺序或停止条件。

## 结果

| 指标 | 结果 |
|---|---:|
| 完整 FPGA-correct 池 | 5/5 |
| 在线派发 | 1/5 |
| 在线候选 paired ratio | 0.837109 |
| 完整池 oracle paired ratio | 0.831173 |
| 在线 regret | 0.714% |
| 在线是否进入 oracle+2% | 是，trial 1 |
| 在线是否找到 exact oracle | 否 |
| 候选 build/dispatch 节省 | 80% / 80% |
| stock+候选 model-build 阶段 | 50.494 s vs 160.815 s，-68.60% |
| common static/FSim | 1.748 s / 1.041 s，必须单列 |

冻结控制顺序中，bytes/calls 在 trial 1 进入 +2%、trial 2 找到 exact oracle；Random 在 trial 3 才
进入 +2% 并找到 exact oracle；fixed-front 与 peeling 均在 trial 1 进入 +2%，但其有限序列不包含
exact oracle。由于本池没有 invalid 候选，peeling 和 fixed-front 完全相同，不能把本结果写成
invalid-dominator fallback 的正向验证。

same-tile 机制方面，R50GF01/R50GF06 input-stationary 的最终 fused DMA bytes 分别下降
82.52%/82.26%，配对 latency 分别改善 4.44%/2.49%；R50GF01 weight barrier 的 DMA bytes 下降
3.36%、calls 下降 46.64%，但 latency 只改善 0.18%。因此静态 DMA 是候选资格/排序信息，不是
等比例性能模型。

## 口径限制与下一步

- R50G 完整图来自 `relay.testing.resnet` 的确定性随机参数，只证明调度等价与相对时延，不证明
  ImageNet accuracy，也不是预训练模型结果。
- P7R394 记录了本轮前瞻 stock+候选构建过程墙钟，但 P7R395/P7R397 旧执行器未把完整外层板端
  进程墙钟写入原始 summary。因此当前只能报告 model-build 阶段和板端资源计数，不能宣称完整
  `T_search` 已降低。执行器已补充该字段供下一轮使用。
- 下一未见池必须优先寻找自然出现的 FPGA-invalid 静态支配点，真正验证 peeling 是否能按冻结规则
  暴露后继；同时与 fixed-front、bytes、calls、Random 在同一完整外层预算下比较。
