# P7R266--P7R280：ResNet50 调用点剂量、融合 TIR 与成组准入

日期：2026-09-12  
开发板 boot：`aa7a3e5c-021d-4d6b-88ae-7f696faa567c`  
状态：`TWO_DEVELOPMENT_DOMAINS_SUPPORT_GROUP_FIRST_ADMISSION`

## 本轮回答的问题

此前已经能把一个已资格化驻留 schedule 放回完整 ResNet50，也能用 Graph JSON/DSO 组合只替换一个
调用点。本轮继续回答三个问题：

1. 多个相同 workload 调用点的共享内存变化是否可加？
2. 单点收益是否也能线性相加成整图 latency？
3. 部署搜索应逐点付费，还是利用静态可加性先测一组？

## R50B：DMA 线性，latency 非线性

P7R266 冻结节点 `[57,72,85,98]` 的 `k=0..4` 前缀剂量，P7R267 用四个独立 clean start 完成
24 次正确性和 56 次计时。每增加一个 weight-resident 节点，都严格产生：

```text
weight/total LOAD bytes  -393,216
weight/total LOAD calls  -432
synchronize/driver runs  +16
```

但 `k=1/2/3/4` 相对 `k=0` 的延迟中位差为 `-0.036/-1.428/-1.314/-0.791 ms`。P7R268--P7R269
再测八条极端上下文边，也得到“DMA 完全相同、latency 边际随图位置和上下文变化”的结果。因此
DMA 是访问身份与候选分组依据，不是 latency 的线性替代标签。

P7R270--P7R271 随后执行严格逐点 greedy：必须三 seed 正确、DMA 精确、7/7 获胜且中位差为负才
接受。四点分别只有 4/7、6/7、5/7、4/7，因此全拒绝；但 all-resident 在最终九轮对照中相对
all-original 快 `1.000060 ms`、9/9 获胜。逐点门在低信噪比下出现 false negative。

## R50A：孤立模板模型失败，融合 TIR 模型闭合

P7R272 把 R50A input-stationary 候选编入预训练 ResNet50，三个图节点为 `[67,80,93]`。P7R273 的
输出 6/6 正确，但“孤立算子静态差值×3”预测总 LOAD calls `-192`，板端为 `-204`，因此按合同在
计时前停止。

P7R274 捕获 original/input 两种模式各 68 个 `CPUAccessRewrite` 后的模块。P7R275 按最终 Graph
JSON 聚合 fused TIR，发现每个目标节点除 input/weight 请求变化外还减少 4 次 ACC LOAD；三点合计：

```text
total LOAD       -1,204,224 B / -204 calls
input LOAD       -1,204,224 B /  -96 calls
weight LOAD               0 B /  -96 calls
ACC LOAD                  0 B /  -12 calls
STORE                     0 B /    0 calls
```

十项字段与 P7R273 板端记录完全一致。P7R276 在该预测冻结后重新上板，correctness/timing profile
再次逐项精确命中，6 次正确性和 14 次计时输出相同非零；中位延迟 `337.272702→330.124981 ms`，
提升 `2.165%`、7/7 获胜。

## 两种准入策略

P7R277 将 R50A 三点展开为八个精确子集，P7R278 对同一个 7/7 严格门比较：

| 策略 | 结果 | 比较对数 | 整网执行数 | 关键现象 |
|---|---:|---:|---:|---|
| 逐点 greedy | `111` | 3 | 60 | 三个单点均通过 |
| 整组优先 | `111` | 1 | 20 | `-6.977 ms`、7/7，执行数 -66.67% |

P7R279 与 R50B 汇总后形成算法原型：

```text
按相同驻留语义、资源合同兼容、fused-TIR DMA 增量可加形成 bundle
  -> 相对当前 incumbent 测整个 bundle
  -> 正确性 + exact DMA + 配对 latency 门全部通过：接受整组
  -> 失败且组大小 > 1：二分并相对更新后的 incumbent 递归评估
```

R50A 表明它能在同一选择下减少验证成本；R50B 表明它还能避免低信噪比逐点门漏掉稳定组合。

P7R280 再用既有 YOLO 四变体做强制异构负向控制：整组 `Y00+Y02` 为 `+75.012 ms、0/8`，拆分
后 Y00 为 `-4.196 ms、8/8`，再相对更新后的 Y00 incumbent 测 Y02 为 `+79.285 ms、0/8`，最终
只保留 Y00，与既有 planner 一致。该路径需要 3 次 pair test，而逐点只需 2 次，清楚显示递归策略
不是在最坏情况下免费。

## 结论边界

- 两个域都已经暴露过相关标签，只能作为方法开发证据，不是新 holdout。
- R50B 最终 bundle 证据来自三方验证：实际共 36 次执行，其中目标 A/B 两模式相关 24 次；不能写成
  实际只做了 20 次。
- 两个自然 same-mode call-site bundle 都没有失败；P7R280 只有既有板端数据上的异构强制负控。
  它验证了递归决策语义并显示 50% pair-test 额外成本，但不能代替自然失败域。
- 节点身份仍来自冻结 Graph JSON，未原生进入 Relay/AutoTVM 搜索键。
- profiler 统计逻辑 VTA LOAD/STORE，不是物理 AXI burst；没有 ImageNet accuracy 或跨 boot 结论。

## 下一步

优先冻结一个未见调用点域，不读取 partial-mask latency：先依据 fused TIR 形成 bundle，执行 whole-
group gate；若失败则实际跑通正常分组规则内的递归拆分。主指标是 time-to-accepted-route、整网执行次数、逻辑 DMA、
最终 latency 与逐点 greedy 的选择差异。若找不到未见多调用点 workload，则至少在新 boot 上复测
R50B bundle，并把当前结果严格保留为开发消融。

## 关键产物

- `20260912_p7r267_resnet50_callsite_dose_board_run01`
- `20260912_p7r269_resnet50_callsite_marginal_board_run01`
- `20260912_p7r271_resnet50_callsite_greedy_board_run01`
- `20260912_p7r273_r50a_resnet50_generic_pair_board_run01`（预期失败，计时前停止）
- `20260912_p7r274_r50a_resnet50_fused_tir_audit_run01`
- `20260912_p7r275_r50a_fused_tir_occurrence_delta_run01`
- `20260912_p7r276_r50a_fused_tir_pair_board_run01`
- `20260912_p7r278_r50a_grouped_admission_board_run01`
- `20260912_p7r279_resnet50_grouped_admission_analysis_run01`
- `20260912_p7r280_hierarchical_bundle_negative_control_run01`
