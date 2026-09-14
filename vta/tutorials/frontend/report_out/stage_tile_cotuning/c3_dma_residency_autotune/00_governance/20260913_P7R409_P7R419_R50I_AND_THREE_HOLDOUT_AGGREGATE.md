# P7R409--P7R424：R50I 在线留出、独立基线与三留出汇总

日期：2026-09-13  
状态：`COMPLETE`  
开发板启动：`4d232b6f-6e2a-4395-aafe-853a021403e8`

## 结论

R50I 是在目标完整图、FPGA 正确性和 latency 标签均不可见时冻结的第二个
source-consistent invalid-dominator-peeling 在线留出。在线搜索按既定规则派发初始两点前沿；两点
均通过三 seed FPGA 正确性，因此没有人为触发第二波，搜索在 213.527 s 后正常停止。事后补齐的
五点完整池为 4/5 FPGA-correct，在线第二点就是精确 oracle。

R50G/R50H/R50I 三个前瞻留出合并为 25 个 lowering/FSim 合法候选、23 个 FPGA-correct 候选；
在线实际派发 7 点，相对穷举候选数减少 72%，3/3 进入 pool-oracle+2%，2/3 命中 exact oracle。
三个留出分别覆盖“无错误近优停止”“错误支配点剥离后展开”“无错误且精确停止”。真正发生
invalid-dominator expansion 的仍只有 R50H，因此不主张已经重复两次自然错误剥离。

## 时间顺序与身份

1. P7R409 从 `relay.testing.resnet` 自动核验并冻结 `stage3_unit1_conv1`：CI512→CO256、输入
   28x28、输出 14x14、1x1/stride-2/pad0。完整 ConfigEntity 域为 1920 点，解析资格保留 328 点，
   固定哈希选出 8 个 family、24 个生产身份。该几何没有历史性能标签冲突。
2. P7R410 在目标标签不可见时冻结 R50H 已使用的 peeling 规则，不因 R50I 改阈值。
3. P7R411 真实 lowering 与三 seed FSim 只保留 5/24 点；失败身份全部保留，不补点。
4. P7R412 在任何目标完整图、FPGA 或 latency 标签前冻结两点 wave0、支配关系以及
   bytes/calls/Random/fixed-front 控制顺序。
5. P7R413 逐点构建最终图并上板。wave0 的 weight barrier 与 input-stationary 均通过，故搜索按
   冻结停止条件结束，无标签驱动扩展，完整外层进程墙钟为 213.526957 s。
6. P7R414--P7R415 在在线结果锁定后补构建并测量其余三点，以形成不可选择性遗漏的完整 pool
   oracle。补测中两点通过，一点 fail closed。
7. P7R418 只在全池完成后计算 oracle、控制 replay 和成本差；P7R419 再对 R50G/H/I 做汇总。

## R50I 结果

| 指标 | 在线搜索 | 完整池/穷举 | 相对变化 |
|---|---:|---:|---:|
| candidate dispatch | 2 | 5 | -60.00% |
| complete candidate build action | 50.794 s | 138.805 s | -63.41% |
| FPGA kernel invocation | 40 | 82 | -51.22% |
| logical DMA bytes | 6,546,313,216 | 13,700,435,968 | -52.22% |
| logical DMA calls | 4,105,364 | 8,549,048 | -51.98% |

完整正确池 oracle 是 R50IF07 input-stationary，候选中位时延 681.139921 ms，paired ratio
0.895434509。在线在第 2 次派发命中 exact oracle。冻结控制命中 exact 所需派发为：calls-lazy 1、
online/fixed-front 2、bytes-lazy 3、Random 5。因此这一池不能用于宣称方法全面优于所有简单规则；
它证明在线停止行为、精确质量保留及相对穷举的实测成本下降。

## 机制与正确性边界

- R50IF05 original/input 构成可解释 same-tile 对。input-stationary 的逻辑 DMA bytes 仅下降
  1.14%，calls 下降 72.79%，完整图 latency 改善 7.49%，说明请求碎片/固定请求开销不可从总字节
  指标中删除。
- R50IF07 original 在 FPGA 上错误，而同 family 的 input 与 weight 模式通过。因 original 不具备
  正确 latency 标签，不计算或宣称该 family 的纯驻留机制加速。
- 全部运行时计数是逻辑 VTA LOAD/STORE 与 kernel invocation，不是物理 AXI burst 或总线计数。

## 三留出综合解释

| 留出 | 完整池 | FPGA-correct | 在线派发 | exact | oracle+2% | 实际展开 |
|---|---:|---:|---:|---:|---:|---:|
| R50G | 5 | 5 | 1 | 否，regret 0.714% | 是 | 否 |
| R50H | 15 | 14 | 4 | 是 | 是 | 是 |
| R50I | 5 | 4 | 2 | 是 | 是 | 否 |
| 合计/计数 | 25 | 23 | 7 | 2/3 | 3/3 | 1/3 |

正式论文应把贡献写成：共享内存语义在昂贵 final-graph/FPGA 阶段之前形成多目标候选前沿；真实
FPGA 正确性不是预测标签，而是 fail-closed 的在线反馈，只有错误支配点被移除后才逐层暴露新候选。
这套动作能减少达到近优正确程序所需的昂贵试验，并在 R50H 中恢复静态错误支配点遮蔽的 oracle。

不能写成：Pareto 前沿保证 exact oracle、peeling 总是优于 bytes/calls、已经在两个自然错误支配
workload 上重复 expansion、三个 workload 等于跨模型泛化、随机 Relay 参数证明 ImageNet accuracy，
或逻辑 DMA 等价于物理 AXI 流量。R50G 缺少序列化的完整外层墙钟，因此 P7R419 只汇总 R50H/R50I
可用的 526.469 s，不拿它与三池穷举完整墙钟做虚假比较。

## P7R420--P7R424：同预算独立在线基线

P7R412 已在 R50I 标签前冻结 calls/bytes/Random 顺序。P7R420--P7R422 在完整池完成后才执行，
因此不增加新的前瞻 workload 计数，但三条基线都由独立进程重新做模型准备、stock build、逐候选
final-graph build、clean-start FPGA correctness 和计时，而不是从一次全池测量中截取成本。每条策略
都固定派发前两点，执行期间不读取 oracle；P7R424 仅在运行结束后按不可变 candidate identity 接入
P7R418 的完整池质量标签。

| 策略 | 派发 | 正确/错误 | 完整外层墙钟 | 首次 exact/oracle+2% | 两点预算质量 |
|---|---:|---:|---:|---:|---|
| peeling | 2 | 2/0 | 213.527 s | 213.311 s | exact |
| calls-lazy | 2 | 2/0 | 212.812 s | 146.179 s | exact |
| bytes-lazy | 2 | 2/0 | 214.571 s | 未命中 | 超出 +2% |
| frozen Random | 2 | 1/1 | 199.165 s | 未命中 | 超出 +2% |

这项结果关闭了“R50I 简单控制只有 replay、没有完整进程墙钟”的局部缺口，也直接否定了任何
“peeling 在每个池都最好”的叙述：calls-lazy 在该池第一次派发就命中 oracle，比 peeling 早
67.133 s。另一方面，bytes-lazy 与 Random 在相同两候选预算内均未进入 2% 带；Random 的总墙钟
更短只是因为第二点正确性失败后没有执行 14 次 timing invocation，不能解释为同等质量下更快。
因此主张仍应是多目标前沿和真实错误反馈提供可靠的预算—质量折中，而不是单一排序规则普遍占优。

## 证据入口

- `07_grouped_holdout/20260913_p7r409_r50i_stride2_candidate_contract_run01`
- `07_grouped_holdout/20260913_p7r410_r50i_invalid_dominator_peeling_policy_run01`
- `07_grouped_holdout/20260913_p7r411_r50i_full_local_qualification_run01`
- `07_grouped_holdout/20260913_p7r412_r50i_peeling_wave0_contract_run01`
- `07_grouped_holdout/20260913_p7r413_r50i_invalid_dominator_peeling_online_run01`
- `07_grouped_holdout/20260913_p7r414_r50i_oracle_completion_build_run01`
- `07_grouped_holdout/20260913_p7r415_r50i_oracle_completion_board_run01`
- `07_grouped_holdout/20260913_p7r418_r50i_invalid_dominator_peeling_analysis_run01`
- `07_grouped_holdout/20260913_p7r419_three_peeling_holdout_aggregate_run01`
- `07_grouped_holdout/20260913_p7r420_r50i_calls_lazy_budget2_online_run01`
- `07_grouped_holdout/20260913_p7r421_r50i_bytes_lazy_budget2_online_run01`
- `07_grouped_holdout/20260913_p7r422_r50i_random_lazy_budget2_online_run01`
- `07_grouped_holdout/20260913_p7r424_r50i_independent_online_controls_final_run01`

P7R423 是 P7R424 的较早汇总版本；原始文件保留用于审计，但正式引用 P7R424。
