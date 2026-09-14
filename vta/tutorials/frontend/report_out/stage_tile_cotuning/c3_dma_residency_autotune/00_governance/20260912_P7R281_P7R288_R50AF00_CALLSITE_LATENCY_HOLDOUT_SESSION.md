# P7R281--P7R288：R50AF00 调用点/完整图 latency holdout

日期：2026-09-12  
开发板 boot：`aa7a3e5c-021d-4d6b-88ae-7f696faa567c`

## 目标与证据边界

目标是在 P7R279 已冻结“DMA 可加性引导、整组优先、7/7 接受、失败递归拆分”的规则后，选择一个
新的 tile，在不读取其完整图或 partial-mask latency 的前提下，比较 singleton greedy 与 group-first
admission 的选择和验证成本。

R50AF00 是新的 tile，但不是新的模型或 workload。其算子级 correctness/performance 历史记录已经
存在，只是没有参与本次确定性选择；本次真正隔离的是完整图和八个调用点 mask 的 latency。因此
本文将它称为 **call-site/full-graph latency holdout**，不称独立 workload holdout。

## P7R281：确定性冻结候选

脚本按既有注册表顺序，排除已作为开发域的 F06 后，选择第一个 original/input 都有三 seed FPGA
correctness 记录的 family：

- family：`R50AF00`
- knobs：`tile_h=1, tile_w=7, tile_ci=8, tile_co=2`
- Graph JSON 节点：67、80、93
- original identity：`73aa97bd1989108f0d1fe089241ca12f3ffc07756c42f6d78cf2ca173afe969c`
- input identity：`ea09328abc3e1e14129d29da989026db9a714762ae6c39c81c3f0def741440d2`

合同显式记录 `operator_latency_used_for_selection=false`、
`full_graph_latency_unseen_before_contract=true` 和
`partial_mask_latency_unseen_before_contract=true`。

## P7R282--P7R285：构建、融合 TIR 预测与八子集冻结

P7R282 生成 original/input 两份预训练 ResNet50 DSO；两者 graph 和参数语义保持一致，18 次配置
查询、2 次 schedule 命中通过。P7R283 捕获两模式各 68 个最终 fused TIR 快照。P7R284 在任何板端
latency 前，按节点 67/80/93 聚合出预期差值：

| 范围 | LOAD bytes | LOAD calls | input calls | weight calls | ACC calls | store |
|---|---:|---:|---:|---:|---:|---:|
| 每个节点 | -1,204,224 | -3,024 | -1,344 | -1,344 | -336 | 0 |
| 三节点整组 | -3,612,672 | -9,072 | -4,032 | -4,032 | -1,008 | 0 |

权重和 ACC 的字节数不变，但请求次数减少；这进一步说明只看某一张量字节不足以描述最终融合程序。
P7R285 随后冻结三个调用点的全部八种 mask 及 wrapper、graph、参数、DSO 和合同哈希，且明确尚未
接触板端标签。

## P7R286：真实 FPGA 结果

实验用四个独立 clean start 分别执行三条 singleton edge 和一次 whole-group edge。每条 pair 包含
三 seed correctness 和七轮平衡计时，共 20 次完整模型执行；所有输出均非零且逐元素相等，所有
correctness/timing profile 的十项 LOAD/STORE 差均逐项命中 P7R284 预测。

| 决策 | 中位 latency 差 | paired wins | 结果 |
|---|---:|---:|---|
| node 67 singleton | -15.088 ms | 7/7 | 接受 |
| node 80 singleton | -16.099 ms | 7/7 | 接受 |
| node 93 singleton | -15.993 ms | 7/7 | 接受 |
| `000→111` whole group | -47.839 ms | 7/7 | 接受 |

整组中位延迟为 `422.292→374.452 ms`，提升 12.776%。singleton greedy 与 group-first 最终都选择
`111`，但前者需要 3 对、60 次完整模型执行，后者需要 1 对、20 次，减少 66.67%。

P7R286 复用了旧 runner，其 raw summary 中的 `claim_boundary` 是针对上一开发实验的硬编码文字。
原始产物保持不可变；P7R287 通过预注册哈希、状态、身份和结果复核，在独立摘要中校正本次解释。
P7R288 再把两个开发域与该留出分开汇总，状态为
`first_callsite_latency_holdout_confirms_group_first_cost_reduction`。

## 能主张什么

可以主张：在两个开发域固定 group-first 规则和 7/7 门后，一个确定性选择的新 tile 在未知完整图与
partial-mask latency 上，与 singleton greedy 得到相同 `111` 选择，并把真实 FPGA 完整模型验证执行
从 60 次降到 20 次；编译期 fused-TIR DMA 合同也被板端逻辑 profiler 精确确认。

不能主张：跨 workload/网络泛化、全局最优、最坏情况下永远节省、自然 same-mode 失败 bundle 已
完成递归拆分、ImageNet accuracy 或物理 AXI burst。三个图节点仍通过冻结 Graph JSON/DSO 组合，
尚未成为 Relay/AutoTVM 原生 call-site 搜索维度。

## 下一步

优先级依次为：

1. 冻结独立 workload/模型的多调用点 bundle，或在新 boot 做预注册复验；
2. 找到满足正常同驻留语义分组规则、且 whole group 自然失败的域，真实执行递归拆分；
3. 将稳定 call-site identity 前移到 Relay/AutoTVM 编译和搜索层。

## 关键产物

- `20260912_p7r281_r50a_callsite_latency_holdout_contract_run01`
- `20260912_p7r282_r50af00_resnet50_generic_pair_build_run01`
- `20260912_p7r283_r50af00_resnet50_fused_tir_audit_run01`
- `20260912_p7r284_r50af00_fused_tir_occurrence_contract_run01`
- `20260912_p7r285_r50af00_callsite_holdout_subsets_run01`
- `20260912_p7r286_r50af00_callsite_holdout_board_run01`
- `20260912_p7r287_r50af00_callsite_latency_holdout_audit_run01`
- `20260912_p7r288_grouped_admission_holdout_confirmation_run01`
