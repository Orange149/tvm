# P7R289--P7R296：R50AF01 留出与完整图 DMA 碎片化边界

日期：2026-09-12  
开发板 boot：`aa7a3e5c-021d-4d6b-88ae-7f696faa567c`

## 目标

在 P7R288 完成首个 R50AF00 调用点/完整图 latency holdout 后，保持候选选择规则、三 seed
correctness、最终 fused-TIR 十字段 DMA 合同和 7/7 性能门不变，确定性选择下一个正确性合格 tile。
目的有两个：继续检查 group-first 的选择与成本，并用不同 tile 解释 bytes、calls、请求粒度与 latency
之间的关系。

该实验没有读取目标完整图或 partial-mask latency，但仍使用历史算子 correctness，并与 F00 共享
ResNet50、R50A workload 和 boot。因此它是第二个新 tile latency holdout，不是独立 workload。

## P7R289--P7R293：冻结链

P7R289 排除已使用的 F06/F00 后，按原注册表顺序选择 `R50AF01`：

- original：`e6473088a7dfdba7328a50f0cd0735819794144b1d7a666d7513ae40cd74d906`
- input-stationary：`5dd7a442cff1fb6565284948c9996c64c86410d19ec8508e9a59ab4e97eccc1e`
- knobs：`tile_h=14,tile_w=1,tile_ci=1,tile_co=1`
- 图节点：67、80、93

P7R290 完成两份预训练 ResNet50 交叉构建，graph/params 相同且目标 schedule 命中。P7R291 捕获两种
模式各 68 个最终 fused TIR 模块。P7R292 在上板前冻结每个节点和整个 bundle 的十字段预测：

| 范围 | LOAD bytes | LOAD calls | input calls | weight calls | ACC calls |
|---|---:|---:|---:|---:|---:|
| 每节点 | -2,809,856 | -25,480 | -12,544 | -12,544 | -392 |
| 三节点 | -8,429,568 | -76,440 | -37,632 | -37,632 | -1,176 |

weight/ACC LOAD 字节不变但 calls 下降，说明 loop/tile 改变了相同字节被拆分成多少请求。P7R293
冻结全部八个精确 Graph JSON mask 和 wrapper/DSO/参数哈希。

P7R293 的 `label_exposure` 误沿用 F00 名称，P7R294 raw summary 也沿用旧 development claim boundary；
两份原始证据不修改。P7R295 根据 P7R289/P7R292/P7R293 的上板前内容哈希校正权威解释，代码中的
通用文案也已修复供后续运行使用。

## P7R294：真实 FPGA

run01 因本机未导出 `SSH_ASKPASS`，在读取初始板状态时返回 SSH 255；没有 FPGA reload、RPC stop 或
候选调用，不计作候选失败。run02 显式绑定密码助手后完成四个独立 clean start：

| 决策 | latency 中位差 | paired wins | 结果 |
|---|---:|---:|---|
| node 67 singleton | -153.480 ms | 7/7 | 接受 |
| node 80 singleton | -154.790 ms | 7/7 | 接受 |
| node 93 singleton | -154.191 ms | 7/7 | 接受 |
| `000→111` whole group | -463.302 ms | 7/7 | 接受 |

whole group 为 `898.821→435.519 ms`，latency 降低 51.546%，等价吞吐提升 106.379%。全部 24 次
correctness 和 56 次 timing 的输出非零且 A/B 相等，板端十字段 profile 与预注册 fused TIR 逐项
相同。singleton greedy 和 group-first 都选择 `111`；前者需要 60 次完整模型执行，后者需要 20 次，
验证成本减少 66.67%。whole group 通过，因此仍未触发递归拆分。

## P7R296：F00/F01 受控 tile—共享内存比较

两个 tile 作用于同一 workload、三个相同图节点、同一模式和同一 boot：

| 指标 | R50AF00 | R50AF01 |
|---|---:|---:|
| original LOAD bytes | 27,009,024 | 20,729,856 |
| original LOAD calls | 12,096 | 87,360 |
| original latency | 422.292 ms | 898.821 ms |
| residency LOAD bytes | 23,396,352 | 12,300,288 |
| residency LOAD calls | 3,024 | 10,920 |
| residency latency | 374.452 ms | 435.519 ms |
| input reload ratio | 4 | 8 |
| input residency latency reduction | 11.329% | 51.546% |

F01 original 比 F00 少搬 23.25% LOAD bytes，却多发 622.22% calls，延迟高 112.84%；驻留后 F01
仍少搬 47.43% bytes，却多发 261.11% calls，延迟高 16.31%。其平均 LOAD 请求只有
`237.3→1,126.4 B`，F00 则为 `2,232.9→7,736.9 B`。

这构成一个比“某个驻留方案更快”更重要的搜索规律：

```text
跨 tile 的外存代价 != 只看总字节
跨 tile 的外存代价 = bytes/BW + calls×请求启动代价 + 请求形态/同步/计算上下文
```

因此最终 fused program 的 LOAD bytes 和 LOAD calls 应共同进入共享内存服务代价；真实 FPGA
latency 仍是准入标签。两个点不足以拟合通用系数，也不能把逻辑 calls 写成物理 AXI burst。

## 下一步

1. 冻结独立 workload/模型的多调用点 bundle，避免继续只增加 R50A tile；
2. 寻找正常 same-mode whole group 自然失败域，实际跑通递归拆分；
3. 在新 boot 预注册复测一个 bundle，分离启动差异；
4. 把稳定 call-site identity 前移到 Relay/AutoTVM 搜索层。

## 关键产物

- `20260912_p7r289_r50a_second_callsite_latency_holdout_contract_run01`
- `20260912_p7r290_r50af01_resnet50_generic_pair_build_run01`
- `20260912_p7r291_r50af01_resnet50_fused_tir_audit_run01`
- `20260912_p7r292_r50af01_fused_tir_occurrence_contract_run01`
- `20260912_p7r293_r50af01_callsite_holdout_subsets_run01`
- `20260912_p7r294_r50af01_callsite_holdout_board_run01`（环境失败，未执行候选）
- `20260912_p7r294_r50af01_callsite_holdout_board_run02`（有效）
- `20260912_p7r295_r50af01_callsite_latency_holdout_audit_run01`
- `20260912_p7r296_r50a_callsite_tile_holdout_comparison_run01`
