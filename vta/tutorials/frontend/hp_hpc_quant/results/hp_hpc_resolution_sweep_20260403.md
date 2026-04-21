# HP vs HPC Resolution Sweep

日期：`2026-04-03`

板端：
- RPC：`192.168.1.247:9090`
- 方案：`three_stage_e`
- 分辨率：`224`、`320`
- 对比对象：
  - `HP-baseline`
  - `HP-boundary-opt`
  - `HPC/coherent`

本报告只回答一个问题：把输入从 `224x224` 提到 `320x320` 后，`HP` 是否开始出现现实优势。

测试口径固定为：
- `pipeline_cycle_run_ms = max(stage0_cpu.run + stage2_cpu.run, stage1_vta.run)`
- 串行解释口径：`total.service avg`
- 重点观测：
  - `boundary_bytes`
  - `stage0_cpu.out`
  - `stage1_vta.run`
  - `flush_cache_*`
  - `device_run_wait`

数据来源：
- `HP 224 baseline`：用户提供的 `three_stage_e` 日志
- `HP 224 boundary-opt`：用户提供的 `three_stage_e` 日志
- `HPC 224`：[`three_stage_e.log`](/tmp/hpc_scheme_sweep_20260403/three_stage_e.log)
- `HP 320 baseline`：[`hp_res_sweep_three_stage_e_hp_base_320.log`](/tmp/hp_res_sweep_three_stage_e_hp_base_320.log)
- `HP 320 boundary-opt`：[`hp_res_sweep_three_stage_e_hp_opt_320.log`](/tmp/hp_res_sweep_three_stage_e_hp_opt_320.log)
- `HPC 320`：[`hpc_res_sweep_three_stage_e_320.log`](/tmp/hpc_res_sweep_three_stage_e_320.log)

## 总结

结论很明确：

1. `320x320` 没有让 `HP` 开始超过 `HPC`
2. `HP-boundary-opt` 在 `224` 和 `320` 下都能把 `flush_cache` 降到 `0`
3. 但它带来的收益仍然只是局部收益：
   - `stage1_vta.run` 变好
   - `stage0_cpu.out` 明显变差
4. 在 `224` 和 `320` 两档分辨率下，最终都是 `HPC/coherent` 更快
5. 因此当前可以更有把握地认为：
   - `HP` 的问题不只是小分辨率下固定开销太大
   - 即使 feature map 边界增大到 `320`，`HP` 仍没有现实优势

## 主结果表

| mode | image | boundary_bytes | stage0_out_ms | stage1_run_ms | stage2_run_ms | pipeline_cycle_run_ms | total.service_ms | flush_calls | flush_time_ms | device_run_wait_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `HP-baseline` | 224 | 903168 | 2.140 | 80.924 | 24.680 | 80.924 | 224.767 | 140 | 5.258 | 59.640 |
| `HP-boundary-opt` | 224 | 903168 | 10.350 | 79.347 | 24.391 | 79.347 | 231.071 | 0 | 0.000 | 59.624 |
| `HPC/coherent` | 224 | 903168 | 2.698 | 66.181 | 24.668 | 77.676 | 210.709 | 0 | 0.000 | 59.623 |
| `HP-baseline` | 320 | 1843200 | 2.924 | 1437.804 | 70.025 | 1437.804 | 1780.833 | 1580 | 57.571 | 1093.174 |
| `HP-boundary-opt` | 320 | 1843200 | 19.527 | 1381.895 | 69.125 | 1381.895 | 1740.506 | 0 | 0.000 | 1092.810 |
| `HPC/coherent` | 320 | 1843200 | 3.605 | 1200.603 | 69.167 | 1200.603 | 1544.650 | 0 | 0.000 | 1116.642 |

## 表 1：`pipeline_cycle_run_ms / total.service`

| image | mode | pipeline_cycle_run_ms | total.service_ms |
| ---: | --- | ---: | ---: |
| 224 | `HP-baseline` | 80.924 | 224.767 |
| 224 | `HP-boundary-opt` | 79.347 | 231.071 |
| 224 | `HPC/coherent` | 77.676 | 210.709 |
| 320 | `HP-baseline` | 1437.804 | 1780.833 |
| 320 | `HP-boundary-opt` | 1381.895 | 1740.506 |
| 320 | `HPC/coherent` | 1200.603 | 1544.650 |

观察：

- `224` 下，`HPC` 已经是三条线里最优
- `320` 下，三条线都明显变慢，但 `HPC` 仍然最优
- `HP-boundary-opt` 在 `320` 下相对 `HP-baseline` 确实改善了：
  - `pipeline_cycle_run_ms` 降低约 `55.9 ms`
  - `total.service` 降低约 `40.3 ms`
- 但这仍然不足以追上 `HPC`
- `320` 下 `HPC` 相比 `HP-boundary-opt` 仍快约：
  - `181.3 ms` 的 `pipeline_cycle_run_ms`
  - `195.9 ms` 的 `total.service`

## 表 2：`flush_cache_*` 与 `stage0_cpu.out`

| image | mode | flush_calls | flush_time_ms | stage0_cpu.out_ms |
| ---: | --- | ---: | ---: | ---: |
| 224 | `HP-baseline` | 140 | 5.258 | 2.140 |
| 224 | `HP-boundary-opt` | 0 | 0.000 | 10.350 |
| 224 | `HPC/coherent` | 0 | 0.000 | 2.698 |
| 320 | `HP-baseline` | 1580 | 57.571 | 2.924 |
| 320 | `HP-boundary-opt` | 0 | 0.000 | 19.527 |
| 320 | `HPC/coherent` | 0 | 0.000 | 3.605 |

观察：

- 边界优化是有效的：
  - `224` 和 `320` 下都把 `flush_cache` 降到了 `0`
- 但代价也很稳定：
  - `stage0_cpu.out` 大幅上升
  - `224` 下从 `2.140` 变到 `10.350`
  - `320` 下从 `2.924` 变到 `19.527`
- 这说明边界优化本质上是在做成本迁移：
  - 从 `flush_cache` 迁到 `CPU` 直接写 `ext_dev/uncached` 边界 buffer
- 分辨率从 `224` 增到 `320` 后，这个迁移并没有变得更“划算”

## 表 3：`stage1_vta.run / device_run_wait`

| image | mode | stage1_vta.run_ms | device_run_wait_ms |
| ---: | --- | ---: | ---: |
| 224 | `HP-baseline` | 80.924 | 59.640 |
| 224 | `HP-boundary-opt` | 79.347 | 59.624 |
| 224 | `HPC/coherent` | 66.181 | 59.623 |
| 320 | `HP-baseline` | 1437.804 | 1093.174 |
| 320 | `HP-boundary-opt` | 1381.895 | 1092.810 |
| 320 | `HPC/coherent` | 1200.603 | 1116.642 |

观察：

- `HP-boundary-opt` 在两档分辨率下都改善了 `stage1_vta.run`
- 这说明先前的判断是对的：
  - `CPU -> VTA` 边界上的 `flush` 确实会拖慢 VTA 侧
- 但 `HPC` 在两档分辨率下都更快：
  - `224`：`66.181 ms` vs `79.347 ms`
  - `320`：`1200.603 ms` vs `1381.895 ms`
- 更重要的是：
  - `320` 下 `device_run_wait` 在 `HP-boundary-opt` 和 `HPC` 已经非常接近
  - 但 `stage1_vta.run` 仍有明显差距
- 这说明当前差距不能简单理解成“只要把 `flush` 清掉，`HP` 就会自然赢”

## 分辨率放大后的判断

这轮实验最关键的问题是：

- feature map 边界从 `903168 B` 增大到 `1843200 B` 后，`HP` 是否开始显示现实优势？

答案是否定的。

更具体地说：

1. `HP-baseline -> HP-boundary-opt`
   - 在 `320` 下收益比 `224` 更明显
   - 说明大一些的边界上，去掉 `flush` 的价值确实在增大

2. 但 `HP-boundary-opt -> HPC`
   - `HPC` 仍然更快
   - 而且差距并没有消失

3. 因此最合理的解释是：
   - `HP` 的问题不是只有 `224` 下固定开销太大
   - 即使边界更大，`HP` 也没有展现出足以超过 `HPC` 的净收益

## 结论

这轮 `224 vs 320` sweep 的结论落在第三类：

- `HP` 在 `224` 和 `320` 下都没有现实优势，可停止主线投入

更细一点的表述是：

- `HP` 边界优化能降低 `flush`
- 也能让 `stage1_vta.run` 变好
- 但总体收益仍然不如 `HPC/coherent`
- 分辨率增大到 `320` 后，这个结论没有发生反转

当前最合理的工程选择是：

1. 把 `HPC/coherent` 作为默认主线
2. 不再继续把主要精力投在纯 `HP-only` 路径
3. 如果还要继续研究，优先方向应该是：
   - 优化切图结构
   - 优化 `HPC` 路径本身
   - 而不是继续扩大 `HP` 专项优化面
