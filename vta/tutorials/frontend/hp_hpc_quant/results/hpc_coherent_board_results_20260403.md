# HPC/Coherent Board Results

日期：`2026-04-03`

板端：
- RPC：`192.168.1.247:9090`
- bitstream：`HPC/coherent`
- runtime：`coherent`
- 启动方式：`./start_axu5evb_hpc_rpc.sh`

本报告与 [`hp_only_board_results_20260403.md`](/home/orange/code/tvm/vta/tutorials/frontend/hp_hpc_quant/results/hp_only_board_results_20260403.md) 保持同口径，所有对照都尽量复用同一组脚本与参数。

主要产物：

- 单算子 3-case CSV：[hpc_conv_compare.csv](/tmp/hpc_conv_compare.csv)
- 单算子 profiler：`/tmp/hpc_conv_profile`
- 整网 profiler：[hpc_e2e_profile](/tmp/hpc_e2e_profile)
- 切图日志：`/tmp/hpc_scheme_sweep_20260403/*.log`
- 重复稳定性 CSV：`/tmp/hpc_repeat_1.csv` 到 `/tmp/hpc_repeat_5.csv`
- 吞吐/一致性微基准：
  - CSV：[coherence_throughput.csv](/tmp/hpc_coherence_bw_20260403/coherence_throughput.csv)
  - JSON：[summary.json](/tmp/hpc_coherence_bw_20260403/summary.json)

## 环境与状态

当前 `HPC/coherent` 版本已经满足：

- 单算子 3-case 全部 `ok=True`
- 5 次重复稳定性全部 `ok=True`
- 整网 hetero benchmark 可稳定运行
- `all_vta`、`three_stage_d`、`three_stage_e` 都可 build/run
- 无 timeout，无 RPC/runtime warning 回归

一个与 `HP-only` 的环境差异：

- `vta.runtime.profiler_clear/status` 存在
- `vta.runtime.profiler_events` 在当前 HPC runtime 中不存在

这不影响主结论，因为这轮对照主要依赖 `profiler_status` 的汇总字段；只是少了 event 级时间线。

## 单算子结果

来源：

- HP：[hp_conv_compare.csv](/tmp/hp_conv_compare.csv)
- HPC：[hpc_conv_compare.csv](/tmp/hpc_conv_compare.csv)

| case | HP kernel_ms | HPC kernel_ms | HP total_ms | HPC total_ms |
| --- | ---: | ---: | ---: | ---: |
| `s1_conv3x3_64_64` | 43.248 | 38.887 | 357.601 | 330.630 |
| `s2_conv3x3_128_128` | 30.134 | 26.540 | 430.184 | 412.497 |
| `s4_conv3x3_512_512` | 87.414 | 65.165 | 2893.721 | 2746.674 |

单算子结论：

- `HPC/coherent` 在这 3 个 case 上 kernel 时间都优于 `HP-only`
- 改善最明显的是 `s4_conv3x3_512_512`
  - `87.414 ms -> 65.165 ms`
- total 时间也同步下降，但下降幅度小于 kernel，因为 host 侧准备/拷贝仍然占一部分

### 重复稳定性

对 `s1_conv3x3_64_64` 连续独立执行 5 次：

| config | kernel_avg_ms | kernel_std_ms | total_avg_ms | total_std_ms |
| --- | ---: | ---: | ---: | ---: |
| `HP-only` | 43.422 | 0.029 | 390.641 | 3.599 |
| `HPC/coherent` | 38.865 | 0.024 | 335.765 | 3.365 |

结论：

- `HPC/coherent` 不只更快，而且稳定性同样良好
- 当前对照不是偶发波动造成的

## 整网与切图结果

### 整网 hetero benchmark

来源：

- HP：[hp_e2e_profile/params_once_benchmark_totals_status.json](/tmp/hp_e2e_profile/params_once_benchmark_totals_status.json)
- HPC：[hpc_e2e_profile/params_once_benchmark_totals_status.json](/tmp/hpc_e2e_profile/params_once_benchmark_totals_status.json)

关键对照：

| metric | HP-only | HPC/coherent |
| --- | ---: | ---: |
| `device_run_wait_us` | 1628640.0 | 1628640.0 |
| `load_buffer_2d_bytes` | 366868480 | 366868480 |
| `store_buffer_2d_bytes` | 33617920 | 33617920 |
| `mem_copy_from_host_calls` | 20 | 20 |
| `invalidate_cache_calls` | 20 | 0 |
| `flush_cache_calls` | 0 | 0 |

单次 `after_run` 窗口的差异更直观：

| metric | HP-only | HPC/coherent |
| --- | ---: | ---: |
| `flush_cache_calls` | 692 | 0 |
| `flush_cache_bytes` | 11176448 | 0 |
| `invalidate_cache_calls` | 1 | 0 |
| `load_buffer_2d_bytes` | 18343424 | 18343424 |
| `store_buffer_2d_bytes` | 1680896 | 1680896 |
| `device_run_wait_us` | 81579.5 | 81444.3 |

结论：

- 整网 steady-state 下，HPC 的 DMA 数据量与 HP 基本相同
- 但 `HPC/coherent` 不再需要显式 `flush/invalidate`
- 单次 window 中，HP 的显式 cache maintenance 很明显，HPC 则为 0

### 切图对照

来源：

- HP：`/tmp/hp_scheme_sweep_20260403/*.log`
- HPC：`/tmp/hpc_scheme_sweep_20260403/*.log`

统一口径：

- `pipeline_cycle_run_ms = max(stage0_run_ms + stage2_run_ms, stage1_run_ms)`
- `all_vta` 直接用 VTA 单段 `run_mean_ms`

| scheme | HP pipeline_ms | HPC pipeline_ms | HP total.service_ms | HPC total.service_ms |
| --- | ---: | ---: | ---: | ---: |
| `all_vta` | 115.251 | 103.762 | 178.645 | 167.487 |
| `three_stage_d` | 94.692 | 95.074 | 237.330 | 219.913 |
| `three_stage_e` | 79.626 | 75.691 | 223.321 | 210.709 |

切图结论：

- 两边最优切图都仍然是 `three_stage_e`
- `HPC/coherent` 下：
  - `all_vta` 更快
  - `three_stage_e` 也更快
- `three_stage_d` 基本与 HP-only 持平，说明不是所有切图都同等受益

## 吞吐与一致性开销

来源：

- HP：[summary.json](/tmp/hp_coherence_bw_20260403/summary.json)
- HPC：[summary.json](/tmp/hpc_coherence_bw_20260403/summary.json)

聚合对照：

| metric | HP-only | HPC/coherent |
| --- | ---: | ---: |
| `total_bw_gbps avg` | 0.1621 | 0.1624 |
| `total_bw_gbps max` | 0.3307 | 0.3315 |
| `load_bw_gbps avg` | 0.1588 | 0.1591 |
| `coherence_overhead_ms avg` | 79.712 | 0.000 |
| `coherence_overhead_ms max` | 226.452 | 0.000 |
| `coherence_pct_of_driver_run avg` | 469.091% | 0.000% |

最强带宽 case 两边相同：

- `s4_conv3x3_512_512`

吞吐/一致性结论：

- `HP` 与 `HPC` 的**有效 DMA 吞吐几乎一样**
- 当前差异不在“总线带宽明显不同”，而在**一致性处理方式**
- `HP-only` 的显式 `flush/invalidate` 开销非常大
- `HPC/coherent` 把这部分显式开销基本消掉了

这是本轮最关键的结论：

1. `HP` 没有表现出比 `HPC` 更高的有效 DMA 吞吐
2. `HPC` 明确消除了 `HP` 那部分显式一致性开销
3. 因此当前 workload 下，`HPC/coherent` 整体更优是合理且可解释的

## 最终结论

这轮 `HPC/coherent` 对照已经足够支撑结论：

1. 单算子、整网、切图、吞吐微基准全部完成
2. `HPC/coherent` 在当前 3-case 单算子上全面优于 `HP-only`
3. `three_stage_e` 在两边都仍然是最优切图
4. `HP` 与 `HPC` 的有效 DMA 吞吐非常接近
5. 真正拉开差距的是一致性成本：
   - `HP-only` 需要显式 `flush/invalidate`
   - `HPC/coherent` 这部分几乎为 0

因此当前更准确的判断是：

- 这套 VTA workload 下，`HPC/coherent` 优势主要来自**省掉显式一致性维护**
- 不是来自 `HPC` 明显更高的原始 DMA 吞吐

如果下一步继续做 `HP+HPC` 混合实验，最值得验证的方向不是“HP 会不会更快”，而是：

- 能否把大吞吐但不敏感的流量分到 `HP`
- 同时保留 `HPC` 处理一致性敏感部分
- 并且最终总收益要大于混合拓扑带来的复杂度
