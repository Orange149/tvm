# HP-only Board Results

日期：`2026-04-03`

板端：
- RPC：`192.168.1.247:9090`
- bitstream：`HP-only`
- runtime：`non-coherent`
- 驱动状态：`u-dma-buf` sysfs 同步路径已修通，板端已不再出现 `msync failed` warning

本报告收口 `HP-only + non-coherent` 这一轮在切回 `coherent/HPC` 前的最小测试闭环。使用的主要产物：

- 单算子 3-case CSV：[hp_conv_compare.csv](/tmp/hp_conv_compare.csv)
- 单算子细粒度 profiler：[hp_conv_profile](/tmp/hp_conv_profile)
- 整网 hetero profiler：[hp_e2e_profile](/tmp/hp_e2e_profile)
- 分阶段 profiler：[hp_stage_profile](/tmp/hp_stage_profile)
- 多方案切图 sweep 日志：`/tmp/hp_scheme_sweep_20260403/*.log`
- 吞吐/一致性微基准：
  - CSV：[coherence_throughput.csv](/tmp/hp_coherence_bw_20260403/coherence_throughput.csv)
  - JSON：[summary.json](/tmp/hp_coherence_bw_20260403/summary.json)
- 重复稳定性 CSV：`/tmp/hp_repeat_1.csv` 到 `/tmp/hp_repeat_5.csv`

## 环境与状态

当前 `HP-only` 版本已经满足以下条件：

- 单算子 smoke 通过，`ok=True`
- 3 个代表性卷积 case 均通过，`ok=True`
- 整网 hetero benchmark 能稳定运行
- 分阶段 `three_stage_a` 可 build 和 run
- 板端 cache-maintenance 路径工作正常，warning 已消失

现有 profiler 结果也支持这一点：

- 单算子 `after_run` 能看到真实 `flush_cache` 活动，例如：
  - [`s1 after_run_status.json`](/tmp/hp_conv_profile/single_op/s1_conv3x3_64_64/after_run_status.json)：`flush_cache_calls=240`
  - [`s4 after_run_status.json`](/tmp/hp_conv_profile/single_op/s4_conv3x3_512_512/after_run_status.json)：`flush_cache_calls=1248`
- 整网 `after_run` 同样能看到 host->device flush：
  - [`after_run_status.json`](/tmp/hp_e2e_profile/after_run_status.json)：`flush_cache_calls=692`，`flush_cache_bytes=11176448`
- 所有当前采样窗口中：
  - `driver_timeout_calls=0`

结论：

- `HP-only` 功能正确
- cache-maintenance 已生效
- 当前数据已可以作为切回 `coherent/HPC` 的对照基线

## 单算子结果

来源：[hp_conv_compare.csv](/tmp/hp_conv_compare.csv)

| case | kernel_ms | total_ms | submit_ms | h2d_ms | d2h_ms | sync_ms | gops | ok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `s1_conv3x3_64_64` | 43.248 | 357.601 | 44.048 | 55.078 | 16.582 | 0.838 | 5.346 | `True` |
| `s2_conv3x3_128_128` | 30.134 | 430.184 | 31.026 | 44.402 | 8.951 | 0.747 | 7.673 | `True` |
| `s4_conv3x3_512_512` | 87.414 | 2893.721 | 88.301 | 239.286 | 4.670 | 0.748 | 2.645 | `True` |

这组结果说明：

- `s4_conv3x3_512_512` 是当前 `HP-only` 下最明显的重负载热点
- `sync_ms` 在 3 个 case 中都很低，不是当前单算子主瓶颈
- `h2d_ms` 和 `submit_ms` 占比明显，尤其 `s4` 的 host/device 准备成本很大

### 重复稳定性

来源：`/tmp/hp_repeat_1.csv` 到 `/tmp/hp_repeat_5.csv`

对 `s1_conv3x3_64_64` 连续独立执行 5 次，结果全部 `ok=True`。

- `kernel_ms avg = 43.422`
- `kernel_ms std = 0.029`
- `total_ms avg = 390.641`
- `total_ms std = 3.599`

结论：

- `HP-only` 当前没有出现偶发错误或 warning 回归
- kernel 时间波动很小，稳定性足够进入对照测试

## 整网、分阶段与多方案切图结果

### 整网 hetero profiler

来源：

- [`after_run_status.json`](/tmp/hp_e2e_profile/after_run_status.json)
- [`after_get_output_status.json`](/tmp/hp_e2e_profile/after_get_output_status.json)
- [`params_once_benchmark_totals_status.json`](/tmp/hp_e2e_profile/params_once_benchmark_totals_status.json)

`after_run` 关键字段：

- `flush_cache_calls = 692`
- `flush_cache_bytes = 11176448`
- `invalidate_cache_calls = 1`
- `driver_run_calls = 19`
- `driver_run_total_us = 81452`
- `device_run_wait_us = 81579.5`

`params_once benchmark` totals：

- `driver_run_calls = 380`
- `driver_timeout_calls = 0`
- `invalidate_cache_calls = 20`
- `invalidate_cache_bytes = 501760`
- `mem_copy_from_host_calls = 20`
- `mem_copy_from_host_bytes = 12042240`
- `load_buffer_2d_calls = 42200`
- `store_buffer_2d_calls = 2520`

解释：

- 整网窗口里 cache maintenance 是活跃的，不是“表面能跑但同步没发生”
- `params_once` totals 中 `flush_cache_calls=0` 是窗口定义导致的：steady-state benchmark 前数据已常驻，不代表 `HP-only` 不需要 flush

### 分阶段 `three_stage_a`

来源：

- [`steady_state_window_status.json`](/tmp/hp_stage_profile/stage_profile/steady_state_window_status.json)
- [`stage1_vta/after_run_status.json`](/tmp/hp_stage_profile/stage_profile/stage1_vta/after_run_status.json)

阶段平均服务时间：

- `stage0_cpu service avg = 139.151 ms`
- `stage1_vta service avg = 53.188 ms`
- `stage2_cpu service avg = 46.163 ms`
- `total.service avg = 238.503 ms`

纯 `run` 时间：

- `stage0_cpu run_mean_ms = 78.304`
- `stage1_vta run_mean_ms = 49.842`
- `stage2_cpu run_mean_ms = 43.250`

`steady_state_window` 关键 profiler：

- `driver_run_calls = 50`
- `driver_timeout_calls = 0`
- `invalidate_cache_calls = 5`
- `load_buffer_2d_calls = 5420`
- `store_buffer_2d_calls = 300`
- `synchronize_insns = 11640`

`stage1_vta` 单独窗口：

- `driver_run_calls = 10`
- `driver_timeout_calls = 0`
- `driver_run_total_us = 39123.6`
- `device_run_wait_us = 39192.2`
- `load_buffer_2d_calls = 1084`
- `store_buffer_2d_calls = 60`

结论：

- `stage1_vta` 已经被单独剖出，且运行稳定
- 但当前整图主要耗时仍然是 `stage0_cpu`，不是 `stage1_vta`
- 所以 `HP-only` 的当前整图端到端表现，还不能简单归因于“VTA/HP 数据面就是最大瓶颈”

### 其他可编译切图方案

本轮额外补跑了 6 组已知可编译方案：

- `all_vta`
- `three_stage_b`
- `three_stage_d`
- `three_stage_e`
- `block_stage_b`
- `block_stage_c`

日志位于：`/tmp/hp_scheme_sweep_20260403/*.log`

统一吞吐口径沿用当前主线 `4CPU shared`：

- `pipeline_cycle_run_ms = max(stage0_run_ms + stage2_run_ms, stage1_vta_run_ms)`
- `all_vta` 则直接使用其 `stage0_vta run_mean_ms`

| rank | scheme | stage0_run_ms | stage1_run_ms | stage2_run_ms | pipeline_cycle_run_ms | total.service_ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `three_stage_e` | 51.736 | 79.626 | 23.358 | 79.626 | 223.321 |
| 2 | `three_stage_d` | 51.617 | 69.623 | 43.075 | 94.692 | 237.330 |
| 3 | `all_vta` | 115.251 | - | - | 115.251 | 178.645 |
| 4 | `three_stage_b` | 116.375 | 23.993 | 43.118 | 159.493 | 253.926 |
| 5 | `block_stage_b` | 97.696 | 27.944 | 63.645 | 161.341 | 257.664 |
| 6 | `block_stage_c` | 78.438 | 27.886 | 84.022 | 162.460 | 264.472 |

这组结果说明：

- 在当前 `HP-only` 下，`three_stage_d` 和 `three_stage_e` 仍然优于 `all_vta`
- 其中 `three_stage_e` 最好，`pipeline_cycle_run_ms = 79.626`
- 相比 `all_vta = 115.251`，`three_stage_e` 明显更优
- 这和之前 `coherent/HPC` 报告里的总体趋势一致，不是 `HP-only` 才偶然出现的单点结果
- 但从串行 `total.service` 看，`all_vta` 仍然更小，因此这里的优势是吞吐口径上的，不是串行端到端绝对时间上的

### 吞吐与一致性微基准

本轮新增了专门的数据通路实验脚本：

- [measure_vta_coherence_throughput.py](/home/orange/code/tvm/vta/tutorials/frontend/hp_hpc_quant/measure_vta_coherence_throughput.py)

这个脚本复用现有单算子编译/运行路径，但单独输出以下口径：

- `load_buffer_2d_bytes`
- `store_buffer_2d_bytes`
- `device_run_wait_us`
- `flush_cache_us`
- `invalidate_cache_us`
- `total_bw_gbps = (load_bytes + store_bytes) / device_run_wait_us`
- `coherence_overhead_ms = flush_cache_us + invalidate_cache_us`

注意：

- 这里的 `coherence_overhead_ms` 是一次完整 host->device->host 窗口里的显式 cache-maintenance 时间
- 它不是 `driver_run_total_us` 的子集
- 所以它可以大于 `driver_run_total_us`；这不表示统计错误，而是表示“run 前后的 flush/invalidate”已经超过纯设备 run 等待时间

本轮 `HP-only` 默认测了 6 个 case：

| case | total_bw_gbps | coherence_overhead_ms | load_bytes | store_bytes |
| --- | ---: | ---: | ---: | ---: |
| `s1_conv3x3_64_64` | 0.124 | 43.081 | 4458496 | 200704 |
| `s2_proj1x1_64_128_s2` | 0.037 | 27.832 | 1032192 | 100352 |
| `s2_conv3x3_128_128` | 0.147 | 33.455 | 3619840 | 100352 |
| `s3_proj1x1_128_256_s2` | 0.043 | 46.200 | 1247232 | 50176 |
| `s4_proj1x1_256_512_s2` | 0.291 | 101.251 | 1517568 | 25088 |
| `s4_conv3x3_512_512` | 0.331 | 226.452 | 18708480 | 25088 |

聚合结果：

- `total_bw_gbps avg = 0.162`
- `total_bw_gbps max = 0.331`
- `coherence_overhead_ms avg = 79.712`
- `coherence_overhead_ms max = 226.452`
- 最强带宽 case：
  - `s4_conv3x3_512_512`
- 最重一致性开销 case：
  - `s4_conv3x3_512_512`

这组结果说明：

- 当前 `HP-only` 下，DMA 有效吞吐会随 case 尺寸明显变化，不是一个固定值
- 大 case 才能把 `HP` 数据通路逼到更高带宽区间
- 在当前 workload 组织下，显式一致性开销已经足够大，必须和 `HPC/coherent` 做同口径对照，不能只看端到端时间猜

## 结论与是否切回 Coherent

当前 `HP-only` 最小充分集已经完成：

- 3-case 单算子：完成
- 整网 hetero benchmark：完成
- stage profile：完成
- 5 次重复稳定性：完成
- 额外可编译切图方案 sweep：完成
- 吞吐/一致性微基准：完成

因此当前结论是：

1. `HP-only + non-coherent` 已经具备可信的功能正确性数据
2. cache-maintenance 路径已被验证为真实生效
3. 没有 timeout，也没有 warning 回归
4. `s4_conv3x3_512_512` 是当前最重单算子热点
5. 整图级上，`stage0_cpu` 仍比 `stage1_vta` 更重
6. 在当前 `HP-only` 下，`three_stage_d/e` 依然比 `all_vta` 更有吞吐潜力，其中 `three_stage_e` 最好
7. `HP-only` 的数据通路吞吐和显式一致性开销已经有单独基线，可直接与 `HPC/coherent` 做同口径对照

是否还需要别的测试：

- 切回 `coherent/HPC` 前，不再缺必测项
- 可以直接切回并用同一组命令做对照测试
- 这次已经补过“其他可编译切图方案”，不需要再为了切图覆盖率额外停留在 `HP-only`
- 这次也已经补过专门的 `HP-only` 吞吐/一致性实验，不需要再额外发明新的 `HP` 单独基准
- 可选项只有 `50-run long_run`：
  - 如果你想验证长期稳定性，再补做
  - 如果你的目标是尽快进入 `HP-only vs HPC/coherent` 对照，这一项可以先不做

下一步建议：

1. 切回 `coherent/HPC` bitstream 和匹配 runtime
2. 复用同样的 4 组测试入口：
   - 3-case 单算子
   - 5 次重复稳定性
   - 整网 hetero benchmark
   - `three_stage_a` stage profile
3. 再补一轮切图对照，至少包括：
   - `all_vta`
   - `three_stage_d`
   - `three_stage_e`
4. 用同一个吞吐/一致性脚本再跑一轮：
   - `vta/tutorials/frontend/hp_hpc_quant/measure_vta_coherence_throughput.py --config-label hpc_coherent`
5. 用完全相同的产物结构再出一份 `HPC/coherent` 报告，与本报告逐项对照
