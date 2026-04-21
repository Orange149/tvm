# VTA Per-Op Breakdown

日期：`2026-04-03`

板端：
- RPC：`192.168.1.247:9090`
- 模式：`vta + hetero`
- 分辨率：`224`、`320`

本报告基于新脚本 [`profile_vta_per_op_breakdown.py`](/home/orange/code/tvm/vta/tutorials/frontend/profile_vta_per_op_breakdown.py)，目标是拿到“一次推理里每个图节点的时间与 VTA runtime 增量”。

产物：
- `224`
  - [`per_node_summary.csv`](/tmp/vta_per_op_breakdown_224/per_node_summary.csv)
  - [`summary.json`](/tmp/vta_per_op_breakdown_224/summary.json)
  - [`per_op_table.txt`](/tmp/vta_per_op_breakdown_224/per_op_table.txt)
- `320`
  - [`per_node_summary.csv`](/tmp/vta_per_op_breakdown_320/per_node_summary.csv)
  - [`summary.json`](/tmp/vta_per_op_breakdown_320/summary.json)
  - [`per_op_table.txt`](/tmp/vta_per_op_breakdown_320/per_op_table.txt)

## Summary

这轮逐节点 profiling 有两个非常清楚的结论：

1. 当前板端配置下，**没有任何节点触发 `flush_cache`**
2. 当前推理主瓶颈是 **VTA device wait**，而且主要集中在后段几个大卷积节点上

也就是说，这次结果落在第三类：

- 逐节点看下来，真正主瓶颈主要是 `VTA device wait`，不是显式一致性

从行为上看，这更像是 `HPC/coherent` 路径：
- `nodes_with_flush = 0`
- `sum_flush_ms = 0`
- 所有热点节点都直接表现为 `device_run_wait_ms` 很高

## 全局统计

| image | total_nodes | nodes_with_loads | nodes_with_stores | nodes_with_flush | sum_seq_ms | sum_bench_ms | sum_device_run_wait_ms | sum_flush_ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | 81 | 19 | 19 | 0 | 155.011 | 109.990 | 81.435 | 0.000 |
| 320 | 81 | 19 | 19 | 0 | 1678.925 | 1617.079 | 1465.705 | 0.000 |

观察：

- 两档分辨率下，真正发起 VTA DMA 的节点数都稳定在 `19`
- 两档分辨率下，`flush_cache` 都是 `0`
- `320` 相比 `224`，`sum_device_run_wait_ms` 从 `81.435` 暴涨到 `1465.705`
- 这说明分辨率放大后，主要变重的是 VTA 执行等待，不是边界同步

## Top N 节点按 `seq_ms`

### 224

| rank | node | seq_ms | bench_ms | device_run_wait_ms |
| ---: | --- | ---: | ---: | ---: |
| 1 | `tvmgen_default_fused_nn_conv2d_add_nn_relu` | 20.547 | 19.650 | 0.000 |
| 2 | `...clip_cast_1` | 7.094 | 6.392 | 6.271 |
| 3 | `...clip_cast` | 6.962 | 6.391 | 6.264 |
| 4 | `...add_right_shift_clip_cast_1` | 6.823 | 6.125 | 6.008 |
| 5 | `...add_right_shift_clip_cast` | 6.650 | 6.125 | 6.008 |

### 320

| rank | node | seq_ms | bench_ms | device_run_wait_ms |
| ---: | --- | ---: | ---: | ---: |
| 1 | `...add_right_shift_clip_cast_7` | 113.816 | 112.058 | 98.783 |
| 2 | `...add_right_shift_clip_cast_6` | 112.980 | 112.122 | 98.817 |
| 3 | `...relu_add_right_shift_clip_cast_7` | 112.109 | 111.484 | 97.658 |
| 4 | `...add_right_shift_clip_cast_4` | 89.881 | 89.012 | 82.193 |
| 5 | `...add_right_shift_clip_cast_4_1` | 89.729 | 88.897 | 82.206 |

观察：

- `224` 下最慢节点是第一层大卷积 `tvmgen_default_fused_nn_conv2d_add_nn_relu`
- `320` 下热点明显后移，最慢节点集中在后段 `..._4/_6/_7` 这些大通道卷积
- `seq_ms` 和 `bench_ms` 很接近，说明脚本逐节点执行的墙钟和 `debug_executor` 的 per-op 时间是一致的

## Top N 节点按 `flush_cache_ms`

### 224

所有节点 `flush_cache_ms = 0.000`

### 320

所有节点 `flush_cache_ms = 0.000`

结论：

- 当前逐节点结果里，`flush` 不是瓶颈，也不是局部热点
- 如果你要研究 `flush` 对性能的影响，这份结果本身不支持，因为当前板端模式没有把 `flush` 暴露出来
- 要研究 `flush`，应在 `HP/non-coherent` 路径下重跑同一脚本

## Top N 节点按 `device_run_wait_ms`

### 224

| rank | node | device_run_wait_ms | load_bytes | store_bytes | synchronize_insns |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `...relu_add_right_shift_clip_cast_1` | 6.271 | 747264 | 200704 | 332 |
| 2 | `...relu_add_right_shift_clip_cast` | 6.264 | 747264 | 200704 | 332 |
| 3 | `...add_right_shift_clip_cast` | 6.008 | 747264 | 200704 | 316 |
| 4 | `...add_right_shift_clip_cast_1` | 6.008 | 747264 | 200704 | 316 |
| 5 | `...relu_add_right_shift_clip_cast_3` | 5.420 | 726016 | 100352 | 296 |

### 320

| rank | node | device_run_wait_ms | load_bytes | store_bytes | synchronize_insns |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `...add_right_shift_clip_cast_6` | 98.817 | 28200960 | 51200 | 33622 |
| 2 | `...add_right_shift_clip_cast_7` | 98.783 | 28200960 | 51200 | 33622 |
| 3 | `...relu_add_right_shift_clip_cast_7` | 97.658 | 28200960 | 51200 | 33944 |
| 4 | `...add_right_shift_clip_cast_4_1` | 82.206 | 16568320 | 102400 | 18262 |
| 5 | `...add_right_shift_clip_cast_4` | 82.193 | 16568320 | 102400 | 18262 |

观察：

- `device_run_wait_ms` 和节点总时间几乎同涨同跌
- `320` 下最重的节点都具有：
  - 极高的 `load_bytes`
  - 极高的 `synchronize_insns`
  - 相对很小的 `store_bytes`
- 这说明当前 VTA 热点更像是“重输入读取 + 长执行等待”的卷积节点，而不是输出回写节点

## Per-op 表的补充结论

从 [`per_op_table.txt`](/tmp/vta_per_op_breakdown_224/per_op_table.txt) 和 [`per_op_table.txt`](/tmp/vta_per_op_breakdown_320/per_op_table.txt) 看：

- `224` 下：
  - 第一层 `tvmgen_default_fused_nn_conv2d_add_nn_relu` 占比最高，约 `17.77%`
  - 后续多个 `conv2d + add + clip/cast` 节点分散在 `4%` 到 `11%`
- `320` 下：
  - 热点完全由一组后段 `conv2d + add + clip/cast` 节点主导
  - 单节点占比大约在 `4.7%` 到 `11.0%`
  - 第一层大卷积反而只剩约 `2.38%`

这和逐节点 `device_run_wait_ms` 的分布是吻合的：

- `320` 下真正的热点不是输入边界，而是后段主干卷积

## 结论

这轮逐节点细粒度 profiling 支持以下判断：

1. 当前板端模式下，**显式 `flush` 不是问题**
2. 当前瓶颈主要集中在少数 VTA 卷积节点，特别是 `320` 下的后段卷积
3. `320` 相比 `224` 的主要变化不是“边界同步更多”，而是：
   - 部分 VTA 卷积节点的 `device_run_wait_ms` 成倍增长
   - `load_buffer_2d_bytes` 与 `synchronize_insns` 成倍增长

因此，这份结果更支持下面这个结论：

- 如果要继续优化当前推理路径，优先级应该放在：
  - 后段大卷积节点的 VTA 执行路径
  - 或切图/打包方式对这些节点的影响
- 而不是继续把主要精力放在 `flush/invalidate` 上

如果你要继续下一步，我建议直接做这两件事之一：

1. 在 `HP/non-coherent` 下重跑这同一脚本，对比逐节点 `flush` 是否真的集中在边界节点
2. 继续在当前 `HPC/coherent` 路径下，只盯 `320` 的后段热点卷积做更细的 schedule / packing 诊断
