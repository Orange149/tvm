# VTA Per-Op Breakdown on HPC

日期：`2026-04-03`

板端：
- RPC：`192.168.1.247:9090`
- bitstream/runtime：`HPC/coherent`
- 模式：`vta + hetero`
- 分辨率：`224`、`320`

产物：
- `224`
  - [`per_node_summary.csv`](/tmp/vta_per_op_breakdown_hpc_224/per_node_summary.csv)
  - [`summary.json`](/tmp/vta_per_op_breakdown_hpc_224/summary.json)
  - [`per_op_table.txt`](/tmp/vta_per_op_breakdown_hpc_224/per_op_table.txt)
- `320`
  - [`per_node_summary.csv`](/tmp/vta_per_op_breakdown_hpc_320/per_node_summary.csv)
  - [`summary.json`](/tmp/vta_per_op_breakdown_hpc_320/summary.json)
  - [`per_op_table.txt`](/tmp/vta_per_op_breakdown_hpc_320/per_op_table.txt)

## Summary

这轮 `HPC/coherent` 的逐节点 breakdown 很明确：

- `224` 和 `320` 都是 `nodes_with_flush = 0`
- 热点都在 `ext_dev` 的卷积融合节点
- 分辨率从 `224` 提到 `320` 后，主要增长的是后段卷积节点的 `device_run_wait_ms`

也就是说，在 `HPC` 下：

- 显式一致性维护不是问题
- 真正的主瓶颈是后段 VTA 卷积的执行等待

## 全局统计

| image | total_nodes | nodes_with_loads | nodes_with_stores | nodes_with_flush | sum_seq_ms | sum_bench_ms | sum_device_run_wait_ms | sum_flush_ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 224 | 81 | 19 | 19 | 0 | 155.364 | 110.051 | 81.437 | 0.000 |
| 320 | 81 | 19 | 19 | 0 | 1678.741 | 1616.788 | 1465.610 | 0.000 |

## Top 节点按 `seq_ms`

### 224

| rank | node | seq_ms | bench_ms | device_run_wait_ms |
| ---: | --- | ---: | ---: | ---: |
| 1 | `tvmgen_default_fused_nn_conv2d_add_nn_relu` | 20.509 | 20.208 | 0.000 |
| 2 | `...clip_cast_1` | 7.147 | 6.396 | 6.271 |
| 3 | `...clip_cast` | 7.116 | 6.389 | 6.264 |
| 4 | `...add_right_shift_clip_cast_1` | 6.803 | 6.125 | 6.008 |
| 5 | `...add_right_shift_clip_cast` | 6.773 | 6.127 | 6.008 |

### 320

| rank | node | seq_ms | bench_ms | device_run_wait_ms |
| ---: | --- | ---: | ---: | ---: |
| 1 | `...add_right_shift_clip_cast_6` | 113.179 | 112.102 | 98.850 |
| 2 | `...relu_add_right_shift_clip_cast_7` | 113.077 | 111.479 | 97.967 |
| 3 | `...add_right_shift_clip_cast_7` | 112.824 | 112.060 | 98.798 |
| 4 | `...add_right_shift_clip_cast_4_1` | 90.154 | 88.897 | 82.207 |
| 5 | `...add_right_shift_clip_cast_4` | 90.036 | 89.035 | 82.205 |

## Top 节点按 `device_run_wait_ms`

### 224

| rank | node | device_run_wait_ms | load_bytes | store_bytes |
| ---: | --- | ---: | ---: | ---: |
| 1 | `...relu_add_right_shift_clip_cast_1` | 6.271 | 747264 | 200704 |
| 2 | `...relu_add_right_shift_clip_cast` | 6.264 | 747264 | 200704 |
| 3 | `...add_right_shift_clip_cast` | 6.008 | 747264 | 200704 |
| 4 | `...add_right_shift_clip_cast_1` | 6.008 | 747264 | 200704 |
| 5 | `...relu_add_right_shift_clip_cast_3` | 5.420 | 726016 | 100352 |

### 320

| rank | node | device_run_wait_ms | load_bytes | store_bytes |
| ---: | --- | ---: | ---: | ---: |
| 1 | `...add_right_shift_clip_cast_6` | 98.850 | 28200960 | 51200 |
| 2 | `...add_right_shift_clip_cast_7` | 98.798 | 28200960 | 51200 |
| 3 | `...relu_add_right_shift_clip_cast_7` | 97.967 | 28200960 | 51200 |
| 4 | `...add_right_shift_clip_cast_4_1` | 82.207 | 16568320 | 102400 |
| 5 | `...add_right_shift_clip_cast_4` | 82.205 | 16568320 | 102400 |

## Conclusion

当前 `HPC/coherent` 逐节点结果支持这两个判断：

1. `flush/invalidate` 不是当前主问题
2. `320` 下真正变重的是后段几个大卷积融合节点，它们的 `device_run_wait_ms` 和 `load_bytes` 成倍增长

这意味着如果接下来还要继续挖性能，优先方向应该是：

- 继续看后段 VTA 卷积节点
- 看它们的 packing / schedule / 切图方式
- 而不是继续把重点放在显式一致性维护上
