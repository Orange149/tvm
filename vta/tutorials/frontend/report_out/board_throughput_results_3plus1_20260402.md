# Board Throughput Results (3+1 CPU Version)

日期：`2026-04-02`

板端：
- RPC: `192.168.1.247:9090`
- 主测量口径：`time_evaluator("run")`
- 串行解释口径：`total.service avg`

## 口径说明

这份报告单独保存旧的 `3:1` CPU 结果：

- `stage0_cpu = 3` 线程
- `stage2_cpu = 1` 线程
- `stage1_vta` 为单 VTA 执行

吞吐周期定义为：

- `pipeline_cycle_run_ms = max(stage0_run_ms, stage1_vta_run_ms, stage2_run_ms)`

说明：
- 这不是当前默认口径
- 当前代码主线已经切到 `4CPU shared`
- 这份文件只作为旧结果归档，方便对照

## `all_vta` baseline

- `all_vta run_mean_ms = 103.456`
- `all_vta total.service avg = 169.763 ms`

## 13 组结果

这 13 组包含：
- `all_vta`
- `three_stage_a`
- 11 个已知可编译窗口

按 `pipeline_cycle_run_ms` 排序：

| rank | candidate | stage0_run_ms | stage1_run_ms | stage2_run_ms | pipeline_cycle_run_ms | ratio_vs_all_vta | total.service_ms |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `all_vta` | - | 103.456 | - | 103.456 | 1.000 | 169.763 |
| 2 | `three_stage_a` | 123.803 | 43.899 | 134.744 | 134.744 | 1.302 | 370.638 |
| 3 | `window_layer2_block1_main_preadd__layer3_block1_add_relu_tail` | 151.678 | 32.099 | 134.557 | 151.678 | 1.466 | 389.649 |
| 4 | `three_stage_b` | 177.503 | 20.782 | 134.006 | 177.503 | 1.716 | 400.353 |
| 5 | `window_layer3_block0_main_preadd__layer3_block0_add_relu_tail` | 177.413 | 10.764 | 201.358 | 201.358 | 1.946 | 457.727 |
| 6 | `block_stage_b` | 151.743 | 21.880 | 201.450 | 201.450 | 1.947 | 446.984 |
| 7 | `window_layer3_block1_main_preadd__layer4_block0_add_relu_tail` | 204.542 | 19.571 | 65.891 | 204.542 | 1.977 | 360.557 |
| 8 | `window_layer3_block1_main_preadd__layer4_block1_add_relu_tail` | 205.335 | 29.397 | 2.807 | 205.335 | 1.985 | 306.112 |
| 9 | `window_layer4_block0_main_preadd__layer4_block0_add_relu_tail` | 230.145 | 9.294 | 66.522 | 230.145 | 2.225 | 373.613 |
| 10 | `window_layer4_block0_main_preadd__layer4_block1_add_relu_tail` | 230.655 | 19.095 | 2.836 | 230.655 | 2.229 | 320.381 |
| 11 | `block_stage_c` | 123.776 | 24.763 | 275.961 | 275.961 | 2.667 | 493.386 |
| 12 | `window_layer2_block0_main_preadd__layer2_block0_add_relu_tail` | 123.935 | 13.713 | 340.177 | 340.177 | 3.288 | 545.393 |
| 13 | `window_layer1_block1_main_preadd__layer2_block0_add_relu_tail` | 79.131 | 26.844 | 341.221 | 341.221 | 3.298 | 516.164 |

## 结论

- 在 `3:1` 口径下，没有任何异构切图超过 `all_vta`
- 最接近 `all_vta` 的异构方案是：
  - `three_stage_a`
- 如果只看那 11 个已知可编译窗口，最接近 `all_vta` 的是：
  - `window_layer2_block1_main_preadd__layer3_block1_add_relu_tail`

## 与当前主线的关系

- 当前主线已经切到 `4CPU shared` 口径
- 因此这份文件只保留为历史对照
- 不应用这份结果去覆盖当前 `4CPU` 结论
