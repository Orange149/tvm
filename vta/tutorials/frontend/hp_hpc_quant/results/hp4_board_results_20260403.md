# AXU5EVB 4HP Board Results 2026-04-03

## Summary

- 配置：`4HP bitstream + HP/non-coherent runtime`
- 板端：`192.168.1.247:9090`
- 本轮重新确认：
  - `4HP` 已正确加载
  - 单算子和切图均可稳定运行
  - 不混入 `VTA_STAGE_INPUT_EXT_DEV` / `VTA_STAGE_BOUNDARY_INPUT_UNCACHED`

本轮最重要的结论是：

1. `4HP` 不是“完全没意义”
2. 在 `224` 和小单算子场景下，它没有带来稳定的端到端收益
3. 在 `320x320 + three_stage_e` 这类更大的边界场景下，`4HP` 开始显著优于旧 `1HP`
4. `4HP` 提高了有效 DMA 吞吐，但没有降低 `HP/non-coherent` 的一致性维护开销

## Artifacts

- 单算子 CSV：[`/tmp/hp4_conv_compare.csv`](/tmp/hp4_conv_compare.csv)
- 单算子 profiler：[`/tmp/hp4_conv_profile`](/tmp/hp4_conv_profile)
- 重复稳定性：`/tmp/hp4_repeat_1.csv` 到 `5.csv`
- 整网 profiler：[`/tmp/hp4_e2e_profile`](/tmp/hp4_e2e_profile)
- 切图日志：[`/tmp/hp4_scheme_sweep`](/tmp/hp4_scheme_sweep)
- `320` 切图日志：[`/tmp/hp4_res_sweep_three_stage_e_320.log`](/tmp/hp4_res_sweep_three_stage_e_320.log)
- throughput/coherence：[`/tmp/hp4_coherence_bw/coherence_throughput.csv`](/tmp/hp4_coherence_bw/coherence_throughput.csv)
- throughput/coherence summary：[`/tmp/hp4_coherence_bw/summary.json`](/tmp/hp4_coherence_bw/summary.json)

## Single-Op

| case | 1HP kernel_ms | 4HP kernel_ms | HPC kernel_ms | 1HP total_ms | 4HP total_ms | HPC total_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `s1_conv3x3_64_64` | 43.248 | 31.405 | 38.887 | 357.601 | 325.467 | 330.630 |
| `s2_conv3x3_128_128` | 30.134 | 23.949 | 26.540 | 430.184 | 618.351 | 412.497 |
| `s4_conv3x3_512_512` | 87.414 | 74.762 | 65.165 | 2893.721 | 2816.194 | 2746.674 |

观察：

- `4HP` 的 `kernel_ms` 比旧 `1HP` 全面更好。
- `s1/s2` 的 `kernel_ms` 甚至优于当前 `HPC`。
- 但 `total_ms` 没有同步改善：
  - `s2` 明显更差，主要被 `h2d` 拖慢。
  - `s4` 虽优于旧 `1HP`，仍落后于 `HPC`。
- 这说明 `4HP` 更像是提升了 VTA 内核/数据通路局部效率，但 host-side/提交/传输成本仍然很重。

### Repeat Stability

`s1_conv3x3_64_64` 5 次重复：

- `kernel_ms avg = 31.592`
- `kernel_ms std = 0.017`
- `total_ms avg = 379.587`
- `total_ms std = 2.380`
- 全部 `ok=True`

结论：

- `4HP` 在最小单算子上是稳定的。
- 当前不是“偶发快/偶发错”，而是稳定地落在这组性能区间。

## End-to-End

### Single Run

- `set_params = 1592.820 ms`
- `set_data = 61.131 ms`
- `run = 276.638 ms`
- `get_output = 21.995 ms`
- `total = 1952.585 ms`

`VTA-RUNTIME` single run：

- `flush_cache = 127.473 ms`
- `device_run_wait = 80.948 ms`

### Params-Once Steady-State

- `set_data = 60.739 ms`
- `run = 122.308 ms`
- `get_output = 2.437 ms`
- `total = 185.485 ms`

`VTA-RUNTIME` params-once totals：

- `flush_cache = 0`
- `invalidate_cache avg_per_run = 0.193 ms`
- `device_run_wait avg_per_run = 80.892 ms`

对照旧 `1HP`：

- 旧 `1HP params-once total = 187.282 ms`
- 新 `4HP params-once total = 185.485 ms`

结论：

- `4HP` 对整图 steady-state 的改善非常小，只有约 `1.8 ms`。
- `single run` 里显式 `flush_cache` 仍然很重，和旧 `1HP` 基本同量级。
- 这说明 `4HP` 没有改变 `HP/non-coherent` 的本质一致性成本，只是略微改善了 steady-state 数据路径。

## Stage Sweep

### 224x224

| scheme | 1HP pipeline_ms | 4HP pipeline_ms | HPC pipeline_ms | 1HP total.service_ms | 4HP total.service_ms | HPC total.service_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `all_vta` | 115.251 | 114.586 | 103.762 | 178.645 | 179.494 | 167.487 |
| `three_stage_d` | 94.692 | 95.074 | 95.074 | 237.330 | 233.183 | 219.913 |
| `three_stage_e` | 79.626 | 78.836 | 75.691 | 223.321 | 226.039 | 210.709 |

补充：

- `three_stage_a` 4HP：
  - `stage1_vta run_mean_ms = 49.297`
  - `pipeline_cycle_run_ms = max(78.289 + 43.057, 49.297) = 121.346`
  - `total.service_ms = 244.710`

结论：

- `224` 下，`4HP` 对 `all_vta` 基本没有改善。
- `three_stage_d/e` 的 `pipeline_ms` 与旧 `1HP` 基本持平，只是小幅波动。
- `HPC` 在 `224` 下仍然更强。

### 320x320 `three_stage_e`

| mode | total.service_ms | stage0_out_ms | stage1_vta.run_ms | stage2_cpu.run_ms | flush_time_ms | device_run_wait_ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1HP` | 1780.833 | 2.924 | 1437.804 | 70.025 | 57.571 | 1093.174 |
| `4HP` | 1472.485 | 3.253 | 1129.058 | 69.655 | 57.509 | 783.996 |
| `HPC` | 1544.650 | - | 1199.655 | - | 0.000 | - |

结论：

- `320` 是本轮唯一明确看到 `4HP` 价值的场景。
- 相比旧 `1HP`：
  - `total.service` 降低约 `308 ms`
  - `stage1_vta.run` 降低约 `309 ms`
  - `device_run_wait` 降低约 `309 ms`
- 这说明在更大的 boundary / 更重的 VTA 主段下，4HP 分流终于开始真正发挥作用。
- 而且这组里：
  - `4HP total.service = 1472.485 ms`
  - `HPC total.service = 1544.650 ms`
- 也就是说，`4HP` 在这个大边界 `three_stage_e 320` 场景下，已经优于当前 `HPC` 基线。

## Throughput / Coherence

### Summary

| config | avg total_bw_gbps | avg coherence_overhead_ms | avg kernel_ms | avg total_ms |
| --- | ---: | ---: | ---: | ---: |
| `1HP` | 0.162 | 79.712 | 41.428 | 131.000 |
| `4HP` | 0.222 | 79.771 | 32.620 | 122.683 |
| `HPC` | 0.162 | 0.000 | 33.499 | 119.411 |

关键点：

- `4HP` 把平均有效 DMA 吞吐从 `0.162` 拉到了 `0.222 Gbps`。
- 但 `coherence_overhead_ms` 几乎没变：
  - `1HP = 79.712 ms`
  - `4HP = 79.771 ms`
- 这说明：
  - 4HP 真正改善的是数据通路吞吐
  - 没有改善 `HP/non-coherent` 的显式一致性维护成本

### Interpretation

- `4HP` 不是“更少 flush”，而是“相同 coherence 成本下，更多有效 DMA 带宽”。
- 这解释了为什么：
  - 在小图、小边界下，收益不明显
  - 在 `320x320 three_stage_e` 这种更重的数据路径场景下，收益开始放大

## Conclusion

本轮可以得出更细的结论，而不是简单说 “4HP 好” 或 “4HP 没用”：

1. `4HP` 在功能上已经稳定，可正确运行单算子、整图、切图。
2. `4HP` 没有降低 `HP` 的一致性开销；`flush/invalidate` 问题本质上还在。
3. `4HP` 明显提高了有效 DMA 吞吐。
4. 在 `224` 和小 workload 下，这个收益不足以转化成稳定的端到端优势。
5. 在 `320x320 + three_stage_e` 这种更大边界、更重 VTA 段的场景下，`4HP` 开始显著优于旧 `1HP`，并且已经超过当前 `HPC` 基线。

工程建议：

- 如果目标是默认配置，当前还不能直接用 `4HP` 替代所有场景下的 `HPC`。
- 如果目标是更大输入、更重中间 feature map 的切图场景，`4HP` 已经值得继续投入。
- 下一步最值得做的是：
  - 继续围绕 `three_stage_e` 和更大分辨率做 sweep
  - 而不是再回去优化小图场景
