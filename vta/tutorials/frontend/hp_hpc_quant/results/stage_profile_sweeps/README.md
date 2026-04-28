# Stage Profile Sweeps

这份目录只保留三组 1HPC stage profile 实验，三组都已经重跑成统一口径：

- board runtime: coherent + sleep polling
- CPU 线程: `stage0=4`, `stage2=4`
- `--timer-number 1 --timer-repeat 3`
- `--events-limit 4096`

实际运行入口是：

- `vta/tutorials/frontend/hp_hpc_quant/run_stage_profile_sweep.py`

它内部调用：

- `vta/tutorials/frontend/profile_split_resnet18_stages.py`

一次 sweep 会按 `--schemes` 批量生成多个 scheme 子目录，所以 `1hpc/` 下面有多个结果目录是正常的，不是多次手工跑不同脚本。

## Run Commands

从 repo root 运行。

环境：

```bash
unset LD_LIBRARY_PATH
source /home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux
export TEST_DATA_ROOT_PATH=/tmp/tvm_test_data
export MPLCONFIGDIR=/tmp/mpl
export PYTHONUNBUFFERED=1
```

### 1hpc

完整 split sweep：

```bash
/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/hp_hpc_quant/run_stage_profile_sweep.py \
  --host 192.168.1.133 --port 9090 \
  --config-label 1hpc \
  --schemes all_vta,three_stage_a,three_stage_b,three_stage_d,three_stage_e \
  --repeat 10 --warmup-repeat 1 \
  --stage0-num-threads 4 --stage2-num-threads 4 \
  --timer-number 1 --timer-repeat 3 \
  --events-limit 4096
```

### 1hpc_sleep1us_coherent

聚焦 `all_vta` 和 `three_stage_e`：

```bash
/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/hp_hpc_quant/run_stage_profile_sweep.py \
  --host 192.168.1.133 --port 9090 \
  --config-label 1hpc_sleep1us_coherent \
  --schemes all_vta,three_stage_e \
  --repeat 10 --warmup-repeat 1 \
  --stage0-num-threads 4 --stage2-num-threads 4 \
  --timer-number 1 --timer-repeat 3 \
  --events-limit 4096
```

### 1hpc_sleep1us_coherent_cpu4

最小主线复现实验，只保留 `three_stage_e`：

```bash
/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/hp_hpc_quant/run_stage_profile_sweep.py \
  --host 192.168.1.133 --port 9090 \
  --config-label 1hpc_sleep1us_coherent_cpu4 \
  --schemes three_stage_e \
  --repeat 10 --warmup-repeat 1 \
  --stage0-num-threads 4 --stage2-num-threads 4 \
  --timer-number 1 --timer-repeat 3 \
  --events-limit 4096
```

## Output Layout

每组结果目录下包含：

- `sweep_manifest.json`
  - 本次 sweep 的参数、scheme 列表、执行命令、返回码
- `<scheme>/stage_profile.log`
  - 原始终端日志，包含 build/run/timing/VTA runtime 摘要
- `<scheme>/partition.svg`
  - 切图可视化
- `<scheme>/profile/*_status.json`
  - 累计统计
- `<scheme>/profile/*_events.json`
  - 事件明细；这三组现在都统一为 `4096` 条
- `<scheme>/profile/*_meta.json`
  - snapshot 元信息

## Results

### 1hpc: split sweep

`1hpc` 的主要用途是横向比较切图方案。当前结果：

| scheme | total.service avg (ms) | 备注 |
| --- | ---: | --- |
| `all_vta` | 180.502 | 单阶段 VTA 基线 |
| `three_stage_a` | 248.867 | CPU 头尾都偏重 |
| `three_stage_b` | 263.906 | 最慢，stage0 CPU 过重 |
| `three_stage_d` | 235.146 | 好于 a/b，但仍明显慢于 e |
| `three_stage_e` | 223.397 | split 方案里最优 |

`three_stage_e` 的 stage 级时间：

- `stage0_cpu.service = 119.113 ms`
- `stage1_vta.service = 75.921 ms`
- `stage2_cpu.service = 28.363 ms`
- `stage1_vta run_mean = 68.721 ms`

结论：在当前 1HPC + coherent + sleep polling + `4/4` 线程设置下，`three_stage_e` 仍然是最好的 split 方案。

### 1hpc_sleep1us_coherent: focused baseline

这一组只保留 `all_vta` 和 `three_stage_e`，用于聚焦主线对比。

| scheme | total.service avg (ms) | stage run mean (ms) |
| --- | ---: | --- |
| `all_vta` | 180.808 | `stage0_vta = 108.379` |
| `three_stage_e` | 224.568 | `stage0_cpu = 51.385`, `stage1_vta = 68.790`, `stage2_cpu = 23.392` |

这组和新的 `1hpc` 结果几乎一致，说明现在两者本质上已经是同一口径：

- 同一套 board runtime
- 同样的 `4/4` CPU 线程
- 同样的 `4096` event 抓取

因此这个目录的意义是“主线对比版”，不是新的实验条件。

### 1hpc_sleep1us_coherent_cpu4: minimal reproduction

这一组只跑 `three_stage_e`，当前结果：

- `total.service avg = 223.007 ms`
- `stage0_cpu.service = 119.188 ms`
- `stage1_vta.service = 75.760 ms`
- `stage2_cpu.service = 28.058 ms`
- `stage1_vta run_mean = 68.717 ms`

它和 `1hpc_sleep1us_coherent/three_stage_e` 也几乎重合。现在保留这个目录的意义主要是：

- 保留一个最小主线复现集
- 后续只分析 `three_stage_e` 时，不必带上完整 split sweep

## Runtime / DMA Summary

当前最有代表性的 VTA runtime 指标是 `three_stage_e`：

- `load_buffer_2d_calls/run = 1598`
- `store_buffer_2d_calls/run = 94`
- `device_run_wait_ms/run ≈ 60.17`
- `runtime enqueue/run`:
  - `load = 0.835 ms`
  - `store = 0.042 ms`
  - `gemm = 0.249 ms`
  - `alu = 0.212 ms`
  - `total ≈ 1.338 ms`
- `top load signatures`:
  - `inp:x=14,y=14,stride=14,count=2240`
  - `wgt:x=9,y=2,stride=72,count=1920`
  - `wgt:x=9,y=4,stride=144,count=1920`

这说明当前主要问题仍然是 VTA 子图内部的 input-side DMA 比较碎，典型形态仍然是 `14x14` input load。

`all_vta` 的 VTA runtime 指标：

- `load_buffer_2d_calls/run = 2110`
- `store_buffer_2d_calls/run = 126`
- `device_run_wait_ms/run ≈ 82.11`
- `runtime enqueue/run`:
  - `load = 1.015 ms`
  - `store = 0.057 ms`
  - `gemm = 0.321 ms`
  - `alu = 0.277 ms`
  - `total ≈ 1.670 ms`
- `top load signature`:
  - `wgt:x=9,y=2,stride=36,count=2240`

结论：

- `all_vta` 总时延最短，但 VTA 负载更重，DMA 次数更多
- `three_stage_e` 作为 split 方案最佳，仍值得作为后续 runtime / event / DMA 优化主线

## Event Analysis Results

这部分结果来自 `analyze_vta_profile_events.py` 对 `1hpc/three_stage_e` 和
`1hpc/all_vta` 的 `4096` 条事件窗口分析。

### `three_stage_e` event summary

- `events_count = 4096`
- `device_run_wait_us = 601661`
- `driver_poll_wait_us = 589650`
- `load_buffer_2d_calls = 15980`
- `store_buffer_2d_calls = 940`
- `load_buffer_2d_small_calls = 4780`
- `load_buffer_2d_strided_calls = 8480`
- `load_buffer_2d_padded_calls = 6560`
- `top load signatures`:
  - `inp:x=14,y=14,stride=14,count=2240`
  - `wgt:x=9,y=2,stride=72,count=1920`
  - `wgt:x=9,y=4,stride=144,count=1920`
- `event kind` 前几项：
  - `load_buffer_2d: 2192`
  - `push_gemm_op: 1152`
  - `push_alu_op: 610`
  - `store_buffer_2d: 114`
  - `synchronize: 22`
- `padded_ratio = 0.2148`
- `dma_signature_top` 前几项：
  - `load_buffer_2d:x=14,y=14,stride=14 -> 384`
  - `load_buffer_2d:x=9,y=4,stride=144 -> 320`
  - `load_buffer_2d:x=28,y=15,stride=28 -> 192`
  - `load_buffer_2d:x=9,y=2,stride=72 -> 192`
  - `load_buffer_2d:x=7,y=7,stride=7 -> 128`

这组结果和 `status.json` 的聚合统计一致：`three_stage_e` 的 VTA 子图仍然以
`14x14` input load 为最显著特征，small/strided/padded load 也都不少，说明
当前主要瓶颈还是碎 input DMA，而不是 enqueue 开销。

### `all_vta` event summary

- `events_count = 4096`
- `device_run_wait_us = 820869`
- `driver_poll_wait_us = 804981`
- `load_buffer_2d_calls = 21100`
- `store_buffer_2d_calls = 1260`
- `load_buffer_2d_small_calls = 6380`
- `load_buffer_2d_strided_calls = 10880`
- `load_buffer_2d_padded_calls = 8960`
- `top load signatures`:
  - `wgt:x=9,y=2,stride=36,count=2240`
  - `inp:x=14,y=14,stride=14,count=2240`
  - `wgt:x=9,y=2,stride=72,count=1920`
- `event kind` 前几项：
  - `load_buffer_2d: 2122`
  - `push_gemm_op: 1124`
  - `push_alu_op: 696`
  - `store_buffer_2d: 128`
  - `synchronize: 20`
- `padded_ratio = 0.22`
- `dma_signature_top` 前几项：
  - `load_buffer_2d:x=9,y=2,stride=36 -> 224`
  - `load_buffer_2d:x=14,y=14,stride=14 -> 224`
  - `load_buffer_2d:x=7,y=7,stride=7 -> 197`
  - `load_buffer_2d:x=9,y=16,stride=288 -> 197`
  - `load_buffer_2d:x=28,y=15,stride=28 -> 192`

`all_vta` 的事件窗口显示出更高的 VTA 侧压力：`device_run_wait`、
`driver_poll_wait`、`load/store calls` 都高于 `three_stage_e`。它总时延更短，但
代价是把更多工作和更多 DMA 负担压到了 VTA 上。

### Comparison and takeaways

- 两个方案的事件窗口都显示 `load_buffer_2d` 是最主要的 event 类型，DMA 仍然是主问题，而不是 enqueue 本身。
- `three_stage_e` 的总 load/store call 数明显少于 `all_vta`，这和它作为“最佳 split 方案”的定位一致。
- 但 `three_stage_e` 的 input side 仍然保留了非常强的 `14x14` load 形态，说明当前主要优化瓶颈依然是 VTA 子图内部的碎 input DMA。
- `all_vta` 的 `device_run_wait`、`driver_poll_wait`、`load/store calls` 都更高，说明它把更多工作压到了 VTA 上，虽然总时延更短，但 VTA 侧负担更重。
- 两边 `padded_ratio` 都在约 `0.21~0.22`，说明边界相关的 padded DMA 不是个别异常，而是当前 schedule 的稳定特征。
- 下一步的优化方向仍然应该是：优先减少 input-side 的 `14x14`、`7x7`、`strided/padded` load；再考虑 weight side 的进一步聚合；event 结果支持继续把 `three_stage_e` 作为 DMA 形态优化主线。

## Event Analysis Script

分析单个 scheme 的 `profile/` 目录：

```bash
/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/hp_hpc_quant/analyze_vta_profile_events.py \
  --profile-dir vta/tutorials/frontend/hp_hpc_quant/results/stage_profile_sweeps/1hpc/three_stage_e/profile
```

```bash
/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/hp_hpc_quant/analyze_vta_profile_events.py \
  --profile-dir vta/tutorials/frontend/hp_hpc_quant/results/stage_profile_sweeps/1hpc/all_vta/profile
```

脚本会输出：

- `kind` 分布
- `duration_us` 的总和 / 均值
- `memory_type` 分布
- `padded` 比例
- DMA event 的 `(x_size,y_size,x_stride)` top signatures
- 与 `status.json` 的交叉核对：
  - `event_count`
  - `device_run_wait_us`
  - `driver_poll_wait_us`
  - `load/store calls`
  - `small/strided/padded`
  - `top_signatures`

如果某次运行的 `events.json` 为空，脚本会自动退回用 `status.json` 做摘要；但对这三组当前结果来说，这已经不是主路径，因为所有 `events.json` 都是 `4096` 条。

## Summary and optimization directions

基于当前这批 `1hpc` profile，可以先得出下面几条结论。

### 1. `three_stage_e` 仍然是最好的 split 方案

数据依据：

- `1hpc/all_vta total.service = 180.502 ms`
- `1hpc/three_stage_e total.service = 223.397 ms`
- `1hpc/three_stage_d total.service = 235.146 ms`
- `1hpc/three_stage_a total.service = 248.867 ms`
- `1hpc/three_stage_b total.service = 263.906 ms`

结论：

- 如果允许整图都压到 VTA，`all_vta` 总时延最短。
- 如果限定在 split 方案里，`three_stage_e` 仍然是当前最优切图。
- 后续如果要继续做异构图优化，`three_stage_e` 仍然应该作为主线。

### 2. 当前主要瓶颈仍然是 VTA 子图内部的碎 DMA，尤其是 input-side load

数据依据：

- `three_stage_e load_buffer_2d_calls/run = 1598`
- `three_stage_e load_buffer_2d_small_calls/run = 478`
- `three_stage_e load_buffer_2d_strided_calls/run = 848`
- `three_stage_e load_buffer_2d_padded_calls/run = 656`
- `three_stage_e top load signature = inp:x=14,y=14,stride=14,count=2240`
- `three_stage_e` event top signatures:
  - `load_buffer_2d:x=14,y=14,stride=14 -> 384`
  - `load_buffer_2d:x=7,y=7,stride=7 -> 128`

结论：

- 当前 VTA 内部最突出的坏味道不是单一的大块 DMA，而是大量 `14x14`、`7x7`、带 stride / padding 的 input load。
- 这说明主要问题仍然在 VTA 子图内部的 tile / DMA 形态，而不是单纯的 host-device 边界传输。

### 3. `all_vta` 虽然总时延更短，但 VTA 侧负担明显更重

数据依据：

- `all_vta device_run_wait_ms/run ≈ 82.11`
- `three_stage_e device_run_wait_ms/run ≈ 60.17`
- `all_vta load_buffer_2d_calls/run = 2110`
- `three_stage_e load_buffer_2d_calls/run = 1598`
- `all_vta store_buffer_2d_calls/run = 126`
- `three_stage_e store_buffer_2d_calls/run = 94`
- `all_vta load_buffer_2d_padded_calls/run = 896`
- `three_stage_e load_buffer_2d_padded_calls/run = 656`

结论：

- `all_vta` 的更短总时延是以更高的 VTA 侧运行等待时间和更高的 DMA 次数换来的。
- 这使它更像“纯性能上界参考”，而不是最适合继续做异构 runtime 优化的主线。

### 4. event 结果说明 DMA 才是主问题，不是 enqueue 本身

数据依据：

- `three_stage_e` event kinds:
  - `load_buffer_2d = 2192`
  - `push_gemm_op = 1152`
  - `push_alu_op = 610`
  - `store_buffer_2d = 114`
- `all_vta` event kinds:
  - `load_buffer_2d = 2122`
  - `push_gemm_op = 1124`
  - `push_alu_op = 696`
  - `store_buffer_2d = 128`
- 两边 `padded_ratio` 都在 `0.21 ~ 0.22`

结论：

- 事件窗口里最主要的 event 类型都是 `load_buffer_2d`，说明 DMA 仍然是主要压力来源。
- `push_gemm_op` / `push_alu_op` 虽然数量也不少，但当前数据并不支持把“enqueue 开销”当成第一优先级。

### 5. VTA runtime API 本身的时间占比不大，但静态化仍然有明确上限收益

数据依据：

- `three_stage_e runtime enqueue total ≈ 1.338 ms/run`
  - `load 0.835 ms`
  - `store 0.042 ms`
  - `gemm 0.249 ms`
  - `alu 0.212 ms`
- `three_stage_e stage1_vta run_mean = 68.721 ms`
- `all_vta runtime enqueue total ≈ 1.670 ms/run`
- `all_vta stage0_vta run_mean = 108.658 ms`

结论：

- 单看平均时延，runtime API 组织/下发指令的时间不是主瓶颈。
- 对 `three_stage_e` 来说，静态化 runtime 的理论直接收益上限大约是 `1.338 ms/run`，约等于 `stage1_vta run_mean` 的 `1.9%`。
- 对 `all_vta` 来说，这个上限大约是 `1.670 ms/run`，约等于 `stage0_vta run_mean` 的 `1.5%`。
- 所以如果目标只是压低平均 latency，静态化收益明确但不会是决定性提升；它更适合用来减少每次推理的 runtime 抖动，给后续 pipeline / steady-state frame pacing 打基础。

### 6. 当前最值得尝试的优化方向

按优先级建议：

1. 优先优化 input-side DMA 形态  
   目标是减少 `14x14`、`7x7`、`strided`、`padded` load。

2. 再优化 weight-side 聚合  
   例如减少 `wgt:x=9,y=2` / `wgt:x=9,y=4` 这类碎 weight load 的次数。

3. 继续以 `three_stage_e` 为主线做 runtime / compiler 优化  
   因为它是当前最优 split，同时又比 `all_vta` 更能代表“异构图真实优化空间”。

4. 如果后续要做自动化搜索，cost model 应重点吸收这些特征  
   例如：
   - `load_buffer_2d_small_calls`
   - `load_buffer_2d_strided_calls`
   - `load_buffer_2d_padded_calls`
   - top DMA signatures
   - `device_run_wait_us`
   - runtime enqueue totals

简化成一句话：

> 当前 profile 的核心结论是：`three_stage_e` 仍然是最佳 split 主线，而真正值得继续优化的点不是 host 侧接口，而是 VTA 子图内部仍然很碎的 input-side DMA 形态。
