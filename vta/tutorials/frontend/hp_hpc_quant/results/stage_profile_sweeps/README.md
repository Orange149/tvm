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
- `top load signatures`:
  - `inp:x=14,y=14,stride=14,count=2240`
  - `wgt:x=9,y=2,stride=72,count=1920`
  - `wgt:x=9,y=4,stride=144,count=1920`

这说明当前主要问题仍然是 VTA 子图内部的 input-side DMA 比较碎，典型形态仍然是 `14x14` input load。

`all_vta` 的 VTA runtime 指标：

- `load_buffer_2d_calls/run = 2110`
- `store_buffer_2d_calls/run = 126`
- `device_run_wait_ms/run ≈ 82.11`
- `top load signature`:
  - `wgt:x=9,y=2,stride=36,count=2240`

结论：

- `all_vta` 总时延最短，但 VTA 负载更重，DMA 次数更多
- `three_stage_e` 作为 split 方案最佳，仍值得作为后续 runtime / event / DMA 优化主线

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
