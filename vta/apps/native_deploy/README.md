# VTA ResNet18 Native Stage Pipeline 部署说明

这条路径用于在 AXU5EVB 上运行 ResNet18 的三段 native pipeline：

```text
stage0_cpu -> stage1_vta -> stage2_cpu
```

它不启动、不连接、不依赖 VTA RPC server。correctness baseline 仍然使用现有 RPC `all_vta` 版本；这个 native stage pipeline 自带 `--serial` 模式，只用于确认三段拆分和板端 native runner 没有发生输入/输出串扰。

当前 v1 仍然使用 AXU5EVB driver 的 `sleep + poll` 等待方式。区别是 VTA 等待只阻塞 stage1 的调度线程，stage0/stage2 的 CPU 调度线程可以继续处理其他帧。

CPU runtime threadpool 是 thread-local 的：stage0 和 stage2 在不同外层线程里运行时，会各自拥有自己的 TVM threadpool。因此 pipeline 默认运行时线程分配为 `stage0=3, stage1=1, stage2=1`，避免两个 CPU stage 同时各拉 4 个 worker 抢 4 核。串行 sanity check 默认仍使用 `stage0=4, stage1=1, stage2=4`。

## 1. 文件职责与提交范围

这次工作其实有两条不同的 pipeline 路线，需要在提交时分清：

```text
GraphExecutor runtime pipeline:
  让一个 graph executor module 自身支持多 slot submit/wait/poll。

Native stage pipeline 实验:
  把 ResNet18 手动拆成 CPU -> VTA -> CPU 三段，用独立 native runner 在板端并行跑多帧。
```

核心 runtime pipeline 文件：

| 文件 | 内容 | 是否属于“让 runtime pipeline 起来” |
|---|---|---|
| `src/runtime/graph_executor/graph_executor.cc` | 给 `GraphExecutor` 增加实验性 pipeline API 和实现：`pipeline_init`、`pipeline_submit`、`pipeline_try_submit`、`pipeline_wait`、`pipeline_poll`、`pipeline_stats`、`pipeline_close`；内部维护多个 executor slot、请求队列、worker thread、状态和错误传播。 | 是 |
| `src/runtime/graph_executor/graph_executor.h` | 声明 pipeline slot/status、队列、mutex/condition variable、worker thread、保存 graph json/params 等状态。 | 是 |
| `tests/python/unittest/test_runtime_graph.py` | 增加 CPU graph executor pipeline 单元测试，覆盖 submit/wait/poll/stats/try_submit/close 的基本语义。 | 是 |
| `vta/tutorials/frontend/deploy_classification_native.py` | 构建一个完整 VTA ResNet graph，交叉编译 `vta_native_runner` 并部署到板端；`--pipeline` 开关用于调用 GraphExecutor runtime pipeline。 | 是，作为板端验证入口 |
| `vta/apps/native_deploy/vta_native_runner.cc` | `deploy_classification_native.py` 使用的板端 C++ runner，运行完整 graph；在 `--pipeline` 模式下走 GraphExecutor pipeline API。 | 是，作为 native runtime pipeline runner |

Native stage pipeline 实验文件：

| 文件 | 内容 | 是否属于 runtime pipeline 核心 |
|---|---|---|
| `vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py` | Host 侧把 ResNet18 按 scheme 拆成三段、分别 build、打包、scp、ssh 运行，并拉回结果。 | 否，实验脚本 |
| `vta/apps/native_deploy/vta_stage_pipeline_runner.cc` | 板端三段 runner，手动调度 `stage0_cpu -> stage1_vta -> stage2_cpu`，输出 `native_result.jsonl`。 | 否，实验 runner |
| `vta/tutorials/frontend/split_resnet18_stages.py` | 定义 ResNet18 拆分 scheme；当前包含主线 `three_stage_e` 和保留的对照 `three_stage_f`。 | 否，实验拆图工具 |
| `vta/tutorials/frontend/sweep_stage_pipeline_poll_sleep.py` | 批量搜索 AXU5EVB driver `POST_START_SLEEP_NS/POLL_SLEEP_NS` 对三段 pipeline 吞吐的影响，并归档结果。 | 否，实验脚本 |
| `vta/tutorials/frontend/report_out/native_stage_pipeline_runs/` | 本轮 AXU5EVB 实验结果归档，包括 README、summary、manifest、jsonl 和部分 profile。 | 否，实验数据 |
| `vta/apps/native_deploy/README.md` | 本文件，记录部署方法、数据、结论、复现命令和后续优化方向。 | 否，实验记录 |

辅助 profiling 文件：

| 文件 | 内容 | 是否建议混入同一提交 |
|---|---|---|
| `vta/tutorials/frontend/hp_hpc_quant/measure_vta_coherence_throughput.py` | 测 VTA DMA/cache coherence 开销和吞吐，用于解释 driver/runtime profile 中 flush/invalidate 等成本。 | 可以单独提交；如果本次 commit 主题是“AXU5EVB pipeline 性能分析”，可以一起放入 |
| `3rdparty/vta-hw` | VTA HW/driver submodule 指针更新。当前主仓库只看到 submodule commit 从 `c339ef5...` 移到 `2fc44f...`。 | 只有当该 submodule commit 已推送且确实包含本实验依赖的 driver/profiler 改动时才一起提交 |

因此提交可以拆成：

```text
Commit A: GraphExecutor runtime pipeline
  src/runtime/graph_executor/graph_executor.cc
  src/runtime/graph_executor/graph_executor.h
  tests/python/unittest/test_runtime_graph.py
  vta/apps/native_deploy/vta_native_runner.cc
  vta/tutorials/frontend/deploy_classification_native.py

Commit B: Native VTA stage pipeline experiments
  vta/apps/native_deploy/vta_stage_pipeline_runner.cc
  vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py
  vta/tutorials/frontend/sweep_stage_pipeline_poll_sleep.py
  vta/tutorials/frontend/split_resnet18_stages.py
  vta/tutorials/frontend/report_out/native_stage_pipeline_runs/
  vta/apps/native_deploy/README.md

Commit C: VTA coherence/profiler helper, if kept
  vta/tutorials/frontend/hp_hpc_quant/measure_vta_coherence_throughput.py
  3rdparty/vta-hw
```

## 2. Host 环境准备

在 WSL/Linux 的 TVM repo 根目录执行：

```bash
cd /home/orange/code/tvm
unset LD_LIBRARY_PATH
source /home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux
export TEST_DATA_ROOT_PATH=/tmp/tvm_test_data
export MPLCONFIGDIR=/tmp/mpl
export PYTHONUNBUFFERED=1
```

确认交叉编译器、sysroot 和板端 runtime 库存在：

```bash
which aarch64-xilinx-linux-g++
echo "$SDKTARGETSYSROOT"
ls build_axu_aarch64/libtvm_runtime.so build_axu_aarch64/libvta.so
```

如果你修改过 runtime 或 VTA driver，需要先重新构建：

```bash
ninja -C build_axu_aarch64 tvm_runtime vta
```

## 3. 只打包，不上板

你在实验室外面时，可以先生成部署包：

```bash
BUILD_DIR=/tmp/vta_stage_pipeline_pkg
IMG_DIR=/mnt/c/path/to/images

/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py \
  --package-only \
  --build-dir "$BUILD_DIR" \
  --image-dir "$IMG_DIR" \
  --max-images 20 \
  --runs 20 \
  --scheme three_stage_e \
  --queue-depth 2 \
  --runtime-num-threads 4 \
  --run-serial-before-pipeline \
  --compare-serial-pipeline \
  --rpc-baseline-result /tmp/rpc_all_vta_result.jsonl \
  --vta-runtime-profile-dir profile \
  --keep-build-dir
```

部署包会生成在：

```text
$BUILD_DIR/package
$BUILD_DIR/package.tar.gz
```

包内关键文件：

```text
vta_stage_pipeline_runner
stages/stage0_cpu/graph.json
stages/stage0_cpu/graphlib.so
stages/stage0_cpu/params.params
stages/stage1_vta/graph.json
stages/stage1_vta/graphlib.so
stages/stage1_vta/params.params
stages/stage2_cpu/graph.json
stages/stage2_cpu/graphlib.so
stages/stage2_cpu/params.params
inputs/input_000000.bin
inputs/input_000001.bin
inputs.txt
manifest.json
libtvm_runtime.so
libvta.so
run_stage_serial.sh
run_stage_pipeline.sh
```

## 4. 一条命令编译、拷贝、运行

在能访问开发板时：

```bash
BOARD=root@192.168.1.247
REMOTE_DIR=/mnt/sd/vta_stage_pipeline
IMG_DIR=/mnt/c/path/to/images
OUT_DIR=/tmp/vta_stage_pipeline_results

/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py \
  --board "$BOARD" \
  --remote-dir "$REMOTE_DIR" \
  --image-dir "$IMG_DIR" \
  --max-images 20 \
  --runs 20 \
  --scheme three_stage_e \
  --queue-depth 2 \
  --runtime-num-threads 4 \
  --run-serial-before-pipeline \
  --compare-serial-pipeline \
  --rpc-baseline-result /tmp/rpc_all_vta_result.jsonl \
  --vta-runtime-profile-dir profile \
  --vta-runtime-profile-events-limit 200 \
  --fetch-results-dir "$OUT_DIR" \
  --keep-build-dir
```

脚本会完成：

```text
host 编译三段 graph
host 预处理多张图片
host 交叉编译 vta_stage_pipeline_runner
scp package.tar.gz 到开发板
ssh 先执行 run_stage_serial.sh，再执行 run_stage_pipeline.sh
可选 scp 拉回 stage_serial_result.jsonl、native_result.jsonl 和 profile/
如果提供 `--rpc-baseline-result`，host 会再比较 RPC all_vta baseline 与 native serial 的 `input_index/top1`
```

## 5. 手动 SCP 和板端运行

如果已经用 `--package-only` 生成了包：

```bash
BOARD=root@192.168.1.247
REMOTE_DIR=/mnt/sd/vta_stage_pipeline
TAR=/tmp/vta_stage_pipeline_pkg/package.tar.gz

scp "$TAR" "$BOARD:/tmp/package.tar.gz"
ssh "$BOARD" "rm -rf '$REMOTE_DIR' && mkdir -p '$REMOTE_DIR' && tar -xzf /tmp/package.tar.gz -C '$REMOTE_DIR'"
ssh "$BOARD" "cd '$REMOTE_DIR' && ./run_stage_serial.sh"
ssh "$BOARD" "cd '$REMOTE_DIR' && ./run_stage_pipeline.sh"
```

拉回结果：

```bash
mkdir -p /tmp/vta_stage_pipeline_results
scp "$BOARD:$REMOTE_DIR/native_result.jsonl" /tmp/vta_stage_pipeline_results/
scp "$BOARD:$REMOTE_DIR/stage_serial_result.jsonl" /tmp/vta_stage_pipeline_results/
scp "$BOARD:$REMOTE_DIR/manifest.json" /tmp/vta_stage_pipeline_results/
scp -r "$BOARD:$REMOTE_DIR/profile" /tmp/vta_stage_pipeline_results/
```

## 6. 板端实际执行命令

`run_stage_pipeline.sh` 会设置：

```sh
export LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$PWD/libtvm_runtime.so:$PWD/libvta.so"
export TVM_NUM_THREADS=4
export TVM_THREAD_POOL_SPIN_COUNT=0
export AXU5EVB_DRIVER_POST_START_SLEEP_NS=1000
export AXU5EVB_DRIVER_POLL_SLEEP_NS=1000
```

然后执行：

```sh
./vta_stage_pipeline_runner \
  --stage0-graph stages/stage0_cpu/graph.json \
  --stage0-lib stages/stage0_cpu/graphlib.so \
  --stage0-params stages/stage0_cpu/params.params \
  --stage0-input-names data0 \
  --stage1-graph stages/stage1_vta/graph.json \
  --stage1-lib stages/stage1_vta/graphlib.so \
  --stage1-params stages/stage1_vta/params.params \
  --stage1-input-names data0 \
  --stage2-graph stages/stage2_cpu/graph.json \
  --stage2-lib stages/stage2_cpu/graphlib.so \
  --stage2-params stages/stage2_cpu/params.params \
  --stage2-input-names data0 \
  --input-list inputs.txt \
  --runs 20 \
  --queue-depth 2 \
  --runtime-num-threads 4 \
  --stage0-runtime-num-threads 3 \
  --stage1-runtime-num-threads 1 \
  --stage2-runtime-num-threads 1 \
  --output-jsonl native_result.jsonl \
  --vta-runtime-profile-dir profile/pipeline
```

`run_stage_serial.sh` 会使用同一套 stage module，但加 `--serial`，输出 `stage_serial_result.jsonl`，profiler 写到 `profile/serial`。`run_stage_pipeline.sh` 的 profiler 写到 `profile/pipeline`。pipeline 模式只 dump `benchmark_totals_*` profiler 文件；serial 模式才会生成 `single_run_*` 和 checkpoint profiler 文件。

## 7. 结果检查

板端运行后检查：

```bash
cd /mnt/sd/vta_stage_pipeline
wc -l native_result.jsonl
wc -l stage_serial_result.jsonl
head -n 3 native_result.jsonl
ls profile
```

`native_result.jsonl` 每行对应一帧，包含：

```text
frame_id
input_index
input_file
top1
stage0_ms / stage1_ms / stage2_ms
stage0_start_ms / stage0_end_ms
stage1_start_ms / stage1_end_ms
stage2_start_ms / stage2_end_ms
total_latency_ms
```

验证 overlap 时，看不同 `frame_id` 的时间区间是否交叠。例如：

```text
frame N   : stage1_start_ms ... stage1_end_ms
frame N+1 : stage0_start_ms ... stage0_end_ms
frame N-1 : stage2_start_ms ... stage2_end_ms
```

如果这些区间重叠，说明 CPU/VTA/CPU 三段已经在不同帧之间并行推进。

## 8. 当前实验数据摘要

以下数据来自 AXU5EVB，`three_stage_e`，默认 `input.bin` 重复提交，`runs=20`，统计时跳过前 3 帧。所有列出的 native serial/pipeline 对比都满足 `top1` 一致。

### 8.1 当前最佳 pipeline 形态

当前最稳定的配置仍是：

```text
stage0_runtime_num_threads = 3
stage1_runtime_num_threads = 1
stage2_runtime_num_threads = 1
AXU5EVB_DRIVER_POST_START_SLEEP_NS = 1000
AXU5EVB_DRIVER_POLL_SLEEP_NS = 1000
```

对应结果：

```text
stage0 avg                 89.51 ms
stage1 avg                 72.76 ms
stage2 avg                 72.00 ms
stage0 start interval avg  89.56 ms
finite-window throughput   10.24 fps
steady-state upper bound   1000 / 89.56 ~= 11.17 fps
```

`10.24 fps` 是 `runs=20` 的有限窗口实际吞吐，包含 pipeline 填充和排空成本。`11.17 fps` 是按稳态瓶颈 stage0 启动间隔推导的上限；更长 `runs` 且只看中间稳态时才可能接近这个值。

Pipeline 时间关系可以理解为：

```text
time --->
frame0: [ stage0 CPU 89.6 ][ stage1 VTA 72.8 ][ stage2 CPU 72.0 ]
frame1:                 [ stage0 CPU 89.6 ][ stage1 VTA 72.8 ][ stage2 CPU 72.0 ]
frame2:                                  [ stage0 CPU 89.6 ][ stage1 VTA 72.8 ][ stage2 CPU 72.0 ]
frame3:                                                   [ stage0 CPU 89.6 ][ stage1 VTA 72.8 ][ stage2 CPU 72.0 ]
```

stage0 比 stage1/stage2 慢，因此当前 admission bottleneck 是 stage0，后一帧进入 pipeline 的节奏主要由 `stage0_start_interval_ms` 决定。

### 8.2 CPU 线程分配实验

| 配置 stage0/stage1/stage2 | Pipeline fps | stage0 ms | stage1 ms | stage2 ms | stage0 interval ms | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `3/1/1` | `10.24` | `89.51` | `72.76` | `72.00` | `89.56` | 当前最佳 |
| `2/1/1` | `9.53` | `96.77` | `72.94` | `70.82` | `96.80` | stage0 线程不足 |
| `4/1/1` | `8.69` | `85.90` | `74.34` | `102.13` | `90.79` | stage0 抢 CPU，stage2 被拖慢 |
| `3/1/2` | `8.23` | `115.00` | `73.13` | `61.35` | `115.34` | stage2 变快但 stage0 严重变慢 |

结论：stage0/stage2 的 TVM CPU threadpool 竞争是主要限制。简单给 stage0 或 stage2 加线程都会破坏另一端，`3/1/1` 是目前最好的折中。

### 8.3 Stage split 实验

`three_stage_e` 当前作为 native stage pipeline 主线：

```text
stage0_cpu : stem + layer1_block0
stage1_vta : layer1_block1 + layer2 + layer3 + layer4_block0
stage2_cpu : layer4_block1 + head
```

尝试过的更激进切分是 `three_stage_f`：

```text
stage0_cpu : stem
stage1_vta : layer1 + layer2 + layer3 + layer4_block0
stage2_cpu : layer4_block1 + head
```

实测结果：

| Scheme | Pipeline fps | stage0 ms | stage1 ms | stage2 ms | stage0 interval ms | top1 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `three_stage_e` | `10.24` | `89.51` | `72.76` | `72.00` | `89.56` | `285` | 当前最佳 |
| `three_stage_f` | `8.80` | `36.81` | `98.79` | `69.47` | `90.46` | `282` | stage0 降低，但 stage1 过重且 top1 改变 |

结论：

```text
three_stage_f 证明“减轻 stage0”这个方向有效，但把整个 layer1_block0 放进 VTA 太激进。
stage1 从 72.76 ms 增加到 98.79 ms，吞吐从 10.24 fps 掉到 8.80 fps。
top1 从 three_stage_e 的 285 变成 282；serial/pipeline 自身一致，但仍需要 RPC all_vta baseline 判断 correctness。
```

还尝试过更细的候选：`stage0_cpu = stem + layer1_block0_main_preadd`，`stage1_vta` 从 `layer1_block0_add_relu_tail` 开始。该方向已经从代码里回退，因为当前 VTA `graph_pack` 不支持 VTA stage 以这种 tuple residual tail 输入作为入口，会触发：

```text
vta/python/vta/top/graphpack.py:326
assert not self.start_pack
```

因此三段 runner 下暂不继续测试这种细粒度切法；后续要么改 `graph_pack` 支持该入口，要么升级为更多 stage 的 pipeline。

### 8.4 Poll sleep 实验

固定 `3/1/1`，只改变：

```text
AXU5EVB_DRIVER_POST_START_SLEEP_NS
AXU5EVB_DRIVER_POLL_SLEEP_NS
```

| Poll sleep | Pipeline fps | stage0 ms | stage1 ms | stage2 ms | stage0 interval ms |
|---:|---:|---:|---:|---:|---:|
| `0.5us` | `10.199` | `89.87` | `72.74` | `72.39` | `89.88` |
| `1us repeat` | `10.199` | `89.88` | `72.73` | `72.00` | `89.86` |
| `2us` | `10.177` | `90.06` | `72.89` | `72.17` | `90.24` |
| `5us` | `10.200` | `89.81` | `72.75` | `72.17` | `89.74` |
| `10us` | `10.158` | `90.25` | `74.88` | `80.90` | `89.76` |
| `20us` | `9.789` | `93.18` | `73.85` | `72.16` | `94.86` |
| `50us` | `10.205` | `89.77` | `72.68` | `72.06` | `89.79` |
| `100us` | `9.937` | `92.34` | `73.04` | `72.00` | `93.58` |
| `1ms` | `9.022` | `102.61` | `72.62` | `72.04` | `103.41` |

结论：

```text
0.5us / 1us / 5us / 50us 基本同档，差距小于当前 runs=20 的测量噪声。
20us / 100us / 1ms 明显更差，不应作为默认。
暂时保留 1us 默认最稳妥；如果要改默认，需要用 runs=100 复测 1us、5us、50us。
```

### 8.5 当前可优化方向

优先级从高到低：

1. Correctness baseline 先补强。`three_stage_f` 的 native serial/pipeline 自身一致，但 top1 从 `285` 变成 `282`，不能直接当优化成功；后续每个新切分都需要和 RPC `all_vta` baseline 对齐，而不只是 serial/pipeline 互相比。
2. 用更长 runs 确认 poll sleep。`0.5us/1us/5us/50us` 在 `runs=20` 下基本同档，差距小于测量噪声；如果要改默认值，先对 `1us/5us/50us` 做 `runs=100` 或 `runs=200` 复测。
3. 继续找比 `three_stage_e` 更平衡的 stage split。方向是降低 stage0，但不能把 VTA stage 拉到 90ms 以上；`three_stage_f` 已证明把整个 `layer1_block0` 放进 VTA 太重。更细的 residual-tail 入口当前被 VTA `graph_pack` 限制，除非改 graph_pack 或升级到五段 pipeline，否则三段方案要优先在 block/layer 边界上找。
4. 跑真实多图输入。当前大部分数据是默认 `input.bin` 重复提交，适合测 pipeline 调度和吞吐，但不覆盖前处理后的多图输入稳定性；正式结论需要带 `--image-dir` 跑一轮。
5. 控制 CPU 调度干扰。当前 stage0/stage2 都会使用各自 thread-local TVM threadpool，`4/1/1` 和 `3/1/2` 已经说明 CPU 抢占会显著拖慢另一端。后续可以试 CPU affinity、减少外部负载，或固定 stage0/stage2 的核心分配。
6. 自动化 timeline/backlog 分析。现在靠 `native_result.jsonl` 手动看 stage start/end；建议加脚本输出 Gantt、stage0 interval、stage2 backlog 和稳态窗口 fps，避免每轮人工判断。
7. 研究中间 tensor 拷贝和 buffer 复用。当前三段之间通过 NDArray 交接，主要瓶颈仍是 stage0 compute，但如果后续 stage split 更平衡，减少中间分配/拷贝会变得更重要。

暂时不建议继续投入的方向：

```text
poll sleep 放大到 20us/100us/1ms：已有数据明显变差。
three_stage_f 直接作为优化结果：吞吐下降且 top1 改变。
简单增加 CPU runtime threads：4/1/1 和 3/1/2 都已经劣化。
```

## 9. 实验复现命令与数据位置

所有命令默认在 repo 根目录 `/home/orange/code/tvm` 下执行：

```bash
unset LD_LIBRARY_PATH
source /home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux
export TEST_DATA_ROOT_PATH=/tmp/tvm_test_data
export MPLCONFIGDIR=/tmp/mpl
export PYTHONUNBUFFERED=1
```

当前主线 baseline `three_stage_e + 3/1/1 + 1us poll`：

```bash
OUT_DIR=/tmp/vta_stage_pipeline_results_three_stage_e_3_1_1

AXU5EVB_DRIVER_POST_START_SLEEP_NS=1000 \
AXU5EVB_DRIVER_POLL_SLEEP_NS=1000 \
/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py \
  --board root@192.168.1.133 \
  --ssh-option HostKeyAlgorithms=+ssh-rsa \
  --ssh-option PubkeyAcceptedAlgorithms=+ssh-rsa \
  --remote-dir /mnt/sd/vta_stage_pipeline \
  --scheme three_stage_e \
  --runs 20 \
  --queue-depth 2 \
  --runtime-num-threads 4 \
  --stage0-runtime-num-threads 3 \
  --stage1-runtime-num-threads 1 \
  --stage2-runtime-num-threads 1 \
  --run-serial-before-pipeline \
  --compare-serial-pipeline \
  --fetch-results-dir "$OUT_DIR"
```

线程分配实验只改这三个参数和 `OUT_DIR`：

```text
--stage0-runtime-num-threads 2 --stage1-runtime-num-threads 1 --stage2-runtime-num-threads 1
--stage0-runtime-num-threads 4 --stage1-runtime-num-threads 1 --stage2-runtime-num-threads 1
--stage0-runtime-num-threads 3 --stage1-runtime-num-threads 1 --stage2-runtime-num-threads 2
```

`three_stage_f` 对照实验：

```bash
OUT_DIR=/tmp/vta_stage_pipeline_results_three_stage_f_3_1_1

AXU5EVB_DRIVER_POST_START_SLEEP_NS=1000 \
AXU5EVB_DRIVER_POLL_SLEEP_NS=1000 \
/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py \
  --board root@192.168.1.133 \
  --ssh-option HostKeyAlgorithms=+ssh-rsa \
  --ssh-option PubkeyAcceptedAlgorithms=+ssh-rsa \
  --remote-dir /mnt/sd/vta_stage_pipeline \
  --scheme three_stage_f \
  --runs 20 \
  --queue-depth 2 \
  --runtime-num-threads 4 \
  --stage0-runtime-num-threads 3 \
  --stage1-runtime-num-threads 1 \
  --stage2-runtime-num-threads 1 \
  --run-serial-before-pipeline \
  --compare-serial-pipeline \
  --fetch-results-dir "$OUT_DIR"
```

更长 poll confirmation 可以用 sweep 脚本限制候选值：

```bash
/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/sweep_stage_pipeline_poll_sleep.py \
  --board root@192.168.1.133 \
  --ssh-option HostKeyAlgorithms=+ssh-rsa \
  --ssh-option PubkeyAcceptedAlgorithms=+ssh-rsa \
  --remote-dir /mnt/sd/vta_stage_pipeline \
  --runs 100 \
  --queue-depth 2 \
  --runtime-num-threads 4 \
  --stage0-runtime-num-threads 3 \
  --stage1-runtime-num-threads 1 \
  --stage2-runtime-num-threads 1 \
  --poll-value poll_1us_confirm:1000 \
  --poll-value poll_5us_confirm:5000 \
  --poll-value poll_50us_confirm:50000
```

原始拉回结果在 `/tmp/vta_stage_pipeline_results*`。长期归档在：

```text
vta/tutorials/frontend/report_out/native_stage_pipeline_runs/
vta/tutorials/frontend/report_out/native_stage_pipeline_runs/poll_sleep_sweeps/
```

每个归档目录通常包含：

```text
README.md
run_command.sh
summary.json
manifest.json
native_result.jsonl
stage_serial_result.jsonl
profile/
```

当前已经归档的关键实验：

```text
20260429_cat_three_stage_e_runs20_default_threads
20260429_cat_three_stage_e_runs20_threads_2_1_1
20260429_cat_three_stage_e_runs20_threads_3_1_2
20260429_cat_three_stage_e_runs20_threads_4_1_1
20260429_cat_three_stage_f_runs20_threads_3_1_1
20260429_cat_three_stage_e_runs20_threads_3_1_1_poll_0p5us
20260429_cat_three_stage_e_runs20_threads_3_1_1_poll_1us_repeat
20260429_cat_three_stage_e_runs20_threads_3_1_1_poll_2us
20260429_cat_three_stage_e_runs20_threads_3_1_1_poll_5us
20260429_cat_three_stage_e_runs20_threads_3_1_1_poll_10us
20260429_cat_three_stage_e_runs20_threads_3_1_1_poll_20us
20260429_cat_three_stage_e_runs20_threads_3_1_1_poll_50us
20260429_cat_three_stage_e_runs20_threads_3_1_1_poll_100us
20260429_cat_three_stage_e_runs20_threads_3_1_1_poll_1ms
poll_sleep_sweeps/20260429_cat_three_stage_e_runs20_threads_3_1_1_poll_sweep
```

看聚合摘要：

```bash
cat vta/tutorials/frontend/report_out/native_stage_pipeline_runs/*/README.md
cat vta/tutorials/frontend/report_out/native_stage_pipeline_runs/poll_sleep_sweeps/*/summary.csv
```

## 10. Host 静态检查

```bash
python3 -m py_compile vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py
python3 -m py_compile vta/tutorials/frontend/sweep_stage_pipeline_poll_sleep.py

g++ -std=c++17 -fsyntax-only \
  -I include \
  -I 3rdparty/dlpack/include \
  -I 3rdparty/dmlc-core/include \
  -I 3rdparty/vta-hw/include \
  -I vta/include \
  vta/apps/native_deploy/vta_stage_pipeline_runner.cc
```

确认新的 native stage 路径没有 RPC 调用：

```bash
rg "rpc\\.connect|remote\\.upload|remote\\.load_module|request_remote" \
  vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py \
  vta/apps/native_deploy/vta_stage_pipeline_runner.cc
```

## 11. Poll sleep sweep

当前默认 AXU5EVB driver 等待参数是：

```sh
AXU5EVB_DRIVER_POST_START_SLEEP_NS=1000
AXU5EVB_DRIVER_POLL_SLEEP_NS=1000
```

也就是 `1us`。如果要系统搜索 `0.5us ~ 100us`，使用：

```bash
source /home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux

/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/sweep_stage_pipeline_poll_sleep.py \
  --board root@192.168.1.133 \
  --ssh-option HostKeyAlgorithms=+ssh-rsa \
  --ssh-option PubkeyAcceptedAlgorithms=+ssh-rsa \
  --remote-dir /mnt/sd/vta_stage_pipeline \
  --runs 20 \
  --queue-depth 2 \
  --runtime-num-threads 4 \
  --stage0-runtime-num-threads 3 \
  --stage1-runtime-num-threads 1 \
  --stage2-runtime-num-threads 1
```

默认会依次测试：

```text
poll_0p5us:500
poll_1us_repeat:1000
poll_2us:2000
poll_5us:5000
poll_20us:20000
poll_50us:50000
poll_100us:100000
```

每组都会执行 serial + pipeline，并比较 top1。结果会拉回 `/tmp/vta_stage_pipeline_results_3_1_1_<label>`，同时归档到：

```text
vta/tutorials/frontend/report_out/native_stage_pipeline_runs/<run_label>
vta/tutorials/frontend/report_out/native_stage_pipeline_runs/poll_sleep_sweeps/<sweep_label>
```

聚合结果看：

```bash
cat vta/tutorials/frontend/report_out/native_stage_pipeline_runs/poll_sleep_sweeps/*/README.md
cat vta/tutorials/frontend/report_out/native_stage_pipeline_runs/poll_sleep_sweeps/*/summary.csv
```
