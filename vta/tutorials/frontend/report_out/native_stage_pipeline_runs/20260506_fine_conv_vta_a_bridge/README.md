# 2026-05-06 fine_conv_vta_a boundary bridge native pipeline

## 分支

`fix-vta-bridge`

## 代码改动

- `vta/python/vta/top/graphpack.py`
  - 新增 `graph_pack(..., boundary_bridge=False)` 参数。
  - 默认 `False` 保持原来的 `ExprPack` 路径不变。
  - `boundary_bridge=True` 时使用新的 bridge-aware packer，直接消费调用方显式插入的 `annotation.bitpack_start/end` marker。
  - 为每个 Relay expr 跟踪 layout 状态：普通 4D tensor、packed 6D tensor、tuple fields。
  - 将 `bitpack_start(x)` 降成 4D NCHW 到 packed `NCHWbnc` 的 pack bridge。
  - 将 `bitpack_end(x)` 降成 packed `NCHWbnc` 到逻辑 4D NCHW 的 unpack bridge。
  - 支持 `Tuple` 和 `TupleGetItem` 的 field layout 跟踪；packed tensor 喂给普通 4D consumer 时自动 unpack，普通 4D tensor 喂给 packed VTA conv path 时自动 pack。

- `vta/tutorials/frontend/profile_split_resnet18_stages.py`
  - staged VTA build 路径改成在 VTA stage 边界显式插入 bitpack marker。
  - VTA stage 调用 `graph_pack(..., start_name=None, stop_name=None, boundary_bridge=True)`。
  - 旧的 all-VTA `use_graph_pack=True` 路径保持不变，避免扩大回归面。

- `vta/tests/python/unittest/test_graphpack_boundary_bridge.py`
  - 新增最小 Relay 单测，覆盖 tuple 输出、packed 到普通 4D consumer、普通 4D 输入到 packed conv path。

- `vta/apps/native_deploy/vta_stage_pipeline_runner.cc`
  - 将原来写死的 3 段 `stage0/stage1/stage2` runner 改成动态 `--stageN-*` 参数解析。
  - 每个 stage 支持显式 name、device、input names、runtime thread count。
  - native pipeline 根据 stage 数动态创建 queue/thread 链。
  - 结果 JSON 新增 `stage_count`，并输出每个 `stage{i}` 的 timing 字段。

- `vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py`
  - native stage 校验从固定 `cpu/vta/cpu` 放宽到 `cpu/.../vta/.../cpu`。
  - run script 生成动态 `--stageN-*` 参数，package check 也改成按实际 stage 列表检查。
  - `manifest.json` 记录每个 stage 的 runtime thread 配置。

- `vta/tutorials/frontend/split_resnet18_stages.py`
  - 新增手工 5-stage 方案 `fine_conv_vta_a`：
    - `stage0_cpu`: `stem + layer1`
    - `stage1_vta`: `layer2_block0_main_preadd + layer2_block0_skip_proj`
    - `stage2_cpu`: `layer2_block0_add_relu_tail`
    - `stage3_vta`: `layer2_block1_main_preadd`
    - `stage4_cpu`: `layer2_block1_add_relu_tail + layer3 + layer4 + head`

## 本地验证

graph_pack bridge 单测：

```bash
/home/orange/miniconda3/envs/vta-resnet/bin/python -m pytest \
  vta/tests/python/unittest/test_graphpack_boundary_bridge.py -q
```

结果：

```text
3 passed in 0.03s
```

boundary bridge buildability sweep：

```bash
TEST_DATA_ROOT_PATH=/tmp/tvm_test_data \
MPLCONFIGDIR=/tmp/mpl \
VTA_HW_PATH=/home/orange/code/tvm/3rdparty/vta-hw \
PYTHONUNBUFFERED=1 \
/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/profile_split_resnet18_stages.py \
  --scheme auto_resource_aware \
  --vta-stage-mode vta_build \
  --resource-aware-buildability-sweep \
  --resource-aware-buildability-max-items 8 \
  --resource-aware-top-k 8
```

结果：

```text
15/15 candidates buildable
tuple_output_partial: 6/6 buildable
未再出现 4D/6D shape mismatch。
```

新 5-stage 方案 native package-only 验证：

```bash
unset LD_LIBRARY_PATH
source /home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux

TEST_DATA_ROOT_PATH=/tmp/tvm_test_data \
MPLCONFIGDIR=/tmp/mpl \
PYTHONUNBUFFERED=1 \
VTA_HW_PATH=/home/orange/code/tvm/3rdparty/vta-hw \
/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py \
  --scheme fine_conv_vta_a \
  --package-only \
  --keep-build-dir \
  --build-dir /tmp/fine_conv_vta_a_pkg \
  --runs 1 \
  --queue-depth 1 \
  --runtime-num-threads 4 \
  --stage0-runtime-num-threads 3 \
  --stage1-runtime-num-threads 1 \
  --stage2-runtime-num-threads 1
```

结果：

```text
stage0_cpu -> CPU build ok
stage1_vta -> VTA build ok
stage2_cpu -> CPU build ok
stage3_vta -> VTA build ok
stage4_cpu -> CPU build ok
native runner cross-compile ok
package: /tmp/fine_conv_vta_a_pkg/package.tar.gz
```

旧 3-stage 方案兼容性验证：

```bash
unset LD_LIBRARY_PATH
source /home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux

TEST_DATA_ROOT_PATH=/tmp/tvm_test_data \
MPLCONFIGDIR=/tmp/mpl \
PYTHONUNBUFFERED=1 \
VTA_HW_PATH=/home/orange/code/tvm/3rdparty/vta-hw \
/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py \
  --scheme three_stage_a \
  --package-only \
  --build-dir /tmp/three_stage_a_nstage_runner_pkg \
  --runs 1 \
  --queue-depth 1 \
  --runtime-num-threads 4 \
  --stage0-runtime-num-threads 3 \
  --stage1-runtime-num-threads 1 \
  --stage2-runtime-num-threads 1
```

结果：

```text
three_stage_a package-only build passed with the dynamic N-stage runner.
```

## 上板实验

运行命令：

```bash
unset LD_LIBRARY_PATH
source /home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux

TEST_DATA_ROOT_PATH=/tmp/tvm_test_data \
MPLCONFIGDIR=/tmp/mpl \
PYTHONUNBUFFERED=1 \
VTA_HW_PATH=/home/orange/code/tvm/3rdparty/vta-hw \
/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py \
  --board root@192.168.1.133 \
  --remote-dir /mnt/sd/vta_stage_pipeline_fine_conv_vta_a \
  --scheme fine_conv_vta_a \
  --runs 10 \
  --queue-depth 2 \
  --runtime-num-threads 4 \
  --stage0-runtime-num-threads 3 \
  --stage1-runtime-num-threads 1 \
  --stage2-runtime-num-threads 1 \
  --run-serial-before-pipeline \
  --compare-serial-pipeline \
  --fetch-results-dir /tmp/fine_conv_vta_a_board \
  --vta-runtime-profile-dir profile \
  --vta-runtime-profile-events-limit 80 \
  --ssh-option HostKeyAlgorithms=+ssh-rsa \
  --ssh-option PubkeyAcceptedAlgorithms=+ssh-rsa
```

结果文件拉回到 `/tmp/fine_conv_vta_a_board`：

```text
manifest.json
native_result.jsonl
stage_serial_result.jsonl
profile/pipeline/benchmark_totals_status.json
profile/serial/benchmark_totals_status.json
```

功能正确性结果：

```text
10/10 pipeline frames top1 = 285
serial/pipeline top1 comparison passed
```

板上 stage timing：

```text
Pipeline averages:
  total_latency_ms avg = 1536.675 ms
  stage0_cpu avg      = 138.371 ms
  stage1_vta avg      = 18.713 ms
  stage2_cpu avg      = 1.316 ms
  stage3_vta avg      = 18.119 ms
  stage4_cpu avg      = 292.172 ms
  completion gap avg  = 290.467 ms/frame

Serial averages:
  total_latency_ms avg = 247.050 ms
  stage0_cpu avg      = 125.288 ms
  stage1_vta avg      = 17.971 ms
  stage2_cpu avg      = 0.995 ms
  stage3_vta avg      = 17.182 ms
  stage4_cpu avg      = 85.338 ms
```

VTA runtime profile 摘要：

```text
driver_run_calls = 50
driver_timeout_calls = 0
mem_copy_from_host_calls = 20
mem_copy_to_host_calls = 40
driver_run_total_us ~= 203293 us
device_run_wait_us ~= 204093 us
```

## 结果解读

- 更细粒度的 5-stage native pipeline 已经能在板子上跑通。
- 显式 graph_pack boundary bridge 可以支撑 tuple-output VTA 子图，以及 `CPU -> VTA -> CPU -> VTA -> CPU` 这种分段方式。
- 这次手工切法不是一个好的性能点：
  - pipeline 稳态吞吐约 `290 ms/frame`，瓶颈是 `stage4_cpu`。
  - `stage4_cpu` 在 pipeline 模式下默认只用了 1 个 runtime thread，所以比 serial 中的 `stage4_cpu` 慢很多。
  - 当前切法把 `layer3 + layer4 + head` 都留在 CPU，CPU tail 计算量太大。
- 边界拷贝有开销，但不是这次的主要瓶颈：
  - Host->device 加 device->host 拷贝总时间约 `94.4 ms / 10 frames ~= 9.4 ms/frame`。

## 后续建议

- 增加通用的 stage thread CLI，例如 `--stage-runtime-num-threads 3,1,1,1,4`，让 `stage3/stage4` 这类更多 stage 也能显式配置线程数。
- 继续尝试更细的切法，把 `layer3/layer4` 的重 conv 区域也交给 VTA，只把 add/relu tail 这类小算子留给 CPU。
