#!/bin/sh
set -eu
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$PWD/libtvm_runtime.so:$PWD/libvta.so"
export TVM_NUM_THREADS=1
: "${TVM_THREAD_POOL_SPIN_COUNT:=0}"
: "${AXU5EVB_DRIVER_POST_START_SLEEP_NS:=1000}"
: "${AXU5EVB_DRIVER_POLL_SLEEP_NS:=1000}"
export TVM_THREAD_POOL_SPIN_COUNT
export AXU5EVB_DRIVER_POST_START_SLEEP_NS
export AXU5EVB_DRIVER_POLL_SLEEP_NS

exec ./vta_stage_pipeline_runner \
  --stage0-graph stages/cpu_residual_3x3/graph.json \
  --stage0-lib stages/cpu_residual_3x3/graphlib.so \
  --stage0-params stages/cpu_residual_3x3/params.params \
  --stage0-input-names data \
  --stage0-name cpu_residual_3x3 \
  --stage0-device cpu \
  --stage0-runtime-num-threads 1 \
  --stage1-graph stages/vta_conv3x3_c_small/graph.json \
  --stage1-lib stages/vta_conv3x3_c_small/graphlib.so \
  --stage1-params stages/vta_conv3x3_c_small/params.params \
  --stage1-input-names data \
  --stage1-name vta_conv3x3_c_small \
  --stage1-device vta \
  --stage1-runtime-num-threads 1 \
  --input input.bin \
  --runs 8 \
  --queue-depth 1 \
  --runtime-num-threads 1 \
  --output-mode raw_all_stages \
  --output-jsonl stage_serial_result.jsonl \
  --serial \
  --vta-runtime-profile-dir profile/serial \
  --vta-runtime-profile-events-limit 512 \
  --vta-runtime-profile-checkpoint-every 0
