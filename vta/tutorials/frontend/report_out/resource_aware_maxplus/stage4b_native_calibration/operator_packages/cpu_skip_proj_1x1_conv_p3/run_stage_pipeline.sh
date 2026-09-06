#!/bin/sh
set -eu
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$PWD/libtvm_runtime.so:$PWD/libvta.so"
export TVM_NUM_THREADS=4
: "${TVM_THREAD_POOL_SPIN_COUNT:=0}"
: "${AXU5EVB_DRIVER_POST_START_SLEEP_NS:=1000}"
: "${AXU5EVB_DRIVER_POLL_SLEEP_NS:=1000}"
export TVM_THREAD_POOL_SPIN_COUNT
export AXU5EVB_DRIVER_POST_START_SLEEP_NS
export AXU5EVB_DRIVER_POLL_SLEEP_NS

exec ./vta_stage_pipeline_runner \
  --stage0-graph stages/cpu_skip_proj_1x1_conv_p3/graph.json \
  --stage0-lib stages/cpu_skip_proj_1x1_conv_p3/graphlib.so \
  --stage0-params stages/cpu_skip_proj_1x1_conv_p3/params.params \
  --stage0-input-names data \
  --stage0-name cpu_skip_proj_1x1_conv_p3 \
  --stage0-device cpu \
  --stage0-runtime-num-threads 4 \
  --input input.bin \
  --runs 25 \
  --queue-depth 1 \
  --runtime-num-threads 4 \
  --output-mode raw_all_stages \
  --output-jsonl native_result.jsonl
