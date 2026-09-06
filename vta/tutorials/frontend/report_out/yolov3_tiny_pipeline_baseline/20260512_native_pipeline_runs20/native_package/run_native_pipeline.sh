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
  --stage0-graph stages/stage0_all_vta/graph.json \
  --stage0-lib stages/stage0_all_vta/graphlib.so \
  --stage0-params stages/stage0_all_vta/params.params \
  --stage0-input-names data \
  --stage0-name stage0_all_vta \
  --stage0-device vta \
  --stage0-runtime-num-threads 1 \
  --input input.bin \
  --runs 20 \
  --queue-depth 2 \
  --runtime-num-threads 4 \
  --output-mode raw \
  --output-jsonl native_pipeline_result.jsonl
