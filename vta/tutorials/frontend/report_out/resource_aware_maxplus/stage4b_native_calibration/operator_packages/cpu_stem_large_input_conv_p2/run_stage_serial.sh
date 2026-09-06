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
  --stage0-graph stages/cpu_stem_large_input_conv_p2/graph.json \
  --stage0-lib stages/cpu_stem_large_input_conv_p2/graphlib.so \
  --stage0-params stages/cpu_stem_large_input_conv_p2/params.params \
  --stage0-input-names data \
  --stage0-name cpu_stem_large_input_conv_p2 \
  --stage0-device cpu \
  --stage0-runtime-num-threads 4 \
  --input input.bin \
  --runs 25 \
  --queue-depth 1 \
  --runtime-num-threads 4 \
  --output-mode raw_all_stages \
  --output-jsonl stage_serial_result.jsonl \
  --serial
