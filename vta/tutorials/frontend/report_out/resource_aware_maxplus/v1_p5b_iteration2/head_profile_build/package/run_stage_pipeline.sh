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
  --stage0-graph stages/stem/graph.json \
  --stage0-lib stages/stem/graphlib.so \
  --stage0-params stages/stem/params.params \
  --stage0-input-names data0 \
  --stage0-name stem \
  --stage0-device cpu \
  --stage0-runtime-num-threads 1 \
  --stage0-cpu-affinity 0 \
  --stage1-graph stages/vta_body/graph.json \
  --stage1-lib stages/vta_body/graphlib.so \
  --stage1-params stages/vta_body/params.params \
  --stage1-input-names data0 \
  --stage1-name vta_body \
  --stage1-device vta \
  --stage1-runtime-num-threads 1 \
  --stage2-graph stages/regular_block/graph.json \
  --stage2-lib stages/regular_block/graphlib.so \
  --stage2-params stages/regular_block/params.params \
  --stage2-input-names data0 \
  --stage2-name regular_block \
  --stage2-device cpu \
  --stage2-runtime-num-threads 1 \
  --stage2-cpu-affinity 0 \
  --stage3-graph stages/head/graph.json \
  --stage3-lib stages/head/graphlib.so \
  --stage3-params stages/head/params.params \
  --stage3-input-names data0 \
  --stage3-name head \
  --stage3-device cpu \
  --stage3-runtime-num-threads 1 \
  --stage3-cpu-affinity 0 \
  --input input.bin \
  --runs 7 \
  --queue-depth 2 \
  --runtime-num-threads 4 \
  --output-mode raw \
  --output-jsonl native_result.jsonl \
  --output-dump-dir correctness_outputs/pipeline
