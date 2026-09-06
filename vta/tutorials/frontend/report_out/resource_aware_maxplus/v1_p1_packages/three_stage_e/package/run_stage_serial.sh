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
  --stage0-graph stages/stage0_cpu/graph.json \
  --stage0-lib stages/stage0_cpu/graphlib.so \
  --stage0-params stages/stage0_cpu/params.params \
  --stage0-input-names data0 \
  --stage0-name stage0_cpu \
  --stage0-device cpu \
  --stage0-runtime-num-threads 4 \
  --stage0-cpu-affinity 0,1,2,3 \
  --stage1-graph stages/stage1_vta/graph.json \
  --stage1-lib stages/stage1_vta/graphlib.so \
  --stage1-params stages/stage1_vta/params.params \
  --stage1-input-names data0 \
  --stage1-name stage1_vta \
  --stage1-device vta \
  --stage1-runtime-num-threads 1 \
  --stage2-graph stages/stage2_cpu/graph.json \
  --stage2-lib stages/stage2_cpu/graphlib.so \
  --stage2-params stages/stage2_cpu/params.params \
  --stage2-input-names data0 \
  --stage2-name stage2_cpu \
  --stage2-device cpu \
  --stage2-runtime-num-threads 4 \
  --stage2-cpu-affinity 0,1,2,3 \
  --input input.bin \
  --runs 12 \
  --queue-depth 2 \
  --runtime-num-threads 4 \
  --output-mode raw_all_stages \
  --output-jsonl stage_serial_result.jsonl \
  --serial \
  --vta-runtime-profile-dir profile/serial \
  --vta-runtime-profile-events-limit 256 \
  --vta-runtime-profile-checkpoint-every 0
