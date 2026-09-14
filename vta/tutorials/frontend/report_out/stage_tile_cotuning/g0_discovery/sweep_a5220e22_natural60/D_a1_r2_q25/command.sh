#!/bin/sh
set -eu
cd /media/sd-mmcblk1p2/v1_p7_top20/legacy_profile_rank13
export LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$PWD/libtvm_runtime.so:$PWD/libvta.so"
export TVM_NUM_THREADS=4
: "${TVM_THREAD_POOL_SPIN_COUNT:=0}"
: "${AXU5EVB_DRIVER_POST_START_SLEEP_NS:=1000}"
: "${AXU5EVB_DRIVER_POLL_SLEEP_NS:=1000}"
export TVM_THREAD_POOL_SPIN_COUNT
export AXU5EVB_DRIVER_POST_START_SLEEP_NS
export AXU5EVB_DRIVER_POLL_SLEEP_NS

exec /media/sd-mmcblk1p2/vta_stage_pair_runner_g0 --stage0-graph stages/stage0/graph.json --stage0-lib stages/stage0/graphlib.so --stage0-params stages/stage0/params.params --stage0-input-names data0 --stage0-name stage0 --stage0-device cpu --stage0-runtime-num-threads 4 --stage0-cpu-affinity 0,1,2,3 --stage1-graph stages/stage1/graph.json --stage1-lib stages/stage1/graphlib.so --stage1-params stages/stage1/params.params --stage1-input-names data0 --stage1-name stage1 --stage1-device vta --stage1-runtime-num-threads 1 --stage2-graph stages/stage2/graph.json --stage2-lib stages/stage2/graphlib.so --stage2-params stages/stage2/params.params --stage2-input-names data0 --stage2-name stage2 --stage2-device cpu --stage2-runtime-num-threads 1 --stage2-cpu-affinity 0 --stage3-graph stages/stage3/graph.json --stage3-lib stages/stage3/graphlib.so --stage3-params stages/stage3/params.params --stage3-input-names data0 --stage3-name stage3 --stage3-device vta --stage3-runtime-num-threads 1 --stage4-graph stages/stage4/graph.json --stage4-lib stages/stage4/graphlib.so --stage4-params stages/stage4/params.params --stage4-input-names data0 --stage4-name stage4 --stage4-device cpu --stage4-runtime-num-threads 1 --stage4-cpu-affinity 0 --input input.bin --runs 10 --queue-depth 2 --runtime-num-threads 4 --output-mode raw --output-jsonl /media/sd-mmcblk1p2/sweep_a5220e22_natural60/D_a1_r2_q25.jsonl --vta-runtime-profile-dir '' --vta-runtime-profile-events-limit 200 --vta-runtime-profile-checkpoint-every 0 --serial --warmup-runs 5 --pair-active 1 --pair-ready 2 --pair-offset-ms 2.534
