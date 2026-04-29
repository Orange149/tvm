# Native Stage Pipeline Run: 20260429 cat three_stage_e runs20 threads 2_1_1

## Command
```bash
cd /home/orange/code/tvm
BOARD=root@192.168.1.133
REMOTE_DIR=/mnt/sd/vta_stage_pipeline
OUT_DIR=/tmp/vta_stage_pipeline_results_2_1_1

/home/orange/miniconda3/envs/vta-resnet/bin/python   vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py   --board "$BOARD"   --ssh-option HostKeyAlgorithms=+ssh-rsa   --ssh-option PubkeyAcceptedAlgorithms=+ssh-rsa   --remote-dir "$REMOTE_DIR"   --runs 20   --scheme three_stage_e   --queue-depth 2   --runtime-num-threads 4   --stage0-runtime-num-threads 2   --stage1-runtime-num-threads 1   --stage2-runtime-num-threads 1   --run-serial-before-pipeline   --compare-serial-pipeline   --vta-runtime-profile-dir profile   --vta-runtime-profile-events-limit 200   --fetch-results-dir "$OUT_DIR"   --keep-build-dir
```

## Result Summary
- Correctness: serial vs pipeline top1 matched for 20 frames.
- Serial steady state, skip first 3: total 232.767 ms, stage0 94.741 ms, stage1 71.602 ms, stage2 66.284 ms.
- Pipeline steady state, skip first 3: stage0 96.769 ms, stage1 72.939 ms, stage2 70.818 ms.
- Pipeline stage0 start interval, skip first 3: 96.801 ms, throughput 9.528 fps.

## Interpretation
This 2/1/1 run is worse than default 3/1/1. Stage0 becomes the bottleneck at about 96.8 ms/frame, and serial sanity is also slower because the same per-stage thread override is used there. Keep this as a negative result; next useful sweep should try reducing CPU overlap differently or separating serial baseline from pipeline thread config.
