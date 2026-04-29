# Native Stage Pipeline Run: 20260429 cat three_stage_e runs20 default threads
## Command
```bash
cd /home/orange/code/tvm
BOARD=root@192.168.1.133
REMOTE_DIR=/mnt/sd/vta_stage_pipeline
OUT_DIR=/tmp/vta_stage_pipeline_results

/home/orange/miniconda3/envs/vta-resnet/bin/python   vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py   --board "$BOARD"   --ssh-option HostKeyAlgorithms=+ssh-rsa   --ssh-option PubkeyAcceptedAlgorithms=+ssh-rsa   --remote-dir "$REMOTE_DIR"   --runs 20   --scheme three_stage_e   --queue-depth 2   --runtime-num-threads 4   --run-serial-before-pipeline   --compare-serial-pipeline   --vta-runtime-profile-dir profile   --vta-runtime-profile-events-limit 200   --fetch-results-dir "$OUT_DIR"   --keep-build-dir
```
## Result Summary
- Correctness: serial vs pipeline top1 matched for 20 frames.
- Serial steady state, skip first 3: total 148.102 ms, stage0 52.734 ms, stage1 71.705 ms, stage2 23.529 ms.
- Pipeline steady state, skip first 3: stage0 89.514 ms, stage1 72.763 ms, stage2 71.997 ms.
- Pipeline stage0 start interval, skip first 3: 89.564 ms, throughput 10.238 fps.

## Interpretation
Pipeline overlap is present, but CPU stage0/stage2 contend heavily: serial stage0/stage2 are about 52.7/23.5 ms, while pipeline stage0/stage2 are about 89.5/72.0 ms. VTA stage remains around 72 ms. Next sweep should tune per-stage CPU runtime threads.

## Files
- `stage_serial_result.jsonl`
- `native_result.jsonl`
- `manifest.json`
- `profile/serial/`
- `profile/pipeline/`
- `summary.json`
