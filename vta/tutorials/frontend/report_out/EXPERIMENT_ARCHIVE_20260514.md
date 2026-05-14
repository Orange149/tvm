# VTA Stage Pipeline Experiment Archive 2026-05-14

This branch records the experiment metadata, summaries, and reproducible command
structure for the ResNet18 and YOLOv3-tiny native stage pipeline work.

The full `report_out` tree is intentionally not suitable for a normal git
commit: it is about 85 GB and includes build caches, package archives,
parameters, and raw output dumps. Keep those large artifacts on the local
machine. Commit only the lightweight manifests, summaries, CSV/JSON reports,
and analysis notes needed to reproduce or audit the results.

## Board Setup

Target board used for the latest YOLO runs:

```bash
root@192.168.1.185
```

After board reboot:

```bash
mkdir -p /mnt/sd
mount -t ext4 /dev/mmcblk1p2 /mnt/sd
ls /mnt/sd/tvm_deploy
cp /mnt/sd/tvm_deploy/firmware/vta_hpc.bit /lib/firmware/
insmod /mnt/sd/tvm_deploy/hp/u-dma-buf.ko udmabuf0=201326592
cat /sys/class/u-dma-buf/udmabuf0/size
echo vta_hpc.bit > /sys/class/fpga_manager/fpga0/firmware
cat /sys/class/fpga_manager/fpga0/state
```

Host environment:

```bash
unset LD_LIBRARY_PATH
source /home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux
export TEST_DATA_ROOT_PATH=/tmp/tvm_test_data
export MPLCONFIGDIR=/tmp/mpl
export PYTHONUNBUFFERED=1
export VTA_HW_PATH=/home/orange/code/tvm/3rdparty/vta-hw
PY=/home/orange/miniconda3/envs/vta-resnet/bin/python
```

## ResNet18 Native Stage Search

Important directories:

- `native_stage_pipeline_searches/20260507_v23_warm100`
- `native_stage_pipeline_searches/20260512_theory_build100_board`
- `native_stage_pipeline_build_cache/v23_20260506`

Summary:

- V2.3 uses static full enumeration, calibrated shortlist, package-only
  buildability, and native board measurement.
- Warm100 and theory-guided Build100/board runs produced 200 measured native
  ResNet18 candidates.
- Results motivated a heuristic centered on stage-time balance, CPU/VTA
  occupancy, boundary transfer, DMA fragmentation, SRAM risk, and avoiding
  overly small VTA islands.

Do not commit the persistent build cache as normal git content. It contains
large packages and `params.params` files.

## YOLOv3-tiny Baselines

Important directories:

- `yolov3_tiny_pipeline_baseline/20260512_rpc_all_vta_runs20`
- `yolov3_tiny_pipeline_baseline/20260512_native_serial_smoke`
- `yolov3_tiny_pipeline_baseline/20260513_packed_graph_split_smoke`
- `yolov3_tiny_pipeline_baseline/20260513_packed_hetero_top20`

Known reference points:

- RPC all-VTA baseline: `3.724 fps`
- Packed graph smoke: `6.5215 fps`
- Packed hetero best: `4.2922 fps`

## YOLOv3-tiny CPU/VTA/CPU Search

Latest 100-candidate run:

- `yolov3_tiny_pipeline_baseline/20260514_resnet_style_cpu_vta_cpu_100`
- Main report: `FINAL_ANALYSIS.md`
- Full results: `summary.json`, `summary.csv`
- Candidate list: `safe_candidates.csv`
- Build report: `buildability_summary.json`, `buildability.csv`

Command shape:

```bash
$PY vta/tutorials/frontend/search_yolov3_tiny_stage_splits.py \
  --mode build,measure \
  --split-backend relay \
  --candidate-count 36 \
  --runtime-config-search-count 100 \
  --max-runtime-configs-per-split 6 \
  --split-quantization-mode per_stage \
  --tail-device cpu \
  --runner-output-mode raw_all_stages \
  --correctness-policy detection_gate \
  --output-root vta/tutorials/frontend/report_out/yolov3_tiny_pipeline_baseline/20260514_resnet_style_cpu_vta_cpu_100 \
  --rpc-baseline-json vta/tutorials/frontend/report_out/yolov3_tiny_pipeline_baseline/20260512_rpc_all_vta_runs20/rpc_all_vta_baseline.json \
  --board root@192.168.1.185 \
  --port 9090 \
  --remote-dir /var/volatile/yolov3_tiny_pipeline_search \
  --measure-count 100 \
  --serial-runs 2 \
  --runs 20 \
  --queue-depth 2 \
  --serial-timeout-s 240 \
  --pipeline-timeout-s 600 \
  --fetch-timeout-s 120 \
  --ssh-option HostKeyAlgorithms=+ssh-rsa \
  --ssh-option PubkeyAcceptedAlgorithms=+ssh-rsa
```

Latest result:

- Generated candidates: 100
- Buildable configs: 88
- Serial smoke passed: 88
- Pipeline measured OK: 88
- Pipeline timeout/error: 0
- Best candidate:
  `yolo_pool2_to_dual_pre_logits__rt_s04_s22_q2_poll1000_post1000`
- Best throughput: `5.113142 fps`
- Best stage times: `[195.9976552, 165.1157442, 137.6137912] ms`
- Correctness policy: detection gate passed, raw gate recorded as diagnostic

Practical YOLO heuristic:

Choose a coarse CPU/VTA/CPU split that keeps VTA occupied with the main
conv-heavy trunk while keeping both CPU prefix and CPU tail below the VTA-stage
time. For this board and implementation, `pool2 -> dual_pre_logits` is the best
observed split family. Earlier starts overload VTA; later exits make the CPU
tail too heavy or the VTA island too short.

## Recommended Git Scope

Commit:

- Search scripts and small source changes needed to reproduce the runs.
- `EXPERIMENT_ARCHIVE_20260514.md`.
- Lightweight `README.md`, `FINAL_ANALYSIS.md`, `summary.json`, `summary.csv`,
  `buildability*.json/csv`, `safe_candidates.csv`, `rejected_candidates.csv`,
  `*_baseline.json`, and measured-id text files.

Do not commit:

- `packages/`
- `native_stage_pipeline_build_cache/`
- `serial_output_dumps/`
- `pipeline_output_dumps/`
- `*.tar.gz`
- large `params.params` / shared libraries generated by package builds

