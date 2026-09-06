# RAMPS v1 Experiment Protocol

This directory is the persistent root for the Resource-Aware Max-Plus Stage
Search (RAMPS) publication experiment. Generated measurements must remain here;
`/tmp` is only used for local smoke tests.

The normative system formulation, theoretical foundations, CPU-demand model,
queue semantics, and portability contract are documented in
[RAMPS_SYSTEM_MODEL.md](../RAMPS_SYSTEM_MODEL.md). Paper-ready evidence is
maintained in [PAPER_EVIDENCE.md](../PAPER_EVIDENCE.md). The frozen order below
describes the existing v1 implementation and is retained for reproducibility,
not as the final experimental-design justification.

## Frozen Order

1. Measure all CPU, VTA, DMA, bridge, submit, and synchronization buckets in
   three independent board sessions. No fallback bucket is accepted.
2. Run the 12-anchor by 3-queue-depth by 2-thread-policy ResNet18 design. Each
   of its 72 configurations is measured by three independent native processes.
3. Fit resource parameters from the controlled design and a monotonic residual
   from the historical ResNet18 200 cohort.
4. Write and hash `models/resnet_to_yolo_frozen.json` before opening any YOLO
   throughput label.
5. Evaluate the frozen model once on the historical YOLOv3-tiny 169 cohort.
6. Select and measure the frozen prospective YOLO30 cohort without updating the
   model until every buildable candidate has completed all three sessions.

Candidate performance is always measured by the native pipeline. RPC is used
only for the all-VTA correctness reference.

## Invocation

```bash
unset LD_LIBRARY_PATH
source /home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux
export TEST_DATA_ROOT_PATH=/tmp/tvm_test_data
export MPLCONFIGDIR=/tmp/mpl
export PYTHONUNBUFFERED=1
export VTA_HW_PATH=/home/orange/code/tvm/3rdparty/vta-hw
export PYTHONPATH=python:vta/python:vta/tutorials/frontend

/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/run_ramps_experiment.py \
  --board root@192.168.1.228 \
  --resume
```

Progress is recorded in `orchestration/progress.json`; long board phases also
write their own `progress.txt` and `progress.json`.

## Publication Gates

- `calibration/cost_model.json` must have `source=ramps_measured_calibration`
  and contain no `fallback` or `estimated_from` field.
- `identification/resnet72/completion.json` must report all 72 publication-valid
  configurations and all 216 processes.
- `models/freeze_receipt.json` must predate zero-shot result generation and its
  model hash must remain unchanged through prospective30.
- Zero-shot and prospective reports include all failures and interval coverage,
  not only the best throughput.
- Claims remain provisional until executor-matched all-VTA, energy, search cost,
  an unseen third model, and hardware-sensitivity experiments are complete.

## Current Measurement Status

No publication-grade board measurement has been started under the revised
systematic formulation. Before measurement, rerun preflight on
`root@192.168.1.228`, record the hardware fingerprint, and verify that every
formal calibration bucket is measured rather than estimated.
