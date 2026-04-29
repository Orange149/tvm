#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="/tmp/vta_stage_pipeline_results_3_1_1_poll_0p5us"
mkdir -p "$OUT_DIR"
AXU5EVB_DRIVER_POST_START_SLEEP_NS=500 \
AXU5EVB_DRIVER_POLL_SLEEP_NS=500 \
TEST_DATA_ROOT_PATH=${TEST_DATA_ROOT_PATH:-/tmp/tvm_test_data} \
MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl} \
PYTHONUNBUFFERED=1 \
  /home/orange/miniconda3/envs/vta-resnet/bin/python \
  /home/orange/code/tvm/vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py \
  --board \
  root@192.168.1.133 \
  --remote-dir \
  /mnt/sd/vta_stage_pipeline \
  --runs \
  20 \
  --scheme \
  three_stage_e \
  --queue-depth \
  2 \
  --runtime-num-threads \
  4 \
  --stage0-runtime-num-threads \
  3 \
  --stage1-runtime-num-threads \
  1 \
  --stage2-runtime-num-threads \
  1 \
  --run-serial-before-pipeline \
  --compare-serial-pipeline \
  --fetch-results-dir \
  /tmp/vta_stage_pipeline_results_3_1_1_poll_0p5us \
  --ssh-option \
  HostKeyAlgorithms=+ssh-rsa \
  --ssh-option \
  PubkeyAcceptedAlgorithms=+ssh-rsa
