#!/usr/bin/env bash
set -euo pipefail
OUT_DIR=/tmp/vta_stage_pipeline_results_3_1_1_poll_10us
mkdir -p "$OUT_DIR"
source /home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux
AXU5EVB_DRIVER_POST_START_SLEEP_NS=10000 AXU5EVB_DRIVER_POLL_SLEEP_NS=10000 TEST_DATA_ROOT_PATH=/tmp/tvm_test_data MPLCONFIGDIR=/tmp/mpl PYTHONUNBUFFERED=1 /home/orange/miniconda3/envs/vta-resnet/bin/python vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py   --board root@192.168.1.133   --ssh-option HostKeyAlgorithms=+ssh-rsa   --ssh-option PubkeyAcceptedAlgorithms=+ssh-rsa   --remote-dir /mnt/sd/vta_stage_pipeline   --runs 20   --queue-depth 2   --runtime-num-threads 4   --stage0-runtime-num-threads 3   --stage1-runtime-num-threads 1   --stage2-runtime-num-threads 1   --run-serial-before-pipeline   --compare-serial-pipeline   --fetch-results-dir "$OUT_DIR"
