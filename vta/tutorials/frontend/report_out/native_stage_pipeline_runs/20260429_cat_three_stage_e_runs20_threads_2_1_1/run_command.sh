cd /home/orange/code/tvm
BOARD=root@192.168.1.133
REMOTE_DIR=/mnt/sd/vta_stage_pipeline
OUT_DIR=/tmp/vta_stage_pipeline_results_2_1_1

/home/orange/miniconda3/envs/vta-resnet/bin/python   vta/tutorials/frontend/deploy_classification_stage_pipeline_native.py   --board "$BOARD"   --ssh-option HostKeyAlgorithms=+ssh-rsa   --ssh-option PubkeyAcceptedAlgorithms=+ssh-rsa   --remote-dir "$REMOTE_DIR"   --runs 20   --scheme three_stage_e   --queue-depth 2   --runtime-num-threads 4   --stage0-runtime-num-threads 2   --stage1-runtime-num-threads 1   --stage2-runtime-num-threads 1   --run-serial-before-pipeline   --compare-serial-pipeline   --vta-runtime-profile-dir profile   --vta-runtime-profile-events-limit 200   --fetch-results-dir "$OUT_DIR"   --keep-build-dir
