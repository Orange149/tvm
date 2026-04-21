HP/HPC coherence quantification assets live here.

Scripts:
- `measure_vta_coherence_throughput.py`
- `run_hp_vta_profile_suite.py`
- `minimal_three_stage_resnet18.py`
- `vta_runtime_profile_utils.py`

Results:
- `results/`

Run these commands from the repo root.

Environment example:

```bash
unset LD_LIBRARY_PATH
source /home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux
export TEST_DATA_ROOT_PATH=/tmp/tvm_test_data
export MPLCONFIGDIR=/tmp/mpl
export PYTHONUNBUFFERED=1
```

Measure HP-only throughput/coherence:

```bash
/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/hp_hpc_quant/measure_vta_coherence_throughput.py \
  --host 192.168.1.247 \
  --port 9090 \
  --config-label hp_only \
  --output-dir /tmp/hp_coherence_bw
```

Measure HPC/coherent throughput/coherence:

```bash
/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/hp_hpc_quant/measure_vta_coherence_throughput.py \
  --host 192.168.1.247 \
  --port 9090 \
  --config-label hpc_coherent \
  --output-dir /tmp/hpc_coherence_bw
```

Run the HP profiling suite:

```bash
/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/hp_hpc_quant/run_hp_vta_profile_suite.py \
  --host 192.168.1.247 \
  --port 9090 \
  --output-dir vta/tutorials/frontend/hp_hpc_quant/results/hp_profile_suite
```

Run the minimal three-stage ResNet18 build/run experiment:

```bash
/home/orange/miniconda3/envs/vta-resnet/bin/python \
  vta/tutorials/frontend/hp_hpc_quant/minimal_three_stage_resnet18.py \
  --scheme three_stage_e \
  --host 192.168.1.247 \
  --port 9090 \
  --run
```

Useful outputs:
- `measure_vta_coherence_throughput.py` writes `coherence_throughput.csv` and `summary.json`.
- `run_hp_vta_profile_suite.py` writes logs, CSVs, and profiler JSON files under its `--output-dir`.
- checked-in historical reports and CSVs are under `results/`.
