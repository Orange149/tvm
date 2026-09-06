# Rejected RPC Calibration Smoke

The initial single-operator smoke used `benchmark_resnet18_single_ops.py`, whose execution
path is TVM RPC. It is rejected from RAMPS calibration for the following reasons:

1. RAMPS performance calibration must match the native static-package executor.
2. RPC H2D/D2H includes transport semantics that are not native stage boundary service.
3. Both CPU and VTA builds reported fallback schedules.
4. CSV emission failed because the legacy field list omitted memory-bandwidth columns.

The observed terminal values are diagnostic only. They must not be copied into
`service_model.json`, fitted parameters, paper tables or uncertainty analysis. RPC remains
allowed only for the persisted all-VTA correctness reference.
