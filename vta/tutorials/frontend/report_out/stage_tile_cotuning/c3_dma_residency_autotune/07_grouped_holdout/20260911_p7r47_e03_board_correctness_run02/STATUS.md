# E03 board correctness run 02

- Status: `failed_correctness_gate_stopped_before_timing`
- Frozen contract: E03 DMA-Pareto priority config17 versus boundary config576.
- Original config17: 3/3 seeds exact.
- Input-stationary config17: 0/3 seeds exact; mismatch counts 36536, 36879, 37254.
- Boundary config576: not executed because the first mismatch stops the contract.
- Performance measurements: none.
- Safety result: the FPGA canary rejected the candidate before it could enter the latency model.
- Post-failure board state: unchanged boot ID, FPGA `operating`, RPC PID 830, no new storage error.
