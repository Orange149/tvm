# AutoTVM pre-measure certificate gate integration

- Status: `passed`
- Input: E03 input-stationary task plus P7R49 hardware-certified dispatch.
- Dispatch contents: one failed exact hardware identity, one unknown identity, no `allow_timing` entry.
- Expected behavior: refuse tuning before creating any `MeasureInput` or RPC request.
- Observed behavior: `ValueError: dispatch contains no exact hardware-certified configuration to time`.
- AutoTVM log files created: none.
- FPGA module execution or timing: none.
- Regression tests: 27 passed and 12 environment-dependent FSim tests skipped; focused dispatch tests 6/6 passed.

This closes the distinction between checking correctness inside a Runner and filtering before the
AutoTVM model. The latter is required because native `ModelBasedTuner.update` records failed
measurements with zero throughput.
