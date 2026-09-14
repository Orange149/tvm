# Handoff

## Reproduce

Run the command in `command.txt` from the TVM repository root. It writes `results.jsonl` and `summary.json`. Then run the two verification commands recorded there.

## Review points

1. Confirm the source hashes and 80 = 32 pass + 7 wrong-answer + 41 compile accounting.
2. Treat budgets as total dispatched archive positions, including failures.
3. Treat W03/W06 as unscorable, not as zero-performance workloads.
4. B2 is the only evaluable baseline. Do not use B4's leaky diagnostic upper bound or B5/B6's not-evaluable diagnostics as effectiveness results; their ordering depends on post-dispatch outcome/runtime data.
5. A later prospective run must compute static compile/SRAM/DMA features for every candidate before dispatch, preserve the incumbent, use grouped holdout, and measure on the board before claiming G5.

No schedule, runtime, driver, hardware, or tuning script was changed by this P5-local task.
