# HANDOFF

`results.jsonl` is the candidate-level AXU5EVB build/export ledger. Fresh candidates were process-isolated, re-lowered with an exact TIR-hash check, built, and exported with the aarch64 Xilinx toolchain. Protected incumbents cite P2 run02. Shared objects were deleted with their temporary directories; only size/hash remain. Passing this stage does not establish board correctness, loadability on the current boot, latency, or FPS.

All 155 fresh records passed their exact ConfigEntity and re-lowered TIR hash checks. The 165 records comprise 55 original, 32 input-stationary, 43 weight-stationary, and 35 paper-inspired-hybrid candidates. Compile success does not imply that weight or hybrid reduces DMA or improves performance.
