# P7R6 adaptive recovery contract

Frozen after config330's same-tile original failed real-FPGA correctness, and before any board
execution of config455 or config461. This development-only adaptation does not use latency labels:
it removes config330 and retains the two orientation probes that were selected in the original
contract but never executed. Each remains paired with its exact original ConfigEntity and the batch
is bracketed by protected config463 sentinels. No timing is allowed unless all 18 seed checks pass.
