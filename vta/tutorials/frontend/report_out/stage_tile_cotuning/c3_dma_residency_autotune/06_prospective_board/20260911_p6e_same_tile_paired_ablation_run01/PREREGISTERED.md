# C3-P6e same-tile paired ablation preregistration

Frozen after P6d and before observing any P6e board output. Six mechanism candidates are paired with mode-0 controls having the identical workload and ConfigEntity. Five unique controls are used because W02 input-stationary and weight-barrier share config 177.

Before timing, all 11 unique candidates must pass exact board output equality for all three frozen seeds (33 checks total). Each unique candidate receives three warmups. Each pair then receives 30 measurements per side using the deterministic order in `protocol.json`; treatment order is balanced 15/15 and every pair occupies every order position five times. Compilation, upload, allocation, correctness, and warmup are excluded from latency.

This run isolates the residency-mode change at a fixed tile/thread configuration. It does not test whether a search policy finds the configuration efficiently, whether the configuration beats TopHub, or whether an operator effect transfers to stage/FPS.
