# P7Q post-qualification timing addendum

Frozen on 2026-09-11 after correctness qualification and before any P7 latency label.

The original P7 contract required 80/80 candidates to pass before confirmatory timing. Its result is 79 correct candidates and one reproducible wrong-answer candidate: W04 `input_stationary` ConfigSpace index 139 (`906c1bdc...`). Therefore the original 80/80 confirmatory endpoint is not claimed.

P7Q is a clearly labeled post-qualification salvage analysis:

- Physically time exactly the 79 candidates that passed all frozen exact checks; do not replace the failure.
- Retain the invalid candidate in every gross search-policy order. If a policy reaches it, that dispatch consumes budget, produces no latency label, and is not used to train B3.
- Keep correctness-only diagnostic candidates outside the timing pool.
- Use three warmups and five deterministic randomized complete blocks per workload, with the incumbent before/after each block as drift sentinels.
- Use one reused board buffer set per workload to avoid the u-dma-buf exhaustion exposed by the first W08 harness.
- Report this analysis as P7Q rather than retroactively calling it the original preregistered P7 result.

Frozen artifacts:

- Qualified contract SHA-256: `dbed44f6260f6acade9dd723875f86f53b843402603447c4a927b7fb618f2246`.
- Timing runner SHA-256: `8bfe549490c8ddd22eec9d1b526be9dfccf222faa3e159b13ce385b8981562d5`.
- B3 sequential-XGB replay SHA-256: `3cb0413a31af8d903aa2d12d64404869889c8075a734b68ff891e6c12346f0f5`.
- XGBoost version: 3.2.0; seed 20250901; seven numeric ConfigEntity knobs only; four deterministic gross warmup dispatches; 64 trees, depth 3, learning rate 0.1, hist tree method, one thread.

The current RPC process has no `VTA_*` environment variables, so runtime profiling, replay, and command-threshold submission are disabled for timing.
