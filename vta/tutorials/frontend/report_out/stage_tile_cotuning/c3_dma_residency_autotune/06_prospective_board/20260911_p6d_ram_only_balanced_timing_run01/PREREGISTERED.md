# C3-P6d RAM-only balanced timing preregistration

Frozen on 2026-09-11 after P6c correctness completed and before observing any P6d latency sample.

- Correctness precondition: P6c passed 9/9 candidates and 27/27 exact seed checks; `results.jsonl` SHA-256 is `cc59f507b4ccdfafe0edbe863abeca48b3096d3d54454d03102183e35a9af348`, and `summary.json` SHA-256 is `accb956f2309d4f6c14a8bf7dd6ff48a011c17aee2c25687f7157b22ba75098f`.
- Candidate and ordering contract: immutable P6b `mechanism_canary_manifest.json` and `dispatch_plan.json`.
- Warmup: 3 invocations per candidate, excluded from timing.
- Measurement: 30 samples per candidate with `time_evaluator("main", number=1, repeat=1)`.
- Ordering: the exact seeded randomized complete crossover frozen in P6b; all six workload and within-workload candidate permutations occur five times.
- Setup: compilation, upload, allocation, and warmup are excluded.
- Counters: clear immediately before each measured invocation, then record the required per-inference runtime counters.
- Decision: compare each mechanism median with its workload's incumbent median. Improvement is at least 5%; equivalence is absolute difference at most 2%; regression is at most -5%; values in between are indeterminate.
- Safety: current boot ID must remain `a68a7983-719f-47bf-94d7-41f974c5342c`, `/dev/mmcblk1p2` remains read-only, experiment files remain RAM-only, and any new storage error after dmesg timestamp 467 invalidates/stops the run.
