# P7e1 W08 correctness resume

Frozen after the P7e infrastructure stop and before this execution.

- Contract SHA-256: `696bf95b25b4db7e9f54252231e70073e35e203d982c2d9a8d2b1f25b85f1bc0`.
- Resume-runner SHA-256: `fe93d7a179f90e19afa745daa5522f19744fc990e2e090db89e56837cef3fbbf`.
- Scope: only previously unobserved W08 correctness positions 19--21; positions 1--18 must not be repeated.
- Oracle: exact elementwise NumPy equality for seeds 0, 20250901, and 20260910.
- The runner allocates one shape-compatible data/weight/output set and reuses it, avoiding cumulative u-dma-buf exhaustion.
- This run collects no latency, FPS, search-efficiency, or timing-profiler label.
- First wrong answer or infrastructure error stops the suffix; no replacement candidate is allowed.
