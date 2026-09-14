# P7R120 Y02 RPC-only board contract

- Status: `frozen_after_local_cross_qualification_before_rpc_labels`
- Candidate pool: 6 (2 families x 3 real modes)
- Fresh AXU cross qualification: 6/6; sealed TopHub canary: passed
- Board path: direct RPC 192.168.1.247:9090; SSH/restart/persistent writes forbidden
- Correctness gate: all 18 elementwise seed checks before timing
- Timing: 7 balanced rounds, TopHub bracketed but excluded from search/training
- Boot ID: `unknown_not_exposed_by_rpc`
