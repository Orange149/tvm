# C3 P2-local run02 handoff

Run02 closes the source-guard gap left by run01. Under a stable post-refactor source state, the original schedule reproduces the frozen ten-workload DMA artifact exactly, passes 30/30 local reference checks, cross-compiles 10/10 AXU artifacts, and passes the two requested regression suites (19 and 7 tests).

Primary evidence:

- `static_dma.json` plus `logs/static_compare.stdout.log`;
- `fsim_results.json` plus `logs/fsim.stdout.log`;
- `cross_compile_results.json` plus `logs/cross_compile.stdout.log`;
- `tir_results.json`, `lowered_tir/`, and `logs/tir_compare_run01.stdout.log`;
- raw pytest stdout/stderr under `logs/`;
- source/input guards in `source_hashes_start.txt` and `source_hashes_end.txt`.

The run-local `audit_original.py` is evidence tooling, not production code. Its AXU target guard was corrected after a pre-compilation diagnostic failure; this does not affect any production source or measured result.

Accept this as `P2_LOCAL_BASELINE_PASS`. Do not convert it into a board-performance claim or full-G2 claim: no SSH, board RPC, physical AXI measurement, FPS, or latency measurement occurred.
