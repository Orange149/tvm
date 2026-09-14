# C3-P3e explicit-drain compiler proof

- Status: `completed_positive_local_proof`
- Decision: **GO for the isolated compiler-level explicit-drain mechanism**
- Scope: local lowering, dependency audit, build, and FSim correctness only
- Board/SSH: not used
- Performance timing: not collected; FSim time is not a performance result
- Probe identity: mode 4 remains excluded from production candidates and winners

The implementation repairs the explicit-sync lowering at
`vta/python/vta/transform.py:346` by constructing the registered
`tir.vta.coproc_sync` Op. `CoProcInstDepDetector` recognizes that Op at
`src/tir/transforms/coproc_sync.cc:377`, closes unmatched tokens in the segment before
the full drain, and clears dependency state before the following segment. It does not
remove the runtime dependency check.

The mode-4 schedule keeps a weight tile at the outer output-channel scope and places a
full drain after that tile's complete spatial region (`vta_conv2d.py:339`). W00 and W02
reduce statically counted weight DMA bytes by 85.71% and 75.00%; W09 reduces weight LOAD
calls from 32 to 2 but not bytes. All three lowered modules have balanced push/pop pairs
and no direct `1 <-> 3` pair. All nine FSim executions (three workloads times seeds 0,
20250901, and 20260910) exactly match the independent NumPy reference.

The final combined regression is 50/50 passed. It covers the original CoProcSync tests,
new pure-TIR barrier segmentation tests, mode-4 static/FSim qualification, frozen
original TIR hashes, and modes 0--3.

This is a compiler dependency-handling enhancement enabling explicit draining. It is
not hardware prefetch and does not establish board performance.
