# STATUS

- Task: C3-P3b weight-residency dependency audit
- State: `completed_read_only`
- Date: 2026-09-10
- Source edits: none by this audit
- Board/SSH: not used
- RTL edits: none
- Primary finding: failed weight-cache hoisting creates illegal loop-carried `STORE(3) -> LOAD(1)`; VTA provides only `1 <-> 2` and `2 <-> 3` token paths.
- Schedule-only verdict: feasible only as an explicit-drain proof or if the compute region is restructured to mediate the boundary; plain `compute_at` hoisting is not feasible.
- Preferred implementation direction: VTA-specific post-`CoProcSync` dependency legalization, then consider a weight-load liveness/dedup pass.
- Performance claim: none; in particular this run does not claim recovery of any board FPS.

The worktree was already dirty and contains concurrent user/agent changes. All evidence is bound to the hashes in `manifest.json`; no attempt was made to restore or alter those files.

