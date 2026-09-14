# HANDOFF

Read `WEIGHT_REUSE_FEASIBILITY.md` first. The immediate engineering decision is:

1. Do not retry outer `ckernel.compute_at` by itself and do not remove the runtime check.
2. If a minimal experiment is authorized, first add an explicit drain at the outer weight-residency group and inspect the post-sync TIR for forbidden pairs. This is only a feasibility probe.
3. For an asynchronous solution, design a VTA-specific legalization pass between `InjectCoProcSync` and intrinsic lowering. Its hard invariant is that no emitted VTA dependency pair directly connects queues 1 and 3.
4. Only after that invariant is enforced should a general load-lifetime/dedup pass be attempted.

Mandatory offline acceptance criteria for future implementation:

- forbidden dependency-pair count is zero after VTA lowering;
- push/pop tokens balance on every loop/branch path;
- existing schedules without forbidden edges remain unchanged;
- weight DMA decreases statically;
- SRAM/ACC/UOP/command limits pass;
- FSim correctness passes representative padding, stride, tail, and randomized scheduling cases.

This audit did not edit source, use SSH, access the board, or establish an FPS result.

