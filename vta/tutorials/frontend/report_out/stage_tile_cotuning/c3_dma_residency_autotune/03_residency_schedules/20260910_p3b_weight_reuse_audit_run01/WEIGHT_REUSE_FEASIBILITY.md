# C3-P3b: VTA weight-residency feasibility audit

Date: 2026-09-10  
Status: read-only engineering decision; no source/RTL change and no board claim

## Decision first

The failed outer `ckernel.compute_at` experiment does **not** show that VTA cannot reuse weights. It shows that the particular TE lifetime transformation exposes a direct `STORE(3) -> LOAD(1)` loop-carried dependence which VTA cannot encode. VTA has only four physical token paths, `LOAD <-> COMPUTE` and `COMPUTE <-> STORE`; it has no `STORE <-> LOAD` FIFO. Therefore removing the runtime assertion is unsafe.

Recommended order:

1. **Schedule-only feasibility probe with an explicit drain/synchronization boundary** — smallest change, useful to prove semantic feasibility, but likely loses overlap and may not improve runtime.
2. **VTA-specific compiler-pass legalization after `InjectCoProcSync`** — most plausible non-RTL production path: rewrite unsupported logical `3 <-> 1` dependencies through compute stage 2, with strong token-balance tests.
3. **Compile-flow weight-load lifetime/dedup pass before dependency injection** — highest potential and closest to a general weight-residency contribution, but widest correctness surface.
4. Direct runtime translation is not recommended; never merely delete the assertion.

The present safe schedule is a correctness baseline, not weight-residency: it keeps `ckernel` at `conv2d_stage/k_o`, so it avoids the illegal edge but does not reduce weight DMA.

## Exact failing dependence chain

The failed implementation moved the weight cache from the inner convolution region to an outer output loop (`ckernel.compute_at(output, x_co0)` for weight mode and analogous `x_cog` for hybrid). The resulting chain is:

1. `ckernel` is a weight cache read. `InjectDMAIntrin` recognizes the weight SRAM scope and emits `VTALoadBuffer2D` under coprocessor queue id 1 (`vta/python/vta/transform.py:560-577,615-643`; queue ids are defined in `vta/python/vta/environment.py:76-81`).
2. The convolution/GEMM and ALU body uses queue id 2 (`vta/python/vta/intrin.py:95-98`; `environment.py:79-81`).
3. The output write is emitted as `VTAStoreBuffer2D` under queue id 3 (`transform.py:529-557`).
4. At the outer residency-loop boundary, the previous iteration exits with output STORE(3), while the next iteration enters with the hoisted weight LOAD(1).
5. Generic `CoProcInstDepDetector` explicitly connects a loop body's last state to its first state (`src/tir/transforms/coproc_sync.cc:376-394`). Its singleton fast path blindly emits `push(from,to)` and `pop(from,to)` for any unequal numeric contexts (`coproc_sync.cc:469-485`); it has no VTA topology rule. It therefore creates `coproc_dep_push(3,1)` after STORE and `coproc_dep_pop(3,1)` before the next weight LOAD.
6. VTA intrinsic lowering converts those calls to `VTADepPush(...,3,1)` and `VTADepPop(...,3,1)` (`vta/python/vta/build_module.py:188-199`).
7. Runtime queue construction rejects the pop exactly because STORE-to-LOAD is impossible (`vta/runtime/runtime.cc:1569-1585`). This matches `failed_outer_cache_fsim.log`: `Check failed: (from != kStoreStage || to != kLoadStage) is false` in `VTADepPop` for W00/W02/W09.

This assertion is protective. `DepPush(3,1)` would otherwise reuse the generic `push_prev_dep` bit (`runtime.cc:1587-1611`), but a STORE's `push_prev_dep` physically feeds STORE-to-COMPUTE, not STORE-to-LOAD. Likewise a LOAD pop bit consumes from COMPUTE-to-LOAD. The simulator maps the four bit directions only to `l2g`, `s2g`, `g2l`, and `g2s` (`3rdparty/vta-hw/include/vta/sim_tlpp.h:148-155`; `3rdparty/vta-hw/src/sim/sim_tlpp.cc:99-123`). There is no `s2l` queue. The instruction format also carries relative previous/next dependency bits rather than arbitrary source/destination ids (`3rdparty/vta-hw/include/vta/hw_spec.h:57-115,144-176`).

## Why input residency is legal but outer weight hoisting is not

The working input mode changes output nesting to place multiple output-channel tiles inside one spatial region, while leaving both `cdata` and `ckernel` attached to `conv2d_stage/k_o` (`vta/python/vta/top/vta_conv2d.py:241-253,293-329`). It saves input loads because the same input region covers several output-channel computations; it does not put an input LOAD directly after a STORE at the outer lifetime boundary.

A read-only lowering audit of workload W00, config entity 252, found the same supported dependency-pair sequence for current original/input/weight/hybrid modes:

```text
3->2, 3->2, 2->1, 2->1, 1->2, 1->2,
2->1, 2->1, 2->3, 2->3, 3->2, 3->2
```

The duplicated entries are push/pop calls. The unique edges are only `1 <-> 2` and `2 <-> 3`; no `1 <-> 3` edge occurs. In the input schedule, the loop carry is mediated as `STORE(3) -> COMPUTE(2) -> LOAD(1)`. The failed weight transform instead moved only the weight LOAD outward, making the boundary directly `3 -> 1`. The recovered safe weight mode keeps both caches inner and therefore remains legal, but its static weight traffic is unchanged.

Note: `tvm.lower` still displays `tir.vta.coproc_dep_push/pop`; the later registered intrinsic lowering turns these into external `VTADepPush/Pop` calls. Searching only for extern calls in the pre-build TIR would produce a false negative.

## Where the behavior is introduced

The VTA build order is decisive (`vta/python/vta/build_module.py:69-85`):

```text
InjectDMAIntrin -> Lift scopes/allocations -> InjectCoProcSync
-> EarlyRewrite -> intrinsic lowering -> CPUAccessRewrite
```

- `InjectDMAIntrin` decides that a cache read becomes queue-1 weight DMA and an output write becomes queue-3 STORE.
- Generic `CoProcSync` derives dependencies from lexical/control-flow ordering, and creates the illegal loop-carried pair.
- `CPUAccessRewrite` only rewrites CPU BufferLoad/BufferStore/Allocate addresses to `VTABufferCPUPtr` (`vta/python/vta/transform.py:144-237`). It neither assigns VTA queues nor fixes command dependencies.
- `sim_driver.cc` executes the encoded load/compute/store operations and hands them to `TlppVerify` (`3rdparty/vta-hw/src/sim/sim_driver.cc:316-377,532-537,592-597`); it is not the source of the illegal pair.

Thus this is primarily a schedule/lowering interaction, not a CPU-access rewrite or simulator bug.

## Candidate implementation paths

### Path A — schedule-only explicit drain (feasibility probe)

Restore the outer weight-cache lifetime, but end each outer residency group with an explicit VTA `coproc_sync`/drain so the next weight LOAD starts after prior stores complete. `InjectCoProcSync` already rewrites an explicit sync pragma (`vta/python/vta/transform.py:331-367`), and the build lowers it to `VTASynchronize` (`vta/python/vta/build_module.py:169-185`).

- Scope: experimental schedule/template and its tests only; no runtime or RTL.
- Expected value: establishes whether the SRAM lifetime and numeric results are feasible independently of cross-queue overlap.
- Risks: synchronization may destroy the DMA benefit; placement inside TE may be awkward; group granularity changes submit/wait overhead; existing pad/compact-2D and ACC-capacity failures remain.
- Offline gate: dump TIR immediately before and after sync injection; assert no pair `{1,3}`; verify static weight bytes decrease; verify SRAM/ACC/UOP limits; FSim all representative shapes and seeds; count `VTASynchronize` calls.

Verdict: **schedule-only is plausible as a proof, not yet credible as the final performance solution.** A pure reorder/`compute_at` change without a drain or compute-mediated boundary has already been falsified.

### Path B — schedule-only compute-mediated region restructuring

Keep `ckernel` within a compute region, but enlarge that region to cover several spatial/output tiles and arrange the boundary so computation queue 2 lies between the last STORE and next LOAD. This may require holding multiple partial outputs in ACC before storing them.

- Scope: schedule/template/tests only.
- Risks: severe ACC pressure (W02 has already exceeded 1,048,576 bits at 1,605,632 bits), compact-2D/padding proof failures, extra no-op/ALU work, and possibly no actual weight-DMA reduction.
- Offline gate: storage-bound proof, exact TIR dependency-pair audit, static DMA counts, FSim correctness, and unchanged baseline schedule hash.

Verdict: **possible only if a legal tiling exists within SRAM; lower priority than Path A and the pass solution.**

### Path C — VTA-specific dependency legalization pass (recommended production direction)

Add a VTA-only TIR pass after generic `InjectCoProcSync` and before intrinsic lowering/`CPUAccessRewrite`. Detect logical `3 -> 1` and `1 -> 3` push/pop pairs and lower them into two legal neighbor edges through queue 2, e.g. `3 -> 2` then `2 -> 1`. Do not modify generic `src/tir/transforms/coproc_sync.cc` unless the API is generalized by topology, because that pass serves coprocessors beyond VTA.

- Scope: new/revised function in `vta/python/vta/transform.py`, registration in the pass list in `vta/python/vta/build_module.py`, focused Python/TIR and FSim tests.
- Risks: a naive four-call textual rewrite is wrong. Push and pop must remain balanced across loops/branches, queue-2 may need a legal no-op, existing dependencies may duplicate tokens, and latency/command footprint can grow.
- Offline gate: synthetic TIR containing both forbidden directions; post-pass invariant `all(abs(from-to)==1)`; balance every pushed/popped edge including control flow; randomized FSim scheduler/seeds; compare original schedules byte-for-byte or structurally when no forbidden edge exists; inspect encoded dependency flags.

Verdict: **most likely non-RTL route to retain asynchronous execution**, but requires compiler work rather than only TE scheduling.

### Path D — compile-flow weight-load lifetime/deduplication pass

Before `CoProcSync`, identify identical loop-invariant `VTALoadBuffer2D` weight transfers, reserve stable weight-SRAM ranges, and hoist/deduplicate loads only when no intervening write aliases or overwrites the range. If hoisting exposes `1 <-> 3`, combine it with Path C or a drain.

- Scope: VTA transform and pass order, allocation/range analysis, possibly GEMM weight-index rewriting, and extensive tests.
- Risks: alias/range soundness, SRAM capacity/banking, dynamic descriptors, interaction with `StorageRewrite` and UOP folding, and dependency legalization.
- Offline gate: interval/liveness proof per SRAM allocation; differential static DMA; post-pass dependency invariant; FSim correctness on padding/stride/tail cases; capacity and command-footprint reporting.

Verdict: **best candidate for a general thesis mechanism**, but largest implementation effort.

### Rejected shortcut — remove the runtime check

Do not delete `runtime.cc:1583-1584`. The hardware encoding and simulator have no direct STORE-to-LOAD channel, so deletion can misencode the edge or deadlock. A runtime state machine could theoretically defer the separated push/pop calls and materialize compute no-ops, but it is harder to reason about than a TIR pass and would hide compiler mistakes.

## Tests currently missing

The generic test `tests/python/unittest/test_tir_transform_coproc_sync.py:28-123` checks generic insertion and includes a supported `2 -> 3` case, but it has no VTA topology rejection/legalization case for `1 <-> 3`. `vta/tutorials/frontend/test_vta_residency_schedule.py:145-189` checks the recovered safe boundary and FSim modes, not the failed outer lifetime. Any implementation needs an explicit forbidden-edge regression test.

## Interpretation of the paper claim (inference, not fact)

The paper's high-level wording about a “compile-flow optimization” that reuses on-chip weight memory or prevents overwrite is consistent with one of: (a) a post-schedule load-lifetime/dedup pass around `InjectDMAIntrin`, (b) dependency/command-stream legalization after `CoProcSync`, or (c) schedule transformation plus explicit synchronization. This is an **engineering inference only** from the claimed behavior and the local VTA pipeline. Without the paper's released compiler/code or lower-level description, this audit cannot identify the exact layer and must not present reproduction as complete.

## Final go/no-go

- GO: Path A as a bounded offline semantic probe.
- GO: design and unit-test Path C before another large TE search.
- CONDITIONAL GO: Path D after a stable SRAM liveness model exists.
- NO-GO: outer `ckernel.compute_at` alone.
- NO-GO: deleting the runtime STORE/LOAD check or claiming a board/FPS result from this audit.

