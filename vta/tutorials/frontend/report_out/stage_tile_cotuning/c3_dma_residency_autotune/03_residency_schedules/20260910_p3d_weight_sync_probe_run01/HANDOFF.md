# HANDOFF

The schedule-only explicit-drain path is frozen as a negative result. Mode 4 is a probe,
not a production candidate identity or performance winner.

Two independent blockers are demonstrated:

1. The explicit pragma path in `InjectCoProcSync` uses the old string call form
   `tvm.tir.Call("int32", "vta.coproc_sync", [])`, which current TVM rejects before
   dependency insertion.
2. Even when that call is removed only for counterfactual inspection, generic
   `CoProcInstDepDetector` does not treat a full drain as a queue-state boundary and still
   emits `3 -> 1` push/pop pairs. They are numerically balanced but unsupported by VTA.

Do not delete the runtime check and do not claim FSim correctness for mode 4. A future
positive implementation requires at least a transform/build-flow change that both creates
the intrinsic with `tir.vta.coproc_sync` and resets/legalizes dependency state across it;
that is outside this schedule-only probe.

