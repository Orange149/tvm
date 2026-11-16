"""Bind finite window-audit events to one actual runtime instruction submission."""

from c3_five_tradeoff_commands import completion_dag, decode_submissions
from c3_valid_window_tps_audit import bind_dependencies, bind_material, require


def native_binding_details(events, text):
    """Return evidence plus event-indexed ancestors for material events.

    The static trace has no explicit UOP LOAD events. Runtime caching can put a
    pending dependency pop on a UOP LOAD preceding its GEMM. Per-lane binding
    keeps that actual completion point instead of inventing a GEMM pop bit.
    Control events are mapped in the evidence; their ancestor entries remain
    empty because the data/version checker compares material events only.
    """
    blocks = decode_submissions(text)
    require(len(blocks) == 1, "NATIVE_BINDING_REQUIRES_ONE_SUBMISSION")
    records = blocks[0]
    converted, kernels = [], []
    for eid, event in enumerate(events):
        kind = event["kind"]
        value = {"event": eid, "kind": kind}
        if kind == "kernel":
            uops = event["uops"]
            require(uops, "NATIVE_BINDING_EMPTY_KERNEL")
            loops = uops[0]["loops"]
            require(all(uop["loops"] == loops for uop in uops),
                    "NATIVE_BINDING_MIXED_HARDWARE_LOOPS")
            require(0 <= len(loops) <= 2, "NATIVE_BINDING_HARDWARE_LOOP_ARITY")
            # GEMM emits loop[0] as outer; any absent loop is an extent-one
            # zero-factor loop. ALU's one-loop convention is checked below.
            gemm_loops = [list(loop) for loop in loops]
            while len(gemm_loops) < 2:
                gemm_loops.append([1, 0, 0, 0])
            mode, reset = event["mode_reset"]
            value.update(kernel=len(kernels), mode=mode, reset=reset,
                         uops=len(uops), loops=gemm_loops)
            kernels.append(uops)
        elif kind in ("load", "store"):
            if kind == "load":
                values = [event["offset"], event["x"], event["y"], event["stride"],
                          *event["pad"], event["sram"], event["domain"]]
            else:
                values = [event["sram"], event["domain"], event["offset"],
                          event["x"], event["y"], event["stride"]]
            value.update(object=event["tensor"], values=values)
        elif kind in ("push", "pop"):
            value["lane"] = event["pair"]
        elif kind != "sync":
            raise ValueError("NATIVE_BINDING_UNKNOWN_EVENT: " + kind)
        converted.append(value)
    material = bind_material(converted, kernels, records)
    dependencies, drain, pairs = bind_dependencies(converted, records)
    native_ancestors = completion_dag(records)
    mapping = {item["tir_event"]: item["instruction"] for item in material}
    ancestors = [0] * len(events)
    for target, native_target in mapping.items():
        ancestors[target] = sum(1 << source for source, native_source in mapping.items()
                                if native_ancestors[native_target] & (1 << native_source))
    return dict(status="ACTUAL_NATIVE_FIFO_TOKEN_BINDING_PASSED", submission_count=1,
                instruction_count=len(records), ancestors=ancestors,
                material_map=material, dependency_map=dependencies,
                runtime_drain_events=drain, token_pairs=pairs,
                limits=["Control events have no synthesized ancestor entries",
                        "DumpInsn does not expose serialized UOP values or kernel signatures"])


def bind_native(events, text):
    """Return ancestors suitable for c3_valid_window.certify_trace."""
    return native_binding_details(events, text)["ancestors"]
