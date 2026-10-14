"""Experiment-only STORE -> Compute NOP -> LOAD bridge for node34 mode4.

The original mode4 schedule and its group-end coprocessor sync remain intact.
This pass runs *after* CoProcSync has closed each group, and replaces only the
target function's group-end sync, leaving its function-end sync unchanged.
It is not installed in production VTA build_config or the public mode registry.
"""

from __future__ import annotations

import json
from pathlib import Path

import tvm
from tvm import tir


SYMBOL = "tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast_5"


def assert_group_boundary(address_audit):
    """Require precisely 64 STORE-terminated groups and a final sync.

    The address audit expands TIR in host issue order, not device completion
    order.  The bridge below supplies the missing cross-queue completion edge.
    """
    events = address_audit["ordered_events"]
    positions = address_audit["sync_event_positions"]
    if address_audit["status"] != "passed" or len(positions) != 65:
        raise ValueError("node34 mode4 does not have 64 groups plus final sync")
    if positions[-2:] != [len(events), len(events)]:
        raise ValueError("node34 final sync shape changed")
    for index, position in enumerate(positions[:64]):
        if position != (index + 1) * 498:
            raise ValueError("node34 group length changed")
        if events[position - 1][0] != "VTAStoreBuffer2D":
            raise ValueError("group does not end in STORE")
        if index < 63 and events[position][0] != "VTALoadBuffer2D":
            raise ValueError("next group does not begin in LOAD")
    return {"groups": 64, "events_per_group": 498,
            "boundary": "last STORE -> first LOAD", "function_end_sync": True}


def make_pass(expected_symbol=SYMBOL):
    observations = []
    sync_op = tvm.ir.Op.get("tir.vta.coproc_sync")

    @tir.transform.prim_func_pass(opt_level=0, name="c3.Node34GroupTokenBridge")
    def replace_group_sync(function, *_):
        attrs = function.attrs
        if attrs is None or "global_symbol" not in attrs or str(attrs["global_symbol"]) != expected_symbol:
            return function
        seen = []

        def visit(stmt):
            if not isinstance(stmt, tir.Evaluate) or not isinstance(stmt.value, tir.Call):
                return None
            if not stmt.value.op.same_as(sync_op):
                return None
            seen.append(len(seen))
            if len(seen) > 1:
                return None  # Preserve the function-end completion boundary.
            # DepPush(STORE, COMPUTE) attaches to the last STORE.  The pending
            # DepPop(STORE, COMPUTE) is committed as a zero-length Compute NOP
            # before DepPush(COMPUTE, LOAD) attaches to that same NOP.  The
            # following DepPop(COMPUTE, LOAD) waits on the next group's LOAD.
            # Thus all four memory domains wait for the prior group's final
            # STORE, without a host VTADeviceRun per group.
            edges = ((3, 2, "push"), (3, 2, "pop"),
                     (2, 1, "push"), (2, 1, "pop"))
            return tir.SeqStmt([
                tir.Evaluate(tvm.tir.call_intrin(
                    "int32", "tir.vta.coproc_dep_" + kind, source, destination))
                for source, destination, kind in edges
            ])

        body = tir.stmt_functor.ir_transform(function.body, None, visit, ["tir.Evaluate"])
        if len(seen) != 2:
            raise ValueError("expected exactly one mode4 group sync and one function-end sync")
        observations.append({"symbol": expected_symbol,
                             "group_sync_replaced": 1, "function_end_sync_preserved": 1})
        return function.with_body(body)

    return replace_group_sync, observations


def build_config_with_bridge(expected_symbol=SYMBOL):
    """Return a fresh experiment-only PassContext and its change ledger."""
    import vta

    context = vta.build_config(
        opt_level=3, disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"})
    passes = list(context.config["tir.add_lower_pass"])
    positions = [i for i, (_, value) in enumerate(passes)
                 if str(value.info.name) == "tir.vta.InjectCoProcSync"]
    if len(positions) != 1:
        raise ValueError("unable to locate the exact CoProcSync lowering pass")
    bridge, observations = make_pass(expected_symbol)
    passes.insert(positions[0] + 1, (1, bridge))
    result = tvm.transform.PassContext(
        opt_level=3, config={"tir.add_lower_pass": passes},
        disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"})
    return result, observations


def fused_relay_from_mode4_evidence(evidence_path):
    from tvm import relay

    evidence = json.loads(Path(evidence_path).read_text())
    text = evidence["fusion"]["actual_relay_block"].split("} /* ty=fn", 1)[0] + "}"
    # The parsed standalone @main is a wrapper, not an internal Primitive.
    text = text.replace(", Primitive=1", "").replace("fn (", "def @main(", 1)
    return relay.parse('#[version = "0.0.5"]\n' + text)
