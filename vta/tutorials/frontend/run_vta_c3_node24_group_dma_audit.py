#!/usr/bin/env python3
"""Fail-closed read-only audit for batching node24's eight WGT loads.

It checks the frozen F13 mapping/lifetime and both new programs' symbolic
address and queue-token structure.  It does not claim cycle-level overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import tvm
from tvm import tir
import vta

from vta.top.vta_staged_weight import _calls, _integer, _weight_load_info


HERE = Path(__file__).resolve().parent
ROOT = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
OLD = ROOT / "20261009_c3_node24_partial_run01/binaries/f13/node24_final.tir.json"
RUNTIME_CONTRACT = ROOT / "20260922_c3_thread_sync_freeze_run04/contract.json"
RUNTIME_SOURCE = HERE.parents[1] / "runtime/runtime.cc"
PROTOTYPE = ROOT / "20261009_c3_node24_group_dma_run01/binaries"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _guarded_calls(stmt, guards=()):
    """Collect calls and their enclosing control-flow guards, in statement order."""
    if isinstance(stmt, tir.SeqStmt):
        for item in stmt.seq:
            yield from _guarded_calls(item, guards)
    elif isinstance(stmt, tir.AttrStmt):
        yield from _guarded_calls(stmt.body, guards)
    elif isinstance(stmt, tir.For):
        yield from _guarded_calls(stmt.body, guards)
    elif isinstance(stmt, tir.IfThenElse):
        yield from _guarded_calls(stmt.then_case, guards + (stmt.condition,))
        if stmt.else_case is not None:
            yield from _guarded_calls(stmt.else_case, guards + (tir.Not(stmt.condition),))
    elif isinstance(stmt, tir.Evaluate) and isinstance(stmt.value, tir.Call):
        yield stmt.value, guards


def _enabled(guards, spatial, group, p, g):
    substitutions = {spatial.loop_var: tir.IntImm("int32", p),
                     group.loop_var: tir.IntImm("int32", g)}
    analyzer = tvm.arith.Analyzer()
    return all(_integer(analyzer.simplify(tir.stmt_functor.substitute(
        guard, substitutions)), "DMA/token guard") != 0 for guard in guards)


def _events(stmt, spatial, group, p, g):
    result = []
    for call, guards in _guarded_calls(stmt):
        if not _enabled(guards, spatial, group, p, g):
            continue
        name = call.op.name
        if name in ("tir.vta.coproc_dep_pop", "tir.vta.coproc_dep_push"):
            result.append((name.split("_")[-1], tuple(_integer(x, "token lane") for x in call.args)))
        elif _weight_load_info(call) is not None:
            result.append(("WGT_LOAD", ()))
        elif name == "tir.vta.uop_push" and _integer(call.args[0], "GEMM mode") == 0 and \
                _integer(call.args[1], "GEMM reset") == 0:
            result.append(("GEMM", ()))
    return result


def audit_prototype(name):
    path = PROTOTYPE / name / "node24_final.tir.json"
    function = tvm.ir.load_json(path.read_text())
    loops = {}
    tir.stmt_functor.post_order_visit(
        function.body, lambda node: loops.setdefault(str(node.loop_var), []).append(node)
        if isinstance(node, tir.For) and str(node.loop_var) in ("ax2_outer", "c_o") else None,
    )
    if any(len(loops.get(x, [])) != 1 for x in ("ax2_outer", "c_o")):
        raise ValueError("prototype loops changed")
    spatial, group = loops["ax2_outer"][0], loops["c_o"][0]
    if not isinstance(group.body, tir.SeqStmt) or len(group.body.seq) != 11:
        raise ValueError("prototype not reset/load/eight MACs/final pop")
    pieces = list(group.body.seq)
    for p in range(7):
        for g in range(16):
            loaded = name == "a13_b" or p == 0 or g >= 13
            expected = [
                [("push", (2, 1))],
                ([("pop", (2, 1)), ("WGT_LOAD", ()), ("push", (1, 2))]
                 if loaded else []),
            ]
            expected.extend([
                ([("pop", (1, 2))] if loaded and k == 0 else []) +
                [("GEMM", ())] +
                ([("push", (2, 1))] if loaded and k == 7 else [])
                for k in range(8)
            ])
            expected.append([("pop", (2, 1))])
            actual = [_events(piece, spatial, group, p, g) for piece in pieces]
            if actual != expected:
                raise ValueError("prototype queue token/order mismatch: {} p{} g{} {} != {}".format(
                    name, p, g, actual, expected))
    loads = _calls(function.body, lambda c: _weight_load_info(c) is not None)
    macs = _calls(function.body, lambda c: isinstance(c, tir.Call) and
                  c.op.name == "tir.vta.uop_push" and
                  _integer(c.args[0], "GEMM mode") == 0 and
                  _integer(c.args[1], "GEMM reset") == 0)
    if len(loads) != 1 or len(macs) != 8:
        raise ValueError("prototype has unexpected WGT LOAD or GEMM read")
    load = loads[0]
    if tuple(_weight_load_info(load)[x] for x in ("x_size", "y_size", "stride")) != (9, 8, 9):
        raise ValueError("prototype full-group descriptor changed")
    for g in range(16):
        substitutions = {group.loop_var: tir.IntImm("int32", g)}
        source = _integer(tir.stmt_functor.substitute(load.args[3], substitutions), "source")
        target = _integer(tir.stmt_functor.substitute(load.args[11], substitutions), "destination")
        if (source, target) != (g * 72, min(g, 13) * 72):
            raise ValueError("prototype group DMA source/destination drift")
        for k, mac in enumerate(macs):
            index = _integer(tir.stmt_functor.substitute(mac.args[4], substitutions), "GEMM WGT")
            if index != min(g, 13) * 72 + k * 9:
                raise ValueError("prototype GEMM reads a different WGT address")
    return {"final_tir_sha256": digest(path), "cases_checked": 7 * 16,
            "wgt_load_sites": 1, "gemm_read_sites": 8,
            "queue_argument": "load: reset token→DMA→first GEMM; last GEMM completion→final pop; "
                              "skip: reset token→final pop; Compute FIFO orders next reset "
                              "after all eight GEMMs before a streaming-slot overwrite"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    contract = json.loads(RUNTIME_CONTRACT.read_text())
    if digest(RUNTIME_SOURCE) != contract["source_bindings"].get(str(RUNTIME_SOURCE)):
        raise ValueError("current runtime source differs from frozen runtime")
    for name, path in contract["runtime_paths"].items():
        if digest(Path(path)) != contract["runtime_sha256"][name]:
            raise ValueError("frozen runtime binary drift: " + name)
    source = RUNTIME_SOURCE.read_text()
    if ("case VTA_MEM_ID_WGT:" not in source or
            "elem_bytes = VTA_WGT_ELEM_BYTES;" not in source or
            "next.sram_base = dst_sram_index;" not in source or
            "next.x_stride = x_stride;" not in source):
        raise ValueError("frozen runtime WGT DMA semantics unrecognized")
    env = vta.get_env()
    element_bytes = int(env.WGT_ELEM_BYTES)
    if element_bytes != 256 or int(env.WGT_BUFF_SIZE) != 262144:
        raise ValueError("node24 hardware WGT element/capacity changed")
    function = tvm.ir.load_json(OLD.read_text())
    loops = {}
    tir.stmt_functor.post_order_visit(
        function.body,
        lambda node: loops.setdefault(str(node.loop_var), []).append(node)
        if isinstance(node, tir.For) and str(node.loop_var) in ("ax2_outer", "c_o") else None,
    )
    if any(len(loops.get(name, [])) != 1 for name in ("ax2_outer", "c_o")):
        raise ValueError("spatial/group loop no longer unique")
    spatial, group = loops["ax2_outer"][0], loops["c_o"][0]
    if [_integer(loop.extent, "loop extent") for loop in (spatial, group)] != [7, 16]:
        raise ValueError("node24 geometry changed")
    nested = []
    tir.stmt_functor.post_order_visit(
        spatial.body, lambda node: nested.append(node)
        if isinstance(node, tir.For) and node.same_as(group) else None,
    )
    if len(nested) != 1 or not isinstance(group.body, tir.SeqStmt) or len(group.body.seq) != 18:
        raise ValueError("not spatial→group→eight load/MAC pairs")
    seq = list(group.body.seq)
    pairs = []
    for k in range(8):
        load_node, mac_node = seq[1 + 2 * k], seq[2 + 2 * k]
        loads = _calls(load_node, lambda x: _weight_load_info(x) is not None)
        macs = _calls(mac_node, lambda x: isinstance(x, tir.Call) and
                      x.op.name == "tir.vta.uop_push" and
                      _integer(x.args[0], "GEMM mode") == 0 and
                      _integer(x.args[1], "GEMM reset") == 0)
        if len(loads) != 1 or len(macs) != 1:
            raise ValueError("group body is not eight strict WGT LOAD→GEMM pairs")
        load, mac = loads[0], macs[0]
        info = _weight_load_info(load)
        if (info["x_size"], info["y_size"], info["stride"]) != (9, 1, 9):
            raise ValueError("old WGT DMA shape changed")
        for g in range(16):
            old_source = _integer(
                tir.stmt_functor.substitute(load.args[3], {group.loop_var: tir.IntImm("int32", g)}),
                "old WGT source element",)
            old_target = _integer(
                tir.stmt_functor.substitute(load.args[11], {group.loop_var: tir.IntImm("int32", g)}),
                "old WGT SRAM element",)
            mac_target = _integer(
                tir.stmt_functor.substitute(mac.args[4], {group.loop_var: tir.IntImm("int32", g)}),
                "GEMM WGT base element",)
            slot = min(g, 13)
            if (old_source, old_target, mac_target) != \
                    (g * 72 + k * 9, slot * 72 + k * 9, slot * 72 + k * 9):
                raise ValueError("WGT source, SRAM destination, or GEMM base drift")
        pairs.append({"subblock": k, "source_delta_elements": k * 9,
                      "target_delta_elements": k * 9,
                      "width_elements": 9, "bytes": 9 * element_bytes})
    all_loads = _calls(function.body, lambda x: _weight_load_info(x) is not None)
    all_macs = _calls(function.body, lambda x: isinstance(x, tir.Call) and
                      x.op.name == "tir.vta.uop_push" and
                      _integer(x.args[0], "GEMM mode") == 0 and
                      _integer(x.args[1], "GEMM reset") == 0)
    if len(all_loads) != 8 or len(all_macs) != 8:
        raise ValueError("unexpected additional WGT LOAD or GEMM read")
    # Abstract execution of the frozen loop order.  The slot holds one whole
    # group in the proposed form; a tag mismatch is a constructive overwrite
    # counterexample.  Queue-level safety is separately required later.
    slots = {}
    consumed = {}
    stream_transitions = []
    for p in range(7):
        for g in range(16):
            slot = min(g, 13)
            need_load = p == 0 or g >= 13
            if need_load:
                previous = slots.get(slot)
                previous_consumed = previous is None or consumed.get(slot) == 8
                if not previous_consumed:
                    raise ValueError("stream group overwritten before its eighth GEMM")
                slots[slot] = [(g, k) for k in range(8)]
                consumed[slot] = 0
                if slot == 13:
                    stream_transitions.append({
                        "spatial_block": p, "group": g,
                        "previous_group": previous[0][0] if previous else None,
                        "slot_element_range": [13 * 72, 14 * 72],
                        "load_before_first_gemm": True,
                        "overwrite_after_previous_eighth_gemm": previous_consumed,
                    })
            for k in range(8):
                if slots.get(slot, [None] * 8)[k] != (g, k):
                    raise ValueError("stream slot overwritten before subblock use")
                consumed[slot] += 1
    report = {
        "status": "MAPPING_AND_ABSTRACT_LIFETIME_PASSED_QUEUE_PROOF_PENDING",
        "old_final_tir_sha256": digest(OLD),
        "runtime_source_sha256": digest(RUNTIME_SOURCE),
        "runtime_binary_sha256": contract["runtime_sha256"],
        "wgt_element_bytes": element_bytes, "wgt_capacity_bytes": int(env.WGT_BUFF_SIZE),
        "resident_groups": list(range(13)), "streaming_groups": [13, 14, 15],
        "old_per_subblock_dma": pairs,
        "proposed_group_dma": {"x_size": 9, "y_size": 8, "x_stride": 9,
                               "payload_bytes": 9 * 8 * element_bytes},
        "stream_transitions": stream_transitions,
        "expected_a13_b_loads_if_queue_safe": 16 * 7,
        "expected_f13_b_loads_if_queue_safe": 13 + 3 * 7,
        "explicit_non_claim": "Does not establish device-cycle overlap or measured wait cycles",
    }
    report["prototype_queue_and_address"] = {
        name: audit_prototype(name) for name in ("a13_b", "f13_b")
    }
    report["status"] = "MAPPING_LIFETIME_AND_PROTOTYPE_QUEUE_STRUCTURE_PASSED"
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# Node24 whole-group DMA preflight", "",
             "Status: mapping, abstract single-slot lifetime, and prototype queue structure passed.",
             "", "The frozen TIR is spatial→group→eight ordered WGT LOAD/GEMM pairs.",
             "Groups 13–15 share slot 13, but each group consumes all eight subblocks before the next overwrite.",
             "Each old DMA covers nine 256-byte WGT elements at source/group and SRAM/slot offsets `9k`.",
             "A proposed `x=9,y=8,stride=9` descriptor maps to the same 72 elements (18 KiB).", "",
             "The prototype retains one Load→Compute handshake and gates the next overwrite "
             "on the eighth GEMM's Compute→Load completion. All 112 spatial/group cases were checked "
             "for both A13-B and F13-B.",
             "This is a static queue-structure argument, not a device-cycle timing trace or "
             "a substitute for FPGA correctness."]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": report["status"], "stream_transitions": len(stream_transitions)}))


if __name__ == "__main__":
    main()
