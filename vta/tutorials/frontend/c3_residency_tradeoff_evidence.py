"""Read-only old-anchor, final-address, and fixed group-dependency certificates."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import time

import tvm
from tvm import tir
import vta

import c3_residency_tradeoff as p
from c3_thread_sync_audit import CAPACITY, ELEMENT_BYTES, final_tir_audit
from run_vta_c3_node24_group_dma_audit import _events, _integer
from run_vta_c3_node24_partial_board import RUNTIME, GROUPED, SYMBOL


HERE = Path(__file__).resolve().parent
OLD = GROUPED / "20261009_c3_node24_partial_run01"
DMA = GROUPED / "20261009_c3_node24_group_dma_run01"
GEMM = GROUPED / "20261009_c3_node24_group_gemm_run01"
ANCHORS = {
    "R13-G": GEMM / "binaries/f13_g",
    "F13-B": DMA / "binaries/f13_b",
    "Input-7x7": OLD / "binaries/input_alt",
}


def function(path):
    return tvm.ir.load_json((path / "node24_final.tir.json").read_text())


def nonweight_events(audit):
    return [event for event in audit["ordered_events"] if event[0] != "kernel"
            and not (event[0] == "VTALoadBuffer2D" and event[1][-1] == 1)]


def compute_without_wgt(audit):
    kernels = deepcopy(audit["kernels"])
    for kernel in kernels:
        for op in kernel:
            if op["args"][0:2] == [0, 0]:
                # The only allowed numeric computation difference across R is WGT base.
                op["args"][4] = "WGT_BASE_ALLOWED_DIFFERENCE"
    return kernels


def check_runtime():
    contract = p.read(RUNTIME)
    required = [HERE.parents[1] / "runtime/runtime.cc",
                HERE.parents[2] / "3rdparty/vta-hw/config/vta_config.json",
                HERE.parents[2] / "3rdparty/vta-hw/include/vta/hw_spec.h",
                HERE.parents[2] / "3rdparty/vta-hw/include/vta/hw_spec_const.h",
                HERE.parents[2] / "3rdparty/vta-hw/src/axu5evb/axu5evb_driver.cc"]
    for path in required:
        if p.digest(path) != contract["source_bindings"].get(str(path)):
            raise ValueError("frozen runtime/hardware source identity mismatch: " + str(path))
    for name, path in contract["runtime_paths"].items():
        if p.digest(path) != contract["runtime_sha256"][name]:
            raise ValueError("frozen runtime binary identity mismatch: " + name)
    env = vta.get_env()
    actual = {"INP": env.INP_BUFF_SIZE // env.INP_ELEM_BYTES,
              "WGT": env.WGT_BUFF_SIZE // env.WGT_ELEM_BYTES,
              "ACC": env.ACC_BUFF_SIZE // env.ACC_ELEM_BYTES,
              "OUT": env.ACC_BUFF_SIZE // env.ACC_ELEM_BYTES,
              "UOP": env.UOP_BUFF_SIZE // 4}
    if actual != CAPACITY or any(int(getattr(env, name + "_ELEM_BYTES")) != value
                                for name, value in ELEMENT_BYTES.items() if name != "UOP"):
        raise ValueError("address-audit units do not match frozen hardware")
    return {"status": "PASSED", "capacity_elements": actual,
            "element_bytes": ELEMENT_BYTES, "runtime_sha256": contract["runtime_sha256"],
            "source_sha256": {str(path): p.digest(path) for path in required},
            "uop_evidence": "final kernel lengths and all loop-adjusted addresses; "
                            "frozen UopQueue allocation/FlushUopLoad and normal synchronization; "
                            "actual LOAD UOP/reference intervals required again in FPGA diagnostic"}


def group_queue_certificate(f, r):
    loops = {}
    tir.stmt_functor.post_order_visit(f.body, lambda node: loops.setdefault(str(node.loop_var), []).append(node)
                                    if isinstance(node, tir.For) else None)
    if any(len(loops.get(name, [])) != 1 for name in ("ax2_outer", "c_o")):
        raise ValueError("grouped loop identity is not unique")
    spatial, group = loops["ax2_outer"][0], loops["c_o"][0]
    if [int(spatial.extent), int(group.extent)] != [7, 16] or \
            not isinstance(group.body, tir.SeqStmt) or len(group.body.seq) != 4:
        raise ValueError("not frozen reset/group-LOAD/merged-MAC/final-pop")
    pieces = list(group.body.seq)
    cases, slots = [], {}
    for block in range(7):
        for g in range(16):
            loaded = block == 0 or g >= r
            expected = [
                [("push", (2, 1))],
                [("pop", (2, 1)), ("WGT_LOAD", ()), ("push", (1, 2))] if loaded else [],
                ([("pop", (1, 2))] if loaded else []) + [("GEMM", ())] * 8
                + ([("push", (2, 1))] if loaded else []),
                [("pop", (2, 1))],
            ]
            actual = [_events(piece, spatial, group, block, g) for piece in pieces]
            if actual != expected:
                raise ValueError("ordered token production/consumption changed at p%d g%d" % (block, g))
            slot = min(g, r)
            if loaded:
                slots[slot] = g
            if slots.get(slot) != g:
                raise ValueError("stream overwrite or missing first-use load")
            cases.append({"spatial_block": block, "group": g, "slot": slot, "loaded": loaded})
    # One pop is supplied by RESET when no LOAD exists, or by last MAC when LOAD exists.
    # Compute FIFO orders that last MAC before the next reset.  Next LOAD pops that
    # reset completion before overwriting the shared WGT slot.  The unchanged
    # spatial bridge separately closes Store->Compute->Load between input tiles.
    return {"status": "PASSED", "cases": cases,
            "proof": "normal Compute FIFO: last MAC->next RESET completion->WGT overwrite; "
                     "LOAD completion->first MAC; skipped resident LOAD removes matching "
                     "handshake, retains RESET->final pop; spatial bridge and final sync preserved",
            "not_cycle_trace": True}


def audit_local(path, r, reference_path):
    f = function(path)
    requests = p.read(path / "request_semantics.json")
    audit = final_tir_audit(tvm.IRModule({"main": f}), requests)
    record = p.read(path / "result.json")
    exp = p.expectations("R%d-G" % r)
    if record["request_totals"]["load_buffer_2d_wgt_calls"] != exp["wgt_loads"] or \
            record["request_totals"]["load_buffer_2d_wgt_bytes"] != exp["wgt_payload_bytes"]:
        raise ValueError("request model mismatch")
    old = p.read(reference_path / "address_audit.json")
    if nonweight_events(audit) != nonweight_events(old) or \
            compute_without_wgt(audit) != compute_without_wgt(old):
        raise ValueError("unrelated input/ACC/output/computation changed across R")
    if len(audit["sync_event_positions"]) != len(old["sync_event_positions"]):
        raise ValueError("unexpected additional explicit synchronization")
    if r == 13 and not tvm.ir.structural_equal(f, function(reference_path), map_free_vars=True):
        raise ValueError("new R13 branch is not structurally equal to frozen R13 anchor")
    queue = group_queue_certificate(f, r)
    p.write_new(path / "comparison_certificate.json", {
        "status": "PASSED", "resident_groups": r, "reference": str(reference_path),
        "reference_tir_sha256": p.digest(reference_path / "node24_final.tir.json"),
        "new_tir_sha256": p.digest(path / "node24_final.tir.json"),
        "exact_R13_structure": r == 13, "nonweight_requests_matched": True,
        "compute_except_wgt_base_matched": True, "queue_certificate": queue,
        "max_kernel_uops": audit["max_kernel_uops"],
    })


def finish_prepare(output):
    runtime = check_runtime()
    bindings = {}
    reference = ANCHORS["R13-G"]
    for name, directory in ANCHORS.items():
        record = p.read(directory / "result.json")
        if p.digest(directory / "graphlib.so") != record.get("graphlib_sha256") or \
                record.get("address_status") != "passed":
            raise ValueError("frozen anchor identity/address certificate mismatch: " + name)
        bindings[name] = {"directory": str(directory), "binary": str(directory / "graphlib.so"),
                          "binary_sha256": p.digest(directory / "graphlib.so"),
                          "tir_sha256": p.digest(directory / "node24_final.tir.json"),
                          "candidate_id": record["route"]["candidate_id"], "old_binary": True}
    for r in (13, 0, 8, 12):
        directory = output / "local" / ("compat_R13-G" if r == 13 else "R%d-G" % r)
        audit_local(directory, r, reference)
    old, new = p.read(ANCHORS["F13-B"] / "address_audit.json"), p.read(reference / "address_audit.json")
    if nonweight_events(old) != nonweight_events(new) or \
            p.read(ANCHORS["F13-B"] / "request_semantics.json")["request_groups"] != \
            p.read(reference / "request_semantics.json")["request_groups"]:
        raise ValueError("F13-B/R13-G ordered data contract mismatch")
    # Check the complete transformation itself, not just request totals or counts.
    from vta.top.vta_node24_partial import Node24GroupGEMM
    old_f = function(ANCHORS["F13-B"])
    transformed = Node24GroupGEMM()(tvm.IRModule({"main": old_f}))["main"]
    if not tvm.ir.structural_equal(transformed, function(reference), map_free_vars=True):
        raise ValueError("old organization anchor differs by more than frozen GEMM merge")
    p.write_new(output / "organization_certificate.json", {
        "status": "PASSED", "old": bindings["F13-B"], "new": bindings["R13-G"],
        "transform_structurally_reproduces_new_body": True,
        "ordered_requests_equal": True, "nonweight_events_equal": True})
    for r in (0, 8, 12):
        name, directory = "R%d-G" % r, output / "binaries" / ("R%d-G" % r)
        command = [sys.executable, str(HERE / "run_vta_c3_residency_tradeoff.py"), "worker",
                   "--output-dir", str(directory), "--resident-groups", str(r), "--board-compile"]
        sdk = "/home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux"
        import shlex
        # SDK affects only this worker, never the parent or existing services.
        shell = "unset LD_LIBRARY_PATH; source " + shlex.quote(sdk) + "; exec " + shlex.join(command)
        started = time.monotonic()
        with (output / ("cross_R%d.log" % r)).open("x") as log:
            process = subprocess.run(["bash", "-c", shell], stdout=log, stderr=subprocess.STDOUT,
                                     check=False, timeout=1200)
        if process.returncode:
            raise ValueError("cross compilation failed; attempt retained")
        local_f = function(output / "local" / name)
        if not tvm.ir.structural_equal(local_f, function(directory), map_free_vars=True):
            raise ValueError("local/board compilation changed final TIR")
        row = p.read(directory / "result.json")
        bindings[name] = {"directory": str(directory), "binary": str(directory / "graphlib.so"),
                          "binary_sha256": p.digest(directory / "graphlib.so"),
                          "tir_sha256": p.digest(directory / "node24_final.tir.json"),
                          "candidate_id": row["route"]["candidate_id"], "old_binary": False,
                          "cross_compile_seconds": time.monotonic() - started}
    p.write_new(output / "prepared.json", {
        "schema": p.SCHEMA, "status": "ALL_LOCAL_GATES_PASSED_BOARD_NOT_QUALIFIED",
        "programs": bindings, "runtime": runtime,
        "aliases": {"R13-G": "historical isolated f13_g; same binary",
                    "F13G_graph": "historical graph packaging, not a board arm"},
        "protocol_sha256": p.digest(output / "protocol.json"),
    })
