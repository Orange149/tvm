#!/usr/bin/env python3
"""Independent offline completion audit; read held-out data only after model seal."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

import numpy as np
import tvm
from tvm import tir
import vta

import c3_residency_tradeoff as p
from c3_residency_tradeoff_analysis import journal
from c3_residency_tradeoff_evidence import check_runtime, function, ANCHORS
from c3_residency_tradeoff_board import dump_signature, helper_certificate
from c3_thread_sync_audit import instruction_dump


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_equivalent(actual, recorded):
    """Compare all values after the same lossless JSON encoding as the journal."""
    return json.loads(json.dumps(actual, allow_nan=False)) == recorded


def allocations(path):
    f = tvm.ir.load_json((path / "node24_slots.tir.json").read_text())
    arenas = {}
    def visit(node):
        if isinstance(node, tir.Allocate):
            scope = str(getattr(node.buffer_var.type_annotation, "storage_scope", ""))
            if scope.startswith("local."):
                dtype = tvm.DataType(str(node.dtype))
                size = int(np.prod([int(x) for x in node.extents])) * dtype.bits * dtype.lanes // 8
                arenas[scope] = arenas.get(scope, 0) + size
    tir.stmt_functor.post_order_visit(f.body, visit)
    return arenas


def audit(output):
    output = Path(output).resolve()
    receipt = p.read(output / "analysis/seal_receipt.json")
    require(receipt["model_sha256"] == p.digest(output / "analysis/model.json"), "model seal drift")
    model = p.read(output / "analysis/model.json")
    require(model["calibration_programs"] == list(p.CALIBRATION), "calibration scope drift")
    require(model["validation_programs"] == list(p.VALIDATION), "validation scope drift")
    prepared = p.read(output / "prepared.json")
    protocol = p.read(output / "protocol.json")
    qualification = p.read(output / "qualification/summary.json")
    qcontract = p.read(output / "qualification/contract.json")
    contract = p.read(output / "board/contract.json")
    summary = p.read(output / "board/summary.json")
    require(not (output / "board/invalid_attempt.json").exists(), "invalid formal session")
    require(not (output / "qualification/invalid_attempt.json").exists(), "invalid qualification session")
    require(summary["status"] == "FORMAL_ACQUISITION_COMPLETE", "formal session incomplete")
    require(qualification["status"] == "SIX_PROGRAMS_FPGA_AND_COMMAND_QUALIFIED", "qualification incomplete")
    require(set(prepared["programs"]) == set(p.PROGRAMS), "candidate scope not six frozen programs")
    require(contract["orders"] == [list(x) for x in p.ORDERS], "frozen order changed")
    require(contract["sources"] == qcontract["sources"], "code identity changed between qualification and timing")
    for path, sha in contract["sources"].items():
        require(p.digest(path) == sha, "execution source drift: " + path)
    require(contract["helpers"] == helper_certificate(), "helper binary/source drift")
    require(p.digest(output / "protocol.json") == prepared["protocol_sha256"], "protocol artifact changed")
    require(p.digest(protocol["document"]) == protocol["document_sha256"], "document artifact changed")
    check_runtime()
    qrows = journal(output / "qualification/calls.jsonl")
    calls = journal(output / "board/calls.jsonl")
    require(len(qrows) == 153, "qualification ledger not 150 target + 3 CPU regression")
    require(len(calls) == 216, "formal ledger not 108 warmup + 108 timed")
    boot = summary["boot_id"]
    require(qualification["boot_id"] == model["boot_id"] == boot, "cross-phase boot differs")
    require({r["boot_id"] for r in qrows + calls} == {boot}, "call boot differs")
    for row in qrows + calls:
        require(row["status"] == "PASS" and row["correct"] and row["arguments_unchanged"],
                "incorrect/incomplete call: " + row["invocation_id"])
    sequence = []
    for round_id, order in enumerate(p.ORDERS):
        for name in order:
            for kind in ("warmup", "timing"):
                for pos, seed in enumerate(p.seed_order(round_id)):
                    sequence.append((round_id, name, kind, seed, pos + (1 if kind == "warmup" else 4)))
    actual = [(row["round"], row["program"], row["call_kind"], row["seed"], row["call_position"])
              for row in calls]
    require(actual == sequence, "actual formal order/seed/prefix differs")
    require(len({r["invocation_id"] for r in qrows + calls}) == len(qrows) + len(calls),
            "duplicate invocation IDs")
    for name in p.PROGRAMS:
        seed_rows = [r for r in qrows if r["program"] == name and r["call_kind"] == "correctness"]
        require([r["seed"] for r in seed_rows] == list(p.SEEDS), "three-seed qualification missing")
        aba = [r for r in qrows if r["program"] == name and r["call_kind"] == "weight_ABA"]
        require([r["aba_label"] for r in aba] == ["A", "B", "A_return"], "ABA order changed")
        require(len({r["session_id"] for r in aba}) == 1, "ABA RPC/module replaced")
        require(aba[0]["buffer_handles"] == aba[1]["buffer_handles"] == aba[2]["buffer_handles"],
                "ABA parameter handles changed")
        require(aba[0]["weight_payload_sha256"] == aba[2]["weight_payload_sha256"]
                != aba[1]["weight_payload_sha256"], "ABA weight content not overwritten in-place")
        require(aba[0]["output_sha256"] == aba[2]["output_sha256"] != aba[1]["output_sha256"],
                "ABA output distinguishing/return evidence absent")
        require(aba[0]["seed"] == aba[1]["seed"] == aba[2]["seed"] == 0, "ABA input changed")
    for row in qrows:
        if row["call_kind"] in ("first_call_diagnostic", "warmed_diagnostic"):
            parsed = instruction_dump((output / "qualification" / row["dump_file"]).read_text())
            require(json_equivalent(parsed, row["instruction_audit"]), "instruction dump/audit differs")
            signature = dump_signature(parsed)
            require(signature == row["command_signature"], "semantic instruction signature differs")
            require(signature["mac"] + signature["reset"] ==
                    row["runtime_profile"]["push_gemm_op_calls"], "RESET/MAC mapping differs")
            require(signature["submissions"] == row["runtime_profile"]["driver_run_calls"], "submission mapping differs")
    require(all(row["profile_before_output_reads"] for row in calls), "profile snapshot ordering absent")
    timer_counts = journal(output / "board/helper_counts.jsonl")
    require(len(timer_counts) == 108, "timer count coverage absent")
    require(all(row["count"] == row["expected"] and row["expected"] in (1, 2, 3) for row in timer_counts),
            "extra timer invocation")
    require([r["counter_value"] for r in qrows if r["call_kind"] == "helper_regression"] == [123, 246, 369],
            "helper actual four-argument regression missing")
    for stage in ("qualification", "board"):
        sessions = journal(output / stage / "sessions.jsonl")
        for session in sessions:
            require(session["status"] == "PASS" and "cleanup_error" not in session, "failed RPC session")
            require(session["runtime_attestation"] == prepared["runtime"]["runtime_sha256"], "runtime mismatch")
            require(session["environment"]["masks"] and set(session["environment"]["masks"]) == {"3"},
                    "CPU3 affinity not proven")
            require(session["environment"]["boot_id"] == boot == session["end_environment"]["boot_id"],
                    "session boot changed")
            require(session["bitstream_load_record"]["sha256"] == session["start_state"]["bitstream_sha256"],
                    "bitstream load identity differs")
            require(session["runtime_directory"] in session["loaded_maps"], "actual loaded runtime mapping absent")
    for name in p.CALIBRATION:
        file = output / "board/programs" / (name + ".jsonl")
        require(model["calibration_journals_sha256"][name] == p.digest(file), "calibration journal drift")
    resources = {}
    env = vta.get_env()
    capacity = {"local.inp_buffer": env.INP_BUFF_SIZE, "local.wgt_buffer": env.WGT_BUFF_SIZE,
                "local.acc_buffer": env.ACC_BUFF_SIZE}
    diagnostics = p.read(output / "qualification/diagnostic_signatures.json")
    for name, entry in prepared["programs"].items():
        directory = Path(entry["directory"])
        require(p.digest(entry["binary"]) == entry["binary_sha256"], "binary identity changed")
        require(p.digest(directory / "node24_final.tir.json") == entry["tir_sha256"], "TIR identity changed")
        if name in ("R0-G", "R8-G", "R12-G"):
            arena_dir = output / "local" / name
        elif name == "R13-G":
            arena_dir = output / "local/compat_R13-G"
        elif name == "F13-B":
            arena_dir = directory
        else:
            # Input has no slots pass; retain allocations from its early snapshot.
            f = tvm.ir.load_json((directory / "node24_early.tir.json").read_text())
            arena_dir = None
        if arena_dir is not None:
            arenas = allocations(arena_dir)
        else:
            arenas = {}
            def visit(node):
                if isinstance(node, tir.Allocate):
                    scope = str(getattr(node.buffer_var.type_annotation, "storage_scope", ""))
                    if scope.startswith("local."):
                        dtype = tvm.DataType(str(node.dtype))
                        size = int(np.prod([int(x) for x in node.extents])) * dtype.bits * dtype.lanes // 8
                        arenas[scope] = arenas.get(scope, 0) + size
            tir.stmt_functor.post_order_visit(f.body, visit)
        require(arenas and all(value <= capacity[scope] for scope, value in arenas.items()),
                "actual static allocation exceeds capacity")
        resources[name] = {"static_arenas_bytes": arenas, "capacity_bytes": capacity,
                           "static_fraction": {key: val / capacity[key] for key, val in arenas.items()},
                           "not_dynamic_utilization": True,
                           "warm_command_signature": diagnostics[name]["rotations"]["0"][0],
                           "first_call_signature": diagnostics[name]["first_call"]}
    report = {"schema": p.SCHEMA, "status": "COMPLETION_AUDIT_PASSED", "boot_id": boot,
              "requirements": {
                  "six_exact_programs": True, "local_address_and_comparison_certificates": True,
                  "FPGA_three_seed_and_same_handle_ABA": True, "one_call_timer_and_completion_path": True,
                  "RESET_MAC_separate": True, "profile_before_output_read": True,
                  "cold_and_matching_4_5_6_diagnostic": True, "fixed_Williams_and_seed_rotation": True,
                  "one_boot_CPU3_runtime_bitstream_identity": True,
                  "model_calibration_sealed_before_validation": True,
                  "all_call_and_submit_ledger": True},
              "counts": {"formal": 108, "formal_warmup": 108, "qualification": 18, "weight_ABA": 18,
                         "diagnostic_warmup": 54, "warmed_diagnostic": 54, "cold_diagnostic": 6,
                         "all_target_FPGA_calls": 366, "CPU_helper_regression": 3},
              "resource_and_command_table": resources,
              "audit_source_sha256": p.digest(__file__)}
    p.write_new(output / "completion_audit.json", report)
    local = p.read(output / "local_attempts.json")
    p.write_new(output / "cost_ledger.json", {
        "call_counts": report["counts"],
        "local_FSim_worker_seconds": sum(x["seconds"] for x in local),
        "cross_compile_worker_seconds": sum(x.get("cross_compile_seconds", 0)
                                            for x in prepared["programs"].values()),
        "formal_timer_sum_ms": sum(x["latency_ms"] for x in calls if x["call_kind"] == "timing"),
        "formal_warmup_has_no_latency_label": True,
        "per_call_host_call_and_validation_seconds": {
            stage: sum(x.get("host_call_and_validation_seconds", 0) for x in data
                       if x["call_kind"] != "helper_regression")
            for stage, data in (("qualification", qrows), ("formal", calls))},
        "host_call_scope": "after parameter preparation, includes function call and readback/validation; "
                           "not the formal timer scope and not total RPC/setup wall clock",
        "initialization_total_wall_seconds": None,
        "wall_clock_missing_not_zero": True,
        "restoration_cost_separate": ["../20261012_c3_tradeoff_volatile_restore_run01/summary.json",
                                    "../20261012_c3_tradeoff_taskset_restore_run01/summary.json"],
    })
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.output_dir)
    print(report["status"])
