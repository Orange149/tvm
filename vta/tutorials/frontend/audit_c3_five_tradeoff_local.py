#!/usr/bin/env python3
"""Fail-closed local certificate for the one declared two-slot candidate."""

import argparse
from pathlib import Path

import tvm

from c3_residency_tradeoff import read, digest, write_new
from c3_residency_tradeoff_evidence import (check_runtime, nonweight_events,
                                           compute_without_wgt)
from c3_five_tradeoff_commands import certify_prefetch_dump
from c3_five_tradeoff_queue import certify_stream_window


def require_normal_debug_modes(address):
    if "debug_mode_transitions" not in address or address["debug_mode_transitions"] != []:
        raise ValueError("runtime mode transitions missing or non-normal")


def audit(root):
    root = Path(root).resolve()
    path = root / "local/prefetch_attempt03"
    old = root.parent / "20261012_c3_residency_instruction_tradeoff_run01/local/R12-G"
    row, before = read(path / "result.json"), read(old / "result.json")
    if row["request_totals"] != before["request_totals"]:
        raise ValueError("R12 two-slot changed logical data request totals")
    if row["request_totals"]["load_buffer_2d_wgt_calls"] != 40 or \
            row["request_totals"]["load_buffer_2d_wgt_bytes"] != 720 * 1024:
        raise ValueError("weight request contract differs")
    address, old_address = read(path / "address_audit.json"), read(old / "address_audit.json")
    if nonweight_events(address) != nonweight_events(old_address):
        raise ValueError("nonweight ordered request contract changed")
    if compute_without_wgt(address) != compute_without_wgt(old_address):
        raise ValueError("actual reduction/RESET/ALU order changed")
    require_normal_debug_modes(address)
    if address["sync_event_positions"] != old_address["sync_event_positions"]:
        # Numeric event positions may move with reordered WGT commands, but
        # complete old/new nonweight and kernel equality requires same total
        # events; no sync may be silently added/dropped.
        raise ValueError("explicit synchronization boundary changed")
    if row["max_kernel_uops"] != 224 or row["address_status"] != "passed":
        raise ValueError("effective address/kernel evidence incomplete")
    if [x["seed"] for x in row["fsim"]] != [0, 20250901, 20260910, 0] or not all(
            x["correct"] and x["arguments_unchanged"] for x in row["fsim"]):
        raise ValueError("full fusion FSim/argument stability missing")
    if row["fsim"][0]["actual_sha256"] != row["fsim"][-1]["actual_sha256"] or \
            row["fsim"][0]["expected_sha256"] == row["fsim"][1]["expected_sha256"]:
        raise ValueError("local repeated weight update does not distinguish/return A")
    # Local test changes both X and W with seed; it does not substitute for
    # the required FPGA fixed-X same-W-handle ABA qualification.
    native_path = root / "local/prefetch_command_diagnostic02.log"
    native = certify_prefetch_dump(native_path.read_text())
    f = tvm.ir.load_json((path / "node24_slots.tir.json").read_text())
    arenas = []
    from tvm import tir
    tir.stmt_functor.post_order_visit(f.body, lambda n: arenas.append(n)
        if isinstance(n, tir.Allocate) and
        str(getattr(n.buffer_var.type_annotation, "storage_scope", "")) == "local.wgt_buffer" else None)
    if len(arenas) != 1 or list(map(int, arenas[0].extents)) != [252 * 1024]:
        raise ValueError("actual arena is not 252 KiB")
    diag_function = tvm.ir.load_json((root / "local/prefetch_command_diagnostic02/node24_final.tir.json").read_text())
    plain_function = tvm.ir.load_json((path / "node24_final.tir.json").read_text())
    if not tvm.ir.structural_equal(diag_function, plain_function, map_free_vars=True):
        raise ValueError("diagnostic and FSim source program differ")
    certificate = {
        "status": "LOCAL_FUSION_ADDRESS_AND_NATIVE_QUEUE_CERTIFIED_BOARD_PENDING",
        "candidate_id": "N24-R12-G2-PREFETCH",
        "builder_route_label_not_full_identity": row["route"]["candidate_id"],
        "final_tir_sha256": digest(path / "node24_final.tir.json"),
        "reference_tir_sha256": digest(old / "node24_final.tir.json"),
        "math_and_nonweight_requests_matched": True, "weight_payload_bytes": 720 * 1024,
        "weight_loads": 40, "WGT_static_allocation_bytes": 252 * 1024,
        "single_kernel_uops": 224, "native_dump_sha256": digest(native_path),
        "source_bindings": {str(Path(__file__).parent / name): digest(Path(__file__).parent / name)
                            for name in ("c3_five_tradeoff_prefetch.py", "c3_five_tradeoff_queue.py",
                                         "c3_five_tradeoff_commands.py", "audit_c3_five_tradeoff_local.py")},
        "runtime_source_and_board_binary_check": check_runtime(),
        "local_FSim_runtime_binary_sha256": digest(Path(__file__).resolve().parents[3] / "build/libvta_fsim.so"),
        "stream_interleavings": certify_stream_window(), "native_queue": native,
        "local_same_graph_ABA_changes_X_and_W_not_FPGA_ABA_certificate": True,
        "FPGA_ABA_correctness_and_actual_command_dump_required": True,
        "latency_ms": None, "board_contacted": False,
    }
    certificate["certificate_version"] = 2
    certificate["supersedes"] = "local_qualification.json; same measured data, mandatory debug_mode_transitions check"
    write_new(root / "local_qualification_v2.json", certificate)
    return certificate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = audit(parser.parse_args().output_dir)
    print(result["status"])
