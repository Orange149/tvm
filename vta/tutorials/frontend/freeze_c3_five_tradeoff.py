#!/usr/bin/env python3
"""Seal static features, equal-budget prefixes and identities before new FPGA latency."""

import argparse
from pathlib import Path

import tvm

from c3_residency_tradeoff import read, digest, write_new
from c3_five_tradeoff_policy import PROGRAMS, NEW, prefixes
from c3_residency_tradeoff_evidence import check_runtime


def freeze(root):
    root = Path(root).resolve()
    old = root.parent / "20261012_c3_residency_instruction_tradeoff_run01"
    old_manifest = read(old / "closeout_manifest.json")
    for name, sha in old_manifest["artifacts"].items():
        if digest(old / name) != sha:
            raise ValueError("historical artifact changed: " + name)
    prepared = read(old / "prepared.json")
    resources = read(old / "completion_audit.json")["resource_and_command_table"]
    certificate = read(root / "local_qualification_v2.json")
    entries, features = {}, {}
    for name in PROGRAMS:
        directory = root / "binaries/prefetch_compile01" if name == NEW else Path(prepared["programs"][name]["directory"])
        record = read(directory / "result.json")
        binary = directory / "graphlib.so"
        if digest(binary) != record["graphlib_sha256"] or record["address_status"] != "passed":
            raise ValueError("unbound binary or unqualified address")
        if name == NEW:
            plain = tvm.ir.load_json((root / "local/prefetch_attempt03/node24_final.tir.json").read_text())
            arm = tvm.ir.load_json((directory / "node24_final.tir.json").read_text())
            if not tvm.ir.structural_equal(plain, arm, map_free_vars=True):
                raise ValueError("ARM/fusion FSim actual programs differ")
            arena, windows = 252 * 1024, len(certificate["native_queue"]["invocations"][0]["independent_windows"])
            if certificate["final_tir_sha256"] != digest(root / "local/prefetch_attempt03/node24_final.tir.json"):
                raise ValueError("certificate/program identity mismatch")
            submitted = certificate["native_queue"]["capacity"]["submission_count"] / 4
            instructions = certificate["native_queue"]["invocations"][0]["instructions"]
        else:
            if digest(binary) != prepared["programs"][name]["binary_sha256"]:
                raise ValueError("historical binary identity mismatch")
            arena, windows = resources[name]["static_arenas_bytes"]["local.wgt_buffer"], 0
            submitted = resources[name]["warm_command_signature"]["submissions"]
            instructions = resources[name]["warm_command_signature"]["instructions"]
        totals = record["request_totals"]
        signature = resources[name]["warm_command_signature"] if name != NEW else \
                    certificate["native_queue"]["capacity"]["submissions"][0]["semantic"]
        mac = signature["mac"] if name != NEW else signature["MAC_GEMM"]
        reset = signature["reset"] if name != NEW else signature["RESET_GEMM"]
        features[name] = {"local_qualified": True, "inp_bytes": totals["load_buffer_2d_inp_bytes"],
                          "wgt_bytes": totals["load_buffer_2d_wgt_bytes"],
                          "acc_bytes": totals["load_buffer_2d_acc_bytes"],
                          "out_bytes": totals["store_buffer_2d_out_bytes"],
                          "wgt_arena_bytes": arena, "independent_windows": windows,
                          "mac": mac, "reset": reset, "submissions": submitted,
                          "instructions": instructions, "normal_completion": True,
                          "latency_ms": None}
        entries[name] = {"directory": str(directory), "binary": str(binary),
                         "binary_sha256": digest(binary),
                         "tir_sha256": digest(directory / "node24_final.tir.json"),
                         "candidate_id": name, "legacy_builder_route_label": record["route"]["candidate_id"],
                         "historical_latency_exposed": name != NEW,
                         "new_FPGA_correctness_required": True}
    proposal = prefixes(features)
    if proposal["rule"] != [NEW] or proposal["bytes_only"] != ["R12-G"]:
        raise ValueError("actual static features do not establish frozen expected prefixes")
    sources = {str(Path(__file__).parent / name): digest(Path(__file__).parent / name)
               for name in ("c3_five_tradeoff_policy.py", "c3_five_tradeoff_prefetch.py",
                            "c3_five_tradeoff_queue.py", "c3_five_tradeoff_commands.py",
                            "freeze_c3_five_tradeoff.py")}
    write_new(root / "rule_freeze.json", {
        "status": "STATIC_RULE_AND_POOL_FROZEN_BEFORE_NEW_FPGA_LATENCY",
        "programs": entries, "features": features, "prefixes": proposal,
        "source_bindings": sources, "hardware_runtime": check_runtime(),
        "protocol_sha256": digest(root / "PROTOCOL.md"),
        "local_certificate_sha256": digest(root / "local_qualification_v2.json"),
        "historical_manifest_verified_entries": len(old_manifest["artifacts"]),
        "main_metric": "independent confirmation latency of frozen returned program",
        "same_exposure_calls_per_method": 6, "physical_selection_calls": 12,
        "no_numerical_five_factor_model_identified": True,
        "node34_role": "latency-exposed Input deployment validation, not new algorithm holdout",
        "new_FPGA_latency_observed": False,
    })
    print("STATIC_RULE_AND_POOL_FROZEN_BEFORE_NEW_FPGA_LATENCY")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    freeze(parser.parse_args().output_dir)
