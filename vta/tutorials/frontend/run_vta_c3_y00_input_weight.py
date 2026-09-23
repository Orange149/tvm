#!/usr/bin/env python3
"""Prepare, freeze, run, and analyze Y00F06 input plus staged-weight reuse."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import c3_thread_sync_audit as audit
import c3_y00_input_weight_protocol as protocol
import run_vta_c3_reuse_supplement as s
import run_vta_c3_vthread_staged_residency as base


HERE, ROOT = s.HERE, s.ROOT
POOL = (
    HERE
    / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
    / "20260912_p7r127_y00_v2_local_pool_run01/candidates_v2.jsonl"
)
SOURCE_CANDIDATE = "7b958e8562ed5e04fd14a52e940aa7aa5518ba600d8284c41d5ecd0d9bf18e3c"
SCHEMA = "c3_y00_input_weight_staged_v1"
SCOPE = (
    "YOLOv3-tiny Y00 conv2, frozen Y00F06 h16/w52 Tile: original, "
    "input-only, and input plus no-drain first-use full-weight residency"
)
_base_bindings = base.bindings


def bindings():
    result = _base_bindings()
    for name in (
        Path(__file__),
        ROOT / "vta/python/vta/top/vta_conv2d.py",
        HERE / "c3_y00_input_weight_protocol.py",
        HERE / "c3_y00_input_weight_board.py",
        HERE / "c3_y00_input_weight_results.py",
    ):
        result[str(name.resolve())] = s.file_hash(name)
    return result


def _source_row():
    rows = [json.loads(line) for line in POOL.read_text().splitlines() if line.strip()]
    return next(row for row in rows if row["candidate_id"] == SOURCE_CANDIDATE)


def make_candidates(fingerprint, schedule_version):
    source = _source_row()
    config = copy.deepcopy(source["identity"]["complete_config_entity"])
    visible = dict(source["knobs"])
    expected = {
        "tile_b": 1,
        "tile_h": 16,
        "tile_w": 52,
        "tile_ci": 1,
        "tile_co": 1,
        "oc_nthread": 1,
        "h_nthread": 1,
    }
    if visible != expected:
        raise ValueError("frozen Y00F06 Tile identity changed")
    shared = {
        "tile": "Y00F06",
        "family_id": "Y00-F06-input-plus-staged-weight",
        "workload_id": "Y00",
        "complete_config_entity": config,
        "visible_features": visible,
        "hardware_fingerprint": fingerprint,
        "applicability": {"full_weight_fits_sram": True, "full_weight_bytes": 4608},
        "schedule_version": schedule_version,
        "source_candidate_id": SOURCE_CANDIDATE,
    }
    result = []
    for variant, mode, name, source_tophub, build_variant in (
        ("O", 0, "original", True, "stock"),
        ("I", 1, "input_stationary", False, "stock"),
        ("IW", 9, "input_weight_resident_staged", False, "staged_weight_residency"),
    ):
        row = {
            **shared,
            "candidate_id": "Y00F06_" + variant,
            "variant": variant,
            "implementation_mode": mode,
            "residence_mode": name,
            "source_tophub": source_tophub,
            "build_config_variant": build_variant,
            "identity": {
                "workload": source["identity"]["workload"],
                "complete_config_entity": config,
            },
        }
        row["identity_sha256"] = s.digest(
            {key: value for key, value in row.items() if key != "candidate_id"}
        )
        result.append(row)
    return result


def pair_audit(rows, addresses):
    by_id = {row["candidate_id"]: row for row in rows}
    input_only, combined = by_id["Y00F06_I"], by_id["Y00F06_IW"]
    if input_only["status"] != "passed" or combined["status"] != "passed":
        return {"eligible_program_comparison": False, "reason": "candidate unqualified"}
    left = input_only["local_qualification"]
    right = combined["local_qualification"]
    left_totals, right_totals = left["semantics"]["totals"], right["semantics"]["totals"]
    structural = audit.control_match(
        addresses["Y00F06_I"], addresses["Y00F06_IW"], None, None, "residency"
    )
    checks = {
        "same_complete_config": input_only["complete_config_entity"]
        == combined["complete_config_entity"],
        "same_compute_context": structural["checks"]["compute_context"],
        "same_nonweight_requests": structural["checks"]["nonweight_data"],
        "same_sync_count": structural["checks"]["sync_count"],
        "same_storage_arenas": left["storage_arenas"]["scope_arena_span_sum_upper_bound_bytes"]
        == right["storage_arenas"]["scope_arena_span_sum_upper_bound_bytes"],
        "input_only_reloads_weight": left_totals["load_buffer_2d_wgt_bytes"] == 239616
        and left_totals["load_buffer_2d_wgt_calls"] == 52,
        "combined_loads_weight_once": right_totals["load_buffer_2d_wgt_bytes"] == 4608
        and right_totals["load_buffer_2d_wgt_calls"] == 1,
    }
    return {
        "eligible_program_comparison": all(checks.values()),
        "checks": checks,
        "weight_request_reduction_pct": 100
        * (1 - right_totals["load_buffer_2d_wgt_bytes"] / left_totals["load_buffer_2d_wgt_bytes"]),
        "declared_mechanism_difference": (
            "same input-prioritized Tile, compute, non-WGT requests, arenas, and sync; "
            "combined mode guards the full WGT load to the first spatial tile"
        ),
        "interpretation": "incremental program-level weight residency on input reuse",
    }


def configure_base():
    base.protocol = protocol
    base.SCHEMA = SCHEMA
    base.SCOPE = SCOPE
    base.bindings = bindings
    base.make_candidates = make_candidates
    base.pair_audit = pair_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "freeze", "board", "analyze"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract")
    parser.add_argument("--board-dir", type=Path)
    parser.add_argument("--execute-board", action="store_true")
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    parser.add_argument("--sdk-environment", default=base.SDK)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    configure_base()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        if args.phase == "prepare":
            base.prepare(args, out)
        elif args.phase == "freeze":
            base.freeze(args, out)
        elif args.phase == "board":
            from c3_y00_input_weight_board import run

            run(args, out)
        else:
            from c3_y00_input_weight_results import analyze

            analyze(args, out)
    except BaseException as error:
        s.write(
            out / "invalid_session.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "all_timing_invalid": True,
                "wall_seconds": time.monotonic() - started,
            },
        )
        raise
    finally:
        s.write(out / "phase_cost.json", {"wall_seconds": time.monotonic() - started})
        s.seal(out)


if __name__ == "__main__":
    main()
