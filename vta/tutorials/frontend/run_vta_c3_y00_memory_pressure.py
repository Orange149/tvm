#!/usr/bin/env python3
"""Prepare, freeze, run, and analyze the Y00 memory-pressure residency pair."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import c3_thread_sync_audit as audit
import c3_y00_memory_pressure_protocol as protocol
import run_vta_c3_reuse_supplement as s
import run_vta_c3_vthread_staged_residency as base


HERE, ROOT = s.HERE, s.ROOT
POOL = (
    HERE
    / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
    / "20260912_p7r127_y00_v2_local_pool_run01/candidates_v2.jsonl"
)
SOURCE_CANDIDATE = "423345af4098fd1b9f901f8b46e6f5a141d3cd187b6fce2b6951439c586a12a5"
SCHEMA = "c3_y00_memory_pressure_residency_v1"
SCOPE = (
    "YOLOv3-tiny Y00 (16->32, 208x208, 3x3), frozen h4/w26 Tile: "
    "same-compute O2 versus first-use two-vthread full WGT residency"
)


_base_bindings = base.bindings


def bindings():
    result = _base_bindings()
    for name in (
        Path(__file__),
        HERE / "c3_y00_memory_pressure_protocol.py",
        HERE / "c3_y00_memory_pressure_board.py",
        HERE / "c3_y00_memory_pressure_results.py",
    ):
        result[str(name.resolve())] = s.file_hash(name)
    return result


def _source_row():
    for line in POOL.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["candidate_id"] == SOURCE_CANDIDATE:
            return row
    raise ValueError("frozen Y00 source candidate is missing")


def make_candidates(fingerprint, schedule_version):
    source = _source_row()
    config = copy.deepcopy(source["identity"]["complete_config_entity"])
    for item in config["entity"]:
        if item[0] == "oc_nthread":
            item[2] = 2
    visible = dict(source["knobs"], oc_nthread=2)
    expected = {
        "tile_b": 1,
        "tile_h": 4,
        "tile_w": 26,
        "tile_ci": 1,
        "tile_co": 1,
        "oc_nthread": 2,
        "h_nthread": 1,
    }
    if visible != expected:
        raise ValueError("frozen Y00F05 Tile identity changed")
    shared = {
        "tile": "Y00F05",
        "family_id": "Y00-F05-O2-staged-full-residency",
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
        ("O2", 0, "original_o2", True, "stock"),
        ("PS", 8, "weight_vthread_resident_first_use", False, "staged_weight_residency"),
    ):
        row = {
            **shared,
            "candidate_id": "Y00F05_" + variant,
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
    original, resident = by_id["Y00F05_O2"], by_id["Y00F05_PS"]
    if original["status"] != "passed" or resident["status"] != "passed":
        return {"eligible_program_comparison": False, "reason": "candidate unqualified"}
    original_local = original["local_qualification"]
    resident_local = resident["local_qualification"]
    original_totals = original_local["semantics"]["totals"]
    resident_totals = resident_local["semantics"]["totals"]
    structural = audit.control_match(
        addresses["Y00F05_O2"], addresses["Y00F05_PS"], None, None, "residency"
    )
    weight_groups = [
        row
        for row in resident_local["semantics"]["request_groups"]
        if row["direction"] == "load" and row["memory_name"] == "wgt"
    ]
    checks = {
        "same_complete_config": original["complete_config_entity"]
        == resident["complete_config_entity"],
        "same_two_vthread_knobs": original["visible_features"]["oc_nthread"]
        == resident["visible_features"]["oc_nthread"]
        == 2,
        "same_compute_context": structural["checks"]["compute_context"],
        "same_nonweight_requests": structural["checks"]["nonweight_data"],
        "same_sync_count": structural["checks"]["sync_count"],
        "same_storage_arenas": original_local["storage_arenas"]
        ["scope_arena_span_sum_upper_bound_bytes"]
        == resident_local["storage_arenas"]["scope_arena_span_sum_upper_bound_bytes"],
        "original_reloads_weight": original_totals["load_buffer_2d_wgt_bytes"] == 1916928
        and original_totals["load_buffer_2d_wgt_calls"] == 832,
        "resident_loads_weight_once": resident_totals["load_buffer_2d_wgt_bytes"] == 4608
        and resident_totals["load_buffer_2d_wgt_calls"] == 2,
        "resident_has_two_thread_groups": len(weight_groups) == 2
        and all(row["bytes_per_request"] == 2304 for row in weight_groups),
    }
    return {
        "eligible_program_comparison": all(checks.values()),
        "checks": checks,
        "weight_request_reduction_pct": 100
        * (1 - resident_totals["load_buffer_2d_wgt_bytes"] / original_totals["load_buffer_2d_wgt_bytes"]),
        "declared_mechanism_difference": (
            "same Tile, two vthreads, compute, non-WGT requests, arenas, and synchronization; "
            "only repeated WGT requests and derived instruction organization differ"
        ),
        "interpretation": "program-level residency effect, not isolated DMA wait cycles",
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
            from c3_y00_memory_pressure_board import run

            run(args, out)
        else:
            from c3_y00_memory_pressure_results import analyze

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
