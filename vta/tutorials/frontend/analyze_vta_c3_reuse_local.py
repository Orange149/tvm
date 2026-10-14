#!/usr/bin/env python3
"""Independently verify and render Node-B local evidence (never FPGA latency)."""
import argparse
from collections import Counter
import csv
from html import escape
import json
from pathlib import Path
import time

from run_vta_c3_reuse_supplement import verify, read, write, seal, file_hash, digest, MODES, SEEDS


def check_candidate(row):
    static, fsim, cross = row["local_qualification"], row["fsim_qualification"], row["cross_qualification"]
    if any(value["status"] != "passed" for value in (static, fsim, cross)):
        raise ValueError("unqualified local identity")
    if fsim["relowered_tir_sha256"] != static["tir_sha256"] or cross["relowered_tir_sha256"] != static["tir_sha256"]:
        raise ValueError("lowered identity mismatch")
    if [seed["seed"] for seed in fsim["seeds"]] != list(SEEDS):
        raise ValueError("incomplete seed protocol")
    for seed in fsim["seeds"]:
        if not seed["correct"] or seed["mismatch_count"] or seed["actual_sha256"] != seed["expected_sha256"] or seed["complete_function_invocations"] != 1:
            raise ValueError("seed correctness or normalization mismatch")
        for key, value in static["semantics"]["totals"].items():
            if seed["runtime_logical_dma"].get(key) != value:
                raise ValueError("independent static/runtime mismatch")
        for sim_key, key in {"inp_load_nbytes": "load_buffer_2d_inp_bytes", "wgt_load_nbytes": "load_buffer_2d_wgt_bytes",
                             "out_store_nbytes": "store_buffer_2d_out_bytes"}.items():
            if seed["fsim_instruction_bytes"][sim_key] != seed["runtime_logical_dma"][key]:
                raise ValueError("independent FSim instruction/runtime mismatch")
    command = fsim["command_features"]
    if len(set(command["per_seed_sha256"].values())) != 1 or not command["values"]["drain_check"]["matches_p3e"]:
        raise ValueError("inconsistent command/drain protocol")
    if not static["storage_arenas"]["allocations"] or not all(request["ddr_bounds_checked"] for request in static["semantics"]["request_groups"]):
        raise ValueError("missing storage or source-bounds evidence")
    return True


def table_row(row):
    check_candidate(row)
    static, command = row["local_qualification"], row["fsim_qualification"]["command_features"]["values"]
    result = {"workload_id": row["workload_id"], "family_id": row["family_id"], "mode": row["residence_mode"],
              "control": row.get("experimental_reload_control", False), "candidate_id": row["candidate_id"],
              "tile_h": row["visible_features"]["tile_h"], "tile_w": row["visible_features"]["tile_w"],
              "tile_ci": row["visible_features"]["tile_ci"], "tile_co": row["visible_features"]["tile_co"],
              "submissions": command["submissions"], "residency_drains": static["explicit_residency_drains"],
              "instruction_peak_bytes": command["peaks"]["insn_bytes"], "uop_peak_bytes": command["peaks"]["uop_bytes"],
              "tir_sha256": static["tir_sha256"], "binary_sha256": row["cross_qualification"]["binary_sha256"],
              "fpga_latency_ms": None, "simultaneous_sram_peak": None}
    for operand in ("inp", "wgt", "acc", "uop", "out"):
        direction = "store" if operand == "out" else "load"
        for metric in ("bytes", "calls"):
            # Zero absent LOAD types are verified by the retained runtime profile.
            key = direction + "_buffer_2d_" + operand + "_" + metric
            result[operand + "_" + metric] = static["semantics"]["totals"].get(key, 0)
        result[operand + "_unique_source_bytes"] = static["semantics"]["unique_source_region_bytes"].get(direction + "_" + operand)
        result[operand + "_reload_ratio"] = static["semantics"]["reload_ratios"].get(direction + "_" + operand)
    for operand in ("inp", "wgt", "acc"):
        result[operand + "_arena_bytes"] = static["storage_arenas"]["scope_arena_span_sum_upper_bound_bytes"].get("local." + operand + "_buffer")
    return result


def scope_svg(rows):
    width, height = 1100, 390
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="white"/>',
              '<style>text{font-family:monospace;font-size:12px} .title{font-size:15px;font-weight:bold}</style>',
              '<text x="20" y="25" class="title">Exact lowered LOAD scopes: local evidence, FPGA timing pending</text>']
    first_family = rows[0]["family_id"]
    selected = {row["residence_mode"]: row for row in rows if row["family_id"] == first_family}
    for index, mode in enumerate(MODES):
        x = 20 + 270 * index
        row = selected[mode]
        static = row["local_qualification"]
        pieces += [f'<rect x="{x}" y="50" width="260" height="290" fill="#f5f7fa" stroke="#9aa7b0"/>',
                   f'<text x="{x+8}" y="72" class="title">{escape(mode.replace("_resident_barrier", ""))}</text>',
                   f'<text x="{x+8}" y="92">{escape(first_family)}</text>']
        for operand, offset in (("inp", 0), ("wgt", 105)):
            sites = [site for site in static["semantics"]["load_store_sites"] if site["extern"] == "VTALoadBuffer2D"
                     and site["symbolic_arguments"][-1] == ("2" if operand == "inp" else "1")]
            pieces.append(f'<text x="{x+8}" y="{115+offset}">{operand.upper()} producer scope:</text>')
            if sites:
                for depth, loop in enumerate(sites[0]["loop_scope"][:4]):
                    label = f'{loop["var"]} x {loop["extent"]}'
                    pieces.append(f'<text x="{x+12+depth*8}" y="{133+offset+depth*17}">{escape(label)}</text>')
                if not sites[0]["loop_scope"]:
                    pieces.append(f'<text x="{x+12}" y="{133+offset}">function / batch scope</text>')
        totals = static["semantics"]["totals"]
        pieces += [f'<text x="{x+8}" y="315">I/W B: {totals["load_buffer_2d_inp_bytes"]}/{totals["load_buffer_2d_wgt_bytes"]}</text>',
                   f'<text x="{x+8}" y="331">residency drains: {static["explicit_residency_drains"]}</text>']
    pieces += ['<text x="20" y="365">Scope describes LOAD placement; this is not a complete consumer/overwrite proof.</text>', '</svg>']
    return "\n".join(pieces)


def run(local_directory, output):
    started = time.monotonic()
    local_directory, output = Path(local_directory).resolve(), Path(output).resolve()
    contract = verify(local_directory)
    if contract["phase"] != "local":
        raise ValueError("requires local evidence")
    summary = read(local_directory / "summary.json")
    candidates, controls = read(local_directory / "candidates.json"), read(local_directory / "controls.json")
    tables = [table_row(row) for row in candidates + controls]
    for row in candidates + controls:
        if file_hash(local_directory / "binaries" / (row["candidate_id"] + ".so")) != row["cross_qualification"]["binary_sha256"]:
            raise ValueError("ARM artifact mismatch")
    classification = []
    for family_id in sorted({row["family_id"] for row in candidates}):
        rows = [row for row in tables if row["family_id"] == family_id and not row["control"]]
        if len(rows) != 4 or {row["mode"] for row in rows} != set(MODES):
            raise ValueError("not an exact four-mode family")
        original = next(row for row in rows if row["mode"] == "original")
        combined = next(row for row in rows if row["mode"] == "input_weight_resident_barrier")
        classification.append({"family_id": family_id, "workload_id": original["workload_id"],
                               "four_local_correct": True, "four_fpga_measured": False,
                               "distinct_tir_programs": len({row["tir_sha256"] for row in rows}),
                               "distinct_arm_binaries": len({row["binary_sha256"] for row in rows}),
                               "input_delta_pct": 100 * (combined["inp_bytes"] / original["inp_bytes"] - 1),
                               "weight_delta_pct": 100 * (combined["wgt_bytes"] / original["wgt_bytes"] - 1),
                               "effective_dual_reuse": combined["inp_bytes"] < original["inp_bytes"] and combined["wgt_bytes"] < original["wgt_bytes"]})
    output.mkdir(parents=True, exist_ok=False)
    write(output / "contract.json", {"phase": "local_analysis", "input_directory": str(local_directory),
                                     "input_manifest_sha256": file_hash(local_directory / "artifact_hashes.json"),
                                     "source_sha256": file_hash(Path(__file__)), "board_contacted": False})
    with (output / "four_mode_structure.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tables[0]))
        writer.writeheader()
        writer.writerows({key: "unknown" if value is None else value for key, value in row.items()} for row in tables)
    write(output / "family_classification.json", classification)
    if candidates:
        (output / "load_scopes.svg").write_text(scope_svg(candidates))
    report = {"status": summary["status"], "selected_main_identities": len(candidates), "accepted_controls": len(controls),
              "selected_four_local_correct_families": len(classification),
              "effective_dual_families": sum(row["effective_dual_reuse"] for row in classification),
              "four_distinct_arm_binary_families": sum(row["distinct_arm_binaries"] == 4 for row in classification),
              "selected_geometries": dict(Counter(row["workload_id"] for row in classification)),
              "fpga_measured_families": 0, "all_candidate_counters_independently_verified": True,
              "board_contacted": False, "input_costs": summary, "analysis_wall_seconds": time.monotonic() - started}
    write(output / "summary.json", report)
    (output / "results.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in classification) + "\n")
    (output / "timeline.jsonl").write_text(json.dumps({"event": "analysis_complete", "elapsed_seconds": time.monotonic() - started}) + "\n")
    verify(local_directory)
    seal(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    run(args.local_dir, args.output_dir)
