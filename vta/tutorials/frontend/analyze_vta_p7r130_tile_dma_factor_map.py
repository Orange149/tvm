#!/usr/bin/env python3
"""Build a cross-geometry tile/DMA/qualification map from immutable C3 artifacts.

This is a descriptive, no-latency-leak analysis.  Same-family residence-mode
pairs isolate schedule lifetime changes; one-knob pairs expose which tile knob
changed together with DMA/SRAM/command metrics.  Neither table is promoted to
a universal causal law.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P7 = ROOT / "07_grouped_holdout"
DEFAULT_LOCAL = {
    "Y00": P7 / "20260912_p7r127_y00_v2_local_pool_run01",
    "Y01": P7 / "20260912_p7r124_y01_v2_local_pool_run01",
    "Y02": P7 / "20260911_p7r119_y02_weight_resident_barrier_pilot_v2_run01",
}
DEFAULT_BOARD = (
    P7 / "20260912_p7r122_y02_health_gated_correctness_run01",
    P7 / "20260912_p7r125_y01_complete_board_pool_run01",
    P7 / "20260912_p7r126_y01_remaining_correctness_run01",
)
DEFAULT_TIMING = P7 / "20260912_p7r123_y02_b00_paired_timing_run01"
DEFAULT_OUTPUT = P7 / "20260912_p7r130_tile_dma_factor_map_run02"

KNOBS = ("tile_b", "tile_h", "tile_w", "tile_ci", "tile_co", "oc_nthread", "h_nthread")
DMA_FIELDS = (
    "input_dma_bytes", "weight_dma_bytes", "output_dma_bytes", "load_dma_bytes",
    "input_dma_calls", "weight_dma_calls", "output_dma_calls", "load_dma_calls",
    "small_dma_calls", "strided_dma_calls", "padded_dma_calls", "input_reload",
    "weight_reload", "output_reload",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_run(directory: Path) -> str:
    ledger_path = directory / "artifact_hashes.json"
    ledger = read_json(ledger_path)
    for name, expected in ledger["artifacts"].items():
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"artifact hash mismatch: {path}")
    return sha256(ledger_path)


def aligned_4k(value):
    if value is None:
        return None
    return int(math.ceil(float(value) / 4096.0) * 4096)


def fsim_command(row):
    try:
        values = row["command_evidence"]["values"]
        peaks = values["peaks"]
        return (
            int(peaks["insn_bytes"]),
            int(peaks["uop_bytes"]),
            int(values["submissions"]),
        )
    except (KeyError, TypeError):
        return None, None, None


def canonical_local_rows(workload_id, directory):
    static_rows = read_jsonl(directory / "static_results.jsonl")
    fsim_rows = {row["candidate_id"]: row for row in read_jsonl(directory / "fsim_results.jsonl")}
    rows = []
    for source in static_rows:
        fsim = fsim_rows.get(source["candidate_id"])
        feature = dict(source.get("static_feature_vector") or {})
        dma = source.get("dma_summary") or {}
        for target, origin in {
            "input_dma_bytes": "input_bytes", "weight_dma_bytes": "weight_bytes",
            "output_dma_bytes": "output_bytes", "load_dma_bytes": "load_bytes",
            "input_dma_calls": "input_calls", "weight_dma_calls": "weight_calls",
            "output_dma_calls": "output_calls", "load_dma_calls": "load_calls",
        }.items():
            if target not in feature and origin in dma:
                feature[target] = dma[origin]
        transfer = source.get("transfer_signature") or {}
        reloads = transfer.get("reload_ratio") or {}
        totals = transfer.get("totals") or {}
        for target, origin in {
            "input_reload": "input_load", "weight_reload": "weight_load",
            "output_reload": "output_store",
        }.items():
            if target not in feature and origin in reloads:
                feature[target] = reloads[origin]
        for target, origin in {
            "small_dma_calls": "load_buffer_2d_small_calls",
            "strided_dma_calls": "load_buffer_2d_strided_calls",
        }.items():
            if target not in feature and origin in totals:
                feature[target] = totals[origin]
        if "padded_dma_calls" not in feature:
            padded = transfer.get("padded_calls") or {}
            if padded:
                feature["padded_dma_calls"] = sum(padded.values())
        sram = source.get("sram_features") or {}
        required = sram.get("required_vectors") or {}
        insn, uop, submissions = fsim_command(fsim or {})
        knobs = source.get("knobs") or {}
        if not knobs:
            raise ValueError(f"{workload_id}: static row lacks frozen knobs")
        row = {
            "workload_id": workload_id,
            "family_id": source["family_id"],
            "candidate_id": source["candidate_id"],
            "mode": source["public_mode"],
            **{name: int(knobs[name]) for name in KNOBS},
            "static_status": source["status"],
            "static_failure": (source.get("failure") or {}).get("subcategory"),
            "fsim_status": "not_run" if fsim is None else fsim["status"],
            "fpga_status": "not_observed",
            "latency_ms": None,
            **{name: feature.get(name) for name in DMA_FIELDS},
            "input_sram_vectors": required.get("input_vectors"),
            "weight_sram_vectors": required.get("weight_vectors"),
            "accumulator_sram_vectors": required.get("accumulator_vectors"),
            "residency_drains": (source.get("sync") or {}).get("residency_drains"),
            "insn_peak_bytes": insn,
            "uop_peak_bytes": uop,
            "submissions": submissions,
            "command_backing_bytes": (
                None if insn is None or uop is None else aligned_4k(insn) + aligned_4k(uop)
            ),
        }
        rows.append(row)
    return rows


def board_labels(board_dirs):
    result = {}
    for directory in board_dirs:
        name = "candidate_correctness.jsonl" if (directory / "candidate_correctness.jsonl").is_file() \
            else "correctness.jsonl"
        for row in read_jsonl(directory / name):
            status = row.get("status")
            if status not in ("passed", "failed"):
                continue
            previous = result.get(row["candidate_id"])
            if previous is not None and previous != status:
                raise ValueError("conflicting FPGA labels for candidate " + row["candidate_id"])
            result[row["candidate_id"]] = status
    return result


def timing_labels(directory):
    summary = read_json(directory / "summary.json")
    if not summary.get("all_timed_calls_correct"):
        raise ValueError("timing input is not correctness-qualified")
    return {
        candidate_id: float(stats["median_ms"])
        for candidate_id, stats in summary["analysis"]["candidate_stats"].items()
    }


def fractional_delta(value, control):
    if value is None or control in (None, 0):
        return None
    return float(value) / float(control) - 1.0


def residence_pairs(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[(row["workload_id"], row["family_id"])][row["mode"]] = row
    result = []
    for (workload, family), modes in sorted(groups.items()):
        control = modes.get("original")
        if control is None:
            continue
        for mode, row in sorted(modes.items()):
            if mode == "original":
                continue
            item = {
                "workload_id": workload,
                "family_id": family,
                "mode": mode,
                "static_status": row["static_status"],
                "fsim_status": row["fsim_status"],
                "fpga_status": row["fpga_status"],
                "latency_delta_fraction": fractional_delta(row["latency_ms"], control["latency_ms"]),
            }
            for field in DMA_FIELDS + (
                "input_sram_vectors", "weight_sram_vectors", "accumulator_sram_vectors",
                "residency_drains", "insn_peak_bytes", "uop_peak_bytes", "command_backing_bytes",
            ):
                item[field + "_delta_fraction"] = fractional_delta(row.get(field), control.get(field))
                item[field + "_delta"] = (
                    None if row.get(field) is None or control.get(field) is None
                    else row[field] - control[field]
                )
            result.append(item)
    return result


def one_knob_pairs(rows):
    result = []
    grouped = defaultdict(list)
    for row in rows:
        if row["static_status"] == "ok":
            grouped[(row["workload_id"], row["mode"])].append(row)
    for (workload, mode), candidates in sorted(grouped.items()):
        for index, left in enumerate(candidates):
            for right in candidates[index + 1:]:
                changed = [name for name in KNOBS if left[name] != right[name]]
                if len(changed) != 1:
                    continue
                knob = changed[0]
                low, high = sorted((left, right), key=lambda row: row[knob])
                item = {
                    "workload_id": workload,
                    "mode": mode,
                    "knob": knob,
                    "low_value": low[knob],
                    "high_value": high[knob],
                    "low_family_id": low["family_id"],
                    "high_family_id": high["family_id"],
                }
                for field in DMA_FIELDS + (
                    "input_sram_vectors", "weight_sram_vectors", "accumulator_sram_vectors",
                    "residency_drains", "command_backing_bytes",
                ):
                    item[field + "_delta_fraction"] = fractional_delta(high.get(field), low.get(field))
                result.append(item)
    return result


def median(values):
    values = [value for value in values if value is not None]
    return None if not values else statistics.median(values)


def summarize(rows, mode_pairs, knob_pairs):
    by_workload = {}
    for workload in sorted({row["workload_id"] for row in rows}):
        subset = [row for row in rows if row["workload_id"] == workload]
        by_workload[workload] = {
            "identities": len(subset),
            "static_pass": sum(row["static_status"] == "ok" for row in subset),
            "fsim_pass": sum(row["fsim_status"] == "passed" for row in subset),
            "fpga_observed": sum(row["fpga_status"] != "not_observed" for row in subset),
            "fpga_pass": sum(row["fpga_status"] == "passed" for row in subset),
            "timed": sum(row["latency_ms"] is not None for row in subset),
            "static_failure_categories": dict(Counter(
                row["static_failure"] for row in subset if row["static_failure"]
            )),
        }
    effects = {}
    for workload, mode in sorted({(row["workload_id"], row["mode"]) for row in mode_pairs}):
        subset = [row for row in mode_pairs if row["workload_id"] == workload and row["mode"] == mode]
        effects[f"{workload}:{mode}"] = {
            "pair_count": len(subset),
            "load_bytes_delta_fraction_median": median(
                row["load_dma_bytes_delta_fraction"] for row in subset
            ),
            "load_calls_delta_fraction_median": median(
                row["load_dma_calls_delta_fraction"] for row in subset
            ),
            "command_backing_delta_fraction_median": median(
                row["command_backing_bytes_delta_fraction"] for row in subset
            ),
            "fpga_pass_pairs": sum(row["fpga_status"] == "passed" for row in subset),
            "timed_pairs": sum(row["latency_delta_fraction"] is not None for row in subset),
        }
    knob_effects = {}
    for workload, mode, knob in sorted({
        (row["workload_id"], row["mode"], row["knob"]) for row in knob_pairs
    }):
        subset = [row for row in knob_pairs if row["workload_id"] == workload
                  and row["mode"] == mode and row["knob"] == knob]
        knob_effects[f"{workload}:{mode}:{knob}"] = {
            "pair_count": len(subset),
            "load_bytes_delta_fraction_median": median(
                row["load_dma_bytes_delta_fraction"] for row in subset
            ),
            "load_calls_delta_fraction_median": median(
                row["load_dma_calls_delta_fraction"] for row in subset
            ),
            "input_sram_delta_fraction_median": median(
                row["input_sram_vectors_delta_fraction"] for row in subset
            ),
            "accumulator_sram_delta_fraction_median": median(
                row["accumulator_sram_vectors_delta_fraction"] for row in subset
            ),
        }
    return {
        "schema": "c3_p7r130_tile_dma_factor_map_v1",
        "scope": "descriptive cross-geometry map; no universal causal or search-speed claim",
        "identities": len(rows),
        "same_tile_residence_pairs": len(mode_pairs),
        "one_knob_pairs": len(knob_pairs),
        "by_workload": by_workload,
        "same_tile_effects": effects,
        "one_knob_effects": knob_effects,
        "board_execution_state": (
            "W05 seed0 health canary timed out on current boot; no new candidate dispatched"
        ),
        "frozen_search_rule_v3": {
            "hard_gates": [
                "static tensorize/SRAM/padding/compact/UOP checks",
                "three-seed FSim",
                "FPGA correctness fail-fast; pass requires three seeds",
            ],
            "priority": (
                "within each geometry, keep the nondominated set over total LOAD bytes, "
                "LOAD calls, padded calls and synchronization drains; order by normalized "
                "LOAD bytes then calls; preserve one candidate per residence mode before repeats"
            ),
            "performance_labels_used_to_define_rule": False,
            "tophub_used_by_selector": False,
            "evaluation": "gross-budget trials/regret to frozen full-pool oracle",
        },
    }


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_results(path, summary):
    lines = [
        "# P7R130 tile—DMA—qualification factor map", "",
        "> Descriptive analysis of immutable local/board artifacts; no new FPGA candidate was dispatched.", "",
        "## Coverage", "",
        "| workload | identities | static pass | FSim pass | FPGA observed/pass | timed |", "|---|---:|---:|---:|---:|---:|",
    ]
    for workload, item in summary["by_workload"].items():
        lines.append(
            f"| {workload} | {item['identities']} | {item['static_pass']} | {item['fsim_pass']} | "
            f"{item['fpga_observed']}/{item['fpga_pass']} | {item['timed']} |"
        )
    lines.extend([
        "", "## Same-tile residence effects", "",
        "| workload:mode | pairs | median LOAD bytes delta | median LOAD-call delta | FPGA-pass pairs | timed pairs |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, item in summary["same_tile_effects"].items():
        byte_delta = item["load_bytes_delta_fraction_median"]
        call_delta = item["load_calls_delta_fraction_median"]
        byte_text = "--" if byte_delta is None else f"{byte_delta * 100:.2f}%"
        call_text = "--" if call_delta is None else f"{call_delta * 100:.2f}%"
        lines.append(
            f"| {name} | {item['pair_count']} | {byte_text} | {call_text} | "
            f"{item['fpga_pass_pairs']} | {item['timed_pairs']} |"
        )
    lines.extend([
        "", "## What is now frozen for the search experiment", "",
        "- Hard legality/correctness gates are separated from performance ranking.",
        "- Ranking uses a simple label-free shared-memory Pareto rule; no complex model is retained without evidence.",
        "- Same-tile mode diversity is preserved before repeatedly sampling one mode.",
        "- TopHub is not exposed to the selector; the primary endpoint is trials/regret to a completed pool oracle.",
        "", "## Claim boundary", "",
        "The CSV files expose exact per-identity and paired deltas. One-knob pairs are controlled associations inside the sampled pool, not universal causal laws. Current-board W05 health failed before candidate dispatch, so this run adds no latency or FPGA-performance claim.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    local_dirs = {workload: Path(getattr(args, workload.lower())) for workload in DEFAULT_LOCAL}
    board_dirs = [Path(value) for value in args.board_dirs]
    timing_dir = Path(args.timing_dir)
    bindings = {
        "local": {workload: {"path": str(path), "ledger_sha256": verify_run(path)}
                  for workload, path in local_dirs.items()},
        "board": [{"path": str(path), "ledger_sha256": verify_run(path)} for path in board_dirs],
        "timing": {"path": str(timing_dir), "ledger_sha256": verify_run(timing_dir)},
    }
    rows = []
    for workload, directory in local_dirs.items():
        rows.extend(canonical_local_rows(workload, directory))
    labels = board_labels(board_dirs)
    timings = timing_labels(timing_dir)
    known_ids = {row["candidate_id"] for row in rows}
    if not set(labels).issubset(known_ids) or not set(timings).issubset(known_ids):
        raise ValueError("board/timing label does not bind to the local v2 identity set")
    for row in rows:
        row["fpga_status"] = labels.get(row["candidate_id"], "not_observed")
        row["latency_ms"] = timings.get(row["candidate_id"])
    mode_pairs = residence_pairs(rows)
    knob_pairs = one_knob_pairs(rows)
    summary = summarize(rows, mode_pairs, knob_pairs)
    output.mkdir(parents=True)
    (output / "input_bindings.json").write_text(json.dumps(bindings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output / "identity_factor_map.csv", rows)
    write_csv(output / "same_tile_residence_pairs.csv", mode_pairs)
    write_csv(output / "one_knob_pairs.csv", knob_pairs)
    write_results(output / "RESULTS.md", summary)
    artifacts = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.name != "artifact_hashes.json"}
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": artifacts, "source_sha256": sha256(Path(__file__))}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--y00", type=Path, default=DEFAULT_LOCAL["Y00"])
    parser.add_argument("--y01", type=Path, default=DEFAULT_LOCAL["Y01"])
    parser.add_argument("--y02", type=Path, default=DEFAULT_LOCAL["Y02"])
    parser.add_argument("--board-dirs", type=Path, nargs="+", default=list(DEFAULT_BOARD))
    parser.add_argument("--timing-dir", type=Path, default=DEFAULT_TIMING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
