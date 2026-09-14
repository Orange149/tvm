#!/usr/bin/env python3
"""Analyze Cheng-style four-scheme same-tile choices after the complete R18 pool."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import run_vta_c3_literature_baselines as baseline


MODES = (
    "original",
    "input_stationary",
    "weight_resident_barrier",
    "input_weight_resident_barrier",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def verify(directory):
    directory = Path(directory).resolve()
    artifacts = read_json(directory / "artifact_hashes.json")["artifacts"]
    for name, expected in artifacts.items():
        if isinstance(expected, str) and baseline.sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: {}".format(directory / name))
    return baseline.sha256_file(directory / "artifact_hashes.json")


def applicable(mode, original):
    limits = original.get("applicability", {})
    acc = bool(limits.get("accumulator_fits_sram", True))
    weight = bool(limits.get("full_weight_fits_sram", True))
    if mode == "original":
        return True
    if mode == "input_stationary":
        return acc
    if mode == "weight_resident_barrier":
        return weight
    if mode == "input_weight_resident_barrier":
        return acc and weight
    raise ValueError(mode)


def scheme_row(mode, candidate, observed, original, original_observed):
    if candidate is not None:
        row = observed[candidate["candidate_id"]]
        return {
            "mode": mode,
            "status": "measured" if row["fpga_correct"] else "fpga_invalid",
            "candidate_id": candidate["candidate_id"],
            "fallback_to": None,
            **{
                key: row.get(key)
                for key in (
                    "input_dma_bytes",
                    "weight_dma_bytes",
                    "acc_dma_bytes",
                    "output_dma_bytes",
                    "uop_dma_bytes",
                    "total_dma_bytes",
                    "total_dma_calls",
                    "submissions",
                    "explicit_residency_drains",
                    "instruction_peak_bytes",
                    "uop_peak_bytes",
                    "pure_instruction_ms",
                    "operator_latency_ms",
                )
            },
        }
    if not applicable(mode, original):
        return {
            "mode": mode,
            "status": "not_applicable_fallback_original",
            "candidate_id": original["candidate_id"],
            "fallback_to": "original",
            **{
                key: original_observed.get(key)
                for key in (
                    "input_dma_bytes",
                    "weight_dma_bytes",
                    "acc_dma_bytes",
                    "output_dma_bytes",
                    "uop_dma_bytes",
                    "total_dma_bytes",
                    "total_dma_calls",
                    "submissions",
                    "explicit_residency_drains",
                    "instruction_peak_bytes",
                    "uop_peak_bytes",
                    "pure_instruction_ms",
                    "operator_latency_ms",
                )
            },
        }
    return {
        "mode": mode,
        "status": "unavailable_after_lowering_fsim_pool_qualification",
        "candidate_id": None,
        "fallback_to": None,
        "operator_latency_ms": None,
        "pure_instruction_ms": None,
        "total_dma_bytes": None,
    }


def analyze_family(rows, observed):
    by_mode = {row["residence_mode"]: row for row in rows}
    original = by_mode.get("original")
    if original is None:
        return None
    original_observed = observed[original["candidate_id"]]
    schemes = [
        scheme_row(mode, by_mode.get(mode), observed, original, original_observed)
        for mode in MODES
    ]
    selectable = [
        row
        for row in schemes
        if row.get("operator_latency_ms") is not None
        and row["status"] in ("measured", "not_applicable_fallback_original")
    ]
    minimum = min(
        selectable,
        key=lambda row: (row["total_dma_bytes"], MODES.index(row["mode"])),
    ) if selectable else None
    fastest = min(
        selectable,
        key=lambda row: (row["operator_latency_ms"], MODES.index(row["mode"])),
    ) if selectable else None
    return {
        "workload_id": original["workload_id"],
        "family_id": original["family_id"],
        "config_entity": original["complete_config_entity"],
        "schemes": schemes,
        "all_four_semantically_resolved": len(selectable) == 4,
        "minimum_access_mode": minimum["mode"] if minimum else None,
        "minimum_access_candidate_id": minimum["candidate_id"] if minimum else None,
        "fastest_mode": fastest["mode"] if fastest else None,
        "fastest_candidate_id": fastest["candidate_id"] if fastest else None,
        "minimum_access_is_fastest": (
            minimum is not None
            and fastest is not None
            and minimum["candidate_id"] == fastest["candidate_id"]
        ),
    }


def run(args):
    adapter_dir = Path(args.adapter_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    input_hash = verify(adapter_dir)
    adapter = read_json(adapter_dir / "summary.json")
    if adapter.get("status") != "complete_post_board_replay_adapter":
        raise RuntimeError("complete post-board adapter is required")
    observed = {
        row["candidate_id"]: row
        for row in read_jsonl(adapter_dir / "cheng_rows.jsonl")
    }
    groups = defaultdict(list)
    for workload_id in ("R18-H1", "R18-H2", "R18-H3"):
        contract = read_json(
            adapter_dir / "{}_workload_contract.json".format(workload_id.lower())
        )
        for row in contract["candidates"]:
            groups[(workload_id, row["family_id"])].append(row)
    families = []
    for rows in groups.values():
        value = analyze_family(rows, observed)
        if value is not None:
            families.append(value)
    families.sort(key=lambda row: (row["workload_id"], row["family_id"]))
    output.mkdir(parents=True)
    (output / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in families),
        encoding="utf-8",
    )
    by_workload = {}
    for workload_id in ("R18-H1", "R18-H2", "R18-H3"):
        rows = [row for row in families if row["workload_id"] == workload_id]
        resolved = [row for row in rows if row["all_four_semantically_resolved"]]
        by_workload[workload_id] = {
            "families_with_original": len(rows),
            "four_scheme_resolved": len(resolved),
            "minimum_access_equals_fastest": sum(
                row["minimum_access_is_fastest"] for row in resolved
            ),
            "minimum_access_differs_from_fastest": sum(
                not row["minimum_access_is_fastest"] for row in resolved
            ),
            "scheme_status_counts": dict(
                Counter(
                    scheme["status"]
                    for row in rows
                    for scheme in row["schemes"]
                )
            ),
        }
    summary = {
        "schema": "c3_resnet18_cheng_four_scheme_analysis_v1",
        "status": "complete_method_consistent_cheng_four_scheme_analysis",
        "workloads": by_workload,
        "family_count": len(families),
        "input_adapter_manifest_sha256": input_hash,
        "fallback_rule": "resource-not-applicable modes inherit exact same-tile original; compiler/FSim/FPGA failures do not silently fall back",
        "claim_boundary": (
            "Functional four-scheme reproduction with explicit barrier residency; not "
            "the authors' unpublished non-overwriting-address source implementation. "
            "Logical DMA is not physical AXI traffic."
        ),
    }
    write_json(output / "summary.json", summary)
    write_json(
        output / "contract.json",
        {
            "schema": summary["schema"],
            "source_adapter_manifest_sha256": input_hash,
            "modes": list(MODES),
            "board_contacted": False,
            "post_board_analysis_only": True,
        },
    )
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                path.name: baseline.sha256_file(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "source_sha256": baseline.sha256_file(Path(__file__).resolve()),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
