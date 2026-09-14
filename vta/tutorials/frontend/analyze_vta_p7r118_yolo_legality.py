#!/usr/bin/env python3
"""Explain Y02 VTA tile legality without TopHub or performance labels.

The complete P7R115 hardware-eligible Y02 domain is first lowered in original
mode.  Only original-legal entities are then lowered in the other three modes,
which is exhaustive for the four-mode intersection because a failed original
can never belong to that intersection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import tvm
from tvm import autotvm
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from c3_candidate_identity import canonical_json_bytes, normalize_config_entity
from generate_vta_residency_candidates import classify_lower_failure


SCHEMA = "c3_p7r118_yolo_legality_v1"
MODE_NUMBERS = {
    "original": 0,
    "input_stationary": 1,
    "weight_stationary": 2,
    "paper_inspired_hybrid": 3,
}
MINIMUM_USEFUL_INTERSECTION = 8

HERE = Path(__file__).resolve().parent
C3 = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P7R115 = C3 / "07_grouped_holdout" / "20260911_p7r115_yolo_confirmation_protocol_run01"
P7R117B = C3 / "07_grouped_holdout" / "20260911_p7r117b_yolo96_static_run01"
DEFAULT_CANDIDATES = P7R115 / "candidates.jsonl"
DEFAULT_DOMAIN = P7R115 / "complete_domain.jsonl"
DEFAULT_CONTEXT = P7R117B / "static_results.jsonl"
DEFAULT_OUTPUT = C3 / "07_grouped_holdout" / "20260911_p7r118_y02_legality_run01"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_frozen(path):
    path = Path(path)
    ledger = load_json(path.parent / "artifact_hashes.json")["artifacts"]
    observed = sha256_file(path)
    if ledger.get(path.name) != observed:
        raise ValueError("frozen input hash mismatch: {}".format(path))
    return {"path": str(path.resolve()), "sha256": observed}


def config_knobs(entity):
    return {
        name: int(value[-1] if kind == "sp" else value)
        for name, kind, value in entity["entity"]
    }


def original_legality_rule(knobs, output_height):
    """Y02 scoped rule inferred from DMA padding and 2-D representation limits."""

    return bool(
        knobs["tile_ci"] == 1
        and (knobs["tile_co"] == 1 or knobs["tile_h"] in (1, output_height))
    )


def four_mode_intersection_rule(knobs, output_height):
    """Y02 scoped extension for current width-grouping residency schedules."""

    return bool(
        original_legality_rule(knobs, output_height)
        and (
            knobs["tile_w"] == 2
            or (knobs["tile_co"] == 1 and knobs["tile_h"] == 1)
        )
    )


def confusion_matrix(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("actual/predicted length mismatch")
    tp = sum(a and p for a, p in zip(actual, predicted))
    tn = sum(not a and not p for a, p in zip(actual, predicted))
    fp = sum(not a and p for a, p in zip(actual, predicted))
    fn = sum(a and not p for a, p in zip(actual, predicted))
    total = len(actual)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
        "accuracy": None if not total else (tp + tn) / total,
        "precision": None if tp + fp == 0 else tp / (tp + fp),
        "recall": None if tp + fn == 0 else tp / (tp + fn),
        "specificity": None if tn + fp == 0 else tn / (tn + fp),
    }


def create_tasks(workload, env):
    return {
        name: autotvm.task.create(
            "conv2d_packed_residency.vta",
            args=tuple(workload[1:]) + (number,),
            target=env.target,
            target_host=env.target_host,
        )
        for name, number in MODE_NUMBERS.items()
    }


def lower_one(task, index, expected_entity):
    start = time.monotonic()
    phase = "identity"
    try:
        config = task.config_space.get(int(index))
        actual = normalize_config_entity(config.to_json_dict())
        if canonical_json_bytes(actual) != canonical_json_bytes(expected_entity):
            raise RuntimeError("ConfigEntity differs from frozen complete domain")
        phase = "instantiate"
        with task.target:
            schedule, tensors = task.instantiate(config)
        phase = "tir_lower"
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            module = tvm.lower(schedule, tensors, name="main")
        return {
            "status": "ok",
            "failure": None,
            "tir_sha256": hashlib.sha256(
                tvm.ir.save_json(module).encode("utf-8")
            ).hexdigest(),
            "wall_seconds": time.monotonic() - start,
            "wall_time_semantics": "compiler feasibility cost only; not latency",
        }
    except Exception as error:  # TVM exposes multiple rejection exception types.
        message = str(error) or type(error).__name__
        subcategory = (
            "identity_mismatch"
            if phase == "identity"
            else classify_lower_failure(message, phase)
        )
        return {
            "status": "failed",
            "failure": {
                "category": "environment" if phase == "identity" else "lower",
                "subcategory": subcategory,
                "phase": phase,
                "exception_type": type(error).__name__,
                "message": message[:4000],
            },
            "tir_sha256": None,
            "wall_seconds": time.monotonic() - start,
            "wall_time_semantics": "compiler feasibility cost only; not latency",
        }


def context_summary(rows):
    selected = [row for row in rows if row.get("workload_id") == "Y02"]
    return {
        "rows": len(selected),
        "status_counts": dict(sorted(Counter(row["status"] for row in selected).items())),
        "failure_subcategories": dict(
            sorted(
                Counter(
                    row.get("failure", {}).get("subcategory", "unspecified")
                    for row in selected
                    if row.get("failure")
                ).items()
            )
        ),
        "role": "context-only; not used to select or fit the complete-domain scan",
    }


def build_report(original_rows, intersections, output_height, context, inputs):
    actual_original = [row["original"]["status"] == "ok" for row in original_rows]
    padding_only = [row["knobs"]["tile_ci"] == 1 for row in original_rows]
    original_rule = [
        original_legality_rule(row["knobs"], output_height) for row in original_rows
    ]
    intersection_indices = {
        row["debug"]["config_index"]
        for row in intersections
        if row["all_four_modes_legal"]
    }
    actual_intersection = [
        row["debug"]["config_index"] in intersection_indices for row in original_rows
    ]
    intersection_rule = [
        four_mode_intersection_rule(row["knobs"], output_height)
        for row in original_rows
    ]
    original_legal = sum(actual_original)
    four_legal = sum(actual_intersection)
    failures = Counter(
        row["original"]["failure"]["subcategory"]
        for row in original_rows
        if row["original"]["status"] != "ok"
    )
    mode_counts = {
        mode: dict(
            sorted(Counter(row["modes"][mode]["status"] for row in intersections).items())
        )
        for mode in MODE_NUMBERS
    }
    mode_failures = {
        mode: dict(
            sorted(
                Counter(
                    row["modes"][mode]["failure"]["subcategory"]
                    for row in intersections
                    if row["modes"][mode]["status"] != "ok"
                ).items()
            )
        )
        for mode in MODE_NUMBERS
    }
    return {
        "schema": SCHEMA,
        "status": "completed_local_legality_only",
        "scope": "Y02 packed conv geometry on the frozen VTA schedule/hardware fingerprint",
        "inputs": inputs,
        "p7r117b_context": context,
        "population": {
            "p7r115_hardware_eligible": len(original_rows),
            "original_lower_legal": original_legal,
            "original_lower_legal_fraction": original_legal / len(original_rows),
            "original_failure_subcategories": dict(sorted(failures.items())),
            "other_modes_scanned": "all original-legal entities",
            "four_mode_legal": four_legal,
            "four_mode_fraction_of_hardware_eligible": four_legal / len(original_rows),
            "four_mode_fraction_of_original_legal": four_legal / original_legal,
            "per_mode_status_on_original_legal": mode_counts,
            "per_mode_failures_on_original_legal": mode_failures,
        },
        "rules": {
            "prior_hardware_predicate_only": {
                "expression": "True within this already-filtered 343-point population",
                "confusion": confusion_matrix(actual_original, [True] * len(original_rows)),
                "interpretation": "SRAM-vector/reuse necessary conditions alone do not predict DMA lowering legality",
            },
            "padding_rule_only": {
                "expression": "tile_ci == 1",
                "confusion": confusion_matrix(actual_original, padding_only),
                "interpretation": "eliminates all channel-inner padding failures but not unsupported 2-D DMA layouts",
            },
            "original_legality_v1": {
                "expression": "tile_ci == 1 and (tile_co == 1 or tile_h in {1, output_height})",
                "output_height": output_height,
                "confusion": confusion_matrix(actual_original, original_rule),
                "scope_limit": "empirically exact for Y02 and current source hashes; compiler remains final authority",
            },
            "four_mode_intersection_v1": {
                "expression": "original_legality_v1 and (tile_w == 2 or (tile_co == 1 and tile_h == 1))",
                "confusion": confusion_matrix(actual_intersection, intersection_rule),
                "scope_limit": "empirically exact for Y02 and current mode-0..3 schedule sources",
            },
        },
        "mechanism_explanation": [
            {
                "failure": "dma_pad_innermost",
                "tile_condition": "tile_ci > 1",
                "reason": "the 3x3 padding reaches a reduction/channel-inner dimension after tiling; InjectDMAIntrin accepts padding only on the two spatial dimensions and rejects nonzero innermost-block padding",
            },
            {
                "failure": "dma_2d_pattern",
                "tile_condition": "tile_ci == 1, tile_co > 1, and 1 < tile_h < output_height",
                "reason": "the remaining tiled strides cannot be folded into VTA's supported 2-D DMA descriptor; tile_h=1 or a full-height tile removes that intermediate stride",
            },
            {
                "failure": "dma_compact_buffer in weight/hybrid modes",
                "tile_condition": "most original-legal points with even output-width outer extent",
                "reason": "the current modes group two adjacent outer-width tiles; the promoted/grouped buffer then has gaps, so _check_compact rejects it. tile_w=2 makes 26/2=13 outer tiles and disables factor-2 grouping; the degenerate tile_co=tile_h=1 layouts also remain compact",
            },
            {
                "failure": "allocation_capacity",
                "tile_condition": "mode-dependent promoted/grouped lifetime",
                "reason": "the earlier vector formula is necessary-only; schedule placement may keep a larger accumulator/input region live. Compiler allocation bounds are the final memory-legality certificate",
            },
        ],
        "replacement_decision": {
            "minimum_required_four_mode_entities": MINIMUM_USEFUL_INTERSECTION,
            "observed_four_mode_entities": four_legal,
            "sufficient": four_legal >= MINIMUM_USEFUL_INTERSECTION,
            "other_yolo_geometry_scan": "not_triggered" if four_legal >= MINIMUM_USEFUL_INTERSECTION else "required",
            "reason": "Y02 retains enough entities for an N=8 per-geometry confirmation rule" if four_legal >= MINIMUM_USEFUL_INTERSECTION else "Y02 intersection is too small",
        },
        "claim_boundary": "no FSim correctness, board latency, FPS, or performance winner claim",
        "tophub_entity_or_cost_read": False,
        "board_contacted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--complete-domain", type=Path, default=DEFAULT_DOMAIN)
    parser.add_argument("--p7r117b-context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(args.output_dir))

    inputs = {
        "candidates": verify_frozen(args.candidates),
        "complete_domain": verify_frozen(args.complete_domain),
        "p7r117b_context": verify_frozen(args.p7r117b_context),
    }
    # Candidate/domain sources contain no TopHub fields.  The P7R115 contract is
    # deliberately not opened by this analysis.
    candidate_bytes = args.candidates.read_bytes().lower()
    domain_bytes = args.complete_domain.read_bytes().lower()
    if b"tophub" in candidate_bytes or b"tophub" in domain_bytes:
        raise ValueError("legality inputs unexpectedly contain TopHub material")
    candidates = load_jsonl(args.candidates)
    y02_candidates = [row for row in candidates if row["workload_id"] == "Y02"]
    workloads = {canonical_json_bytes(row["identity"]["workload"]) for row in y02_candidates}
    if len(workloads) != 1:
        raise ValueError("Y02 candidates do not have exactly one workload")
    workload = y02_candidates[0]["identity"]["workload"]
    output_height = int(workload[1][1][2])
    domain = [
        row
        for row in load_jsonl(args.complete_domain)
        if row["workload_id"] == "Y02" and row["hardware_predicate"]["passed"]
    ]
    if len(domain) != 343:
        raise ValueError("frozen Y02 hardware-eligible population must be 343, got {}".format(len(domain)))
    args.output_dir.mkdir(parents=True)
    preregistered = {
        "schema": "c3_p7r118_legality_protocol_v1",
        "status": "frozen_before_complete_original_scan",
        "population": "all 343 P7R115 Y02 hardware-predicate-passed entities",
        "order": "ascending audit config_index; index is not an identity or feature",
        "stage_1": "lower original for all 343 entities",
        "stage_2": "lower input/weight/hybrid for every original-legal entity",
        "minimum_useful_four_mode_intersection": MINIMUM_USEFUL_INTERSECTION,
        "replacement_rule": "scan another unexposed YOLO geometry only if intersection has fewer than eight entities",
        "performance_labels": "none",
        "tophub_entity_or_cost": "not read",
        "inputs": inputs,
    }
    write_json(args.output_dir / "preregistered.json", preregistered)
    write_json(
        args.output_dir / "pre_scan_hashes.json",
        {"artifacts": {"preregistered.json": sha256_file(args.output_dir / "preregistered.json")}},
    )

    env = vta.get_env()
    tasks = create_tasks(workload, env)
    original_rows = []
    original_path = args.output_dir / "original_scan.jsonl"
    original_path.touch(exist_ok=False)
    for position, row in enumerate(sorted(domain, key=lambda item: item["debug"]["config_index"])):
        entity = row["complete_config_entity"]
        knobs = config_knobs(entity)
        result = lower_one(tasks["original"], row["debug"]["config_index"], entity)
        record = {
            "schema": SCHEMA,
            "scan_position": position,
            "semantic_config_sha256": row["semantic_config_sha256"],
            "debug": row["debug"],
            "complete_config_entity": entity,
            "knobs": knobs,
            "p7r115_hardware_predicate": row["hardware_predicate"],
            "rule_predictions": {
                "padding_only": knobs["tile_ci"] == 1,
                "original_legality_v1": original_legality_rule(knobs, output_height),
                "four_mode_intersection_v1": four_mode_intersection_rule(knobs, output_height),
            },
            "original": result,
        }
        original_rows.append(record)
        with original_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    intersections = []
    intersection_path = args.output_dir / "four_mode_scan.jsonl"
    intersection_path.touch(exist_ok=False)
    for record in original_rows:
        if record["original"]["status"] != "ok":
            continue
        modes = {"original": record["original"]}
        for mode in ("input_stationary", "weight_stationary", "paper_inspired_hybrid"):
            modes[mode] = lower_one(
                tasks[mode],
                record["debug"]["config_index"],
                record["complete_config_entity"],
            )
        item = {
            "schema": SCHEMA,
            "semantic_config_sha256": record["semantic_config_sha256"],
            "debug": record["debug"],
            "knobs": record["knobs"],
            "modes": modes,
            "all_four_modes_legal": all(result["status"] == "ok" for result in modes.values()),
        }
        intersections.append(item)
        with intersection_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, sort_keys=True) + "\n")

    context = context_summary(load_jsonl(args.p7r117b_context))
    report = build_report(original_rows, intersections, output_height, context, inputs)
    write_json(args.output_dir / "report.json", report)
    write_json(args.output_dir / "legality_rules.json", report["rules"])
    population = report["population"]
    status = [
        "# P7R118 Y02 legality exploration",
        "",
        "- Status: `completed_local_legality_only`",
        "- Hardware-eligible domain: {}".format(population["p7r115_hardware_eligible"]),
        "- Original lower legal: {} ({:.2%})".format(
            population["original_lower_legal"], population["original_lower_legal_fraction"]
        ),
        "- Four-mode intersection: {} ({:.2%} of complete eligible domain)".format(
            population["four_mode_legal"], population["four_mode_fraction_of_hardware_eligible"]
        ),
        "- Scoped original rule: `tile_ci=1 && (tile_co=1 || tile_h in {1,H})`",
        "- Scoped four-mode rule adds: `tile_w=2 || (tile_co=1 && tile_h=1)`",
        "- Replacement geometry scan: `{}`".format(report["replacement_decision"]["other_yolo_geometry_scan"]),
        "- TopHub entity/cost read: no",
        "- Board contacted: no",
    ]
    (args.output_dir / "STATUS.md").write_text("\n".join(status) + "\n", encoding="utf-8")
    (args.output_dir / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(args.output_dir.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(args.output_dir / "artifact_hashes.json", {"artifacts": hashes})
    print(json.dumps(population, indent=2, sort_keys=True))


if __name__ == "__main__":
    import sys

    main()
