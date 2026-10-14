#!/usr/bin/env python3
"""Build the versioned C3 selector-v2 development table.

This adapter deliberately preserves evidence provenance.  In particular,
command UOP bytes/peaks are never relabelled as cumulative LOAD-UOP bytes.
Unknown quantities are written as null and are handled by selector-v2's
training-fold imputation plus missing indicators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


HERE = Path(__file__).resolve().parent
C3 = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune"
GROUPED = C3 / "07_grouped_holdout"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_selector_v2_dataset_run01"
SCHEMA = "c3_selector_v2_dataset_v1"

CANONICAL_POOLS = (
    "20260912_p7r144_y00_layout_preserving_search_board_run01",
    "20260912_p7r154_y03_layout_preserving_search_board_run01",
    "20260912_p7r211_r50a_canonical_completed_pool_run01",
    "20260912_p7r222_r50b_canonical_completed_pool_run01",
    "20260913_p7r303_r50c_canonical_completed_pool_run01",
)
P7Q_CONTRACT = GROUPED / "20260911_p7q_qualified_timing_contract_run01/contract.json"
P7Q_PAIRS = GROUPED / "20260911_p7q5_search_analysis_run01/same_tile_effects.json"
RECENT_POOL = GROUPED / "20260924_c3_real_task_board_pool_freeze_run01/candidates.jsonl"
RECENT_SELECTOR = GROUPED / "20260924_c3_real_task_selector_freeze_run01/program_features.jsonl"
RECENT_BOOT1 = GROUPED / "20260924_c3_real_task_board_boot1_run02/reference_summary.json"
RECENT_BOOT2 = GROUPED / "20260924_c3_real_task_confirmation_board_boot2_run01/reference_summary.json"
RECENT_CROSS = GROUPED / "20260924_c3_real_task_cross_boot_analysis_run01/summary.json"

LOGICAL_HARDWARE_FINGERPRINT = {
    "target_family": "axu5evb",
    "batch": 1,
    "block_in": 16,
    "block_out": 16,
    "inp_sram_vectors": 2048,
    "wgt_sram_vectors": 1024,
    "acc_sram_vectors": 2048,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def blank_metrics() -> Dict[str, Any]:
    return {
        "inp_request_bytes": None,
        "wgt_request_bytes": None,
        "acc_request_bytes": None,
        "out_request_bytes": None,
        "total_request_bytes": None,
        "inp_request_calls": None,
        "wgt_request_calls": None,
        "acc_request_calls": None,
        "out_request_calls": None,
        "total_request_calls": None,
        "uop_load_bytes": None,
        "uop_load_calls": None,
        "command_uop_bytes": None,
        "instruction_bytes": None,
        "submissions": None,
        "sync_count": None,
        "inp_arena_fraction": None,
        "wgt_arena_fraction": None,
        "acc_arena_fraction": None,
        "uop_arena_fraction": None,
    }


def normalize_mode(value: Any) -> str:
    return {
        "weight_resident_barrier": "weight_stationary_barrier",
        "input_weight_resident_barrier": "input_weight_staged",
    }.get(str(value), str(value))


def tile_from_entity(entity: Mapping[str, Any]) -> Dict[str, Any]:
    result = {name: None for name in ("tile_b", "tile_h", "tile_w", "tile_ci", "tile_co",
                                      "oc_nthread", "h_nthread")}
    for item in (entity or {}).get("entity", []):
        if len(item) != 3 or item[0] not in result:
            continue
        value = item[2]
        if item[1] == "sp" and isinstance(value, list):
            value = value[-1]
        result[item[0]] = value
    return result


def canonical_metrics(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    values = candidate["predispatch"]["validity"]
    metrics = blank_metrics()
    metrics.update({
        "inp_request_bytes": values.get("static_input_dma_bytes"),
        "wgt_request_bytes": values.get("static_weight_dma_bytes"),
        "out_request_bytes": values.get("static_output_dma_bytes"),
        "total_request_bytes": values.get("static_dma_total_bytes"),
        "inp_request_calls": values.get("static_input_dma_calls"),
        "wgt_request_calls": values.get("static_weight_dma_calls"),
        "out_request_calls": values.get("static_output_dma_calls"),
        "total_request_calls": values.get("static_dma_total_calls"),
        # This source only records a command-buffer peak.  It is intentionally
        # not copied into uop_load_bytes.
        "command_uop_bytes": values.get("static_uop_peak_bytes"),
        "instruction_bytes": values.get("static_insn_peak_bytes"),
        "submissions": values.get("static_submissions"),
        "sync_count": values.get("static_residency_drains"),
    })
    for prefix in ("input", "weight", "accumulator"):
        required = values.get(f"static_{prefix}_sram_required_vectors")
        headroom = values.get(f"static_{prefix}_sram_headroom_vectors")
        if required is not None and headroom is not None and required + headroom > 0:
            key = {"input": "inp", "weight": "wgt", "accumulator": "acc"}[prefix]
            metrics[f"{key}_arena_fraction"] = required / (required + headroom)
    return metrics


def canonical_fingerprint_ok(candidates: List[Mapping[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    observed = {"inp": set(), "wgt": set(), "acc": set()}
    names = {"inp": "input", "wgt": "weight", "acc": "accumulator"}
    for candidate in candidates:
        values = candidate["predispatch"]["validity"]
        for short, prefix in names.items():
            required = values.get(f"static_{prefix}_sram_required_vectors")
            headroom = values.get(f"static_{prefix}_sram_headroom_vectors")
            if required is not None and headroom is not None:
                observed[short].add(int(required + headroom))
    expected = {"inp": {2048}, "wgt": {1024}, "acc": {2048}}
    return observed == expected, {key: sorted(value) for key, value in observed.items()}


def make_row(*, task: str, context: str, candidate_id: str, incumbent_id: str,
             candidate_mode: str, incumbent_mode: str, candidate_tile: Mapping[str, Any],
             incumbent_tile: Mapping[str, Any], candidate_metrics: Mapping[str, Any],
             incumbent_metrics: Mapping[str, Any], candidate_latency: float,
             incumbent_latency: float, source: str, incumbent_policy: str,
             feature_provenance: Mapping[str, Any], fingerprint_evidence: Mapping[str, Any],
             fpga_correct: bool = True) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "execution_task_id": task,
        "comparison_context_id": context,
        "candidate_id": candidate_id,
        "incumbent_id": incumbent_id,
        "incumbent_policy": incumbent_policy,
        "candidate_mode": candidate_mode,
        "incumbent_mode": incumbent_mode,
        "candidate_tile": dict(candidate_tile),
        "incumbent_tile": dict(incumbent_tile),
        "candidate_metrics": dict(candidate_metrics),
        "incumbent_metrics": dict(incumbent_metrics),
        "candidate_latency_ms": candidate_latency,
        "incumbent_latency_ms": incumbent_latency,
        "target_log_latency_ratio": math.log(candidate_latency / incumbent_latency),
        "fpga_correct": bool(fpga_correct),
        "hardware_fingerprint": LOGICAL_HARDWARE_FINGERPRINT,
        "hardware_fingerprint_evidence": dict(fingerprint_evidence),
        "feature_provenance": dict(feature_provenance),
        "source": source,
    }


def adapt_canonical_pool(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    completed = load_json(path / "completed_pool.json")
    task, payload = next(iter(completed["workloads"].items()))
    candidates = payload["candidates"]
    fingerprint_ok, observed = canonical_fingerprint_ok(candidates)
    if not fingerprint_ok:
        return [], {"source": str(path), "accepted": False,
                    "reason": "logical_hardware_fingerprint_mismatch", "observed": observed}
    correct = [candidate for candidate in candidates
               if candidate.get("oracle", {}).get("correctness_seed_checks") == 3
               and candidate.get("oracle", {}).get("latency_ms") is not None]
    incumbent_pool = [candidate for candidate in correct
                      if candidate.get("residence_mode") in ("original", "input_stationary")]
    if not incumbent_pool:
        return [], {"source": str(path), "accepted": False, "reason": "no_incumbent_candidate"}
    # These are development labels, so selecting the current known best
    # already-deployed mechanism is permitted.  It is recorded explicitly and
    # is never reused to define an unseen holdout.
    incumbent = min(incumbent_pool,
                    key=lambda row: (float(row["oracle"]["latency_ms"]), row["candidate_id"]))
    incumbent_metrics = canonical_metrics(incumbent)
    incumbent_tile = dict(incumbent["predispatch"]["knobs"])
    rows = []
    for candidate in correct:
        if candidate["candidate_id"] == incumbent["candidate_id"]:
            continue
        rows.append(make_row(
            task=task,
            context=f"{task}:current-known-incumbent",
            candidate_id=candidate["candidate_id"],
            incumbent_id=incumbent["candidate_id"],
            candidate_mode=normalize_mode(candidate["residence_mode"]),
            incumbent_mode=normalize_mode(incumbent["residence_mode"]),
            candidate_tile=candidate["predispatch"]["knobs"],
            incumbent_tile=incumbent_tile,
            candidate_metrics=canonical_metrics(candidate),
            incumbent_metrics=incumbent_metrics,
            candidate_latency=float(candidate["oracle"]["latency_ms"]),
            incumbent_latency=float(incumbent["oracle"]["latency_ms"]),
            source=str(path / "completed_pool.json"),
            incumbent_policy="fastest FPGA-correct original-or-input-stationary development candidate",
            feature_provenance={
                "requests_and_calls": "predispatch.validity static_*",
                "command_uop_bytes": "static_uop_peak_bytes; command peak, not LOAD-UOP cumulative bytes",
                "instruction_bytes": "static_insn_peak_bytes",
                "arena": "required_vectors/(required_vectors+headroom_vectors)",
                "latency": "oracle.latency_ms from completed pool",
            },
            fingerprint_evidence={
                "kind": "logical_sram_capacity_signature",
                "observed_vectors": observed,
                "status": "matched",
            },
        ))
    return rows, {"source": str(path), "accepted": True, "task": task,
                  "rows": len(rows), "incumbent_id": incumbent["candidate_id"],
                  "logical_fingerprint": observed}


def p7q_metrics(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    values = candidate["static_metrics"]
    metrics = blank_metrics()
    metrics.update({
        "inp_request_bytes": values.get("input_dma_bytes"),
        "wgt_request_bytes": values.get("weight_dma_bytes"),
        "out_request_bytes": values.get("output_dma_bytes"),
        "total_request_bytes": values.get("dma_total_bytes"),
        "inp_request_calls": values.get("input_dma_calls"),
        "wgt_request_calls": values.get("weight_dma_calls"),
        "out_request_calls": values.get("output_dma_calls"),
        "total_request_calls": values.get("dma_total_calls"),
        "command_uop_bytes": values.get("total_uop_bytes"),
        "instruction_bytes": values.get("total_insn_bytes"),
        "submissions": values.get("submissions"),
        "sync_count": max(0, int(values.get("source_derived_finish_count", 1)) - 1),
    })
    return metrics


def adapt_p7q() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    contract = load_json(P7Q_CONTRACT)
    pairs = load_json(P7Q_PAIRS)["pairs"]
    candidates: Dict[str, Dict[str, Any]] = {}
    for payload in contract["workloads"].values():
        for candidate in payload["gross_candidates"]:
            candidates[candidate["candidate_id"]] = candidate
    rows = []
    rejected = []
    for pair in pairs:
        candidate = candidates.get(pair["mechanism_candidate_id"])
        incumbent = candidates.get(pair["control_candidate_id"])
        if candidate is None or incumbent is None:
            rejected.append({"pair": pair, "reason": "identity_missing_from_frozen_contract"})
            continue
        if "AXU cross-compile" not in str(candidate.get("qualification")):
            rejected.append({"candidate_id": candidate["candidate_id"],
                             "reason": "missing_axu_cross_compile_qualification"})
            continue
        task = pair["workload_id"]
        context = f"{task}:same-tile:{pair['control_candidate_id']}"
        rows.append(make_row(
            task=task,
            context=context,
            candidate_id=candidate["candidate_id"],
            incumbent_id=incumbent["candidate_id"],
            candidate_mode=normalize_mode(candidate["residence_mode"]),
            incumbent_mode=normalize_mode(incumbent["residence_mode"]),
            candidate_tile=tile_from_entity(candidate["complete_config_entity"]),
            incumbent_tile=tile_from_entity(incumbent["complete_config_entity"]),
            candidate_metrics=p7q_metrics(candidate),
            incumbent_metrics=p7q_metrics(incumbent),
            candidate_latency=float(pair["mechanism_median_ms"]),
            incumbent_latency=float(pair["control_median_ms"]),
            source=str(P7Q_PAIRS),
            incumbent_policy="same-tile original control; development ablation, not unique production incumbent",
            feature_provenance={
                "requests_calls_command": "frozen P7Q contract gross_candidates.static_metrics",
                "uop_load_bytes": "unknown; total_uop_bytes retained separately as command_uop_bytes",
                "latency": "same_tile_effects median-of-five qualified FPGA timing",
            },
            fingerprint_evidence={
                "kind": "frozen AXU qualification and source-guard chain",
                "status": "matched_logical_target_family",
                "qualification": candidate["qualification"],
                "source_guards_sha256": candidate.get("source_guards_sha256"),
            },
        ))
    return rows, {"source": str(P7Q_CONTRACT), "accepted": True,
                  "tasks": sorted({row["execution_task_id"] for row in rows}),
                  "rows": len(rows), "rejected_pairs": rejected}


def recent_metrics(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = blank_metrics()
    dma = candidate.get("dma_features", {})
    logical = dma.get("logical_bytes", {})
    totals = dma.get("totals", {})
    command = candidate.get("command_features", {}).get("values", {})
    command_totals = command.get("totals", {})
    metrics.update({
        "inp_request_bytes": logical.get("inp"),
        "wgt_request_bytes": logical.get("wgt"),
        "acc_request_bytes": logical.get("acc"),
        "out_request_bytes": logical.get("out"),
        "total_request_bytes": (totals.get("load_buffer_2d_bytes", 0)
                                + totals.get("store_buffer_2d_bytes", 0)),
        "inp_request_calls": totals.get("load_buffer_2d_inp_calls"),
        "wgt_request_calls": totals.get("load_buffer_2d_wgt_calls"),
        "acc_request_calls": totals.get("load_buffer_2d_acc_calls"),
        "out_request_calls": totals.get("store_buffer_2d_out_calls"),
        "total_request_calls": (totals.get("load_buffer_2d_calls", 0)
                                + totals.get("store_buffer_2d_calls", 0)),
        "command_uop_bytes": command_totals.get("uop_bytes"),
        "instruction_bytes": command_totals.get("insn_bytes"),
        "submissions": command.get("submissions"),
        "sync_count": candidate.get("sync", {}).get("explicit_residency_drains"),
    })
    return metrics


def adapt_recent_cross_boot() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidates = load_jsonl(RECENT_POOL)
    by_candidate = {row["candidate_id"]: row for row in candidates}
    program_bindings = load_jsonl(RECENT_SELECTOR)
    by_program = {row["program_id"]: by_candidate[row["candidate_id"]]
                  for row in program_bindings}
    boot1 = load_json(RECENT_BOOT1)
    boot2 = load_json(RECENT_BOOT2)
    cross = load_json(RECENT_CROSS)
    rows = []
    task_summaries = {}
    expected_fp = {
        "target": "axu5evb", "batch": 1, "block_in": 16, "block_out": 16,
        "inp_buff_size": 32768, "wgt_buff_size": 262144, "acc_buff_size": 131072,
    }
    for task, result in cross["workloads"].items():
        incumbent_id = result["incumbent_program_id"]
        correct = (set(boot1[task]["fpga_correct_program_ids"])
                   & set(boot2[task]["cross_boot_correct_program_ids"]))
        if incumbent_id not in correct:
            raise RuntimeError(f"{task}: incumbent not FPGA-correct across both boots")
        incumbent = by_program[incumbent_id]
        fingerprint = incumbent["hardware_fingerprint"]
        for key, value in expected_fp.items():
            if fingerprint.get(key) != value:
                raise RuntimeError(f"{task}: hardware fingerprint mismatch for {key}")
        def latency(program_id: str) -> float:
            return float(median([
                boot1[task]["program_statistics"][program_id]["median_of_round_medians_ms"],
                boot2[task]["program_statistics"][program_id]["median_of_round_medians_ms"],
            ]))
        incumbent_latency = latency(incumbent_id)
        task_rows = 0
        for program_id in sorted(correct):
            if program_id == incumbent_id:
                continue
            candidate = by_program[program_id]
            rows.append(make_row(
                task=task,
                context=f"{task}:cross-boot-current-incumbent",
                candidate_id=program_id,
                incumbent_id=incumbent_id,
                candidate_mode=normalize_mode(candidate["residence_mode"]),
                incumbent_mode=normalize_mode(incumbent["residence_mode"]),
                candidate_tile=candidate["visible_features"],
                incumbent_tile=incumbent["visible_features"],
                candidate_metrics=recent_metrics(candidate),
                incumbent_metrics=recent_metrics(incumbent),
                candidate_latency=latency(program_id),
                incumbent_latency=incumbent_latency,
                source=str(RECENT_CROSS),
                incumbent_policy="frozen current incumbent from selector-v1 study",
                feature_provenance={
                    "requests_calls": "real lowered DMA features",
                    "command_uop_bytes": "command totals.uop_bytes; not LOAD-UOP cumulative bytes",
                    "latency": "median of the two independent-boot median-of-round-medians",
                },
                fingerprint_evidence={
                    "kind": "explicit VTA hardware fingerprint plus identical bitstream/runtime attestation",
                    "status": "matched",
                    "bitstream_sha256": "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6",
                    "runtime_sha256": "04e894baf305311315fd0c79d03ec2a6375b9591916e9c6f429cf3697c4457b2",
                },
            ))
            task_rows += 1
        task_summaries[task] = {"rows": task_rows, "incumbent_id": incumbent_id,
                                "fpga_correct_programs": len(correct)}
    return rows, {"source": str(RECENT_CROSS), "accepted": True, "tasks": task_summaries}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable output: {args.output}")
    args.output.mkdir(parents=True)

    rows: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    inputs = [P7Q_CONTRACT, P7Q_PAIRS, RECENT_POOL, RECENT_SELECTOR,
              RECENT_BOOT1, RECENT_BOOT2, RECENT_CROSS]
    for name in CANONICAL_POOLS:
        path = GROUPED / name
        adapted, status = adapt_canonical_pool(path)
        rows.extend(adapted)
        sources.append(status)
        inputs.extend([path / "completed_pool.json", path / "summary.json", path / "artifact_hashes.json"])
    adapted, status = adapt_p7q()
    rows.extend(adapted)
    sources.append(status)
    adapted, status = adapt_recent_cross_boot()
    rows.extend(adapted)
    sources.append(status)

    rows.sort(key=lambda row: (row["execution_task_id"], row["comparison_context_id"], row["candidate_id"]))
    write_jsonl(args.output / "training_rows.jsonl", rows)
    write_json(args.output / "source_admission.json", sources)
    hashes = {str(path.resolve()): sha256(path) for path in inputs}
    hashes[str((args.output / "training_rows.jsonl").resolve())] = sha256(args.output / "training_rows.jsonl")
    write_json(args.output / "input_hashes.json", hashes)
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "row_count": len(rows),
        "execution_task_count": len({row["execution_task_id"] for row in rows}),
        "execution_tasks": sorted({row["execution_task_id"] for row in rows}),
        "target_rows": sum(row.get("target_log_latency_ratio") is not None for row in rows),
        "target_coverage": (sum(row.get("target_log_latency_ratio") is not None for row in rows) / len(rows)
                            if rows else 0.0),
        "hardware_fingerprint": LOGICAL_HARDWARE_FINGERPRINT,
        "claim_boundary": (
            "development-only table; P7Q rows use matching same-tile original controls, while completed "
            "pools use a unique current-known incumbent; no row is a latency-unseen confirmation label"
        ),
        "unknown_policy": "null; never converted to observed zero",
    }
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "artifact_hashes.json", {
        name: sha256(args.output / name)
        for name in ("training_rows.jsonl", "source_admission.json", "input_hashes.json", "summary.json")
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
