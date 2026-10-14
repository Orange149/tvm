#!/usr/bin/env python3
"""Read-only primary-evidence audit, writing only a new audit directory."""
import argparse
from collections import Counter
from pathlib import Path
import statistics
import time

import run_vta_c3_reuse_supplement as s


def audit(board_dir, out):
    started = time.monotonic()
    contract = s.verify(board_dir)
    summary = s.read(board_dir / "summary.json")
    assert summary["status"] == "COMPLETE" and summary["timing_valid"]
    assert not (board_dir / "invalid_session.json").exists()
    s.check_bindings(contract["source_bindings"])
    prepared = Path(contract["prepared_directory"])
    s.verify(prepared)
    assert s.file_hash(prepared / "artifact_hashes.json") == contract["prepared_manifest_sha256"]
    candidates = s.read(prepared / "candidates.json")
    lookup = {r["candidate_id"]: r for r in [contract["health"]] + candidates}
    calls = [__import__("json").loads(line) for line in (board_dir / "results.jsonl").read_text().splitlines()]
    phases = Counter(r["phase"] for r in calls)
    for call in calls:
        assert call["correct"] and call["mismatch_count"] == 0
        assert call["actual_sha256"] == call["expected_sha256"]
        assert call["complete_operator_calls"] == 1 and call["hidden_warmup_calls"] == 0
        profile = call["runtime_profile_complete"]
        static = lookup[call["candidate_id"]]["local_qualification"]
        for key, value in static["semantics"]["totals"].items():
            if key in profile:
                assert profile[key] == value, (call["candidate_id"], key, profile[key], value)
        assert profile["driver_timeout_calls"] == 0
        assert profile["driver_run_calls"] == sum(static["sync_multiplicity"])
    assert phases["health"] == 3
    qualified = [r for r in candidates if r.get("cross_qualification", {}).get("status") == "passed"]
    assert phases["correctness"] == 3 * len(qualified)
    assert phases["warmup"] == len(qualified)
    assert phases["timing"] == summary["timing_calls"] == 3 * 10 * (len(qualified) - 1)
    for row in qualified:
        seeds = [r["seed"] for r in calls if r["phase"] == "correctness" and r["candidate_id"] == row["candidate_id"]]
        assert seeds == list(s.SEEDS)
    pairs = s.read(board_dir / "paired_observations.json")
    assert len(pairs) * 3 == phases["timing"]
    baseline_id = contract["baseline_candidate_id"]
    baseline_calls = [r for r in calls if r["phase"] == "timing" and r["candidate_id"] == baseline_id]
    # Rebuild A-X-A observations directly from the raw dispatch log.
    timing = [r for r in calls if r["phase"] == "timing"]
    for index, pair in enumerate(pairs):
        a, x, b = timing[3*index:3*index+3]
        assert a["candidate_id"] == b["candidate_id"] == baseline_id
        assert x["candidate_id"] == pair["candidate_id"]
        assert a["seed"] == x["seed"] == b["seed"]
        assert a["round"] == x["round"] == b["round"] == pair["round"]
        assert pair["baseline_ms"] == (a["latency_ms"] + b["latency_ms"])/2
        assert pair["candidate_ms"] == x["latency_ms"]
    mode_best = {}
    for comparison in summary["comparisons"]:
        selected = [p for p in pairs if p["candidate_id"] == comparison["candidate_id"]]
        assert len(selected) == 10
        assert comparison["median_ms"] == statistics.median(p["candidate_ms"] for p in selected)
        row = lookup[comparison["candidate_id"]]
        enriched = {**comparison, "logical_dma": row["local_qualification"]["semantics"]["totals"],
                    "sync_multiplicity": row["local_qualification"]["sync_multiplicity"]}
        mode_best.setdefault(comparison["residence_mode"], enriched)
    baseline = lookup[baseline_id]
    top_i = next(r for r in summary["comparisons"] if r["family_id"] == "TOPHUB_TILE" and r["residence_mode"] == "input_stationary")
    same_o = next(r for r in qualified if r["family_id"] == "TOPHUB_TILE" and r["implementation_mode"] == 0)
    assert same_o["local_qualification"]["tir_sha256"] == baseline["local_qualification"]["tir_sha256"]
    assert contract["official_record"]["input"][2] == contract["workload"][1:]
    assert contract["timer_module_retained_for_whole_session"]
    result = dict(status="PRIMARY_EVIDENCE_AUDIT_PASS", total_calls=len(calls), phases=dict(phases),
        all_calls_correct=True, all_calls_match_static_dma=True, all_calls_match_static_submissions=True,
        TopHub_original_median_ms=statistics.median(r["latency_ms"] for r in baseline_calls),
        TopHub_original_min_ms=min(r["latency_ms"] for r in baseline_calls),
        TopHub_original_max_ms=max(r["latency_ms"] for r in baseline_calls),
        same_tophub_tile_input=top_i, best_per_mode=mode_best,
        gross_local_candidates=len(candidates), qualified_candidates=len(qualified),
        distinct_qualified_tir=len({r["local_qualification"]["tir_sha256"] for r in qualified}),
        lower_rejects=[dict(family_id=r["family_id"], mode=r["residence_mode"],
                          failure=r["local_qualification"].get("failure"))
                       for r in candidates if r["local_qualification"]["status"] != "passed"],
        boot_id=contract["boot_id"], reference_kind=contract["reference_kind"],
        whole_graph_tested=False, no_exhaustive_optimality_claim=True, wall_seconds=time.monotonic()-started)
    s.write(out / "contract.json", dict(phase="tophub_primary_evidence_audit", board_directory=str(board_dir),
            board_manifest_sha256=s.file_hash(board_dir / "artifact_hashes.json"),
            source_bindings={str(Path(__file__).resolve()):s.file_hash(__file__)}))
    s.write(out / "summary.json", result)
    s.seal(out)
    print(__import__("json").dumps({k:result[k] for k in ("status", "total_calls", "phases", "TopHub_original_median_ms", "same_tophub_tile_input")}, indent=2))
    print("BEST_PER_MODE")
    for mode, row in mode_best.items():
        print(mode, row["family_id"], row["median_ms"], row["latency_reduction_percent"], row["wins"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    audit(Path(args.board_dir).resolve(), output)
