#!/usr/bin/env python3
"""Build the P5c label-free, command-aware shortlist over the unified VTA pool.

Qualification records decide pool membership only.  Ranking reads a fixed whitelist
of static DMA and timing-free structural command fields; it never reads latency,
correctness, cost, FPS, or other post-measurement outcomes as ordering features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path

from c3_candidate_identity import canonical_json_bytes


SCHEMA = "c3_p5c_full_pool_command_shortlist_v1"
PLANNER_VERSION = "c3_p5c_command_pareto_stratified_v1"
FEATURE_VERSION = "c3_static_dma_fsim_command_shape_v1"
DEFAULT_BUDGETS = (4, 8, 16)
EXPECTED_BASE = 165
EXPECTED_BARRIER = 32
EXPECTED_TOTAL = EXPECTED_BASE + EXPECTED_BARRIER

# Every objective is minimized.  The list and equal-rank policy are frozen before
# reading P4j.  Mode is deliberately absent: it is used only to interleave peers
# within the same Pareto front for exploration coverage.
PARETO_OBJECTIVES = (
    "dma_total_bytes",
    "dma_total_calls",
    "input_dma_bytes",
    "input_dma_calls",
    "weight_dma_bytes",
    "weight_dma_calls",
    "output_dma_bytes",
    "output_dma_calls",
    "input_reload",
    "weight_reload",
    "output_reload",
    "small_dma_calls",
    "strided_dma_calls",
    "padded_dma_calls",
    "submissions",
    "total_insn_bytes",
    "peak_insn_bytes",
    "total_uop_bytes",
    "peak_uop_bytes",
    "source_derived_finish_count",
)

# FINISH is still an audited Pareto field, but source derivation makes it an exact
# alias of submissions.  Exclude that one alias from the within-front rank sum so
# the tie-break does not silently double-weight submission count.
RANK_SUM_OBJECTIVES = tuple(
    name for name in PARETO_OBJECTIVES if name != "source_derived_finish_count"
)

FORBIDDEN_RANK_KEYS = frozenset(
    {
        "latency",
        "latency_ms",
        "correct",
        "correctness",
        "cost",
        "costs",
        "costs_s",
        "fps",
        "runtime_ms",
        "time_ms",
        "measurement_result",
        "device_run_us",
        "poll_wait_us",
        "submit_mmio_us",
        "finalize_us",
    }
)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def payload_sha256(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def unique(rows, source):
    result = {}
    for row in rows:
        candidate_id = row["candidate_id"]
        if candidate_id in result:
            raise ValueError("duplicate candidate ID in {}: {}".format(source, candidate_id))
        result[candidate_id] = row
    return result


def artifact_entries(ledger):
    if "output_sha256" in ledger:
        return ledger["output_sha256"]
    if "artifacts" in ledger:
        return ledger["artifacts"]
    if "files" in ledger:
        # P5b's older ledger also lists external source/input paths.  Verify all
        # run-local basename entries here; their input hashes remain in its manifest.
        return {name: digest for name, digest in ledger["files"].items() if "/" not in name}
    return {
        name: digest
        for name, digest in ledger.items()
        if isinstance(name, str) and isinstance(digest, str) and len(digest) == 64
    }


def verify_run_artifacts(run_dir):
    run_dir = Path(run_dir).resolve()
    ledger_path = run_dir / "artifact_hashes.json"
    ledger = load_json(ledger_path)
    expected = artifact_entries(ledger)
    if not expected:
        raise ValueError("artifact ledger has no output hashes: {}".format(ledger_path))
    verified = {}
    for name, wanted in expected.items():
        path = run_dir / name
        if not path.is_file():
            raise ValueError("missing frozen artifact: {}".format(path))
        observed = sha256_file(path)
        if observed != wanted:
            raise ValueError("frozen artifact hash mismatch: {}".format(path))
        verified[name] = observed
    return {
        "run_dir": str(run_dir),
        "artifact_hashes_sha256": sha256_file(ledger_path),
        "verified_files": len(verified),
    }


def join_qualified_pool(p4b_rows, p4e_rows, p4f_rows, p4g_rows, p4i_rows):
    """Join qualification certificates without exposing outcomes to the ranker."""

    p4b = unique(p4b_rows, "P4b")
    p4e = unique(p4e_rows, "P4e")
    p4f = unique(p4f_rows, "P4f")
    p4g = unique(p4g_rows, "P4g")
    p4i = unique(p4i_rows, "P4i")

    base_ids = {
        candidate_id
        for candidate_id, row in p4e.items()
        if row.get("local_status") == "fsim_passed"
    }
    exported_base_ids = {
        candidate_id
        for candidate_id, row in p4f.items()
        if row.get("overall_status") == "passed"
    }
    if base_ids != exported_base_ids or len(base_ids) != EXPECTED_BASE:
        raise ValueError(
            "P4e/P4f qualified base-set mismatch: {}/{}".format(
                len(base_ids), len(exported_base_ids)
            )
        )

    barrier_ids = {
        candidate_id
        for candidate_id, row in p4g.items()
        if row.get("status") == "ok"
        and (row.get("dependency_audit") or {}).get("valid") is True
        and (row.get("fsim_qualification") or {}).get("overall_status") == "passed"
    }
    exported_barrier_ids = {
        candidate_id
        for candidate_id, row in p4i.items()
        if row.get("overall_status") == "passed"
    }
    if barrier_ids != exported_barrier_ids or len(barrier_ids) != EXPECTED_BARRIER:
        raise ValueError(
            "P4g/P4i qualified barrier-set mismatch: {}/{}".format(
                len(barrier_ids), len(exported_barrier_ids)
            )
        )
    if base_ids & barrier_ids:
        raise ValueError("base and weight-barrier candidate IDs overlap")

    joined = {}
    for candidate_id in sorted(base_ids):
        static = p4b.get(candidate_id)
        if static is None or static.get("status") != "ok":
            raise ValueError("qualified base candidate lacks P4b static record: {}".format(candidate_id))
        if not (
            static.get("tir_sha256") == p4e[candidate_id].get("tir_sha256")
            == p4f[candidate_id].get("tir_sha256")
        ):
            raise ValueError("base candidate TIR certificate mismatch: {}".format(candidate_id))
        joined[candidate_id] = static
    for candidate_id in sorted(barrier_ids):
        static = p4g[candidate_id]
        if static.get("tir_sha256") != p4i[candidate_id].get("p4g_tir_sha256"):
            raise ValueError("weight-barrier TIR certificate mismatch: {}".format(candidate_id))
        joined[candidate_id] = static

    if len(joined) != EXPECTED_TOTAL:
        raise AssertionError("unified pool is not exactly 197 candidates")
    incumbents = [
        row for row in joined.values() if row.get("candidate_role") == "protected_original_incumbent"
    ]
    if len(incumbents) != 10 or len({row["workload_id"] for row in incumbents}) != 10:
        raise ValueError("unified pool must contain one protected incumbent for each workload")
    return joined


def structural_command(record):
    signature = record.get("command_signature")
    if not isinstance(signature, dict):
        raise ValueError("P4j record lacks command_signature: {}".format(record.get("candidate_id")))
    structural = signature.get("structural", signature)
    if not isinstance(structural, dict):
        raise ValueError("P4j structural command signature is invalid")
    if "sha256" in signature and signature["sha256"] != payload_sha256(structural):
        raise ValueError("P4j command signature hash mismatch: {}".format(record["candidate_id"]))
    return structural


def join_p4j_commands(static_by_id, p4j_rows):
    """Attach only structural command data; do not inspect correctness or timing labels."""

    p4j = unique(p4j_rows, "P4j")
    expected_ids = set(static_by_id)
    if set(p4j) != expected_ids:
        raise ValueError(
            "P4j candidate set incomplete: missing={} extra={}".format(
                len(expected_ids - set(p4j)), len(set(p4j) - expected_ids)
            )
        )
    joined = []
    for candidate_id in sorted(expected_ids):
        static = static_by_id[candidate_id]
        command_record = p4j[candidate_id]
        if command_record.get("schema") != "c3_full_pool_fsim_command_signature_v1":
            raise ValueError("unexpected P4j result schema: {}".format(candidate_id))
        if command_record.get("execution_transport") != "direct_local_module_no_rpc":
            raise ValueError("P4j candidate was not executed by the no-RPC path: {}".format(candidate_id))
        if command_record.get("relowered_tir_sha256") not in (None, static.get("tir_sha256")):
            raise ValueError("P4j/static TIR mismatch: {}".format(candidate_id))
        structural = structural_command(command_record)
        finish = structural.get("finish") or {}
        if finish.get("observed_in_queue_json") is not False:
            raise ValueError("FINISH must be explicitly source-derived: {}".format(candidate_id))
        if int(finish.get("source_derived_count", -1)) != int(structural.get("submissions", -2)):
            raise ValueError("source-derived FINISH/submission mismatch: {}".format(candidate_id))
        joined.append((static, structural))
    return joined


def assert_rank_payload_label_free(value, location="$"):
    if isinstance(value, dict):
        forbidden = sorted(set(value) & FORBIDDEN_RANK_KEYS)
        if forbidden:
            raise ValueError("forbidden rank label at {}: {}".format(location, forbidden))
        for key, child in value.items():
            assert_rank_payload_label_free(child, "{}.{}".format(location, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_rank_payload_label_free(child, "{}[{}]".format(location, index))


def extract_feature(static, structural):
    """Whitelisted static feature extraction; no outcome field is accepted here."""

    signature = static["transfer_signature"]
    aggregates = signature["descriptor_aggregates"]
    by_memory = aggregates["by_memory"]
    totals = signature["totals"]
    reload_ratio = signature["reload_ratio"]
    command_totals = structural["totals"]
    command_peaks = structural["peaks"]
    metrics = {
        "dma_total_bytes": int(aggregates["expanded_bytes"]),
        "dma_total_calls": int(aggregates["expanded_calls"]),
        "input_dma_bytes": int(by_memory["inp"]["bytes"]),
        "input_dma_calls": int(by_memory["inp"]["calls"]),
        "weight_dma_bytes": int(by_memory["wgt"]["bytes"]),
        "weight_dma_calls": int(by_memory["wgt"]["calls"]),
        "output_dma_bytes": int(by_memory["out"]["bytes"]),
        "output_dma_calls": int(by_memory["out"]["calls"]),
        "input_reload": float(reload_ratio["input_load"]),
        "weight_reload": float(reload_ratio["weight_load"]),
        "output_reload": float(reload_ratio["output_store"]),
        "small_dma_calls": int(totals["load_buffer_2d_small_calls"])
        + int(totals["store_buffer_2d_small_calls"]),
        "strided_dma_calls": int(totals["load_buffer_2d_strided_calls"])
        + int(totals["store_buffer_2d_strided_calls"]),
        "padded_dma_calls": int(aggregates["padded_calls"].get("load", 0))
        + int(aggregates["padded_calls"].get("store", 0)),
        "submissions": int(structural["submissions"]),
        "total_insn_bytes": int(command_totals["insn_bytes"]),
        "peak_insn_bytes": int(command_peaks["insn_bytes"]),
        "total_uop_bytes": int(command_totals["uop_bytes"]),
        "peak_uop_bytes": int(command_peaks["uop_bytes"]),
        "source_derived_finish_count": int(structural["finish"]["source_derived_count"]),
    }
    if set(metrics) != set(PARETO_OBJECTIVES):
        raise AssertionError("rank feature contract drift")
    payload = {
        "feature_version": FEATURE_VERSION,
        "candidate_id": static["candidate_id"],
        "workload_id": static["workload_id"],
        "candidate_role": static["candidate_role"],
        "residence_mode": static["residence_mode"],
        "config_index_debug_only": int(static["debug"]["config_index"]),
        "metrics": metrics,
    }
    assert_rank_payload_label_free(payload)
    payload["feature_sha256"] = payload_sha256(payload)
    return payload


def dominates(left, right):
    lhs = tuple(left["metrics"][name] for name in PARETO_OBJECTIVES)
    rhs = tuple(right["metrics"][name] for name in PARETO_OBJECTIVES)
    return all(a <= b for a, b in zip(lhs, rhs)) and any(a < b for a, b in zip(lhs, rhs))


def pareto_fronts(candidates):
    remaining = list(candidates)
    fronts = []
    while remaining:
        front = [
            candidate
            for candidate in remaining
            if not any(
                other["candidate_id"] != candidate["candidate_id"]
                and dominates(other, candidate)
                for other in remaining
            )
        ]
        if not front:
            raise RuntimeError("Pareto decomposition made no progress")
        rank_sum = {candidate["candidate_id"]: 0 for candidate in front}
        for objective in RANK_SUM_OBJECTIVES:
            distinct = sorted({candidate["metrics"][objective] for candidate in front})
            ranks = {value: index for index, value in enumerate(distinct)}
            for candidate in front:
                rank_sum[candidate["candidate_id"]] += ranks[candidate["metrics"][objective]]
        prepared = []
        for candidate in front:
            item = dict(candidate)
            item["pareto_front"] = len(fronts)
            item["equal_objective_rank_sum"] = rank_sum[candidate["candidate_id"]]
            prepared.append(item)
        fronts.append(prepared)
        selected = {candidate["candidate_id"] for candidate in front}
        remaining = [candidate for candidate in remaining if candidate["candidate_id"] not in selected]
    return fronts


def stratify_front(front):
    """Round-robin modes inside one front; never cross a Pareto-front boundary."""

    groups = {}
    for candidate in front:
        groups.setdefault(candidate["residence_mode"], []).append(candidate)
    for group in groups.values():
        group.sort(key=lambda item: (item["equal_objective_rank_sum"], item["candidate_id"]))
    modes = sorted(
        groups,
        key=lambda mode: (
            groups[mode][0]["equal_objective_rank_sum"],
            groups[mode][0]["candidate_id"],
            mode,
        ),
    )
    ordered = []
    offset = 0
    while True:
        emitted = False
        for mode in modes:
            if offset < len(groups[mode]):
                ordered.append(groups[mode][offset])
                emitted = True
        if not emitted:
            return ordered
        offset += 1


def rank_workload(features, incumbent_id):
    incumbents = [feature for feature in features if feature["candidate_id"] == incumbent_id]
    if len(incumbents) != 1 or incumbents[0]["candidate_role"] != "protected_original_incumbent":
        raise ValueError("protected incumbent missing or malformed: {}".format(incumbent_id))
    ordered = []
    for front in pareto_fronts(features):
        ordered.extend(stratify_front(front))
    ranked_incumbent = next(item for item in ordered if item["candidate_id"] == incumbent_id)
    ordered = [ranked_incumbent] + [
        item for item in ordered if item["candidate_id"] != incumbent_id
    ]
    for position, item in enumerate(ordered, 1):
        item["ranking_position"] = position
        item["protected_first"] = position == 1
    return ordered


def budget_record(ordered, budget):
    prefix = ordered[: min(int(budget), len(ordered))]
    return {
        "requested_budget": int(budget),
        "effective_budget": len(prefix),
        "candidate_ids": [item["candidate_id"] for item in prefix],
        "mode_counts": dict(sorted(Counter(item["residence_mode"] for item in prefix).items())),
    }


def build_shortlist(joined, budgets=DEFAULT_BUDGETS):
    features = [extract_feature(static, structural) for static, structural in joined]
    if len(features) != EXPECTED_TOTAL or len({item["candidate_id"] for item in features}) != EXPECTED_TOTAL:
        raise ValueError("feature pool is not exactly 197 unique candidates")
    by_workload = {}
    for feature in features:
        by_workload.setdefault(feature["workload_id"], []).append(feature)
    if len(by_workload) != 10:
        raise ValueError("feature pool must contain exactly ten workloads")
    output, ranked_rows = {}, []
    for workload_id in sorted(by_workload):
        rows = by_workload[workload_id]
        incumbents = [row for row in rows if row["candidate_role"] == "protected_original_incumbent"]
        if len(incumbents) != 1:
            raise ValueError("expected one incumbent for {}".format(workload_id))
        incumbent_id = incumbents[0]["candidate_id"]
        ordered = rank_workload(rows, incumbent_id)
        ranked_rows.extend(ordered)
        output[workload_id] = {
            "protected_original_incumbent": incumbent_id,
            "eligible_candidates": len(rows),
            "ranking_candidate_ids": [item["candidate_id"] for item in ordered],
            "budgets": {str(budget): budget_record(ordered, budget) for budget in budgets},
        }
    feature_set = sorted(
        ({"candidate_id": item["candidate_id"], "feature_sha256": item["feature_sha256"]} for item in features),
        key=lambda item: item["candidate_id"],
    )
    shortlist = {
        "schema": SCHEMA,
        "planner_version": PLANNER_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_set_sha256": payload_sha256(feature_set),
        "budgets": list(budgets),
        "candidate_count": len(features),
        "label_policy": {
            "qualification_outcomes_used_for_membership_only": True,
            "latency_used_for_ranking": False,
            "correctness_used_for_ranking": False,
            "learned_model_used": False,
            "feature_whitelist_only": True,
        },
        "rule": {
            "incumbent": "protected incumbent is always position one",
            "objectives": list(PARETO_OBJECTIVES),
            "objective_direction": "all minimized",
            "fronts": "standard nondominated Pareto fronts",
            "within_front_score": "equal-weight unit-free distinct-value rank sum",
            "finish_tie_policy": "FINISH remains a Pareto field but is not double-counted in rank sum because it equals submissions by source derivation",
            "mode_exploration": "round-robin by residence_mode only within one Pareto front",
            "final_tie_break": "candidate_id",
        },
        "workloads": output,
    }
    ranked_rows.sort(key=lambda row: (row["workload_id"], row["ranking_position"]))
    return shortlist, ranked_rows


def compare_to_p5b(shortlist, p5b, budgets=DEFAULT_BUDGETS):
    comparisons = {}
    old_pool, new_pool = set(), set()
    for workload_id in sorted(shortlist["workloads"]):
        new_workload = shortlist["workloads"][workload_id]
        old_workload = p5b["workloads"][workload_id]
        old_ranking = old_workload["strategies"]["B7"]["ranking_candidate_ids"]
        new_ranking = new_workload["ranking_candidate_ids"]
        old_pool.update(old_ranking)
        new_pool.update(new_ranking)
        budget_rows = {}
        for budget in budgets:
            old_ids = old_workload["strategies"]["B7"]["budgets"][str(budget)]["candidate_ids"]
            new_ids = new_workload["budgets"][str(budget)]["candidate_ids"]
            budget_rows[str(budget)] = {
                "p5b_b7_candidate_ids": old_ids,
                "p5c_candidate_ids": new_ids,
                "same_prefix": old_ids == new_ids,
                "overlap_count": len(set(old_ids) & set(new_ids)),
                "added_in_p5c": [candidate_id for candidate_id in new_ids if candidate_id not in old_ids],
                "removed_from_p5b_prefix": [candidate_id for candidate_id in old_ids if candidate_id not in new_ids],
            }
        comparisons[workload_id] = {
            "p5b_eligible": len(old_ranking),
            "p5c_eligible": len(new_ranking),
            "candidate_space_delta": len(new_ranking) - len(old_ranking),
            "budgets": budget_rows,
        }
    return {
        "schema": "c3_p5c_comparison_to_p5b_v1",
        "comparison_baseline": "P5b B7 label-free mode-diverse static shortlist",
        "p5b_candidate_space": len(old_pool),
        "p5c_candidate_space": len(new_pool),
        "shared_candidates": len(old_pool & new_pool),
        "added_candidates": len(new_pool - old_pool),
        "removed_candidates": len(old_pool - new_pool),
        "workloads": comparisons,
        "claim_boundary": "prefix change only; no search-efficiency or performance conclusion",
    }


def summarize(shortlist, ranked_rows, comparison):
    budget_summary = {}
    for budget in shortlist["budgets"]:
        rows = [workload["budgets"][str(budget)] for workload in shortlist["workloads"].values()]
        modes = Counter()
        for row in rows:
            modes.update(row["mode_counts"])
        budget_summary[str(budget)] = {
            "workloads": len(rows),
            "selected_total": sum(row["effective_budget"] for row in rows),
            "incumbent_first": all(
                row["candidate_ids"][0]
                == shortlist["workloads"][workload_id]["protected_original_incumbent"]
                for workload_id, row in zip(sorted(shortlist["workloads"]), rows)
            ),
            "prefixes_changed_vs_p5b_b7": sum(
                not item["budgets"][str(budget)]["same_prefix"]
                for item in comparison["workloads"].values()
            ),
            "mode_counts": dict(sorted(modes.items())),
            "workloads_with_weight_barrier_exploration": sum(
                row["mode_counts"].get("weight_stationary_barrier", 0) > 0 for row in rows
            ),
        }
    return {
        "schema": "c3_p5c_full_pool_command_shortlist_summary_v1",
        "status": "completed",
        "qualified_candidates": len(ranked_rows),
        "workloads": len(shortlist["workloads"]),
        "mode_counts": dict(sorted(Counter(row["residence_mode"] for row in ranked_rows).items())),
        "budget_summary": budget_summary,
        "p5b_comparison": {
            key: comparison[key]
            for key in (
                "p5b_candidate_space",
                "p5c_candidate_space",
                "shared_candidates",
                "added_candidates",
                "removed_candidates",
            )
        },
        "label_free_ranking": True,
        "board_access": False,
        "performance_measurement": "not_collected",
        "search_efficiency_claim": False,
        "g5_board_labels_required": True,
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def default_paths():
    base = Path(__file__).resolve().parent / "report_out/stage_tile_cotuning/c3_dma_residency_autotune"
    p4 = base / "04_dma_command_signatures"
    return {
        "p4b": p4 / "20260910_p4b_local_residency_pool_run01/results.jsonl",
        "p4e": p4 / "20260911_p4e_unified_local_qualification_run01/results.jsonl",
        "p4f": p4 / "20260911_p4f_axu_cross_compile_run01/results.jsonl",
        "p4g": p4 / "20260911_p4g_weight_barrier_pool_run01/results.jsonl",
        "p4i": p4 / "20260911_p4i_weight_barrier_cross_compile_run01/results.jsonl",
        # run01 is retained only as a superseded protocol-deviation audit because
        # it used rpc.LocalSession.  run02 is the frozen direct-local, no-RPC input.
        "p4j": p4 / "20260911_p4j_full_pool_fsim_command_run02/results.jsonl",
        "p5b": base / "05_grouped_replay/20260910_p5b_local_predispatch_shortlist_run01/shortlist.json",
    }


def main():
    defaults = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in defaults.items():
        parser.add_argument("--" + name, default=str(default))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", default="4,8,16")
    args = parser.parse_args()
    paths = {name: Path(getattr(args, name)).resolve() for name in defaults}
    budgets = tuple(sorted({int(value) for value in args.budgets.split(",") if value}))
    if budgets != DEFAULT_BUDGETS:
        raise ValueError("P5c frozen budgets must be exactly 4,8,16")

    evidence = {}
    for name, path in paths.items():
        evidence[name] = verify_run_artifacts(path.parent)
        evidence[name].update(path=str(path), sha256=sha256_file(path))
    static = join_qualified_pool(
        load_jsonl(paths["p4b"]),
        load_jsonl(paths["p4e"]),
        load_jsonl(paths["p4f"]),
        load_jsonl(paths["p4g"]),
        load_jsonl(paths["p4i"]),
    )
    joined = join_p4j_commands(static, load_jsonl(paths["p4j"]))
    shortlist, ranked_rows = build_shortlist(joined, budgets)
    comparison = compare_to_p5b(shortlist, load_json(paths["p5b"]), budgets)
    summary = summarize(shortlist, ranked_rows, comparison)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "shortlist.json", shortlist)
    (output / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ranked_rows),
        encoding="utf-8",
    )
    write_json(output / "comparison_to_p5b.json", comparison)
    write_json(output / "summary.json", summary)
    write_json(
        output / "preregistered.json",
        {
            "schema": "c3_p5c_preregistered_protocol_v1",
            "planner_version": PLANNER_VERSION,
            "population": "exact 165 P4e/P4f-qualified base candidates plus exact 32 P4g/P4i-qualified mode4 candidates",
            "required_p4j_coverage": 197,
            "budgets": list(DEFAULT_BUDGETS),
            "incumbent_rule": "always first",
            "pareto_objectives": list(PARETO_OBJECTIVES),
            "objective_direction": "all minimized",
            "tie_rule": "equal-weight distinct-value rank sum then candidate_id",
            "finish_tie_policy": "source-derived FINISH is Pareto-audited but not double-counted against its exact submissions alias",
            "mode_rule": "round-robin exploration only within the same Pareto front",
            "p4j_protocol": "consume only direct-local no-RPC run02; run01 is superseded",
            "feature_firewall": "only enumerated static DMA and timing-free command fields enter ranking",
            "forbidden_rank_labels": sorted(FORBIDDEN_RANK_KEYS),
            "learned_model": False,
            "claim_boundary": "no search-efficiency or performance claim before G5 board labels",
        },
    )
    write_json(
        output / "manifest.json",
        {
            "schema": "c3_p5c_full_pool_command_shortlist_manifest_v1",
            "date": "2026-09-11",
            "environment": {
                "python": sys.executable,
                "python_version": platform.python_version(),
                "ssh_used": False,
                "rpc_used": False,
                "board_contacted": False,
            },
            "inputs": evidence,
            "superseded_input": {
                "path": str(paths["p4j"].parent.parent / "20260911_p4j_full_pool_fsim_command_run01"),
                "consumed": False,
                "reason": "used rpc.LocalSession and violates the strict no-RPC P5c protocol",
            },
            "planner": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(__file__)},
            "feature_set_sha256": shortlist["feature_set_sha256"],
            "results": summary,
        },
    )
    (output / "command.txt").write_text(
        "PYTHONPATH={} {}\n".format(os.environ.get("PYTHONPATH", ""), " ".join([sys.executable] + sys.argv))
    )
    (output / "stdout.log").write_text(
        "status=completed candidates=197 workloads=10 budgets=4,8,16 board=false labels=false\n"
    )
    (output / "stderr.log").write_text("")
    (output / "STATUS.md").write_text(
        "# C3-P5c full-pool command-aware shortlist\n\n"
        "Status: `completed`. The label-free unified pool expands P5b's 165 candidates to "
        "197 by adding 32 mode-4 candidates; all ten incumbents are first in the budget "
        "4/8/16 prefixes. All three budget prefixes change for all ten workloads. Ranking "
        "uses only frozen static DMA and timing-free command structure. No SSH, RPC, board, "
        "latency label, correctness label, learned model, or performance claim was used.\n",
        encoding="utf-8",
    )
    (output / "HANDOFF.md").write_text(
        "# HANDOFF\n\n"
        "Use `shortlist.json` as the prospective G5 dispatch input. `results.jsonl` records "
        "the complete 197-candidate rank/features, and `comparison_to_p5b.json` records only "
        "candidate-space and prefix changes versus P5b B7. These changes do not demonstrate "
        "search efficiency or speed; G5 board labels are required for both conclusions. P4j "
        "run02 is the consumed direct-local no-RPC evidence; run01 is explicitly superseded.\n",
        encoding="utf-8",
    )
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    test_path = Path(__file__).with_name("test_plan_vta_full_pool_command_shortlist.py")
    sources = {str(Path(__file__).resolve()): sha256_file(__file__)}
    if test_path.is_file():
        sources[str(test_path.resolve())] = sha256_file(test_path)
    write_json(output / "artifact_hashes.json", {"output_sha256": artifacts, "source_sha256": sources})
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
