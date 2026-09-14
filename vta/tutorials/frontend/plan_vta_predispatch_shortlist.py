"""Freeze label-free VTA board-dispatch shortlists from P4b static features."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


SCHEMA = "c3_predispatch_shortlist_v1"
FEATURE_VERSION = "c3_static_request_shape_v1"
PLANNER_VERSION = "c3_p5b_shortlist_planner_v1"
STRATEGIES = ("B4", "B5", "B6", "B7")
DEFAULT_BUDGETS = (4, 8, 16)
MODE_ORDER = ("original", "input_stationary", "weight_stationary", "paper_inspired_hybrid")
EXPERIMENTAL_MODES = MODE_ORDER[1:]
FORBIDDEN_LABEL_KEYS = frozenset(
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
    }
)

# All objectives are minimized. Negative request-size objectives therefore favor
# larger contiguous transfers while calls/bytes/small/strided/padded are minimized.
PARETO_OBJECTIVES = (
    "total_dma_bytes",
    "total_dma_calls",
    "load_calls",
    "store_calls",
    "small_calls",
    "strided_calls",
    "padded_calls",
    "input_reload",
    "weight_reload",
    "output_reload",
    "negative_max_input_request_bytes",
    "negative_max_weight_request_bytes",
    "negative_max_output_request_bytes",
    "histogram_distinct_sizes",
    "negative_histogram_min_request_bytes",
)


def canonical_json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def payload_sha256(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _assert_label_free(value, location="$"):
    if isinstance(value, dict):
        forbidden = sorted(set(value).intersection(FORBIDDEN_LABEL_KEYS))
        if forbidden:
            raise ValueError("forbidden post-dispatch label(s) at {}: {}".format(location, forbidden))
        for key, child in value.items():
            _assert_label_free(child, "{}.{}".format(location, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_label_free(child, "{}[{}]".format(location, index))


def _pool_candidate_ids(pool):
    ids = set()
    protected = {}
    for workload in pool["workloads"]:
        workload_id = workload["workload_id"]
        incumbent = workload["protected_original_incumbent"]["candidate_id"]
        if workload_id in protected or incumbent in ids:
            raise ValueError("duplicate workload or protected candidate: {}".format(workload_id))
        protected[workload_id] = incumbent
        ids.add(incumbent)
        for mapping in workload["mapping"]["records"]:
            for candidate_id in mapping["candidate_ids"].values():
                if candidate_id in ids:
                    raise ValueError("duplicate candidate_id in candidate pool: {}".format(candidate_id))
                ids.add(candidate_id)
    return ids, protected


def _histogram_metrics(histogram):
    rows = []
    normalized = {}
    for direction in sorted(histogram):
        normalized[direction] = {}
        for request_bytes, count in sorted(
            histogram[direction].items(), key=lambda item: int(item[0])
        ):
            request_bytes = int(request_bytes)
            count = int(count)
            normalized[direction][str(request_bytes)] = count
            rows.append((request_bytes, count))
    if not rows:
        raise ValueError("successful candidate has an empty request histogram")
    return {
        "exact": normalized,
        "distinct_sizes": len({request_bytes for request_bytes, _ in rows}),
        "min_request_bytes": min(request_bytes for request_bytes, _ in rows),
        "expanded_calls": sum(count for _, count in rows),
        "expanded_bytes": sum(request_bytes * count for request_bytes, count in rows),
    }


def extract_prefeature(record):
    """Extract only static, pre-dispatch request features from one successful record."""
    if record.get("status") != "ok" or record.get("failure") is not None:
        raise ValueError("prefeatures require a successful lower record")
    signature = record.get("transfer_signature")
    if not signature:
        raise ValueError("successful candidate lacks transfer_signature")
    totals = signature["totals"]
    aggregates = signature["descriptor_aggregates"]
    reload_ratio = signature["reload_ratio"]
    maximum = signature["max_request_bytes_by_memory"]
    histogram = _histogram_metrics(aggregates["exact_request_bytes_histogram"])
    required_totals = (
        "load_buffer_2d_bytes",
        "load_buffer_2d_calls",
        "load_buffer_2d_small_calls",
        "load_buffer_2d_strided_calls",
        "store_buffer_2d_bytes",
        "store_buffer_2d_calls",
        "store_buffer_2d_small_calls",
        "store_buffer_2d_strided_calls",
    )
    missing = [key for key in required_totals if key not in totals]
    if missing:
        raise ValueError("successful candidate lacks static total(s): {}".format(missing))
    padded = aggregates["padded_calls"]
    metrics = {
        "total_dma_bytes": int(aggregates["expanded_bytes"]),
        "total_dma_calls": int(aggregates["expanded_calls"]),
        "load_bytes": int(totals["load_buffer_2d_bytes"]),
        "store_bytes": int(totals["store_buffer_2d_bytes"]),
        "load_calls": int(totals["load_buffer_2d_calls"]),
        "store_calls": int(totals["store_buffer_2d_calls"]),
        "small_calls": int(totals["load_buffer_2d_small_calls"])
        + int(totals["store_buffer_2d_small_calls"]),
        "strided_calls": int(totals["load_buffer_2d_strided_calls"])
        + int(totals["store_buffer_2d_strided_calls"]),
        "padded_calls": int(padded.get("load", 0)) + int(padded.get("store", 0)),
        "input_reload": float(reload_ratio["input_load"]),
        "weight_reload": float(reload_ratio["weight_load"]),
        "output_reload": float(reload_ratio["output_store"]),
        "max_input_request_bytes": int(maximum.get("inp", 0)),
        "max_weight_request_bytes": int(maximum.get("wgt", 0)),
        "max_output_request_bytes": int(maximum.get("out", 0)),
        "histogram_distinct_sizes": histogram["distinct_sizes"],
        "histogram_min_request_bytes": histogram["min_request_bytes"],
    }
    metrics.update(
        {
            "negative_max_input_request_bytes": -metrics["max_input_request_bytes"],
            "negative_max_weight_request_bytes": -metrics["max_weight_request_bytes"],
            "negative_max_output_request_bytes": -metrics["max_output_request_bytes"],
            "negative_histogram_min_request_bytes": -metrics["histogram_min_request_bytes"],
        }
    )
    payload = {
        "feature_version": FEATURE_VERSION,
        "candidate_id": record["candidate_id"],
        "metrics": metrics,
        "request_histogram": histogram,
    }
    return dict(payload, feature_sha256=payload_sha256(payload))


def _dominates(left, right):
    values_left = tuple(left["prefeature"]["metrics"][name] for name in PARETO_OBJECTIVES)
    values_right = tuple(right["prefeature"]["metrics"][name] for name in PARETO_OBJECTIVES)
    return all(a <= b for a, b in zip(values_left, values_right)) and any(
        a < b for a, b in zip(values_left, values_right)
    )


def pareto_order(candidates):
    """Return deterministic nondominated-front order with a unit-free rank-sum tie break."""
    remaining = list(candidates)
    ordered = []
    front_index = 0
    while remaining:
        front = [
            candidate
            for candidate in remaining
            if not any(
                other["candidate_id"] != candidate["candidate_id"]
                and _dominates(other, candidate)
                for other in remaining
            )
        ]
        if not front:
            raise RuntimeError("Pareto ordering made no progress")
        rank_sum = {candidate["candidate_id"]: 0 for candidate in front}
        for objective in PARETO_OBJECTIVES:
            distinct = sorted(
                {candidate["prefeature"]["metrics"][objective] for candidate in front}
            )
            ranks = {value: rank for rank, value in enumerate(distinct)}
            for candidate in front:
                rank_sum[candidate["candidate_id"]] += ranks[
                    candidate["prefeature"]["metrics"][objective]
                ]
        front.sort(key=lambda candidate: (rank_sum[candidate["candidate_id"]], candidate["candidate_id"]))
        for candidate in front:
            candidate["pareto_front"] = front_index
            candidate["pareto_tie_rank_sum"] = rank_sum[candidate["candidate_id"]]
        ordered.extend(front)
        selected = {candidate["candidate_id"] for candidate in front}
        remaining = [candidate for candidate in remaining if candidate["candidate_id"] not in selected]
        front_index += 1
    return ordered


def _protect_incumbent(ordered, incumbent_id):
    incumbent = [candidate for candidate in ordered if candidate["candidate_id"] == incumbent_id]
    if len(incumbent) != 1:
        raise ValueError("protected incumbent missing or duplicated: {}".format(incumbent_id))
    return incumbent + [candidate for candidate in ordered if candidate["candidate_id"] != incumbent_id]


def _b7_diverse_order(b6_order, incumbent_id):
    selected = []
    seen = set()

    def add(candidate):
        if candidate["candidate_id"] not in seen:
            selected.append(candidate)
            seen.add(candidate["candidate_id"])

    add(next(candidate for candidate in b6_order if candidate["candidate_id"] == incumbent_id))
    for mode in EXPERIMENTAL_MODES:
        candidate = next((candidate for candidate in b6_order if candidate["residence_mode"] == mode), None)
        if candidate is not None:
            add(candidate)
    for candidate in b6_order:
        add(candidate)
    return selected


def _candidate_brief(candidate):
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_role": candidate["candidate_role"],
        "residence_mode": candidate["residence_mode"],
        "config_index": candidate["config_index"],
        "feature_version": FEATURE_VERSION,
        "feature_sha256": candidate["prefeature"]["feature_sha256"],
    }


def _budget_record(ordered, budget):
    prefix = ordered[: min(int(budget), len(ordered))]
    ids = [candidate["candidate_id"] for candidate in prefix]
    if len(ids) != len(set(ids)):
        raise AssertionError("board budget contains duplicate candidate_id")
    return {
        "requested_budget": int(budget),
        "effective_budget": len(prefix),
        "candidate_ids": ids,
        "candidates": [_candidate_brief(candidate) for candidate in prefix],
        "mode_counts": dict(sorted(Counter(candidate["residence_mode"] for candidate in prefix).items())),
    }


def build_shortlists(pool, records, budgets=DEFAULT_BUDGETS):
    """Build deterministic B4--B7 shortlists without consuming board labels."""
    _assert_label_free(pool, "$.candidate_pool")
    _assert_label_free(records, "$.results")
    budgets = tuple(sorted(set(int(budget) for budget in budgets)))
    if not budgets or any(budget <= 0 for budget in budgets):
        raise ValueError("budgets must be positive")
    pool_ids, protected = _pool_candidate_ids(pool)
    if len(records) != len({record["candidate_id"] for record in records}):
        raise ValueError("results contain duplicate candidate_id")
    result_ids = {record["candidate_id"] for record in records}
    if result_ids != pool_ids:
        raise ValueError(
            "candidate pool/results identity mismatch: missing={} extra={}".format(
                sorted(pool_ids - result_ids), sorted(result_ids - pool_ids)
            )
        )

    failures = []
    candidates_by_workload = {workload_id: [] for workload_id in protected}
    features = []
    for record in records:
        workload_id = record["workload_id"]
        if workload_id not in candidates_by_workload:
            raise ValueError("unknown workload_id in results: {}".format(workload_id))
        if record.get("status") != "ok" or record.get("failure") is not None:
            failure = record.get("failure") or {}
            failures.append(
                {
                    "candidate_id": record["candidate_id"],
                    "workload_id": workload_id,
                    "candidate_role": record["candidate_role"],
                    "residence_mode": record["residence_mode"],
                    "config_index": record["debug"]["config_index"],
                    "category": failure.get("category", "unknown"),
                    "subcategory": failure.get("subcategory", "unknown"),
                    "phase": failure.get("phase"),
                    "retryable": failure.get("retryable"),
                }
            )
            continue
        prefeature = extract_prefeature(record)
        candidate = {
            "candidate_id": record["candidate_id"],
            "workload_id": workload_id,
            "candidate_role": record["candidate_role"],
            "residence_mode": record["residence_mode"],
            "config_index": record["debug"]["config_index"],
            "protected": bool(record.get("protected", False)),
            "prefeature": prefeature,
        }
        candidates_by_workload[workload_id].append(candidate)
        features.append(dict(_candidate_brief(candidate), metrics=prefeature["metrics"], request_histogram=prefeature["request_histogram"]))

    workload_outputs = {}
    for workload_id in sorted(candidates_by_workload):
        candidates = candidates_by_workload[workload_id]
        incumbent_id = protected[workload_id]
        if not any(
            candidate["candidate_id"] == incumbent_id
            and candidate["candidate_role"] == "protected_original_incumbent"
            and candidate["residence_mode"] == "original"
            for candidate in candidates
        ):
            raise ValueError("successful protected original incumbent missing for {}".format(workload_id))

        b4 = _protect_incumbent(sorted(candidates, key=lambda candidate: candidate["candidate_id"]), incumbent_id)
        b5 = _protect_incumbent(
            sorted(
                candidates,
                key=lambda candidate: (
                    candidate["prefeature"]["metrics"]["total_dma_bytes"],
                    candidate["prefeature"]["metrics"]["load_calls"],
                    candidate["candidate_id"],
                ),
            ),
            incumbent_id,
        )
        b6 = _protect_incumbent(pareto_order([dict(candidate) for candidate in candidates]), incumbent_id)
        b7 = _b7_diverse_order(b6, incumbent_id)
        rankings = {"B4": b4, "B5": b5, "B6": b6, "B7": b7}
        workload_failures = sorted(
            (failure for failure in failures if failure["workload_id"] == workload_id),
            key=lambda failure: failure["candidate_id"],
        )
        workload_outputs[workload_id] = {
            "protected_original_incumbent": incumbent_id,
            "eligible_lower_successes": len(candidates),
            "filtered_lower_failures": len(workload_failures),
            "strategies": {
                strategy: {
                    "ranking_candidate_ids": [candidate["candidate_id"] for candidate in rankings[strategy]],
                    "budgets": {
                        str(budget): _budget_record(rankings[strategy], budget)
                        for budget in budgets
                    },
                }
                for strategy in STRATEGIES
            },
        }

    features.sort(key=lambda feature: feature["candidate_id"])
    failures.sort(key=lambda failure: failure["candidate_id"])
    feature_set = [
        {"candidate_id": feature["candidate_id"], "feature_sha256": feature["feature_sha256"]}
        for feature in features
    ]
    shortlist = {
        "schema": SCHEMA,
        "planner_version": PLANNER_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_set_sha256": payload_sha256(feature_set),
        "budgets": list(budgets),
        "budget_unit": "unique_candidate_id_per_workload_and_strategy",
        "label_policy": {
            "status": "strict_input_firewall",
            "forbidden_keys": sorted(FORBIDDEN_LABEL_KEYS),
            "latency_used": False,
            "correctness_used": False,
            "regret_evaluated": False,
        },
        "strategy_definitions": {
            "B4": "protected original incumbent first, then all successful lowers by candidate_id",
            "B5": "B4 eligible set; incumbent protected, remainder by total DMA bytes, load calls, candidate_id",
            "B6": "B4 eligible set; incumbent protected, remainder by full request-shape Pareto fronts and stable rank-sum/candidate_id tie-break",
            "B7": "B6 with incumbent first and best available input/weight/hybrid representative before Pareto fill",
        },
        "pareto_objectives": list(PARETO_OBJECTIVES),
        "workloads": workload_outputs,
    }
    failure_counts = Counter((failure["category"], failure["subcategory"]) for failure in failures)
    screening = {
        "schema": "c3_predispatch_screening_cost_v1",
        "input_records": len(records),
        "unique_candidate_ids": len(result_ids),
        "lower_successes_board_eligible": len(features),
        "lower_failures_filtered_before_board": len(failures),
        "board_dispatches_consumed_by_screening": 0,
        "screening_cost_unit": "P4b static candidate record",
        "failure_counts": {
            "{}:{}".format(category, subcategory): count
            for (category, subcategory), count in sorted(failure_counts.items())
        },
    }
    return shortlist, features, failures, screening


def _write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite frozen artifact: {}".format(path))
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", default="4,8,16")
    args = parser.parse_args()
    pool_path = Path(args.candidate_pool)
    results_path = Path(args.results)
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    budgets = tuple(int(value) for value in args.budgets.split(",") if value)
    shortlist, features, failures, screening = build_shortlists(pool, records, budgets)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_new(output_dir / "shortlist.json", shortlist)
    _write_new(output_dir / "candidate_features.json", features)
    _write_new(output_dir / "filtered_failures.json", failures)
    _write_new(output_dir / "screening_cost.json", screening)
    manifest = {
        "schema": "c3_predispatch_shortlist_manifest_v1",
        "planner_version": PLANNER_VERSION,
        "planner_source_sha256": file_sha256(__file__),
        "candidate_pool": {"path": str(pool_path), "sha256": file_sha256(pool_path)},
        "results": {"path": str(results_path), "sha256": file_sha256(results_path)},
        "feature_version": FEATURE_VERSION,
        "feature_set_sha256": shortlist["feature_set_sha256"],
        "label_free": True,
        "board_access": False,
    }
    _write_new(output_dir / "input_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "frozen_label_free_shortlist",
                "workloads": len(shortlist["workloads"]),
                "eligible": len(features),
                "filtered": len(failures),
                "feature_set_sha256": shortlist["feature_set_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
