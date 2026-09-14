#!/usr/bin/env python3
"""Freeze the exposed-workload development fit for the C3 service-cost proxy.

This analysis deliberately includes Y04, whose latency exposed the failure of
the earlier bytes-first rule.  It therefore selects coefficients but cannot be
used as confirmation evidence.  A latency-unseen workload must be evaluated
after this artifact is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P7 = ROOT / "07_grouped_holdout"
DEFAULT_POOLS = (
    P7 / "20260912_p7r131_search_rule_ablation_p7q_development_run02",
    P7 / "20260912_p7r144_y00_layout_preserving_search_board_run01",
    P7 / "20260912_p7r154_y03_layout_preserving_search_board_run01",
    P7 / "20260912_p7r165_y04_clean_start_search_board_run01",
)
DEFAULT_OUTPUT = P7 / "20260912_p7r166_shared_memory_service_proxy_development_run01"
GRID = (0, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288)
FROZEN_REQUEST_EQUIVALENT = 65536
FROZEN_EXTRA_SUBMISSION_EQUIVALENT = 131072


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_run(directory):
    directory = Path(directory)
    ledger_path = directory / "artifact_hashes.json"
    ledger = read_json(ledger_path)
    entries = ledger.get("files", ledger.get("artifacts"))
    if not isinstance(entries, dict):
        raise ValueError("unknown artifact ledger schema: " + str(ledger_path))
    for name, expected in entries.items():
        if sha256(directory / name) != expected:
            raise ValueError("artifact hash mismatch: " + str(directory / name))
    return sha256(ledger_path)


def pool_path(directory):
    for name in ("pool_adapter_snapshot.json", "completed_pool.json"):
        path = Path(directory) / name
        if path.exists():
            return path
    raise FileNotFoundError("no canonical pool in " + str(directory))


def metric(candidate, name):
    return float(candidate["predispatch"]["validity"]["static_" + name])


def score(candidate, request_equivalent, submission_equivalent):
    return (
        metric(candidate, "dma_total_bytes")
        + request_equivalent * metric(candidate, "dma_total_calls")
        + submission_equivalent * max(metric(candidate, "submissions") - 1.0, 0.0)
    )


def measured_candidates(workload):
    result = [
        candidate for candidate in workload["candidates"]
        if isinstance(candidate.get("oracle", {}).get("latency_ms"), (int, float))
    ]
    if not result:
        raise ValueError("workload has no measured candidate")
    return result


def chosen_record(workload_id, candidates, policy, key):
    oracle_latency = min(float(item["oracle"]["latency_ms"]) for item in candidates)
    chosen = min(candidates, key=lambda item: (key(item), item["candidate_id"]))
    latency = float(chosen["oracle"]["latency_ms"])
    return {
        "workload_id": workload_id,
        "policy": policy,
        "candidate_id": chosen["candidate_id"],
        "residence_mode": chosen["residence_mode"],
        "latency_ms": latency,
        "oracle_latency_ms": oracle_latency,
        "regret_percent": (latency / oracle_latency - 1.0) * 100.0,
        "dma_total_bytes": metric(chosen, "dma_total_bytes"),
        "dma_total_calls": metric(chosen, "dma_total_calls"),
        "submissions": metric(chosen, "submissions"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-run", action="append", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    runs = tuple(args.pool_run or DEFAULT_POOLS)
    bindings = []
    workloads = {}
    for run in runs:
        pool = pool_path(run)
        bindings.append({
            "run": str(run.resolve()),
            "ledger_sha256": verify_run(run),
            "pool": str(pool.resolve()),
            "pool_sha256": sha256(pool),
        })
        value = read_json(pool)
        overlap = set(workloads).intersection(value["workloads"])
        if overlap:
            raise ValueError("duplicate workload ids: " + ",".join(sorted(overlap)))
        workloads.update(value["workloads"])

    grid = []
    for request_equivalent, submission_equivalent in itertools.product(GRID, GRID):
        regrets = []
        for workload in workloads.values():
            candidates = measured_candidates(workload)
            oracle = min(float(item["oracle"]["latency_ms"]) for item in candidates)
            chosen = min(candidates, key=lambda item: (
                score(item, request_equivalent, submission_equivalent), item["candidate_id"]
            ))
            regrets.append((float(chosen["oracle"]["latency_ms"]) / oracle - 1.0) * 100.0)
        grid.append({
            "request_equivalent_bytes": request_equivalent,
            "extra_submission_equivalent_bytes": submission_equivalent,
            "max_regret_percent": max(regrets),
            "mean_regret_percent": sum(regrets) / len(regrets),
            "all_workloads_within_oracle_2_percent": max(regrets) <= 2.0,
        })
    feasible = [item for item in grid if item["all_workloads_within_oracle_2_percent"]]
    selected = min(feasible, key=lambda item: (
        item["request_equivalent_bytes"] + item["extra_submission_equivalent_bytes"],
        item["max_regret_percent"], item["mean_regret_percent"],
        item["request_equivalent_bytes"], item["extra_submission_equivalent_bytes"],
    ))
    if (selected["request_equivalent_bytes"], selected["extra_submission_equivalent_bytes"]) != (
        FROZEN_REQUEST_EQUIVALENT, FROZEN_EXTRA_SUBMISSION_EQUIVALENT
    ):
        raise RuntimeError("frozen service weights do not reproduce deterministic grid selection")

    policies = {
        "bytes_only": lambda item: (metric(item, "dma_total_bytes"),),
        "legacy_bytes_first_lexicographic": lambda item: (
            metric(item, "dma_total_bytes"), metric(item, "dma_total_calls"),
            metric(item, "padded_dma_calls"), metric(item, "submissions"),
        ),
        "calls_first_diagnostic": lambda item: (
            metric(item, "dma_total_calls"), metric(item, "submissions"),
            metric(item, "dma_total_bytes"),
        ),
        "shared_memory_service_proxy": lambda item: (
            score(item, FROZEN_REQUEST_EQUIVALENT, FROZEN_EXTRA_SUBMISSION_EQUIVALENT),
        ),
    }
    selections = []
    for workload_id, workload in sorted(workloads.items()):
        candidates = measured_candidates(workload)
        for name, key in policies.items():
            selections.append(chosen_record(workload_id, candidates, name, key))

    analysis = {
        "schema": "c3_shared_memory_service_proxy_development_v1",
        "status": "development_frozen_before_y01_latency_recovery_confirmation",
        "workloads": sorted(workloads),
        "candidate_count": sum(len(measured_candidates(item)) for item in workloads.values()),
        "grid_values_bytes": list(GRID),
        "grid_point_count": len(grid),
        "within_2_percent_grid_point_count": len(feasible),
        "selection_rule": (
            "among grid points whose first dispatch is within each exposed workload pool oracle+2%, "
            "minimize coefficient sum, then max regret, mean regret, request weight, submission weight"
        ),
        "frozen_proxy": {
            "formula": "dma_bytes + 65536*dma_calls + 131072*max(submissions-1,0)",
            "request_equivalent_bytes": FROZEN_REQUEST_EQUIVALENT,
            "extra_submission_equivalent_bytes": FROZEN_EXTRA_SUBMISSION_EQUIVALENT,
            "max_regret_percent": selected["max_regret_percent"],
            "mean_regret_percent": selected["mean_regret_percent"],
            "interpretation": (
                "byte-equivalent acquisition penalties for request launch and extra submission service; "
                "not physical AXI traffic or literal hardware constants"
            ),
        },
        "selections": selections,
        "grid": grid,
        "claim_boundary": (
            "P7Q, Y00, Y03 and Y04 latency labels all participated in coefficient selection. "
            "This artifact is development evidence only; Y01 latency remains unseen and is the next holdout."
        ),
    }

    rows = []
    by_workload = {}
    for item in selections:
        by_workload.setdefault(item["workload_id"], {})[item["policy"]] = item
    for workload_id in sorted(by_workload):
        items = by_workload[workload_id]
        rows.append(
            "| {} | {:.3f}% | {:.3f}% | {:.3f}% | {} |".format(
                workload_id,
                items["bytes_only"]["regret_percent"],
                items["legacy_bytes_first_lexicographic"]["regret_percent"],
                items["shared_memory_service_proxy"]["regret_percent"],
                items["shared_memory_service_proxy"]["residence_mode"],
            )
        )
    report = "\n".join([
        "# P7R166 shared-memory service-cost proxy development freeze", "",
        "> This is exposed-label development, not confirmation. Y04 is part of the fit; Y01 latency is the frozen next holdout.", "",
        "The earlier bytes-first rule failed on Y04 because it rewarded fewer bytes even when a candidate issued more DMA requests and submissions. The frozen replacement is:", "",
        "`score = DMA bytes + 65,536 * DMA calls + 131,072 * max(submissions - 1, 0)`", "",
        "The coefficients are byte-equivalent search penalties, not physical AXI bytes or measured hardware constants.", "",
        "| workload | bytes-only regret | legacy lexicographic regret | service-proxy regret | proxy-selected mode |",
        "|---|---:|---:|---:|---|", *rows, "",
        "Across the seven exposed workloads, the frozen proxy has maximum first-dispatch regret {:.6f}% and mean {:.6f}%.".format(
            selected["max_regret_percent"], selected["mean_regret_percent"]
        ),
        "It must now be tested unchanged on Y01; these seven workloads cannot establish generalization.", "",
    ])

    output = args.output_dir
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output: " + str(output))
    output.mkdir(parents=True)
    command = " ".join([Path(sys.executable).name, Path(__file__).name] + sys.argv[1:]) + "\n"
    files = {
        "input_bindings.json": json.dumps(bindings, indent=2, sort_keys=True) + "\n",
        "analysis.json": json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        "RESULTS.md": report,
        "command.txt": command,
    }
    for name, content in files.items():
        (output / name).write_text(content, encoding="utf-8")
    ledger = {name: sha256(output / name) for name in sorted(files)}
    (output / "artifact_hashes.json").write_text(
        json.dumps({"schema": "c3_immutable_artifact_hashes_v1", "files": ledger},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": analysis["status"], "workloads": len(workloads),
                      "output_dir": str(output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
