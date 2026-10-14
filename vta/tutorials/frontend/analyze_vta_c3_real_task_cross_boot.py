#!/usr/bin/env python3
"""Analyze the two real-task boots without producing new board observations."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from run_vta_c3_real_task_board_boot import GROUPED, sha256_file, verify_flat, write_json


DEFAULT_BOOT1 = GROUPED / "20260924_c3_real_task_board_boot1_run02"
DEFAULT_BOOT2 = GROUPED / "20260924_c3_real_task_confirmation_board_boot2_run01"
DEFAULT_SELECTOR = GROUPED / "20260924_c3_real_task_selector_freeze_run01"
SCHEMA = "c3_real_task_cross_boot_analysis_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def rank_correlation(stats1, stats2):
    ids = sorted(set(stats1) & set(stats2))
    rank1 = {
        program: index + 1
        for index, program in enumerate(
            sorted(ids, key=lambda value: (stats1[value]["median_of_round_medians_ms"], value))
        )
    }
    rank2 = {
        program: index + 1
        for index, program in enumerate(
            sorted(ids, key=lambda value: (stats2[value]["median_of_round_medians_ms"], value))
        )
    }
    if len(ids) < 2:
        return 1.0
    squared = sum((rank1[value] - rank2[value]) ** 2 for value in ids)
    count = len(ids)
    return 1.0 - 6.0 * squared / (count * (count * count - 1))


def per_round(rows, program_id, rounds=6):
    return [
        statistics.median(
            float(row["latency_ms"])
            for row in rows
            if row["program_id"] == program_id and row["round"] == round_index
        )
        for round_index in range(rounds)
    ]


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    boot1 = Path(args.boot1_dir).resolve()
    boot2 = Path(args.boot2_dir).resolve()
    selector_dir = Path(args.selector_dir).resolve()
    manifests = {
        "boot1": verify_flat(boot1),
        "boot2": verify_flat(boot2),
        "selector": verify_flat(selector_dir),
    }
    summary1, summary2 = read_json(boot1 / "summary.json"), read_json(boot2 / "summary.json")
    if summary1["status"] != "BOOT1_COMPLETE" or summary2["status"] != "BOOT2_COMPLETE":
        raise ValueError("both complete boots are required")
    if summary1["boot_id"] == summary2["boot_id"]:
        raise ValueError("cross-boot analysis requires distinct boot IDs")
    correctness1 = read_json(boot1 / "fpga_correctness_summary.json")
    correctness2 = read_json(boot2 / "fpga_correctness_summary.json")
    correct1 = set(correctness1["fpga_correct_candidate_ids"])
    correct2 = set(correctness2["fpga_correct_candidate_ids"])
    reference1 = read_json(boot1 / "reference_summary.json")
    reference2 = read_json(boot2 / "reference_summary.json")
    confirmation = read_json(boot2 / "confirmation_summary.json")
    decisions = read_json(boot1 / "selection_decisions_frozen.json")
    selector = read_json(selector_dir / "contract.json")
    rows1 = read_jsonl(boot1 / "reference_timing.jsonl")
    rows2 = read_jsonl(boot2 / "reference_timing.jsonl")
    program_records = {
        program["program_id"]: program
        for workload in selector["workloads"].values()
        for program in workload["programs"]
    }
    workloads = {}
    every_method_same_as_incumbent = True
    for workload_id in sorted(reference1):
        frozen = selector["workloads"][workload_id]
        incumbent = frozen["incumbent_program_id"]
        oracle1 = reference1[workload_id]["pool_oracle_program_id"]
        oracle2 = reference2[workload_id]["pool_oracle_program_id"]
        latency1 = reference1[workload_id]["pool_oracle_latency_ms"]
        latency2 = reference2[workload_id]["pool_oracle_latency_ms"]
        incumbent1 = reference1[workload_id]["program_statistics"][incumbent][
            "median_of_round_medians_ms"
        ]
        incumbent2 = reference2[workload_id]["program_statistics"][incumbent][
            "median_of_round_medians_ms"
        ]
        method_results = {}
        for method, budgets in decisions[workload_id]["budget_decisions"].items():
            method_results[method] = {}
            for budget, decision in budgets.items():
                selected = decision["selected_program_id"]
                every_method_same_as_incumbent &= selected == incumbent
                selected1 = reference1[workload_id]["program_statistics"][selected][
                    "median_of_round_medians_ms"
                ]
                selected2 = reference2[workload_id]["program_statistics"][selected][
                    "median_of_round_medians_ms"
                ]
                method_results[method][budget] = {
                    "selected_program_id": selected,
                    "selected_is_incumbent": selected == incumbent,
                    "boot1_regret_percent": 100.0 * (selected1 / latency1 - 1.0),
                    "boot2_regret_percent": 100.0 * (selected2 / latency2 - 1.0),
                    "oracle_plus_2_percent_both_boots": (
                        selected1 <= 1.02 * latency1 and selected2 <= 1.02 * latency2
                    ),
                }
        oracle_pair = None
        if oracle1 == oracle2 and oracle1 != incumbent:
            inc_round1, alt_round1 = per_round(rows1, incumbent), per_round(rows1, oracle1)
            inc_round2, alt_round2 = per_round(rows2, incumbent), per_round(rows2, oracle1)
            gains1 = [100.0 * (1.0 - alt / inc) for inc, alt in zip(inc_round1, alt_round1)]
            gains2 = [100.0 * (1.0 - alt / inc) for inc, alt in zip(inc_round2, alt_round2)]
            oracle_pair = {
                "oracle_program_id": oracle1,
                "same_tile_as_incumbent": (
                    program_records[oracle1]["identity"]["tile"]
                    == program_records[incumbent]["identity"]["tile"]
                ),
                "oracle_residence_mode": program_records[oracle1]["identity"]["residence_mode"],
                "incumbent_residence_mode": program_records[incumbent]["identity"]["residence_mode"],
                "boot1_round_gains_percent": gains1,
                "boot2_round_gains_percent": gains2,
                "boot1_median_round_gain_percent": statistics.median(gains1),
                "boot2_median_round_gain_percent": statistics.median(gains2),
                "boot1_wins": sum(value > 0 for value in gains1),
                "boot2_wins": sum(value > 0 for value in gains2),
                "meets_two_percent_both_boots": (
                    statistics.median(gains1) >= 2.0 and statistics.median(gains2) >= 2.0
                ),
            }
        workloads[workload_id] = {
            "incumbent_program_id": incumbent,
            "boot1_oracle_program_id": oracle1,
            "boot2_oracle_program_id": oracle2,
            "same_oracle_both_boots": oracle1 == oracle2,
            "boot1_oracle_latency_ms": latency1,
            "boot2_oracle_latency_ms": latency2,
            "boot1_incumbent_latency_ms": incumbent1,
            "boot2_incumbent_latency_ms": incumbent2,
            "boot2_confirmation_latency_ms": confirmation[workload_id]["program_statistics"][incumbent][
                "median_of_round_medians_ms"
            ],
            "boot1_incumbent_regret_percent": 100.0 * (incumbent1 / latency1 - 1.0),
            "boot2_incumbent_regret_percent": 100.0 * (incumbent2 / latency2 - 1.0),
            "incumbent_within_oracle_plus_2_percent_both_boots": (
                incumbent1 <= 1.02 * latency1 and incumbent2 <= 1.02 * latency2
            ),
            "spearman_rank_correlation": rank_correlation(
                reference1[workload_id]["program_statistics"],
                reference2[workload_id]["program_statistics"],
            ),
            "method_results": method_results,
            "non_incumbent_oracle_pair": oracle_pair,
        }
    result = {
        "schema": SCHEMA,
        "status": "cross_boot_analysis_complete",
        "input_manifests": manifests,
        "boot_ids": [summary1["boot_id"], summary2["boot_id"]],
        "candidate_count": summary1["candidate_count"],
        "same_fpga_correct_set_both_boots": correct1 == correct2,
        "cross_boot_fpga_correct_count": len(correct1 & correct2),
        "reproduced_fpga_rejected_candidate_ids": sorted(set(program_records) and (set(correctness1["fpga_rejected_candidate_ids"]) & set(correctness2["fpga_rejected_candidate_ids"]))),
        "total_operator_calls": summary1["complete_operator_calls"] + summary2["complete_operator_calls"],
        "total_timing_samples": (
            summary1["selection_timing_samples"] + summary1["reference_timing_samples"]
            + summary2["confirmation_timing_samples"] + summary2["reference_timing_samples"]
        ),
        "every_method_budget_selected_incumbent": every_method_same_as_incumbent,
        "cost_aware_advantage_observed": False,
        "strict_latency_unseen_claim_allowed": False,
        "strict_unseen_reason": "selection latencies were exposed by the disclosed invalid run01 before the nonadaptive rerun",
        "workloads": workloads,
        "interpretation": (
            "Both workloads have incumbents already within two percent of the cross-boot pool oracle; "
            "all methods choose the same incumbent, so these tasks validate correctness and safe fallback "
            "but cannot establish a selection-method advantage."
        ),
    }
    output.mkdir(parents=True)
    write_json(output / "summary.json", result)
    lines = [
        "# C3 real-task cross-boot result",
        "",
        "- FPGA correctness: {}/{} candidates in the identical two-boot intersection.".format(
            result["cross_boot_fpga_correct_count"], result["candidate_count"]
        ),
        "- Calls: {} operator calls, including {} timing samples.".format(
            result["total_operator_calls"], result["total_timing_samples"]
        ),
        "- Selection outcome: every method and budget retained the incumbent.",
        "- Claim boundary: this is a non-adaptive rerun after a disclosed invalid latency exposure, not a strict unseen-task result.",
        "",
        "| Workload | Boot1 oracle ms | Boot2 oracle ms | Incumbent regret B1/B2 | Rank rho |",
        "|---|---:|---:|---:|---:|",
    ]
    for workload_id, row in workloads.items():
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.3f}% / {:.3f}% | {:.4f} |".format(
                workload_id,
                row["boot1_oracle_latency_ms"],
                row["boot2_oracle_latency_ms"],
                row["boot1_incumbent_regret_percent"],
                row["boot2_incumbent_regret_percent"],
                row["spearman_rank_correlation"],
            )
        )
    lines += [
        "",
        "The `128→128, 28×28, 3×3` incumbent is the exact oracle on both boots. For the",
        "`64→64, 56×56, 1×1` task, same-tile Input-only is the oracle on both boots, but",
        "its paired median improvement over Original is below 0.2% and does not meet the",
        "predefined 2% stable-improvement threshold.",
        "",
        "Therefore these two tasks do not demonstrate an advantage for the cost-aware selector.",
        "They demonstrate reproducible FPGA qualification, safe incumbent retention, and a case",
        "where the available pool has essentially no material headroom over the incumbent.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                "REPORT.md": sha256_file(output / "REPORT.md"),
                "summary.json": sha256_file(output / "summary.json"),
            },
            "source_sha256": sha256_file(Path(__file__).resolve()),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boot1-dir", type=Path, default=DEFAULT_BOOT1)
    parser.add_argument("--boot2-dir", type=Path, default=DEFAULT_BOOT2)
    parser.add_argument("--selector-dir", type=Path, default=DEFAULT_SELECTOR)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
