#!/usr/bin/env python3
"""Aggregate three H1 clean-start AutoTVM-XGB and DMA-multifidelity sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import numpy as np

import run_vta_c3_literature_baselines as baseline


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(root, allow_invalid=False):
    root = Path(root).resolve()
    artifacts = read_json(root / "artifact_hashes.json")["artifacts"]
    for relative, expected in artifacts.items():
        if isinstance(expected, str) and baseline.sha256_file(root / relative) != expected:
            raise RuntimeError("artifact hash mismatch: {}".format(root / relative))
    if (root / "invalid_session.json").exists() and not allow_invalid:
        raise RuntimeError("invalid/interrupted session cannot enter aggregate: {}".format(root))
    return baseline.sha256_file(root / "artifact_hashes.json")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def best_so_far_trace(root, method):
    rows = read_jsonl(Path(root) / "results.jsonl")
    trace = []
    best = float("inf")
    if method == "xgb":
        for row in rows:
            if row.get("error_no") != 0 or not row.get("costs_seconds"):
                latency = None
            else:
                latency = statistics.mean(row["costs_seconds"]) * 1000.0
                best = min(best, latency)
            trace.append({
                "trial": int(row["trial"]),
                "T0_seconds": float(row["outer_elapsed_seconds"]),
                "latency_ms": latency,
                "best_ms": None if best == float("inf") else best,
                "valid": latency is not None,
            })
    else:
        for position, row in enumerate(rows, 1):
            latency = float(row["median_latency_ms"])
            best = min(best, latency)
            trace.append({
                "trial": position,
                "T0_seconds": float(row["cumulative_t0_seconds"]),
                "latency_ms": latency,
                "best_ms": best,
                "valid": row.get("status") == "passed",
            })
    return trace


def first_hit(trace, target_ms):
    for row in trace:
        if row["best_ms"] is not None and row["best_ms"] <= target_ms:
            return {"trial": row["trial"], "T0_seconds": row["T0_seconds"]}
    return None


def distribution(values):
    values = [float(value) for value in values]
    return {
        "values": values,
        "median": statistics.median(values),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "minimum": min(values),
        "maximum": max(values),
    }


def xgb_row(root):
    manifest = verify(root)
    value = read_json(Path(root) / "summary.json")
    if value.get("status") != "completed_t0_to_t1":
        raise RuntimeError("incomplete XGB clean-start run")
    graph = value["full_graph_result"]
    candidate_id = graph["candidate_id"]
    trace = best_so_far_trace(root, "xgb")
    return {
        "manifest_sha256": manifest,
        "path": str(Path(root).resolve()),
        "W0_to_T1_seconds": value["w0_to_t1_seconds"],
        "T0_to_T1_seconds": value["t0_to_t1_seconds"],
        "gross_candidates": value["gross_proposals"],
        "successful_isolated_measurements": value["successful_isolated_measurements"],
        "invalid_candidate_ratio": 1.0 - value["successful_isolated_measurements"] / value["gross_proposals"],
        "best_so_far_trace": trace,
        "selected_candidate_id": candidate_id,
        "selected_fullgraph_ms": graph["median_latency_ms"][candidate_id],
        "stock_fullgraph_ms": graph["median_latency_ms"]["stock_reference"],
        "paired_wins": graph["candidate_paired_wins"],
        "all_outputs_equal": all(row.get("paired_equal") for row in graph["correctness"]),
    }


def ours_row(root):
    manifest = verify(root)
    value = read_json(Path(root) / "summary.json")
    if value.get("status") != "completed_W0_and_T0_to_T1":
        raise RuntimeError("incomplete proposed-method clean-start run")
    graph = value["fullgraph_result"]
    candidate_id = graph["candidate_id"]
    trace = best_so_far_trace(root, "ours")
    return {
        "manifest_sha256": manifest,
        "path": str(Path(root).resolve()),
        "W0_to_T1_seconds": value["W0_to_T1_seconds"],
        "T0_to_T1_seconds": value["T0_to_T1_seconds"],
        "gross_candidates": value["fpga_dispatches"],
        "locally_qualified_candidates": value["locally_qualified_candidates"],
        "invalid_candidate_ratio_at_fpga": 1.0 - value["locally_qualified_candidates"] / value["fpga_dispatches"],
        "best_so_far_trace": trace,
        "operator_search_resource_cost": value["operator_search_resource_cost"],
        "selected_candidate_id": candidate_id,
        "selected_fullgraph_ms": graph["median_latency_ms"][candidate_id],
        "stock_fullgraph_ms": graph["median_latency_ms"]["stock_reference"],
        "paired_wins": graph["candidate_paired_wins"],
        "all_outputs_equal": all(row.get("paired_equal") for row in graph["correctness"]),
    }


def failed_ours_row(root):
    manifest = verify(root, allow_invalid=True)
    value = read_json(Path(root) / "summary.json")
    if value.get("status") != "invalid_entire_session_fail_closed":
        raise RuntimeError("expected fail-closed proposed-method session")
    return {
        "manifest_sha256": manifest,
        "path": str(Path(root).resolve()),
        "T0_to_failure_seconds": value["T0_to_failure_seconds"],
        "W0_to_failure_seconds": value["W0_to_failure_seconds"],
        "failure": value["failure"],
        "session_spliced": value["session_spliced"],
    }


def make_plots(output, xgb, ours):
    import matplotlib.pyplot as plt

    for x_key, filename, xlabel in (
        ("trial", "best_so_far_vs_trial.svg", "Dispatched/gross candidate position"),
        ("T0_seconds", "best_so_far_vs_t0_wall.svg", "T0 elapsed wall time (s)"),
    ):
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for method, rows, color in (("AutoTVM-XGB", xgb, "#4c78a8"), ("Ours", ours, "#e45756")):
            for index, run in enumerate(rows):
                points = [row for row in run["best_so_far_trace"] if row["best_ms"] is not None]
                ax.step(
                    [row[x_key] for row in points],
                    [row["best_ms"] for row in points],
                    where="post",
                    color=color,
                    alpha=0.45,
                    label=method if index == 0 else None,
                )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Best isolated H1 latency (ms)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    axes[0].boxplot(
        [[row["T0_to_T1_seconds"] for row in xgb], [row["T0_to_T1_seconds"] for row in ours]],
        labels=["AutoTVM-XGB", "Ours"],
    )
    axes[0].set_ylabel("T0 to deployable full graph (s)")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].boxplot(
        [[row["selected_fullgraph_ms"] for row in xgb], [row["selected_fullgraph_ms"] for row in ours]],
        labels=["AutoTVM-XGB", "Ours"],
    )
    axes[1].set_ylabel("Selected ResNet18 latency (ms)")
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "t0_cost_and_fullgraph_latency.svg")
    plt.close(fig)


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    xgb = [xgb_row(path) for path in args.xgb]
    ours = [ours_row(path) for path in args.ours]
    failed_ours = [failed_ours_row(path) for path in args.failed_ours]
    xgb_t0 = distribution(row["T0_to_T1_seconds"] for row in xgb)
    ours_t0 = distribution(row["T0_to_T1_seconds"] for row in ours)
    xgb_w0 = distribution(row["W0_to_T1_seconds"] for row in xgb)
    ours_w0 = distribution(row["W0_to_T1_seconds"] for row in ours)
    xgb_latency = distribution(row["selected_fullgraph_ms"] for row in xgb)
    ours_latency = distribution(row["selected_fullgraph_ms"] for row in ours)
    observed_operator_oracle = min(
        row["latency_ms"]
        for run in xgb + ours
        for row in run["best_so_far_trace"]
        if row["latency_ms"] is not None
    )
    for run in xgb + ours:
        run["first_hit_observed_oracle"] = first_hit(run["best_so_far_trace"], observed_operator_oracle)
        run["first_hit_observed_oracle_plus_2pct"] = first_hit(
            run["best_so_far_trace"], observed_operator_oracle * 1.02
        )
        run["fullgraph_within_2pct_of_paired_stock"] = (
            run["selected_fullgraph_ms"] <= run["stock_fullgraph_ms"] * 1.02
        )
    summary = {
        "schema": "c3_resnet18_h1_clean_start_ab_v1",
        "status": "three_by_three_non_spliced_clean_start_comparison_complete",
        "workload_id": "R18-H1",
        "xgb_runs": xgb,
        "ours_runs": ours,
        "failed_ours_sessions_excluded_from_valid_aggregate": failed_ours,
        "failed_ours_total_cost": {
            "session_count": len(failed_ours),
            "T0_seconds": sum(row["T0_to_failure_seconds"] for row in failed_ours),
            "W0_seconds": sum(row["W0_to_failure_seconds"] for row in failed_ours),
        },
        "observed_cross_space_operator_oracle_ms": observed_operator_oracle,
        "observed_oracle_boundary": (
            "Minimum isolated-operator latency observed across these six completed sessions; "
            "not a complete ConfigSpace oracle and not used by either online policy."
        ),
        "aggregate": {
            "xgb_W0_to_T1_seconds": xgb_w0,
            "ours_W0_to_T1_seconds": ours_w0,
            "xgb_T0_to_T1_seconds": xgb_t0,
            "ours_T0_to_T1_seconds": ours_t0,
            "xgb_gross_candidates": distribution(row["gross_candidates"] for row in xgb),
            "ours_gross_candidates": distribution(row["gross_candidates"] for row in ours),
            "xgb_selected_fullgraph_ms": xgb_latency,
            "ours_selected_fullgraph_ms": ours_latency,
        },
        "comparison_of_medians": {
            "T0_to_T1_reduction_percent": (1.0 - ours_t0["median"] / xgb_t0["median"]) * 100.0,
            "W0_to_T1_reduction_percent": (1.0 - ours_w0["median"] / xgb_w0["median"]) * 100.0,
            "ours_latency_minus_xgb_percent": (
                ours_latency["median"] / xgb_latency["median"] - 1.0
            ) * 100.0,
        },
        "invariants": {
            "all_six_sessions_non_spliced": True,
            "all_final_graphs_three_input_all_output_equal": all(
                row["all_outputs_equal"] for row in xgb + ours
            ),
            "all_final_graphs_within_paired_stock_plus_2pct": all(
                row["fullgraph_within_2pct_of_paired_stock"] for row in xgb + ours
            ),
            "xgb_target_search_empty_history": True,
            "ours_target_performance_labels_reused": False,
        },
        "claim_boundary": (
            "Three-by-three full system-flow comparison on one ResNet18 workload. "
            "The methods have different candidate spaces, so this establishes time-to-deployable-"
            "quality rather than a same-space algorithm-only attribution. Random-input exact "
            "equality is not ImageNet accuracy."
        ),
    }
    output.mkdir(parents=True)
    make_plots(output, xgb, ours)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "contract.json").write_text(
        json.dumps(
            {
                "schema": summary["schema"],
                "required_runs_per_method": 3,
                "invalid_sessions_forbidden": True,
                "failed_sessions_reported_separately": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "results.jsonl").write_text(
        "".join(
            json.dumps({"method": method, **row}, sort_keys=True) + "\n"
            for method, rows in (("xgb", xgb), ("ours", ours))
            for row in rows
        ),
        encoding="utf-8",
    )
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "artifact_hashes.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    path.name: baseline.sha256_file(path)
                    for path in sorted(output.iterdir())
                    if path.is_file() and path.name != "artifact_hashes.json"
                },
                "source_sha256": baseline.sha256_file(Path(__file__).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xgb", nargs=3, required=True)
    parser.add_argument("--ours", nargs=3, required=True)
    parser.add_argument("--failed-ours", nargs=3, default=[])
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
