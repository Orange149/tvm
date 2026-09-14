#!/usr/bin/env python3
"""Render publication-readable variants of the immutable P7R524 C3 figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position

import plot_vta_c3_literature_results as base


def verify(directory):
    directory = Path(directory).resolve()
    ledger = base.read_json(directory / "artifact_hashes.json")["artifacts"]
    for relative, expected in ledger.items():
        if base.sha256(directory / relative) != expected:
            raise RuntimeError("artifact hash mismatch: {}".format(directory / relative))
    return base.sha256(directory / "artifact_hashes.json")


def save(fig, output, stem):
    fig.savefig(output / (stem + ".pdf"), bbox_inches="tight")
    fig.savefig(output / (stem + ".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_search_curves(rows, output, wall=False):
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8), sharey=True)
    longest = base.longest_budget_runs(rows)
    for axis, workload_id in zip(axes, ("R18-H1", "R18-H2", "R18-H3")):
        for policy in base.POLICIES:
            runs = [
                row
                for row in longest
                if row["workload_id"] == workload_id and row["policy"] == policy
            ]
            curve = base.median_curve(
                runs, "cumulative_wall_seconds" if wall else "dispatch"
            )
            if curve:
                axis.step(
                    [item[0] for item in curve],
                    [item[1] for item in curve],
                    where="post",
                    label=base.LABELS[policy],
                    color=base.COLORS[policy],
                    linewidth=1.6,
                )
        axis.axhline(2.0, color="black", linestyle="--", linewidth=0.8)
        axis.axhline(5.0, color="grey", linestyle=":", linewidth=0.8)
        axis.set_yscale("symlog", linthresh=2.0, linscale=1.0)
        axis.set_ylim(bottom=0.0)
        axis.set_title(workload_id, pad=6)
        axis.set_xlabel("Counterfactual wall time (s)" if wall else "Gross candidate dispatch")
        axis.grid(alpha=0.22, which="both")
    axes[0].set_ylabel("Median best-so-far regret (%)\n(symmetric-log scale above 2%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    if wall:
        fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0))
        fig.subplots_adjust(top=0.9, bottom=0.3, left=0.07, right=0.99, wspace=0.16)
    else:
        fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.99))
        fig.subplots_adjust(top=0.76, bottom=0.18, left=0.07, right=0.99, wspace=0.16)
    save(fig, output, "figure2_best_vs_wall" if wall else "figure1_best_vs_trial")


def plot_actual_t0_curves(clean_start, output):
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    groups = (
        ("AutoTVM-XGB", clean_start["xgb_runs"], "#4c78a8"),
        ("Ours DMA-MF", clean_start["ours_runs"], "#e45756"),
    )
    for method, runs, color in groups:
        for index, run in enumerate(runs):
            points = [row for row in run["best_so_far_trace"] if row["best_ms"] is not None]
            axis.step(
                [row["T0_seconds"] for row in points],
                [row["best_ms"] for row in points],
                where="post",
                color=color,
                alpha=0.55,
                linewidth=1.6,
                label=method if index == 0 else None,
            )
    observed = float(clean_start["observed_cross_space_operator_oracle_ms"])
    axis.axhline(observed, color="black", linewidth=0.9, label="Observed minimum")
    axis.axhline(
        observed * 1.02,
        color="black",
        linestyle="--",
        linewidth=0.9,
        label="Observed minimum +2%",
    )
    axis.set_xlabel("Actual T0 elapsed wall time (s)")
    axis.set_ylabel("Best isolated R18-H1 latency (ms)")
    axis.set_title("Three independent clean-start runs per method")
    axis.grid(alpha=0.22)
    axis.legend(ncol=2, fontsize=9)
    fig.subplots_adjust(bottom=0.15, top=0.9, left=0.13, right=0.98)
    save(fig, output, "figure2_best_vs_actual_t0_wall")


def plot_invalid_stability(rows, hw_summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    workloads = ("R18-H1", "R18-H2", "R18-H3")
    levels = ("random_initialization", "neighbor_presampling", "balanced_e0")
    names = ("Random E0", "Neighbour E0", "Balanced E0")
    width = 0.24
    x = np.arange(len(workloads))
    for index, (level, name) in enumerate(zip(levels, names)):
        medians = [
            hw_summary["workloads"][workload][level]["e0_invalid_ratio"]["median"] * 100.0
            for workload in workloads
        ]
        axes[0].bar(x + (index - 1) * width, medians, width, label=name)
    axes[0].set_xticks(x, workloads)
    axes[0].set_ylabel("Invalid configurations in E0 (%)")
    axes[0].set_title("Initialization validity")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.22)

    data = []
    for policy in base.POLICIES:
        data.append([
            row["target"]["final_regret_fraction"] * 100.0
            for row in rows
            if row["policy"] == policy
            and row["requested_budget"] == 50
            and row["target"]["final_regret_fraction"] is not None
        ])
    axes[1].boxplot(data, tick_labels=[base.LABELS[p] for p in base.POLICIES], showfliers=True)
    axes[1].set_yscale("symlog", linthresh=0.05, linscale=1.0)
    axes[1].set_ylim(bottom=-0.01)
    axes[1].tick_params(axis="x", rotation=32)
    for label in axes[1].get_xticklabels():
        label.set_ha("right")
    axes[1].set_ylabel("Final regret at budget 50 (%)\n(symmetric-log scale above 0.05%)")
    axes[1].set_title("20-seed distribution across three geometries")
    axes[1].grid(axis="y", alpha=0.22, which="both")
    fig.subplots_adjust(bottom=0.32, top=0.9, left=0.08, right=0.99, wspace=0.28)
    save(fig, output, "figure3_invalid_ratio_and_stability")


def plot_transfer(cheng_rows, fullgraph, output):
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.8))
    metrics = (
        ("total_dma_bytes", "Logical DMA change (%)"),
        ("pure_instruction_ms", "Pure-instruction change (%)"),
        ("operator_latency_ms", "Operator latency change (%)"),
    )
    modes = (
        "input_stationary",
        "weight_resident_barrier",
        "input_weight_resident_barrier",
    )
    mode_labels = ("Input", "Weight", "Combined")
    for axis, (metric, label) in zip(axes[:3], metrics):
        changes = base.paired_scheme_changes(cheng_rows, metric)
        data = [changes.get(mode, []) for mode in modes]
        axis.boxplot(data, tick_labels=mode_labels, showfliers=False)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.22)
        for position, values in enumerate(data, 1):
            axis.text(position, 0.98, "n={}".format(len(values)), transform=axis.get_xaxis_transform(), ha="center", va="top", fontsize=8)
    axes[0].set_title("Shared-memory traffic")
    axes[1].set_title("Instruction execution")
    axes[2].set_title("Isolated operator")

    stock = fullgraph["stock_reference"]["median_ms"]
    policies = (
        "stock_mode_aware_xgb",
        "cheng_minimum_access",
        "ml2tuner_pva",
        "ours_dma_multifidelity",
    )
    changes = [
        (fullgraph["policy_results"][policy]["median_ms"] / stock - 1.0) * 100.0
        for policy in policies
    ]
    axes[3].bar(np.arange(len(policies)), changes, color=[base.COLORS[p] for p in policies])
    axes[3].axhline(0.0, color="black", linewidth=0.8)
    axes[3].set_xticks(np.arange(len(policies)), [base.LABELS[p] for p in policies], rotation=32, ha="right")
    axes[3].set_ylabel("Full-graph latency vs stock (%)")
    axes[3].set_title("ResNet18 transfer after safe fallback")
    axes[3].grid(axis="y", alpha=0.22)
    fig.subplots_adjust(bottom=0.28, top=0.88, left=0.055, right=0.995, wspace=0.34)
    save(fig, output, "figure4_dma_instruction_operator_fullgraph_transfer")


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    inputs = {
        "pool_analysis": Path(args.pool_analysis_dir).resolve(),
        "hw_analysis": Path(args.hw_analysis_dir).resolve(),
        "cheng_analysis": Path(args.cheng_analysis_dir).resolve(),
        "fullgraph_board": Path(args.fullgraph_board_dir).resolve(),
        "original_p7r524": Path(args.original_figure_dir).resolve(),
        "clean_start_analysis": Path(args.clean_start_analysis_dir).resolve(),
    }
    manifests = {name: verify(path) for name, path in inputs.items()}
    pool_rows = base.read_jsonl(inputs["pool_analysis"] / "results.jsonl")
    hw_summary = base.read_json(inputs["hw_analysis"] / "summary.json")
    cheng_rows = base.read_jsonl(inputs["cheng_analysis"] / "results.jsonl")
    fullgraph = base.read_json(inputs["fullgraph_board"] / "summary.json")
    clean_start = base.read_json(inputs["clean_start_analysis"] / "summary.json")
    if fullgraph.get("status") != "complete_non_spliced_selected_fullgraph_comparison":
        raise RuntimeError("complete selected full-graph comparison is required")
    if clean_start.get("status") != "three_by_three_non_spliced_clean_start_comparison_complete":
        raise RuntimeError("complete three-by-three clean-start analysis is required")
    output.mkdir(parents=True)
    plot_search_curves(pool_rows, output, wall=False)
    plot_actual_t0_curves(clean_start, output)
    plot_invalid_stability(pool_rows, hw_summary, output)
    plot_transfer(cheng_rows, fullgraph, output)
    summary = {
        "schema": "c3_literature_publication_figure_amendment_v1",
        "status": "publication_readability_amendment_complete",
        "data_or_policy_changed": False,
        "changes": [
            "symlog regret axes expose both early outliers and the preregistered 2/5-percent bands",
            "legend and subplot margins prevent title overlap",
            "figure 2 uses measured P7R523 T0 process wall rather than replayed counterfactual wall",
            "budget-50 stability retains all 20-seed outcomes and uses symlog instead of a clipped linear axis",
            "Cheng transfer panels annotate the measured same-tile sample count",
        ],
        "input_manifests": manifests,
        "figures": sorted(path.name for path in output.glob("figure*")),
        "logical_dma_is_physical_axi": False,
        "random_input_equivalence_is_imagenet_accuracy": False,
    }
    for name, value in (
        ("summary.json", summary),
        ("contract.json", {"schema": summary["schema"], "input_manifests": manifests}),
    ):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    (output / "results.jsonl").write_text(
        "".join(json.dumps({"figure": name}, sort_keys=True) + "\n" for name in summary["figures"]),
        encoding="utf-8",
    )
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "artifact_hashes.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    path.name: base.sha256(path)
                    for path in sorted(output.iterdir())
                    if path.is_file() and path.name != "artifact_hashes.json"
                },
                "source_hashes": {
                    str(Path(__file__).resolve()): base.sha256(Path(__file__).resolve()),
                    str(Path(base.__file__).resolve()): base.sha256(Path(base.__file__).resolve()),
                },
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
    parser.add_argument("--pool-analysis-dir", required=True)
    parser.add_argument("--hw-analysis-dir", required=True)
    parser.add_argument("--cheng-analysis-dir", required=True)
    parser.add_argument("--fullgraph-board-dir", required=True)
    parser.add_argument("--original-figure-dir", required=True)
    parser.add_argument("--clean-start-analysis-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
