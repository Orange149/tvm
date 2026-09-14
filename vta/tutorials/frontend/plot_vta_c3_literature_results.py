#!/usr/bin/env python3
"""Generate the four preregistered C3 literature-alignment paper figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import numpy as np  # pylint: disable=wrong-import-position


POLICIES = (
    "random",
    "stock_mode_aware_xgb",
    "rieber_hw_init_xgb",
    "ml2tuner_pva",
    "cheng_minimum_access",
    "ours_dma_multifidelity",
)
LABELS = {
    "random": "Random",
    "stock_mode_aware_xgb": "AutoTVM-XGB",
    "rieber_hw_init_xgb": "HW-Aware+XGB",
    "ml2tuner_pva": "ML2Tuner P/V/A",
    "cheng_minimum_access": "Cheng min-access",
    "ours_dma_multifidelity": "Ours DMA-MF",
}
COLORS = dict(zip(POLICIES, plt.get_cmap("tab10").colors))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def save(fig, output, stem):
    fig.tight_layout()
    fig.savefig(output / (stem + ".pdf"), bbox_inches="tight")
    fig.savefig(output / (stem + ".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def longest_budget_runs(rows):
    maximum = max(row["requested_budget"] for row in rows)
    return [row for row in rows if row["requested_budget"] == maximum]


def median_curve(runs, x_key):
    points = defaultdict(list)
    for run in runs:
        for point in run["target"]["best_so_far"]:
            if point["best_latency_ms"] is not None:
                points[int(point["dispatch"])].append(
                    (float(point[x_key]), float(point["regret_fraction"]) * 100.0)
                )
    curve = []
    for dispatch in sorted(points):
        values = points[dispatch]
        curve.append(
            (
                statistics.median(value[0] for value in values),
                statistics.median(value[1] for value in values),
            )
        )
    return curve


def plot_search_curves(rows, output, wall=False):
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.45), sharey=True)
    for axis, workload_id in zip(axes, ("R18-H1", "R18-H2", "R18-H3")):
        for policy in POLICIES:
            runs = [
                row
                for row in longest_budget_runs(rows)
                if row["workload_id"] == workload_id and row["policy"] == policy
            ]
            curve = median_curve(
                runs, "cumulative_wall_seconds" if wall else "dispatch"
            )
            if curve:
                axis.step(
                    [item[0] for item in curve],
                    [item[1] for item in curve],
                    where="post",
                    label=LABELS[policy],
                    color=COLORS[policy],
                    linewidth=1.5,
                )
        axis.axhline(2.0, color="black", linestyle="--", linewidth=0.8)
        axis.axhline(5.0, color="grey", linestyle=":", linewidth=0.8)
        axis.set_title(workload_id)
        axis.set_xlabel("Counterfactual wall time (s)" if wall else "Gross candidate dispatch")
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Median best-so-far regret (%)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    save(fig, output, "figure2_best_vs_wall" if wall else "figure1_best_vs_trial")


def plot_invalid_stability(rows, hw_summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7))
    workloads = ("R18-H1", "R18-H2", "R18-H3")
    levels = ("random_initialization", "neighbor_presampling", "balanced_e0")
    names = ("Random E0", "Neighbour E0", "Balanced E0")
    width = 0.24
    x = np.arange(len(workloads))
    for index, (level, name) in enumerate(zip(levels, names)):
        medians = [
            hw_summary["workloads"][workload][level]["e0_invalid_ratio"]["median"]
            * 100.0
            for workload in workloads
        ]
        axes[0].bar(x + (index - 1) * width, medians, width, label=name)
    axes[0].set_xticks(x, workloads)
    axes[0].set_ylabel("Invalid configurations in E0 (%)")
    axes[0].set_title("Initialization validity")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.22)

    budget = 50
    positions = np.arange(len(POLICIES))
    medians, lower, upper = [], [], []
    for policy in POLICIES:
        values = [
            row["target"]["final_regret_fraction"] * 100.0
            for row in rows
            if row["policy"] == policy
            and row["requested_budget"] == budget
            and row["target"]["final_regret_fraction"] is not None
        ]
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        medians.append(median)
        lower.append(median - q1)
        upper.append(q3 - median)
    axes[1].errorbar(
        positions,
        medians,
        yerr=np.array([lower, upper]),
        fmt="o",
        capsize=3,
        color="black",
    )
    axes[1].set_xticks(
        positions, [LABELS[policy] for policy in POLICIES], rotation=32, ha="right"
    )
    axes[1].set_ylabel("Final regret at budget 50 (%)")
    axes[1].set_title("20-seed median and IQR")
    axes[1].grid(axis="y", alpha=0.22)
    save(fig, output, "figure3_invalid_ratio_and_stability")


def paired_scheme_changes(cheng_rows, metric):
    values = defaultdict(list)
    for family in cheng_rows:
        schemes = {row["mode"]: row for row in family["schemes"]}
        original = schemes["original"].get(metric)
        if original in (None, 0):
            continue
        for mode in (
            "input_stationary",
            "weight_resident_barrier",
            "input_weight_resident_barrier",
        ):
            value = schemes[mode].get(metric)
            if value is not None and schemes[mode]["status"] == "measured":
                values[mode].append((float(value) / float(original) - 1.0) * 100.0)
    return values


def plot_transfer(cheng_rows, fullgraph, output):
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.65))
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
        changes = paired_scheme_changes(cheng_rows, metric)
        data = [changes.get(mode, []) for mode in modes]
        axis.boxplot(data, labels=mode_labels, showfliers=False)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_title("Shared-memory traffic")
    axes[1].set_title("Instruction execution")
    axes[2].set_title("Isolated operator")

    stock = fullgraph["stock_reference"]["median_ms"]
    policy_order = (
        "stock_mode_aware_xgb",
        "cheng_minimum_access",
        "ml2tuner_pva",
        "ours_dma_multifidelity",
    )
    changes = [
        (fullgraph["policy_results"][policy]["median_ms"] / stock - 1.0) * 100.0
        for policy in policy_order
    ]
    axes[3].bar(
        np.arange(len(policy_order)),
        changes,
        color=[COLORS[policy] for policy in policy_order],
    )
    axes[3].axhline(0.0, color="black", linewidth=0.8)
    axes[3].set_xticks(
        np.arange(len(policy_order)),
        [LABELS[policy] for policy in policy_order],
        rotation=32,
        ha="right",
    )
    axes[3].set_ylabel("Full-graph latency vs stock (%)")
    axes[3].set_title("ResNet18 transfer")
    axes[3].grid(axis="y", alpha=0.22)
    save(fig, output, "figure4_dma_instruction_operator_fullgraph_transfer")


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    pool_rows = read_jsonl(Path(args.pool_analysis_dir) / "results.jsonl")
    hw_summary = read_json(Path(args.hw_analysis_dir) / "summary.json")
    cheng_rows = read_jsonl(Path(args.cheng_analysis_dir) / "results.jsonl")
    fullgraph = read_json(Path(args.fullgraph_board_dir) / "summary.json")
    if fullgraph.get("status") != "complete_non_spliced_selected_fullgraph_comparison":
        raise RuntimeError("complete selected full-graph comparison is required")
    plot_search_curves(pool_rows, output, wall=False)
    plot_search_curves(pool_rows, output, wall=True)
    plot_invalid_stability(pool_rows, hw_summary, output)
    plot_transfer(cheng_rows, fullgraph, output)
    manifest = {
        "status": "four_literature_alignment_figures_complete",
        "figures": sorted(path.name for path in output.iterdir()),
        "logical_dma_is_physical_axi": False,
        "random_input_equivalence_is_imagenet_accuracy": False,
    }
    (output / "summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "contract.json").write_text(
        json.dumps(
            {
                "schema": "c3_literature_figure_contract_v1",
                "inputs": {
                    "pool_analysis": sha256(Path(args.pool_analysis_dir) / "artifact_hashes.json"),
                    "hw_analysis": sha256(Path(args.hw_analysis_dir) / "artifact_hashes.json"),
                    "cheng_analysis": sha256(Path(args.cheng_analysis_dir) / "artifact_hashes.json"),
                    "fullgraph_board": sha256(Path(args.fullgraph_board_dir) / "artifact_hashes.json"),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    (output / "results.jsonl").write_text(
        "".join(
            json.dumps({"figure": name}, sort_keys=True) + "\n"
            for name in manifest["figures"]
        ),
        encoding="utf-8",
    )
    (output / "artifact_hashes.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    path.name: sha256(path)
                    for path in sorted(output.iterdir())
                    if path.is_file() and path.name != "artifact_hashes.json"
                },
                "source_sha256": sha256(Path(__file__).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-analysis-dir", required=True)
    parser.add_argument("--hw-analysis-dir", required=True)
    parser.add_argument("--cheng-analysis-dir", required=True)
    parser.add_argument("--fullgraph-board-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
