#!/usr/bin/env python3
"""Generate summary tables/plots for AXU5EVB VTA vs CPU measurements.

This script uses the measured numbers collected during the debugging session and
produces:
1. Markdown tables printed to stdout
2. CSV files written to an output directory
3. Optional PNG charts when matplotlib is available
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


WHOLE_GRAPH = {
    "community_run_only_ms": {"vta": 107.797, "cpu": 722.714},
    "params_once_run_ms": {"vta": 110.119, "cpu": 709.517},
    "vta_stage_avg_ms": {
        "set_params": 1644.526,
        "set_data": 67.129,
        "run": 111.417,
        "get_output": 4.800,
        "total": 1827.871,
    },
}


CONV_KERNEL_MS = [
    {"layer": "C2", "shape": "56x56 ic64 oc64 k3 s1", "vta": 5.8494, "cpu": 11.6888},
    {"layer": "C3", "shape": "56x56 ic64 oc128 k3 s2", "vta": 3.0101, "cpu": 6.2386},
    {"layer": "C4", "shape": "56x56 ic64 oc128 k1 s2", "vta": 1.2439, "cpu": 0.8780},
    {"layer": "C5", "shape": "28x28 ic128 oc128 k3 s1", "vta": 5.2621, "cpu": 10.9056},
    {"layer": "C6", "shape": "28x28 ic128 oc256 k3 s2", "vta": 2.6880, "cpu": 6.3599},
    {"layer": "C7", "shape": "28x28 ic128 oc256 k1 s2", "vta": 0.9416, "cpu": 0.7984},
    {"layer": "P3", "shape": "14x14 ic256 oc512 k1 s2", "vta": 0.9247, "cpu": 0.8510},
]


HOT_OPS_MS = [
    ("tvmgen_default_fused_nn_conv2d_add_nn_relu", 21.41256),
    ("tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast", 11.73471),
    ("tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast_2", 10.54847),
    ("tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast_4", 9.93679),
    ("tvmgen_default_fused_nn_conv2d_add_nn_relu_add_right_shift_clip_cast_1", 6.40302),
    ("tvmgen_default_fused_nn_conv2d_add_nn_relu_add_right_shift_clip_cast", 6.39750),
]


def markdown_table(headers, rows):
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(h)) for h in headers]
    for row in str_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def fmt_row(row):
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    lines = [fmt_row([str(h) for h in headers]), sep]
    lines.extend(fmt_row(row) for row in str_rows)
    return "\n".join(lines)


def write_csv(path: Path, headers, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def try_plot(output_dir: Path):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except Exception:
        print("[plot] matplotlib not available, skip PNG generation")
        return

    palette = {
        "vta": "#0B6E4F",
        "cpu": "#C84C09",
        "accent": "#1F2937",
        "muted": "#6B7280",
        "grid": "#D1D5DB",
        "memory": "#EAB308",
        "run": "#0EA5E9",
        "output": "#A855F7",
    }

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": palette["accent"],
            "axes.labelcolor": palette["accent"],
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "xtick.color": palette["accent"],
            "ytick.color": palette["accent"],
            "font.size": 11,
            "grid.color": palette["grid"],
            "grid.alpha": 0.6,
            "grid.linestyle": "--",
        }
    )

    def annotate_bars(ax, bars, fmt="{:.1f}", suffix=" ms", color=None):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                fmt.format(height) + suffix,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                color=color or palette["accent"],
            )

    def ms_formatter(_, __):
        return "{:.0f} ms".format(_)

    whole_labels = ["community_run_only", "params_once_run"]
    vta_vals = [
        WHOLE_GRAPH["community_run_only_ms"]["vta"],
        WHOLE_GRAPH["params_once_run_ms"]["vta"],
    ]
    cpu_vals = [
        WHOLE_GRAPH["community_run_only_ms"]["cpu"],
        WHOLE_GRAPH["params_once_run_ms"]["cpu"],
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = range(len(whole_labels))
    width = 0.35
    bars_vta = ax.bar(
        [i - width / 2 for i in x],
        vta_vals,
        width=width,
        label="VTA",
        color=palette["vta"],
    )
    bars_cpu = ax.bar(
        [i + width / 2 for i in x],
        cpu_vals,
        width=width,
        label="CPU",
        color=palette["cpu"],
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Community run-only", "Params-once run"])
    ax.set_ylabel("ms")
    ax.set_title("AXU5EVB ResNet-18 Whole-Graph Runtime")
    ax.yaxis.set_major_formatter(FuncFormatter(ms_formatter))
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    annotate_bars(ax, bars_vta)
    annotate_bars(ax, bars_cpu)
    ratio = WHOLE_GRAPH["community_run_only_ms"]["cpu"] / WHOLE_GRAPH["community_run_only_ms"]["vta"]
    ax.text(
        0.02,
        0.95,
        "Community benchmark: VTA is {:.2f}x faster than CPU".format(ratio),
        transform=ax.transAxes,
        fontsize=11,
        color=palette["accent"],
        va="top",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "whole_graph_compare.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    labels = [item["layer"] for item in CONV_KERNEL_MS]
    vta_vals = [item["vta"] for item in CONV_KERNEL_MS]
    cpu_vals = [item["cpu"] for item in CONV_KERNEL_MS]
    x = range(len(labels))
    bars_vta = ax.bar(
        [i - width / 2 for i in x],
        vta_vals,
        width=width,
        label="VTA",
        color=palette["vta"],
    )
    bars_cpu = ax.bar(
        [i + width / 2 for i in x],
        cpu_vals,
        width=width,
        label="CPU",
        color=palette["cpu"],
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("ms")
    ax.set_title("Representative Single-Conv Kernel Compare")
    ax.yaxis.set_major_formatter(FuncFormatter(ms_formatter))
    ax.grid(axis="y")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    annotate_bars(ax, bars_vta, fmt="{:.2f}")
    annotate_bars(ax, bars_cpu, fmt="{:.2f}")
    fig.tight_layout()
    fig.savefig(output_dir / "conv_kernel_compare.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    op_labels = [name.replace("tvmgen_default_fused_", "") for name, _ in HOT_OPS_MS]
    op_vals = [val for _, val in HOT_OPS_MS]
    bars = ax.barh(op_labels, op_vals, color=palette["run"])
    ax.set_xlabel("ms")
    ax.set_title("Top VTA Hot Ops")
    ax.grid(axis="x")
    ax.xaxis.set_major_formatter(FuncFormatter(ms_formatter))
    for bar in bars:
        width_val = bar.get_width()
        ax.annotate(
            "{:.2f} ms".format(width_val),
            xy=(width_val, bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=10,
            color=palette["accent"],
        )
    fig.tight_layout()
    fig.savefig(output_dir / "vta_hot_ops.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    stage = WHOLE_GRAPH["vta_stage_avg_ms"]
    stage_labels = ["set_params", "set_data", "run", "get_output"]
    stage_vals = [stage[label] for label in stage_labels]
    stage_colors = [palette["memory"], "#F59E0B", palette["run"], palette["output"]]
    bars = ax.bar(stage_labels, stage_vals, color=stage_colors)
    ax.set_title("VTA Stage Breakdown")
    ax.set_ylabel("ms")
    ax.yaxis.set_major_formatter(FuncFormatter(ms_formatter))
    ax.grid(axis="y")
    annotate_bars(ax, bars, fmt="{:.1f}")
    total = stage["total"]
    memory_share = 100.0 * (stage["set_params"] + stage["set_data"] + stage["get_output"]) / total
    ax.text(
        0.02,
        0.95,
        "Memory-side share: {:.1f}% of total".format(memory_share),
        transform=ax.transAxes,
        fontsize=11,
        color=palette["accent"],
        va="top",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "vta_stage_breakdown.png", dpi=160)
    plt.close(fig)

    print("[plot] wrote PNG charts to", output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="vta/tutorials/frontend/report_out",
        help="Directory for CSV/PNG outputs",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path("/home/orange/code/tvm") / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    whole_rows = [
        [
            "community_run_only",
            f'{WHOLE_GRAPH["community_run_only_ms"]["vta"]:.3f}',
            f'{WHOLE_GRAPH["community_run_only_ms"]["cpu"]:.3f}',
            f'{WHOLE_GRAPH["community_run_only_ms"]["cpu"] / WHOLE_GRAPH["community_run_only_ms"]["vta"]:.2f}',
        ],
        [
            "params_once_run",
            f'{WHOLE_GRAPH["params_once_run_ms"]["vta"]:.3f}',
            f'{WHOLE_GRAPH["params_once_run_ms"]["cpu"]:.3f}',
            f'{WHOLE_GRAPH["params_once_run_ms"]["cpu"] / WHOLE_GRAPH["params_once_run_ms"]["vta"]:.2f}',
        ],
    ]
    conv_rows = []
    for item in CONV_KERNEL_MS:
        faster = "VTA" if item["vta"] < item["cpu"] else "CPU"
        ratio = max(item["vta"], item["cpu"]) / min(item["vta"], item["cpu"])
        conv_rows.append(
            [
                item["layer"],
                item["shape"],
                f'{item["vta"]:.4f}',
                f'{item["cpu"]:.4f}',
                faster,
                f"{ratio:.2f}",
            ]
        )

    stage_rows = [
        [key, f"{val:.3f}"]
        for key, val in WHOLE_GRAPH["vta_stage_avg_ms"].items()
    ]
    hot_rows = [[name, f"{val:.5f}"] for name, val in HOT_OPS_MS]

    print("\n# Whole-Graph Compare")
    print(markdown_table(["metric", "vta_ms", "cpu_ms", "cpu_over_vta"], whole_rows))

    print("\n# Single-Conv Kernel Compare")
    print(
        markdown_table(
            ["layer", "shape", "vta_ms", "cpu_ms", "faster", "speedup"],
            conv_rows,
        )
    )

    print("\n# VTA Stage Avg")
    print(markdown_table(["stage", "ms"], stage_rows))

    print("\n# VTA Hot Ops")
    print(markdown_table(["op", "ms"], hot_rows))

    write_csv(output_dir / "whole_graph_compare.csv", ["metric", "vta_ms", "cpu_ms", "cpu_over_vta"], whole_rows)
    write_csv(
        output_dir / "single_conv_kernel_compare.csv",
        ["layer", "shape", "vta_ms", "cpu_ms", "faster", "speedup"],
        conv_rows,
    )
    write_csv(output_dir / "vta_stage_avg.csv", ["stage", "ms"], stage_rows)
    write_csv(output_dir / "vta_hot_ops.csv", ["op", "ms"], hot_rows)

    try_plot(output_dir)
    print("[report] wrote CSV files to", output_dir)


if __name__ == "__main__":
    main()
