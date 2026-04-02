#!/usr/bin/env python3
"""Convert ResNet18 single-op benchmark CSV into a memory-focused report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="CSV produced by benchmark_resnet18_single_ops.py")
    parser.add_argument(
        "--output-dir",
        default="vta/tutorials/frontend/report_out",
        help="Directory for derived CSV/Markdown outputs",
    )
    parser.add_argument(
        "--worth-threshold",
        type=float,
        default=1.05,
        help="cpu_over_vta_total above this threshold is classified as worth offloading",
    )
    parser.add_argument(
        "--not-worth-threshold",
        type=float,
        default=0.95,
        help="cpu_over_vta_total below this threshold is classified as not worth offloading",
    )
    return parser.parse_args()


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def as_float(value: str) -> float:
    if value == "":
        return 0.0
    return float(value)


def load_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open() as f:
        for raw in csv.DictReader(f):
            row = dict(raw)
            row["implemented"] = as_bool(raw["implemented"])
            row["kernel_ms"] = as_float(raw["kernel_ms"])
            row["total_ms"] = as_float(raw["total_ms"])
            row["submit_ms"] = as_float(raw["submit_ms"])
            row["sync_ms"] = as_float(raw["sync_ms"])
            row["h2d_ms"] = as_float(raw["h2d_ms"])
            row["d2h_ms"] = as_float(raw["d2h_ms"])
            row["pack_ms"] = as_float(raw["pack_ms"])
            row["gops"] = as_float(raw["gops"])
            row["copy_ms"] = row["pack_ms"] + row["h2d_ms"] + row["d2h_ms"]
            row["memory_over_kernel"] = (
                row["copy_ms"] / row["kernel_ms"] if row["kernel_ms"] else float("inf")
            )
            row["copy_share_pct"] = (
                100.0 * row["copy_ms"] / row["total_ms"] if row["total_ms"] else 0.0
            )
            rows.append(row)
    return rows


def write_csv(path: Path, headers: List[str], rows: Iterable[Iterable[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def implemented_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [row for row in rows if row["implemented"] and row["status"] == "ok"]


def top_rows(rows: List[Dict[str, object]], sort_key: str) -> List[List[object]]:
    ordered = sorted(rows, key=lambda row: row[sort_key], reverse=True)
    return [
        [
            row["device"],
            row["case_id"],
            f'{row["kernel_ms"]:.4f}',
            f'{row["copy_ms"]:.4f}',
            f'{row["total_ms"]:.4f}',
            f'{row["copy_share_pct"]:.1f}',
            f'{row["memory_over_kernel"]:.2f}',
        ]
        for row in ordered
    ]


def pair_by_case(rows: List[Dict[str, object]]) -> List[List[object]]:
    grouped: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), {})[str(row["device"])] = row

    paired: List[List[object]] = []
    for case_id in sorted(grouped):
        devices = grouped[case_id]
        cpu = devices.get("arm_cpu")
        vta = devices.get("vta")
        if not cpu or not vta:
            continue
        cpu_kernel = cpu["kernel_ms"]
        vta_kernel = vta["kernel_ms"]
        cpu_copy = cpu["copy_ms"]
        vta_copy = vta["copy_ms"]
        cpu_total = cpu["total_ms"]
        vta_total = vta["total_ms"]
        paired.append(
            [
                case_id,
                f"{cpu_kernel:.4f}",
                f"{vta_kernel:.4f}",
                f"{(cpu_kernel / vta_kernel) if vta_kernel else 0.0:.2f}",
                f"{cpu_copy:.4f}",
                f"{vta_copy:.4f}",
                f"{(cpu_copy / vta_copy) if vta_copy else 0.0:.2f}",
                f"{cpu_total:.4f}",
                f"{vta_total:.4f}",
                f"{(cpu_total / vta_total) if vta_total else 0.0:.2f}",
            ]
        )
    return paired


def summary_rows(rows: List[Dict[str, object]]) -> List[List[object]]:
    out = []
    for device in sorted({str(row["device"]) for row in rows}):
        dev_rows = [row for row in rows if row["device"] == device]
        if not dev_rows:
            continue
        count = len(dev_rows)
        kernel_avg = sum(row["kernel_ms"] for row in dev_rows) / count
        copy_avg = sum(row["copy_ms"] for row in dev_rows) / count
        total_avg = sum(row["total_ms"] for row in dev_rows) / count
        copy_share_avg = sum(row["copy_share_pct"] for row in dev_rows) / count
        out.append(
            [
                device,
                count,
                f"{kernel_avg:.4f}",
                f"{copy_avg:.4f}",
                f"{total_avg:.4f}",
                f"{copy_share_avg:.1f}",
            ]
        )
    return out


def conv_type_label(row: Dict[str, object]) -> str:
    if row["op_name"] != "conv2d":
        return str(row["op_name"])
    return "{}x{}_s{}".format(
        int(row["kernel_h"]), int(row["kernel_w"]), int(row["stride_h"])
    )


def spatial_bucket_label(row: Dict[str, object]) -> str:
    return "{}x{}".format(int(row["height"]), int(row["width"]))


def threshold_label(cpu_over_vta_total: float, worth_threshold: float, not_worth_threshold: float) -> str:
    if cpu_over_vta_total >= worth_threshold:
        return "worth_offloading"
    if cpu_over_vta_total <= not_worth_threshold:
        return "not_worth_offloading"
    return "borderline"


def paired_case_dicts(rows: List[Dict[str, object]], worth_threshold: float, not_worth_threshold: float):
    grouped: Dict[str, Dict[str, Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), {})[str(row["device"])] = row

    paired = []
    for case_id in sorted(grouped):
        devices = grouped[case_id]
        cpu = devices.get("arm_cpu")
        vta = devices.get("vta")
        if not cpu or not vta:
            continue
        record = {
            "case_id": case_id,
            "conv_type": conv_type_label(cpu),
            "spatial_bucket": spatial_bucket_label(cpu),
            "cpu_kernel_ms": cpu["kernel_ms"],
            "vta_kernel_ms": vta["kernel_ms"],
            "cpu_copy_ms": cpu["copy_ms"],
            "vta_copy_ms": vta["copy_ms"],
            "cpu_total_ms": cpu["total_ms"],
            "vta_total_ms": vta["total_ms"],
            "cpu_over_vta_kernel": (cpu["kernel_ms"] / vta["kernel_ms"]) if vta["kernel_ms"] else 0.0,
            "cpu_over_vta_copy": (cpu["copy_ms"] / vta["copy_ms"]) if vta["copy_ms"] else 0.0,
            "cpu_over_vta_total": (cpu["total_ms"] / vta["total_ms"]) if vta["total_ms"] else 0.0,
            "vta_copy_share_pct": vta["copy_share_pct"],
        }
        record["offload_decision"] = threshold_label(
            record["cpu_over_vta_total"], worth_threshold, not_worth_threshold
        )
        paired.append(record)
    return paired


def grouped_threshold_rows(records: List[Dict[str, object]], group_key: str) -> List[List[object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record[group_key]), []).append(record)
    rows = []
    for key in sorted(grouped):
        items = grouped[key]
        count = len(items)
        avg_cpu_over_vta_total = sum(item["cpu_over_vta_total"] for item in items) / count
        avg_vta_copy_share = sum(item["vta_copy_share_pct"] for item in items) / count
        worth = sum(1 for item in items if item["offload_decision"] == "worth_offloading")
        borderline = sum(1 for item in items if item["offload_decision"] == "borderline")
        not_worth = sum(1 for item in items if item["offload_decision"] == "not_worth_offloading")
        rows.append(
            [
                key,
                count,
                f"{avg_cpu_over_vta_total:.2f}",
                f"{avg_vta_copy_share:.1f}",
                worth,
                borderline,
                not_worth,
            ]
        )
    return rows


def decision_table_rows(records: List[Dict[str, object]]) -> List[List[object]]:
    ordered = sorted(records, key=lambda item: item["vta_total_ms"])
    rows = []
    for item in ordered:
        rows.append(
            [
                item["case_id"],
                item["conv_type"],
                item["spatial_bucket"],
                f"{item['cpu_total_ms']:.4f}",
                f"{item['vta_total_ms']:.4f}",
                f"{item['cpu_over_vta_total']:.2f}",
                f"{item['vta_copy_share_pct']:.1f}",
                item["offload_decision"],
            ]
        )
    return rows


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path("/home/orange/code/tvm") / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_csv)
    ok_rows = implemented_rows(rows)

    kernel_rank = top_rows(ok_rows, "kernel_ms")
    copy_rank = top_rows(ok_rows, "copy_ms")
    total_rank = top_rows(ok_rows, "total_ms")
    paired = pair_by_case(ok_rows)
    summary = summary_rows(ok_rows)
    paired_dicts = paired_case_dicts(ok_rows, args.worth_threshold, args.not_worth_threshold)
    spatial_summary = grouped_threshold_rows(paired_dicts, "spatial_bucket")
    conv_summary = grouped_threshold_rows(paired_dicts, "conv_type")
    decision_rows = decision_table_rows(paired_dicts)

    print("\n# Device Summary")
    print(
        markdown_table(
            ["device", "count", "avg_kernel_ms", "avg_copy_ms", "avg_total_ms", "avg_copy_share_pct"],
            summary,
        )
    )

    print("\n# Kernel-Dominant Ranking")
    print(
        markdown_table(
            ["device", "case", "kernel_ms", "copy_ms", "total_ms", "copy_share_pct", "copy_over_kernel"],
            kernel_rank,
        )
    )

    print("\n# Copy-Dominant Ranking")
    print(
        markdown_table(
            ["device", "case", "kernel_ms", "copy_ms", "total_ms", "copy_share_pct", "copy_over_kernel"],
            copy_rank,
        )
    )

    print("\n# Total-Latency Ranking")
    print(
        markdown_table(
            ["device", "case", "kernel_ms", "copy_ms", "total_ms", "copy_share_pct", "copy_over_kernel"],
            total_rank,
        )
    )

    print("\n# CPU vs VTA Comparison")
    print(
        markdown_table(
            [
                "case",
                "cpu_kernel_ms",
                "vta_kernel_ms",
                "cpu_over_vta_kernel",
                "cpu_copy_ms",
                "vta_copy_ms",
                "cpu_over_vta_copy",
                "cpu_total_ms",
                "vta_total_ms",
                "cpu_over_vta_total",
            ],
            paired,
        )
    )

    stem = input_csv.stem
    write_csv(
        output_dir / f"{stem}_summary.csv",
        ["device", "count", "avg_kernel_ms", "avg_copy_ms", "avg_total_ms", "avg_copy_share_pct"],
        summary,
    )
    common_headers = ["device", "case", "kernel_ms", "copy_ms", "total_ms", "copy_share_pct", "copy_over_kernel"]
    write_csv(output_dir / f"{stem}_kernel_rank.csv", common_headers, kernel_rank)
    write_csv(output_dir / f"{stem}_copy_rank.csv", common_headers, copy_rank)
    write_csv(output_dir / f"{stem}_total_rank.csv", common_headers, total_rank)
    write_csv(output_dir / "copy_dominant_rank.csv", common_headers, copy_rank)
    write_csv(output_dir / "total_latency_rank.csv", common_headers, total_rank)
    write_csv(
        output_dir / f"{stem}_cpu_vta_compare.csv",
        [
            "case",
            "cpu_kernel_ms",
            "vta_kernel_ms",
            "cpu_over_vta_kernel",
            "cpu_copy_ms",
            "vta_copy_ms",
            "cpu_over_vta_copy",
            "cpu_total_ms",
            "vta_total_ms",
            "cpu_over_vta_total",
        ],
        paired,
    )
    write_csv(
        output_dir / "cpu_vta_compare.csv",
        [
            "case",
            "cpu_kernel_ms",
            "vta_kernel_ms",
            "cpu_over_vta_kernel",
            "cpu_copy_ms",
            "vta_copy_ms",
            "cpu_over_vta_copy",
            "cpu_total_ms",
            "vta_total_ms",
            "cpu_over_vta_total",
        ],
        paired,
    )
    write_csv(
        output_dir / f"{stem}_spatial_threshold_summary.csv",
        [
            "spatial_bucket",
            "count",
            "avg_cpu_over_vta_total",
            "avg_vta_copy_share_pct",
            "worth_offloading",
            "borderline",
            "not_worth_offloading",
        ],
        spatial_summary,
    )
    write_csv(
        output_dir / f"{stem}_convtype_threshold_summary.csv",
        [
            "conv_type",
            "count",
            "avg_cpu_over_vta_total",
            "avg_vta_copy_share_pct",
            "worth_offloading",
            "borderline",
            "not_worth_offloading",
        ],
        conv_summary,
    )
    write_csv(
        output_dir / f"{stem}_threshold_decision.csv",
        [
            "case",
            "conv_type",
            "spatial_bucket",
            "cpu_total_ms",
            "vta_total_ms",
            "cpu_over_vta_total",
            "vta_copy_share_pct",
            "offload_decision",
        ],
        decision_rows,
    )
    write_csv(
        output_dir / "threshold_decision.csv",
        [
            "case",
            "conv_type",
            "spatial_bucket",
            "cpu_total_ms",
            "vta_total_ms",
            "cpu_over_vta_total",
            "vta_copy_share_pct",
            "offload_decision",
        ],
        decision_rows,
    )

    report_path = output_dir / f"{stem}_single_op_threshold_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Device Summary",
                markdown_table(
                    ["device", "count", "avg_kernel_ms", "avg_copy_ms", "avg_total_ms", "avg_copy_share_pct"],
                    summary,
                ),
                "",
                "# Kernel-Dominant Ranking",
                markdown_table(
                    ["device", "case", "kernel_ms", "copy_ms", "total_ms", "copy_share_pct", "copy_over_kernel"],
                    kernel_rank,
                ),
                "",
                "# Copy-Dominant Ranking",
                markdown_table(
                    ["device", "case", "kernel_ms", "copy_ms", "total_ms", "copy_share_pct", "copy_over_kernel"],
                    copy_rank,
                ),
                "",
                "# Total-Latency Ranking",
                markdown_table(
                    ["device", "case", "kernel_ms", "copy_ms", "total_ms", "copy_share_pct", "copy_over_kernel"],
                    total_rank,
                ),
                "",
                "# CPU vs VTA Comparison",
                markdown_table(
                    [
                        "case",
                        "cpu_kernel_ms",
                        "vta_kernel_ms",
                        "cpu_over_vta_kernel",
                        "cpu_copy_ms",
                        "vta_copy_ms",
                        "cpu_over_vta_copy",
                        "cpu_total_ms",
                        "vta_total_ms",
                        "cpu_over_vta_total",
                    ],
                    paired,
                ),
                "",
                "# Offload Decision Table",
                markdown_table(
                    [
                        "case",
                        "conv_type",
                        "spatial_bucket",
                        "cpu_total_ms",
                        "vta_total_ms",
                        "cpu_over_vta_total",
                        "vta_copy_share_pct",
                        "offload_decision",
                    ],
                    decision_rows,
                ),
                "",
                "# Spatial Bucket Summary",
                markdown_table(
                    [
                        "spatial_bucket",
                        "count",
                        "avg_cpu_over_vta_total",
                        "avg_vta_copy_share_pct",
                        "worth_offloading",
                        "borderline",
                        "not_worth_offloading",
                    ],
                    spatial_summary,
                ),
                "",
                "# Conv Type Summary",
                markdown_table(
                    [
                        "conv_type",
                        "count",
                        "avg_cpu_over_vta_total",
                        "avg_vta_copy_share_pct",
                        "worth_offloading",
                        "borderline",
                        "not_worth_offloading",
                    ],
                    conv_summary,
                ),
                "",
                "# Threshold Interpretation",
                "",
                "- `worth_offloading`: `cpu_over_vta_total >= {:.2f}`".format(args.worth_threshold),
                "- `not_worth_offloading`: `cpu_over_vta_total <= {:.2f}`".format(args.not_worth_threshold),
                "- `borderline`: between the two thresholds.",
                "",
                "This report treats `total_ms` as the single-op latency target. `copy_ms` is used to explain when tile/copy cost starts to erode VTA gains.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "single_op_threshold_report.md").write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"\n[report] wrote derived files to {output_dir}")


if __name__ == "__main__":
    main()
