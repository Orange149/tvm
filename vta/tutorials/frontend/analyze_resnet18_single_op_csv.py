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

    report_path = output_dir / f"{stem}_memory_report.md"
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
            ]
        ),
        encoding="utf-8",
    )
    print(f"\n[report] wrote derived files to {output_dir}")


if __name__ == "__main__":
    main()
