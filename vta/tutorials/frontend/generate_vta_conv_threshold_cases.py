#!/usr/bin/env python3
"""Generate representative conv2d benchmark cases for VTA tile-copy threshold studies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPRESENTATIVE_CASES = [
    {
        "case_id": "thr_s1_conv3x3_64_64_56",
        "op_name": "conv2d",
        "batch": 1,
        "channels_in": 64,
        "height": 56,
        "width": 56,
        "channels_out": 64,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 1,
        "stride_w": 1,
        "pad_h": 1,
        "pad_w": 1,
        "note": "3x3 stride1 large spatial",
    },
    {
        "case_id": "thr_s2_down3x3_64_128_56",
        "op_name": "conv2d",
        "batch": 1,
        "channels_in": 64,
        "height": 56,
        "width": 56,
        "channels_out": 128,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 2,
        "stride_w": 2,
        "pad_h": 1,
        "pad_w": 1,
        "note": "3x3 stride2 stage2 downsample",
    },
    {
        "case_id": "thr_s2_proj1x1_64_128_56",
        "op_name": "conv2d",
        "batch": 1,
        "channels_in": 64,
        "height": 56,
        "width": 56,
        "channels_out": 128,
        "kernel_h": 1,
        "kernel_w": 1,
        "stride_h": 2,
        "stride_w": 2,
        "pad_h": 0,
        "pad_w": 0,
        "note": "1x1 stride2 shortcut",
    },
    {
        "case_id": "thr_s2_conv3x3_128_128_28",
        "op_name": "conv2d",
        "batch": 1,
        "channels_in": 128,
        "height": 28,
        "width": 28,
        "channels_out": 128,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 1,
        "stride_w": 1,
        "pad_h": 1,
        "pad_w": 1,
        "note": "3x3 stride1 stage2 main path",
    },
    {
        "case_id": "thr_s3_down3x3_128_256_28",
        "op_name": "conv2d",
        "batch": 1,
        "channels_in": 128,
        "height": 28,
        "width": 28,
        "channels_out": 256,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 2,
        "stride_w": 2,
        "pad_h": 1,
        "pad_w": 1,
        "note": "3x3 stride2 stage3 downsample",
    },
    {
        "case_id": "thr_s3_proj1x1_128_256_28",
        "op_name": "conv2d",
        "batch": 1,
        "channels_in": 128,
        "height": 28,
        "width": 28,
        "channels_out": 256,
        "kernel_h": 1,
        "kernel_w": 1,
        "stride_h": 2,
        "stride_w": 2,
        "pad_h": 0,
        "pad_w": 0,
        "note": "1x1 stride2 stage3 shortcut",
    },
    {
        "case_id": "thr_s3_conv3x3_256_256_14",
        "op_name": "conv2d",
        "batch": 1,
        "channels_in": 256,
        "height": 14,
        "width": 14,
        "channels_out": 256,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 1,
        "stride_w": 1,
        "pad_h": 1,
        "pad_w": 1,
        "note": "3x3 stride1 stage3 main path",
    },
    {
        "case_id": "thr_s4_down3x3_256_512_14",
        "op_name": "conv2d",
        "batch": 1,
        "channels_in": 256,
        "height": 14,
        "width": 14,
        "channels_out": 512,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 2,
        "stride_w": 2,
        "pad_h": 1,
        "pad_w": 1,
        "note": "3x3 stride2 stage4 downsample",
    },
    {
        "case_id": "thr_s4_proj1x1_256_512_14",
        "op_name": "conv2d",
        "batch": 1,
        "channels_in": 256,
        "height": 14,
        "width": 14,
        "channels_out": 512,
        "kernel_h": 1,
        "kernel_w": 1,
        "stride_h": 2,
        "stride_w": 2,
        "pad_h": 0,
        "pad_w": 0,
        "note": "1x1 stride2 stage4 shortcut",
    },
    {
        "case_id": "thr_s4_conv3x3_512_512_7",
        "op_name": "conv2d",
        "batch": 1,
        "channels_in": 512,
        "height": 7,
        "width": 7,
        "channels_out": 512,
        "kernel_h": 3,
        "kernel_w": 3,
        "stride_h": 1,
        "stride_w": 1,
        "pad_h": 1,
        "pad_w": 1,
        "note": "3x3 stride1 stage4 main path",
    },
]


INTERPOLATION_CASES = [
    (112, 64, 64, 3, 1, "interp_3x3_s1_64_64_112", "Large spatial interpolation"),
    (56, 128, 128, 3, 1, "interp_3x3_s1_128_128_56", "Channel-up interpolation"),
    (56, 64, 128, 1, 2, "interp_1x1_s2_64_128_56", "Shortcut interpolation"),
    (28, 256, 256, 3, 1, "interp_3x3_s1_256_256_28", "Mid spatial interpolation"),
    (28, 128, 256, 3, 2, "interp_3x3_s2_128_256_28", "Stage3-like interpolation"),
    (14, 512, 512, 3, 1, "interp_3x3_s1_512_512_14", "High channel interpolation"),
    (14, 256, 512, 1, 2, "interp_1x1_s2_256_512_14", "Shortcut interpolation high channel"),
    (7, 256, 256, 3, 1, "interp_3x3_s1_256_256_7", "Small spatial low channel"),
    (7, 512, 512, 1, 1, "interp_1x1_s1_512_512_7", "1x1 small spatial"),
    (28, 64, 64, 3, 1, "interp_3x3_s1_64_64_28", "Reduced spatial baseline"),
    (14, 128, 128, 3, 1, "interp_3x3_s1_128_128_14", "Reduced spatial channel baseline"),
    (56, 64, 64, 1, 2, "interp_1x1_s2_64_64_56", "Stride2 1x1 low channel"),
]


def build_interpolation_case(height, cin, cout, kernel, stride, case_id, note):
    pad = 1 if kernel == 3 else 0
    return {
        "case_id": case_id,
        "op_name": "conv2d",
        "batch": 1,
        "channels_in": cin,
        "height": height,
        "width": height,
        "channels_out": cout,
        "kernel_h": kernel,
        "kernel_w": kernel,
        "stride_h": stride,
        "stride_w": stride,
        "pad_h": pad,
        "pad_w": pad,
        "note": note,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="vta/tutorials/frontend/report_out/vta_conv_threshold_cases.csv",
        help="Output CSV or JSON path",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format; defaults to csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = list(REPRESENTATIVE_CASES)
    rows.extend(build_interpolation_case(*item) for item in INTERPOLATION_CASES)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path("/home/orange/code/tvm") / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "json" or out_path.suffix.lower() == ".json":
        out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    else:
        fieldnames = [
            "case_id",
            "op_name",
            "batch",
            "channels_in",
            "height",
            "width",
            "channels_out",
            "kernel_h",
            "kernel_w",
            "stride_h",
            "stride_w",
            "pad_h",
            "pad_w",
            "note",
        ]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print("[casegen] wrote {} cases to {}".format(len(rows), out_path))


if __name__ == "__main__":
    main()
