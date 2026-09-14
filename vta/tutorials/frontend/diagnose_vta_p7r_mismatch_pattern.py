#!/usr/bin/env python3
"""Collect one correctness-only FPGA mismatch map for a frozen P7R config."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from tvm import rpc
import tvm
import vta

from qualify_vta_residency_fsim import array_sha256, reference_data_from_workload
from run_vta_p7r_joint_correctness import build_module, load_contract, sha256_file, ssh_preflight


def axis_counts(mask, axis):
    reduce_axes = tuple(index for index in range(mask.ndim) if index != axis)
    return [int(value) for value in np.sum(mask, axis=reduce_axes)]


def mismatch_samples(actual, expected, limit=32):
    rows = []
    for location in np.argwhere(actual != expected)[:limit]:
        index = tuple(int(value) for value in location)
        rows.append(
            {
                "index": list(index),
                "expected": int(expected[index]),
                "actual": int(actual[index]),
            }
        )
    return rows


def shifted_matches(actual, expected):
    rows = []
    for delta_h in range(-2, 3):
        for delta_w in range(-2, 3):
            h0 = max(0, delta_h)
            h1 = expected.shape[2] + min(0, delta_h)
            w0 = max(0, delta_w)
            w1 = expected.shape[3] + min(0, delta_w)
            lhs = actual[:, :, h0:h1, w0:w1, :, :]
            rhs = expected[
                :,
                :,
                h0 - delta_h : h1 - delta_h,
                w0 - delta_w : w1 - delta_w,
                :,
                :,
            ]
            rows.append(
                {
                    "delta_h": delta_h,
                    "delta_w": delta_w,
                    "equal_fraction": float(np.mean(lhs == rhs)),
                }
            )
    return sorted(rows, key=lambda row: -row["equal_fraction"])[:5]


def channel_block_matches(actual, expected):
    rows = []
    for delta in range(expected.shape[1]):
        shifted = np.roll(expected, delta, axis=1)
        rows.append(
            {"co_block_roll": delta, "equal_fraction": float(np.mean(actual == shifted))}
        )
    return sorted(rows, key=lambda row: -row["equal_fraction"])[:5]


def packed_intermediate_outputs(workload, data, weight):
    """Return the int8 out-buffer image after GEMM and each ALU instruction."""

    _, _, strides, padding, _, _, _ = workload[1:]
    batch_outer, ci_outer, height, width, batch_inner, block_in = data.shape
    co_outer, _, kernel_h, kernel_w, block_out, _ = weight.shape
    nchw = data.transpose(0, 4, 1, 5, 2, 3).reshape(
        batch_outer * batch_inner, ci_outer * block_in, height, width
    )
    kernel = weight.transpose(0, 4, 1, 5, 2, 3).reshape(
        co_outer * block_out, ci_outer * block_in, kernel_h, kernel_w
    )
    if len(padding) == 2:
        pad_top, pad_left = padding
        pad_bottom, pad_right = pad_top, pad_left
    else:
        pad_top, pad_left, pad_bottom, pad_right = padding
    padded = np.pad(
        nchw.astype("int32"),
        ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
    )
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (kernel_h, kernel_w), axis=(2, 3)
    )[:, :, :: strides[0], :: strides[1], :, :]
    accum = np.einsum(
        "ncyxij,ocij->noyx", windows, kernel.astype("int32"), optimize=True
    )

    def low_int8(value):
        return np.ascontiguousarray((value.astype("int64") & 255).astype("uint8").view("int8"))

    def pack(value):
        output_h, output_w = value.shape[2:]
        return np.ascontiguousarray(
            value.reshape(batch_outer, batch_inner, co_outer, block_out, output_h, output_w)
            .transpose(0, 2, 4, 5, 1, 3)
        )

    shifted = accum >> 8
    clipped_high = np.minimum(shifted, 127)
    clipped_both = np.maximum(clipped_high, 0)
    return {
        "gemm_low8_before_alu": pack(low_int8(accum)),
        "after_shift": pack(low_int8(shifted)),
        "after_min_127": pack(low_int8(clipped_high)),
        "after_max_0_final": pack(low_int8(clipped_both)),
    }


def stage_match_summary(actual, stages):
    result = {}
    for name, value in stages.items():
        result[name] = {
            "total_equal_fraction": float(np.mean(actual == value)),
            "equal_fraction_by_co_block": [
                float(np.mean(actual[:, block] == value[:, block]))
                for block in range(actual.shape[1])
            ],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--config-index", type=int, required=True)
    parser.add_argument("--mode", choices=("original", "input_stationary"), default="original")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefill", type=int, default=-113)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    contract = load_contract(args.contract)
    before = ssh_preflight(contract, args.host)
    env = vta.get_env()
    remote = rpc.connect(args.host, args.port, session_timeout=120)
    device = remote.ext_dev(0)
    with tempfile.TemporaryDirectory(prefix="c3_p7r_mismatch_map_") as temporary:
        binary, tir_hash, config = build_module(
            contract["workload"], args.mode, args.config_index, env, Path(temporary)
        )
        binary_hash = sha256_file(binary)
        remote.upload(str(binary))
        function = remote.load_module(binary.name)["main"]
        data, weight, expected = reference_data_from_workload(contract["workload"], args.seed)
        stages = packed_intermediate_outputs(contract["workload"], data, weight)
        out = tvm.nd.array(np.full(expected.shape, args.prefill, dtype="int8"), device)
        function(tvm.nd.array(data, device), tvm.nd.array(weight, device), out)
        actual = out.numpy()
    after = ssh_preflight(contract, args.host)
    mask = actual != expected
    coordinates = np.argwhere(mask)
    value_histogram = Counter(int(value) for value in actual[mask])
    summary = {
        "schema": "c3_p7r_fpga_mismatch_pattern_v1",
        "scope": "single-seed exact-output diagnostic; no latency",
        "workload_id": contract["workload_id"],
        "config_index": args.config_index,
        "mode": args.mode,
        "seed": args.seed,
        "prefill": args.prefill,
        "config": config,
        "tir_sha256": tir_hash,
        "binary_sha256": binary_hash,
        "shape": list(expected.shape),
        "mismatch_count": int(mask.sum()),
        "mismatch_fraction": float(mask.mean()),
        "prefill_remaining_count": int(np.count_nonzero(actual == args.prefill)),
        "expected_prefill_count": int(np.count_nonzero(expected == args.prefill)),
        "actual_sha256": array_sha256(actual),
        "expected_sha256": array_sha256(expected),
        "mismatch_bbox_min": coordinates.min(axis=0).tolist() if coordinates.size else None,
        "mismatch_bbox_max": coordinates.max(axis=0).tolist() if coordinates.size else None,
        "mismatch_counts_by_axis": {
            "batch_outer": axis_counts(mask, 0),
            "co_block": axis_counts(mask, 1),
            "height": axis_counts(mask, 2),
            "width": axis_counts(mask, 3),
            "batch_inner": axis_counts(mask, 4),
            "co_inner": axis_counts(mask, 5),
        },
        "mismatch_actual_value_histogram": {
            str(key): value for key, value in sorted(value_histogram.items())
        },
        "mismatch_samples": mismatch_samples(actual, expected),
        "best_spatial_shift_matches": shifted_matches(actual, expected),
        "best_co_block_roll_matches": channel_block_matches(actual, expected),
        "hardware_pipeline_stage_matches": stage_match_summary(actual, stages),
        "board_before": before,
        "board_after": after,
        "performance_measurement": "not_collected",
    }
    result = output / "mismatch_pattern.json"
    result.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "STATUS.md").write_text(
        "# P7R FPGA mismatch-pattern diagnostic\n\n"
        "- Status: `completed`\n"
        "- Workload/config: `{}/{}`\n"
        "- Mismatches: `{}` of `{}` ({:.2%})\n"
        "- Prefill remaining: `{}`\n"
        "- Performance timing: not collected\n".format(
            contract["workload_id"], args.config_index, int(mask.sum()), mask.size,
            float(mask.mean()), int(np.count_nonzero(actual == args.prefill))
        ),
        encoding="utf-8",
    )
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": artifacts}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: summary[key] for key in (
        "mismatch_count", "mismatch_fraction", "prefill_remaining_count",
        "mismatch_bbox_min", "mismatch_bbox_max")}, indent=2))


if __name__ == "__main__":
    main()
