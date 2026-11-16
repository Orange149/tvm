#!/usr/bin/env python3
"""Validate frozen fusion-aware predictions against later full-build truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fusion_aware_transfer import DMA_KEYS, digest_value, read_json, sha256_file


HOST_KEYS = ("cpu_read_bytes", "cpu_write_bytes", "cpu_helper_calls", "convert_calls")
BYTE_KEYS = (
    "load_buffer_2d_inp_bytes",
    "load_buffer_2d_wgt_bytes",
    "load_buffer_2d_acc_bytes",
    "store_buffer_2d_out_bytes",
)


def truth_rows(census, verification):
    helpers = {row["segment_id"]: row for row in verification["rows"]}
    result = {}
    for row in census["rows"]:
        segment_id = row["segment_id"]
        helper = helpers[segment_id]
        if not helper["cpu_logical_access_totals_complete"]:
            raise ValueError("incomplete CPU truth for {}".format(segment_id))
        convert_calls = sum(
            int(item["invocations"])
            for item in helper["cpu_helpers"]
            if item["status"] == "static_logical_accesses" and (
                set(item["ops"]) <= {"cast", "reshape", "transpose", "multiply", "round", "clip"}
                or bool(set(item["ops"]) & {"reshape", "transpose", "multiply", "round", "clip"})
            )
        )
        result[segment_id] = {
            "dma": {key: int(row["logical_descriptor_totals"].get(key, 0)) for key in DMA_KEYS},
            "host": {
                "cpu_read_bytes": int(helper["cpu_logical_access_totals"].get("read_bytes", 0)),
                "cpu_write_bytes": int(helper["cpu_logical_access_totals"].get("write_bytes", 0)),
                "cpu_helper_calls": sum(int(item["invocations"]) for item in helper["cpu_helpers"]),
                "convert_calls": convert_calls,
            },
        }
    return result


def compare(predictions, truth):
    rows = []
    for prediction in predictions:
        segment_id = prediction["segment_id"]
        if segment_id not in truth:
            continue
        expected = truth[segment_id]
        dma_errors = {
            key: int(prediction["total_dma"].get(key, 0)) - expected["dma"][key]
            for key in DMA_KEYS
        }
        host_errors = {
            key: int(prediction["components"]["host"].get(key, 0)) - expected["host"][key]
            for key in HOST_KEYS
        }
        predicted_bytes = sum(int(prediction["total_dma"].get(key, 0)) for key in BYTE_KEYS)
        truth_bytes = sum(expected["dma"][key] for key in BYTE_KEYS)
        relative_error = abs(predicted_bytes - truth_bytes) / truth_bytes if truth_bytes else float(predicted_bytes != 0)
        exact = not any(dma_errors.values()) and not any(host_errors.values())
        rows.append({
            "segment_id": segment_id,
            "prediction_status": prediction["status"],
            "prediction_usable_for_ranking": prediction["usable_for_ranking"],
            "exact": exact,
            "dma_errors": dma_errors,
            "host_errors": host_errors,
            "total_dma_byte_relative_error": relative_error,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--truth-census", type=Path, required=True)
    parser.add_argument("--truth-verification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-exact-fraction", type=float, default=0.95)
    parser.add_argument("--maximum-mape", type=float, default=0.01)
    parser.add_argument("--allow-truth-subset", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    args.output_dir.mkdir(parents=True)
    payload = read_json(args.predictions)
    predictions = payload["rows"]
    if "prediction_sha256" in payload and digest_value(predictions) != payload["prediction_sha256"]:
        raise ValueError("frozen prediction hash mismatch")
    truth = truth_rows(read_json(args.truth_census), read_json(args.truth_verification))
    prediction_ids = {row["segment_id"] for row in predictions}
    if not args.allow_truth_subset and prediction_ids != set(truth):
        raise ValueError("prediction/truth segment sets differ")
    rows = compare(predictions, truth)
    if not rows:
        raise ValueError("no overlapping prediction/truth segments")
    exact_fraction = sum(row["exact"] for row in rows) / len(rows)
    mape = sum(row["total_dma_byte_relative_error"] for row in rows) / len(rows)
    fail_closed_ok = all(
        row["prediction_status"] == "exact_reuse" or not row["prediction_usable_for_ranking"]
        for row in rows
    )
    known_class_silent_errors = [
        row["segment_id"] for row in rows
        if row["prediction_status"] == "exact_reuse" and not row["exact"]
    ]
    gates = {
        "exact_fraction": exact_fraction >= args.minimum_exact_fraction,
        "mape": mape <= args.maximum_mape,
        "fail_closed": fail_closed_ok,
        "no_known_class_silent_error": not known_class_silent_errors,
    }
    result = {
        "schema": "vta_fixed_tile_fusion_transfer_validation_v1",
        "status": "pass" if all(gates.values()) else "fail",
        "segments_compared": len(rows),
        "exact_segments": sum(row["exact"] for row in rows),
        "exact_fraction": exact_fraction,
        "total_dma_byte_mape": mape,
        "known_class_silent_errors": known_class_silent_errors,
        "gates": gates,
        "thresholds": {
            "minimum_exact_fraction": args.minimum_exact_fraction,
            "maximum_mape": args.maximum_mape,
        },
        "rows": rows,
        "sources": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in (
            ("predictions", args.predictions),
            ("truth_census", args.truth_census),
            ("truth_verification", args.truth_verification),
        )},
    }
    (args.output_dir / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "STATUS.md").write_text(
        "# C1 fusion-transfer validation\n\n"
        "- Status: `{}`.\n"
        "- Compared: {} segments; exact: {} ({:.2%}).\n"
        "- Total-DMA-byte MAPE: {:.4%}.\n"
        "- Known-class silent errors: {}.\n".format(
            result["status"], len(rows), result["exact_segments"], exact_fraction,
            mape, len(known_class_silent_errors)
        )
    )
    print(json.dumps({key: result[key] for key in (
        "status", "segments_compared", "exact_segments", "exact_fraction", "total_dma_byte_mape", "gates"
    )}, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
