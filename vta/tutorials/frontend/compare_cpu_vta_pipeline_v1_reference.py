#!/usr/bin/env python3
"""Compare a dumped native ResNet18 logits tensor with the frozen P5A reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as inp:
        for chunk in iter(lambda: inp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference(path):
    reference_path = Path(path).resolve()
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    tensor_path = reference_path.parent / payload["tensor_file"]
    if file_sha256(tensor_path) != payload["tensor_sha256"]:
        raise ValueError("reference tensor SHA256 does not match its manifest")
    dtype = np.dtype(payload["dtype"])
    shape = tuple(int(value) for value in payload["shape"])
    tensor = np.fromfile(tensor_path, dtype=dtype)
    if tensor.size != int(np.prod(shape)):
        raise ValueError("reference tensor element count does not match its shape")
    return payload, tensor.reshape(shape)


def compare_tensors(reference, actual, gate):
    expected = np.asarray(reference, dtype="float64").reshape(-1)
    observed = np.asarray(actual, dtype="float64").reshape(-1)
    result = {
        "shape_match": tuple(reference.shape) == tuple(actual.shape),
        "finite": bool(np.isfinite(observed).all()),
    }
    if not result["shape_match"] or not result["finite"]:
        result.update({"passed": False, "failed_checks": [key for key, value in result.items() if not value]})
        return result

    delta = observed - expected
    ref_rms = float(np.sqrt(np.mean(expected * expected)))
    rmse = float(np.sqrt(np.mean(delta * delta)))
    denominator = float(np.linalg.norm(expected) * np.linalg.norm(observed))
    cosine = float(np.dot(expected, observed) / denominator) if denominator > 0.0 else 0.0
    top_k = int(gate["top_k"])
    expected_top = set(np.argsort(expected)[-top_k:].tolist())
    observed_top = set(np.argsort(observed)[-top_k:].tolist())
    result.update(
        {
            "mae": float(np.mean(np.abs(delta))),
            "max_abs_error": float(np.max(np.abs(delta))),
            "rmse": rmse,
            "normalized_rmse": rmse / max(ref_rms, 1.0e-12),
            "cosine_similarity": cosine,
            "top1_reference": int(np.argmax(expected)),
            "top1_actual": int(np.argmax(observed)),
            "top1_match": int(np.argmax(expected)) == int(np.argmax(observed)),
            "top_k": top_k,
            "top_k_overlap": len(expected_top & observed_top),
        }
    )
    checks = {
        "shape_match": result["shape_match"],
        "finite": result["finite"],
        "cosine_similarity": result["cosine_similarity"] >= float(gate["min_cosine_similarity"]),
        "normalized_rmse": result["normalized_rmse"] <= float(gate["max_normalized_rmse"]),
        "top_k_overlap": result["top_k_overlap"] >= int(gate["min_top_k_overlap"]),
    }
    result["checks"] = checks
    result["failed_checks"] = [key for key, value in checks.items() if not value]
    result["passed"] = not result["failed_checks"]
    return result


def compare_files(reference_json, actual_bin):
    payload, expected = load_reference(reference_json)
    actual_path = Path(actual_bin)
    actual = np.fromfile(actual_path, dtype=np.dtype(payload["dtype"]))
    expected_count = int(np.prod(expected.shape))
    if actual.size != expected_count:
        return {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_p5_tensor_comparison",
            "reference_json": str(Path(reference_json).resolve()),
            "actual_file": str(actual_path.resolve()),
            "actual_sha256": file_sha256(actual_path),
            "actual_element_count": int(actual.size),
            "expected_element_count": expected_count,
            "passed": False,
            "failed_checks": ["element_count"],
        }
    result = compare_tensors(expected, actual.reshape(expected.shape), payload["qualification_gate"])
    return {
        "schema_version": 1,
        "kind": "cpu_vta_pipeline_v1_p5_tensor_comparison",
        "reference_json": str(Path(reference_json).resolve()),
        "reference_tensor_sha256": payload["tensor_sha256"],
        "actual_file": str(actual_path.resolve()),
        "actual_sha256": file_sha256(actual_path),
        **result,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-json", required=True)
    parser.add_argument("--actual-bin", required=True)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    result = compare_files(args.reference_json, args.actual_bin)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
