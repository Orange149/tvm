#!/usr/bin/env python3
"""Route VTA residency candidates through static and exact hardware certificates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_vta_hardware_certificate_ledger import (
    HARDWARE_CERTIFICATE_LEDGER_SCHEMA,
    canonical_hash,
    certificate_identity,
    hardware_fingerprint,
    sha256_file,
)


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def dma_pareto(candidate):
    original = candidate["original"]["dma"]
    residency = candidate[candidate["residence_mode"]]["dma"]
    before_bytes = original["load_buffer_2d_bytes"] + original["store_buffer_2d_bytes"]
    after_bytes = residency["load_buffer_2d_bytes"] + residency["store_buffer_2d_bytes"]
    before_calls = original["load_buffer_2d_calls"] + original["store_buffer_2d_calls"]
    after_calls = residency["load_buffer_2d_calls"] + residency["store_buffer_2d_calls"]
    reductions = {
        "input_bytes": candidate["input_dma_reduction_fraction"],
        "total_dma_bytes": (before_bytes - after_bytes) / before_bytes,
        "total_dma_calls": (before_calls - after_calls) / before_calls,
    }
    return {
        "certified": reductions["input_bytes"] > 0
        and reductions["total_dma_bytes"] >= 0
        and reductions["total_dma_calls"] >= 0,
        "reductions": reductions,
    }


def route_candidate(candidate, contract, certificates):
    result = {
        "candidate_id": candidate["candidate_id"],
        "workload_id": candidate["workload_id"],
        "mode": candidate["residence_mode"],
        "config_index": candidate["config_index"],
        "tir_sha256": candidate["tir_sha256"],
        "complete_config_entity": candidate["identity"]["complete_config_entity"],
    }
    if not candidate.get("locally_eligible"):
        result.update(
            route="reject_static",
            reason=candidate.get("reject_reason"),
            dma_pareto={"certified": False, "reductions": None},
        )
        return result
    pareto = dma_pareto(candidate)
    result["dma_pareto"] = pareto
    if not pareto["certified"]:
        result.update(route="abstain_keep_original", reason="not_dma_pareto")
        return result
    probe = {
        "mode": candidate["residence_mode"],
        "config_index": candidate["config_index"],
        "complete_config_entity": candidate["identity"]["complete_config_entity"],
        "tir_sha256": candidate["tir_sha256"],
    }
    key, _ = certificate_identity(contract, probe, hardware_fingerprint(contract)["key"])
    result["hardware_certificate_key"] = key
    certificate = certificates.get(key)
    if certificate is None:
        result.update(route="correctness_canary_only", reason="unknown_exact_hardware_identity")
    elif certificate["status"] == "passed":
        result.update(route="allow_timing", reason="exact_hardware_certificate_passed")
    else:
        result.update(route="reject_hardware", reason="exact_hardware_certificate_failed")
    return result


def dispatch_plan(candidates, contract, ledger):
    if ledger.get("schema") != HARDWARE_CERTIFICATE_LEDGER_SCHEMA:
        raise ValueError(
            "hardware certificate ledger must use {}; refusing legacy or unversioned ledger".format(
                HARDWARE_CERTIFICATE_LEDGER_SCHEMA
            )
        )
    certificates = {row["certificate_key"]: row for row in ledger["certificates"]}
    routed = [route_candidate(row, contract, certificates) for row in candidates]
    priority = {
        "protected_incumbent": 0,
        "allow_timing": 1,
        "correctness_canary_only": 2,
        "abstain_keep_original": 3,
        "reject_hardware": 4,
        "reject_static": 5,
    }
    records = [{
        "route": "protected_incumbent",
        "reason": "zero-regret fallback is dispatched before experimental candidates",
        "workload_id": contract["workload_id"],
    }] + sorted(routed, key=lambda row: (priority[row["route"]], row["config_index"]))
    counts = {}
    for row in records:
        counts[row["route"]] = counts.get(row["route"], 0) + 1
    return records, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    paths = {name: Path(value).resolve() for name, value in (
        ("candidates", args.candidates), ("contract", args.contract), ("ledger", args.ledger)
    )}
    contract = json.loads(paths["contract"].read_text())
    all_candidates = load_jsonl(paths["candidates"])
    candidates = [row for row in all_candidates if row["workload_id"] == contract["workload_id"]]
    if not candidates:
        raise ValueError("candidate file has no rows for contract workload")
    ledger = json.loads(paths["ledger"].read_text())
    records, counts = dispatch_plan(candidates, contract, ledger)
    result = {
        "schema": "c3_vta_hardware_certified_dispatch_v2",
        "status": "completed",
        "performance_labels_used": False,
        "policy_order": [
            "protected incumbent",
            "static and DMA-Pareto certificate",
            "exact hardware-certificate lookup",
            "unknown identities receive correctness only",
            "timing only after exact pass",
        ],
        "counts": counts,
        "records": records,
        "inputs": {name: {"path": str(path), "sha256": sha256_file(path)}
                   for name, path in paths.items()},
    }
    result["dispatch_id"] = canonical_hash({key: value for key, value in result.items() if key != "dispatch_id"})
    plan_path = output / "dispatch.json"
    plan_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "STATUS.md").write_text(
        "# Hardware-certified dispatch plan\n\n"
        "- Performance labels used for routing: no\n"
        "- Route counts: `{}`\n"
        "- Only `allow_timing` entries may enter a latency cost model.\n".format(
            json.dumps(counts, sort_keys=True)
        )
    )
    hashes = {path.name: sha256_file(path) for path in output.iterdir()
              if path.is_file() and path.name != "artifact_hashes.json"}
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": hashes}, indent=2) + "\n"
    )
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
