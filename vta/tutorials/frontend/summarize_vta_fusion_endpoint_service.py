#!/usr/bin/env python3
"""Aggregate C1 endpoint service measurements from independent board boots."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from fusion_aware_transfer import read_json, sha256_file
from measure_vta_fusion_endpoint_service_boot import SEGMENT_IDS, TRIPLETS


SERVICE_FIELDS = (
    "stage_service_ms",
    "vta_driver_service_ms",
    "host_helper_plus_graph_remainder_ms",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def seal(payload):
    result = dict(payload)
    result["artifact_sha256"] = hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()
    return result


def contract_identity(contract):
    """Hash protocol-invariant fields while excluding the cold-boot snapshot."""
    normalized = dict(contract)
    normalized.pop("boot_id", None)
    normalized.pop("board_before", None)
    return hashlib.sha256(canonical(normalized).encode("utf-8")).hexdigest()


def expected_sequence(repeat):
    sequence = []
    for case in range(3):
        for repeat_index in range(repeat):
            offset = (case + repeat_index) % len(SEGMENT_IDS)
            order = SEGMENT_IDS[offset:] + SEGMENT_IDS[:offset]
            sequence.extend((segment_id, case, repeat_index) for segment_id in order)
    return sequence


def load_boot(directory):
    directory = Path(directory)
    summary = read_json(directory / "summary.json")
    if summary.get("schema") != "vta_c1_endpoint_service_boot_v1":
        raise ValueError("unexpected boot summary schema: {}".format(directory))
    if summary.get("status") != "complete":
        raise ValueError("incomplete boot run: {}".format(directory))
    sample_path = directory / "samples.jsonl"
    if sha256_file(sample_path) != summary["sample_file_sha256"]:
        raise ValueError("sample hash mismatch: {}".format(directory))
    contract = read_json(directory / "contract.json")
    if contract.get("schema") != "vta_c1_endpoint_service_boot_contract_v1":
        raise ValueError("unexpected boot contract schema: {}".format(directory))
    if sha256_file(directory / "contract.json") != summary["contract_sha256"]:
        raise ValueError("contract hash mismatch: {}".format(directory))
    if contract.get("boot_id") != summary.get("boot_id"):
        raise ValueError("summary/contract boot-id mismatch: {}".format(directory))
    if contract.get("segments") != list(SEGMENT_IDS):
        raise ValueError("contract segment order mismatch: {}".format(directory))
    if contract.get("triplets") != [list(row) for row in TRIPLETS]:
        raise ValueError("contract triplet mismatch: {}".format(directory))
    if contract.get("input_cases") != [0, 1, 2] or int(contract.get("repeat", 0)) != 3:
        raise ValueError("contract is not the frozen 3-case x 3-repeat protocol: {}".format(directory))
    for name in ("board_before", "board_after"):
        if summary.get(name, {}).get("boot") != summary["boot_id"]:
            raise ValueError("{} boot snapshot mismatch: {}".format(name, directory))
    if contract.get("board_before") != summary.get("board_before"):
        raise ValueError("contract/summary board-before mismatch: {}".format(directory))
    samples = [json.loads(line) for line in sample_path.read_text().splitlines() if line.strip()]
    expected_keys = {
        (segment_id, case, repeat)
        for segment_id in SEGMENT_IDS
        for case in range(3)
        for repeat in range(int(contract["repeat"]))
    }
    actual_keys = {(row["segment_id"], row["input_case"], row["repeat"]) for row in samples}
    if actual_keys != expected_keys or len(samples) != len(expected_keys):
        raise ValueError("boot does not contain the registered 81-sample set: {}".format(directory))
    if summary.get("samples") != len(samples) or summary.get("expected_samples") != len(expected_keys):
        raise ValueError("summary sample count mismatch: {}".format(directory))
    ordered = sorted(samples, key=lambda row: row["sequence"])
    if [row["sequence"] for row in ordered] != list(range(len(ordered))):
        raise ValueError("sample sequence is not contiguous: {}".format(directory))
    observed_sequence = [
        (row["segment_id"], row["input_case"], row["repeat"]) for row in ordered
    ]
    if observed_sequence != expected_sequence(int(contract["repeat"])):
        raise ValueError("sample order differs from the registered rotation: {}".format(directory))
    if any(row["boot_id"] != summary["boot_id"] for row in samples):
        raise ValueError("sample boot-id mismatch: {}".format(directory))
    if any(row.get("schema") != "vta_c1_endpoint_service_sample_v1" for row in samples):
        raise ValueError("unexpected sample schema: {}".format(directory))
    if any(row["dma_counter_differences"] for row in samples):
        raise ValueError("timed sample has a DMA mismatch: {}".format(directory))
    for row in samples:
        values = [float(row[field]) for field in SERVICE_FIELDS]
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("invalid service value: {}".format(directory))
        if abs(values[0] - values[1] - values[2]) > 1.0e-9:
            raise ValueError("service decomposition does not close: {}".format(directory))
        if int(row.get("profiler_invocations", 0)) != 2:
            raise ValueError("unexpected profiler invocation count: {}".format(directory))
        expected_output = contract["independent_llvm_reference_output_sha256"][
            row["segment_id"]
        ][str(row["input_case"])]
        if row["output_sha256"] != expected_output:
            raise ValueError("sample output differs from frozen LLVM reference: {}".format(directory))
    hashes = defaultdict(set)
    for row in samples:
        hashes[row["segment_id"], row["input_case"]].add(tuple(row["output_sha256"]))
    if any(len(values) != 1 for values in hashes.values()):
        raise ValueError("non-deterministic full output within a boot: {}".format(directory))
    return summary, contract, samples


def median_fields(rows):
    return {field: statistics.median(float(row[field]) for row in rows) for field in SERVICE_FIELDS}


def subtract(left, right):
    return {field: left[field] - right[field] for field in SERVICE_FIELDS}


def summarize_runs(runs):
    if len(runs) != 3:
        raise ValueError("E2 service aggregation requires exactly three cold-boot runs")

    loaded = [load_boot(directory) for directory in runs]
    boot_ids = [summary["boot_id"] for summary, _contract, _samples in loaded]
    if len(set(boot_ids)) != 3:
        raise ValueError("E2 runs must come from three distinct board boots")
    contract_identities = [contract_identity(contract) for _summary, contract, _samples in loaded]
    if len(set(contract_identities)) != 1:
        raise ValueError("E2 runs do not share one frozen protocol/artifact contract")
    cross_boot_hashes = defaultdict(set)
    for _summary, _contract, samples in loaded:
        for row in samples:
            cross_boot_hashes[row["segment_id"], row["input_case"]].add(
                tuple(row["output_sha256"])
            )
    if any(len(values) != 1 for values in cross_boot_hashes.values()):
        raise ValueError("full endpoint outputs differ across cold boots")

    by_segment_boot = defaultdict(list)
    for summary, _contract, samples in loaded:
        for row in samples:
            by_segment_boot[row["segment_id"], summary["boot_id"]].append(row)
    segment_results = {}
    for segment_id in SEGMENT_IDS:
        boot_medians = {
            boot_id: median_fields(by_segment_boot[segment_id, boot_id])
            for boot_id in boot_ids
        }
        segment_results[segment_id] = {
            "boot_medians": boot_medians,
            "cold_boot_median": {
                field: statistics.median(values[field] for values in boot_medians.values())
                for field in SERVICE_FIELDS
            },
        }

    triplets = []
    for base_id, projection_id, tail_id in TRIPLETS:
        base = segment_results[base_id]["cold_boot_median"]
        projection = segment_results[projection_id]["cold_boot_median"]
        tail = segment_results[tail_id]["cold_boot_median"]
        triplets.append({
            "segments": [base_id, projection_id, tail_id],
            "base_to_projection_service_delta_ms": subtract(projection, base),
            "projection_to_cpu_tail_service_delta_ms": subtract(tail, projection),
        })

    result = seal({
        "schema": "vta_c1_endpoint_service_three_boot_summary_v1",
        "status": "complete",
        "boot_ids": boot_ids,
        "statistical_unit": "independent cold boot",
        "service_scope": {
            "stage_service_ms": "measured full GraphExecutor stage",
            "vta_driver_service_ms": "VTA driver profiler component",
            "host_helper_plus_graph_remainder_ms": (
                "difference proxy; includes graph dispatch and must not be called pure CPU helper time"
            ),
        },
        "segments": segment_results,
        "triplets": triplets,
        "interpretation": (
            "use deltas to test resource attribution; do not require lower transfer to imply higher FPS"
        ),
        "contract_identity_sha256": contract_identities[0],
        "sources": [
            {
                "directory": str(Path(directory).resolve()),
                "summary_sha256": sha256_file(Path(directory) / "summary.json"),
                "contract_sha256": sha256_file(Path(directory) / "contract.json"),
                "samples_sha256": sha256_file(Path(directory) / "samples.jsonl"),
            }
            for directory in runs
        ],
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__)),
        },
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output_dir))
    result = summarize_runs(args.runs)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "service_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": result["status"],
        "boot_ids": result["boot_ids"],
        "triplets": result["triplets"],
        "artifact_sha256": result["artifact_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
