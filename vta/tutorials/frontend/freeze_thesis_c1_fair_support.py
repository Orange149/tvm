#!/usr/bin/env python3
"""Freeze the thesis C1 8-topology/16-configuration label-free support and orders."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from shared_memory_topk import pareto_layers


SCHEMA = "thesis_c1_fair_support_input_v1"
OUTPUT_SCHEMA = "thesis_c1_fair_support_contract_v1"
SEMANTIC_DIMENSIONS = (
    "framework_materialization_bytes",
    "vta_dma_calls",
    "vta_internal_dma_bytes",
    "additional_slot_bytes",
    "vta_islands",
)
RANDOM_SEEDS = tuple(range(58001, 58021))
FORBIDDEN_LABEL_FRAGMENTS = (
    "observed_ii",
    "measured_ii",
    "board_latency",
    "performance_label",
    "throughput_label",
    "oracle_rank",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_label_free(value, path="root"):
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = key.lower()
            if key != "new_performance_labels_read" and any(
                fragment in normalized for fragment in FORBIDDEN_LABEL_FRAGMENTS
            ):
                raise ValueError("forbidden new performance label field at {}.{}".format(path, key))
            validate_label_free(nested, "{}.{}".format(path, key))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            validate_label_free(nested, "{}[{}]".format(path, index))


def validate_candidate(row):
    required = {
        "candidate_id",
        "topology_id",
        "m1_score_ms",
        "vta_islands",
        "stage_devices",
        "thread_vector",
        "static_legal",
        "realized_policy",
        *SEMANTIC_DIMENSIONS[:-1],
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError("candidate missing fields: {}".format(", ".join(missing)))
    if not row["candidate_id"] or not row["topology_id"]:
        raise ValueError("candidate/topology identity must be non-empty")
    if row["static_legal"] is not True:
        raise ValueError("input pool must remove statically illegal candidates before freezing")
    if row["realized_policy"] != "ordinary_copy":
        raise ValueError("C1 fair support requires one ordinary-copy realization")
    score = float(row["m1_score_ms"])
    if not math.isfinite(score) or score <= 0:
        raise ValueError("invalid static M1 score")
    devices = list(row["stage_devices"])
    threads = [int(value) for value in row["thread_vector"]]
    if len(devices) != len(threads) or not devices or any(value <= 0 for value in threads):
        raise ValueError("invalid full stage thread vector")
    if any(device not in {"cpu", "vta"} for device in devices):
        raise ValueError("unknown stage device")
    if any(threads[index] != 1 for index, device in enumerate(devices) if device == "vta"):
        raise ValueError("VTA stages must retain one runtime thread")
    if int(row["vta_islands"]) not in {1, 2, 3}:
        raise ValueError("vta_islands must be one of 1/2/3")
    for field in SEMANTIC_DIMENSIONS[:-1]:
        value = row[field]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError("invalid semantic field {}".format(field))


def row_key(row):
    return float(row["m1_score_ms"]), row["candidate_id"]


def thread_target(devices, cpu_threads):
    return tuple(cpu_threads if device == "cpu" else 1 for device in devices)


def select_two_variants(topology_rows):
    ordered = sorted(topology_rows, key=row_key)
    best = ordered[0]
    devices = list(best["stage_devices"])
    if any(list(row["stage_devices"]) != devices for row in ordered):
        raise ValueError("stage device vector differs within one topology")
    target = thread_target(devices, 1)
    if tuple(best["thread_vector"]) == target:
        target = thread_target(devices, 4)
    matches = [row for row in ordered if tuple(row["thread_vector"]) == target]
    if len(matches) != 1:
        raise ValueError(
            "topology {} lacks exactly one prescribed second thread vector {}".format(
                best["topology_id"], target
            )
        )
    second = matches[0]
    if second["candidate_id"] == best["candidate_id"]:
        raise ValueError("topology thread variants collapse to one candidate")
    return best, second


def select_topologies(rows):
    by_topology = {}
    for row in rows:
        by_topology.setdefault(row["topology_id"], []).append(row)
    summaries = []
    for topology_id, variants in by_topology.items():
        best, second = select_two_variants(variants)
        summaries.append(
            {
                "topology_id": topology_id,
                "m1_score_ms": float(best["m1_score_ms"]),
                "candidate_id": best["candidate_id"],
                "vta_islands": int(best["vta_islands"]),
                "reasonable_a": any(row.get("reasonable_a") is True for row in variants),
                "best": best,
                "second": second,
            }
        )
    summaries.sort(key=lambda row: (row["m1_score_ms"], row["candidate_id"], row["topology_id"]))
    if len(summaries) < 8:
        raise ValueError("fewer than eight statically legal topologies")
    selected = []
    seen = set()

    def add(row):
        if row is not None and row["topology_id"] not in seen:
            selected.append(row)
            seen.add(row["topology_id"])

    add(summaries[0])
    add(next((row for row in summaries if row["reasonable_a"]), None))
    for islands in (1, 2, 3):
        add(next((row for row in summaries if row["vta_islands"] == islands), None))
    for row in summaries:
        if len(selected) == 8:
            break
        add(row)
    if len(selected) != 8:
        raise ValueError("static topology selection could not produce eight identities")
    return selected


def semantic_order(first_rows):
    ordered = sorted(first_rows, key=row_key)
    winner = ordered[0]
    limit = float(winner["m1_score_ms"]) * 1.02
    band = [row for row in ordered[1:] if float(row["m1_score_ms"]) <= limit + 1.0e-12]
    outside = [row for row in ordered[1:] if float(row["m1_score_ms"]) > limit + 1.0e-12]
    unknown = sorted(
        row["candidate_id"]
        for row in band
        if any(row.get(field) is None for field in SEMANTIC_DIMENSIONS)
    )
    result = [winner]
    layers = []
    if unknown:
        result.extend(sorted(band, key=row_key))
        fallback = True
    else:
        fallback = False
        for layer_index, layer in enumerate(pareto_layers(band, SEMANTIC_DIMENSIONS), 1):
            ids = []
            for row in layer:
                result.append(row)
                ids.append(row["candidate_id"])
            layers.append({"layer": layer_index, "candidate_ids": ids})
    result.extend(sorted(outside, key=row_key))
    return result, {
        "band_limit_ms": limit,
        "band_candidate_ids": [row["candidate_id"] for row in band],
        "unknown_candidate_ids": unknown,
        "fallback_to_m1": fallback,
        "pareto_layers": layers,
    }


def freeze(payload):
    if payload.get("schema") != SCHEMA:
        raise ValueError("unexpected fair-support input schema")
    if payload.get("new_performance_labels_read") is not False:
        raise ValueError("support must be frozen before reading new performance labels")
    validate_label_free(payload)
    rows = [dict(row) for row in payload.get("candidates", [])]
    if len({row.get("candidate_id") for row in rows}) != len(rows):
        raise ValueError("candidate IDs must be unique")
    for row in rows:
        validate_candidate(row)
    selected_topologies = select_topologies(rows)
    first = [row["best"] for row in selected_topologies]
    second = [row["second"] for row in selected_topologies]
    support = first + second
    support_ids = {row["candidate_id"] for row in support}
    if len(support) != 16 or len(support_ids) != 16:
        raise ValueError("selected support is not 16 unique full candidate identities")
    o0 = sorted(support, key=row_key)
    first_m1 = sorted(first, key=row_key)
    second_m1 = sorted(second, key=row_key)
    o1 = first_m1 + second_m1
    semantic_first, semantic_audit = semantic_order(first_m1)
    o2 = semantic_first + second_m1
    common_first = o1[0]
    random_orders = {}
    rest = sorted(
        row["candidate_id"]
        for row in support
        if row["candidate_id"] != common_first["candidate_id"]
    )
    for seed in RANDOM_SEEDS:
        permutation = np.random.Generator(np.random.PCG64(seed)).permutation(len(rest))
        random_orders[str(seed)] = [common_first["candidate_id"]] + [rest[int(i)] for i in permutation]
    orders = {
        "O0_m1": [row["candidate_id"] for row in o0],
        "O1_topology_first": [row["candidate_id"] for row in o1],
        "O2_semantic_band": [row["candidate_id"] for row in o2],
        "O3_random": random_orders,
    }
    all_orders = [orders["O0_m1"], orders["O1_topology_first"], orders["O2_semantic_band"]]
    all_orders.extend(random_orders.values())
    if any(len(order) != 16 or set(order) != support_ids for order in all_orders):
        raise AssertionError("an order does not preserve the exact support")
    if any(order[0] != common_first["candidate_id"] for order in all_orders):
        raise AssertionError("orders do not share the fixed first candidate")
    measurement_base = sorted(support_ids, key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value))
    measurement_orders = {}
    for boot_index, offset in enumerate((0, 5, 10)):
        rotation = offset % len(measurement_base)
        measurement_orders[str(boot_index)] = measurement_base[rotation:] + measurement_base[:rotation]
    topology_records = []
    for row in selected_topologies:
        topology_records.append(
            {
                "topology_id": row["topology_id"],
                "vta_islands": row["vta_islands"],
                "reasonable_a": row["reasonable_a"],
                "best_candidate_id": row["best"]["candidate_id"],
                "second_candidate_id": row["second"]["candidate_id"],
            }
        )
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "frozen_before_new_performance_labels",
        "label_free": True,
        "actual_policy": "ordinary_copy",
        "support_size": 16,
        "topology_count": 8,
        "topologies": topology_records,
        "support": support,
        "orders": orders,
        "semantic_audit": semantic_audit,
        "random_policy": {
            "generator": "numpy.random.PCG64",
            "seeds": list(RANDOM_SEEDS),
            "fixed_first_candidate_id": common_first["candidate_id"],
        },
        "measurement_orders": measurement_orders,
        "measurement_rotation_offsets": [0, 5, 10],
        "support_identity_sha256": canonical_hash(sorted(support_ids)),
        "claim_boundary": "label-free static support/order contract only; no board correctness or performance",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    input_path = Path(args.input).resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    result = freeze(payload)
    output.mkdir(parents=True)
    result["input"] = {"path": str(input_path), "sha256": sha256_file(input_path)}
    contract = output / "fair_support_contract.json"
    write_json(contract, result)
    write_json(
        output / "artifact_hashes.json",
        {"artifacts": {contract.name: sha256_file(contract)}},
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "support_size": result["support_size"],
                "topology_count": result["topology_count"],
                "support_identity_sha256": result["support_identity_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
