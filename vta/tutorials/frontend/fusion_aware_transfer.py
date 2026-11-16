#!/usr/bin/env python3
"""Pure helpers for fixed-tile, fusion-aware VTA transfer accounting.

The model deliberately describes logical compiler-visible work.  It does not
claim that logical bytes are physical DDR transactions or predict latency.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


DMA_KEYS = (
    "load_buffer_2d_calls",
    "load_buffer_2d_bytes",
    "load_buffer_2d_small_calls",
    "load_buffer_2d_strided_calls",
    "load_buffer_2d_inp_calls",
    "load_buffer_2d_inp_bytes",
    "load_buffer_2d_wgt_calls",
    "load_buffer_2d_wgt_bytes",
    "load_buffer_2d_acc_calls",
    "load_buffer_2d_acc_bytes",
    "store_buffer_2d_calls",
    "store_buffer_2d_bytes",
    "store_buffer_2d_small_calls",
    "store_buffer_2d_strided_calls",
    "store_buffer_2d_out_calls",
    "store_buffer_2d_out_bytes",
)

CONVERSION_OPS = {"cast", "reshape", "transpose", "multiply", "round", "clip"}
ENVIRONMENT_FIELDS = (
    "git_head",
    "sources",
    "target",
    "host",
    "tophub",
    "quantize",
    "disabled_passes",
)
STATUS_PRIORITY = (
    "environment_mismatch",
    "new_workload",
    "new_fusion_class",
    "unsupported_dynamic_dma",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest_value(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def workload_key(workload, config_index):
    return digest_value({"workload": workload, "config_index": int(config_index)})


def weight_key(weight):
    return canonical({key: weight[key] for key in ("shape", "dtype", "data_sha256")})


def environment_contract(provenance):
    contract = {key: provenance.get(key) for key in ENVIRONMENT_FIELDS}
    return {"fields": contract, "sha256": digest_value(contract)}


def zero_dma():
    return {key: 0 for key in DMA_KEYS}


def add_vectors(*vectors):
    keys = set().union(*(vector for vector in vectors)) if vectors else set()
    return {key: sum(vector.get(key, 0) for vector in vectors) for key in sorted(keys)}


def subtract_vectors(left, right):
    return {key: left.get(key, 0) - right.get(key, 0) for key in sorted(set(left) | set(right))}


def dma_vector(totals):
    return {key: int(totals.get(key, 0)) for key in DMA_KEYS}


def dtype_bytes(dtype):
    digits = "".join(character for character in str(dtype) if character.isdigit())
    if not digits:
        raise ValueError("dtype has no bit width: {}".format(dtype))
    bits = int(digits)
    if bits % 8:
        raise ValueError("sub-byte dtype is unsupported: {}".format(dtype))
    return bits // 8


def contract_bytes(contract):
    total = 0
    for slot in contract.get("slots", []):
        total += math.prod(int(extent) for extent in slot["shape"]) * dtype_bytes(slot["dtype"])
    return total


def is_conversion_helper(ops):
    ops = set(ops)
    if not ops:
        return False
    return ops <= CONVERSION_OPS or bool(ops & (CONVERSION_OPS - {"cast"}))


def _scan_rows(scan_dir):
    summary = read_json(Path(scan_dir) / "summary.json")
    if not summary.get("complete") or len(summary.get("rows", [])) != 87:
        raise ValueError("scan must be the completed 87-segment audit")
    return summary


def _workload_records(scan, static_workloads):
    static = {
        workload_key(row["workload"], row["config_index"]): row
        for row in static_workloads["rows"]
    }
    records = {}
    signature_workloads = defaultdict(set)
    for segment in scan["rows"]:
        for conv in segment["convolutions"]:
            if conv.get("is_fallback"):
                raise ValueError("fallback workload in {}".format(segment["segment_id"]))
            key = workload_key(conv["workload"], conv["config"]["index"])
            if key not in static:
                raise ValueError("missing static DMA workload {}".format(key))
            signature_workloads[conv["primitive_signature"]].add(key)
            candidate = {
                "workload_key": key,
                "workload": conv["workload"],
                "config": conv["config"],
                "tile_identity_sha256": digest_value(conv["config"]),
                "base_dma": dma_vector(static[key]["static_dma"]["totals"]),
                "capacity_qualification": "existing_static_lowering",
            }
            previous = records.setdefault(key, candidate)
            if previous != candidate:
                raise ValueError("inconsistent fixed-tile workload {}".format(key))
    ambiguous = {signature: sorted(keys) for signature, keys in signature_workloads.items() if len(keys) != 1}
    if ambiguous:
        raise ValueError("primitive signature maps to multiple fixed workloads: {}".format(ambiguous))
    return records, {signature: next(iter(keys)) for signature, keys in signature_workloads.items()}


def build_model(*, scan_dir, archive_dir, census_path, verification_path, workloads_path):
    """Build a strict calibration model from the 23 representative segments."""
    scan_dir, archive_dir = Path(scan_dir), Path(archive_dir)
    scan = _scan_rows(scan_dir)
    archive = read_json(archive_dir / "summary.json")
    census = read_json(census_path)
    verification = read_json(verification_path)
    static_workloads = read_json(workloads_path)
    if not archive.get("complete") or len(archive.get("rows", [])) != 23:
        raise ValueError("archive must contain 23 complete representative builds")
    archive_segments = [row["segment_id"] for row in archive["rows"]]
    if set(archive_segments) != {row["segment_id"] for row in census["rows"]}:
        raise ValueError("census/archive segment mismatch")
    if set(archive_segments) != {row["segment_id"] for row in verification["rows"]}:
        raise ValueError("verification/archive segment mismatch")

    provenance = read_json(scan_dir / "provenance.json")
    archive_provenance = read_json(archive_dir / "provenance.json")
    scan_environment = environment_contract(provenance)
    # The representative archive deliberately references the richer scan
    # provenance instead of duplicating it.  Bind that reference and the
    # independently repeated target/host fields before inheriting the scan
    # environment contract.
    if archive_provenance.get("scan_provenance_sha256") != sha256_file(scan_dir / "provenance.json"):
        raise ValueError("representative archive is not bound to the scan provenance")
    for field in ("target", "host"):
        if archive_provenance.get(field) != provenance.get(field):
            raise ValueError("scan and representative {} differ".format(field))
    profile_source = "vta/tutorials/frontend/profile_split_resnet18_stages.py"
    if archive_provenance.get("profile_builder_sha256") != provenance.get("sources", {}).get(profile_source):
        raise ValueError("representative profile builder differs from scan source")

    workloads, signature_workload = _workload_records(scan, static_workloads)
    scan_by_segment = {row["segment_id"]: row for row in scan["rows"]}
    census_by_segment = {row["segment_id"]: row for row in census["rows"]}
    verification_by_segment = {row["segment_id"]: row for row in verification["rows"]}
    observations = defaultdict(list)

    for segment_id in archive_segments:
        local = archive_dir / segment_id.replace(":", "_")
        inventory = read_json(local / "primitive_inventory.json")
        graph = read_json(local / "graph.json")
        by_hash = {row["compiler_hash"]: row for row in inventory}
        function_to_inventory = {}
        calls = Counter()
        for node in graph["nodes"]:
            if node["op"] != "tvm_op":
                continue
            function = node["attrs"]["func_name"]
            item = by_hash[node["attrs"]["hash"]]
            calls[function] += 1
            previous = function_to_inventory.setdefault(function, item)
            if previous["signature"] != item["signature"]:
                raise ValueError("compiler function has multiple signatures")

        census_functions = census_by_segment[segment_id]["functions"]
        helper_rows = {row["function"]: row for row in verification_by_segment[segment_id]["cpu_helpers"]}
        scan_weights = {weight_key(conv["weight"]): conv for conv in scan_by_segment[segment_id]["convolutions"]}
        if set(census_functions) != set(function_to_inventory):
            raise ValueError("function coverage mismatch in {}".format(segment_id))

        for function, item in function_to_inventory.items():
            counted = census_functions[function]
            if int(counted["invocations"]) != calls[function]:
                raise ValueError("invocation mismatch in {}:{}".format(segment_id, function))
            base_keys = []
            for weight in item["weights"]:
                conv = scan_weights[weight_key(weight)]
                base_keys.append(workload_key(conv["workload"], conv["config"]["index"]))
            base = zero_dma()
            for key in base_keys:
                base = add_vectors(base, workloads[key]["base_dma"])
            full = dma_vector(counted["static_dma"]["totals"])
            helper = helper_rows.get(function)
            host = {
                "cpu_read_bytes": int((helper or {}).get("accesses_per_invocation", {}).get("read_bytes", 0)),
                "cpu_write_bytes": int((helper or {}).get("accesses_per_invocation", {}).get("write_bytes", 0)),
                "cpu_helper_calls": int(helper is not None),
                "convert_calls": int(helper is not None and is_conversion_helper(item["ops"])),
            }
            supported = helper is None or helper.get("status") == "static_logical_accesses"
            observations[item["signature"]].append({
                "segment_id": segment_id,
                "function": function,
                "ops": item["ops"],
                "base_workload_keys": base_keys,
                "full_dma": full,
                "fusion_residual_dma": subtract_vectors(full, base),
                "host": host,
                "supported": supported,
            })

    classes = {}
    for signature, rows in sorted(observations.items()):
        comparable = {
            canonical({key: row[key] for key in ("ops", "base_workload_keys", "full_dma", "fusion_residual_dma", "host", "supported")})
            for row in rows
        }
        if len(comparable) != 1:
            raise ValueError("fusion class is not invariant: {}".format(signature))
        record = {key: rows[0][key] for key in (
            "ops", "base_workload_keys", "full_dma", "fusion_residual_dma", "host", "supported"
        )}
        record["signature"] = signature
        record["calibration_occurrences"] = len(rows)
        record["calibration_segments"] = sorted({row["segment_id"] for row in rows})
        classes[signature] = record

    expected_signatures = {
        primitive["signature"]
        for row in scan["rows"]
        for primitive in read_json(scan_dir / row["segment_id"].replace(":", "_") / "primitives.json")
    }
    missing = sorted(expected_signatures - set(classes))
    if missing:
        raise ValueError("23 representatives do not cover {} primitive classes".format(len(missing)))

    return {
        "schema": "vta_fixed_tile_fusion_transfer_model_v1",
        "scope": "logical compiler-visible transfer and CPU helper accesses; not physical DDR or latency",
        "environment": scan_environment,
        "runtime_boundary_policy": "set_get_copy_once_per_standalone_segment_contract",
        "calibration_segments": sorted(archive_segments),
        "workloads": workloads,
        "signature_workload": signature_workload,
        "fusion_classes": classes,
        "coverage": {
            "domain_segments": len(scan["rows"]),
            "calibration_segments": len(archive_segments),
            "workload_records": len(workloads),
            "fusion_classes": len(classes),
            "expected_fusion_classes": len(expected_signatures),
        },
        "sources": {
            "scan_summary": sha256_file(scan_dir / "summary.json"),
            "scan_provenance": sha256_file(scan_dir / "provenance.json"),
            "archive_summary": sha256_file(archive_dir / "summary.json"),
            "archive_provenance": sha256_file(archive_dir / "provenance.json"),
            "census": sha256_file(census_path),
            "verification": sha256_file(verification_path),
            "static_workloads": sha256_file(workloads_path),
        },
    }


def estimate_segment(*, model, scan_dir, segment, candidate_provenance=None):
    """Estimate one audited segment and return a fail-closed confidence status."""
    scan_dir = Path(scan_dir)
    reasons = []
    if candidate_provenance is not None:
        candidate_environment = environment_contract(candidate_provenance)
        if candidate_environment["sha256"] != model["environment"]["sha256"]:
            reasons.append("environment_mismatch")

    base = zero_dma()
    unknown_workloads = []
    for conv in segment["convolutions"]:
        key = workload_key(conv["workload"], conv["config"]["index"])
        if conv.get("is_fallback") or key not in model["workloads"]:
            unknown_workloads.append(key)
            continue
        base = add_vectors(base, model["workloads"][key]["base_dma"])
    if unknown_workloads:
        reasons.append("new_workload")

    residual, host = zero_dma(), {
        "cpu_read_bytes": 0,
        "cpu_write_bytes": 0,
        "cpu_helper_calls": 0,
        "convert_calls": 0,
    }
    unknown_classes, unsupported_classes = [], []
    primitives = read_json(scan_dir / segment["segment_id"].replace(":", "_") / "primitives.json")
    for primitive in primitives:
        record = model["fusion_classes"].get(primitive["signature"])
        if record is None:
            unknown_classes.append(primitive["signature"])
            continue
        if not record["supported"]:
            unsupported_classes.append(primitive["signature"])
            continue
        residual = add_vectors(residual, record["fusion_residual_dma"])
        host = add_vectors(host, record["host"])
    if unknown_classes:
        reasons.append("new_fusion_class")
    if unsupported_classes:
        reasons.append("unsupported_dynamic_dma")

    total_dma = add_vectors(base, residual)
    if any(value < 0 for value in total_dma.values()) and not reasons:
        raise ValueError("negative reconstructed DMA in {}".format(segment["segment_id"]))
    input_bytes = contract_bytes(segment["input_contract"])
    output_bytes = contract_bytes(segment["output_contract"])
    edge = {
        "input_logical_bytes": input_bytes,
        "output_logical_bytes": output_bytes,
        "logical_boundary_bytes": input_bytes + output_bytes,
        "physical_boundary_bytes": input_bytes + output_bytes,
        "physical_policy": model["runtime_boundary_policy"],
    }
    status = next((status for status in STATUS_PRIORITY if status in reasons), "exact_reuse")
    return {
        "segment_id": segment["segment_id"],
        "status": status,
        "all_status_reasons": [status for status in STATUS_PRIORITY if status in reasons],
        "usable_for_ranking": status == "exact_reuse",
        "full_build_required": status != "exact_reuse",
        "next_action": (
            "reuse_frozen_templates"
            if status == "exact_reuse"
            else "run_local_full_build_and_extend_calibration_only_after_validation"
        ),
        "unknown_workload_keys": sorted(set(unknown_workloads)),
        "unknown_fusion_classes": sorted(set(unknown_classes)),
        "unsupported_fusion_classes": sorted(set(unsupported_classes)),
        "primitive_invocations": len(primitives),
        "components": {
            "base_dma": base,
            "fusion_residual_dma": residual,
            "host": host,
            "edge": edge,
        },
        "total_dma": total_dma,
        "structured_vector": {
            "input_dma_bytes": total_dma["load_buffer_2d_inp_bytes"],
            "weight_dma_bytes": total_dma["load_buffer_2d_wgt_bytes"],
            "acc_dma_bytes": total_dma["load_buffer_2d_acc_bytes"],
            "output_dma_bytes": total_dma["store_buffer_2d_out_bytes"],
            "load_calls": total_dma["load_buffer_2d_calls"],
            "store_calls": total_dma["store_buffer_2d_calls"],
            "small_load_calls": total_dma["load_buffer_2d_small_calls"],
            "strided_load_calls": total_dma["load_buffer_2d_strided_calls"],
            "strided_store_calls": total_dma["store_buffer_2d_strided_calls"],
            **edge,
            **host,
        },
    }


def estimate_domain(*, model, scan_dir):
    scan_dir = Path(scan_dir)
    scan = _scan_rows(scan_dir)
    provenance = read_json(scan_dir / "provenance.json")
    return [
        estimate_segment(model=model, scan_dir=scan_dir, segment=segment, candidate_provenance=provenance)
        for segment in scan["rows"]
    ]


def pareto_layers(records, dimensions):
    """Return deterministic non-dominated layers; this orders but never prunes."""
    remaining = list(records)
    layers = []
    while remaining:
        front = []
        for left in remaining:
            dominated = False
            for right in remaining:
                if right is left:
                    continue
                weak = all(right[dimension] <= left[dimension] for dimension in dimensions)
                strict = any(right[dimension] < left[dimension] for dimension in dimensions)
                if weak and strict:
                    dominated = True
                    break
            if not dominated:
                front.append(left)
        # Keep the ordering reproducible when all modeled resources tie.  The
        # topology is the semantic tie-break; candidate_id then resolves
        # thread-count variants of the same topology.
        front.sort(
            key=lambda row: (
                row.get("m1_score_ms", 0.0),
                row.get("topology_id", row["candidate_id"]),
                row["candidate_id"],
            )
        )
        layers.append(front)
        front_ids = {id(row) for row in front}
        remaining = [row for row in remaining if id(row) not in front_ids]
    return layers
