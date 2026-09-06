#!/usr/bin/env python3
"""Generate V1-P1 transition, archive-reuse, and grouped-holdout manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from freeze_cpu_vta_pipeline_v1 import (
    CPU_THREAD_CHOICES,
    PROTOCOL_ID,
    UNIT_ORDER,
    _unit_kind,
    load_sealed_artifact,
    repo_root,
    seal_artifact,
    validate_artifact_sha256,
)
from search_resnet18_stage_splits import (
    enumerate_island_sets,
    native_pipeline_scheme,
    scheme_from_islands,
)


DEFAULT_OUTPUT = (
    repo_root() / "vta/tutorials/frontend/report_out/resource_aware_maxplus"
)
REPRESENTATIVE_SCHEMES = ("three_stage_a", "three_stage_e", "three_stage_b")


def canonical_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def digest(payload):
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def schema_nbytes(schema):
    dtype_bytes = {
        "int8": 1,
        "uint8": 1,
        "int16": 2,
        "uint16": 2,
        "int32": 4,
        "uint32": 4,
        "float32": 4,
        "int64": 8,
        "uint64": 8,
        "float64": 8,
    }
    total = 0
    for slot in schema.get("slots", []):
        elements = 1
        for extent in slot["shape"]:
            elements *= int(extent)
        total += elements * dtype_bytes[slot["dtype"]]
    return total


def unit_cost_key(device, unit):
    payload = {
        "device": device,
        "kind": unit["kind"],
        "input_contract": unit["input_contract"],
        "output_contract": unit["output_contract"],
        "logical_compute_ops_est": unit["static_compute"]["compute_ops_est"],
    }
    return "{}:{}:{}".format(device, unit["kind"], digest(payload)[:16])


def enumerate_reachable(unit_schema):
    unit_by_name = {item["name"]: item for item in unit_schema["units"]}
    candidates = []
    segments = {}
    boundaries = {}
    for islands in enumerate_island_sets(3):
        scheme = scheme_from_islands(islands)
        if not native_pipeline_scheme(scheme):
            continue
        devices = [stage["device"] for stage in scheme]
        if any(left == right for left, right in zip(devices, devices[1:])):
            continue
        candidates.append(scheme)
        for stage in scheme:
            names = tuple(stage["unit_names"])
            device = stage["device"]
            start = UNIT_ORDER.index(names[0])
            end = UNIT_ORDER.index(names[-1])
            segment_id = "{}:{:02d}:{:02d}".format(device, start, end)
            if segment_id not in segments:
                input_contract = unit_by_name[names[0]]["input_contract"]
                output_contract = unit_by_name[names[-1]]["output_contract"]
                mapping = [unit_cost_key(device, unit_by_name[name]) for name in names]
                signature = {
                    "device": device,
                    "unit_kinds": [_unit_kind(name) for name in names],
                    "input_contract": input_contract,
                    "output_contract": output_contract,
                    "frozen_schedule": "cpu_default_or_vta_boundary_bridge_v1",
                }
                logical_compute_ops = sum(
                    int(unit_by_name[name]["static_compute"]["compute_ops_est"])
                    for name in names
                )
                segments[segment_id] = {
                    "segment_id": segment_id,
                    "device": device,
                    "start_index": start,
                    "end_index": end,
                    "unit_names": list(names),
                    "input_contract": input_contract,
                    "output_contract": output_contract,
                    "signature_sha256": digest(signature),
                    "logical_compute_ops_est": logical_compute_ops,
                    "logical_compute_gops_est": float(logical_compute_ops) / 1e9,
                    "thread_choices": list(CPU_THREAD_CHOICES) if device == "cpu" else [1],
                    "cost_mapping": {
                        "model": "additive_atomic_logical_ops_v1",
                        "unit_cost_keys": mapping,
                        "segment_correction_key": None,
                        "nonadditivity_policy": "diagnose_with_grouped_holdout_not_fit_in_P1",
                    },
                    "compiler_status": "shortlist_compile_required",
                    "representative_compile": None,
                }
        for left, right in zip(scheme, scheme[1:]):
            cut = UNIT_ORDER.index(right["unit_names"][0])
            direction = "{}_to_{}".format(left["device"], right["device"])
            boundary_id = "boundary:{:02d}:{}".format(cut, direction)
            if boundary_id in boundaries:
                continue
            tensor_contract = unit_by_name[left["unit_names"][-1]]["output_contract"]
            slots = []
            for ordinal, slot in enumerate(tensor_contract["slots"]):
                host_id = "{}:host_adapter:{}".format(boundary_id, ordinal)
                dma_kind = "load" if direction == "cpu_to_vta" else "store"
                dma_id = "vta_boundary:{:02d}:{}_instruction:{}".format(
                    cut, dma_kind, ordinal
                )
                slots.append(
                    {
                        "slot_index": ordinal,
                        "host_accounting_id": host_id,
                        "dma_accounting_id": dma_id,
                        "host_owner": "boundary_adapter_set_get",
                        "dma_owner": "lowered_vta_{}_including_spill".format(dma_kind),
                    }
                )
            boundaries[boundary_id] = {
                "boundary_id": boundary_id,
                "cut_position": cut,
                "direction": direction,
                "tensor_contract": tensor_contract,
                "logical_bytes": schema_nbytes(tensor_contract),
                "transfer_resource": "stage_dma",
                "slots": slots,
            }
    return candidates, segments, boundaries


def load_representatives(output_dir, segments):
    records = []
    for scheme in REPRESENTATIVE_SCHEMES:
        manifest_path = output_dir / "v1_p1_packages" / scheme / "package" / "manifest.json"
        with manifest_path.open(encoding="utf-8") as inp:
            manifest = json.load(inp)
        package_record = {
            "scheme": scheme,
            "role": "holdout" if scheme == "three_stage_b" else "fit",
            "manifest_path": str(manifest_path.relative_to(repo_root())),
            "stages": [],
        }
        for stage in manifest["stages"]:
            names = stage["unit_names"]
            segment_id = "{}:{:02d}:{:02d}".format(
                stage["device"], UNIT_ORDER.index(names[0]), UNIT_ORDER.index(names[-1])
            )
            proof = {
                "package_scheme": scheme,
                "stage_name": stage["name"],
                "cache_key": stage["stage_build_cache_key"],
                "cache_hit": stage["stage_build_cache_hit"],
                "relay_ir_sha256": stage["relay_ir_sha256"],
                "compiled_module_sources": stage["compiled_module_sources"],
                "artifact_sha256": stage["artifact_sha256"],
            }
            package_record["stages"].append({"segment_id": segment_id, **proof})
            segments[segment_id]["compiler_status"] = "native_compile_passed"
            segments[segment_id]["representative_compile"] = proof
        records.append(package_record)
    return records


def archive_audit(output_dir, hardware_fingerprint):
    current = hardware_fingerprint["stable_hardware_fingerprint_sha256"]
    archives = [
        {
            "archive": "stage4b_native_calibration",
            "path": "vta/tutorials/frontend/report_out/resource_aware_maxplus/stage4b_native_calibration",
            "status": "stale",
            "reasons": [
                "legacy bucket schema is not segment-lowering keyed",
                "six VTA operator correctness groups failed determinism",
                "runner/lowering source is not bound to the V1 stable fingerprint",
            ],
        },
        {
            "archive": "stage4a_q_qualification",
            "path": "vta/tutorials/frontend/report_out/resource_aware_maxplus/stage4a_q_qualification",
            "status": "stale",
            "reasons": [
                "DMA qualification used a different protocol and runtime fingerprint",
                "usable as experiment-design evidence only",
            ],
        },
        {
            "archive": "stage4_native_three_stage_e",
            "path": "vta/tutorials/frontend/report_out/resource_aware_maxplus/stage4_calibration_smoke/native_three_stage_e",
            "status": "stale",
            "reasons": [
                "manifest predates Relay/module/source hashes",
                "timing boundary and CPU affinity do not satisfy V1 protocol",
            ],
        },
        {
            "archive": "v1_p1_native_packages",
            "path": "vta/tutorials/frontend/report_out/resource_aware_maxplus/v1_p1_packages",
            "status": "reusable_compile_only",
            "reasons": [
                "compiled under current source-sensitive cache key",
                "contains no board performance measurements",
            ],
        },
    ]
    return seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_archive_reuse_audit",
            "protocol_id": PROTOCOL_ID,
            "stable_hardware_fingerprint_sha256": current,
            "reusable_timing_record_count": 0,
            "policy": "only exact fingerprint+correctness+timing-boundary matches may be reused",
            "archives": archives,
            "conclusion": "P2 must collect one fresh minimal session; historical values remain diagnostics",
        }
    )


def build_manifests(output_dir):
    unit_schema = load_sealed_artifact(
        output_dir / "v1_unit_and_boundary_schema.json",
        "cpu_vta_pipeline_v1_unit_and_boundary_schema",
    )
    hardware = load_sealed_artifact(
        output_dir / "v1_hardware_fingerprint.json",
        "cpu_vta_pipeline_v1_hardware_fingerprint",
    )
    protocol = load_sealed_artifact(
        output_dir / "v1_protocol.json", "cpu_vta_pipeline_v1_protocol"
    )
    candidates, segments, boundaries = enumerate_reachable(unit_schema)
    representatives = load_representatives(output_dir, segments)
    accounting_ids = []
    for boundary in boundaries.values():
        for slot in boundary["slots"]:
            accounting_ids.extend([slot["host_accounting_id"], slot["dma_accounting_id"]])
    if len(accounting_ids) != len(set(accounting_ids)):
        raise RuntimeError("duplicate DDR/boundary accounting id")

    profile = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_profile_manifest",
            "protocol_id": PROTOCOL_ID,
            "protocol_artifact_sha256": protocol["artifact_sha256"],
            "stable_hardware_fingerprint_sha256": hardware[
                "stable_hardware_fingerprint_sha256"
            ],
            "model": "resnet18_v1",
            "cost_model_scope": "Tarnawski-style additive logical-op service plus composite boundary",
            "coverage": {
                "legal_candidate_count": len(candidates),
                "unique_cpu_segment_count": sum(
                    item["device"] == "cpu" for item in segments.values()
                ),
                "unique_vta_segment_count": sum(
                    item["device"] == "vta" for item in segments.values()
                ),
                "cpu_thread_execution_signature_count": sum(
                    len(item["thread_choices"])
                    for item in segments.values()
                    if item["device"] == "cpu"
                ),
                "heterogeneous_boundary_count": len(boundaries),
                "representative_compiled_segment_count": sum(
                    item["compiler_status"] == "native_compile_passed"
                    for item in segments.values()
                ),
            },
            "compiler_policy": {
                "P1": "compile grouped representatives; manifest every reachable segment",
                "P5": "compile-only every shortlisted candidate before board execution",
                "full_segment_compile_sweep_forbidden": True,
            },
            "segments": [segments[key] for key in sorted(segments)],
            "boundaries": [boundaries[key] for key in sorted(boundaries)],
            "representative_packages": representatives,
            "gate_checks": {
                "every_reachable_segment_has_cost_mapping": all(
                    item["cost_mapping"]["unit_cost_keys"] for item in segments.values()
                ),
                "all_accounting_ids_unique": len(accounting_ids) == len(set(accounting_ids)),
                "representative_native_compile_passed": all(
                    stage["compiled_module_sources"]
                    for package in representatives
                    for stage in package["stages"]
                ),
                "historical_timing_not_silently_reused": True,
            },
        }
    )
    holdout = seal_artifact(
        {
            "schema_version": 1,
            "kind": "cpu_vta_pipeline_v1_grouped_holdout",
            "protocol_id": PROTOCOL_ID,
            "profile_manifest_artifact_sha256": profile["artifact_sha256"],
            "fit_groups": ["three_stage_a", "three_stage_e"],
            "holdout_groups": ["three_stage_b"],
            "leakage_rule": "no stage timing from three_stage_b may update V1 fit coefficients",
            "metrics": [
                "stage_additivity_ape",
                "pipeline_cycle_ape",
                "predicted_vs_measured_bottleneck",
                "serial_pipeline_correctness",
            ],
            "decision_rule": {
                "ranking_gate": "holdout must preserve bottleneck and candidate order",
                "residual_policy": "record concurrent residual for V2; do not fit contention in V1",
            },
        }
    )
    audit = archive_audit(output_dir, hardware)
    return {
        "v1_profile_manifest.json": profile,
        "v1_archive_reuse_audit.json": audit,
        "v1_grouped_holdout.json": holdout,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    outputs = build_manifests(output_dir)
    for name, payload in outputs.items():
        validate_artifact_sha256(payload)
        (output_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(name, payload["artifact_sha256"])


if __name__ == "__main__":
    main()
