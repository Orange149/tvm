#!/usr/bin/env python3
"""Build a fail-closed deployment manifest from a qualified VTA resource certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_vta_command_resource_certificate import (
    canonical_hash,
    load_json,
    sha256_file,
    validate_certificate,
    verify_frozen_file,
)


SCHEMA = "c3_vta_deployment_manifest_v1"


def manifest_hash_payload(manifest):
    """Hash the pre-execution contract; board qualification is attached afterwards."""

    keys = (
        "schema",
        "primary_candidate_id",
        "selection_policy",
        "deployment_identities",
        "architecture_fingerprint",
        "resource_certificate_key",
        "command_memory",
        "runtime_attestation",
        "replay_policy",
        "claim_boundary",
    )
    return {key: manifest[key] for key in keys}


def build_manifest(certificate, primary_candidate_id, queue_instances=1):
    validate_certificate(certificate)
    if int(queue_instances) <= 0:
        raise ValueError("queue_instances must be positive")
    identities = certificate["identities"]
    identity_ids = {row["candidate_id"] for row in identities}
    if primary_candidate_id not in identity_ids:
        raise ValueError("primary candidate is absent from the qualified allowlist")
    qualification = certificate.get("board_qualification")
    binary_by_id = {}
    if qualification is not None:
        binary_by_id = {
            row["candidate_id"]: row["binary_sha256"] for row in qualification["correctness"]
        }
        if set(binary_by_id) != identity_ids:
            raise ValueError("qualified binary set differs from the resource allowlist")
    per_queue = certificate["derivation"]
    per_instance = int(per_queue["total_capacity_bytes"])
    deployment_identities = []
    for identity in identities:
        row = dict(identity)
        row["role"] = "primary" if row["candidate_id"] == primary_candidate_id else "qualified_alternative"
        deployment_identities.append(row)
    payload = {
        "schema": SCHEMA,
        "status": "ready" if qualification is not None else "qualification_pending",
        "primary_candidate_id": primary_candidate_id,
        "selection_policy": (
            "use the protected incumbent unless an explicitly recorded policy selects a qualified alternative"
        ),
        "deployment_identities": sorted(
            deployment_identities, key=lambda row: row["candidate_id"]
        ),
        "architecture_fingerprint": certificate["architecture_fingerprint"],
        "resource_certificate_key": certificate["certificate_key"],
        "command_memory": {
            "formula": per_queue["formula"],
            "alignment_bytes": int(per_queue["alignment_bytes"]),
            "headroom_policy": "none_beyond_declared_alignment",
            "lifetime_policy": "one_thread_local_queue_instance_per_declared_concurrent_executor",
            "queue_instances": int(queue_instances),
            "per_instance": {
                "insn_capacity_bytes": int(per_queue["insn_capacity_bytes"]),
                "uop_capacity_bytes": int(per_queue["uop_capacity_bytes"]),
                "total_capacity_bytes": per_instance,
            },
            "deployment_total_capacity_bytes": per_instance * int(queue_instances),
            "runtime_environment_per_instance": certificate["runtime_contract"]["environment"],
        },
        "runtime_attestation": {
            "status_function": certificate["runtime_contract"]["status_function"],
            "status_schema": certificate["runtime_contract"]["status_schema"],
            "required_checks": certificate["runtime_contract"]["requirements"],
            "on_mismatch": "abort_before_device_submission_and_regenerate_certificate",
        },
        "replay_policy": certificate["replay_policy"],
        "claim_boundary": (
            "single workload and declared queue concurrency; no whole-model, dynamic-shape, "
            "allocator-fragmentation, performance, replay, or cross-boot claim"
        ),
    }
    if qualification is not None:
        payload["board_qualification_key"] = qualification["qualification_key"]
        # Post-execution evidence is intentionally outside manifest_hash_payload:
        # the pending contract ID launched on the board must remain the ID of
        # the subsequently qualified manifest.
        payload["qualification_attestation"] = {
            "binary_sha256_by_candidate_id": binary_by_id,
            "capacity_runtime_sha256": qualification["capacity_runtime_sha256"],
        }
    payload["manifest_id"] = canonical_hash(manifest_hash_payload(payload))
    payload["launcher_environment"] = {
        **certificate["runtime_contract"]["environment"],
        "VTA_COMMAND_MANIFEST_ID": payload["manifest_id"],
        "VTA_REPLAY_POLICY": payload["replay_policy"]["mode"],
    }
    return payload


def validate_manifest(manifest, certificate):
    if manifest.get("schema") != SCHEMA or manifest.get("status") not in {
        "qualification_pending",
        "ready",
    }:
        raise ValueError("unexpected deployment manifest schema or status")
    expected_id = canonical_hash(manifest_hash_payload(manifest))
    if manifest.get("manifest_id") != expected_id:
        raise ValueError("deployment manifest ID mismatch")
    validate_certificate(certificate)
    if manifest["resource_certificate_key"] != certificate["certificate_key"]:
        raise ValueError("deployment resource-certificate key mismatch")
    if certificate["status"] == "qualified_reduced_capacity":
        if manifest.get("status") != "ready":
            raise ValueError("qualified resource certificate requires a ready manifest")
        if manifest.get("board_qualification_key") != certificate["board_qualification"]["qualification_key"]:
            raise ValueError("deployment board-qualification key mismatch")
    elif manifest.get("status") != "qualification_pending" or "board_qualification_key" in manifest:
        raise ValueError("provisional resource certificate requires pending qualification")
    if certificate["status"] == "qualified_reduced_capacity":
        expected_binaries = {
            row["candidate_id"]: row["binary_sha256"]
            for row in certificate["board_qualification"]["correctness"]
        }
        attestation = manifest.get("qualification_attestation", {})
        if attestation.get("binary_sha256_by_candidate_id") != expected_binaries:
            raise ValueError("deployment binary attestation differs from board qualification")
    elif "qualification_attestation" in manifest:
        raise ValueError("pending manifest cannot contain post-execution attestation")
    identities = {row["candidate_id"] for row in manifest["deployment_identities"]}
    certified = {row["candidate_id"] for row in certificate["identities"]}
    if identities != certified:
        raise ValueError("deployment identity set differs from the certificate")
    command = manifest["command_memory"]
    per_instance = command["per_instance"]
    derivation = certificate["derivation"]
    if (
        int(per_instance["insn_capacity_bytes"]) != int(derivation["insn_capacity_bytes"])
        or int(per_instance["uop_capacity_bytes"]) != int(derivation["uop_capacity_bytes"])
        or int(per_instance["total_capacity_bytes"]) != int(derivation["total_capacity_bytes"])
    ):
        raise ValueError("deployment per-instance capacity differs from the certificate")
    expected_total = int(per_instance["total_capacity_bytes"]) * int(command["queue_instances"])
    if int(command["deployment_total_capacity_bytes"]) != expected_total:
        raise ValueError("deployment concurrent-queue capacity is inconsistent")
    if manifest["replay_policy"].get("mode") != "disabled":
        raise ValueError("current qualified evidence does not permit replay")
    expected_environment = {
        **certificate["runtime_contract"]["environment"],
        "VTA_COMMAND_MANIFEST_ID": manifest["manifest_id"],
        "VTA_REPLAY_POLICY": "disabled",
    }
    if manifest.get("launcher_environment") != expected_environment:
        raise ValueError("deployment launcher environment is inconsistent")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualified-resource-certificate", required=True)
    parser.add_argument("--primary-candidate-id", required=True)
    parser.add_argument("--queue-instances", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    certificate_path = Path(args.qualified_resource_certificate).resolve()
    input_guard = verify_frozen_file(certificate_path)
    certificate = load_json(certificate_path)
    manifest = build_manifest(certificate, args.primary_candidate_id, args.queue_instances)
    validate_manifest(manifest, certificate)
    manifest["provenance"] = {
        "qualified_resource_certificate": input_guard,
        "builder_sha256": sha256_file(__file__),
    }
    # Provenance documents the builder/input path but does not alter the frozen
    # pre-execution contract ID.
    manifest.pop("manifest_id")
    manifest.pop("launcher_environment", None)
    manifest["manifest_id"] = canonical_hash(manifest_hash_payload(manifest))
    manifest["launcher_environment"] = {
        **certificate["runtime_contract"]["environment"],
        "VTA_COMMAND_MANIFEST_ID": manifest["manifest_id"],
        "VTA_REPLAY_POLICY": manifest["replay_policy"]["mode"],
    }
    validate_manifest(manifest, certificate)
    manifest_path = output / "deployment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    command = manifest["command_memory"]
    (output / "STATUS.md").write_text(
        "# VTA deployment manifest\n\n"
        "- Status: `{}`\n"
        "- Exact identities: {}\n"
        "- Primary: `{}`\n"
        "- Queue instances: {}\n"
        "- Per-instance command backing: {} B\n"
        "- Deployment command backing: {} B\n"
        "- Replay: disabled\n"
        "- Mismatch policy: abort before device submission\n".format(
            manifest["status"],
            len(manifest["deployment_identities"]),
            manifest["primary_candidate_id"],
            command["queue_instances"],
            command["per_instance"]["total_capacity_bytes"],
            command["deployment_total_capacity_bytes"],
        )
    )
    hashes = {
        path.name: sha256_file(path)
        for path in output.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": hashes}, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "status": manifest["status"],
        "manifest_id": manifest["manifest_id"],
        "deployment_total_capacity_bytes": command["deployment_total_capacity_bytes"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
