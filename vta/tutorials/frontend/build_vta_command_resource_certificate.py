#!/usr/bin/env python3
"""Build an exact, portable VTA command-memory deployment certificate.

The certificate does not encode a board-specific capacity constant.  It derives
page-aligned instruction and UOP capacities from the maximum structural peak of
an exact deployment allowlist, and binds the result to the FPGA architecture,
lowered candidate identities, and runtime allocation ABI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "c3_vta_command_resource_certificate_v1"
DEFAULT_ALIGNMENT_BYTES = 4096
DEFAULT_GUARDED_SOURCES = (
    "vta/python/vta/top/vta_conv2d.py",
    "vta/python/vta/top/vta_conv2d_residency.py",
    "vta/runtime/runtime.cc",
    "vta/runtime/queue_capacity.h",
    "vta/tutorials/frontend/build_vta_command_resource_certificate.py",
    "vta/tutorials/frontend/tune_resnet18_vta.py",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def align_up(value, alignment=DEFAULT_ALIGNMENT_BYTES):
    value, alignment = int(value), int(alignment)
    if value <= 0 or alignment <= 0:
        raise ValueError("capacity peaks and alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def verify_frozen_file(path):
    path = Path(path).resolve()
    ledger_path = path.parent / "artifact_hashes.json"
    if not ledger_path.is_file():
        raise ValueError("missing artifact hash ledger for {}".format(path))
    ledger = load_json(ledger_path)
    expected = ledger.get("artifacts", ledger).get(path.name)
    observed = sha256_file(path)
    if expected != observed:
        raise ValueError("frozen artifact hash mismatch: {}".format(path))
    return {"path": str(path), "sha256": observed}


def candidate_config_index(candidate):
    if "config_index" in candidate:
        return int(candidate["config_index"])
    return int(candidate["debug"]["config_index"])


def candidate_tir_sha256(candidate):
    return candidate.get("tir_sha256") or candidate.get("expected_tir_sha256")


def selected_candidate_ids(dispatch, explicit_ids=(), selected_ids=()):
    dispatched = {
        row["candidate_id"]
        for row in dispatch.get("records", [])
        if row.get("route") == "allow_timing" and row.get("candidate_id")
    }
    eligible = dispatched | set(explicit_ids)
    selected = set(selected_ids) if selected_ids else eligible
    unauthorized = sorted(selected - eligible)
    if unauthorized:
        raise ValueError("selected identities are not dispatched or explicit fallbacks: {}".format(unauthorized))
    if not selected:
        raise ValueError("deployment allowlist is empty")
    return selected


def build_identity(candidate, signature):
    if signature.get("outcome") != "passed_local_signature":
        raise ValueError("candidate {} lacks a passing structural signature".format(candidate["candidate_id"]))
    if signature.get("workload_id") != candidate.get("workload_id"):
        raise ValueError("workload mismatch for candidate {}".format(candidate["candidate_id"]))
    mode = candidate.get("public_mode") or candidate.get("residence_mode")
    if signature.get("public_mode") != mode:
        raise ValueError("residence-mode mismatch for candidate {}".format(candidate["candidate_id"]))
    expected_tir = candidate_tir_sha256(candidate)
    if not expected_tir or signature.get("relowered_tir_sha256") != expected_tir:
        raise ValueError("lowered-TIR mismatch for candidate {}".format(candidate["candidate_id"]))
    structural = signature["command_signature"]["structural"]
    peaks = structural["peaks"]
    insn_peak, uop_peak = int(peaks["insn_bytes"]), int(peaks["uop_bytes"])
    submissions = int(structural["submissions"])
    finish_count = int(structural["finish"]["source_derived_count"])
    if min(insn_peak, uop_peak, submissions, finish_count) <= 0:
        raise ValueError("non-positive structural command observation")
    if finish_count != submissions:
        raise ValueError("FINISH/submission mismatch for candidate {}".format(candidate["candidate_id"]))
    execution_runtime = signature.get("execution_runtime") or {}
    runtime_sha256 = execution_runtime.get("sha256")
    if (
        not isinstance(runtime_sha256, str)
        or len(runtime_sha256) != 64
        or any(character not in "0123456789abcdef" for character in runtime_sha256)
    ):
        raise ValueError(
            "candidate {} lacks an actual FSim execution-runtime hash".format(
                candidate["candidate_id"]
            )
        )
    return {
        "candidate_id": candidate["candidate_id"],
        "workload_id": candidate["workload_id"],
        "residence_mode": mode,
        "config_index": candidate_config_index(candidate),
        "tir_sha256": expected_tir,
        "complete_config_entity": candidate["identity"]["complete_config_entity"],
        "command_signature_sha256": signature["command_signature"]["sha256"],
        "fsim_execution_runtime_sha256": runtime_sha256,
        "insn_peak_bytes": insn_peak,
        "uop_peak_bytes": uop_peak,
        "submissions": submissions,
        "finish_count": finish_count,
    }


def build_certificate(
    dispatch,
    candidates,
    signatures,
    architecture_config,
    source_guards,
    explicit_ids=(),
    selected_ids=(),
    alignment_bytes=DEFAULT_ALIGNMENT_BYTES,
    inputs=None,
):
    if dispatch.get("schema") not in {
        "c3_vta_hardware_certified_dispatch_v1",
        "c3_vta_hardware_certified_dispatch_v2",
    }:
        raise ValueError("unexpected hardware dispatch schema")
    if dispatch.get("performance_labels_used") is not False:
        raise ValueError("hardware dispatch must be label-free")
    selected = selected_candidate_ids(dispatch, explicit_ids, selected_ids)
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    signature_by_id = {row["candidate_id"]: row for row in signatures}
    missing_candidates = sorted(selected - set(candidate_by_id))
    missing_signatures = sorted(selected - set(signature_by_id))
    if missing_candidates or missing_signatures:
        raise ValueError(
            "selected identities missing candidate/signature rows: candidates={} signatures={}".format(
                missing_candidates, missing_signatures
            )
        )
    identities = sorted(
        (build_identity(candidate_by_id[key], signature_by_id[key]) for key in selected),
        key=lambda row: row["candidate_id"],
    )
    execution_runtime_hashes = sorted(
        {row["fsim_execution_runtime_sha256"] for row in identities}
    )
    if len(execution_runtime_hashes) != 1:
        raise ValueError(
            "certificate identities were executed by different FSim runtime binaries"
        )
    alignment_bytes = int(alignment_bytes)
    insn_peak = max(row["insn_peak_bytes"] for row in identities)
    uop_peak = max(row["uop_peak_bytes"] for row in identities)
    insn_capacity = align_up(insn_peak, alignment_bytes)
    uop_capacity = align_up(uop_peak, alignment_bytes)
    architecture = {
        "schema": "vta_architecture_fingerprint_v1",
        "config": architecture_config,
        "key": canonical_hash(architecture_config),
    }
    identity_set_key = canonical_hash(identities)
    invalidation_material = {
        "architecture_fingerprint_key": architecture["key"],
        "identity_set_key": identity_set_key,
        "source_guards_sha256": source_guards,
        "fsim_execution_runtime_sha256": execution_runtime_hashes[0],
        "allocation_abi": {
            "alignment_bytes": alignment_bytes,
            "queues": ["instruction", "uop"],
            "capacity_policy": "align(max exact per-submit peak over deployment allowlist)",
        },
    }
    return {
        "schema": SCHEMA,
        "status": "local_provisional",
        "derivation": {
            "formula": "C_q(S)=align_A(max_{c in S,b in submits(c)} B_q(c,b))",
            "alignment_bytes": alignment_bytes,
            "insn_peak_bytes": insn_peak,
            "uop_peak_bytes": uop_peak,
            "insn_capacity_bytes": insn_capacity,
            "uop_capacity_bytes": uop_capacity,
            "total_capacity_bytes": insn_capacity + uop_capacity,
        },
        "architecture_fingerprint": architecture,
        "source_guards_sha256": source_guards,
        "fsim_execution_runtime": {
            "sha256": execution_runtime_hashes[0],
            "binding": "actual /proc/self/maps entry in each isolated execution worker",
        },
        "identity_set_key": identity_set_key,
        "identities": identities,
        "invalidation_material": invalidation_material,
        "certificate_key": canonical_hash(invalidation_material),
        "runtime_contract": {
            "environment": {
                "VTA_INSN_BUFFER_BYTES": str(insn_capacity),
                "VTA_UOP_BUFFER_BYTES": str(uop_capacity),
            },
            "status_function": "vta.runtime.queue_capacity_status",
            "status_schema": "vta_queue_capacity_status_v1",
            "requirements": [
                "runtime-reported capacities equal the certificate",
                "runtime-reported peaks do not exceed the certificate",
                "all deployed candidate IDs are covered by the certificate",
            ],
        },
        "replay_policy": {
            "mode": "disabled",
            "reason": "capture/replay command-resource records are not yet observable in the frozen evidence",
            "deployment_requirement": "VTA replay APIs must not be used under this certificate",
        },
        "invalidation_rule": (
            "rebuild when the FPGA architecture config, deployment identity set, lowered TIR, "
            "complete ConfigEntity, schedule/runtime source, or allocation alignment changes"
        ),
        "boot_id_bound": False,
        "claim_boundary": (
            "exact deployment-allowlist command backing; not a universal constant, allocator "
            "fragmentation bound, latency result, or whole-model concurrency bound"
        ),
        "inputs": inputs or {},
    }


def validate_certificate(certificate, required_candidate_ids=(), repo_root=None):
    if certificate.get("schema") != SCHEMA or certificate.get("status") not in {
        "local_provisional",
        "qualified_reduced_capacity",
    }:
        raise ValueError("unexpected command-resource certificate status")
    identities = certificate.get("identities", [])
    if not identities:
        raise ValueError("command-resource certificate has no identities")
    available = {row["candidate_id"] for row in identities}
    missing = sorted(set(required_candidate_ids) - available)
    if missing:
        raise ValueError("command-resource certificate misses required identities: {}".format(missing))
    derivation = certificate["derivation"]
    alignment = int(derivation["alignment_bytes"])
    expected_insn = align_up(max(int(row["insn_peak_bytes"]) for row in identities), alignment)
    expected_uop = align_up(max(int(row["uop_peak_bytes"]) for row in identities), alignment)
    if (
        int(derivation["insn_capacity_bytes"]) != expected_insn
        or int(derivation["uop_capacity_bytes"]) != expected_uop
        or int(derivation["total_capacity_bytes"]) != expected_insn + expected_uop
    ):
        raise ValueError("command-resource capacity derivation is inconsistent")
    if canonical_hash(identities) != certificate["identity_set_key"]:
        raise ValueError("command-resource identity-set hash mismatch")
    runtime_hashes = {row.get("fsim_execution_runtime_sha256") for row in identities}
    runtime_hash = certificate.get("fsim_execution_runtime", {}).get("sha256")
    if len(runtime_hashes) != 1 or runtime_hash not in runtime_hashes:
        raise ValueError("FSim execution-runtime binding mismatch")
    material = certificate["invalidation_material"]
    if canonical_hash(material) != certificate["certificate_key"]:
        raise ValueError("command-resource invalidation key mismatch")
    if repo_root is not None:
        repo_root = Path(repo_root)
        for relpath, expected in certificate["source_guards_sha256"].items():
            path = repo_root / relpath
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError("command-resource source guard changed: {}".format(relpath))
        config_path = repo_root / "3rdparty/vta-hw/config/vta_config.json"
        observed_config = load_json(config_path)
        if canonical_hash(observed_config) != certificate["architecture_fingerprint"]["key"]:
            raise ValueError("VTA architecture fingerprint changed")
    return {
        "certificate_key": certificate["certificate_key"],
        "candidate_ids": sorted(available),
        "environment": certificate["runtime_contract"]["environment"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--signatures", required=True)
    parser.add_argument("--include-candidate-id", action="append", default=[])
    parser.add_argument(
        "--select-candidate-id",
        action="append",
        default=[],
        help="Optional exact deployment subset of dispatched identities plus explicit fallbacks",
    )
    parser.add_argument("--alignment-bytes", type=int, default=DEFAULT_ALIGNMENT_BYTES)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    paths = {
        "dispatch": Path(args.dispatch).resolve(),
        "candidates": Path(args.candidates).resolve(),
        "signatures": Path(args.signatures).resolve(),
    }
    inputs = {name: verify_frozen_file(path) for name, path in paths.items()}
    root = Path(__file__).resolve().parents[3]
    config_path = root / "3rdparty/vta-hw/config/vta_config.json"
    source_guards = {relpath: sha256_file(root / relpath) for relpath in DEFAULT_GUARDED_SOURCES}
    inputs["architecture_config"] = {
        "path": str(config_path),
        "sha256": sha256_file(config_path),
    }
    preregistered = {
        "schema": "c3_vta_command_resource_certificate_contract_v1",
        "status": "frozen_before_certificate_composition",
        "selected_routes": ["allow_timing"],
        "explicit_candidate_ids": sorted(args.include_candidate_id),
        "selected_candidate_ids": sorted(args.select_candidate_id),
        "alignment_bytes": args.alignment_bytes,
        "inputs": inputs,
        "source_guards_sha256": source_guards,
        "performance_labels_used": False,
    }
    preregistered_path = output / "preregistered.json"
    preregistered_path.write_text(json.dumps(preregistered, indent=2, sort_keys=True) + "\n")
    certificate = build_certificate(
        load_json(paths["dispatch"]),
        load_jsonl(paths["candidates"]),
        load_jsonl(paths["signatures"]),
        load_json(config_path),
        source_guards,
        explicit_ids=args.include_candidate_id,
        selected_ids=args.select_candidate_id,
        alignment_bytes=args.alignment_bytes,
        inputs=inputs,
    )
    validate_certificate(certificate, repo_root=root)
    certificate_path = output / "command_resource_certificate.json"
    certificate_path.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    status_path = output / "STATUS.md"
    status_path.write_text(
        "# VTA command-resource deployment certificate\n\n"
        "- Status: `local_provisional`\n"
        "- Exact deployment identities: {}\n"
        "- Peak instruction/UOP: {}/{} B\n"
        "- Capacity instruction/UOP: {}/{} B\n"
        "- Total requested backing: {} B\n"
        "- Boot-bound: no; architecture/TIR/runtime-bound: yes\n"
        "- Replay policy: disabled until capture/replay has an observed resource signature\n"
        "- This is not a universal capacity constant.\n".format(
            len(certificate["identities"]),
            certificate["derivation"]["insn_peak_bytes"],
            certificate["derivation"]["uop_peak_bytes"],
            certificate["derivation"]["insn_capacity_bytes"],
            certificate["derivation"]["uop_capacity_bytes"],
            certificate["derivation"]["total_capacity_bytes"],
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
    print(json.dumps(certificate["derivation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
