#!/usr/bin/env python3
"""Validate the final manifest-to-runtime VTA command-resource evidence chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_vta_command_resource_certificate import load_json, load_jsonl, sha256_file
from build_vta_deployment_manifest import validate_manifest


SCHEMA = "c3_vta_runtime_manifest_attestation_v1"


def verify_frozen(path, relative_name=None):
    path = Path(path).resolve()
    ledger = load_json(path.parent / "artifact_hashes.json")["artifacts"]
    key = relative_name or path.name
    if ledger.get(key) != sha256_file(path):
        raise ValueError("frozen artifact hash mismatch: {}".format(path))
    return {"path": str(path), "sha256": sha256_file(path)}


def validate_attestation(manifest, certificate, board_summary, queue_status, correctness, contract):
    validate_manifest(manifest, certificate)
    if manifest.get("status") != "ready" or board_summary.get("status") != "passed":
        raise ValueError("manifest or board qualification is not ready")
    manifest_id = manifest["manifest_id"]
    if (
        not board_summary.get("manifest_attested")
        or board_summary.get("manifest_id") != manifest_id
        or contract.get("manifest", {}).get("manifest_id") != manifest_id
    ):
        raise ValueError("board experiment did not attest the ready manifest ID")
    expected_environment = manifest["launcher_environment"]
    if (
        queue_status.get("schema") != "vta_queue_capacity_status_v1"
        or queue_status.get("command_manifest_id") != manifest_id
        or queue_status.get("replay_policy") != expected_environment["VTA_REPLAY_POLICY"]
    ):
        raise ValueError("same-session runtime manifest/replay status mismatch")
    per_instance = manifest["command_memory"]["per_instance"]
    if (
        int(queue_status["insn_capacity_bytes"]) != int(per_instance["insn_capacity_bytes"])
        or int(queue_status["uop_capacity_bytes"]) != int(per_instance["uop_capacity_bytes"])
        or int(queue_status["insn_peak_bytes"]) > int(per_instance["insn_capacity_bytes"])
        or int(queue_status["uop_peak_bytes"]) > int(per_instance["uop_capacity_bytes"])
        or int(queue_status["submissions"])
        != int(contract["command_capacity"]["expected_submissions"])
    ):
        raise ValueError("runtime command capacity/peak/submission status mismatch")
    expected_ids = {row["candidate_id"] for row in manifest["deployment_identities"]}
    observed_ids = {row.get("candidate_id") for row in correctness}
    if observed_ids != expected_ids or len(correctness) != len(expected_ids):
        raise ValueError("board correctness set differs from the manifest allowlist")
    for row in correctness:
        seeds = row.get("seeds", [])
        if len(seeds) < 3 or not row.get("correct") or any(
            not seed.get("correct") or int(seed.get("mismatch_count", -1)) != 0
            for seed in seeds
        ):
            raise ValueError("manifest identity lacks three-seed exact correctness")
    expected_runtime = manifest["qualification_attestation"]["capacity_runtime_sha256"]
    if contract["board"]["remote_capacity_runtime_sha256"] != expected_runtime:
        raise ValueError("executed board runtime hashes differ from the ready manifest")
    qualification = certificate["board_qualification"]
    if qualification.get("command_manifest_id") != manifest_id or qualification.get(
        "replay_policy"
    ) != "disabled":
        raise ValueError("qualified certificate is not directly manifest-bound")
    if board_summary.get("performance_measurement") != "not_collected":
        raise ValueError("attestation experiment unexpectedly contains a performance label")
    if board_summary.get("restored_default_rpc", {}).get("CWD") != contract["board"][
        "default_rpc_cwd"
    ]:
        raise ValueError("default RPC was not restored")
    return {
        "schema": SCHEMA,
        "status": "attested",
        "manifest_id": manifest_id,
        "resource_certificate_key": certificate["certificate_key"],
        "qualification_key": qualification["qualification_key"],
        "candidate_ids": sorted(expected_ids),
        "correctness_seed_checks": sum(len(row["seeds"]) for row in correctness),
        "command_memory": {
            "formula": manifest["command_memory"]["formula"],
            "insn_peak_bytes": int(queue_status["insn_peak_bytes"]),
            "uop_peak_bytes": int(queue_status["uop_peak_bytes"]),
            "insn_capacity_bytes": int(per_instance["insn_capacity_bytes"]),
            "uop_capacity_bytes": int(per_instance["uop_capacity_bytes"]),
            "total_capacity_bytes": int(per_instance["total_capacity_bytes"]),
            "legacy_total_bytes": int(board_summary["legacy_total_bytes"]),
            "backing_reduction_fraction": float(
                board_summary["requested_byte_reduction_fraction"]
            ),
        },
        "runtime": {
            "command_manifest_id": queue_status["command_manifest_id"],
            "replay_policy": queue_status["replay_policy"],
            "submissions": int(queue_status["submissions"]),
            "binary_sha256": expected_runtime,
        },
        "safety": {
            "default_rpc_restored": True,
            "sd_writes_for_experiment": board_summary["sd_writes_for_experiment"],
            "reboot_or_poweroff": False,
        },
        "claim_boundary": contract["claim_boundary"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-manifest", required=True)
    parser.add_argument("--board-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)

    manifest_path = Path(args.ready_manifest).resolve()
    manifest_guard = verify_frozen(manifest_path)
    manifest = load_json(manifest_path)
    certificate_path = Path(
        manifest["provenance"]["qualified_resource_certificate"]["path"]
    )
    certificate_guard = verify_frozen(certificate_path)
    certificate = load_json(certificate_path)
    board_dir = Path(args.board_run_dir).resolve()
    board_files = {
        "summary": board_dir / "summary.json",
        "queue_status": board_dir / "queue_capacity_status.json",
        "correctness": board_dir / "correctness/correctness.jsonl",
        "preregistered": board_dir / "preregistered.json",
    }
    board_ledger = load_json(board_dir / "artifact_hashes.json")["artifacts"]
    board_guards = {}
    for name, path in board_files.items():
        relative = str(path.relative_to(board_dir))
        observed = sha256_file(path)
        if board_ledger.get(relative) != observed:
            raise ValueError("frozen board artifact hash mismatch: {}".format(path))
        board_guards[name] = {"path": str(path), "sha256": observed}
    preregistered = load_json(board_files["preregistered"])
    contract_path = Path(preregistered["contract"]["path"])
    if sha256_file(contract_path) != preregistered["contract"]["sha256"]:
        raise ValueError("board contract hash mismatch")
    result = validate_attestation(
        manifest,
        certificate,
        load_json(board_files["summary"]),
        load_json(board_files["queue_status"]),
        load_jsonl(board_files["correctness"]),
        load_json(contract_path),
    )
    result["inputs"] = {
        "ready_manifest": manifest_guard,
        "qualified_certificate": certificate_guard,
        "board": board_guards,
        "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
    }
    result_path = output / "runtime_manifest_attestation.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    command = result["command_memory"]
    (output / "STATUS.md").write_text(
        "# VTA runtime manifest attestation\n\n"
        "- Status: `attested`\n"
        "- Manifest: `{}`\n"
        "- Exact identities/seeds: {}/{}\n"
        "- Observed peak instruction/UOP: {}/{} B\n"
        "- Certified capacity instruction/UOP: {}/{} B\n"
        "- Replay: disabled\n"
        "- Default RPC restored: yes\n".format(
            result["manifest_id"], len(result["candidate_ids"]),
            result["correctness_seed_checks"], command["insn_peak_bytes"],
            command["uop_peak_bytes"], command["insn_capacity_bytes"],
            command["uop_capacity_bytes"],
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
        "status": result["status"],
        "manifest_id": result["manifest_id"],
        "command_memory": result["command_memory"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
