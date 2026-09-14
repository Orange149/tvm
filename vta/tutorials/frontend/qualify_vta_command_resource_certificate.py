#!/usr/bin/env python3
"""Upgrade a provisional VTA command certificate with frozen FPGA evidence."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from build_vta_command_resource_certificate import (
    canonical_hash,
    load_json,
    load_jsonl,
    sha256_file,
    validate_certificate,
    verify_frozen_file,
)


def match_identity(row, identities):
    if row.get("candidate_id"):
        matches = [item for item in identities if item["candidate_id"] == row["candidate_id"]]
    else:
        matches = [
            item
            for item in identities
            if item["residence_mode"] == row.get("mode")
            and int(item["config_index"]) == int(row.get("config_index", -1))
            and item["tir_sha256"] == row.get("tir_sha256")
        ]
    if len(matches) != 1:
        raise ValueError("board row does not resolve to exactly one certified identity")
    return matches[0]


def qualify_certificate(certificate, board_summary, correctness_rows, queue_status, board_contract):
    validate_certificate(certificate)
    if certificate["status"] != "local_provisional":
        raise ValueError("only a local provisional certificate can be upgraded")
    if certificate.get("replay_policy", {}).get("mode") != "disabled":
        raise ValueError("this qualifier has no capture/replay evidence")
    if board_summary.get("status") != "passed":
        raise ValueError("board reduced-capacity run did not pass")
    derivation = certificate["derivation"]
    expected_capacity = {
        "insn_bytes": int(derivation["insn_capacity_bytes"]),
        "uop_bytes": int(derivation["uop_capacity_bytes"]),
    }
    if board_summary.get("capacity") != expected_capacity:
        raise ValueError("board requested capacity differs from the certificate")
    if queue_status.get("schema") != "vta_queue_capacity_status_v1":
        raise ValueError("missing same-session runtime capacity status")
    if (
        int(queue_status["insn_capacity_bytes"]) != expected_capacity["insn_bytes"]
        or int(queue_status["uop_capacity_bytes"]) != expected_capacity["uop_bytes"]
    ):
        raise ValueError("runtime-reported capacity differs from the certificate")
    if (
        int(queue_status["insn_peak_bytes"]) > expected_capacity["insn_bytes"]
        or int(queue_status["uop_peak_bytes"]) > expected_capacity["uop_bytes"]
    ):
        raise ValueError("runtime-reported peak exceeds the certificate")
    manifest_environment = board_contract.get("manifest_environment") or {}
    if manifest_environment and (
        queue_status.get("command_manifest_id")
        != manifest_environment.get("VTA_COMMAND_MANIFEST_ID")
        or queue_status.get("replay_policy")
        != manifest_environment.get("VTA_REPLAY_POLICY")
    ):
        raise ValueError("runtime-reported manifest/replay contract differs")
    identities = certificate["identities"]
    if len(correctness_rows) != len(identities):
        raise ValueError("board evidence does not cover the exact deployment identity set")
    qualified_rows = []
    covered = set()
    for row in correctness_rows:
        identity = match_identity(row, identities)
        if identity["candidate_id"] in covered:
            raise ValueError("duplicate board identity evidence")
        covered.add(identity["candidate_id"])
        seeds = row.get("seeds", [])
        seed_ids = [int(seed["seed"]) for seed in seeds]
        if len(seeds) < 3 or len(set(seed_ids)) != len(seed_ids):
            raise ValueError("board qualification requires at least three unique seeds")
        if not row.get("correct") or any(
            not seed.get("correct") or int(seed.get("mismatch_count", -1)) != 0 for seed in seeds
        ):
            raise ValueError("board correctness evidence contains a mismatch")
        if row.get("tir_sha256") != identity["tir_sha256"]:
            raise ValueError("board TIR differs from the resource identity")
        qualified_rows.append(
            {
                "candidate_id": identity["candidate_id"],
                "binary_sha256": row["binary_sha256"],
                "seed_ids": seed_ids,
                "passed_seed_count": len(seeds),
                "mismatch_count": 0,
            }
        )
    expected_sources = certificate["source_guards_sha256"]
    observed_sources = board_contract.get("source_guards_sha256", {})
    board_relevant_sources = {
        "vta/python/vta/top/vta_conv2d.py",
        "vta/python/vta/top/vta_conv2d_residency.py",
        "vta/runtime/runtime.cc",
        "vta/runtime/queue_capacity.h",
    }
    changed = sorted(
        path
        for path, digest in expected_sources.items()
        if path in board_relevant_sources and observed_sources.get(path) != digest
    )
    if changed:
        raise ValueError("board runtime/schedule source guards differ: {}".format(changed))
    board = board_contract["board"]
    if int(board["udmabuf_bytes"]) < int(derivation["total_capacity_bytes"]):
        raise ValueError("u-dma-buf cannot hold the certified command capacity")
    qualification = {
        "schema": "c3_vta_command_resource_board_qualification_v1",
        "resource_certificate_key": certificate["certificate_key"],
        "boot_id": board["boot_id"],
        "boot_id_bound": True,
        "fpga_state": board["fpga_state"],
        "udmabuf_bytes": int(board["udmabuf_bytes"]),
        "capacity_runtime_sha256": board["remote_capacity_runtime_sha256"],
        "queue_capacity_status": queue_status,
        "command_manifest_id": queue_status.get("command_manifest_id"),
        "replay_policy": queue_status.get("replay_policy", "unreported_legacy_runtime"),
        "correctness": sorted(qualified_rows, key=lambda row: row["candidate_id"]),
        "board_claim_boundary": board_summary["claim_boundary"],
        "performance_measurement": "not_collected",
        "replay_observed": False,
    }
    qualification["qualification_key"] = canonical_hash(qualification)
    result = copy.deepcopy(certificate)
    result["status"] = "qualified_reduced_capacity"
    result["board_qualification"] = qualification
    result["claim_boundary"] = (
        "exact single-queue W05 deployment allowlist under the bound FPGA/runtime evidence; "
        "no replay, allocator-fragmentation, latency, FPS, concurrent-queue, or cross-boot claim"
    )
    validate_certificate(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certificate", required=True)
    parser.add_argument("--board-run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    certificate_path = Path(args.certificate).resolve()
    board_dir = Path(args.board_run_dir).resolve()
    inputs = {"certificate": verify_frozen_file(certificate_path)}
    board_paths = {
        "board_summary": board_dir / "summary.json",
        "correctness": board_dir / "correctness/correctness.jsonl",
        "queue_status": board_dir / "queue_capacity_status.json",
        "preregistered": board_dir / "preregistered.json",
    }
    board_hashes = load_json(board_dir / "artifact_hashes.json")["artifacts"]
    for name, path in board_paths.items():
        expected = board_hashes.get(str(path.relative_to(board_dir)))
        observed = sha256_file(path)
        if expected != observed:
            raise ValueError("board artifact hash mismatch: {}".format(path))
        inputs[name] = {"path": str(path), "sha256": observed}
    preregistered = load_json(board_paths["preregistered"])
    contract_path = Path(preregistered["contract"]["path"])
    if sha256_file(contract_path) != preregistered["contract"]["sha256"]:
        raise ValueError("board contract hash mismatch")
    inputs["board_contract"] = {
        "path": str(contract_path),
        "sha256": preregistered["contract"]["sha256"],
    }
    result = qualify_certificate(
        load_json(certificate_path),
        load_json(board_paths["board_summary"]),
        load_jsonl(board_paths["correctness"]),
        load_json(board_paths["queue_status"]),
        load_json(contract_path),
    )
    result["board_qualification"]["qualifier_sha256"] = sha256_file(__file__)
    result["board_qualification"].pop("qualification_key", None)
    result["board_qualification"]["qualification_key"] = canonical_hash(
        result["board_qualification"]
    )
    result["qualification_inputs"] = inputs
    result_path = output / "qualified_command_resource_certificate.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (output / "STATUS.md").write_text(
        "# Qualified VTA command-resource certificate\n\n"
        "- Status: `qualified_reduced_capacity`\n"
        "- Exact identities: {}\n"
        "- Capacity instruction/UOP: {}/{} B\n"
        "- FPGA seed checks: {}\n"
        "- Replay: disabled and not claimed\n"
        "- Performance: not collected\n".format(
            len(result["identities"]),
            result["derivation"]["insn_capacity_bytes"],
            result["derivation"]["uop_capacity_bytes"],
            sum(row["passed_seed_count"] for row in result["board_qualification"]["correctness"]),
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
        "certificate_key": result["certificate_key"],
        "qualification_key": result["board_qualification"]["qualification_key"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
