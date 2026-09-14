#!/usr/bin/env python3
"""Validate semantic dispatch -> AutoTVM task -> command-resource certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import vta

from build_vta_command_resource_certificate import sha256_file, verify_frozen_file
from tune_resnet18_vta import (
    certified_config_signatures,
    config_signature,
    load_certified_candidate_ids,
    load_command_resource_contract,
    task_from_entry,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch", required=True)
    parser.add_argument("--command-resource-certificate", required=True)
    parser.add_argument("--board-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    paths = {
        "dispatch": Path(args.dispatch).resolve(),
        "command_resource_certificate": Path(args.command_resource_certificate).resolve(),
        "board_contract": Path(args.board_contract).resolve(),
    }
    inputs = {name: verify_frozen_file(path) for name, path in paths.items()}
    contract = json.loads(paths["board_contract"].read_text(encoding="utf-8"))
    env = vta.get_env()
    entry = {
        "workload_id": contract["workload_id"],
        "workload": contract["workload"],
        "residence_mode": "paper_inspired_hybrid",
    }
    task = task_from_entry(entry, env.target, env.target_host)
    expected = certified_config_signatures(
        paths["dispatch"], entry["workload_id"], entry["residence_mode"]
    )
    if expected is None:
        raise ValueError("semantic chain validation requires a v2 dispatch")
    available = {
        config_signature(task.config_space.get(index)._entity_map)
        for index in range(len(task.config_space))
    }
    missing = expected - available
    if missing:
        raise ValueError("dispatch semantic identities are absent from the current ConfigSpace")
    candidate_ids = load_certified_candidate_ids(paths["dispatch"])
    root = Path(__file__).resolve().parents[3]
    resource = load_command_resource_contract(
        paths["command_resource_certificate"], candidate_ids, repo_root=root
    )
    certificate = json.loads(paths["command_resource_certificate"].read_text())
    summary = {
        "schema": "c3_vta_autotune_resource_chain_validation_v1",
        "status": "passed",
        "workload_id": entry["workload_id"],
        "residence_mode": entry["residence_mode"],
        "dispatch_semantic_identities": len(expected),
        "identities_present_in_current_config_space": len(expected),
        "dispatch_candidate_ids": sorted(candidate_ids),
        "resource_identity_count": len(resource["candidate_ids"]),
        "resource_certificate_status": certificate["status"],
        "resource_certificate_key": resource["certificate_key"],
        "deployment_runtime_environment": resource["environment"],
        "board_contacted": False,
        "performance_measurement": "not_collected",
        "claim_boundary": (
            "semantic pre-measure and provisional resource-contract integration only; "
            "no new FPGA, latency, replay, or deployment qualification claim"
        ),
        "inputs": inputs,
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "STATUS.md").write_text(
        "# VTA AutoTune/resource-certificate chain validation\n\n"
        "- Status: `passed`\n"
        "- Semantic dispatch identities found: {}/{}\n"
        "- Resource identities: {}\n"
        "- Runtime environment: `{}`\n"
        "- Board contacted: no\n".format(
            len(expected), len(expected), len(resource["candidate_ids"]),
            json.dumps(resource["environment"], sort_keys=True),
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
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
