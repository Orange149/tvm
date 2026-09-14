#!/usr/bin/env python3
"""Freeze FPGA correctness contracts for support-transferred VTA candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require_fsim(path, expected_ids):
    rows = load_jsonl(path)
    if {row["candidate_id"] for row in rows} != set(expected_ids):
        raise ValueError("FSim candidate coverage mismatch")
    if not all(row.get("overall_status") == "passed" for row in rows):
        raise ValueError("FSim gate failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist", required=True)
    parser.add_argument("--residency-fsim", required=True)
    parser.add_argument("--original-pool", required=True)
    parser.add_argument("--original-fsim", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)

    shortlist = load_jsonl(args.shortlist)
    if len(shortlist) != 3 or len({row["workload_id"] for row in shortlist}) != 3:
        raise ValueError("expected one candidate for each of three workloads")
    originals = load_jsonl(args.original_pool)
    require_fsim(args.residency_fsim, [row["candidate_id"] for row in shortlist])
    require_fsim(args.original_fsim, [row["candidate_id"] for row in originals])
    original_by_key = {(row["workload_id"], row["debug"]["config_index"]): row
                       for row in originals}
    if {(row["workload_id"], row["config_index"]) for row in shortlist} != set(original_by_key):
        raise ValueError("same-tile original coverage mismatch")

    root = Path(__file__).resolve().parents[3]
    sources = [
        root / "vta/tutorials/frontend/run_vta_p7r_joint_correctness.py",
        root / "vta/python/vta/top/vta_conv2d.py",
        root / "vta/python/vta/top/vta_conv2d_residency.py",
    ]
    input_paths = [Path(value).resolve() for value in (
        args.shortlist, args.residency_fsim, args.original_pool, args.original_fsim
    )]
    contracts = {}
    for row in shortlist:
        workload_id, index = row["workload_id"], row["config_index"]
        contract = {
            "schema": "c3_p7r_unseen_board_contract_v1",
            "status": "frozen_before_p7r_board_labels",
            "purpose": "prospective support-transfer correctness gate; no latency",
            "workload_id": workload_id,
            "workload": row["identity"]["workload"],
            "seeds": [0, 20250901, 20260910],
            "selection_rule": row["p7r_selection"],
            "execution_order": [
                {"role": "transferred_original_before", "mode": "original",
                 "config_index": index},
                {"role": "transferred_residency", "mode": "input_stationary",
                 "config_index": index, "candidate_id": row["candidate_id"],
                 "tir_sha256": row["tir_sha256"]},
                {"role": "transferred_original_after", "mode": "original",
                 "config_index": index},
            ],
            "board": {
                "boot_id": "a68a7983-719f-47bf-94d7-41f974c5342c",
                "fpga_state": "operating",
                "udmabuf_bytes": 201326592,
                "rpc_cwd_requirement": "/var/volatile/vta_c3_ram/runtime",
                "sd_mount": "/media/sd-mmcblk1p2",
                "sd_used_percent_at_freeze": 78,
                "dmesg_error_cutoff_seconds": 4057.401259,
                "safety": "abort without reboot/poweroff on mismatch, new storage error, fingerprint change, or timeout",
            },
            "source_guards_sha256": {
                str(path.relative_to(root)): sha256_file(path) for path in sources
            },
            "input_guards_sha256": {str(path): sha256_file(path) for path in input_paths},
        }
        path = output / (workload_id.lower() + "_contract.json")
        path.write_text(json.dumps(contract, indent=2) + "\n")
        contracts[workload_id] = {"path": str(path), "sha256": sha256_file(path)}

    protocol = {
        "schema": "c3_p7r_support_transfer_board_contract_set_v1",
        "status": "all_contracts_frozen_before_first_target_fpga_label",
        "contracts": contracts,
        "per_workload_gate": "original-before, residency, original-after; three seeds each",
        "failure_policy": "a mismatch rejects that exact candidate before timing; other already-frozen contracts remain independent",
        "performance_measurement": "forbidden until a workload passes all nine checks",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (output / "STATUS.md").write_text(
        "# Support-transfer FPGA contracts\n\n"
        "- Status: `all_contracts_frozen_before_first_target_fpga_label`\n"
        "- Workloads: {}\n- Planned seed checks: 27\n"
        "- Performance measurement: forbidden until each independent contract passes.\n".format(
            ", ".join(sorted(contracts))
        )
    )
    hashes = {path.name: sha256_file(path) for path in output.iterdir()
              if path.is_file() and path.name != "artifact_hashes.json"}
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": hashes}, indent=2) + "\n"
    )
    print(json.dumps(contracts, indent=2))


if __name__ == "__main__":
    main()
