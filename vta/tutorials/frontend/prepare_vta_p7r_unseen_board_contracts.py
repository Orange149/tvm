#!/usr/bin/env python3
"""Freeze independent E00--E02 real-FPGA correctness contracts."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist", required=True)
    parser.add_argument("--fsim", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    shortlist = load_jsonl(args.shortlist)
    fsim = {row["candidate_id"]: row for row in load_jsonl(args.fsim)}
    if len(shortlist) != 6 or set(row["workload_id"] for row in shortlist) != {"E00", "E01", "E02"}:
        raise ValueError("shortlist must contain two candidates for each E00--E02")
    if any(fsim.get(row["candidate_id"], {}).get("overall_status") != "passed" for row in shortlist):
        raise ValueError("every unseen shortlist candidate must pass FSim")

    root = Path(__file__).resolve().parents[3]
    source_paths = [
        root / "vta/tutorials/frontend/run_vta_p7r_joint_correctness.py",
        root / "vta/python/vta/top/vta_conv2d.py",
        root / "vta/python/vta/top/vta_conv2d_residency.py",
        root / "vta/tutorials/frontend/qualify_vta_residency_fsim.py",
    ]
    input_guards = {
        str(Path(args.shortlist)): sha256_file(args.shortlist),
        str(Path(args.fsim)): sha256_file(args.fsim),
    }
    contracts = {}
    for workload_id in ("E00", "E01", "E02"):
        entries = [row for row in shortlist if row["workload_id"] == workload_id]
        by_stratum = {row["p7r_selection"]["stratum"]: row for row in entries}
        shape = by_stratum["request_shape"]
        control = by_stratum["contiguous_control"]

        def original(row, role):
            return {"role": role, "mode": "original", "config_index": row["config_index"]}

        def residency(row, role):
            return {
                "role": role,
                "mode": "input_stationary",
                "config_index": row["config_index"],
                "candidate_id": row["candidate_id"],
                "tir_sha256": row["tir_sha256"],
            }

        contract = {
            "schema": "c3_p7r_unseen_board_contract_v1",
            "status": "frozen_before_p7r_board_labels",
            "purpose": "prospective request-shape sign test; correctness only",
            "workload_id": workload_id,
            "workload": shape["identity"]["workload"],
            "seeds": [0, 20250901, 20260910],
            "selection_rule": {
                "uses_this_workload_latency_labels": False,
                "request_shape_prediction": shape["p7r_selection"],
                "contiguous_control_prediction": control["p7r_selection"],
                "timing_gate": "all fifteen seed checks pass",
            },
            "execution_order": [
                original(shape, "request_shape_original_before"),
                residency(shape, "request_shape_residency"),
                original(control, "contiguous_original"),
                residency(control, "contiguous_residency"),
                original(shape, "request_shape_original_after"),
            ],
            "board": {
                "boot_id": "a68a7983-719f-47bf-94d7-41f974c5342c",
                "fpga_state": "operating",
                "udmabuf_bytes": 201326592,
                "rpc_cwd_requirement": "/var/volatile/vta_c3_ram/runtime",
                "sd_mount": "/media/sd-mmcblk1p2",
                "sd_used_percent_at_freeze": 78,
                "dmesg_error_cutoff_seconds": 4057.401259,
                "safety": "abort without reboot/poweroff on any mismatch, new storage error, fingerprint change, or timeout",
            },
            "source_guards_sha256": {str(path.relative_to(root)): sha256_file(path) for path in source_paths},
            "input_guards_sha256": input_guards,
        }
        path = output / (workload_id.lower() + "_contract.json")
        path.write_text(json.dumps(contract, indent=2) + "\n")
        contracts[workload_id] = {"path": str(path), "sha256": sha256_file(path)}

    protocol = {
        "schema": "c3_p7r_unseen_board_contract_set_v1",
        "status": "all_three_frozen_before_first_unseen_board_label",
        "contracts": contracts,
        "failure_policy": "one workload failure stops that workload only; other already-frozen contracts remain selectable without adaptation",
        "performance_measurement": "forbidden_until_each_workload_correctness_contract_passes",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    hashes = {
        path.name: sha256_file(path)
        for path in output.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_text(json.dumps({"artifacts": hashes}, indent=2) + "\n")
    print(json.dumps(contracts, indent=2))


if __name__ == "__main__":
    main()
