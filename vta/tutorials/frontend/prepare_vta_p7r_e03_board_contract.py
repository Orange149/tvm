#!/usr/bin/env python3
"""Freeze the E03 real-FPGA correctness contract after both FSim gates pass."""

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
        raise ValueError("FSim candidate coverage mismatch: {}".format(path))
    if not all(row.get("overall_status") == "passed" for row in rows):
        raise ValueError("FSim gate failed: {}".format(path))


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
    if len(shortlist) != 2 or {row["workload_id"] for row in shortlist} != {"E03"}:
        raise ValueError("shortlist must contain two E03 candidates")
    by_stratum = {row["p7r_selection"]["stratum"]: row for row in shortlist}
    priority = by_stratum["pareto_priority"]
    boundary = by_stratum["pareto_boundary"]
    require_fsim(args.residency_fsim, [row["candidate_id"] for row in shortlist])
    originals = load_jsonl(args.original_pool)
    original_by_config = {row["debug"]["config_index"]: row for row in originals}
    require_fsim(args.original_fsim, [row["candidate_id"] for row in originals])
    if set(original_by_config) != {priority["config_index"], boundary["config_index"]}:
        raise ValueError("same-tile original pool mismatch")

    root = Path(__file__).resolve().parents[3]
    sources = [
        root / "vta/tutorials/frontend/run_vta_p7r_joint_correctness.py",
        root / "vta/tutorials/frontend/run_vta_p7r_unseen_timing.py",
        root / "vta/python/vta/top/vta_conv2d.py",
        root / "vta/python/vta/top/vta_conv2d_residency.py",
    ]

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

    input_paths = [Path(value).resolve() for value in (
        args.shortlist, args.residency_fsim, args.original_pool, args.original_fsim
    )]
    contract = {
        "schema": "c3_p7r_unseen_board_contract_v1",
        "status": "frozen_before_p7r_board_labels",
        "purpose": "prospective E03 DMA-Pareto priority versus boundary test; correctness first",
        "workload_id": "E03",
        "workload": priority["identity"]["workload"],
        "seeds": [0, 20250901, 20260910],
        "selection_rule": {
            "uses_this_workload_latency_labels": False,
            "priority": priority["p7r_selection"],
            "boundary": boundary["p7r_selection"],
            "primary_hypothesis": "same-tile speedup(priority) > same-tile speedup(boundary)",
            "final_choice": "minimum measured correct latency including original fallback",
        },
        "execution_order": [
            original(priority, "request_shape_original_before"),
            residency(priority, "request_shape_residency"),
            original(boundary, "contiguous_original"),
            residency(boundary, "contiguous_residency"),
            original(priority, "request_shape_original_after"),
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
    contract_path = output / "e03_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    (output / "STATUS.md").write_text(
        "# E03 board contract\n\n"
        "- Status: `frozen_before_p7r_board_labels`\n"
        "- Residency FSim: 2/2 passed; original FSim: 2/2 passed\n"
        "- Board at freeze: operating, tmpfs RPC, unchanged boot ID, no new storage error\n"
        "- Performance measurement remains forbidden until all 15 FPGA seed checks pass.\n"
    )
    hashes = {path.name: sha256_file(path) for path in output.iterdir()
              if path.is_file() and path.name != "artifact_hashes.json"}
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": hashes}, indent=2) + "\n"
    )
    print(json.dumps({"contract": str(contract_path), "sha256": sha256_file(contract_path)}, indent=2))


if __name__ == "__main__":
    main()
