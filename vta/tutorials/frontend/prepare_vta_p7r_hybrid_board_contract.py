#!/usr/bin/env python3
"""Freeze the W05 bounded-hybrid FPGA correctness contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import tvm
import vta

from run_vta_p7r_joint_correctness import make_task


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


def lower_tir_hash(workload, mode, index, env):
    task = make_task(workload, mode, env)
    config = task.config_space.get(int(index))
    with task.target:
        schedule, tensors = task.instantiate(config)
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        lowered = tvm.lower(schedule, tensors, name="main")
    return hashlib.sha256(tvm.ir.save_json(lowered).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist", required=True)
    parser.add_argument("--hybrid-fsim", required=True)
    parser.add_argument("--original-pool", required=True)
    parser.add_argument("--original-fsim", required=True)
    parser.add_argument("--cross-results", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)

    paths = {name: Path(value).resolve() for name, value in (
        ("shortlist", args.shortlist), ("hybrid_fsim", args.hybrid_fsim),
        ("original_pool", args.original_pool), ("original_fsim", args.original_fsim),
        ("cross_results", args.cross_results),
    )}
    shortlist = load_jsonl(paths["shortlist"])
    originals = load_jsonl(paths["original_pool"])
    if [row["config_index"] for row in shortlist] != [574, 494]:
        raise ValueError("frozen shortlist identity changed")
    if not all(row["residence_mode"] == "paper_inspired_hybrid" for row in shortlist):
        raise ValueError("unexpected residency mode")
    require_fsim(paths["hybrid_fsim"], [row["candidate_id"] for row in shortlist])
    require_fsim(paths["original_fsim"], [row["candidate_id"] for row in originals])
    original_by_index = {int(row["debug"]["config_index"]): row for row in originals}
    if set(original_by_index) != {574, 494}:
        raise ValueError("same-tile original coverage mismatch")
    cross = load_jsonl(paths["cross_results"])
    if len(cross) != 4 or not all(row.get("status") == "passed" for row in cross):
        raise ValueError("AXU cross-compile gate failed")

    workload = shortlist[0]["identity"]["workload"]
    if not all(row["identity"]["workload"] == workload for row in shortlist + originals):
        raise ValueError("workload identity mismatch")
    env = vta.get_env()
    if env.TARGET != "axu5evb":
        raise RuntimeError("VTA TARGET must be axu5evb")
    incumbent_tir = lower_tir_hash(workload, "original", 575, env)

    execution_order = [{
        "role": "tophub_incumbent_before",
        "mode": "original",
        "config_index": 575,
        "tir_sha256": incumbent_tir,
    }]
    for row in shortlist:
        index = int(row["config_index"])
        execution_order.extend([
            {
                "role": "hybrid_same_tile_original",
                "mode": "original",
                "config_index": index,
                "candidate_id": original_by_index[index]["candidate_id"],
                "tir_sha256": original_by_index[index]["tir_sha256"],
            },
            {
                "role": "bounded_hybrid_t2",
                "mode": "paper_inspired_hybrid",
                "config_index": index,
                "candidate_id": row["candidate_id"],
                "tir_sha256": row["tir_sha256"],
            },
        ])
    execution_order.append({
        "role": "tophub_incumbent_after",
        "mode": "original",
        "config_index": 575,
        "tir_sha256": incumbent_tir,
    })

    root = Path(__file__).resolve().parents[3]
    sources = [
        root / "vta/tutorials/frontend/run_vta_p7r_joint_correctness.py",
        root / "vta/python/vta/top/vta_conv2d.py",
        root / "vta/python/vta/top/vta_conv2d_residency.py",
    ]
    protocol = json.loads((paths["shortlist"].parent / "protocol.json").read_text())
    contract = {
        "schema": "c3_p7r_unseen_board_contract_v1",
        "status": "frozen_before_p7r_board_labels",
        "purpose": "bounded-hybrid t2 correctness gate before any latency",
        "workload_id": "W05",
        "workload": workload,
        "seeds": [0, 20250901, 20260910],
        "selection_rule": {
            "source_protocol": protocol,
            "performance_labels_used": False,
            "frozen_configs": [574, 494],
            "protected_incumbent": 575,
        },
        "execution_order": execution_order,
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
        "input_guards_sha256": {str(path): sha256_file(path) for path in paths.values()},
    }
    contract_path = output / "w05_hybrid_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    (output / "STATUS.md").write_text(
        "# W05 bounded-hybrid FPGA contract\n\n"
        "- Status: `frozen_before_p7r_board_labels`\n"
        "- Order: TopHub before; original/hybrid 574; original/hybrid 494; TopHub after\n"
        "- Planned exact seed checks: 18\n"
        "- Performance timing: forbidden until 18/18 pass\n"
    )
    hashes = {
        path.name: sha256_file(path) for path in output.iterdir()
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    (output / "artifact_hashes.json").write_text(
        json.dumps({"artifacts": hashes}, indent=2) + "\n"
    )
    print(json.dumps({"contract": str(contract_path), "sha256": sha256_file(contract_path)}, indent=2))


if __name__ == "__main__":
    main()
