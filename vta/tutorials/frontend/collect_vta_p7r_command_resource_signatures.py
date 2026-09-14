#!/usr/bin/env python3
"""Collect timing-free command-resource signatures for the frozen W05 winner path.

This is a local FSim structural experiment.  It does not contact the board and
does not reinterpret diagnostic timings as performance measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import tvm
import vta

from collect_vta_full_pool_fsim_commands import (
    run_candidate_no_rpc,
    sanitize_stderr,
    strip_diagnostic_timing,
)
from run_vta_p7r_joint_correctness import make_task


MODE_NUMBERS = {"original": 0, "paper_inspired_hybrid": 3}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify_frozen_file(path):
    path = Path(path).resolve()
    ledger = load_json(path.parent / "artifact_hashes.json")
    hashes = ledger.get("artifacts", ledger.get("output_sha256", ledger))
    expected = hashes.get(path.name)
    observed = sha256_file(path)
    if expected != observed:
        raise ValueError("frozen artifact hash mismatch: {}".format(path))
    return {"path": str(path), "sha256": observed}


def lower_tir(task, config):
    with task.target:
        schedule, tensors = task.instantiate(config)
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        lowered = tvm.lower(schedule, tensors, name="main")
    return hashlib.sha256(tvm.ir.save_json(lowered).encode()).hexdigest()


def stable_id(identity, mode):
    payload = json.dumps(
        {"identity": identity, "residence_mode": mode},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-contract", required=True)
    parser.add_argument("--shortlist", required=True)
    parser.add_argument("--original-pool", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--hybrid-configs",
        default="574",
        help="Comma-separated frozen hybrid config indices. Default preserves the original W05 run.",
    )
    parser.add_argument("--candidate-timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)

    inputs = {
        "board_contract": verify_frozen_file(args.board_contract),
        "shortlist": verify_frozen_file(args.shortlist),
        "original_pool": verify_frozen_file(args.original_pool),
    }
    contract = load_json(args.board_contract)
    shortlist = load_jsonl(args.shortlist)
    originals = load_jsonl(args.original_pool)
    hybrid_configs = [int(value) for value in args.hybrid_configs.split(",") if value.strip()]
    if not hybrid_configs or len(set(hybrid_configs)) != len(hybrid_configs):
        raise ValueError("--hybrid-configs must contain unique config indices")
    hybrid_by_config = {int(row["config_index"]): row for row in shortlist}
    original_by_config = {int(row["debug"]["config_index"]): row for row in originals}
    missing_hybrid = sorted(set(hybrid_configs) - set(hybrid_by_config))
    missing_original = sorted(set(hybrid_configs) - set(original_by_config))
    if missing_hybrid or missing_original:
        raise ValueError(
            "frozen candidate inputs are incomplete: hybrid={} original={}".format(
                missing_hybrid, missing_original
            )
        )
    incumbent_tir = next(
        row["tir_sha256"]
        for row in contract["execution_order"]
        if row["role"] == "tophub_incumbent_before"
    )
    workload = contract["workload"]
    for config_index in hybrid_configs:
        if (
            hybrid_by_config[config_index]["identity"]["workload"] != workload
            or original_by_config[config_index]["identity"]["workload"] != workload
        ):
            raise ValueError("W05 workload identity mismatch")

    root = Path(__file__).resolve().parents[3]
    guarded_sources = [
        root / "vta/python/vta/top/vta_conv2d.py",
        root / "vta/python/vta/top/vta_conv2d_residency.py",
        root / "vta/runtime/runtime.cc",
        root / "vta/runtime/queue_capacity.h",
        root / "vta/tutorials/frontend/collect_vta_full_pool_fsim_commands.py",
    ]
    for relpath, expected in contract["source_guards_sha256"].items():
        if relpath.endswith(("vta_conv2d.py", "vta_conv2d_residency.py")):
            if sha256_file(root / relpath) != expected:
                raise ValueError("board-contract schedule source guard changed: {}".format(relpath))

    env = vta.get_env()
    if env.TARGET != "sim":
        raise RuntimeError("VTA TARGET must be sim")
    incumbent_task = make_task(workload, "original", env)
    incumbent_config = incumbent_task.config_space.get(575)
    if lower_tir(incumbent_task, incumbent_config) != incumbent_tir:
        raise ValueError("TopHub config575 TIR differs from frozen board contract")
    incumbent_identity = {
        "complete_config_entity": incumbent_config.to_json_dict(),
        "schedule_sha256": sha256_file(root / "vta/python/vta/top/vta_conv2d.py"),
        "workload": workload,
    }

    candidates = [
        {
            "candidate_id": stable_id(incumbent_identity, "original"),
            "candidate_role": "protected_tophub_incumbent",
            "workload_id": "W05",
            "public_mode": "original",
            "implementation_mode": 0,
            "identity": incumbent_identity,
            "debug": {"config_index": 575},
            "expected_tir_sha256": incumbent_tir,
            "expected_residency_drains": 0,
        },
    ]
    single_config = len(hybrid_configs) == 1
    for config_index in hybrid_configs:
        original = original_by_config[config_index]
        hybrid = hybrid_by_config[config_index]
        candidates.extend(
            [
                {
                    "candidate_id": original["candidate_id"],
                    "candidate_role": (
                        "same_tile_original"
                        if single_config
                        else "same_tile_original_config{}".format(config_index)
                    ),
                    "workload_id": "W05",
                    "public_mode": "original",
                    "implementation_mode": 0,
                    "identity": original["identity"],
                    "debug": original["debug"],
                    "expected_tir_sha256": original["tir_sha256"],
                    "expected_residency_drains": 0,
                },
                {
                    "candidate_id": hybrid["candidate_id"],
                    "candidate_role": (
                        "bounded_hybrid_candidate"
                        if single_config
                        else "bounded_hybrid_candidate_config{}".format(config_index)
                    ),
                    "workload_id": "W05",
                    "public_mode": "paper_inspired_hybrid",
                    "implementation_mode": MODE_NUMBERS["paper_inspired_hybrid"],
                    "identity": hybrid["identity"],
                    "debug": hybrid["debug"],
                    "expected_tir_sha256": hybrid["tir_sha256"],
                    "expected_residency_drains": 0,
                },
            ]
        )
    candidates_path = output / "candidates.jsonl"
    candidates_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates)
    )
    protocol = {
        "schema": "c3_p7r_command_resource_contract_v1",
        "status": "frozen_before_command_signatures",
        "scope": (
            "W05 TopHub575 and exact original/hybrid config set {}; local FSim only".format(
                hybrid_configs
            )
        ),
        "hybrid_configs": hybrid_configs,
        "candidate_order": [row["candidate_role"] for row in candidates],
        "hypotheses": [
            "all three frozen TIR/config identities pass seed-0 exact correctness",
            "instruction and UOP peaks are far below the legacy 32 MiB per-queue backing",
            "a later capacity plan must include FINISH and use 256-byte-aligned capacities",
        ],
        "claim_boundary": "no board capacity, latency, FPS, full-model, or replay claim",
        "inputs": inputs,
        "source_guards_sha256": {
            str(path.relative_to(root)): sha256_file(path) for path in guarded_sources
        },
    }
    protocol_path = output / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    pre_measure = {
        "candidates.jsonl": sha256_file(candidates_path),
        "protocol.json": sha256_file(protocol_path),
    }
    (output / "pre_measure_hashes.json").write_text(
        json.dumps({"artifacts": pre_measure}, indent=2, sort_keys=True) + "\n"
    )

    results = []
    stderr_sections = []
    for candidate in candidates:
        result, stderr = run_candidate_no_rpc(candidate, args.candidate_timeout_seconds)
        strip_diagnostic_timing(result)
        result["candidate_role"] = candidate["candidate_role"]
        results.append(result)
        stderr_sections.append(
            "candidate_role={} candidate_id={}\n{}".format(
                candidate["candidate_role"], candidate["candidate_id"], sanitize_stderr(stderr)
            )
        )
    results_path = output / "results.jsonl"
    results_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in results))
    (output / "stderr.log").write_text("\n".join(stderr_sections))
    passed = [row for row in results if row.get("outcome") == "passed_local_signature"]
    summary = {
        "schema": "c3_p7r_command_resource_summary_v1",
        "status": "completed" if len(passed) == len(results) else "failed",
        "selected": len(results),
        "passed_local_signature": len(passed),
        "performance_measurement": "not_collected",
        "roles": {
            row["candidate_role"]: {
                "outcome": row.get("outcome"),
                "insn_peak_bytes": row.get("command_signature", {})
                .get("structural", {})
                .get("peaks", {})
                .get("insn_bytes"),
                "uop_peak_bytes": row.get("command_signature", {})
                .get("structural", {})
                .get("peaks", {})
                .get("uop_bytes"),
            }
            for row in results
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "STATUS.md").write_text(
        "# P7R W05 command-resource signatures\n\n"
        "- Status: `{}`\n"
        "- Local FSim signatures passed: {}/{}\n"
        "- Performance timing: not collected\n"
        "- Board contacted: no\n".format(summary["status"], len(passed), len(results))
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
