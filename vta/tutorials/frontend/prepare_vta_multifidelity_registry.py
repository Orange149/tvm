#!/usr/bin/env python3
"""Freeze v2 candidates and a multi-fidelity policy before target observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import replay_vta_multifidelity_search as search
from run_vta_p7r119_yolo_barrier_pilot import load_jsonl, sha256_file, verify_frozen, write_json
from run_vta_p7r127_y00_v2_local_pool import build_candidates, canonical_sha256


SCHEMA = "c3_vta_multifidelity_registry_v1"


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable registry " + str(output))
    input_certificate = verify_frozen(args.candidates)
    candidates = build_candidates(
        load_jsonl(args.candidates), args.workload_id,
        "c3_{}_multifidelity_candidate_v1".format(args.workload_id.lower()),
    )
    policy = {
        "name": "hardware_diverse_frontier_service",
        "frontier_width": int(args.frontier_width),
        "promotion_width": int(args.promotion_width),
        "validity_model_in_primary_policy": False,
        "initialization": (
            "complete an opened same-tile family, otherwise choose the family with maximum "
            "minimum normalized Euclidean distance in knob/SRAM/opportunity space"
        ),
        "promotion": (
            "lower four visible candidates; FSim candidates by currently visible service cost; "
            "compile two FSim-pass candidates; choose FPGA candidate by full service cost"
        ),
        "service_proxy": {
            "formula": "B_dma + 65536*N_dma + 131072*max(N_submit-1,0)",
            "request_equivalent_bytes": search.REQUEST_EQUIVALENT_BYTES,
            "extra_submission_equivalent_bytes": search.EXTRA_SUBMISSION_EQUIVALENT_BYTES,
            "semantics": "acquisition penalties, not physical AXI bytes or fitted latency constants",
        },
        "correctness": "FPGA first-error stop; a pass requires all three frozen seeds",
        "stop_for_confirmation": (
            "record first online candidate reaching the target only after the complete pool oracle "
            "is constructed; continue with unbiased completion for oracle labels"
        ),
    }
    if policy["frontier_width"] != 4 or policy["promotion_width"] != 2:
        raise ValueError("formal Y05 policy is frozen at 4-to-2-to-1")
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_y05_lower_fsim_fpga_or_latency",
        "workload_id": args.workload_id,
        "candidate_source": input_certificate,
        "candidate_count": len(candidates),
        "candidate_commitment_sha256": canonical_sha256(candidates),
        "candidate_order": [row["candidate_id"] for row in candidates],
        "policy": policy,
        "development_evidence": {
            "path": str(Path(args.development_replay).resolve()),
            "artifact_hashes_sha256": sha256_file(
                Path(args.development_replay) / "artifact_hashes.json"
            ),
            "status": "all workloads exposed; hyperparameter development only",
        },
        "source_sha256": {
            str(Path(search.__file__).resolve()): sha256_file(search.__file__),
            str(Path(__file__).resolve()): sha256_file(__file__),
        },
        "baselines": [
            "exhaustive_then_service",
            "random_frontier_service",
            "validity_frontier_service",
            "hardware_diverse_validity_frontier_service",
        ],
        "information_boundary": {
            "F0": "candidate ConfigEntity, mode, parsed SRAM/opportunity features",
            "F1": "exact DMA/request/TIR only after paid lowering",
            "F2": "command peaks/submissions only after paid FSim",
            "F3": "binary hash only after paid cross compile",
            "F4": "correctness/runtime resources only after paid clean-start FPGA canary",
            "F5": "latency only after paid measurement",
        },
        "failure_policy": "retain every failure; no candidate replacement or target-driven retuning",
        "clean_start_contract": (
            "stop exact RPC, reload frozen bitstream, start fresh RPC before any target allocation; "
            "reuse one shape-compatible output-data-weight buffer set without changing layout"
        ),
        "claim_boundary": (
            "candidate and online policy freeze only; no Y05 legality, correctness, latency, "
            "oracle, search saving, or generalization claim"
        ),
    }
    output.mkdir(parents=True)
    write_json(output / "registry_contract.json", contract)
    (output / "candidates_v2.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates), encoding="utf-8"
    )
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    artifacts = {
        path.name: sha256_file(path) for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(output / "artifact_hashes.json", {
        "artifacts": artifacts,
        "source_sha256": {str(Path(__file__).resolve()): sha256_file(__file__)},
    })
    print(json.dumps(contract, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--workload-id", default="Y05")
    parser.add_argument("--development-replay", type=Path, required=True)
    parser.add_argument("--frontier-width", type=int, default=4)
    parser.add_argument("--promotion-width", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
