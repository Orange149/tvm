#!/usr/bin/env python3
"""Freeze candidates and the survival-adaptive policy for a fresh holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import replay_vta_adaptive_multifidelity_search as adaptive
import replay_vta_multifidelity_search as base
from run_vta_p7r119_yolo_barrier_pilot import load_jsonl, sha256_file, verify_frozen, write_json
from run_vta_p7r127_y00_v2_local_pool import build_candidates, canonical_sha256


SCHEMA = "c3_vta_adaptive_multifidelity_registry_v1"


def verify(directory):
    directory = Path(directory)
    ledger = json.loads((directory / "artifact_hashes.json").read_text(encoding="utf-8"))
    for name, expected in ledger["artifacts"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError("artifact hash mismatch: " + str(directory / name))
    return sha256_file(directory / "artifact_hashes.json")


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    source = verify_frozen(args.candidates)
    candidates = build_candidates(
        load_jsonl(args.candidates), args.workload_id,
        "c3_{}_adaptive_multifidelity_candidate_v1".format(args.workload_id.lower()),
    )
    development_dir = Path(args.development_dir)
    development_ledger = verify(development_dir)
    policy = {
        "name": adaptive.POLICY,
        "first_family_modes": 3,
        "first_family_selection": "hardware max-min diversity in F0 knob/SRAM/opportunity space",
        "sparse_threshold_lower_passes": 1,
        "sparse_path": "immediate minimum-service family-wave promotion",
        "dense_path": {"frontier_width": 4, "promotion_width": 2,
                       "policy": "hardware_diverse_frontier_service"},
        "service_proxy": {
            "formula": "B_dma + 65536*N_dma + 131072*max(N_submit-1,0)",
            "request_equivalent_bytes": base.REQUEST_EQUIVALENT_BYTES,
            "extra_submission_equivalent_bytes": base.EXTRA_SUBMISSION_EQUIVALENT_BYTES,
        },
        "decision_visibility": "only the three paid lowering outcomes of the first family",
        "validity_model_in_primary_policy": False,
    }
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_target_lower_fsim_fpga_or_latency",
        "workload_id": args.workload_id,
        "candidate_source": source,
        "candidate_count": len(candidates),
        "candidate_commitment_sha256": canonical_sha256(candidates),
        "candidate_order": [row["candidate_id"] for row in candidates],
        "policy": policy,
        "development_evidence": {
            "path": str(development_dir.resolve()),
            "ledger_sha256": development_ledger,
            "status": "Y00/Y01/Y03/Y04/Y05 labels exposed; development only",
        },
        "source_sha256": {
            str(Path(adaptive.__file__).resolve()): sha256_file(adaptive.__file__),
            str(Path(base.__file__).resolve()): sha256_file(base.__file__),
            str(Path(__file__).resolve()): sha256_file(__file__),
        },
        "failure_policy": "retain all failures; no target-driven replacement or threshold change",
        "clean_start_contract": (
            "stop exact RPC, reload frozen bitstream, start fresh RPC before target allocation; "
            "preserve output-data-weight physical allocation order"
        ),
        "claim_boundary": "policy/candidates only; no target outcome has been observed",
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
    artifacts = {path.name: sha256_file(path) for path in sorted(output.iterdir())
                 if path.is_file() and path.name != "artifact_hashes.json"}
    write_json(output / "artifact_hashes.json", {
        "artifacts": artifacts,
        "source_sha256": {str(Path(__file__).resolve()): sha256_file(__file__)},
    })
    print(json.dumps(contract, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--development-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
