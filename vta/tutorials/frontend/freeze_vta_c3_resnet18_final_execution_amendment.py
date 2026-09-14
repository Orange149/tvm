#!/usr/bin/env python3
"""Freeze a label-free final-protocol analysis-semantics amendment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import run_vta_c3_literature_baselines as baseline


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def verify(directory):
    directory = Path(directory).resolve()
    payload = read_json(directory / "artifact_hashes.json")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("unknown artifact ledger: {}".format(directory))
    for name, expected in artifacts.items():
        if isinstance(expected, str):
            if baseline.sha256_file(directory / name) != expected:
                raise ValueError("artifact hash mismatch: {}".format(directory / name))
        elif isinstance(expected, dict):
            for child, child_hash in expected.items():
                if baseline.sha256_file(directory / name / child) != child_hash:
                    raise ValueError(
                        "artifact hash mismatch: {}".format(directory / name / child)
                    )
        else:
            raise ValueError("unsupported artifact entry: {}".format(name))
    return baseline.sha256_file(directory / "artifact_hashes.json")


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    scan_dirs = [Path(value).resolve() for value in args.full_scan_dir]
    scans = {}
    for directory in scan_dirs:
        summary = read_json(directory / "summary.json")
        scans[summary["workload_id"]] = verify(directory)
    if set(scans) != {"R18-H1", "R18-H2", "R18-H3"}:
        raise ValueError("all three complete original scans are required")
    here = Path(__file__).resolve().parent
    source_paths = [
        here / "run_vta_c3_literature_baselines.py",
        here / "analyze_vta_c3_resnet18_literature_pool.py",
        here / "test_run_vta_c3_literature_baselines.py",
        here / "test_analyze_vta_c3_resnet18_literature_pool.py",
        here / "run_vta_from_scratch_autotvm_fullgraph.py",
        here / "run_vta_c3_resnet18_from_scratch_ours_fullgraph.py",
        here / "run_vta_c3_resnet18_board_pool.py",
        here / "run_vta_p7r132_y00_search_confirmation.py",
        here / "run_vta_c3_resnet18_fullgraph_programs.py",
        here / "prepare_vta_c3_resnet18_literature_replay.py",
        here / "test_run_vta_c3_resnet18_board_pool.py",
        here / "test_vta_c3_resnet18_fullgraph_protocol.py",
        here / "test_prepare_vta_c3_resnet18_literature_replay.py",
    ]
    input_hashes = {
        "prior_final_protocol": verify(args.prior_final_protocol_dir),
        "complete_original_contracts": verify(args.full_original_dir),
        "hw_aware_union": verify(args.hw_union_dir),
        "full_original_scans": scans,
    }
    contract = {
        "schema": "c3_resnet18_final_execution_amendment_v1",
        "status": "frozen_before_any_merged_pool_fpga_or_resnet18_fullgraph_label",
        "supersedes": {
            "manifest_sha256": input_hashes["prior_final_protocol"],
            "scope": "analysis semantics only; all 214 candidate identities, order, board binaries, seeds, budgets and full-graph protocol remain unchanged",
        },
        "corrections": {
            "ml2tuner_validity_censoring": (
                "lower/cross-compile success without FSim and FPGA observation is unknown, "
                "not a positive Model V label; terminal compile failure remains negative"
            ),
            "ml2tuner_metrics": (
                "post-pool leave-one-workload-out Model V reports precision, recall, F1, "
                "accuracy, nDCG, top-20 valid yield and base valid yield"
            ),
            "hw_aware_cost": (
                "every min(1000, |S|) original-space lowering call and measured wall time "
                "is prepaid before the first E0 performance dispatch and never double charged"
            ),
            "hw_aware_post_e0": (
                "post-E0 choice uses the rank sum of the common visible-feature performance "
                "XGB and the target-presampling lowering-validity XGB"
            ),
            "complete_cost_breakdown": (
                "all policies aggregate gross candidates, compiler attempts, FPGA dispatches, "
                "kernel invocations, DMA and per-phase lowering/FSim/cross-compile/correctness/"
                "timing wall; oracle+2% and +5% both report first trial and first wall"
            ),
            "clean_start_preflight_symmetry": (
                "both empty-history AutoTVM-XGB and the proposed H1 clean-start runner verify "
                "the same default runtime hashes and RPC working directory before T0; the "
                "proposed runner also verifies the source-contract artifact ledger"
            ),
            "long_pool_rpc_lifetime": (
                "the complete 214-point non-spliced board pool uses a 7200-second whole-session "
                "RPC lifetime; TVM session_timeout is total session duration, not per-call timeout"
            ),
            "common_audit_artifacts": (
                "the board-pool, selected-fullgraph and both clean-start runners always create "
                "contract.json, timeline.jsonl, results.jsonl, summary.json and artifact_hashes.json; "
                "an interrupted session writes a fail-closed summary plus invalid_session.json and "
                "remains non-spliceable; an already-existing output is never mutated"
            ),
            "pure_instruction_normalization": (
                "driver_run_total_us is divided by complete time-evaluator main invocations, "
                "not driver_run_calls, because one barrier-resident operator can contain many "
                "submissions; Cheng pure-instruction time is therefore per operator, and a missing "
                "driver timer is rejected rather than imputed as zero"
            ),
            "ssh_authentication": (
                "clean-start board control keeps StrictHostKeyChecking=yes and the dedicated "
                "serial-verified known_hosts file, while allowing public-key authentication "
                "before password fallback; no host-key bypass is permitted"
            ),
        },
        "required_analysis_inputs": {
            "full_original_dir": str(Path(args.full_original_dir).resolve()),
            "full_scan_dirs": [str(value) for value in scan_dirs],
        },
        "source_hashes": {
            str(path): baseline.sha256_file(path) for path in source_paths
        },
        "target_labels_read": False,
        "board_contacted": False,
    }
    output.mkdir(parents=True)
    write_json(output / "contract.json", contract)
    write_json(
        output / "summary.json",
        {
            "status": contract["status"],
            "candidate_identity_changed": False,
            "board_execution_order_changed": False,
            "analysis_semantics_corrected": True,
            "board_contacted": False,
            "target_labels_read": False,
        },
    )
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    (output / "results.jsonl").write_text("", encoding="utf-8")
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                path.name: baseline.sha256_file(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "source_sha256": baseline.sha256_file(Path(__file__).resolve()),
        },
    )
    print(json.dumps(read_json(output / "summary.json"), indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-final-protocol-dir", required=True)
    parser.add_argument("--full-original-dir", required=True)
    parser.add_argument("--full-scan-dir", action="append", required=True)
    parser.add_argument("--hw-union-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
