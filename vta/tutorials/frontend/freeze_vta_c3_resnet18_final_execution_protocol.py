#!/usr/bin/env python3
"""Freeze the final R18 pool, full-graph and clean-start execution protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from run_vta_resnet50_relay_residency_dispatch_pair_board import sha256, write_json


SOURCE_NAMES = (
    "prepare_vta_c3_resnet18_literature_replay.py",
    "analyze_vta_c3_resnet18_literature_pool.py",
    "analyze_vta_c3_cheng_four_schemes.py",
    "freeze_vta_c3_resnet18_fullgraph_selections.py",
    "build_vta_c3_resnet18_fullgraph_programs.py",
    "run_vta_c3_resnet18_fullgraph_programs.py",
    "run_vta_from_scratch_autotvm_fullgraph.py",
    "run_vta_c3_resnet18_from_scratch_ours_fullgraph.py",
    "analyze_vta_c3_resnet18_clean_start_ab.py",
    "plot_vta_c3_literature_results.py",
    "run_vta_c3_literature_baselines.py",
    "run_vta_c3_resnet18_board_pool.py",
    "prepare_vta_c3_resnet18_literature_candidates.py",
    "extend_vta_c3_resnet18_literature_candidates.py",
    "run_vta_c3_resnet18_local_qualification.py",
    "freeze_vta_c3_resnet18_board_pool.py",
    "cross_compile_vta_c3_resnet18_board_pool.py",
)


def verify(directory):
    directory = Path(directory).resolve()
    try:
        verify_artifacts_compatible(directory)
    except IsADirectoryError:
        ledger = json.loads((directory / "artifact_hashes.json").read_text())[
            "artifacts"
        ]
        for name, expected in ledger.items():
            if isinstance(expected, str):
                if sha256(directory / name) != expected:
                    raise RuntimeError("artifact hash mismatch: {}".format(directory / name))
            elif isinstance(expected, dict):
                for child, child_hash in expected.items():
                    if sha256(directory / name / child) != child_hash:
                        raise RuntimeError(
                            "artifact hash mismatch: {}".format(directory / name / child)
                        )
            else:
                raise ValueError("unsupported artifact ledger entry: {}".format(name))
    return sha256(directory / "artifact_hashes.json")


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    old = Path(args.prior_postboard_contract).resolve()
    source = Path(args.fullgraph_source_contract).resolve()
    merged = Path(args.merged_board_pool).resolve()
    cross = Path(args.cross_compile).resolve()
    input_hashes = {
        "superseded_postboard_contract": verify(old),
        "fullgraph_source_contract": verify(source),
        "merged_board_pool": verify(merged),
        "cross_compile": verify(cross),
    }
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    source_paths = [here / name for name in SOURCE_NAMES]
    source_paths.extend(
        (
            repo / "vta/python/vta/top/residency_dispatch.py",
            repo / "vta/python/vta/top/vta_conv2d.py",
            repo / "vta/python/vta/top/vta_conv2d_residency.py",
            repo / "python/tvm/relay/testing/resnet.py",
        )
    )
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    contract = {
        "schema": "c3_resnet18_final_execution_protocol_v1",
        "status": "frozen_before_any_merged_pool_fpga_or_resnet18_fullgraph_label",
        "supersedes": {
            "manifest_sha256": input_hashes["superseded_postboard_contract"],
            "reason": "analysis rows now retain terminal candidate identity needed for immutable full-graph deployment",
        },
        "input_hashes": input_hashes,
        "source_hashes": {str(path): sha256(path) for path in source_paths},
        "operator_pool": {
            "candidate_count": 214,
            "workload_counts": {"R18-H1": 104, "R18-H2": 44, "R18-H3": 66},
            "correctness": "three seeds, first wrong answer stops; interruption invalidates the session",
            "timing": "five balanced rounds for every FPGA-correct identity",
            "oracle": "connect only after the complete non-spliced pool",
        },
        "pool_replay": {
            "policies": [
                "random",
                "stock_mode_aware_xgb",
                "rieber_hw_init_xgb",
                "ml2tuner_pva",
                "cheng_minimum_access",
                "ours_dma_multifidelity",
            ],
            "seeds": list(range(57001, 57021)),
            "budgets": [10, 20, 50, 75, 96],
            "model_A_plus_DMA_is_ablation": True,
            "actual_common_pool_cost_and_counterfactual_lazy_cost_are_separate": True,
        },
        "selected_fullgraph": {
            "selection_seed": 57001,
            "selection_budget": 50,
            "programs": [
                "stock_tophub_reference",
                "pool_stock_mode_aware_xgb",
                "cheng_minimum_access",
                "ml2tuner_pva",
                "ours_dma_multifidelity",
            ],
            "same_route_signature_built_once": True,
            "correctness": "three deterministic inputs, every output, exact equality to stock",
            "timing": "seven balanced rounds in one non-spliced clean-start session",
        },
        "clean_start_H1": {
            "execution_order": [
                "XGB-seed-67001",
                "OURS-run-1",
                "OURS-run-2",
                "XGB-seed-67002",
                "XGB-seed-67003",
                "OURS-run-3",
            ],
            "xgb": {
                "history": "empty",
                "target_tophub_used_by_search": False,
                "trial_cap": 300,
                "wall_budget_seconds": 600,
                "seeds": [67001, 67002, 67003],
            },
            "ours": {
                "candidate_generation": "H1-only deterministic max-min batches until at least 12 three-seed FSim-pass identities",
                "candidate_budget": 13,
                "performance_labels_reused": False,
                "runs": 3,
            },
            "boundaries": {
                "W0": "host process starts before preflight/bitstream/RPC",
                "T0": "clean bitstream and fresh RPC ready before tuning work",
                "T1": "selected full graph passes three inputs and seven paired rounds",
            },
            "both_W0_to_T1_and_T0_to_T1_reported": True,
        },
        "failure_accounting": {
            "compile_failure": "gross plus real wall",
            "wrong_answer": "invalid; never enters performance regression",
            "RPC_or_power_interruption": "invalidate entire session; no clock splice",
            "automatic_recovery": "included in W0/T0 wall and event ledger",
        },
        "claim_boundaries": {
            "logical_VTA_DMA_is_physical_AXI": False,
            "random_input_equivalence_is_ImageNet_accuracy": False,
            "method_consistent_is_source_exact_reproduction": False,
            "pool_XGB_is_official_empty_history_AutoTVM": False,
        },
        "target_performance_labels_read": False,
        "board_contacted": False,
    }
    output.mkdir(parents=True)
    write_json(output / "contract.json", contract)
    write_json(
        output / "summary.json",
        {
            "status": contract["status"],
            "bound_source_count": len(source_paths),
            "operator_candidate_count": 214,
            "clean_start_sessions_planned": 6,
            "board_contacted": False,
            "target_performance_labels_read": False,
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
                path.name: sha256(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "source_sha256": sha256(Path(__file__).resolve()),
        },
    )
    print(json.dumps(read_json(output / "summary.json"), indent=2, sort_keys=True))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-postboard-contract", required=True)
    parser.add_argument("--fullgraph-source-contract", required=True)
    parser.add_argument("--merged-board-pool", required=True)
    parser.add_argument("--cross-compile", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
