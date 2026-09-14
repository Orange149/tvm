#!/usr/bin/env python3
"""Freeze the Node-B literature-aligned C3 experiment protocol without board access."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import run_vta_c3_literature_baselines as policies


SEEDS = tuple(range(57001, 57021))
WORKLOADS = {
    "R18-H1": "64->64, 56x56, 3x3, stride1",
    "R18-H2": "256->256, 14x14, 3x3, stride1",
    "R18-H3": "256->512, 14x14, 1x1, stride2",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    source_paths = (
        Path(__file__).resolve(),
        here / "run_vta_c3_literature_baselines.py",
        here / "prepare_vta_c3_resnet18_literature_candidates.py",
        here / "c3_candidate_identity.py",
        repo / "vta/python/vta/top/vta_conv2d.py",
        repo / "vta/python/vta/top/vta_conv2d_residency.py",
        here
        / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/01_paper_reproduction/PAPER_METHOD_SPEC.md",
    )
    source_hashes = {str(path.relative_to(repo)): sha256_file(path) for path in source_paths}
    protocol = {
        "schema": "c3_literature_aligned_experiment_protocol_v1",
        "status": "frozen_before_resnet18_lowering_fsim_fpga_or_latency",
        "node": "B",
        "policies": list(policies.POLICIES),
        "policy_ablations": {
            "hw_aware_initialization": {
                "levels": list(policies.RIEBER_LEVELS),
                "e0_size": 50,
                "maximum_valid": 25,
                "maximum_invalid": 25,
                "presampling": "min(1000, complete original ConfigSpace cardinality)",
                "seeds": list(SEEDS),
                "performance_tuner_after_initialization": "same mode-aware XGB",
            },
            "ml2tuner": {
                "model_p": "visible workload, tile and residence-mode performance model",
                "model_v": "separate compile/execution validity classifier",
                "model_a": "visible plus lowered compiler-hidden performance model",
                "model_a_dma_ablation": True,
                "N": 10,
                "alpha": 1,
                "compile_candidates_per_round": 20,
                "fpga_candidates_per_round": 10,
                "invalid_candidates_enter_performance_regression": False,
            },
            "cheng": {
                "modes": list(policies.CHENG_MODES),
                "scheme_4_implementation": (
                    "input-prioritized traversal plus full-layer weight SRAM residency and "
                    "explicit end-of-residency synchronization; functional reimplementation, "
                    "not the unpublished non-overwriting-address source patch"
                ),
                "resource_failure": "not_applicable with exact original fallback",
            },
            "ours": {
                "stages": ["static", "lower", "fsim", "compile", "fpga_correctness", "timing"],
                "primary_label_free_priority": "lowered logical total DMA bytes",
                "correctness": "three seeds; first failure stops; all three required to pass",
            },
        },
        "workloads": WORKLOADS,
        "candidate_generation": {
            "complete_original_space": (
                "use the actual per-workload AutoTVM ConfigSpace cardinality; do not truncate to "
                "the plan's assumed 1280 when the real domain differs"
            ),
            "observed_cardinalities_before_candidate_freeze": {
                "R18-H1": 2304,
                "R18-H2": 1600,
                "R18-H3": 480,
            },
            "selected_tiles_per_workload": 24,
            "selection": "deterministic normalized-knob max-min without lowering or latency labels",
            "expanded_modes": list(policies.CHENG_MODES),
            "identities_per_workload": 96,
            "local_legal_minimum": 12,
            "expansion": (
                "append the next 24 max-min tiles only when fewer than 12 identities survive local "
                "qualification; FPGA latency must remain unread"
            ),
        },
        "execution": {
            "order": [
                "analytic static checks",
                "real lowering and hidden-feature extraction",
                "three-seed FSim",
                "ARM cross compile",
                "three-seed fail-fast FPGA correctness",
                "five-round balanced operator timing",
                "connect pool oracle only after every correct candidate is timed",
            ],
            "budgets": [10, 20, 50, 75, 96],
            "search_seeds": list(SEEDS),
            "time_boundaries": {
                "W0": "host process begins before preflight/bitstream/RPC preparation",
                "T0": "same clean bitstream and RPC are ready",
                "T1": "selected configuration finishes whole-graph correctness and timing",
            },
            "failures_count_in_gross_and_wall": True,
            "rpc_or_power_interruption": "invalidate whole session; never splice clocks",
        },
        "required_outputs": [
            "contract.json",
            "timeline.jsonl",
            "results.jsonl",
            "summary.json",
            "artifact_hashes.json",
        ],
        "paper_alignment": {
            "Rieber_2022": "method-consistent Algorithm-1/E0/validity-bias reproduction",
            "ML2Tuner_2024_2025": "method-consistent P/V/A reproduction plus A+DMA ablation",
            "Cheng_2026": "four-scheme functional reimplementation on the current VTA tree",
            "code_exact_claim": False,
        },
        "metric_scope": {
            "logical_vta_dma_is_physical_axi": False,
            "random_input_equivalence_is_accuracy": False,
            "report_counts_and_real_wall_together": True,
        },
        "source_hashes": source_hashes,
        "board_contacted": False,
        "performance_labels_read": False,
    }
    write_json(output / "contract.json", protocol)
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    (output / "results.jsonl").write_text("", encoding="utf-8")
    write_json(
        output / "summary.json",
        {
            "status": "node_b_protocol_frozen",
            "policy_count": len(policies.POLICIES),
            "search_seed_count": len(SEEDS),
            "workload_count": len(WORKLOADS),
            "board_contacted": False,
            "performance_labels_read": False,
        },
    )
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(output / "artifact_hashes.json", {"artifacts": artifacts})
    print(json.dumps(read_json(output / "summary.json"), indent=2, sort_keys=True))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
