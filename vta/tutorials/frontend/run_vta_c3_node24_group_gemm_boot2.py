#!/usr/bin/env python3
"""Independent Linux-boot confirmation of the frozen node24 F13-G candidate."""

from __future__ import annotations

import argparse
from pathlib import Path
import traceback

from c3_thread_sync_board import ExperimentBoard
import run_vta_c3_reuse_board as reuse
from run_vta_c3_node24_group_gemm_board import (
    GEMM, TIMER, VERSIONS, run as run_frozen_protocol,
)
from run_vta_c3_node24_partial_board import read, sha, write


BOOT1 = GEMM / "board_run02"


def run(args):
    previous = read(BOOT1 / "summary.json")
    contract = read(BOOT1 / "contract.json")
    if previous["status"] != "SINGLE_BOOT_COMPLETE" or not previous["all_correct"] or \
            previous["timed_calls"] != 54 or \
            contract["status"] != "PRE_LATENCY_FROZEN" or \
            not contract["no_graph_or_second_boot"]:
        raise ValueError("first-boot certificate incomplete")
    # Verify candidate, runtime and timer identities before contacting the board.
    paths = {
        "f13_b": GEMM.parent / "20261009_c3_node24_group_dma_run01/binaries/f13_b/graphlib.so",
        "f13_g": GEMM / "binaries/f13_g/graphlib.so",
        "input_alt": GEMM.parent / "20261009_c3_node24_partial_run01/binaries/input_alt/graphlib.so",
    }
    if set(contract["binaries_sha256"]) != set(VERSIONS) or \
            any(sha(paths[name]) != contract["binaries_sha256"][name]
                for name in VERSIONS) or sha(TIMER) != contract["timer_sha256"]:
        raise ValueError("frozen first-boot program or timer identity drift")
    state = ExperimentBoard(args, "/var/volatile/" + args.output_dir.name + "_bootcheck",
                            args.port).state()
    reuse.validate_state(state, exclusive=True)
    if state["boot_id"] == previous["boot_id"]:
        raise ValueError("second operator run requires a different Linux boot")
    write(args.output_dir / "boot2_predecision.json", {
        "status": "FROZEN_PRIOR_TO_BOOT2_LATENCY",
        "previous_boot": previous["boot_id"], "current_boot": state["boot_id"],
        "same_three_binaries": True, "same_timer": True,
        "orders": contract["orders"], "seed_rotation": contract["seed_rotation"],
        "qualification": "three versions x three seeds; F13-G WA-WB-WA",
        "comparison": "same six-round paired gates as boot 1",
    })
    run_frozen_protocol(args)
    current = read(args.output_dir / "summary.json")
    current_contract = read(args.output_dir / "contract.json")
    if current["boot_id"] != state["boot_id"] or \
            current_contract["binaries_sha256"] != contract["binaries_sha256"] or \
            current_contract["timer_sha256"] != contract["timer_sha256"] or \
            current_contract["runtime_sha256"] != contract["runtime_sha256"] or \
            current_contract["orders"] != contract["orders"]:
        raise ValueError("cross-boot protocol identity drift")
    first = read(BOOT1 / "analysis.json")
    second = read(args.output_dir / "analysis.json")
    write(args.output_dir / "cross_boot_analysis.json", {
        "status": "TWO_BOOT_OPERATOR_CONFIRMED" if
        first["primary_engineering_gate_passed"] and
        second["primary_engineering_gate_passed"] and
        first["single_boot_deployment_candidate"] and
        second["single_boot_deployment_candidate"] else
        "TWO_BOOT_OPERATOR_GATE_NOT_MET",
        "boot_ids": [previous["boot_id"], current["boot_id"]],
        "boot1": first, "boot2": second,
        "graph_eligible": first["primary_engineering_gate_passed"] and
        second["primary_engineering_gate_passed"] and
        first["single_boot_deployment_candidate"] and
        second["single_boot_deployment_candidate"],
        "no_graph_latency_claim": True,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--known-hosts", default=str(Path.home() / ".ssh/known_hosts"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--execute-board", action="store_true")
    args = parser.parse_args()
    if not args.execute_board:
        raise SystemExit("explicit --execute-board required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except BaseException as error:
        write(args.output_dir / "invalid_attempt.json", {
            "status": "INVALID_NON_SPLICEABLE", "error": str(error),
            "traceback": traceback.format_exc(), "timing_valid": False,
        })
        raise


if __name__ == "__main__":
    main()
