#!/usr/bin/env python3
"""C3 v1.1 experiment orchestration. Never invoke the board before local gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import c3_residency_tradeoff as protocol


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DOCUMENT = GROUPED.parent / "00_governance/20261012_C3_RESIDENCY_INSTRUCTION_TRADEOFF_V1_1.md"


def prepare(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    protocol.write_new(output / "protocol.json", {
        "schema": protocol.SCHEMA, "status": "LOCAL_PREPARATION_STARTED",
        "document": str(DOCUMENT), "document_sha256": protocol.digest(DOCUMENT),
        "programs": protocol.PROGRAMS, "orders": protocol.ORDERS,
        "seeds": protocol.SEEDS, "balance": protocol.balance(),
        "no_linux_reboot": True, "no_graph_or_second_boot": True,
        "calibration": protocol.CALIBRATION, "validation": protocol.VALIDATION,
        "formal_calls": 108, "warmup_calls": 108,
        "qualification_calls": 18, "same_handle_weight_aba_calls": 18,
        "stage_timeout_seconds": 1200,
        "source_sha256": {str(HERE / n): protocol.digest(HERE / n) for n in (
            "run_vta_c3_residency_tradeoff.py", "c3_residency_tradeoff.py",
            "c3_residency_tradeoff_local.py")},
    })
    attempts = []
    # R13 is local-only compatibility, not a seventh board candidate.
    for r in (13, 0, 8, 12):
        local = output / "local" / ("compat_R13-G" if r == 13 else "R%d-G" % r)
        command = [sys.executable, str(Path(__file__).resolve()), "worker",
                   "--output-dir", str(local), "--resident-groups", str(r)]
        started = time.monotonic()
        with (output / ("local_R%d.log" % r)).open("x") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=1200, check=False)
        attempts.append({"resident_groups": r, "command": command,
                         "seconds": time.monotonic() - started, "returncode": process.returncode})
        if process.returncode or not (local / "result.json").is_file() or \
                protocol.read(local / "result.json").get("status") != "LOCAL_FSIM_PASSED_BOARD_NOT_QUALIFIED":
            protocol.write_new(output / "local_attempts.json", attempts)
            raise ValueError("local hard gate failed for R%d; no board or replacement candidate" % r)
        print("Local complete: R%d-G (no FPGA contact)" % r, flush=True)
    protocol.write_new(output / "local_attempts.json", attempts)
    # The rest of prepare is deliberately gated on the exact local evidence.
    from c3_residency_tradeoff_evidence import finish_prepare
    finish_prepare(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "qualify", "board", "analyze", "worker"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resident-groups", type=int, choices=(0, 8, 12, 13))
    parser.add_argument("--board-compile", action="store_true")
    parser.add_argument("--execute-board", action="store_true")
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    try:
        if args.phase == "worker":
            if args.resident_groups is None:
                parser.error("worker requires frozen --resident-groups")
            from c3_residency_tradeoff_local import run_worker
            run_worker(args.output_dir, args.resident_groups, args.board_compile)
        elif args.phase == "prepare":
            prepare(args)
        elif args.phase in ("qualify", "board"):
            if not args.execute_board:
                parser.error("explicit --execute-board required")
            from c3_residency_tradeoff_board import run
            run(args)
        else:
            from c3_residency_tradeoff_analysis import analyze
            analyze(args.output_dir)
    except Exception as error:
        if args.output_dir.is_dir() and not (args.output_dir / "invalid_attempt.json").exists():
            protocol.write_new(args.output_dir / "invalid_attempt.json", {
                "status": "INVALID_NON_SPLICEABLE", "phase": args.phase,
                "error": str(error), "traceback": traceback.format_exc(),
                "board_qualified": False,
            })
        raise


if __name__ == "__main__":
    main()
