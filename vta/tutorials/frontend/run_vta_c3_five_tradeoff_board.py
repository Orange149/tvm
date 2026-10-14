#!/usr/bin/env python3
"""Explicit frozen stages, no reboot, system restore, retry or winner replacement."""

import argparse
from pathlib import Path
import traceback

from c3_residency_tradeoff import write_new
from c3_five_tradeoff_board import Runner, execution_freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "preflight", "qualify", "selection", "confirmation"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--known-hosts", default=str(Path.home() / ".ssh/known_hosts"))
    parser.add_argument("--execute-board", action="store_true")
    args = parser.parse_args()
    if args.phase == "freeze":
        execution_freeze(args.output_dir)
        return
    if args.phase != "preflight" and not args.execute_board:
        raise SystemExit("explicit --execute-board required")
    runner = None
    try:
        runner = Runner(args)
        if args.phase == "preflight":
            write_new(args.output_dir / "environment_preflight.json", runner.initial)
            print("READONLY_PREFLIGHT_PASSED; no FPGA calls")
        else:
            getattr(runner, args.phase)()
    except BaseException as error:
        destination = runner.out if runner is not None and runner.out is not None else args.output_dir
        target = destination / ("invalid_attempt.json" if runner is not None and runner.out is not None
                                else "preflight_failure.json")
        write_new(target, {"status": "STOP_SUBSEQUENT_BOARD_WORK", "phase": args.phase,
                          "error": str(error), "traceback": traceback.format_exc(),
                          "completed_calls": len(runner.rows) if runner else 0,
                          "no_retry_or_system_restore": True})
        raise


if __name__ == "__main__":
    main()
