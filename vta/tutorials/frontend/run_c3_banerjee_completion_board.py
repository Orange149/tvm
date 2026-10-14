#!/usr/bin/env python3
"""Freeze, qualify and measure the completed software baseline on one boot."""
import argparse
from pathlib import Path
import traceback
from c3_banerjee_baseline import write_new
from c3_banerjee_completion_board import Runner, freeze, analyze


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "preflight", "qualify", "select", "formal", "analyze"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    parser.add_argument("--execute-board", action="store_true")
    args = parser.parse_args()
    if args.phase in ("freeze", "analyze"):
        (freeze if args.phase == "freeze" else analyze)(args.output_dir.resolve())
    else:
        if args.phase != "preflight" and not args.execute_board:
            parser.error("board acquisition requires --execute-board")
        runner = None
        try:
            if any(args.output_dir.rglob("invalid_attempt.json")):
                raise ValueError("failed attempt retained; no automatic restart")
            runner = Runner(args)
            if args.phase == "preflight":
                write_new(args.output_dir / "environment_preflight.json", runner.initial)
                print("READONLY_PREFLIGHT_PASSED")
            else:
                getattr(runner, args.phase)()
        except BaseException as error:
            dest = runner.out if runner is not None and runner.out is not None else args.output_dir
            write_new(dest / "invalid_attempt.json", {"status": "STOP_ALL_BOARD_WORK", "phase": args.phase,
                "error": str(error), "traceback": traceback.format_exc(), "completed_calls": len(runner.rows) if runner else 0})
            raise
