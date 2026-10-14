#!/usr/bin/env python3
"""Explicit stages, immutable identities, no transparent board retry."""
import argparse
from pathlib import Path
import traceback
from c3_residency_tradeoff import write_new
from c3_banerjee_compare_board import Runner, freeze, analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "preflight", "qualify", "formal", "analyze"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--known-hosts", default=str(Path.home() / ".ssh/known_hosts"))
    parser.add_argument("--execute-board", action="store_true")
    args = parser.parse_args()
    if args.phase in ("freeze", "analyze"):
        (freeze if args.phase == "freeze" else analyze)(args.output_dir)
        return
    if args.phase != "preflight" and not args.execute_board:
        raise SystemExit("explicit --execute-board required")
    runner = None
    try:
        if any(args.output_dir.rglob("invalid_attempt.json")):
            raise ValueError("previous board failure: no retry permitted")
        runner = Runner(args)
        if args.phase == "preflight":
            write_new(args.output_dir / "environment_preflight.json", runner.initial)
            print("READONLY_PREFLIGHT_PASSED; zero target calls")
        else:
            getattr(runner, args.phase)()
    except BaseException as error:
        destination = runner.out if runner is not None and runner.out is not None else args.output_dir
        write_new(destination / "invalid_attempt.json", dict(status="STOP_ALL_SUBSEQUENT_BOARD_WORK",
            phase=args.phase, error=str(error), traceback=traceback.format_exc(),
            completed_calls=len(runner.rows) if runner else 0, no_retry_or_system_restore=True))
        raise


if __name__ == "__main__":
    main()
