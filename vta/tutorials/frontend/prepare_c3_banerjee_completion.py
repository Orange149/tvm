#!/usr/bin/env python3
"""Qualify all analytical ci ties before freezing any FPGA timing pool."""
import argparse
import json
from pathlib import Path
import sys

from c3_banerjee_baseline import write_new, HERE, GROUPED, digest
from run_c3_banerjee_completion import isolated

TPS = ("TPS-single", "TPS-single-ci2", "TPS-single-ci4", "TPS-single-ci8",
       "TPS-oc-reuse", "TPS-oc-reuse-ci2", "TPS-oc-reuse-ci4",
       "TPS-h-reuse", "TPS-h-reuse-ci2", "TPS-h-reuse-ci4")


def local(root, family):
    from audit_c3_banerjee_completion import audit
    records = {}
    for program in TPS:
        directory = root / family / program
        isolated(root, directory, program, "sim")
        audit(root, program, directory)
        records[program] = str(directory)
        print("Qualified complete fusion and actual dependencies: " + program, flush=True)
    write_new(root / "local_index.json", records)


def board_compile(root):
    index = json.loads((root / "local_index.json").read_text())
    for program in TPS:
        if digest(Path(index[program]) / "native_certificate.json") == "":
            raise ValueError("missing native certificate")
        isolated(root, root / "binaries" / program, program, "board")
        print("ARM target matches qualified final TIR: " + program, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("local", "cross"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--local-attempt", default="local_final04")
    args = parser.parse_args()
    if args.phase == "local":
        if "/" in args.local_attempt or args.local_attempt.startswith("."):
            parser.error("local-attempt must be a new simple directory name")
        local(args.root.resolve(), args.local_attempt)
    else:
        board_compile(args.root.resolve())
