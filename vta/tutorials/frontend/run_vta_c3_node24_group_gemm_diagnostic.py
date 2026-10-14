#!/usr/bin/env python3
"""Separate first-call/prewarmed instruction audit for the node24 GEMM pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np
import tvm

from c3_thread_sync_audit import instruction_dump
from run_vta_c3_node24_group_gemm_board import (
    GEMM, OLD, SEEDS, SYMBOL, VERSIONS, GroupGEMMGate,
)
from run_vta_c3_node24_partial_board import same, sentinel, sha, write
from run_vta_p7r120_yolo_rpc_board import load_ephemeral


HELPER = GEMM / "c3_node24_fused_diagnostic_once.so"


def run(args):
    if not HELPER.is_file():
        raise ValueError("diagnostic helper binary missing")
    gate = GroupGEMMGate(args)
    x = {s: np.load(OLD / "board_qualify03" / ("X_%d.npy" % s)) for s in SEEDS}
    y = {s: np.load(OLD / "board_qualify03" / ("Y24_%d.npy" % s)) for s in SEEDS}
    weight = np.load(OLD / "board_qualify03/W_A.npy")
    add = np.load(OLD / "board_qualify03/ADD.npy")
    rows = []
    for condition in ("first_call", "after_three_seed_warmup"):
        for name in VERSIONS:
            with gate.session(name, gate.libraries[name]) as (remote, lib, functions):
                helper_ref = load_ephemeral(remote, HELPER)
                diagnostic = remote.get_function("c3.node24.fused_diagnostic_once")
                dev = remote.ext_dev(0)
                buffers = (tvm.nd.array(x[0], dev), tvm.nd.array(weight, dev),
                           tvm.nd.array(add, dev), tvm.nd.array(sentinel(y[0]), dev))
                xb, wb, ab, yb = buffers
                if condition == "after_three_seed_warmup":
                    for seed in SEEDS:
                        xb.copyfrom(x[seed])
                        wb.copyfrom(weight)
                        ab.copyfrom(add)
                        yb.copyfrom(sentinel(y[seed]))
                        lib[SYMBOL](*buffers)
                        if not same(yb.numpy(), y[seed]):
                            raise ValueError("diagnostic warmup output incorrect")
                xb.copyfrom(x[0])
                wb.copyfrom(weight)
                ab.copyfrom(add)
                yb.copyfrom(sentinel(y[0]))
                functions["vta.runtime.profiler_clear"]()
                dump = str(diagnostic(lib, SYMBOL, *buffers))
                actual = np.array(yb.numpy(), copy=True)
                profile = json.loads(str(functions["vta.runtime.profiler_status"]()))
                parsed = instruction_dump(dump)
                if (not same(actual, y[0]) or
                        not all(same(got, want) for got, want in
                                zip((xb.numpy(), wb.numpy(), ab.numpy()),
                                    (x[0], weight, add))) or
                        parsed["submission_count"] != profile["driver_run_calls"] or
                        profile["driver_timeout_calls"]):
                    raise ValueError("diagnostic correctness/submission mismatch")
                name_out = condition + "_" + name + ".txt"
                (gate.out / name_out).write_text(dump)
                rows.append({"condition": condition, "variant": name,
                             "dump_file": name_out, "dump_sha256": sha(gate.out / name_out),
                             "instruction_audit": parsed, "runtime_profile": profile,
                             "correct": True})
                del helper_ref
    write(gate.out / "analysis.json", {
        "status": "DIAGNOSTIC_COMPLETE", "boot_id": gate.boot,
        "helper_sha256": sha(HELPER), "rows": rows,
        "not_timing_samples": True,
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
