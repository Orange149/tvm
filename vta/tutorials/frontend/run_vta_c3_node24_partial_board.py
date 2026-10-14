#!/usr/bin/env python3
"""First-board correctness-only gate for node24 A12/F12/A13/F13.

Uses the frozen YOLO-224 G2 prefix for real fused inputs and outputs.  A
candidate is never timed here.  Every RPC and bitstream reload belongs to
this experiment; foreign services are never stopped.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import shlex
import tempfile
import traceback

import numpy as np
import tvm
from tvm.contrib import graph_executor

import c3_yolo224_incremental as protocol
import run_vta_c3_reuse_board as reuse
from c3_thread_sync_board import ExperimentBoard
from run_vta_c3_mode11_graph_context_diagnostic import connect, local_params, upload_module
from run_vta_c3_yolo608_graph_boot import random_input
from run_vta_p7r120_yolo_rpc_board import runtime_inventory


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
G2 = GROUPED / "20261008_c3_yolo224_incremental_prepare_run03/G2"
RUNTIME = GROUPED / "20260922_c3_thread_sync_freeze_run04/contract.json"
SYMBOL = "tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast"
GRAPH_SYMBOL = SYMBOL + "_3"
SEEDS = (0, 20250901, 20260910)
VARIANTS = ("input_fixed", "input_alt", "a12", "f12", "a13", "f13")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def same(actual, expected):
    return actual.shape == expected.shape and actual.dtype == expected.dtype \
        and np.array_equal(actual, expected)


def sentinel(reference):
    return np.where(reference == -113, 112, -113).astype(reference.dtype)


class Gate:
    def __init__(self, args):
        self.args = args
        self.out = args.output_dir
        self.rows = []
        self.sessions = []
        self.variants = tuple(args.variants)
        self.runtime = read(RUNTIME)
        for name, path in self.runtime["runtime_paths"].items():
            if sha(path) != self.runtime["runtime_sha256"][name]:
                raise ValueError("frozen runtime hash changed: " + name)
        reuse.RUNTIME = self.runtime["runtime_sha256"]
        build = read(G2 / "build.json")
        if sha(G2 / "graph.json") != build["graph_sha256"] or \
                sha(G2 / "graphlib.so") != build["graphlib_sha256"]:
            raise ValueError("frozen G2 graph or binary changed")
        self.libraries = {name: args.binaries / name / "graphlib.so" for name in self.variants}
        self.local = {}
        for name, binary in self.libraries.items():
            row = read(binary.parent / "result.json")
            local_dir = {
                "input_fixed": "input_fixed_audit01",
                "input_alt": "input_alt03",
            }.get(name, name)
            local = read(args.binaries.parent / "local" / local_dir / "result.json")
            count = None if name.startswith("input_") else int(name[1:])
            if row.get("resident_groups") != count or \
                    row.get("keep_resident") != name.startswith("f") or \
                    row.get("address_status") != "passed" or \
                    row.get("target") != "axu5evb" or \
                    row.get("graphlib_sha256") != sha(binary):
                raise ValueError("node24 candidate identity/address gate failed: " + name)
            expected_mode = "input_stationary" if name.startswith("input_") else \
                "node24_partial_input_experimental"
            if row["route"]["identity"]["public_mode"] != expected_mode:
                raise ValueError("node24 public mode changed: " + name)
            if local.get("status") != "LOCAL_FSIM_PASSED_BOARD_NOT_QUALIFIED" or \
                    local.get("address_status") != "passed" or \
                    local.get("request_totals") != row.get("request_totals") or \
                    not all(item["correct"] and item["arguments_unchanged"]
                            for item in local.get("fsim", [])) or \
                    len(local.get("fsim", [])) != 4:
                raise ValueError("node24 local FSim/request certificate missing: " + name)
            def final_digest(record):
                return next(item["sha256"] for item in record["snapshots"]
                            if item["pass"].endswith("CPUAccessRewrite"))
            if final_digest(local) != final_digest(row):
                raise ValueError("node24 sim/board final TIR differs: " + name)
            self.local[name] = row
        board = ExperimentBoard(args, "/var/volatile/" + self.out.name + "_preflight", args.port)
        self.initial = board.state()
        reuse.validate_state(self.initial, exclusive=True)
        self.boot = self.initial["boot_id"]
        write(self.out / "preflight.json", {
            "boot_id": self.boot,
            "bitstream_sha256": self.initial["bitstream_sha256"],
            "udmabuf_size": self.initial["udmabuf_size"],
            "g2_graphlib_sha256": sha(G2 / "graphlib.so"),
            "candidate_graphlib_sha256": {k: sha(v) for k, v in self.libraries.items()},
        })

    def record(self, **row):
        row.setdefault("latency_ms", None)
        row.update(index=len(self.rows), boot_id=self.boot)
        self.rows.append(row)
        with (self.out / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()

    @contextmanager
    def session(self, name, binary):
        sequence = len(self.sessions)
        board = ExperimentBoard(
            self.args, "/var/volatile/" + self.out.name + "_" + str(sequence),
            self.args.port + 2 * sequence,
        )
        state = board.state()
        reuse.validate_state(state, exclusive=True)
        if state["boot_id"] != self.boot:
            raise ValueError("Linux boot changed during node24 qualification")
        remote, refs = None, []
        try:
            attestation = board.deploy(self.runtime["runtime_paths"])
            board.reload()
            board.start()
            remote = connect(self.args.host, board.port, self.args.timeout)
            inventory, functions = runtime_inventory(remote)
            environment = board.environment(require_affinity=True)
            if environment["boot_id"] != self.boot:
                raise ValueError("RPC boot identity changed")
            with tempfile.TemporaryDirectory(prefix="c3_n24_board_") as temporary:
                lib = upload_module(remote, binary, temporary, name)
                refs.append(lib)
                yield remote, lib, functions
                self.sessions.append({
                    "variant": name,
                    "runtime": attestation,
                    "environment": environment,
                    "inventory": inventory,
                    "binary_sha256": sha(binary),
                })
        finally:
            refs.clear()
            remote = None
            gc.collect()
            board.stop()
            expected_prefix = "/var/volatile/" + self.out.name + "_"
            if not board.runtime.startswith(expected_prefix):
                raise ValueError("unexpected node24 volatile RPC directory")
            target = shlex.quote(board.runtime)
            board.ssh("test -z \"$(pidof tvm_rpc 2>/dev/null || true)\" && "
                      "test -d " + target + " && test ! -L " + target +
                      " && rm -r " + target)

    def prefix_all(self, end):
        with self.session("G2_prefix_" + str(end), G2 / "graphlib.so") as (remote, lib, functions):
            graph = protocol.prefix_graph(read(G2 / "graph.json"), end, [[end, 0, 0]])
            executor = graph_executor.create(json.dumps(graph), lib, [remote.ext_dev(0), remote.cpu(0)])
            executor.load_params((G2 / "params.bin").read_bytes())
            result = {}
            for seed in SEEDS:
                executor.set_input("data", random_input(seed, resolution=224))
                functions["vta.runtime.profiler_clear"]()
                executor.run()
                output = np.array(executor.get_output(0).numpy(), copy=True)
                profile = json.loads(str(functions["vta.runtime.profiler_status"]()))
                self.record(phase="reference_prefix", end_node=end, seed=seed,
                            output_sha256=hashlib.sha256(output.tobytes()).hexdigest(),
                            runtime_profile=profile, correct=True)
                if profile.get("driver_timeout_calls", 0) or profile.get("driver_run_calls", 0) <= 0:
                    raise ValueError("G2 prefix did not complete")
                result[seed] = output
            return result

    def fused(self, lib, functions, name, buffers, x, w, add, expected, label):
        xbuf, wbuf, abuf, ybuf = buffers
        xbuf.copyfrom(x)
        wbuf.copyfrom(w)
        abuf.copyfrom(add)
        ybuf.copyfrom(sentinel(expected))
        functions["vta.runtime.profiler_clear"]()
        lib[GRAPH_SYMBOL if name == "G2" else SYMBOL](xbuf, wbuf, abuf, ybuf)
        actual = np.array(ybuf.numpy(), copy=True)
        profile = json.loads(str(functions["vta.runtime.profiler_status"]()))
        unchanged = all(same(observed, reference) for observed, reference in
                        zip((xbuf.numpy(), wbuf.numpy(), abuf.numpy()), (x, w, add)))
        correct = same(actual, expected) and unchanged
        self.record(phase="fused_qualification", variant=name, label=label,
                    correct=correct, inputs_unchanged=unchanged,
                    output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                    runtime_profile=profile)
        if not correct or profile.get("driver_timeout_calls", 0) or \
                profile.get("driver_run_calls", 0) <= 0:
            np.save(self.out / ("failure_" + name + "_" + label + ".npy"), actual)
            raise ValueError("node24 fused FPGA correctness/completion failed: " + name + ":" + label)
        return actual

    def run(self):
        # Independent sessions prevent two whole graph executors coexisting in
        # the 192 MiB FPGA-visible DMA pool.
        inputs = self.prefix_all(21)
        outputs = self.prefix_all(24)
        params = local_params(G2)
        graph = read(G2 / "graph.json")
        weight = np.array(params[graph["nodes"][22]["name"]], copy=True)
        add = np.array(params[graph["nodes"][23]["name"]], copy=True)
        changed_weight = np.roll(weight, 1, axis=0).copy()
        for seed in SEEDS:
            np.save(self.out / ("X_%d.npy" % seed), inputs[seed])
            np.save(self.out / ("Y24_%d.npy" % seed), outputs[seed])
        np.save(self.out / "W_A.npy", weight)
        np.save(self.out / "W_B.npy", changed_weight)
        np.save(self.out / "ADD.npy", add)
        with self.session("G2", G2 / "graphlib.so") as (remote, lib, functions):
            dev = remote.ext_dev(0)
            buffers = (
                tvm.nd.array(inputs[0], dev), tvm.nd.array(weight, dev),
                tvm.nd.array(add, dev), tvm.nd.array(sentinel(outputs[0]), dev),
            )
            for seed in SEEDS:
                self.fused(lib, functions, "G2", buffers, inputs[seed], weight, add,
                           outputs[seed], "A_seed_" + str(seed))
            xbuf, wbuf, abuf, ybuf = buffers
            xbuf.copyfrom(inputs[0])
            wbuf.copyfrom(changed_weight)
            abuf.copyfrom(add)
            ybuf.copyfrom(sentinel(outputs[0]))
            functions["vta.runtime.profiler_clear"]()
            lib[GRAPH_SYMBOL](xbuf, wbuf, abuf, ybuf)
            b_ref = np.array(ybuf.numpy(), copy=True)
            distinct = not same(b_ref, outputs[0])
            profile = json.loads(str(functions["vta.runtime.profiler_status"]()))
            self.record(phase="fused_B_reference", variant="G2", correct=distinct,
                        output_sha256=hashlib.sha256(b_ref.tobytes()).hexdigest(),
                        runtime_profile=profile)
            if not distinct or profile.get("driver_timeout_calls", 0):
                raise ValueError("changed node24 weights failed to distinguish output")
        np.save(self.out / "Y_B.npy", b_ref)
        for name in self.variants:
            with self.session(name, self.libraries[name]) as (remote, lib, functions):
                dev = remote.ext_dev(0)
                buffers = (
                    tvm.nd.array(inputs[0], dev), tvm.nd.array(weight, dev),
                    tvm.nd.array(add, dev), tvm.nd.array(sentinel(outputs[0]), dev),
                )
                for seed in SEEDS:
                    self.fused(lib, functions, name, buffers, inputs[seed], weight, add,
                               outputs[seed], "A_seed_" + str(seed))
                for label, candidate_weight, expected in (
                    ("WA_repeat", weight, outputs[0]),
                    ("WB", changed_weight, b_ref),
                    ("WA_return", weight, outputs[0]),
                ):
                    self.fused(lib, functions, name, buffers, inputs[0], candidate_weight,
                               add, expected, label)
        last = ExperimentBoard(self.args, "/var/volatile/" + self.out.name + "_postflight", self.args.port)
        state = last.state()
        reuse.validate_state(state, exclusive=True)
        if state["boot_id"] != self.boot:
            raise ValueError("node24 qualification boot changed")
        write(self.out / "sessions.json", self.sessions)
        write(self.out / "summary.json", {
            "status": "BOARD_CORRECT_NO_TIMING", "boot_id": self.boot,
            "all_correct": True, "calls": len(self.rows), "sessions": len(self.sessions),
            "candidate_graphlib_sha256": {name: sha(path) for name, path in self.libraries.items()},
            "board_latency_collected": False,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=("a12", "f12", "a13", "f13"))
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
        Gate(args).run()
    except BaseException as error:
        write(args.output_dir / "invalid_attempt.json", {
            "status": "INVALID_NON_SPLICEABLE", "error": str(error),
            "traceback": traceback.format_exc(), "timing_valid": False,
        })
        raise


if __name__ == "__main__":
    main()
