#!/usr/bin/env python3
"""Fail-closed FPGA correctness gate for node34's fixed grouped bridge.

No latency samples are produced.  Every RPC session reloads the frozen
bitstream and uses the historical CPU3 runtime; a failed call ends the gate.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import shlex
import traceback

import numpy as np
import tvm
from tvm.contrib import graph_executor

import c3_yolo224_incremental as protocol
import run_vta_c3_reuse_board as reuse
from c3_thread_sync_board import ExperimentBoard
from run_vta_c3_mode11_graph_context_diagnostic import connect, local_params, upload_module
from run_vta_c3_yolo608_graph_boot import random_input, output_summary
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory

HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
PREPARED = GROUPED / "20261008_c3_yolo224_incremental_prepare_run03"
GRAPH = {
    "G2": PREPARED / "G2",
    "O1": GROUPED / "20261008_c3_node34_original_o1_graph_run01",
    "I1": GROUPED / "20261008_c3_node34_input_i1_graph_run01",
    "W1": GROUPED / "20261008_c3_node34_mode4_graph_run01",
    "BRIDGE": GROUPED / "20261008_c3_node34_bridge_graph_run01",
}
RUNTIME = GROUPED / "20260922_c3_thread_sync_freeze_run04/contract.json"
HELPER = GROUPED / "20260926_c3_evaluation_v5_helper_build_run02/contract.json"
POLICY = GROUPED / "20260926_c3_yolo512_evaluation_v5_completion_audit_run01/completion_policy.json"
SYMBOL = "tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast_5"
SEEDS = (0, 20250901, 20260910)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def equal(left, right):
    return len(left) == len(right) and all(
        x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y)
        for x, y in zip(left, right)
    )


def sentinel(reference):
    return np.where(reference == -113, 112, -113).astype(reference.dtype)


class Qualification:
    def __init__(self, args):
        self.args = args
        self.out = args.output_dir
        self.rows = []
        self.sessions = []
        runtime = read(RUNTIME)
        helper = read(HELPER)
        policy = read(POLICY)
        if policy["status"] != "COMPLETION_POLICY_FROZEN" or policy["mode"] != "run_only":
            raise ValueError("run-only completion audit not frozen")
        if sha(helper["binary"]) != helper["binary_sha256"] or \
                policy["helper_sha256"] != helper["binary_sha256"]:
            raise ValueError("single-call helper identity changed")
        for name, path in runtime["runtime_paths"].items():
            if sha(path) != runtime["runtime_sha256"][name]:
                raise ValueError("runtime changed: " + name)
        reuse.RUNTIME = runtime["runtime_sha256"]
        self.runtime, self.helper, self.policy = runtime, helper, policy
        frozen = read(GRAPH["G2"] / "build.json")
        frozen_graph = frozen["graph_sha256"]
        frozen_params = sha(GRAPH["G2"] / "params.bin")
        for name, root in GRAPH.items():
            build = read(root / "build.json")
            graph_sha = build["graph_sha256"]
            binary_sha = build.get("graphlib_sha256", build.get("bridge_graphlib_sha256"))
            if graph_sha != frozen_graph or sha(root / "graph.json") != frozen_graph:
                raise ValueError("graph structure changed: " + name)
            if sha(root / "graphlib.so") != binary_sha:
                raise ValueError("graph binary changed: " + name)
            if name != "G2" and sha(root / "params.bin") != frozen_params:
                # TVM serialization can be byte-different while tensors are equal.
                left, right = local_params(root), local_params(GRAPH["G2"])
                if set(left) != set(right) or not all(np.array_equal(left[k], right[k]) for k in left):
                    raise ValueError("parameters changed: " + name)
        board = ExperimentBoard(args, "/var/volatile/" + self.out.name + "_preflight", args.port)
        self.initial = board.state()
        reuse.validate_state(self.initial, exclusive=True)
        self.environment = board.environment()
        self.boot = self.initial["boot_id"]
        write(self.out / "preflight.json", {"board": self.initial, "environment": self.environment})

    def record(self, **row):
        row.update(boot_id=self.boot, index=len(self.rows))
        self.rows.append(row)
        with (self.out / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()

    @contextmanager
    def session(self, name):
        board = ExperimentBoard(self.args, "/var/volatile/" + self.out.name + "_" + str(len(self.sessions)),
                                self.args.port + 2 * len(self.sessions))
        state = board.state()
        reuse.validate_state(state, exclusive=True)
        if state["boot_id"] != self.boot:
            raise ValueError("Linux boot changed")
        attestation = board.deploy(self.runtime["runtime_paths"])
        board.reload()
        remote, refs = None, []
        try:
            board.start()
            remote = connect(self.args.host, board.port, self.args.timeout)
            inventory, functions = runtime_inventory(remote)
            environment = board.environment(require_affinity=True)
            if environment["boot_id"] != self.boot or \
                    environment["governor"] != self.environment["governor"]:
                raise ValueError("board environment changed")
            refs.append(load_ephemeral(remote, Path(self.helper["binary"])))
            symbols = json.loads(str(remote.get_function("c3.v5.runtime_symbols")()))
            for symbol, observed in symbols.items():
                expected = self.policy["runtime_symbols"][symbol]
                if Path(observed["dso"]).name != Path(expected["dso"]).name:
                    raise ValueError("helper resolved a different VTA runtime")
            with tempfile.TemporaryDirectory(prefix="c3_n34_qualify_") as temporary:
                lib = upload_module(remote, GRAPH[name] / "graphlib.so", temporary, name)
                refs.append(lib)
                remote.get_function("c3.v5.reset_counts")()
                yield remote, lib, functions
                counts = json.loads(str(remote.get_function("c3.v5.counts")()))
                row = {"name": name, "runtime": attestation, "symbols": symbols,
                       "environment": environment, "inventory": inventory,
                       "helper_counts": counts}
                self.sessions.append(row)
                write(self.out / ("session_%03d.json" % (len(self.sessions) - 1)), row)
        finally:
            refs.clear()
            remote = None
            gc.collect()
            board.stop()
            # Each session copied about 85 MiB of runtime to reboot-volatile tmpfs.
            # Keep all local session records; remove only this exact inactive RPC copy.
            expected_prefix = "/var/volatile/" + self.out.name + "_"
            if not board.runtime.startswith(expected_prefix):
                raise ValueError("refuse to clean an unverified runtime directory")
            destination = shlex.quote(board.runtime)
            board.ssh("test -z \"$(pidof tvm_rpc 2>/dev/null || true)\" && "
                      "test -d " + destination + " && test ! -L " + destination
                      + " && rm -r " + destination)

    def graph_call(self, remote, lib, functions, name, seed, end=None, expected=None):
        graph = read(GRAPH[name] / "graph.json")
        if end is not None:
            graph = protocol.prefix_graph(graph, end, [[end, 0, 0]])
        executor = graph_executor.create(json.dumps(graph), lib, [remote.ext_dev(0), remote.cpu(0)])
        executor.load_params((GRAPH[name] / "params.bin").read_bytes())
        executor.set_input("data", random_input(seed, resolution=224))
        functions["vta.runtime.profiler_clear"]()
        executor.run()
        outputs = [np.array(executor.get_output(i).numpy(), copy=True)
                   for i in range(executor.get_num_outputs())]
        profile = json.loads(str(functions["vta.runtime.profiler_status"]()))
        correct = expected is None or equal(outputs, expected)
        self.record(phase="prefix" if end is not None else "graph", variant=name,
                    seed=seed, end_node=end, correct=correct,
                    outputs=output_summary(outputs), runtime_profile=profile,
                    latency_ms=None)
        if not correct or profile.get("driver_timeout_calls", 0) != 0 \
                or profile.get("driver_run_calls", 0) <= 0:
            raise ValueError("graph/prefix output or completion failed: " + name)
        return outputs

    def fused_call(self, lib, functions, name, xbuf, wbuf, abuf, ybuf, expected,
                   x, w, add, label):
        xbuf.copyfrom(x)
        wbuf.copyfrom(w)
        abuf.copyfrom(add)
        ybuf.copyfrom(sentinel(expected))
        functions["vta.runtime.profiler_clear"]()
        lib[SYMBOL](xbuf, wbuf, abuf, ybuf)
        output = np.array(ybuf.numpy(), copy=True)
        profile = json.loads(str(functions["vta.runtime.profiler_status"]()))
        inputs_unchanged = equal([xbuf.numpy(), wbuf.numpy(), abuf.numpy()], [x, w, add])
        correct = equal([output], [expected]) and inputs_unchanged
        self.record(phase="fused_replay", variant=name, label=label, correct=correct,
                    inputs_unchanged=inputs_unchanged, outputs=output_summary([output]),
                    runtime_profile=profile, latency_ms=None)
        if not correct or profile.get("driver_timeout_calls", 0) != 0 \
                or profile.get("driver_run_calls", 0) <= 0:
            np.save(self.out / ("failure_" + name + "_" + label + ".npy"), output)
            raise ValueError("fused replay or completion failed: " + name + ":" + label)
        return output

    def run(self):
        params = local_params(GRAPH["G2"])
        graph = read(GRAPH["G2"] / "graph.json")
        wname = graph["nodes"][32]["name"]
        aname = graph["nodes"][33]["name"]
        weight, add = params[wname], params[aname]
        inputs, outputs = {}, {}
        with self.session("G2") as (remote, lib, functions):
            for seed in SEEDS:
                inputs[seed] = self.graph_call(remote, lib, functions, "G2", seed, end=31)[0]
                outputs[seed] = self.graph_call(remote, lib, functions, "G2", seed, end=34)[0]
                np.save(self.out / ("X_%d.npy" % seed), inputs[seed])
                np.save(self.out / ("Y34_%d.npy" % seed), outputs[seed])
        wb = np.roll(weight, 1, axis=0).copy()
        reference_b = None
        with self.session("G2") as (remote, lib, functions):
            dev = remote.ext_dev(0)
            xbuf = tvm.nd.array(inputs[0], dev)
            wbuf = tvm.nd.array(weight, dev)
            abuf = tvm.nd.array(add, dev)
            ybuf = tvm.nd.array(sentinel(outputs[0]), dev)
            for seed in SEEDS:
                self.fused_call(lib, functions, "G2", xbuf, wbuf, abuf, ybuf, outputs[seed],
                                inputs[seed], weight, add, "A_seed_" + str(seed))
            ybuf.copyfrom(sentinel(outputs[0]))
            xbuf.copyfrom(inputs[0])
            wbuf.copyfrom(wb)
            abuf.copyfrom(add)
            functions["vta.runtime.profiler_clear"]()
            lib[SYMBOL](xbuf, wbuf, abuf, ybuf)
            reference_b = np.array(ybuf.numpy(), copy=True)
            unchanged = equal([xbuf.numpy(), wbuf.numpy(), abuf.numpy()], [inputs[0], wb, add])
            profile = json.loads(str(functions["vta.runtime.profiler_status"]()))
            distinguished = not np.array_equal(reference_b, outputs[0])
            self.record(phase="fused_reference", variant="G2", label="B_weight", correct=unchanged,
                        outputs=output_summary([reference_b]), reference_distinct=distinguished,
                        runtime_profile=profile, latency_ms=None)
            if not unchanged or not distinguished or profile.get("driver_timeout_calls", 0) != 0:
                raise ValueError("W_B reference invalid or not distinguishing")
        np.save(self.out / "W_A.npy", weight)
        np.save(self.out / "W_B.npy", wb)
        np.save(self.out / "ADD.npy", add)
        np.save(self.out / "Y_B.npy", reference_b)
        for name in ("O1", "I1", "W1", "BRIDGE"):
            with self.session(name) as (remote, lib, functions):
                dev = remote.ext_dev(0)
                xbuf = tvm.nd.array(inputs[0], dev)
                wbuf = tvm.nd.array(weight, dev)
                abuf = tvm.nd.array(add, dev)
                ybuf = tvm.nd.array(sentinel(outputs[0]), dev)
                for seed in SEEDS:
                    self.fused_call(lib, functions, name, xbuf, wbuf, abuf, ybuf, outputs[seed],
                                    inputs[seed], weight, add, "A_seed_" + str(seed))
                for label, w, expected in (("WA_repeat", weight, outputs[0]),
                                           ("WB", wb, reference_b),
                                           ("WA_return", weight, outputs[0])):
                    self.fused_call(lib, functions, name, xbuf, wbuf, abuf, ybuf, expected,
                                    inputs[0], w, add, label)
        # Graph calls deliberately follow unlike predecessor programs without
        # an experiment-added drain.  The normal target functions must compose.
        with self.session("G2") as (remote, lib, functions):
            refs = {seed: self.graph_call(remote, lib, functions, "G2", seed)
                    for seed in SEEDS}
        for name in ("O1", "I1", "W1", "BRIDGE"):
            with self.session(name) as (remote, lib, functions):
                for seed in (*SEEDS, 0):
                    self.graph_call(remote, lib, functions, name, seed, expected=refs[seed])
        after = ExperimentBoard(self.args, "/var/volatile/" + self.out.name + "_postflight", self.args.port)
        state = after.state()
        reuse.validate_state(state, exclusive=True)
        if state["boot_id"] != self.boot:
            raise ValueError("boot changed during qualification")
        write(self.out / "postflight.json", state)
        write(self.out / "summary.json", {
            "status": "BOARD_FUSION_AND_GRAPH_QUALIFIED_NO_TIMING", "boot_id": self.boot,
            "all_correct": True, "calls": len(self.rows), "sessions": len(self.sessions),
            "board_latency_collected": False,
            "graphlib_sha256": {key: sha(root / "graphlib.so") for key, root in GRAPH.items()},
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
        Qualification(args).run()
    except BaseException as error:
        write(args.output_dir / "invalid_attempt.json", {
            "status": "INVALID_NON_SPLICEABLE", "error": str(error),
            "traceback": traceback.format_exc(), "timing_valid": False,
        })
        raise


if __name__ == "__main__":
    main()
