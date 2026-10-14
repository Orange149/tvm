#!/usr/bin/env python3
"""Run the no-timing C3 mode11 v6.1 R/S/B-first correctness diagnostic."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import tempfile
import time
import traceback

import numpy as np
import tvm
from tvm import relay, rpc
from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor

import run_vta_c3_reuse_board as reuse_board
from c3_mode11_graph_context import (
    INPUT_NODE,
    SUCCESSOR_NODE,
    TARGET_NODE,
    TARGET_SYMBOL,
    StrictDebugCursor,
    append_jsonl,
    arrays_equal,
    classify_replay,
    core_reference,
    first_difference,
    install_debug_api_traps,
    sha256_file,
    summarize_array,
    write_json,
)
from c3_thread_sync_board import ExperimentBoard
from run_vta_c3_yolo608_graph_boot import random_input
from run_vta_p7r120_yolo_rpc_board import runtime_inventory


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_CONTRACT = GROUPED / "20260926_c3_mode11_graph_context_contract_v61_run03"
DEFAULT_MAIN = GROUPED / "20260926_c3_yolo512_evaluation_v5_graph_build_run01"
DEFAULT_DIAGNOSTIC = (
    GROUPED / "20260926_c3_yolo512_evaluation_v5_graph_correctness_diagnostic_build_run01"
)
DEFAULT_FREEZE = GROUPED / "20260926_c3_yolo512_evaluation_v5_freeze_run01"
DEFAULT_CANDIDATE = (
    GROUPED / "20260924_c3_yolo512_selector_v4_production_cross_compile_run01/binaries/"
    "ce645766d9859fcc4131a3bedc9156945d9101a4e2b61c6e25a720408d5afa82.so"
)
DEFAULT_OUTPUT = GROUPED / "20260926_c3_mode11_graph_context_board_v61_run03"
SEED = 0
SENTINEL = -113


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def connect(host, port, timeout):
    for _ in range(30):
        try:
            return rpc.connect(host, port, session_timeout=timeout)
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("RPC unavailable")


def local_params(root):
    values = relay.load_param_dict((Path(root) / "params.bin").read_bytes())
    return {name: np.array(value.numpy(), copy=True) for name, value in values.items()}


def upload_module(remote, source, temporary, label):
    upload = Path(temporary) / (label + ".so")
    shutil.copyfile(source, upload)
    remote.upload(str(upload))
    module = remote.load_module(upload.name)
    return module


def normal_executor(module, root, remote):
    executor = graph_executor.create(
        (Path(root) / "graph.json").read_text(encoding="utf-8"),
        module,
        [remote.ext_dev(0), remote.cpu(0)],
    )
    executor.load_params((Path(root) / "params.bin").read_bytes())
    return executor


def strict_debug_executor(module, root, remote, dump_root):
    executor = debug_executor.create(
        (Path(root) / "graph.json").read_text(encoding="utf-8"),
        module,
        [remote.ext_dev(0), remote.cpu(0)],
        dump_root=str(dump_root),
    )
    executor.load_params((Path(root) / "params.bin").read_bytes())
    traps = install_debug_api_traps(executor)
    missing = sorted(set(StrictDebugCursor.BANNED) - set(traps))
    if missing:
        raise RuntimeError("debug API trap surface incomplete: " + repr(missing))
    return executor, StrictDebugCursor(executor), traps


class Diagnostic:
    def __init__(self, args, output, board):
        self.args = args
        self.output = output
        self.board = board
        self.session_index = 0
        self.invocation_index = 0
        self.calls = output / "calls.jsonl"
        self.calls.write_text("", encoding="utf-8")

    def record(self, stage, session, api, executes_operator, **fields):
        row = {
            "invocation_index": self.invocation_index,
            "stage": stage,
            "session": session,
            "api": api,
            "executes_operator": bool(executes_operator),
            "latency_ms": None,
            **fields,
        }
        self.invocation_index += 1
        append_jsonl(self.calls, row)
        return row

    def session(self, label, binary, callback):
        port = self.args.port + 2 * self.session_index
        self.session_index += 1
        self.board.port, self.board.pid = port, None
        remote = None
        try:
            self.board.start()
            remote = connect(self.args.host, port, self.args.session_timeout)
            inventory, functions = runtime_inventory(remote)
            write_json(self.output / ("rpc_inventory_" + label + ".json"), inventory)
            with tempfile.TemporaryDirectory(prefix="c3_v61_" + label + "_") as temporary:
                module = upload_module(remote, binary, temporary, label)
                self.board.environment(require_affinity=True)
                return callback(remote, module, functions, Path(temporary))
        finally:
            remote = None
            gc.collect()
            try:
                self.board.stop()
            finally:
                self.board.pid = None

    @staticmethod
    def clear(functions):
        function = functions.get("vta.runtime.profiler_clear")
        if function is not None:
            function()

    @staticmethod
    def profile(functions):
        return json.loads(functions["vta.runtime.profiler_status"]())

    def full_graph(self, label, root, data):
        def body(remote, module, functions, _temporary):
            executor = normal_executor(module, root, remote)
            executor.set_input("data", data)
            self.clear(functions)
            executor.run()
            self.record("R", label, "GraphExecutor.run", True, complete_graph_calls=1)
            outputs = [executor.get_output(i).numpy() for i in range(executor.get_num_outputs())]
            return outputs, self.profile(functions)
        return self.session(label, Path(root) / "graphlib.so", body)

    def capture_s(self, root, data, snapshots):
        node_ops = [row["op"] for row in read(Path(root) / "graph.json")["nodes"]]

        def body(remote, module, functions, temporary):
            executor, cursor, traps = strict_debug_executor(
                module, root, remote, temporary / "debug_dump"
            )
            executor.set_input("data", data)
            captured, profiles = {}, {}
            for node, name in ((INPUT_NODE, "X"), (TARGET_NODE, "Y61_ref"),
                               (SUCCESSOR_NODE, "Y62_ref")):
                self.clear(functions)
                executed = cursor.execute_through(node)
                for current in executed:
                    self.record(
                        "S", "incumbent_debug", "execute_node",
                        node_ops[current] == "tvm_op", node=current, prefix_target=node,
                    )
                value = cursor.get_node_output(node, 0).numpy()
                self.record("S", "incumbent_debug", "get_node_output", False,
                            node=node, output_index=0)
                captured[name] = np.array(value, copy=True)
                profiles[name] = self.profile(functions)
            return captured, profiles, traps, cursor.calls
        captured, profiles, traps, calls = self.session(
            "s_incumbent_debug", Path(root) / "graphlib.so", body
        )
        for name, value in captured.items():
            np.save(snapshots / (name + ".npy"), value, allow_pickle=False)
        return captured, profiles, traps, calls

    def fused_once(self, label, binary, x, p27, p28):
        def body(remote, module, functions, _temporary):
            device = remote.ext_dev(0)
            # Preserve and record the independent allocation order.
            y_dev = tvm.nd.array(np.full((1, 16, 16, 16, 1, 16), SENTINEL, "int8"), device)
            x_dev = tvm.nd.array(x, device)
            p27_dev = tvm.nd.array(p27, device)
            p28_dev = tvm.nd.array(p28, device)
            before = [array.numpy() for array in (x_dev, p27_dev, p28_dev)]
            self.clear(functions)
            module[TARGET_SYMBOL](x_dev, p27_dev, p28_dev, y_dev)
            self.record("B", label, TARGET_SYMBOL, True, complete_target_calls=1,
                        allocation_order=["Y", "X", "p27", "p28"])
            output = y_dev.numpy()
            after = [array.numpy() for array in (x_dev, p27_dev, p28_dev)]
            return {
                "output": np.array(output, copy=True),
                "inputs_unchanged": [arrays_equal(a, b) for a, b in zip(before, after)],
                "sentinel_count": int(np.count_nonzero(output == SENTINEL)),
                "profile": self.profile(functions),
            }
        return self.session(label, binary, body)

    def core_once(self, binary, x, p27):
        def body(remote, module, functions, _temporary):
            device = remote.ext_dev(0)
            y_dev = tvm.nd.array(np.full((1, 16, 16, 16, 1, 16), SENTINEL, "int8"), device)
            x_dev, p27_dev = tvm.nd.array(x, device), tvm.nd.array(p27, device)
            before = [x_dev.numpy(), p27_dev.numpy()]
            self.clear(functions)
            module["main"](x_dev, p27_dev, y_dev)
            self.record("A", "isolated_core", "main", True, complete_core_calls=1,
                        allocation_order=["Y", "X", "p27"])
            output = y_dev.numpy()
            return {
                "output": np.array(output, copy=True),
                "inputs_unchanged": [
                    arrays_equal(before[0], x_dev.numpy()), arrays_equal(before[1], p27_dev.numpy())
                ],
                "sentinel_count": int(np.count_nonzero(output == SENTINEL)),
                "profile": self.profile(functions),
            }
        return self.session("a_isolated_core", binary, body)

    def debug_path(self, label, root, data, node):
        node_ops = [row["op"] for row in read(Path(root) / "graph.json")["nodes"]]

        def body(remote, module, functions, temporary):
            executor, cursor, traps = strict_debug_executor(
                module, root, remote, temporary / "debug_dump"
            )
            executor.set_input("data", data)
            self.clear(functions)
            executed = cursor.execute_through(node)
            for current in executed:
                self.record("C", label, "execute_node",
                            node_ops[current] == "tvm_op",
                            node=current, prefix_target=node)
            value = cursor.get_node_output(node, 0).numpy()
            self.record("C", label, "get_node_output", False, node=node, output_index=0)
            return np.array(value, copy=True), self.profile(functions), traps
        return self.session(label, Path(root) / "graphlib.so", body)


def output_differences(reference, values):
    if len(reference) != len(values):
        return [{"kind": "output_count", "expected": len(reference), "actual": len(values)}]
    return [first_difference(left, right) for left, right in zip(reference, values)]


def save_snapshot_inventory(snapshots):
    return {
        path.name: {"sha256": sha256_file(path), "npy": summarize_array(np.load(path))}
        for path in sorted(snapshots.glob("*.npy"))
    }


def run(args):
    contract_dir = Path(args.contract_dir).resolve()
    contract = read(contract_dir / "contract.json")
    if contract.get("status") != "FULL_FUSION_CONTRACT_RESOLVED":
        raise ValueError("FULL_FUSION_CONTRACT_UNRESOLVED")
    if contract.get("formal_latency_forbidden") is not True or contract.get("seed") != SEED:
        raise ValueError("diagnostic contract drift")
    main, diagnostic = Path(args.main_build).resolve(), Path(args.diagnostic_build).resolve()
    roots = {
        "stock": main / "stock",
        "incumbent": main / "incumbent",
        "mode11": diagnostic / "mode11_only_eec24a99",
    }
    candidate = Path(args.candidate_so).resolve()
    if sha256_file(candidate) != contract["frozen_candidate_so_sha256"]:
        raise ValueError("candidate binary drift")
    for root in roots.values():
        if not (root / "graphlib.so").is_file():
            raise FileNotFoundError(root / "graphlib.so")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    snapshots = output / "snapshots"
    snapshots.mkdir()
    freeze = read(Path(args.freeze_dir) / "contract.json")
    runtime = read(freeze["runtime_contract"])
    for name, path in runtime["runtime_paths"].items():
        if sha256_file(path) != runtime["runtime_sha256"][name]:
            raise ValueError("runtime changed: " + name)
    preregistered = {
        "schema": "c3_mode11_graph_context_board_v61",
        "status": "FROZEN_BEFORE_BOARD_DIAGNOSTIC",
        "contract_dir": str(contract_dir),
        "contract_sha256": sha256_file(contract_dir / "contract.json"),
        "seed": SEED,
        "stage_order": ["R", "S", "B", "A", "C_if_B_passes", "D_if_needed"],
        "timer_calls": 0,
        "latency_fields": None,
        "candidate_reselection": False,
        "candidate_modification": False,
        "transparent_retry": False,
    }
    write_json(output / "preregistered.json", preregistered)
    board = ExperimentBoard(args, "/var/volatile/" + output.name, args.port)
    diagnostic_runner = Diagnostic(args, output, board)
    matrix, first_differences = {}, {}
    initial = None
    try:
        initial = board.state()
        reuse_board.validate_state(initial, exclusive=True)
        write_json(output / "preflight.json", {
            "state": initial, "environment": board.environment(), "board_reboot_requested": False,
        })
        reuse_board.RUNTIME = runtime["runtime_sha256"]
        write_json(output / "runtime_attestation.json", board.deploy(runtime["runtime_paths"]))
        write_json(output / "bitstream_attestation.json", board.reload())
        data = random_input(SEED, resolution=512)

        # R: one ordinary graph call for each frozen graph.  No timer is constructed.
        stock, stock_profile = diagnostic_runner.full_graph("r_stock", roots["stock"], data)
        incumbent, incumbent_profile = diagnostic_runner.full_graph(
            "r_incumbent", roots["incumbent"], data
        )
        mode11, mode11_profile = diagnostic_runner.full_graph("r_mode11", roots["mode11"], data)
        incumbent_diff = output_differences(stock, incumbent)
        mode11_diff = output_differences(stock, mode11)
        matrix["R"] = {
            "status": "passed" if not any(incumbent_diff) and any(mode11_diff) else "failed",
            "incumbent_correct": not any(incumbent_diff),
            "mode11_reproduced_error": any(mode11_diff),
            "incumbent_first_differences": incumbent_diff,
            "mode11_first_differences": mode11_diff,
            "profiles": {"stock": stock_profile, "incumbent": incumbent_profile,
                         "mode11": mode11_profile},
        }
        if matrix["R"]["status"] != "passed":
            earliest, flags = classify_replay(matrix)
            raise RuntimeError(earliest)
        np.save(snapshots / "R_stock_Y62.npy", stock[4], allow_pickle=False)
        np.save(snapshots / "R_incumbent_Y62.npy", incumbent[4], allow_pickle=False)

        # S: a single monotonic incumbent debug path supplies the exact real tensors.
        captured, profiles, traps, cursor_calls = diagnostic_runner.capture_s(
            roots["incumbent"], data, snapshots
        )
        params = local_params(roots["incumbent"])
        p27, p28 = params["p27"], params["p28"]
        np.save(snapshots / "p27.npy", p27, allow_pickle=False)
        np.save(snapshots / "p28.npy", p28, allow_pickle=False)
        x, y61_ref, y62_ref = captured["X"], captured["Y61_ref"], captured["Y62_ref"]
        y62_matches_ordinary = arrays_equal(incumbent[4], y62_ref)
        matrix["S"] = {
            "status": "passed" if y62_matches_ordinary else "DEBUG_PREFIX_REFERENCE_INVALID",
            "y62_matches_ordinary_graph_head4": y62_matches_ordinary,
            "y62_vs_ordinary_graph_head4": first_difference(incumbent[4], y62_ref),
            "trapped_debug_apis": traps, "cursor_calls": cursor_calls,
            "profiles": profiles, "snapshots": save_snapshot_inventory(snapshots),
        }
        if not y62_matches_ordinary:
            raise RuntimeError("DEBUG_PREFIX_REFERENCE_INVALID")

        # B: direct four-argument replay from each exact frozen graph library.
        incumbent_b = diagnostic_runner.fused_once(
            "b_incumbent", roots["incumbent"] / "graphlib.so", x, p27, p28
        )
        np.save(snapshots / "B_incumbent.npy", incumbent_b["output"], allow_pickle=False)
        incumbent_ok = arrays_equal(y61_ref, incumbent_b["output"])
        matrix["B"] = {
            "incumbent_matches_reference": incumbent_ok,
            "incumbent_inputs_unchanged": incumbent_b["inputs_unchanged"],
            "incumbent_sentinel_count": incumbent_b["sentinel_count"],
            "incumbent_profile": incumbent_b["profile"],
        }
        first_differences["B_incumbent_vs_Y61_ref"] = first_difference(
            y61_ref, incumbent_b["output"]
        )
        if not incumbent_ok or not all(incumbent_b["inputs_unchanged"]):
            matrix["B"]["status"] = "REFERENCE_REPLAY_CONTRACT_INVALID"
            raise RuntimeError("REFERENCE_REPLAY_CONTRACT_INVALID")

        mode11_b = diagnostic_runner.fused_once(
            "b_mode11_first", roots["mode11"] / "graphlib.so", x, p27, p28
        )
        mode11_ok = arrays_equal(y61_ref, mode11_b["output"])
        matrix["B"].update({
            "mode11_matches_reference": mode11_ok,
            "mode11_inputs_unchanged": mode11_b["inputs_unchanged"],
            "mode11_sentinel_count": mode11_b["sentinel_count"],
            "mode11_profile": mode11_b["profile"],
        })
        first_differences["B_mode11_vs_Y61_ref"] = first_difference(
            y61_ref, mode11_b["output"]
        )
        if not all(mode11_b["inputs_unchanged"]):
            raise RuntimeError("mode11 B modified an independent input")

        if not mode11_ok:
            repeated = diagnostic_runner.fused_once(
                "b_mode11_repeat", roots["mode11"] / "graphlib.so", x, p27, p28
            )
            repeat_ok = arrays_equal(y61_ref, repeated["output"])
            incumbent_zero = diagnostic_runner.fused_once(
                "b_incumbent_zero_p28", roots["incumbent"] / "graphlib.so",
                x, p27, np.zeros_like(p28),
            )
            mode11_zero = diagnostic_runner.fused_once(
                "b_mode11_zero_p28", roots["mode11"] / "graphlib.so",
                x, p27, np.zeros_like(p28),
            )
            zero_ok = arrays_equal(incumbent_zero["output"], mode11_zero["output"])
            matrix["B"].update({
                "status": "FUSED_FUNCTION_INDEPENDENT_BUFFER_DIVERGENCE",
                "mode11_repeat_matches_reference": repeat_ok,
                "mode11_repeat_same_wrong_output": arrays_equal(
                    mode11_b["output"], repeated["output"]
                ),
                "mode11_zero_p28_matches_incumbent": zero_ok,
                "mode11_zero_p28_inputs_unchanged": mode11_zero["inputs_unchanged"],
                "incumbent_zero_p28_inputs_unchanged": incumbent_zero["inputs_unchanged"],
            })
            first_differences["B_mode11_repeat_vs_Y61_ref"] = first_difference(
                y61_ref, repeated["output"]
            )
            first_differences["B_zero_p28_mode11_vs_incumbent"] = first_difference(
                incumbent_zero["output"], mode11_zero["output"]
            )

        # A is auxiliary and cannot overwrite the B conclusion.
        core_expected = core_reference(x, p27)
        core_actual = diagnostic_runner.core_once(candidate, x, p27)
        matrix["A"] = {
            "status": "CORE_REPLAY_PASS" if arrays_equal(core_expected, core_actual["output"])
            else "CORE_REPLAY_FAIL",
            "inputs_unchanged": core_actual["inputs_unchanged"],
            "sentinel_count": core_actual["sentinel_count"],
            "profile": core_actual["profile"],
        }
        first_differences["A_core_vs_CPU"] = first_difference(
            core_expected, core_actual["output"]
        )

        if not mode11_ok:
            matrix["C"] = {"status": "not_run_B_diverged"}
            matrix["D"] = {"status": "not_run_B_diverged"}
        else:
            c = {}
            for node, field in ((INPUT_NODE, "entry"), (TARGET_NODE, "exit"),
                                (SUCCESSOR_NODE, "successor")):
                inc_value, inc_profile, _ = diagnostic_runner.debug_path(
                    "c_incumbent_" + field, roots["incumbent"], data, node
                )
                mode_value, mode_profile, _ = diagnostic_runner.debug_path(
                    "c_mode11_" + field, roots["mode11"], data, node
                )
                c[field + "_matches"] = arrays_equal(inc_value, mode_value)
                c[field + "_profiles"] = {"incumbent": inc_profile, "mode11": mode_profile}
                first_differences["C_" + field] = first_difference(inc_value, mode_value)
            c["full_graph_correct"] = matrix["R"]["mode11_reproduced_error"] is False
            c["debug_paths_correct"] = all(
                c[key] for key in ("entry_matches", "exit_matches", "successor_matches")
            )
            c["post_target_output"] = "UNOBSERVABLE_WITH_CURRENT_RUNTIME"
            c["status"] = "complete"
            matrix["C"] = c
            # D is deliberately bounded.  The direct fresh target call is B; further state
            # probes are only meaningful if C or the ordinary graph still differs.
            matrix["D"] = {
                "status": "CURRENTLY_ONLY_ORDINARY_GRAPH_PATH_REPRODUCES"
                if c["debug_paths_correct"] and not c["full_graph_correct"]
                else "not_required_no_remaining_failure",
                "fresh_target_call": "B",
                "extra_flush_drain_or_sync_added": False,
            }

        earliest, flags = classify_replay(matrix)
        result = {
            "schema": "c3_mode11_graph_context_replay_matrix_v61",
            "status": "C3_MODE11_GRAPH_CONTEXT_DIAGNOSTIC_COMPLETE",
            "earliest_observed_divergence": earliest,
            "observation_flags": flags,
            "matrix": matrix,
            "latency_values_collected": False,
            "timer_calls": 0,
            "candidate_reselected": False,
            "post_target_output": "UNOBSERVABLE_WITH_CURRENT_RUNTIME",
        }
        write_json(output / "replay_matrix.json", result)
        write_json(output / "first_difference.json", first_differences)
        after = board.state()
        reuse_board.validate_state(after, exclusive=True)
        if after["boot_id"] != initial["boot_id"]:
            raise RuntimeError("board boot changed during diagnostic")
        write_json(output / "summary.json", {
            "status": result["status"], "boot_id": initial["boot_id"],
            "earliest_observed_divergence": earliest, "observation_flags": flags,
            "real_operator_invocations": sum(
                1 for line in diagnostic_runner.calls.read_text().splitlines()
                if json.loads(line)["executes_operator"]
            ),
            "latency_values_collected": False, "timer_calls": 0,
        })
    except BaseException as error:
        if matrix:
            earliest, flags = classify_replay(matrix)
            write_json(output / "replay_matrix.json", {
                "schema": "c3_mode11_graph_context_replay_matrix_v61",
                "status": "C3_MODE11_GRAPH_CONTEXT_DIAGNOSTIC_STOPPED_AT_GATE",
                "earliest_observed_divergence": earliest,
                "observation_flags": flags,
                "matrix": matrix,
                "latency_values_collected": False,
                "timer_calls": 0,
            })
            write_json(output / "first_difference.json", first_differences)
        write_json(output / "invalid_or_stopped.json", {
            "message": str(error), "type": type(error).__name__,
            "traceback": traceback.format_exc()[-16000:],
            "latency_values_collected": False, "timer_calls": 0,
        })
        raise
    finally:
        try:
            board.stop()
        except Exception:
            pass
        write_json(output / "artifact_hashes.json", {"artifacts": {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "artifact_hashes.json"
        }})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--main-build", default=str(DEFAULT_MAIN))
    parser.add_argument("--diagnostic-build", default=str(DEFAULT_DIAGNOSTIC))
    parser.add_argument("--freeze-dir", default=str(DEFAULT_FREEZE))
    parser.add_argument("--candidate-so", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=1800)
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
