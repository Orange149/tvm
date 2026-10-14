#!/usr/bin/env python3
"""No-timing ordinary-GraphExecutor prefix-tap diagnostic for C3 mode11."""

from __future__ import annotations

import json
from pathlib import Path
import traceback

import numpy as np
from tvm.contrib import graph_executor

import run_vta_c3_mode11_graph_context_diagnostic as base
import run_vta_c3_reuse_board as reuse_board
from c3_mode11_graph_context import (
    TARGET_SYMBOL,
    arrays_equal,
    core_reference,
    first_difference,
    ordinary_prefix_tap_graph,
    sha256_bytes,
    sha256_file,
    successor62_reference,
    write_json,
)
from c3_thread_sync_board import ExperimentBoard
from run_vta_c3_yolo608_graph_boot import random_input


GROUPED = base.GROUPED
DEFAULT_OUTPUT = GROUPED / "20260926_c3_mode11_graph_context_board_v63_state_run05"
HISTORICAL_SEEDS = (0, 20250901, 20260910)


def canonical_sha(value):
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def float_difference(expected, actual, atol=5e-6, rtol=1e-6):
    left, right = np.asarray(expected), np.asarray(actual)
    if left.shape != right.shape or left.dtype != right.dtype:
        return first_difference(left, right)
    delta = np.abs(left.astype("float64") - right.astype("float64"))
    bad = delta > (atol + rtol * np.abs(left.astype("float64")))
    flat = np.flatnonzero(bad.reshape(-1))
    if not flat.size:
        return None
    index = int(flat[0])
    return {
        "kind": "float_tolerance",
        "flat_index": index,
        "index": [int(x) for x in np.unravel_index(index, left.shape)],
        "expected": float(left.reshape(-1)[index]),
        "actual": float(right.reshape(-1)[index]),
        "abs_difference": float(delta.reshape(-1)[index]),
        "max_abs_difference": float(delta.max()),
        "mismatch_count": int(flat.size),
        "atol": atol,
        "rtol": rtol,
    }


class FreshBitstreamDiagnostic(base.Diagnostic):
    """Give each clean diagnostic RPC the same frozen FPGA initial state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bitstream_resets = []

    def session(self, label, binary, callback):
        attestation = self.board.reload()
        self.bitstream_resets.append({
            "session": label, "reason": "clean_session", "attestation": attestation,
        })
        return super().session(label, binary, callback)

    def reset_sequence(self, label):
        attestation = self.board.reload()
        self.bitstream_resets.append({
            "session": label, "reason": "shared_state_sequence_start",
            "attestation": attestation,
        })
        return attestation

    def session_without_reload(self, label, binary, callback):
        """Run a fresh RPC while deliberately preserving FPGA state from its predecessor."""
        return super().session(label, binary, callback)


def full_graph_batch_without_reload(diagnostic, label, root, seeds):
    """Match the historical runner: one executor session, several graph calls."""
    seeds = tuple(int(seed) for seed in seeds)

    def body(remote, module, functions, _temporary):
        executor = base.normal_executor(module, root, remote)
        rows = []
        for seed in seeds:
            executor.set_input("data", random_input(seed, resolution=512))
            diagnostic.clear(functions)
            executor.run()
            diagnostic.record(
                "D", label, "GraphExecutor.run", True,
                complete_graph_calls=1, seed=seed, shared_fpga_state=True,
            )
            outputs = [
                np.array(executor.get_output(i).numpy(), copy=True)
                for i in range(executor.get_num_outputs())
            ]
            rows.append({"seed": seed, "outputs": outputs,
                         "profile": diagnostic.profile(functions)})
        return rows

    return diagnostic.session_without_reload(
        label, Path(root) / "graphlib.so", body
    )


def run_state_sequence(diagnostic, label, steps, target_root, clean_reference):
    """Reset once, execute a frozen predecessor sequence, then the target mode11."""
    diagnostic.reset_sequence(label)
    step_rows = []
    for step_name, root, seeds in steps:
        rows = full_graph_batch_without_reload(
            diagnostic, label + "_" + step_name, root, seeds
        )
        step_rows.append({
            "variant": step_name,
            "seeds": [row["seed"] for row in rows],
            "output4_sha256": [sha256_bytes(row["outputs"][4].tobytes()) for row in rows],
        })
    target = full_graph_batch_without_reload(
        diagnostic, label + "_target", target_root, (base.SEED,)
    )[0]
    differences = base.output_differences(clean_reference, target["outputs"])
    return {
        "label": label,
        "predecessors": step_rows,
        "target_correct_vs_clean_stock": not any(differences),
        "target_first_differences": differences,
        "target_output4_sha256": sha256_bytes(target["outputs"][4].tobytes()),
        "target_profile": target["profile"],
    }


def tap_once(diagnostic, label, root, tap_graph, data):
    def body(remote, module, functions, _temporary):
        executor = graph_executor.create(
            json.dumps(tap_graph, separators=(",", ":")),
            module,
            [remote.ext_dev(0), remote.cpu(0)],
        )
        executor.load_params((Path(root) / "params.bin").read_bytes())
        executor.set_input("data", data)
        diagnostic.clear(functions)
        executor.run()
        diagnostic.record(
            "S2", label, "GraphExecutor.run", True,
            complete_prefix_graph_calls=1, last_enabled_node=61,
            disabled_successor_node=62,
        )
        outputs = [executor.get_output(i).numpy() for i in range(2)]
        return [np.array(value, copy=True) for value in outputs], diagnostic.profile(functions)

    return diagnostic.session(label, Path(root) / "graphlib.so", body)


def save_array(root, name, value):
    np.save(Path(root) / (name + ".npy"), np.asarray(value), allow_pickle=False)


def make_tap_contract(roots, output):
    rows, graphs = {}, {}
    for name in ("incumbent", "mode11"):
        source = json.loads((Path(roots[name]) / "graph.json").read_text(encoding="utf-8"))
        tapped = ordinary_prefix_tap_graph(source)
        path = output / (name + "_ordinary_prefix_tap_graph.json")
        path.write_text(json.dumps(tapped, separators=(",", ":")), encoding="utf-8")
        rows[name] = {
            "frozen_graph_sha256": sha256_file(Path(roots[name]) / "graph.json"),
            "tap_graph_sha256": sha256_file(path),
            "nodes_0_through_61_sha256": canonical_sha(source["nodes"][:62]),
            "storage_attrs_sha256": canonical_sha(source["attrs"]),
            "node_row_ptr_sha256": canonical_sha(source["node_row_ptr"]),
            "only_runtime_node_change": 62,
            "heads": tapped["heads"],
            "graphlib_sha256": sha256_file(Path(roots[name]) / "graphlib.so"),
            "params_sha256": sha256_file(Path(roots[name]) / "params.bin"),
        }
        graphs[name] = tapped
    if rows["incumbent"]["nodes_0_through_61_sha256"] != rows["mode11"][
        "nodes_0_through_61_sha256"
    ]:
        # The target node metadata is the same even though graphlib implementations differ.
        raise RuntimeError("frozen graph prefix JSON differs between variants")
    contract = {
        "schema": "c3_mode11_ordinary_prefix_tap_v62",
        "purpose": "capture real X/Y61 through normal GraphExecutor.run without debug APIs",
        "node62_disabled_after_target": True,
        "storage_plan_unchanged": True,
        "nodes_0_through_61_unchanged": True,
        "candidate_binary_modified": False,
        "formal_latency_forbidden": True,
        "variants": rows,
    }
    write_json(output / "tap_contract.json", contract)
    return graphs, contract


def run(args):
    contract_dir = Path(args.contract_dir).resolve()
    contract = base.read(contract_dir / "contract.json")
    if contract.get("status") != "FULL_FUSION_CONTRACT_RESOLVED":
        raise ValueError("FULL_FUSION_CONTRACT_UNRESOLVED")
    main = Path(args.main_build).resolve()
    diagnostic_build = Path(args.diagnostic_build).resolve()
    roots = {
        "stock": main / "stock",
        "incumbent": main / "incumbent",
        "selector": main / "selector_v4",
        "other_mode11": diagnostic_build / "mode11_only_2fa33cab",
        "mode11": diagnostic_build / "mode11_only_eec24a99",
    }
    candidate = Path(args.candidate_so).resolve()
    if sha256_file(candidate) != contract["frozen_candidate_so_sha256"]:
        raise ValueError("candidate binary drift")
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    snapshots = output / "snapshots"
    snapshots.mkdir()
    tap_graphs, tap_contract = make_tap_contract(roots, output)

    freeze = base.read(Path(args.freeze_dir) / "contract.json")
    runtime = base.read(freeze["runtime_contract"])
    for name, path in runtime["runtime_paths"].items():
        if sha256_file(path) != runtime["runtime_sha256"][name]:
            raise ValueError("runtime changed: " + name)
    write_json(output / "preregistered.json", {
        "schema": "c3_mode11_graph_context_board_v63_state",
        "status": "FROZEN_BEFORE_BOARD_DIAGNOSTIC",
        "parent_v61": str(base.DEFAULT_OUTPUT.parent / "20260926_c3_mode11_graph_context_board_v61_run03"),
        "supersedes_attempt": str(output.parent / "20260926_c3_mode11_graph_context_board_v63_state_run04"),
        "supersession_reason": "node62 CPU sigmoid tolerance 1e-6 rejected a 4.0531158447265625e-6 implementation difference",
        "seed": base.SEED,
        "historical_seed_order": list(HISTORICAL_SEEDS),
        "stage_order": ["R_clean", "S2_ordinary_prefix_tap", "B", "A", "D_state_sequences"],
        "timer_calls": 0,
        "candidate_reselection": False,
        "candidate_modification": False,
        "debug_executor_used": False,
        "clean_calls_reload_same_bitstream": True,
        "state_sequences_reload_only_at_sequence_start": True,
        "transparent_retry": False,
        "tap_contract_sha256": sha256_file(output / "tap_contract.json"),
    })

    board = ExperimentBoard(args, "/var/volatile/" + output.name, args.port)
    diagnostic = FreshBitstreamDiagnostic(args, output, board)
    matrix, differences = {}, {}
    initial = None
    try:
        initial = board.state()
        reuse_board.validate_state(initial, exclusive=True)
        write_json(output / "preflight.json", {
            "state": initial,
            "environment": board.environment(),
            "board_reboot_requested": False,
            "bitstream_reloaded": True,
            "clean_sessions_reload_bitstream": True,
            "state_sequences_share_bitstream_after_one_reset": True,
        })
        reuse_board.RUNTIME = runtime["runtime_sha256"]
        write_json(output / "runtime_attestation.json", board.deploy(runtime["runtime_paths"]))
        write_json(output / "bitstream_attestation.json", board.reload())
        data = random_input(base.SEED, resolution=512)

        stock, stock_profile = diagnostic.full_graph("r_stock", roots["stock"], data)
        incumbent, incumbent_profile = diagnostic.full_graph(
            "r_incumbent", roots["incumbent"], data
        )
        mode11, mode11_profile = diagnostic.full_graph("r_mode11", roots["mode11"], data)
        incumbent_diff = base.output_differences(stock, incumbent)
        mode11_diff = base.output_differences(stock, mode11)
        matrix["R"] = {
            "status": "passed" if not any(incumbent_diff) else "failed",
            "clean_incumbent_correct": not any(incumbent_diff),
            "clean_mode11_correct": not any(mode11_diff),
            "clean_mode11_reproduced_historical_error": any(mode11_diff),
            "incumbent_first_differences": incumbent_diff,
            "mode11_first_differences": mode11_diff,
            "profiles": {"stock": stock_profile, "incumbent": incumbent_profile,
                         "mode11": mode11_profile},
            "interpretation": (
                "clean bitstream standalone mode11 is correct"
                if not any(mode11_diff)
                else "historical mode11 failure also reproduces from clean bitstream"
            ),
        }
        if matrix["R"]["status"] != "passed":
            raise RuntimeError("CLEAN_INCUMBENT_REFERENCE_INVALID")
        save_array(snapshots, "R_stock_Y62", stock[4])
        save_array(snapshots, "R_incumbent_Y62", incumbent[4])
        save_array(snapshots, "R_mode11_Y62", mode11[4])

        inc1, inc_profile1 = tap_once(
            diagnostic, "s2_incumbent_first", roots["incumbent"], tap_graphs["incumbent"], data
        )
        inc2, inc_profile2 = tap_once(
            diagnostic, "s2_incumbent_repeat", roots["incumbent"], tap_graphs["incumbent"], data
        )
        mod1, mod_profile1 = tap_once(
            diagnostic, "s2_mode11_first", roots["mode11"], tap_graphs["mode11"], data
        )
        mod2, mod_profile2 = tap_once(
            diagnostic, "s2_mode11_repeat", roots["mode11"], tap_graphs["mode11"], data
        )
        x, y61 = inc1
        mode_x, mode_y61 = mod1
        for name, value in (
            ("X", x), ("Y61_ref", y61), ("X_incumbent_repeat", inc2[0]),
            ("Y61_incumbent_repeat", inc2[1]), ("X_mode11", mode_x),
            ("Y61_mode11", mode_y61), ("X_mode11_repeat", mod2[0]),
            ("Y61_mode11_repeat", mod2[1]),
        ):
            save_array(snapshots, name, value)

        inc_successor = successor62_reference(y61)
        mode_successor = successor62_reference(mode_y61)
        inc_successor_diff = float_difference(incumbent[4], inc_successor)
        mode_successor_diff = float_difference(mode11[4], mode_successor)
        differences["S2_incumbent_successor_reference"] = inc_successor_diff
        differences["S2_mode11_successor_reference"] = mode_successor_diff
        differences["S2_target_exit"] = first_difference(y61, mode_y61)
        s2_ok = (
            arrays_equal(x, inc2[0]) and arrays_equal(y61, inc2[1])
            and arrays_equal(x, mode_x) and arrays_equal(mode_x, mod2[0])
            and arrays_equal(mode_y61, mod2[1])
            and inc_successor_diff is None and mode_successor_diff is None
        )
        matrix["S2"] = {
            "status": "passed" if s2_ok else "ORDINARY_PREFIX_TAP_INVALID",
            "incumbent_repeat_X_equal": arrays_equal(x, inc2[0]),
            "incumbent_repeat_Y61_equal": arrays_equal(y61, inc2[1]),
            "cross_variant_X_equal": arrays_equal(x, mode_x),
            "mode11_repeat_X_equal": arrays_equal(mode_x, mod2[0]),
            "mode11_repeat_Y61_equal": arrays_equal(mode_y61, mod2[1]),
            "target_exit_equal": arrays_equal(y61, mode_y61),
            "incumbent_successor_matches_ordinary_head4": inc_successor_diff is None,
            "mode11_successor_matches_ordinary_head4": mode_successor_diff is None,
            "profiles": {"incumbent_first": inc_profile1, "incumbent_repeat": inc_profile2,
                         "mode11_first": mod_profile1, "mode11_repeat": mod_profile2},
            "tap_contract": tap_contract,
        }
        if not s2_ok:
            raise RuntimeError("ORDINARY_PREFIX_TAP_INVALID")

        params = base.local_params(roots["incumbent"])
        p27, p28 = params["p27"], params["p28"]
        save_array(snapshots, "p27", p27)
        save_array(snapshots, "p28", p28)

        incumbent_b = diagnostic.fused_once(
            "b_incumbent", roots["incumbent"] / "graphlib.so", x, p27, p28
        )
        save_array(snapshots, "B_incumbent", incumbent_b["output"])
        incumbent_ok = arrays_equal(y61, incumbent_b["output"])
        matrix["B"] = {
            "incumbent_matches_tap_reference": incumbent_ok,
            "incumbent_inputs_unchanged": incumbent_b["inputs_unchanged"],
            "incumbent_profile": incumbent_b["profile"],
        }
        differences["B_incumbent_vs_Y61_ref"] = first_difference(y61, incumbent_b["output"])
        if not incumbent_ok or not all(incumbent_b["inputs_unchanged"]):
            matrix["B"]["status"] = "REFERENCE_REPLAY_CONTRACT_INVALID"
            raise RuntimeError("REFERENCE_REPLAY_CONTRACT_INVALID")

        mode_b = diagnostic.fused_once(
            "b_mode11_first", roots["mode11"] / "graphlib.so", x, p27, p28
        )
        save_array(snapshots, "B_mode11_first", mode_b["output"])
        mode_ok = arrays_equal(y61, mode_b["output"])
        matrix["B"].update({
            "mode11_matches_reference": mode_ok,
            "mode11_matches_mode11_tap": arrays_equal(mode_y61, mode_b["output"]),
            "mode11_inputs_unchanged": mode_b["inputs_unchanged"],
            "mode11_profile": mode_b["profile"],
        })
        differences["B_mode11_vs_Y61_ref"] = first_difference(y61, mode_b["output"])
        differences["B_mode11_vs_mode11_tap"] = first_difference(mode_y61, mode_b["output"])
        if not all(mode_b["inputs_unchanged"]):
            raise RuntimeError("mode11 B modified an independent input")

        flags = []
        if not mode_ok:
            repeated = diagnostic.fused_once(
                "b_mode11_repeat", roots["mode11"] / "graphlib.so", x, p27, p28
            )
            save_array(snapshots, "B_mode11_repeat", repeated["output"])
            incumbent_zero = diagnostic.fused_once(
                "b_incumbent_zero_p28", roots["incumbent"] / "graphlib.so",
                x, p27, np.zeros_like(p28),
            )
            mode_zero = diagnostic.fused_once(
                "b_mode11_zero_p28", roots["mode11"] / "graphlib.so",
                x, p27, np.zeros_like(p28),
            )
            save_array(snapshots, "B_incumbent_zero_p28", incumbent_zero["output"])
            save_array(snapshots, "B_mode11_zero_p28", mode_zero["output"])
            repeat_wrong = not arrays_equal(y61, repeated["output"])
            same_wrong = arrays_equal(mode_b["output"], repeated["output"])
            zero_equal = arrays_equal(incumbent_zero["output"], mode_zero["output"])
            matrix["B"].update({
                "status": "FUSED_FUNCTION_INDEPENDENT_BUFFER_DIVERGENCE",
                "mode11_repeat_matches_reference": not repeat_wrong,
                "mode11_repeat_same_wrong_output": same_wrong,
                "mode11_zero_p28_matches_incumbent": zero_equal,
                "mode11_repeat_inputs_unchanged": repeated["inputs_unchanged"],
                "incumbent_zero_inputs_unchanged": incumbent_zero["inputs_unchanged"],
                "mode11_zero_inputs_unchanged": mode_zero["inputs_unchanged"],
            })
            differences["B_mode11_repeat_vs_Y61_ref"] = first_difference(
                y61, repeated["output"]
            )
            differences["B_zero_p28_mode11_vs_incumbent"] = first_difference(
                incumbent_zero["output"], mode_zero["output"]
            )
            flags.append("P28_SENSITIVE" if zero_equal else "P28_ZERO_STILL_FAILS")
            if not (repeat_wrong and same_wrong):
                flags.append("UNSTABLE_REPRODUCTION")
            earliest = "FUSED_FUNCTION_INDEPENDENT_BUFFER_DIVERGENCE"
        else:
            matrix["B"]["status"] = "passed"
            earliest = (
                "GRAPH_CONTEXT_TARGET_EXIT_DIVERGENCE"
                if not arrays_equal(y61, mode_y61)
                else "NO_TARGET_DIVERGENCE"
            )

        core_expected = core_reference(x, p27)
        core_actual = diagnostic.core_once(candidate, x, p27)
        save_array(snapshots, "A_core", core_actual["output"])
        matrix["A"] = {
            "status": "CORE_REPLAY_PASS" if arrays_equal(core_expected, core_actual["output"])
            else "CORE_REPLAY_FAIL",
            "inputs_unchanged": core_actual["inputs_unchanged"],
            "profile": core_actual["profile"],
        }
        differences["A_core_vs_CPU"] = first_difference(core_expected, core_actual["output"])

        matrix["D"] = {
            "status": "not_run_b_failed",
            "purpose": "locate stateful predecessor required by the historical graph failure",
        }
        if mode_ok:
            sequences = [
                ("exact_historical", [
                    ("stock", roots["stock"], HISTORICAL_SEEDS),
                    ("incumbent", roots["incumbent"], HISTORICAL_SEEDS),
                    ("selector", roots["selector"], HISTORICAL_SEEDS),
                    ("other_mode11", roots["other_mode11"], HISTORICAL_SEEDS),
                ]),
                ("stock_only", [
                    ("stock", roots["stock"], HISTORICAL_SEEDS),
                ]),
                ("incumbent_only", [
                    ("incumbent", roots["incumbent"], HISTORICAL_SEEDS),
                ]),
                ("selector_only", [
                    ("selector", roots["selector"], HISTORICAL_SEEDS),
                ]),
                ("other_mode11_only", [
                    ("other_mode11", roots["other_mode11"], HISTORICAL_SEEDS),
                ]),
                ("target_self", [
                    ("mode11", roots["mode11"], HISTORICAL_SEEDS),
                ]),
            ]
            state_rows = [
                run_state_sequence(
                    diagnostic, "d_" + label, steps, roots["mode11"], stock
                )
                for label, steps in sequences
            ]
            failing_sequences = [
                row["label"] for row in state_rows
                if not row["target_correct_vs_clean_stock"]
            ]
            matrix["D"] = {
                "status": (
                    "PREDECESSOR_STATE_DEPENDENT_FULL_GRAPH_DIVERGENCE"
                    if failing_sequences else "HISTORICAL_SEQUENCE_NOT_REPRODUCED"
                ),
                "same_bitstream_preserved_within_each_sequence": True,
                "fresh_rpc_between_program_variants": True,
                "sequences": state_rows,
                "failing_sequences": failing_sequences,
            }
            for row in state_rows:
                differences[row["label"] + "_target_vs_clean_stock"] = row[
                    "target_first_differences"
                ]
            if failing_sequences:
                earliest = "PREDECESSOR_STATE_DEPENDENT_FULL_GRAPH_DIVERGENCE"
                flags.append("STATE_SEQUENCE_SENSITIVE")
                if "d_exact_historical" in failing_sequences:
                    flags.append("HISTORICAL_SEQUENCE_REPRODUCED")
            else:
                flags.append("HISTORICAL_SEQUENCE_NOT_REPRODUCED")

        result = {
            "schema": "c3_mode11_graph_context_replay_matrix_v63_state",
            "status": "C3_MODE11_STATE_DIAGNOSTIC_COMPLETE",
            "earliest_observed_divergence": earliest,
            "observation_flags": flags,
            "matrix": matrix,
            "latency_values_collected": False,
            "timer_calls": 0,
            "candidate_reselected": False,
            "debug_executor_used": False,
        }
        write_json(output / "replay_matrix.json", result)
        write_json(output / "first_difference.json", differences)
        after = board.state()
        reuse_board.validate_state(after, exclusive=True)
        if after["boot_id"] != initial["boot_id"]:
            raise RuntimeError("board boot changed during diagnostic")
        write_json(output / "summary.json", {
            "status": result["status"], "boot_id": initial["boot_id"],
            "earliest_observed_divergence": earliest, "observation_flags": flags,
            "latency_values_collected": False, "timer_calls": 0,
        })
        lines = [
            "# C3 mode11 v6.3 干净状态与前驱状态诊断报告", "",
            "- 最早可靠差异边界：`{}`。".format(earliest),
            "- 干净 bitstream 下 mode11 整图正确：`{}`。".format(
                matrix["R"]["clean_mode11_correct"]
            ),
            "- 普通前缀快照资格：`{}`。".format(matrix["S2"]["status"]),
            "- incumbent 独立 B 门禁：`{}`。".format(matrix["B"]["incumbent_matches_tap_reference"]),
            "- mode11 独立 B 匹配参考：`{}`。".format(matrix["B"].get("mode11_matches_reference")),
            "- 前驱状态矩阵：`{}`。".format(matrix["D"]["status"]),
            "- A 辅助核心：`{}`。".format(matrix["A"]["status"]),
            "- 未采集 latency，未使用 debug executor，未重选或修改候选。", "",
            "该报告区分干净状态函数正确性与跨程序状态依赖，不把具体硬件状态命名为唯一根因。",
        ]
        (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except BaseException as error:
        write_json(output / "replay_matrix.json", {
            "schema": "c3_mode11_graph_context_replay_matrix_v63_state",
            "status": "C3_MODE11_STATE_DIAGNOSTIC_STOPPED_AT_GATE",
            "matrix": matrix,
            "latency_values_collected": False,
            "timer_calls": 0,
            "candidate_reselected": False,
            "debug_executor_used": False,
        })
        write_json(output / "first_difference.json", differences)
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
        write_json(output / "per_session_bitstream_resets.json", {
            "clean_sessions_reset_before_rpc": True,
            "state_sequences_reset_only_at_sequence_start": True,
            "board_os_rebooted": False,
            "sessions": diagnostic.bitstream_resets,
        })
        write_json(output / "artifact_hashes.json", {"artifacts": {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "artifact_hashes.json"
        }})


def parse_args():
    args = base.parse_args()
    if Path(args.output_dir).resolve() == Path(base.DEFAULT_OUTPUT).resolve():
        args.output_dir = str(DEFAULT_OUTPUT)
    return args


if __name__ == "__main__":
    run(parse_args())
