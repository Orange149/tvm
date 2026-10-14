#!/usr/bin/env python3
"""No-timing cross-program FPGA-state diagnostic for the frozen C3 mode11 graph."""

from __future__ import annotations

import json
from pathlib import Path
import traceback

import run_vta_c3_mode11_graph_context_diagnostic as base
import run_vta_c3_reuse_board as reuse_board
from c3_mode11_graph_context import sha256_file, write_json
from c3_thread_sync_board import ExperimentBoard
from run_vta_c3_yolo608_graph_boot import random_input
from run_vta_c3_mode11_ordinary_prefix_tap import (
    FreshBitstreamDiagnostic,
    HISTORICAL_SEEDS,
    run_state_sequence,
)


GROUPED = base.GROUPED
DEFAULT_OUTPUT = GROUPED / "20260926_c3_mode11_cross_program_state_run06"


def run(args):
    main = Path(args.main_build).resolve()
    diagnostic_build = Path(args.diagnostic_build).resolve()
    roots = {
        "stock": main / "stock",
        "incumbent": main / "incumbent",
        "mode11": diagnostic_build / "mode11_only_eec24a99",
        "other_mode11": diagnostic_build / "mode11_only_2fa33cab",
    }
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)

    freeze = base.read(Path(args.freeze_dir) / "contract.json")
    runtime = base.read(freeze["runtime_contract"])
    for name, path in runtime["runtime_paths"].items():
        if sha256_file(path) != runtime["runtime_sha256"][name]:
            raise ValueError("runtime changed: " + name)

    frozen_pairs = [
        (predecessor, target)
        for predecessor in ("stock", "incumbent", "other_mode11")
        for target in ("stock", "incumbent", "mode11")
    ]
    write_json(output / "preregistered.json", {
        "schema": "c3_mode11_cross_program_state_v1",
        "status": "FROZEN_BEFORE_BOARD_DIAGNOSTIC",
        "parent": str(output.parent / "20260926_c3_mode11_graph_context_board_v63_state_run05"),
        "seed_order": list(HISTORICAL_SEEDS),
        "predecessor_target_pairs": [list(pair) for pair in frozen_pairs],
        "fresh_rpc_between_predecessor_and_target": True,
        "same_bitstream_preserved_within_pair": True,
        "bitstream_reloaded_before_each_pair": True,
        "timer_calls": 0,
        "candidate_reselection": False,
        "candidate_modification": False,
    })

    board = ExperimentBoard(args, "/var/volatile/" + output.name, args.port)
    diagnostic = FreshBitstreamDiagnostic(args, output, board)
    matrix = {}
    initial = None
    try:
        initial = board.state()
        reuse_board.validate_state(initial, exclusive=True)
        reuse_board.RUNTIME = runtime["runtime_sha256"]
        write_json(output / "runtime_attestation.json", board.deploy(runtime["runtime_paths"]))
        write_json(output / "initial_bitstream_attestation.json", board.reload())

        data = random_input(base.SEED, resolution=512)
        clean = {}
        clean_profiles = {}
        for target in ("stock", "incumbent", "mode11"):
            clean[target], clean_profiles[target] = diagnostic.full_graph(
                "clean_" + target, roots[target], data
            )
        clean_differences = {
            target: base.output_differences(clean["stock"], clean[target])
            for target in ("incumbent", "mode11")
        }
        if any(clean_differences["incumbent"]) or any(clean_differences["mode11"]):
            raise RuntimeError("CLEAN_TARGET_REFERENCE_INVALID")

        rows = []
        for predecessor, target in frozen_pairs:
            row = run_state_sequence(
                diagnostic,
                "pair_" + predecessor + "_to_" + target,
                [(predecessor, roots[predecessor], HISTORICAL_SEEDS)],
                roots[target],
                clean["stock"],
            )
            row["predecessor"] = predecessor
            row["target"] = target
            rows.append(row)

        failing = [
            {"predecessor": row["predecessor"], "target": row["target"]}
            for row in rows if not row["target_correct_vs_clean_stock"]
        ]
        by_predecessor = {
            predecessor: {
                target: next(
                    row["target_correct_vs_clean_stock"] for row in rows
                    if row["predecessor"] == predecessor and row["target"] == target
                )
                for target in ("stock", "incumbent", "mode11")
            }
            for predecessor in ("stock", "incumbent", "other_mode11")
        }
        general_state_failure = any(
            not by_predecessor[predecessor]["stock"]
            or not by_predecessor[predecessor]["incumbent"]
            for predecessor in ("incumbent", "other_mode11")
        )
        mode11_specific = (
            not general_state_failure
            and any(
                not by_predecessor[predecessor]["mode11"]
                for predecessor in ("incumbent", "other_mode11")
            )
        )
        classification = (
            "GENERAL_CROSS_RPC_FPGA_STATE_CONTAMINATION"
            if general_state_failure
            else "MODE11_SPECIFIC_PREDECESSOR_SENSITIVITY"
            if mode11_specific
            else "NO_CROSS_PROGRAM_FAILURE_REPRODUCED"
        )
        matrix = {
            "schema": "c3_mode11_cross_program_state_v1",
            "status": "C3_MODE11_CROSS_PROGRAM_STATE_DIAGNOSTIC_COMPLETE",
            "classification": classification,
            "clean_targets_equal": True,
            "clean_differences": clean_differences,
            "clean_profiles": clean_profiles,
            "pairs": rows,
            "by_predecessor": by_predecessor,
            "failing_pairs": failing,
            "latency_values_collected": False,
            "timer_calls": 0,
        }
        write_json(output / "replay_matrix.json", matrix)
        after = board.state()
        reuse_board.validate_state(after, exclusive=True)
        if after["boot_id"] != initial["boot_id"]:
            raise RuntimeError("board boot changed during diagnostic")
        write_json(output / "summary.json", {
            "status": matrix["status"],
            "classification": classification,
            "by_predecessor": by_predecessor,
            "failing_pairs": failing,
            "boot_id": initial["boot_id"],
            "latency_values_collected": False,
            "timer_calls": 0,
        })
        lines = [
            "# C3 mode11 跨程序 FPGA 状态诊断", "",
            "- 分类：`{}`。".format(classification),
            "- 干净状态 stock/incumbent/mode11 输出一致：`True`。",
            "- 失败前驱—目标对：`{}`。".format(
                json.dumps(failing, ensure_ascii=False, sort_keys=True)
            ),
            "- 未采集 latency，未修改或重选候选。",
        ]
        (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except BaseException as error:
        write_json(output / "invalid_or_stopped.json", {
            "message": str(error),
            "type": type(error).__name__,
            "traceback": traceback.format_exc()[-16000:],
            "partial_matrix": matrix,
            "latency_values_collected": False,
            "timer_calls": 0,
        })
        raise
    finally:
        try:
            board.stop()
        except Exception:
            pass
        write_json(output / "bitstream_resets.json", {
            "board_os_rebooted": False,
            "clean_sessions_reset_before_rpc": True,
            "pairs_reset_only_at_pair_start": True,
            "sessions": diagnostic.bitstream_resets,
        })
        write_json(output / "artifact_hashes.json", {"artifacts": {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "artifact_hashes.json"
        }})


def parse_args():
    args = base.parse_args()
    args.output_dir = str(DEFAULT_OUTPUT)
    return args


if __name__ == "__main__":
    run(parse_args())
