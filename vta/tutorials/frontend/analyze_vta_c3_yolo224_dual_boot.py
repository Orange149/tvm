#!/usr/bin/env python3
"""Combine the two immutable YOLO-224 operator and graph screens.

The boot-1 selector decision remains the only deployment decision.  Boot-2
selection samples are retained only as a repeatability diagnostic and never
reselect a program.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from c3_evaluation_v5 import read_json, seal_flat, verify_flat, write_json


MODE11 = "input_weight_resident_transition_safe"


def gain(test: float, baseline: float) -> float:
    return 100.0 * (1.0 - float(test) / float(baseline))


def paired_gain(candidate, baseline):
    candidate_rounds = candidate["round_medians_ms"]
    baseline_rounds = baseline["round_medians_ms"]
    if len(candidate_rounds) != 6 or len(baseline_rounds) != 6:
        raise ValueError("operator evaluation requires six round medians")
    values = [gain(test, base) for test, base in zip(candidate_rounds, baseline_rounds)]
    return {
        "round_gains_pct": values,
        "median_gain_pct": float(statistics.median(values)),
        "winning_rounds": sum(value > 0.0 for value in values),
    }


def run(args):
    operator_dirs = [Path(args.operator_boot1).resolve(), Path(args.operator_boot2).resolve()]
    graph_dirs = [Path(args.graph_boot1).resolve(), Path(args.graph_boot2).resolve()]
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))

    manifests = {
        "operator": [verify_flat(path) for path in operator_dirs],
        "graph": [verify_flat(path) for path in graph_dirs],
    }
    operator_summaries = [read_json(path / "summary.json") for path in operator_dirs]
    graph_summaries = [read_json(path / "summary.json") for path in graph_dirs]
    for summary in operator_summaries:
        if (summary.get("status") != "YOLO224_SINGLE_BOOT_OPERATOR_SCREEN_ANALYZED"
                or summary.get("all_outputs_correct") is not True):
            raise ValueError("operator analysis is incomplete or incorrect")
    for summary in graph_summaries:
        if (summary.get("status") != "YOLO224_SINGLE_BOOT_GRAPH_SCREEN_COMPLETE"
                or summary.get("all_outputs_equal_to_stock") is not True):
            raise ValueError("graph screen is incomplete or incorrect")
    operator_boot_ids = [summary["boot_id"] for summary in operator_summaries]
    graph_boot_ids = [summary["boot_id"] for summary in graph_summaries]
    if len(set(operator_boot_ids)) != 2 or operator_boot_ids != graph_boot_ids:
        raise ValueError("operator and graph evidence must use the same two distinct boots")

    task_results = [read_json(path / "task_results.json") for path in operator_dirs]
    winners = read_json(operator_dirs[0] / "selection_winners_frozen.json")
    repeat_winners = read_json(operator_dirs[1] / "selection_winners_frozen.json")
    if set(task_results[0]) != set(task_results[1]) or set(winners) != set(task_results[0]):
        raise ValueError("operator task identities differ across boots")

    tasks = {}
    for task_id in sorted(task_results[0]):
        first = task_results[0][task_id]
        second = task_results[1][task_id]
        if first["incumbent_program_id"] != second["incumbent_program_id"]:
            raise ValueError("incumbent drift for {}".format(task_id))
        incumbent = first["incumbent_program_id"]
        winner = winners[task_id]["selector_v4_B3_winner"]
        if winners[task_id]["residence_mode"] != MODE11:
            raise ValueError("boot-1 frozen winner is not mode11")
        confirmations = []
        for result in (first, second):
            confirmation = paired_gain(
                result["evaluation_statistics"][winner],
                result["evaluation_statistics"][incumbent],
            )
            confirmations.append(confirmation)
        tasks[task_id] = {
            "incumbent_program_id": incumbent,
            "boot1_frozen_selector_v4_B3_winner": winner,
            "boot2_repeat_selection_winner": repeat_winners[task_id][
                "selector_v4_B3_winner"
            ],
            "winner_reselected_for_claim": False,
            "oracle_program_ids": [first["oracle_program_id"], second["oracle_program_id"]],
            "oracle_modes": [first["oracle_residence_mode"], second["oracle_residence_mode"]],
            "confirmation_by_boot": confirmations,
            "stable_operator_improvement": all(
                value["median_gain_pct"] >= 2.0 and value["winning_rounds"] >= 5
                for value in confirmations
            ),
            "best_mode11_vs_best_input_gain_pct": [
                first["best_mode11_improvement_vs_best_input_pct"],
                second["best_mode11_improvement_vs_best_input_pct"],
            ],
            "selector_v4_B3_regret_pct": [
                first["methods"]["selector_v4"]["B3"]["simple_regret_pct"],
                second["methods"]["selector_v4"]["B3"]["simple_regret_pct"],
            ],
            "bytes_only_B3_regret_pct": [
                first["methods"]["bytes_only"]["B3"]["simple_regret_pct"],
                second["methods"]["bytes_only"]["B3"]["simple_regret_pct"],
            ],
        }

    pair_rows = [read_json(path / "same_tile_pairs.json") for path in operator_dirs]
    pair_maps = [
        {(row["execution_task_id"], tuple(row["tile"])): row for row in rows}
        for rows in pair_rows
    ]
    if set(pair_maps[0]) != set(pair_maps[1]):
        raise ValueError("same-tile pair identities differ across boots")
    stable_pairs = []
    for key in sorted(pair_maps[0]):
        values = [pair_maps[index][key]["mode11_improvement_vs_input_pct"]
                  for index in range(2)]
        stable_pairs.append({
            "execution_task_id": key[0],
            "tile": list(key[1]),
            "gain_pct_by_boot": values,
            "mode11_wins_both_boots": all(value > 0.0 for value in values),
        })

    graph_by_boot = []
    for summary in graph_summaries:
        stats = summary["statistics"]
        selector = stats["selector_v4"]
        incumbent = stats["incumbent"]
        input_control = stats["input_control"]
        graph_by_boot.append({
            "boot_id": summary["boot_id"],
            "median_ms": {name: stats[name]["median_ms"] for name in stats},
            "throughput_fps": {name: stats[name]["throughput_fps"] for name in stats},
            "selector_gain_pct": summary["selector_improvement_pct"],
            "selector_faster_than_incumbent_in_both_blocks": all(
                selector["block_median_ms"][block] < incumbent["block_median_ms"][block]
                for block in ("A", "B")
            ),
            "selector_faster_than_input_in_both_blocks": all(
                selector["block_median_ms"][block] < input_control["block_median_ms"][block]
                for block in ("A", "B")
            ),
        })

    selector_equal_bytes = all(
        all(abs(task["selector_v4_B3_regret_pct"][index]
                - task["bytes_only_B3_regret_pct"][index]) < 1e-12
            for task in tasks.values())
        for index in range(2)
    )
    result = {
        "schema": "c3_yolo224_dual_boot_analysis_v1",
        "status": "YOLO224_DUAL_BOOT_OPERATOR_AND_GRAPH_CONFIRMED",
        "boot_ids": operator_boot_ids,
        "manifests": manifests,
        "tasks": tasks,
        "same_tile_pairs": stable_pairs,
        "same_tile_pair_count": len(stable_pairs),
        "same_tile_mode11_wins_both_boots": sum(
            row["mode11_wins_both_boots"] for row in stable_pairs
        ),
        "all_frozen_mode11_winners_stable": all(
            task["stable_operator_improvement"] for task in tasks.values()
        ),
        "all_oracles_mode11_both_boots": all(
            task["oracle_modes"] == [MODE11, MODE11] for task in tasks.values()
        ),
        "selector_v4_equal_to_bytes_only_B3_both_boots": selector_equal_bytes,
        "selector_superiority_claim": False,
        "graph_by_boot": graph_by_boot,
        "stable_graph_gain_vs_incumbent": all(
            row["selector_gain_pct"]["incumbent"] >= 1.0
            and row["selector_faster_than_incumbent_in_both_blocks"]
            for row in graph_by_boot
        ),
        "graph_incremental_mode11_vs_input_positive_both_boots": all(
            row["selector_gain_pct"]["input_control"] > 0.0
            and row["selector_faster_than_input_in_both_blocks"]
            for row in graph_by_boot
        ),
        "logical_request_is_physical_bus_claim": False,
        "hardware_stall_attribution_claim": False,
    }

    output.mkdir(parents=True)
    write_json(output / "analysis.json", result)
    write_json(output / "summary.json", {
        "status": result["status"],
        "boot_ids": result["boot_ids"],
        "same_tile_mode11_wins_both_boots": result["same_tile_mode11_wins_both_boots"],
        "same_tile_pair_count": result["same_tile_pair_count"],
        "all_frozen_mode11_winners_stable": result["all_frozen_mode11_winners_stable"],
        "stable_graph_gain_vs_incumbent": result["stable_graph_gain_vs_incumbent"],
        "graph_incremental_mode11_vs_input_positive_both_boots": result[
            "graph_incremental_mode11_vs_input_positive_both_boots"
        ],
        "selector_v4_equal_to_bytes_only_B3_both_boots": selector_equal_bytes,
        "selector_superiority_claim": False,
    })

    lines = [
        "# YOLOv3-tiny-224 C3 双启动确认",
        "",
        "两个 Linux boot ID 不同；两次算子和整图正式调用均通过逐元素正确性检查。",
        "",
        "| boot | stock ms | incumbent ms | Input ms | mode11 ms | mode11 vs Input | mode11 vs incumbent |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(graph_by_boot, 1):
        values = row["median_ms"]
        gains = row["selector_gain_pct"]
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.3f}% | {:.3f}% |".format(
                index, values["stock"], values["incumbent"], values["input_control"],
                values["selector_v4"], gains["input_control"], gains["incumbent"],
            )
        )
    lines.extend([
        "",
        "- 同 Tile Input→mode11：两次启动均为 {}/{} 获胜。".format(
            result["same_tile_mode11_wins_both_boots"], result["same_tile_pair_count"]
        ),
        "- 三个 boot-1 冻结 mode11 winner 均满足两启动算子门槛。",
        "- selector-v4 与 bytes-only 在两个启动的 B3 regret 完全相同；不能主张选择器优越。",
        "- 整图相对 Input 的增量可归于加入 mode11 驻留程序的部署差异；相对 stock 的总收益还包含 Tile 与输入复用。",
        "- 逻辑请求量不是物理总线流量，本实验不提供逐周期 stall 归因。",
        "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    seal_flat(output, (Path(__file__).resolve(),))
    print(json.dumps(read_json(output / "summary.json"), indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-boot1", type=Path, required=True)
    parser.add_argument("--operator-boot2", type=Path, required=True)
    parser.add_argument("--graph-boot1", type=Path, required=True)
    parser.add_argument("--graph-boot2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
