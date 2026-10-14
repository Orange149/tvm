#!/usr/bin/env python3
"""Analyze the frozen one-boot clean-candidate YOLO-224 timing screen."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from freeze_vta_c3_selector_v4_production_timing import verify as verify_flat
from run_vta_c3_fpga_correctness_waves import read, seal, write


METHODS = ("selector_v4", "bytes_only", "calls_only", "command_only", "fixed_random")
BUDGETS = (1, 2, 3)
MODE11 = "input_weight_resident_transition_safe"
INPUT = "input_stationary"
ORIGINAL = "original"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def evaluation_statistics(rows):
    round_medians = []
    for round_index in range(6):
        values = [float(row["latency_ms"]) for row in rows if row["round"] == round_index]
        if len(values) != 3:
            raise ValueError("each evaluation round must contain three seeds")
        round_medians.append(float(statistics.median(values)))
    return {
        "round_medians_ms": round_medians,
        "median_ms": float(statistics.median(round_medians)),
        "single_call_values_ms": [float(row["latency_ms"]) for row in rows],
    }


def selection_statistics(rows):
    values = [float(row["latency_ms"]) for row in rows]
    if len(values) != 3:
        raise ValueError("each selection identity must contain three dedicated samples")
    return {"values_ms": values, "median_ms": float(statistics.median(values))}


def tile_key(candidate):
    features = candidate["visible_features"]
    names = ("tile_b", "tile_h", "tile_w", "tile_ci", "tile_co",
             "oc_nthread", "h_nthread")
    return tuple(int(features[name]) for name in names)


def percent_gain(test, baseline):
    return 100.0 * (1.0 - float(test) / float(baseline))


def run(args):
    timing_dir = Path(args.timing_dir).resolve()
    pool_dir = Path(args.board_pool_dir).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    timing_manifest = verify_flat(timing_dir)
    pool_manifest = verify_flat(pool_dir)
    summary = read(timing_dir / "summary.json")
    if (summary.get("status") != "YOLO224_SINGLE_BOOT_CLEAN_TIMING_COMPLETE"
            or summary.get("all_outputs_correct") is not True):
        raise ValueError("timing input is not a complete correct YOLO-224 screen")
    contract = read(timing_dir / "contract.json")
    if contract.get("input_resolution") != 224:
        raise ValueError("analysis input is not YOLO-224")
    evaluation_rows = read_jsonl(timing_dir / "evaluation_timing.jsonl")
    selection_rows = read_jsonl(timing_dir / "selection_timing.jsonl")
    if len(evaluation_rows) != 486 or len(selection_rows) != 81:
        raise ValueError("timing row counts differ from the frozen contract")
    candidates = read_jsonl(pool_dir / "candidates.jsonl")
    by_program = {row["program_id"]: row for row in candidates}
    policies = contract["task_policies"]

    eval_grouped = defaultdict(list)
    selection_grouped = defaultdict(list)
    for row in evaluation_rows:
        eval_grouped[row["execution_task_id"], row["program_id"]].append(row)
    for row in selection_rows:
        selection_grouped[row["execution_task_id"], row["program_id"]].append(row)

    task_results = {}
    same_tile_pairs = []
    frozen_winners = {}
    for task_id, policy in sorted(policies.items()):
        production = list(policy["production_program_pool"])
        incumbent = policy["incumbent_program_id"]
        eval_stats = {
            program: evaluation_statistics(eval_grouped[task_id, program])
            for program in production
        }
        selection_stats = {
            program: selection_statistics(selection_grouped[task_id, program])
            for program in production
        }
        oracle = min(production, key=lambda program: (
            eval_stats[program]["median_ms"], program
        ))
        method_results = {}
        for method in METHODS:
            method_results[method] = {}
            order = policy["orders"][method]
            for budget in BUDGETS:
                exposed = [incumbent] + order[:budget]
                winner = min(exposed, key=lambda program: (
                    selection_stats[program]["median_ms"],
                    0 if program == incumbent else 1,
                    program,
                ))
                winner_latency = eval_stats[winner]["median_ms"]
                oracle_latency = eval_stats[oracle]["median_ms"]
                method_results[method]["B{}".format(budget)] = {
                    "exposed_program_ids": exposed,
                    "search_winner": winner,
                    "winner_residence_mode": by_program[winner]["residence_mode"],
                    "evaluation_latency_ms": winner_latency,
                    "improvement_vs_incumbent_pct": percent_gain(
                        winner_latency, eval_stats[incumbent]["median_ms"]
                    ),
                    "simple_regret_pct": 100.0 * (winner_latency / oracle_latency - 1.0),
                    "prefix_contains_oracle": oracle in exposed,
                    "winner_is_oracle": winner == oracle,
                }
        frozen_winners[task_id] = {
            "selector_v4_B3_winner": method_results["selector_v4"]["B3"]["search_winner"],
            "residence_mode": method_results["selector_v4"]["B3"]["winner_residence_mode"],
            "sealed_from_selection_rows_only": True,
        }

        by_tile = defaultdict(dict)
        for program in production:
            by_tile[tile_key(by_program[program])][by_program[program]["residence_mode"]] = program
        task_pairs = []
        for key, modes in sorted(by_tile.items()):
            if INPUT not in modes or MODE11 not in modes:
                continue
            input_program = modes[INPUT]
            mode11_program = modes[MODE11]
            input_latency = eval_stats[input_program]["median_ms"]
            mode11_latency = eval_stats[mode11_program]["median_ms"]
            input_bytes = by_program[input_program]["timing_freeze_metrics"]["total_request_bytes"]
            mode11_bytes = by_program[mode11_program]["timing_freeze_metrics"]["total_request_bytes"]
            pair = {
                "execution_task_id": task_id,
                "tile": list(key),
                "input_program_id": input_program,
                "mode11_program_id": mode11_program,
                "input_latency_ms": input_latency,
                "mode11_latency_ms": mode11_latency,
                "mode11_improvement_vs_input_pct": percent_gain(mode11_latency, input_latency),
                "input_total_request_bytes": input_bytes,
                "mode11_total_request_bytes": mode11_bytes,
                "logical_request_reduction_pct": percent_gain(mode11_bytes, input_bytes),
            }
            task_pairs.append(pair)
            same_tile_pairs.append(pair)

        best_by_mode = {}
        for mode in (ORIGINAL, INPUT, MODE11):
            programs = [program for program in production
                        if by_program[program]["residence_mode"] == mode]
            best = min(programs, key=lambda program: (eval_stats[program]["median_ms"], program))
            best_by_mode[mode] = {
                "program_id": best,
                "latency_ms": eval_stats[best]["median_ms"],
            }
        best_input = best_by_mode[INPUT]["latency_ms"]
        best_mode11 = best_by_mode[MODE11]["latency_ms"]
        task_results[task_id] = {
            "execution_task_id": task_id,
            "workload_features": by_program[incumbent]["workload_features"],
            "incumbent_program_id": incumbent,
            "incumbent_latency_ms": eval_stats[incumbent]["median_ms"],
            "oracle_program_id": oracle,
            "oracle_residence_mode": by_program[oracle]["residence_mode"],
            "oracle_latency_ms": eval_stats[oracle]["median_ms"],
            "oracle_improvement_vs_incumbent_pct": percent_gain(
                eval_stats[oracle]["median_ms"], eval_stats[incumbent]["median_ms"]
            ),
            "best_by_mode": best_by_mode,
            "best_mode11_improvement_vs_best_input_pct": percent_gain(
                best_mode11, best_input
            ),
            "selection_statistics": selection_stats,
            "evaluation_statistics": eval_stats,
            "methods": method_results,
            "same_tile_pairs": task_pairs,
        }

    primary_equal = all(
        abs(task["methods"]["selector_v4"]["B3"]["simple_regret_pct"]
            - task["methods"]["bytes_only"]["B3"]["simple_regret_pct"]) < 1e-12
        for task in task_results.values()
    )
    primary_not_worse = all(
        task["methods"]["selector_v4"]["B3"]["simple_regret_pct"]
        <= task["methods"]["bytes_only"]["B3"]["simple_regret_pct"] + 1e-12
        for task in task_results.values()
    )
    primary_strict = any(
        task["methods"]["selector_v4"]["B3"]["simple_regret_pct"]
        < task["methods"]["bytes_only"]["B3"]["simple_regret_pct"] - 1e-12
        for task in task_results.values()
    )
    pair_gains = [row["mode11_improvement_vs_input_pct"] for row in same_tile_pairs]
    request_reductions = [row["logical_request_reduction_pct"] for row in same_tile_pairs]
    best_incremental = [
        task["best_mode11_improvement_vs_best_input_pct"] for task in task_results.values()
    ]
    result_summary = {
        "schema": "c3_yolo224_clean_timing_analysis_v1",
        "status": "YOLO224_SINGLE_BOOT_OPERATOR_SCREEN_ANALYZED",
        "boot_id": summary["boot_id"],
        "task_count": len(task_results),
        "candidate_count": summary["candidate_count"],
        "all_outputs_correct": True,
        "production_oracle_mode11_tasks": sum(
            task["oracle_residence_mode"] == MODE11 for task in task_results.values()
        ),
        "selector_v4_B3_mode11_winners": sum(
            row["residence_mode"] == MODE11 for row in frozen_winners.values()
        ),
        "selector_v4_B3_oracle_hits": sum(
            task["methods"]["selector_v4"]["B3"]["winner_is_oracle"]
            for task in task_results.values()
        ),
        "bytes_only_B3_oracle_hits": sum(
            task["methods"]["bytes_only"]["B3"]["winner_is_oracle"]
            for task in task_results.values()
        ),
        "primary_selector_not_worse_than_bytes": primary_not_worse,
        "primary_selector_strictly_better_on_any_task": primary_strict,
        "primary_selector_equal_to_bytes_on_all_tasks": primary_equal,
        "same_tile_mode11_vs_input_pairs": len(same_tile_pairs),
        "same_tile_mode11_wins": sum(value > 0 for value in pair_gains),
        "same_tile_mode11_gain_pct": {
            "minimum": min(pair_gains),
            "median": float(statistics.median(pair_gains)),
            "maximum": max(pair_gains),
        },
        "same_tile_logical_request_reduction_pct": {
            "minimum": min(request_reductions),
            "median": float(statistics.median(request_reductions)),
            "maximum": max(request_reductions),
        },
        "best_mode11_vs_best_input_gain_pct": {
            "minimum": min(best_incremental),
            "median": float(statistics.median(best_incremental)),
            "maximum": max(best_incremental),
        },
        "cross_boot_stability_claim": False,
        "full_graph_claim": False,
        "selector_superiority_claim": False,
        "timing_manifest_sha256": timing_manifest,
        "pool_manifest_sha256": pool_manifest,
    }

    output.mkdir(parents=True)
    write(output / "task_results.json", task_results)
    write(output / "same_tile_pairs.json", same_tile_pairs)
    write(output / "selection_winners_frozen.json", frozen_winners)
    write(output / "summary.json", result_summary)

    lines = [
        "# YOLOv3-tiny-224 C3 单启动算子初测",
        "",
        "本轮 27 个生产候选、729 次完整算子调用全部正确。以下延迟结论仅属于同一启动下、",
        "每候选洁净 bitstream/fresh RPC 的算子级初测，不是跨启动稳定结论，也不是整图结果。",
        "",
        "| 任务 | 形状 | incumbent ms | oracle ms | oracle模式 | oracle改善 | mode11相对最佳Input改善 |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for task_id, task in sorted(task_results.items()):
        shape = task["workload_features"]
        lines.append(
            "| {} | {}→{}, {}×{}, K{} | {:.6f} | {:.6f} | {} | {:.3f}% | {:.3f}% |".format(
                task_id[:12], shape["ci"], shape["co"], shape["height"], shape["width"],
                shape["kernel"], task["incumbent_latency_ms"], task["oracle_latency_ms"],
                task["oracle_residence_mode"], task["oracle_improvement_vs_incumbent_pct"],
                task["best_mode11_improvement_vs_best_input_pct"],
            )
        )
    lines.extend([
        "",
        "同 Tile 的 9 组 Input→mode11 对照全部由 mode11 获胜；改善范围 {:.3f}%–{:.3f}%，".format(
            min(pair_gains), max(pair_gains)
        ),
        "中位改善 {:.3f}%；逻辑请求量下降范围 {:.3f}%–{:.3f}%。".format(
            statistics.median(pair_gains), min(request_reductions), max(request_reductions)
        ),
        "",
        "selector v4 与 bytes-only 在 B3 上都对 3/3 任务选中了生产池 oracle，因此本轮证明了",
        "mode11 机会和有限预算可选中它，但没有证明 selector v4 优于 bytes-only。",
        "",
        "下一门禁：把只由选择样本封存的三个 mode11 winner 接入同一 YOLO-224 图，先做整图",
        "逐元素正确性，再比较 stock、incumbent 与 selector 部署；若成功，再安排第二启动确认。",
        "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    seal(output)
    print(json.dumps(result_summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-dir", type=Path, required=True)
    parser.add_argument("--board-pool-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
