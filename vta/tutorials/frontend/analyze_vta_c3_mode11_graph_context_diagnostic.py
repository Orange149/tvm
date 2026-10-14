#!/usr/bin/env python3
"""Analyze and report one C3 mode11 v6.1 no-timing correctness diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from c3_mode11_graph_context import classify_replay, sha256_file, write_json


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_CONTRACT = GROUPED / "20260926_c3_mode11_graph_context_contract_v61_run03"
DEFAULT_BOARD = GROUPED / "20260926_c3_mode11_graph_context_board_v61_run03"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(args):
    contract_dir, board_dir = Path(args.contract_dir), Path(args.board_dir)
    contract = read(contract_dir / "contract.json")
    replay = read(board_dir / "replay_matrix.json")
    calls = [
        json.loads(line)
        for line in (board_dir / "calls.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if contract.get("status") != "FULL_FUSION_CONTRACT_RESOLVED":
        raise ValueError("fusion contract is unresolved")
    if replay.get("latency_values_collected") is not False or replay.get("timer_calls") != 0:
        raise ValueError("v6.1 diagnostic unexpectedly contains timing")
    if any(row.get("latency_ms") is not None for row in calls):
        raise ValueError("a call record contains latency")
    forbidden = {"run_individual", "run_individual_node", "debug_get_output", "profile", "profile_rpc"}
    used = {str(row.get("api")) for row in calls}
    if forbidden & used:
        raise ValueError("forbidden debug API appears in calls: " + repr(sorted(forbidden & used)))
    earliest, flags = classify_replay(replay["matrix"])
    stored_earliest = replay["earliest_observed_divergence"]
    stored_flags = replay["observation_flags"]
    classification_corrected = (
        earliest != stored_earliest or sorted(flags) != sorted(stored_flags)
    )
    correction_allowed = (
        replay.get("status") == "C3_MODE11_GRAPH_CONTEXT_DIAGNOSTIC_STOPPED_AT_GATE"
        and replay["matrix"].get("S", {}).get("status")
        == "DEBUG_PREFIX_REFERENCE_INVALID"
    )
    if classification_corrected and not correction_allowed:
        raise ValueError("stored classification disagrees with deterministic analysis")
    graph_nodes = read(contract_dir / "incumbent" / "graph.json")["nodes"]
    semantic_operator_calls = 0
    null_node_api_calls = 0
    for row in calls:
        if row.get("api") == "execute_node":
            is_operator = graph_nodes[int(row["node"])]["op"] == "tvm_op"
            semantic_operator_calls += int(is_operator)
            null_node_api_calls += int(not is_operator)
        else:
            semantic_operator_calls += int(bool(row["executes_operator"]))
    b = replay["matrix"].get("B", {})
    stable = (
        b.get("mode11_matches_reference") is False
        and b.get("mode11_repeat_matches_reference") is False
        and b.get("mode11_repeat_same_wrong_output") is True
    )
    analysis = {
        "schema": "c3_mode11_graph_context_analysis_v61",
        "status": "C3_MODE11_GRAPH_CONTEXT_ANALYSIS_COMPLETE",
        "earliest_observed_divergence": earliest,
        "observation_flags": flags,
        "stable_minimal_reproduction": stable,
        "primary_B_question_answered": replay["matrix"].get("B") is not None,
        "classification_corrected_in_analysis": classification_corrected,
        "raw_stored_classification": {
            "earliest_observed_divergence": stored_earliest,
            "observation_flags": stored_flags,
        },
        "fusion_contract": contract["fusion_semantics"],
        "stages": replay["matrix"],
        "real_operator_invocations": semantic_operator_calls,
        "execute_node_api_calls": sum(row.get("api") == "execute_node" for row in calls),
        "null_node_api_calls": null_node_api_calls,
        "raw_call_ledger_executes_operator_count": sum(
            bool(row["executes_operator"]) for row in calls
        ),
        "call_ledger_semantics_corrected_in_analysis": null_node_api_calls > 0,
        "non_operator_observations": len(calls) - semantic_operator_calls,
        "latency_values_collected": False,
        "candidate_reselected": False,
    }
    write_json(board_dir / "analysis.json", analysis)
    lines = [
        "# C3 mode11 v6.1 正确性定位报告",
        "",
        "## 结论",
        "",
        "- 最早可靠差异边界：`{}`。".format(earliest),
        "- 主问题（mode11 四参数函数独立重放是否错误）：**尚未回答**；S 参考路径先失效，B 未执行。"
        if earliest == "DEBUG_PREFIX_REFERENCE_INVALID"
        else "- 主问题已进入 B 门禁。",
        "- 稳定最小复现：{}。".format("是" if stable else "否"),
        "- 观察标记：{}。".format(", ".join("`{}`".format(x) for x in flags) or "无"),
        "- 本轮未采集 latency、未重选候选、未修改二进制。",
        "",
        "## 已锁定的完整融合合同",
        "",
        "- ABI：`X, p27, p28, Y`。",
        "- 实际顺序：`conv(int32) + p28(int32) → arithmetic >> 6 → clip[-127,127] → int8`。",
        "- p28 广播：`p28[co_outer,0,0,0,co_inner]`。",
        "- 输出静态 STORE 覆盖：{} bytes，完整。".format(
            contract["fusion_semantics"]["output_store_coverage"]["bytes"]
        ),
        "",
        "## 阶段结果",
        "",
    ]
    for stage in ("R", "S", "B", "A", "C", "D"):
        lines.append("- **{}**：`{}`".format(
            stage, replay["matrix"].get(stage, {}).get("status", "not_recorded")
        ))
    lines += [
        "",
        "## 证据边界",
        "",
        "该结论只定位最早可复现边界，不把 p28 敏感性、UOP、同步或 SRAM 状态直接命名为唯一根因。",
        "",
        "普通 incumbent 图与 stock 完全一致，普通 singleton mode11 错误已复现；但逐节点 debug incumbent 在节点62已与普通 incumbent 图不同。因此 debug 路径生成的 Y61 不能作为 B 的参考。",
    ]
    (board_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(board_dir / "attempt_lineage.json", {
        "run01": "invalid: execute_node was incorrectly treated as execute-prefix",
        "run02": "superseded: full prefix fixed, but ordinary-vs-debug reference gate was not yet present",
        "run03": "authoritative stopped attempt: ordinary-vs-debug reference gate failed before B",
        "latency_values_collected": False,
        "candidate_reselected": False,
    })
    write_json(board_dir / "analysis_hashes.json", {
        "contract.json": sha256_file(contract_dir / "contract.json"),
        "replay_matrix.json": sha256_file(board_dir / "replay_matrix.json"),
        "calls.jsonl": sha256_file(board_dir / "calls.jsonl"),
        "analysis.json": sha256_file(board_dir / "analysis.json"),
        "REPORT.md": sha256_file(board_dir / "REPORT.md"),
        "attempt_lineage.json": sha256_file(board_dir / "attempt_lineage.json"),
    })
    print(json.dumps({
        "status": analysis["status"],
        "earliest_observed_divergence": earliest,
        "stable_minimal_reproduction": stable,
    }, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--board-dir", default=str(DEFAULT_BOARD))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

