#!/usr/bin/env python3
"""Six-round fixed-pool analysis. Incomplete/failed attempts yield no result."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from statistics import median


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(root):
    if list(root.rglob("invalid_attempt.json")):
        raise ValueError("failed board attempt: no complete performance result")
    protocol, prepared = read(root / "board_protocol.json"), read(root / "board_prepared.json")
    from c3_valid_window_board import SCHEMA, execution_sources
    if execution_sources() != protocol["source_sha256"] or sha(protocol["document"]) != protocol["document_sha256"]:
        raise ValueError("frozen execution identity changed")
    boot = protocol["required_boot_id"]
    for phase, status in (("qualification", "FIXED_POOL_FPGA_QUALIFIED"),
                          ("diagnostic", "ALL_ROUND_PREFIX_AND_UOP_DIAGNOSTICS_QUALIFIED"),
                          ("board", "FORMAL_FIXED_POOL_ACQUISITION_COMPLETE")):
        summary = read(root / phase / "summary.json")
        if summary["schema"] != SCHEMA or summary["status"] != status or summary["boot_id"] != boot or not summary["all_correct"]:
            raise ValueError("missing/incomplete phase: " + phase)
        if summary["prepared_sha256"] != sha(root / "board_prepared.json") or summary["protocol_sha256"] != sha(root / "board_protocol.json"):
            raise ValueError("phase frozen identity changed")
    calls, sessions = lines(root / "board/calls.jsonl"), lines(root / "board/sessions.jsonl")
    count = protocol["expected_calls"]["formal_timing"]
    if Counter(row["call_kind"] for row in calls) != {"warmup": count, "timing": count}:
        raise ValueError("formal call ledger differs")
    identifiers = [row["invocation_id"] for row in calls]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate invocation identities")
    for row in calls:
        if row["status"] != "PASS" or row["boot_id"] != boot or not row["correct"] or not row["arguments_unchanged"] or row["complete_operator_calls"] != 1 or row["hidden_warmup_calls"]:
            raise ValueError("formal output/call/status gate failed")
        if row["call_kind"] == "timing" and (not row["diagnostic_signature_matched"] or not math.isfinite(row["latency_ms"]) or row["latency_ms"] <= 0):
            raise ValueError("timing invocation is invalid")
    dispatch = lines(root / "board/dispatch.jsonl")
    if len(dispatch) != 2 * len(calls) or [(row["invocation_id"], row["event"]) for row in dispatch] != [(iid, event) for iid in identifiers for event in ("begin", "end")]:
        raise ValueError("formal invocation dispatch is incomplete or unexpected")
    helper = lines(root / "board/helper_counts.jsonl")
    timings = [row for row in calls if row["call_kind"] == "timing"]
    if [(row["invocation_id"], row["count"]) for row in helper] != [(row["invocation_id"], row["seed_position"] + 1) for row in timings]:
        raise ValueError("timer's one-target-call accounting failed")
    expected_cells = [(r + 1, name) for r, order in enumerate(protocol["orders"]) for name in order]
    if len(sessions) != len(expected_cells) or [row["program"] for row in sessions] != [name for _, name in expected_cells]:
        raise ValueError("fresh-session order differs")
    medians = {name: [] for name in protocol["programs"]}
    cursor = 0
    for (round_id, name), session in zip(expected_cells, sessions):
        if session["status"] != "PASS" or session["environment"]["boot_id"] != boot or any(mask != "3" for mask in session["environment"]["masks"]) or session["runtime_attestation"] != protocol["runtime_sha256"]:
            raise ValueError("session runtime/CPU3/boot identity failed")
        if session["start_state"]["udmabuf_size"] != 201326592 or session["bitstream_load_record"]["sha256"] != "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6" or session["inventory"]["replay_initial"]["mode"] != "disabled":
            raise ValueError("session DMA/bitstream/replay failed")
        cell = calls[cursor:cursor + 6]
        cursor += 6
        seeds = protocol["seed_orders"][round_id - 1]
        if [row["call_kind"] for row in cell] != ["warmup"] * 3 + ["timing"] * 3 or [row["seed"] for row in cell] != seeds * 2:
            raise ValueError("explicit prefix/seed order differs")
        if any(row["program"] != name or row["round"] != round_id or row["session_id"] != session["session_id"] or row["buffer_handles"] != cell[0]["buffer_handles"] for row in cell):
            raise ValueError("cell program/session/parameter handles changed")
        medians[name].append(median(row["latency_ms"] for row in cell[3:]))
    if cursor != len(calls):
        raise ValueError("unexpected additional target calls")
    totals = {name: median(values) for name, values in medians.items()}

    def pair(candidate, baseline):
        improvements = [100 * (1 - c / b) for c, b in zip(medians[candidate], medians[baseline])]
        differences = [b - c for c, b in zip(medians[candidate], medians[baseline])]
        return {"candidate": candidate, "baseline": baseline, "paired_improvements_pct": improvements,
                "G_pct": median(improvements), "wins": sum(c < b for c, b in zip(medians[candidate], medians[baseline])),
                "ties": sum(c == b for c, b in zip(medians[candidate], medians[baseline])),
                "paired_ms_saved": differences, "median_paired_ms_saved": median(differences)}
    aliases = protocol["aliases"]
    primary = pair(aliases["Q8"], aliases["Q1"])
    q_programs = [name for name in protocol["programs"] if name != "TPS"]
    candidate = min(q_programs, key=lambda name: (totals[name], name != aliases["Q1"], name))
    comparisons = {name: {"versus_Q1": pair(name, aliases["Q1"]), "versus_TPS": pair(name, "TPS")} for name in q_programs}
    research = primary["G_pct"] >= 2 and primary["wins"] >= 5
    practical = comparisons[candidate]["versus_Q1"]
    tps = comparisons[candidate]["versus_TPS"]
    deployment = practical["G_pct"] >= 2 and practical["wins"] >= 5 and tps["G_pct"] >= 1 and tps["wins"] >= 5
    result = {"schema": "c3_valid_window_six_round_analysis_v1", "status": "COMPLETE_SINGLE_BOOT_FIXED_POOL_RESULT",
        "boot_id": boot, "round_medians_ms": medians, "T_ms": totals,
        "round_range_ms": {name: [min(values), max(values)] for name, values in medians.items()},
        "primary_Q8_versus_Q1": primary, "comparisons": comparisons,
        "Q_pool_minimum_T_program": candidate, "research_replication_engineering_gate": research,
        "practical_graph_engineering_gate": deployment, "formal_timings": count, "formal_warmups": count,
        "statistical_significance_claimed": False, "graph_performance_measured": False,
        "next_boot_or_graph_executed": False, "pause_after_current_round": True,
        "inputs": {str(root / name): sha(root / name) for name in (
            "board_protocol.json", "board_prepared.json", "board/calls.jsonl", "board/sessions.jsonl",
            "board/helper_counts.jsonl", "qualification/summary.json", "diagnostic/summary.json", "board/summary.json")}}
    with (root / "analysis.json").open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"T_ms": totals, "primary": primary,
        "research_gate": research, "practical_gate": deployment}, sort_keys=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    analyze(parser.parse_args().root.resolve())
