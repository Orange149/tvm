#!/usr/bin/env python3
"""Independent confirmation evaluates sealed choices; never changes a winner."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

import c3_residency_tradeoff as p
from c3_five_tradeoff_policy import PROGRAMS, INCUMBENT, NEW, choose_visible
from c3_five_tradeoff_board import ORDERS


def journal(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def summarize(confirmation, sealed):
    rounds = p.round_medians(confirmation, programs=PROGRAMS)
    overall = {name: median(rounds[r][name] for r in range(6)) for name in PROGRAMS}
    oracle = min(PROGRAMS, key=lambda name: (overall[name], name))
    comparisons = {name: p.compare(rounds, NEW, name) for name in PROGRAMS if name != NEW}
    incumbent, practical = comparisons[INCUMBENT], comparisons["Input-7x7"]
    mechanism = comparisons["R12-G"]
    methods = {}
    for method, name in sealed["winners"].items():
        if name not in PROGRAMS:
            raise ValueError("frozen winner not in valid four-program confirmation pool")
        methods[method] = {"search_winner": name, "confirmation_ms": overall[name],
                           "simple_regret": overall[name] / overall[oracle] - 1,
                           "actual_winner_hits_oracle": name == oracle,
                           "effective_selection_calls": 3 * (1 + len(sealed["prefixes"][method]))}
    # No reference-pool reselection: even a fast NEW cannot enter graph if
    # the rule returned incumbent from its limited panel.
    eligible = (sealed["winners"]["rule"] == NEW and
                all(x["median_improvement_pct"] >= 2 and x["wins"] >= 5
                    for x in (incumbent, practical)))
    return {
        "status": "INDEPENDENT_CONFIRMATION_ANALYZED_NO_RESELECTION",
        "program_medians_ms": overall, "round_medians_ms": rounds,
        "comparisons": comparisons, "methods": methods,
        "oracle_scope": "four frozen FPGA-valid programs, one boot, confirmation only",
        "empirical_oracle": oracle,
        "prediction_new_faster_than_R12": {"frozen_direction": "faster",
            "observed_paired_median_improvement_pct": mechanism["median_improvement_pct"],
            "observed_wins": mechanism["wins"],
            "direction_correct": mechanism["median_improvement_pct"] > 0},
        "rule_vs_bytes": {
            "regret_difference": methods["rule"]["simple_regret"] - methods["bytes_only"]["simple_regret"],
            "same_effective_budget": methods["rule"]["effective_selection_calls"] ==
                                      methods["bytes_only"]["effective_selection_calls"],
            "not_broad_algorithm_generalization": True},
        "node24_new_graph_eligible": eligible,
        "deploy_program": NEW if eligible else INCUMBENT,
        "no_graph_after_reference_reselection": True,
        "single_boot_not_stable_cross_boot": True,
    }


def analyze(root):
    root = Path(root).resolve()
    sealed = p.read(root / "winners.json")
    summary = p.read(root / "confirmation/summary.json")
    selection_summary = p.read(root / "selection/summary.json")
    if summary["status"] != "CONFIRMATION_COMPLETE_NO_RESELECTION" or \
            summary["boot_id"] != sealed["boot_id"] or \
            selection_summary["boot_id"] != sealed["boot_id"]:
        raise ValueError("incomplete or cross-boot phases")
    if summary["winners_sha256"] != p.digest(root / "winners.json") or \
            sealed["selection_journal_sha256"] != p.digest(root / "selection/calls.jsonl"):
        raise ValueError("sealed choice/data modified")
    for phase in ("qualification", "selection", "confirmation"):
        if (root / phase / "invalid_attempt.json").exists():
            raise ValueError("invalid phase cannot enter main results")
    rule = p.read(root / "rule_freeze.json")
    if sealed["rule_sha256"] != p.digest(root / "rule_freeze.json"):
        raise ValueError("rule changed after choice")
    panel = journal(root / "selection/calls.jsonl")
    for method in sealed["winners"]:
        if choose_visible([r for r in panel if r["call_kind"] == "selection"],
                          rule["prefixes"][method]) != sealed["winners"][method]:
            raise ValueError("winner did not follow fixed exposed prefix")
    rows = journal(root / "confirmation/calls.jsonl")
    expected_order = [(r, name, kind, seed, i + (1 if kind == "warmup" else 4))
                      for r, order in enumerate(ORDERS) for name in order
                      for kind in ("warmup", "timing") for i, seed in enumerate(p.seed_order(r))]
    if [(r["round"], r["program"], r["call_kind"], r["seed"], r["call_position"])
        for r in rows] != expected_order:
        raise ValueError("actual independent confirmation order differs")
    result = summarize(rows, sealed)
    result.update(boot_id=sealed["boot_id"], winners_sha256=p.digest(root / "winners.json"),
                  rule_sha256=p.digest(root / "rule_freeze.json"),
                  physical_selection_calls=12, physical_selection_warmup=36,
                  confirmation_calls=72, confirmation_warmup=72,
                  analysis_source_sha256=p.digest(__file__))
    destination = root / "analysis"
    destination.mkdir(exist_ok=False)
    p.write_new(destination / "results.json", result)
    p.write_new(destination / "call_costs.json", {
        "per_phase": {phase: dict(Counter(r["call_kind"] for r in journal(root / phase / "calls.jsonl")))
                      for phase in ("qualification", "selection", "confirmation")},
        "formal_timer_sum_ms": sum(r["latency_ms"] for r in rows if r["call_kind"] == "timing"),
        "not_total_tuning_or_application_wall_time": True,
        "init_upload_readback_costs": "per-call/per-session recorded separately; incomplete totals remain unknown"})
    print(json.dumps({"status": result["status"], "node24_new_graph_eligible": result["node24_new_graph_eligible"],
                      "program_medians_ms": result["program_medians_ms"],
                      "methods": result["methods"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    analyze(parser.parse_args().output_dir)
