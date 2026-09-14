#!/usr/bin/env python3
"""Validate a completed G0 census and emit every cell, including state mismatches."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from run_stage_memory_g0_discovery import percentile, save, sha
from run_stage_memory_g0_pair import analyze


def characterize(rows, cell):
    scored = [r for r in rows if not r["warmup"]]
    result = {}
    for policy in ("allow", "wait"):
        subset = [r for r in scored if r["policy"] == policy]
        overlaps = [max(0, min(s["end_ms"] for s in r["sides"]) -
                           max(s["start_ms"] for s in r["sides"])) for r in subset]
        ages = [r["eligible_ms"]-r["sides"][0]["start_ms"] for r in subset]
        result[policy] = {
            "samples": len(subset), "active_at_eligible": sum(r["active_at_eligible"] for r in subset),
            "positive_graph_overlap": sum(t > .00001 for t in overlaps),
            "overlap_median_ms": statistics.median(overlaps),
            "actual_age_median_ms": statistics.median(ages),
            "age_in_natural_observed_range": sum(cell["natural_age_min_ms"] <= t <=
                                                  cell["natural_age_max_ms"] for t in ages),
            "active_run_median_ms": statistics.median(r["sides"][0]["end_ms"] -
                                                       r["sides"][0]["start_ms"] for r in subset),
            "pending_run_median_ms": statistics.median(r["sides"][1]["end_ms"] -
                                                        r["sides"][1]["start_ms"] for r in subset)}
    assert result["wait"]["positive_graph_overlap"] == 0
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    raw_summary = (root / "summary.json").read_bytes()
    summary = json.loads(raw_summary)
    assert summary["completed"]
    plan_raw = (root / "preregistered.json").read_bytes()
    assert sha(plan_raw) == summary["prereg_sha256"]
    plan = json.loads(plan_raw)
    assert [r["cell"] for r in summary["results"]] == plan["cells"]
    rows_out, details, temperatures = [], [], []
    for item in summary["results"]:
        cell = item["cell"]
        local = root / cell["name"]
        raw = (local / "samples.jsonl").read_bytes()
        assert sha(raw) == item["result_sha256"]
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        audited = analyze(rows)
        assert audited["paired_wait_improvement_median"] == item["paired_wait_improvement_median"]
        assert audited["active_at_eligible_count"] == item["active_at_eligible_count"]
        exposure = characterize(rows, cell)
        details.append({"name": cell["name"], "exposure": exposure})
        for phase in ("before", "after"):
            raw_t = [float(v) for v in (local / f"temp_{phase}.txt").read_text().split()]
            assert len(raw_t) == 6
            temperatures.append({"name": cell["name"], "phase": phase,
                                 "ps_c": (raw_t[0]+raw_t[1])*raw_t[2]/1000,
                                 "pl_c": (raw_t[3]+raw_t[4])*raw_t[5]/1000})
        rows_out.append({"name": cell["name"], "topology": cell["topology"],
                         "active": cell["active"], "ready": cell["ready"], "q": cell["quantile"],
                         "offset_ms": cell["offset_ms"], "natural_encounters": cell["natural_eligible_encounters"],
                         "allow_ms": item["allow_makespan_median_ms"],
                         "wait_ms": item["wait_makespan_median_ms"],
                         "paired_wait_gain_percent": item["paired_wait_improvement_median"]*100,
                         "active_at_eligible": item["active_at_eligible_count"],
                         "allow_positive_overlap": exposure["allow"]["positive_graph_overlap"],
                         "state_qualified": item["state_qualified"],
                         "positive_wait_blocks": item["positive_wait_blocks"],
                         "exploratory_candidate": item["exploratory_candidate"],
                         "release_to_start_p95_ms": item["release_to_start_p95_ms"]})
    rows_out.sort(key=lambda r: r["name"])
    with (root / "all_cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    aggregates = []
    for topology in sorted({r["topology"] for r in rows_out}):
        group = [r for r in rows_out if r["topology"] == topology]
        qualified = [r for r in group if r["state_qualified"]]
        aggregates.append({"topology": topology, "cells": len(group), "state_qualified": len(qualified),
                           "state_mismatch": len(group)-len(qualified),
                           "exploratory_candidates": sum(r["exploratory_candidate"] for r in group),
                           "positive_median_qualified_cells": sum(r["paired_wait_gain_percent"] > 0 for r in qualified),
                           "qualified_gain_range_percent": [min(r["paired_wait_gain_percent"] for r in qualified),
                                                            max(r["paired_wait_gain_percent"] for r in qualified)]})
    result = {"summary_sha256": sha(raw_summary), "analyzer_sha256": sha(Path(__file__).read_bytes()),
              "cells": len(rows_out), "pair_samples": sum(r["samples"] for r in summary["results"]),
              "scored_pair_samples": sum(r["scored_samples"] for r in summary["results"]),
              "state_mismatch_cells": [r["name"] for r in rows_out if not r["state_qualified"]],
              "candidate_cells": [r["name"] for r in rows_out if r["exploratory_candidate"]],
              "topologies": aggregates, "exposure": details, "temperatures": temperatures,
              "formal_protect_allow_labels": None, "opportunity_ceiling": None,
              "limitations": "State mismatches are not negative treatment evidence. CPU affinity overlaps VTA host; "
                             "no DDR attribution, independent boot inference, or full multiway pipeline replay."}
    save(root / "census_audit.json", result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("exposure", "temperatures")}, indent=2))


if __name__ == "__main__":
    main()
