#!/usr/bin/env python3
"""Frozen schedule and boot-level statistics for the thesis Stage-06 ablation."""

from __future__ import annotations

import math
import statistics


CONDITIONS = ("A", "B", "C", "Ca", "D", "E")
BOOT_FORWARD = (
    ("A", "B", "C", "Ca", "D", "E"),
    ("B", "C", "Ca", "D", "E", "A"),
    ("C", "Ca", "D", "E", "A", "B"),
    ("Ca", "D", "E", "A", "B", "C"),
    ("D", "E", "A", "B", "C", "Ca"),
)
EDGES = (("A", "B"), ("B", "C"), ("C", "Ca"), ("Ca", "D"), ("D", "E"), ("A", "E"))
T95_DF4 = 2.776
WINDOW_SCORING_VERSION = "discard_first_two_v2"


def frozen_schedule():
    result = []
    for boot_index, forward in enumerate(BOOT_FORWARD, 1):
        result.append({
            "boot_index": boot_index,
            "block_1": list(forward),
            "block_2": list(reversed(forward)),
        })
    return result


def _positive(values, expected, label):
    if len(values) != expected:
        raise ValueError("{} requires exactly {} samples".format(label, expected))
    result = [float(value) for value in values]
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError("{} samples must be finite and positive".format(label))
    return result


def window_metrics(completion_seconds):
    """Score one 302-completion window after discarding its first two rows."""
    completed = _positive(completion_seconds, 302, "window")
    if any(right <= left for left, right in zip(completed, completed[1:])):
        raise ValueError("completion timestamps must increase strictly")
    retained = completed[2:]
    gaps = [right - left for left, right in zip(retained, retained[1:])]
    ii = statistics.mean(gaps)
    return {
        "rows_total": 302,
        "scoring_version": WINDOW_SCORING_VERSION,
        "rows_discarded_front": 2,
        "rows_retained": 300,
        "gaps_scored": 299,
        "ii_seconds": ii,
        "fps": 1.0 / ii,
    }


def paired_log_ratio(before, after):
    if len(before) != 5 or len(after) != 5:
        raise ValueError("paired log-ratio requires five boot points")
    pairs = list(zip(_positive(before, 5, "before"), _positive(after, 5, "after")))
    ratios = [right / left for left, right in pairs]
    logs = [math.log(value) for value in ratios]
    center = statistics.mean(logs)
    half = T95_DF4 * statistics.stdev(logs) / math.sqrt(5)
    return {
        "n_boots": 5,
        "df": 4,
        "t_critical": T95_DF4,
        "raw_ratios_after_over_before": ratios,
        "geometric_mean_ratio": math.exp(center),
        "ci95_lower": math.exp(center - half),
        "ci95_upper": math.exp(center + half),
    }


def analyze_performance(payload):
    if payload.get("schema") != "thesis_e2e_ablation_performance_v1":
        raise ValueError("unexpected performance schema")
    boots = payload.get("boots", [])
    if len(boots) != 5:
        raise ValueError("five independent boots are required")
    schedule = frozen_schedule()
    boot_points = []
    boot_ids = set()
    for index, (boot, expected) in enumerate(zip(boots, schedule), 1):
        if int(boot.get("boot_index", -1)) != index:
            raise ValueError("boot index/order mismatch")
        boot_id = boot.get("boot_id")
        if not boot_id or boot_id in boot_ids:
            raise ValueError("five distinct nonempty boot_id values are required")
        boot_ids.add(boot_id)
        blocks = boot.get("blocks", [])
        if len(blocks) != 2:
            raise ValueError("each boot requires two condition blocks")
        by_condition = {condition: {"serial": [], "ii": []} for condition in CONDITIONS}
        for block_index, block in enumerate(blocks, 1):
            expected_order = expected["block_{}".format(block_index)]
            rows = block.get("conditions", [])
            if [row.get("condition") for row in rows] != expected_order:
                raise ValueError("boot {} block {} order mismatch".format(index, block_index))
            for row in rows:
                condition = row["condition"]
                serial = _positive(row.get("serial_latency_ms", []), 15, "serial block")
                metric = window_metrics(row.get("completion_seconds", []))
                by_condition[condition]["serial"].append(statistics.median(serial))
                by_condition[condition]["ii"].append(metric["ii_seconds"])
        point = {"boot_index": index, "boot_id": boot_id, "conditions": {}}
        for condition, values in by_condition.items():
            if len(values["serial"]) != 2 or len(values["ii"]) != 2:
                raise ValueError("condition does not have two block observations")
            point["conditions"][condition] = {
                "serial_latency_ms": statistics.mean(values["serial"]),
                "ii_seconds": statistics.mean(values["ii"]),
            }
        boot_points.append(point)
    result = {
        "schema": "thesis_e2e_ablation_boot_analysis_v1",
        "boot_points": boot_points,
        "counts": {
            "boots": 5,
            "windows": 60,
            "completion_rows": 18120,
            "scored_ii_gaps": 17940,
            "serial_scored_calls": 900,
        },
        "paired_effects": {},
    }
    for metric in ("serial_latency_ms", "ii_seconds"):
        result["paired_effects"][metric] = {}
        for before, after in EDGES:
            left = [row["conditions"][before][metric] for row in boot_points]
            right = [row["conditions"][after][metric] for row in boot_points]
            result["paired_effects"][metric]["{}->{}".format(before, after)] = paired_log_ratio(left, right)
    return result
