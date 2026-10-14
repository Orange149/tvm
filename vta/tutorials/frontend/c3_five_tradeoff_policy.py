"""Fixed B1 proposal rule and equal-information choice; no learned new labels."""

import math
from statistics import median

INCUMBENT = "R13-G"
NEW = "N24-R12-G2-PREFETCH"
PROGRAMS = (INCUMBENT, "R12-G", NEW, "Input-7x7")
SEEDS = (0, 20250901, 20260910)


def prefixes(features):
    if set(features) != set(PROGRAMS):
        raise ValueError("not the frozen four-program pool")
    if any(not f["local_qualified"] for f in features.values()):
        raise ValueError("qualification is a hard gate, not a latency prediction")
    baseline, candidate = features[INCUMBENT], features[NEW]
    required_equal = ("inp_bytes", "acc_bytes", "out_bytes", "mac", "reset", "submissions")
    if any(candidate[k] != baseline[k] for k in required_equal) or \
            not candidate["normal_completion"] or candidate["independent_windows"] <= 0 or \
            candidate["wgt_arena_bytes"] > 256 * 1024:
        raise ValueError("fixed prefetch hypothesis conditions not established")
    other = [name for name in PROGRAMS if name != INCUMBENT]
    bytes_order = sorted(other, key=lambda name: (
        features[name]["inp_bytes"] + features[name]["wgt_bytes"] +
        features[name]["acc_bytes"] + features[name]["out_bytes"],
        features[name]["wgt_arena_bytes"], name))
    return {
        "rule": [NEW], "bytes_only": bytes_order[:1], "incumbent_only": [],
        "budget_new_candidates": 1,
        "rule_is_opportunity_priority_not_safe_performance_dominance": True,
        "hypotheses": {"new_vs_R12-G": "faster", "new_vs_R13-G": "unknown",
                       "new_vs_Input-7x7": "unknown"},
        "no_numerical_extrapolation_of_old_local_model": True,
    }


def choose_visible(panel, prefix):
    """See only incumbent+prefix, each exactly three independent single calls."""
    allowed = (INCUMBENT, *prefix)
    medians = {}
    for name in allowed:
        rows = [r for r in panel if r["program"] == name]
        if len(rows) != 3 or {r["seed"] for r in rows} != set(SEEDS):
            raise ValueError("choice requires exactly three fixed-seed selection samples")
        if any(r.get("call_kind") != "selection" or not r.get("correct") or
               r.get("status") != "PASS" or not math.isfinite(r["latency_ms"]) or
               r["latency_ms"] <= 0 for r in rows):
            raise ValueError("invalid or non-selection data cannot choose winner")
        medians[name] = median(r["latency_ms"] for r in rows)
    return min(allowed, key=lambda name: (medians[name], name != INCUMBENT, name))
