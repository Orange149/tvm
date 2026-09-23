"""Frozen experimental design and fail-closed analysis; no board side effects."""
from collections import Counter
import math
import statistics

SEEDS = (0, 20250901, 20260910)
LETTERS = "ABCDEFGH"
QUEUE_ENV = {"VTA_INSN_BUFFER_BYTES": "33554432", "VTA_UOP_BUFFER_BYTES": "33554432",
             "VTA_INSN_SUBMIT_THRESHOLD_BYTES": "33554432", "VTA_REPLAY_POLICY": "disabled",
             "VTA_RUNTIME_TRACE": "0", "VTA_QUEUE_BOUNDARY_AUDIT": "0", "VTA_QUEUE_DRAIN_AUDIT": "0"}
VERSIONS = {
    "A": (0, 2, False), "B": (0, 1, False), "C": (1, 1, False),
    "D": (2, 1, False), "E": (4, 1, True), "F": (4, 1, False),
    "G": (5, 1, True), "H": (5, 1, False),
}
COMPARISONS = (("A", "B", "thread"), ("B", "C", "input"),
               ("B", "D", "traversal"), ("D", "E", "sync"),
               ("C", "G", "sync"), ("E", "F", "residency"),
               ("G", "H", "residency"), ("A", "F", "net"), ("A", "H", "net"))


def williams():
    return [[LETTERS[(n + r) % 8] for n in (0, 1, 7, 2, 6, 3, 5, 4)] for r in range(8)]


def balance(labels):
    rows = [[v for v in row if v in labels] for row in williams()]
    positions, predecessors = Counter(), Counter()
    for row in rows:
        positions.update(f"{v}@{p}" for p, v in enumerate(row))
        predecessors.update(f"{a}>{b}" for a, b in zip(row, row[1:]))
    return dict(rows=rows, position_counts=dict(positions), predecessor_counts=dict(predecessors),
                complete_eight_version_williams=set(labels) == set(LETTERS),
                scope="within rows, candidate groups only; excludes seed and cross-row/tile transitions")


def schedule(accepted):
    """Complete ordered list of actual single operator timing invocations."""
    out = []
    for r, row in enumerate(williams()):
        seeds = SEEDS[r % 3:] + SEEDS[:r % 3]
        for tile in (("T8", "T7") if r % 2 == 0 else ("T7", "T8")):
            for label in row:
                cid = tile + "_" + label
                if cid not in accepted:
                    continue
                for seed in seeds:
                    out.append(dict(ordinal=len(out), round=r, tile=tile, version=label,
                                    candidate_id=cid, seed=seed))
    return out


def validate_timing(rows, accepted):
    expected = schedule(accepted)
    if len(rows) != len(expected):
        raise ValueError("timing invocation coverage differs")
    for row, spec in zip(rows, expected):
        if any(row.get(k) != v for k, v in spec.items()):
            raise ValueError("timing execution order differs")
        if not row.get("correct") or row.get("complete_operator_calls") != 1 or row.get("hidden_warmup_calls") != 0:
            raise ValueError("incorrect output or extra invocation")
        value = row.get("latency_ms")
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError("invalid timing")


def paired(rows, baseline, candidate):
    times = {}
    for cid in (baseline, candidate):
        for r in range(8):
            group = [x for x in rows if x["candidate_id"] == cid and x["round"] == r]
            if len(group) != 3 or {x["seed"] for x in group} != set(SEEDS):
                raise ValueError("incomplete round/seed pair")
            times[cid, r] = statistics.median(x["latency_ms"] for x in group)
    gains = [100 * (1 - times[candidate, r] / times[baseline, r]) for r in range(8)]
    return dict(baseline=baseline, candidate=candidate, round_gains_pct=gains,
                baseline_round_ms=[times[baseline, r] for r in range(8)],
                candidate_round_ms=[times[candidate, r] for r in range(8)],
                paired_median_gain_pct=statistics.median(gains),
                wins=sum(g > 0 for g in gains), losses=sum(g < 0 for g in gains))


def classify(boots, matched=True):
    if not matched:
        return "control_unmatched"
    if len(boots) != 2:
        return "awaiting_second_boot"
    if all(x["paired_median_gain_pct"] >= 2 and x["wins"] >= 6 for x in boots):
        return "stable_improvement"
    if all(x["paired_median_gain_pct"] <= -2 and x["losses"] >= 6 for x in boots):
        return "stable_regression"
    return "no_stable_direction_threshold"


def logical_signature(profile):
    return {k: v for k, v in profile.items()
            if (k.startswith(("load_buffer_2d_", "store_buffer_2d_")) and k.endswith(("_bytes", "_calls")))
            # FSim has no board driver counter. Compare runtime requests here;
            # independently audit actual submissions from queue/dump evidence.
            or k in ("synchronize_calls", "synchronize_insns")}
