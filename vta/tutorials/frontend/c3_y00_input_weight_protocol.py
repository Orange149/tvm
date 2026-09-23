"""Frozen one-boot protocol for Y00F06 original/input/input+weight variants."""

from collections import Counter
import math
import statistics


SEEDS = (0, 20250901, 20260910)
VARIANTS = ("O", "I", "IW")
QUEUE_ENV = {
    "VTA_INSN_BUFFER_BYTES": "33554432",
    "VTA_UOP_BUFFER_BYTES": "33554432",
    "VTA_INSN_SUBMIT_THRESHOLD_BYTES": "33554432",
    "VTA_REPLAY_POLICY": "disabled",
    "VTA_RUNTIME_TRACE": "0",
    "VTA_QUEUE_BOUNDARY_AUDIT": "0",
    "VTA_QUEUE_DRAIN_AUDIT": "0",
}
SEQUENCES = (
    ("O", "I", "IW"),
    ("I", "IW", "O"),
    ("IW", "O", "I"),
    ("IW", "I", "O"),
    ("O", "IW", "I"),
    ("I", "O", "IW"),
)
COMPARISONS = (
    ("Y00F06_O", "Y00F06_I", "input_reuse_vs_original"),
    ("Y00F06_I", "Y00F06_IW", "incremental_staged_weight_residency"),
    ("Y00F06_O", "Y00F06_IW", "combined_vs_original"),
)


def balance(accepted=VARIANTS):
    labels = set(accepted)
    rows = [[item for item in row if item in labels] for row in SEQUENCES]
    positions, predecessors = Counter(), Counter()
    for row in rows:
        positions.update("{}@{}".format(item, index) for index, item in enumerate(row))
        predecessors.update(
            "{}>{}".format(left, right) for left, right in zip(row, row[1:])
        )
    return {
        "rows": rows,
        "position_counts": dict(sorted(positions.items())),
        "predecessor_counts": dict(sorted(predecessors.items())),
        "complete_three_variant_six_permutation": labels == set(VARIANTS),
        "scope": "within-round candidate groups; excludes seeds and cross-round transitions",
    }


def candidate_prefix(accepted_ids):
    prefixes = {candidate_id.rsplit("_", 1)[0] for candidate_id in accepted_ids}
    if len(prefixes) != 1:
        raise ValueError("candidate ids must share one family prefix")
    return next(iter(prefixes))


def schedule(accepted_ids):
    accepted = set(accepted_ids)
    prefix = candidate_prefix(accepted_ids)
    result = []
    for round_index, variants in enumerate(SEQUENCES):
        seed_order = SEEDS[round_index % 3 :] + SEEDS[: round_index % 3]
        for variant in variants:
            candidate_id = prefix + "_" + variant
            if candidate_id not in accepted:
                continue
            for seed in seed_order:
                result.append(
                    {
                        "ordinal": len(result),
                        "round": round_index,
                        "variant": variant,
                        "candidate_id": candidate_id,
                        "seed": seed,
                    }
                )
    return result


def validate_timing(rows, accepted_ids):
    expected = schedule(accepted_ids)
    if len(rows) != len(expected):
        raise ValueError("timing invocation coverage differs")
    for row, spec in zip(rows, expected):
        if any(row.get(key) != value for key, value in spec.items()):
            raise ValueError("timing execution order differs")
        if (
            not row.get("correct")
            or row.get("complete_operator_calls") != 1
            or row.get("hidden_warmup_calls") != 0
        ):
            raise ValueError("incorrect output or extra invocation")
        latency = row.get("latency_ms")
        if not isinstance(latency, (float, int)) or not math.isfinite(latency) or latency <= 0:
            raise ValueError("invalid timing")


def paired(rows, baseline, candidate):
    times = {}
    for candidate_id in (baseline, candidate):
        for round_index in range(len(SEQUENCES)):
            group = [
                row
                for row in rows
                if row["candidate_id"] == candidate_id and row["round"] == round_index
            ]
            if len(group) != 3 or {row["seed"] for row in group} != set(SEEDS):
                raise ValueError("incomplete round/seed pair")
            times[candidate_id, round_index] = statistics.median(
                row["latency_ms"] for row in group
            )
    gains = [
        100 * (1 - times[candidate, index] / times[baseline, index])
        for index in range(len(SEQUENCES))
    ]
    return {
        "baseline": baseline,
        "candidate": candidate,
        "round_gains_pct": gains,
        "baseline_round_ms": [times[baseline, index] for index in range(len(SEQUENCES))],
        "candidate_round_ms": [times[candidate, index] for index in range(len(SEQUENCES))],
        "paired_median_gain_pct": statistics.median(gains),
        "wins": sum(value > 0 for value in gains),
        "losses": sum(value < 0 for value in gains),
    }


def development_classification(observation):
    if observation["paired_median_gain_pct"] >= 2 and observation["wins"] >= 5:
        return "development_improvement_signal"
    if observation["paired_median_gain_pct"] <= -2 and observation["losses"] >= 5:
        return "development_regression_signal"
    return "no_development_direction_threshold"


def logical_signature(profile):
    return {
        key: value
        for key, value in profile.items()
        if (
            key.startswith(("load_buffer_2d_", "store_buffer_2d_"))
            and key.endswith(("_bytes", "_calls"))
        )
        or key in ("synchronize_calls", "synchronize_insns")
    }
