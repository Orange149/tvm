"""Exact layered DP for the frozen thread/affinity configuration library.

Labels retain CPU service maxima, cumulative VTA service and all core-subset
loads.  Keeping only the currently smallest scalar score is unsafe.  No beam
width or intermediate Top-K pruning is used; Top-K is applied at the terminal
layer only.  For this fixed three-stage domain the DP is deliberately small,
and independent scalar enumeration checks every terminal configuration.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

from cpu_vta_partitioning.models import Cost, score
from .domain import matrix, validate_candidate


@dataclass(frozen=True)
class Label:
    cpu_service: float
    vta_service: float
    subset_loads: tuple
    signatures: tuple
    compatible: tuple


def _costs(document):
    if document.get("source") != "fresh_isolated_board_calibration":
        raise ValueError("static DP requires fresh isolated component profiles")
    costs = {key: Cost.from_dict(value) for key, value in document["costs"].items()}
    required = {s["signature"] for c in matrix()["configs"].values() for s in c["stages"]}
    if not required <= set(costs):
        raise ValueError("static DP is missing a required component signature")
    return costs


def rank_dp(document, k=20):
    """Apply the existing generic subset-capacity objective, without feedback."""
    if type(k) is not int or k < 1:
        raise ValueError("Top-K must be a positive integer")
    configs, costs = matrix()["configs"], _costs(document)
    subsets = tuple(range(1, 16))
    stages = {key: validate_candidate(value) for key, value in configs.items()}

    @lru_cache(None)
    def increment(signature):
        stage = next(s for rows in stages.values() for s in rows if s.signature == signature)
        cost = costs[signature]
        mask = sum(1 << core for core in stage.mask)
        work = cost.run_core_ms + cost.boundary_core_ms
        return (cost.service_ms if stage.device == "cpu" else 0.,
                cost.service_ms if stage.device == "vta" else 0.,
                tuple(work / subset.bit_count() if mask & subset == mask else 0.
                      for subset in subsets))

    labels = [Label(0., 0., (0.,) * 15, (), tuple(sorted(configs)))]
    widths = [1]
    for index in range(3):
        following = []
        for label in labels:
            # Compatibility is part of the state: never invent a combination
            # outside the frozen 32-point thread/mask library.
            options = {}
            for key in label.compatible:
                options.setdefault(stages[key][index].signature, []).append(key)
            for signature, compatible in sorted(options.items()):
                cpu, vta, loads = increment(signature)
                following.append(Label(max(label.cpu_service, cpu), label.vta_service + vta,
                                       tuple(a + b for a, b in zip(label.subset_loads, loads)),
                                       (*label.signatures, signature), tuple(compatible)))
        labels = following
        widths.append(len(labels))

    rows = []
    for label in labels:
        value = max(label.cpu_service, label.vta_service, *label.subset_loads)
        for key in label.compatible:
            reference = score(stages[key], costs, 4)
            if not math.isclose(value, reference["G"], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("DP resource state disagrees with the existing scalar model")
            rows.append({"id": key, "score_ms": round(value, 9),
                         "components": reference["components"]})
    rows.sort(key=lambda row: (row["score_ms"], row["id"]))
    if len(rows) != 32 or len({row["id"] for row in rows}) != 32:
        raise ValueError("DP did not cover the complete frozen configuration library")
    return {"algorithm": "exact_layered_resource_state_dp", "model": "existing_generic_subset_capacity",
            "candidate_count": len(rows), "top_k": min(k, len(rows)), "rankings": rows[:k],
            "state_widths": widths, "intermediate_pruning": False,
            "scalar_enumeration_checked": True,
            "ranking_uses_pipeline_feedback": False}


def enumerate_reference(document, k=20):
    """Independent scalar enumeration for tests and full-domain verification."""
    costs = _costs(document)
    rows = [(round(score(validate_candidate(c), costs, 4)["G"], 9), key)
            for key, c in matrix()["configs"].items()]
    return sorted(rows)[:k]
