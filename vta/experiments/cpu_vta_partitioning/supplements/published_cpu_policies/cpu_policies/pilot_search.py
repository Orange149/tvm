"""Join two explicit host policies around the frozen full-graph resource DP.

The interval/resource DP is this project's existing implementation. Pipe-it
supplies the fixed-worker/disjoint-group execution contract; dependent stages
are not Packrat replicas. Isolated costs for other same-size CPU masks remain
an explicit homogeneous-core approximation, not additional measurements.
"""
from copy import deepcopy
import math


METHODS = ("single_core", "prefix", "enhanced")
ENVIRONMENTS = (("shared4", False), ("reserved3", True))
CONFIGURATION_FIELDS = (
    "id", "stages", "affinity_mode", "execution_mode", "host_mask",
    "control_mask", "core_partition",
)


def _legacy(bundle=None):
    from .pilot_common import load_legacy
    load_legacy(bundle)
    from cpu_strategy.policy import domain, search
    return domain, search


def requirements(programs, bundle=None):
    """Return canonical isolated configurations for both host policies.

    CPU calibration has no concurrent VTA/producer/collector, so the shared
    representative prefix table is common to both domains. VTA calibration
    keeps both host AND control affinity, requiring two distinct records.
    """
    domain, _ = _legacy(bundle)
    result = {}
    for _, reserved in ENVIRONMENTS:
        for key, configuration in domain.complete_components(
                programs, "pinned", reserved, True).items():
            if key in result and result[key] != configuration:
                raise ValueError("component identity collision across host policies")
            result[key] = configuration
    return dict(sorted(result.items()))


def _rank_key(row):
    value = row.get("score_ms")
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid static score")
    if not isinstance(row.get("id"), str) or not row["id"]:
        raise ValueError("missing candidate identity")
    return value, row["id"]


def merge_rankings(per_environment, k=20):
    """Merge independently exact domain Top-K lists into the exact union Top-K.

    A candidate below rank K in its own domain cannot be in the union Top-K.
    Host/control settings are part of the underlying candidate identity.
    """
    if type(k) is not int or k < 1:
        raise ValueError("invalid Top-K budget")
    unique = {}
    for name, result in per_environment.items():
        rows = result["rankings"]
        if rows != sorted(rows, key=_rank_key):
            raise ValueError("unsorted source rankings")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("duplicate identity within source ranking")
        for row in rows:
            identifier = row["id"]
            if identifier in unique:
                # The real domains have distinct host identities. Reject
                # altered duplicates rather than allowing order-dependent data.
                previous = {key: value for key, value in unique[identifier].items()
                            if key != "resource_policy"}
                if previous != row:
                    raise ValueError("candidate identity differs between domains")
                continue
            unique[identifier] = {**deepcopy(row), "resource_policy": name}
    return sorted(unique.values(), key=_rank_key)[:k]


def _verify(dp, enumerated):
    a = [(row["id"], row["score_ms"]) for row in dp["rankings"]]
    b = [(row["id"], row["score_ms"]) for row in enumerated["rankings"]]
    if a != b:
        raise ValueError("full-identity DP / independent enumeration Top-20 mismatch")
    if not a:
        raise ValueError("no complete legal CPU-VTA candidate")
    return {"passed": True, "compared_candidates": len(a),
            "compared_fields": ["complete_configuration_id", "score_ms"],
            "reference_algorithm": enumerated["algorithm"],
            "enumerated_terminals": enumerated["visited_terminals"]}


def rank_all(programs, records, bundle=None):
    """Rank all methods on both resource policies before any pipeline timings."""
    domain, search = _legacy(bundle)
    missing = set(requirements(programs, bundle)) - set(records)
    if missing:
        raise ValueError(f"missing {len(missing)} required isolated component costs")
    methods = {}
    total_enumerated = 0
    for method in METHODS:
        results = {}
        for name, reserved in ENVIRONMENTS:
            rule = {"affinity_mode": "pinned", "reserved_host": reserved,
                    "homogeneous": True}
            print(f"[静态 DP] {method} / {name}", flush=True)
            dp = search.rank_dp(programs, records, rule, method=method, k=20)
            reference = search.enumerate_reference(
                programs, records, rule, method=method, k=20)
            verification = _verify(dp, reference)
            total_enumerated += verification["enumerated_terminals"]
            results[name] = {
                **dp, "resource_policy": name,
                "cpu_cores": [0, 1, 2] if reserved else [0, 1, 2, 3],
                "host_mask": [3] if reserved else [0, 1, 2, 3],
                "control_mask": [3] if reserved else [0, 1, 2, 3],
                "verification": verification,
                "enumeration_cost_aliases": reference.get("cost_aliases", {}),
            }
        combined = merge_rankings(results)
        for rank, row in enumerate(combined, 1):
            row["static_rank"] = rank
        methods[method] = {
            "method": method, "rankings": combined, "per_environment": results,
            "verification": {"passed": True, "domains_checked": len(results),
                             "merge_rule": "sort by (score_ms, complete configuration id)",
                             "top_k": 20},
        }
    return {
        "scope": "single-input short-window full-graph pilot static rankings",
        "methods": methods,
        "all_methods_search_both_host_policies": True,
        "ranking_uses_pipeline_feedback": False,
        "unit_count": 21, "maximum_vta_islands": 3,
        "total_enumerated_terminals": total_enumerated,
        "model_assumptions": {
            "cpu_core_permutation":
                "same-size masks use measured prefix-mask isolated costs; other masks are inferred, not measured",
            "isolated_cpu_cost_reuse":
                "shared by both policies because isolated calibration has no concurrent host or control work",
            "control_thread_cost":
                "producer/collector/queue accounting is not separately modeled; not measured zero and not assumed included in VTA boundary",
            "physical_system_core_count": 4,
            "reserved_cpu_core_count": 3,
            "prefix_formula":
                "original global total core-time / 4 retained; CPU-only prefix / 3 applies in reserved policy",
            "runtime_constraints":
                "single_core and enhanced serialize each reused CPU group; prefix retains its gate-off execution semantics",
            "method_attribution":
                "existing project interval/resource DP using source-audited fixed-worker policies, not Pipe-it original allocation search or Packrat replica DP",
        },
    }


def select_pool(rankings):
    """Return bare execution identities: enhanced20 plus each baseline5."""
    methods = rankings["methods"]
    selected = (methods["enhanced"]["rankings"][:20]
                + methods["single_core"]["rankings"][:5]
                + methods["prefix"]["rankings"][:5])
    result = {}
    for row in selected:
        configuration = {field: deepcopy(row[field]) for field in CONFIGURATION_FIELDS}
        if row["id"] in result and result[row["id"]] != configuration:
            raise ValueError("inconsistent duplicated pipeline identity")
        result[row["id"]] = configuration
    if len(result) > 30:
        raise ValueError("pilot pipeline pool exceeds 30 configurations")
    return dict(sorted(result.items()))
