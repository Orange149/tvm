"""Frozen stage-level thread/core configurations; no TVM or board imports."""
from copy import deepcopy
from pathlib import Path

from cpu_vta_partitioning.domain import Stage, candidate_id


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
ALL_CORES = (0, 1, 2, 3)
ENV = {"cores": 4, "cpu_cores": list(ALL_CORES), "host_cores": list(ALL_CORES),
       "poll_sleep_ns": 1000, "post_start_sleep_ns": 1000}
PIPELINE_RANGES = (("cpu", 0, 3), ("vta", 3, 18), ("cpu", 18, 21))
DIAGNOSTIC_UNITS = (0, 2, 20)


def validate_stage(value, programs=None):
    """Return a checked Stage; affinity is a permitted group, not exclusive cores."""
    stage = value if isinstance(value, Stage) else Stage.from_dict(value)
    if stage.device not in ("cpu", "vta"):
        raise ValueError("unknown stage device")
    if (type(stage.begin) is not int or type(stage.end) is not int
            or not 0 <= stage.begin < stage.end <= 21):
        raise ValueError("invalid stage range")
    if (type(stage.threads) is not int or stage.threads < 1
            or not stage.mask or any(type(c) is not int for c in stage.mask)
            or tuple(sorted(set(stage.mask))) != stage.mask
            or not set(stage.mask) <= set(ALL_CORES)):
        raise ValueError("invalid stage threads or core mask")
    if stage.device == "cpu":
        if stage.threads > len(stage.mask):
            raise ValueError("CPU thread count exceeds allowed core count")
        allowed = {(0, 3), (18, 21), *((i, i + 1) for i in DIAGNOSTIC_UNITS)}
        if (stage.begin, stage.end) not in allowed:
            raise ValueError("stage is outside the focused CPU experiment")
    elif (stage.begin, stage.end, stage.threads, stage.mask) != (3, 18, 1, ALL_CORES):
        raise ValueError("VTA stage or host placement changed")
    if isinstance(value, dict):
        for key, actual in (("program", stage.program), ("signature", stage.signature)):
            if key in value and value[key] != actual:
                raise ValueError("stage identity mismatch: " + key)
    if programs is not None:
        if stage.program not in programs:
            raise ValueError("stage program is not in the verified bundle")
        p = programs[stage.program]
        if any(p.get(k) != getattr(stage, k) for k in ("device", "begin", "end")):
            raise ValueError("program range/device differs from stage")
    return stage


def configuration(threads, masks):
    stages = (Stage("cpu", 0, 3, threads[0], tuple(masks[0])),
              Stage("vta", 3, 18, 1, ALL_CORES),
              Stage("cpu", 18, 21, threads[1], tuple(masks[1])))
    for stage in stages:
        validate_stage(stage)
    return {"id": candidate_id(stages) + "__gate_off",
            "stages": [s.to_dict() for s in stages], "execution_mode": "off"}


def validate_candidate(value, programs=None):
    rows = value["stages"] if isinstance(value, dict) else value
    stages = tuple(validate_stage(s, programs) for s in rows)
    if tuple((s.device, s.begin, s.end) for s in stages) != PIPELINE_RANGES:
        raise ValueError("candidate changes the fixed three-stage partition")
    if isinstance(value, dict):
        if value.get("execution_mode", "off") != "off":
            raise ValueError("CPU stage gating is outside this experiment")
        if value.get("id") != candidate_id(stages) + "__gate_off":
            raise ValueError("candidate identity mismatch")
    return stages


def matrix():
    configs, families = {}, {k: [] for k in ("shared", "disjoint", "overlap", "references")}

    def add(family, threads, masks):
        row = configuration(threads, masks)
        key = row["id"]
        if key in configs:
            raise ValueError("frozen matrix unexpectedly contains duplicate configurations")
        configs[key] = row
        families[family].append(key)
        return key

    shared = {}
    for front in range(1, 5):
        for back in range(1, 5):
            shared[(front, back)] = add("shared", (front, back), (ALL_CORES, ALL_CORES))
    separate = {}
    for front, back in ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2)):
        masks = (tuple(range(front)), tuple(range(4 - back, 4)))
        separate[(front, back, "low")] = add("disjoint", (front, back), masks)
        mirrored = tuple(tuple(sorted(3 - c for c in mask)) for mask in masks)
        separate[(front, back, "high")] = add("disjoint", (front, back), mirrored)
    low = add("overlap", (2, 2), ((0, 1), (0, 1)))
    high = add("overlap", (2, 2), ((2, 3), (2, 3)))
    current = add("references", (4, 2), (ALL_CORES, (0, 1)))
    single = add("references", (1, 1), ((1,), (0,)))
    if len(configs) != 32:
        raise ValueError("frozen CPU matrix coverage changed")
    return {"configs": configs, "families": families,
            "references": {"current": current, "single_thread": single},
            "interventions": {"overlap_low": low, "overlap_high": high,
                              "separate_low": separate[(2, 2, "low")],
                              "separate_high": separate[(2, 2, "high")]},
            "warmup_anchors": [shared[(4, 4)], single, separate[(2, 2, "low")]]}


def components():
    result = {}
    for candidate in matrix()["configs"].values():
        for stage in candidate["stages"]:
            result[stage["signature"]] = {"stage": stage, "role": "pipeline", "blocks": 1}
    for unit in DIAGNOSTIC_UNITS:
        for threads in range(1, 5):
            stage = Stage("cpu", unit, unit + 1, threads, ALL_CORES)
            result[stage.signature] = {"stage": stage.to_dict(), "role": "diagnostic", "blocks": 3}
    if len(result) != 34:
        raise ValueError("frozen component signature coverage changed")
    return [result[key] for key in sorted(result)]


def protocol():
    return {"experiment": "resnet18_cpu_strategy", "schema_version": 2, "seed": 20261104,
            "scope": "fixed_partition_stage_level", "environment": deepcopy(ENV),
            "topology": [{"device": d, "begin": b, "end": e} for d, b, e in PIPELINE_RANGES],
            "exploration_boots": 1, "confirmation_boots": 3, "rounds": 2,
            "calibration": {"input_indices": [0], "warmup": 5, "scored": 32,
                            "tail": 0, "diagnostic_blocks": 3},
            "performance": {"input_indices": [16], "warmup": 8, "extended_warmup": 24,
                            "scored": 32, "tail": 4, "warmup_tolerance": .05},
            "qualification_indices": [0, 16, 17, 18], "anchor_drift_limit": .05,
            "max_round_attempts": 2, "anchor_reference": "single_thread",
            "expected": {"pipeline_configurations": 32, "offline_candidate_configurations": 32,
                         "max_measured_pipeline_configurations": 13, "control_configurations": 8,
                         "pipeline_cpu_signatures": 21,
                         "pipeline_vta_signatures": 1, "diagnostic_signatures": 12,
                         "component_signatures": 34, "max_confirmation_configurations": 10},
            "runtime": {"queue_depth": 2, "spin_count": 0, "cpu_gate": "off",
                        "copy": "ordinary", "replay": "disabled",
                        "affinity": "core_group_migration_allowed",
                        "allow_cpu_affinity_superset": True},
            "numeric": {"integer": "exact", "rtol": 1e-4, "atol": 1e-5},
            "boot_policy": "manual_reboot_only", "formal_significance": False,
            "selection": {"profile_disjoint": "fresh_isolated_service_minmax_before_feedback",
                          "local_fastest": "fresh_isolated_service_per_stage_before_feedback",
                          "dp_static": "static_dp_top1_before_pipeline",
                          "dp_budget5": "best_two_round_mean_ii_among_frozen_dp_top5_then_id"},
            "search": {"algorithm": "exact_layered_resource_state_dp",
                       "candidate_count": 32, "top_k": 20, "screening_budget": 5,
                       "objective": "existing_generic_subset_capacity",
                       "intermediate_pruning": False, "enumeration_crosscheck": True,
                       "ranking_uses_pipeline_feedback": False, "default_full_sweep": False},
            "cpu_accounting": {"calibration": "exclusive_process_cpu_time",
                               "pipeline": "whole_process_between_scored_completions"}}
