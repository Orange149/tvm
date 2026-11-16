"""Native output validation with separate isolated and whole-pipeline CPU scopes."""
from copy import deepcopy
import math
from pathlib import Path
import statistics

from cpu_vta_partitioning.common import digest, read
from cpu_vta_partitioning.measurement import (
    frames, input_sequence, affinity, correctness, calibration, window,
)
from .domain import ENV, validate_candidate, validate_stage


def make_task(kind, stages, *, warmup, scored, tail, indices, **labels):
    if (not isinstance(kind, str) or not kind or
            any(type(x) is not int or x < 0 for x in (warmup, scored, tail)) or scored < 1):
        raise ValueError("invalid task kind or measurement window")
    if (not indices or any(type(i) is not int or not 0 <= i < 67 for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError("invalid task input indices")
    checked = tuple(validate_stage(s) for s in stages)
    if kind in ("calibration", "qualify_component"):
        if len(checked) != 1 or tail:
            raise ValueError("isolated component tasks require one stage and no tail")
    else:
        validate_candidate(checked)
    if kind.startswith("qualify") and warmup + scored + tail < len(indices):
        raise ValueError("qualification window does not cover all scheduled inputs")
    reserved = {"kind", "stages", "environment", "input_indices", "warmup", "scored", "tail",
                "experiment", "execution_mode", "task_id", "allow_cpu_affinity_superset"}
    if reserved.intersection(labels):
        raise ValueError("task labels override frozen execution fields")
    document = {"kind": kind, "stages": [s.to_dict() for s in checked], "environment": deepcopy(ENV),
                "input_indices": list(indices), "warmup": warmup, "scored": scored, "tail": tail,
                "experiment": "resnet18_cpu_strategy", "execution_mode": "off",
                "allow_cpu_affinity_superset": True, **labels}
    document["task_id"] = digest(document)[:24]
    return document


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid or missing " + label)
    return float(value)


def whole_pipeline_window(rows, warmup, scored, tail):
    for row in rows:
        _number(row.get("total_latency_ms"), "frame latency")
    result = window(rows, warmup, scored, tail)
    clocks = [_number(r.get("completion_process_cpu_ms"), "whole-process CPU clock") for r in rows]
    if any(b < a for a, b in zip(clocks, clocks[1:])):
        raise ValueError("nonmonotonic whole-process CPU clock")
    first, last = warmup, warmup + scored - 1
    cpu_ms = clocks[last] - clocks[first]
    wall_ms = rows[last]["completion_ms"] - rows[first]["completion_ms"]
    if cpu_ms <= 0:
        raise ValueError("whole-pipeline CPU window must have positive work")
    if cpu_ms / wall_ms > ENV["cores"] + .05:
        raise ValueError("whole-process CPU clock exceeds physical core capacity")
    result.update(process_cpu_ms=cpu_ms, process_core_ms_per_completion=cpu_ms / (scored - 1),
                  average_cpu_cores=cpu_ms / wall_ms,
                  cpu_time_scope="whole_process_between_scored_completions")
    return result


def thread_activity(affinities, stages):
    """Thread CPU counters cover the session, not the scored throughput interval."""
    result = []
    for stage, row in zip(stages, affinities["stages"]):
        active, total = [], 0.
        for entry in row["effective_affinities"]:
            delta = _number(entry.get("cpu_time_delta_ms"), "thread CPU time increment")
            total += delta
            if delta > 0:
                active.append(entry["tid"])
        if len({entry["tid"] for entry in row["effective_affinities"]}) != len(row["effective_affinities"]):
            raise ValueError("duplicate TID in stage affinity report")
        result.append({"configured_threads": stage["threads"],
                       "observed_active_tids": sorted(active), "observed_active_threads": len(active),
                       "observed_thread_cpu_ms": total,
                       "scope": "whole_session_thread_cpu_time"})
    return result


def isolated_components(rows, warmup, scored):
    result = calibration(rows, warmup, scored)
    kept = rows[warmup:]
    for operation in ("set", "run", "get"):
        result[operation + "_ms"] = statistics.mean(_number(r.get(f"stage0_{operation}_ms"),
                                                   operation + " wall time") for r in kept)
        result[operation + "_core_ms"] = statistics.mean(_number(r.get(f"stage0_{operation}_process_cpu_ms"),
                                                        operation + " isolated CPU time") for r in kept)
    result["cpu_time_scope"] = "exclusive_process_cpu_time"
    return result


def validate_summary(rawdir, document, bundle):
    rawdir = Path(rawdir)
    total = sum(document[k] for k in ("warmup", "scored", "tail"))
    rows = frames(rawdir / "result.jsonl", total)
    if any(type(row.get(key)) is not int for row in rows
           for key in ("frame_id", "completion_index", "input_index")):
        raise ValueError("frame and input identities must be integers")
    input_sequence(rows, document)
    affinities = read(rawdir / "affinity.json")
    affinity(affinities, document["stages"])
    activity = thread_activity(affinities, document["stages"])
    if any(r.get("cpu_stage_gate_mode", "off") != "off" for r in rows):
        raise ValueError("CPU stage gating changed the focused experiment")
    if document["kind"].startswith("qualify"):
        if bundle is None:
            raise ValueError("qualification replay requires the verified reference bundle")
        correctness(rows, document, bundle, rawdir)
        return {"correct": True, "thread_activity": activity}
    if document["kind"] == "calibration":
        if len(document["stages"]) != 1 or document["tail"]:
            raise ValueError("calibration is not an isolated component session")
        result = isolated_components(rows, document["warmup"], document["scored"])
    else:
        result = whole_pipeline_window(rows, document["warmup"], document["scored"], document["tail"])
    return {**result, "thread_activity": activity}
