#!/usr/bin/env python3
"""Read-only timing analysis for the frozen CPU/VTA residual experiment.

All times are milliseconds.  A pipeline window is exactly 307 completion rows:
five warmup rows and two settling rows are discarded, leaving 300 completions
and 299 intervals.  Busy times and waits are observations, not additive estimates
of the initiation interval or measurements of physical DDR traffic.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


DEFAULT_SOURCE = Path(__file__).parent / (
    "report_out/stage_tile_cotuning/thesis_supplement_experiments/results/06/"
    "20260916T124859Z_single_image_audit_bfec496d"
)
T95_DF2 = 4.3026527299


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Non-finite or missing timing: " + label)
    return float(value)


def _stats(values):
    values = list(values)
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    values = [_number(v, "statistic") for v in values]
    return {"count": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "min": min(values), "max": max(values)}


def _validate_rows(rows, mode):
    rows = list(rows)
    if not rows:
        raise ValueError("Empty completion rows")
    count = rows[0].get("stage_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("Invalid stage_count")
    previous_completion = None
    previous_index = None
    for row in rows:
        if row.get("mode") != mode or row.get("stage_count") != count:
            raise ValueError("Mixed modes or stage counts")
        completion = _number(row.get("completion_ms"), "completion_ms")
        index = row.get("completion_index", row.get("frame_id"))
        if not isinstance(index, int) or isinstance(index, bool):
            raise ValueError("Missing integer completion index")
        if previous_completion is not None and (completion <= previous_completion or index != previous_index + 1):
            raise ValueError("Completion rows are not consecutive and strictly ordered")
        previous_completion, previous_index = completion, index
        for i in range(count):
            start = _number(row.get(f"stage{i}_start_ms"), "stage start")
            end = _number(row.get(f"stage{i}_end_ms"), "stage end")
            if end < start:
                raise ValueError("Stage ends before it starts")
            for field in ("set_ms", "run_ms", "get_ms", "ms"):
                if _number(row.get(f"stage{i}_{field}"), field) < 0:
                    raise ValueError("Negative stage service")
            components = sum(row[f"stage{i}_{f}"] for f in ("set_ms", "run_ms", "get_ms"))
            if not math.isclose(row[f"stage{i}_ms"], components, abs_tol=0.01, rel_tol=1e-6):
                raise ValueError("Stage service does not match set/run/get")
            if not math.isclose(end - start, row[f"stage{i}_ms"], abs_tol=0.01, rel_tol=1e-6):
                raise ValueError("Stage span does not match inclusive service")
            run_start, run_end = row.get(f"stage{i}_run_start_ms"), row.get(f"stage{i}_run_end_ms")
            if run_start is not None and run_end is not None:
                run_start = _number(run_start, "run start")
                run_end = _number(run_end, "run end")
                if run_start < start - 0.01 or run_end > end + 0.01 or not math.isclose(
                        run_end - run_start, row[f"stage{i}_run_ms"], abs_tol=0.01, rel_tol=1e-6):
                    raise ValueError("Run interval does not match stage/run service")
    return rows, count


def _stage_stats(rows, i):
    result = {"index": i}
    for field in ("set_ms", "run_ms", "get_ms", "scheduled_service_ms",
                  "vta_mutex_wait_ms", "vta_mutex_held_ms", "vta_run_call_ms",
                  "process_cpu_ms", "run_process_cpu_ms"):
        values = [r.get(f"stage{i}_{field}") for r in rows]
        result[field] = _stats(v for v in values if v is not None)
    result["busy_ms"] = _stats(r[f"stage{i}_ms"] for r in rows)
    result["cpu_time_available"] = result["process_cpu_ms"]["count"] == len(rows)
    return result


def analyze_serial(rows, discard=5):
    rows, count = _validate_rows(rows, "serial")
    if not isinstance(discard, int) or isinstance(discard, bool) or discard < 0 or discard >= len(rows):
        raise ValueError("Invalid serial discard count")
    scored = rows[discard:]
    stages = [_stage_stats(scored, i) for i in range(count)]
    stage_max = max(stage["busy_ms"]["mean"] for stage in stages)
    exclusive = (all(r.get("stage_cpu_time_scope") == "exclusive_process_cpu_time" for r in scored)
                 and all(s["process_cpu_ms"]["count"] == len(scored) for s in stages))
    cpu_sum = sum(s["process_cpu_ms"]["mean"] for s in stages) if exclusive else None
    cpu_capacity = cpu_sum / 4 if cpu_sum is not None else None
    bound = max(stage_max, cpu_capacity) if cpu_capacity is not None else stage_max
    resource = {"stage_service_max_ms": stage_max,
                "exclusive_process_cpu_capacity_ms": cpu_capacity,
                "sum_process_cpu_mean_ms": cpu_sum, "host_cpu_capacity": 4,
                "bound_ms": bound, "complete_cpu_demand_observation": exclusive,
                "affinity_capacity_lower_bound_ms": None,
                "assumption": "matched serial demand transfer; concurrent stage process CPU time remains null",
                "affinity_hall_bound": "unknown", "ddr_attribution": "unknown"}
    outer = [_number(r.get("total_latency_ms"), "total_latency_ms") -
             sum(r[f"stage{i}_ms"] for i in range(count)) for r in scored]
    return {"kind": "serial", "raw_row_count": len(rows), "discarded_rows": discard,
            "scored_rows": len(scored), "stages": stages,
            "total_latency_ms": _stats(r["total_latency_ms"] for r in scored),
            "outer_residual_ms": _stats(outer),
            "stage_service_max_ms": stage_max, "resource_lower_bound": resource,
            "m1_ms": bound, "m1_kind": "matched_serial_resource_lower_bound",
            "ddr_attribution": "unknown"}


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Negative interval")
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(end, merged[-1][1])
    return merged


def _intersection_duration(left, right):
    i = j = 0
    total = 0.0
    while i < len(left) and j < len(right):
        total += max(0.0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _cpu_overlap(rows, count):
    if count < 3:
        return {"available": False, "reason": "first and last CPU stage contract required"}
    intervals = []
    for i in (0, count - 1):
        values = []
        for r in rows:
            start, end = r.get(f"stage{i}_run_start_ms"), r.get(f"stage{i}_run_end_ms")
            if start is None or end is None:
                return {"available": False, "reason": "run interval timestamps unavailable"}
            values.append((_number(start, "run start"), _number(end, "run end")))
        intervals.append(_union(values))
    overlap = _intersection_duration(*intervals)
    durations = [sum(end - start for start, end in x) for x in intervals]
    return {"available": True, "stage_indices": [0, count - 1],
            "overlap_run_ms": overlap, "first_cpu_run_ms": durations[0],
            "last_cpu_run_ms": durations[1],
            "first_cpu_fraction": overlap / durations[0] if durations[0] else None,
            "last_cpu_fraction": overlap / durations[1] if durations[1] else None,
            "scope": "intersection of scored run interval unions across frames"}


def _queue_stats(rows):
    edges = {}
    seen = set()
    for row in rows:
        for entry in row.get("queue_timings", []) or []:
            key = (entry.get("edge_index"), entry.get("push_request_ms"),
                   entry.get("enqueue_ms"), entry.get("dequeue_ms"), entry.get("push_return_ms"))
            if key in seen:
                continue
            seen.add(key)
            edge = entry.get("edge_index")
            if not isinstance(edge, int) or isinstance(edge, bool):
                raise ValueError("Invalid queue edge index")
            request = _number(entry.get("push_request_ms"), "push request")
            enqueue = _number(entry.get("enqueue_ms"), "enqueue")
            dequeue = _number(entry.get("dequeue_ms"), "dequeue")
            returned = _number(entry.get("push_return_ms"), "push return")
            if enqueue < request or dequeue < enqueue or returned < enqueue:
                raise ValueError("Queue timestamps violate causality")
            values = edges.setdefault(edge, {"producer_block_ms": [], "residence_ms": [], "push_call_ms": []})
            values["producer_block_ms"].append(enqueue - request)
            values["residence_ms"].append(dequeue - enqueue)
            values["push_call_ms"].append(returned - request)
    return {"available": bool(edges), "edges": [dict(edge_index=i, **{k: _stats(v) for k, v in values.items()})
                                                 for i, values in sorted(edges.items())],
            "scope": "observed waits; never added to stage services or II"}


def analyze_pipeline(rows):
    rows, count = _validate_rows(rows, "pipeline")
    if len(rows) != 307:
        raise ValueError("Pipeline window requires exactly 307 rows (5 + 2 + 300)")
    scored = rows[7:]
    stages = []
    for i in range(count):
        stage = _stage_stats(scored, i)
        gaps = [b[f"stage{i}_start_ms"] - a[f"stage{i}_end_ms"] for a, b in zip(scored, scored[1:])]
        if min(gaps) < -0.01:
            raise ValueError("Same stage overlaps itself or rows are misordered")
        span = scored[-1][f"stage{i}_end_ms"] - scored[0][f"stage{i}_end_ms"]
        if span <= 0:
            raise ValueError("Non-positive stage completion span")
        busy = sum(r[f"stage{i}_ms"] for r in scored[1:])
        stage.update(completion_span_ms=span, completion_ii_ms=span / 299,
                     span_aligned_busy_mean_ms=busy / 299, gap_ms=_stats(gaps),
                     mean_gap_ms=statistics.mean(gaps), busy_fraction=busy / span,
                     span_accounting_error_ms=span - busy - sum(gaps))
        stages.append(stage)
    span = scored[-1]["completion_ms"] - scored[0]["completion_ms"]
    return {"kind": "pipeline", "raw_row_count": 307, "discarded_rows": 7,
            "scored_rows": 300, "completion_gaps": 299, "completion_span_ms": span,
            "ii_ms": span / 299, "fps": 299000 / span,
            "completion_interval_ms": _stats(b["completion_ms"] - a["completion_ms"] for a, b in zip(scored, scored[1:])),
            "total_latency_ms": _stats(r["total_latency_ms"] for r in scored),
            "stages": stages, "cpu_overlap": _cpu_overlap(scored, count),
            "queue": _queue_stats(scored), "ddr_attribution": "unknown",
            "cpu_time_scope": sorted({r.get("stage_cpu_time_scope", "unspecified") for r in scored}),
            "pmu_available": all(r.get("pmu_available") is True for r in scored)}


def identity_compatibility(left, right, allow_thread_change=False):
    """Require actual binary/runtime/program identities, independently of labels."""
    reasons = []
    for field in ("topology", "runner_sha256", "runtime_bindings", "program_identity"):
        a, b = left.get(field), right.get(field)
        if not a or not b:
            reasons.append("missing_" + field)
        elif a != b:
            reasons.append("mismatched_" + field)
    if left.get("boot_id") and right.get("boot_id") and left["boot_id"] != right["boot_id"]:
        reasons.append("mismatched_boot_id")
    if not allow_thread_change and left.get("threads") != right.get("threads"):
        reasons.append("mismatched_threads")
    for field in ("numeric_input_sha256", "cpu_frequency_khz"):
        if field in left or field in right:
            if left.get(field) != right.get(field):
                reasons.append("mismatched_" + field)
    if not allow_thread_change:
        for field in ("affinity", "spin_count", "queue_depth", "telemetry", "profile"):
            if left.get(field) != right.get(field):
                reasons.append("mismatched_" + field)
    return {"compatible": not reasons, "reasons": reasons,
            "layers": ["topology", "actual_runner_binary", "runtime_bindings", "program_identity"]}


def compare_services(serial_summary, pipeline_summary):
    if serial_summary.get("kind") != "serial" or pipeline_summary.get("kind") != "pipeline":
        raise ValueError("Comparison requires serial and pipeline summaries")
    if len(serial_summary["stages"]) != len(pipeline_summary["stages"]):
        raise ValueError("Stage count mismatch")
    identity = None
    if "session_identity" in serial_summary or "session_identity" in pipeline_summary:
        identity = identity_compatibility(serial_summary.get("session_identity", {}),
                                          pipeline_summary.get("session_identity", {}))
        if not identity["compatible"]:
            raise ValueError("Incompatible serial/pipeline identity: " + ", ".join(identity["reasons"]))
    differences = []
    for a, b in zip(serial_summary["stages"], pipeline_summary["stages"]):
        differences.append({"index": a["index"], "run_delta_mean_ms": b["run_ms"]["mean"] - a["run_ms"]["mean"],
                            "run_delta_median_ms": b["run_ms"]["median"] - a["run_ms"]["median"],
                            "run_slowdown_mean": b["run_ms"]["mean"] / a["run_ms"]["mean"] if a["run_ms"]["mean"] else None,
                            "inclusive_service_delta_mean_ms": b["busy_ms"]["mean"] - a["busy_ms"]["mean"]})
    vta = [d for d, stage in zip(differences, pipeline_summary["stages"])
           if stage["vta_mutex_held_ms"]["count"]]
    return {"stage_deltas": differences,
            "ii_minus_m1_ms": pipeline_summary["ii_ms"] - serial_summary["m1_ms"],
            "ii_minus_stage_max_ms": pipeline_summary["ii_ms"] - serial_summary.get("stage_service_max_ms", serial_summary["m1_ms"]),
            "resource_lower_bound": serial_summary.get("resource_lower_bound"),
            "R_outer_ms": serial_summary["outer_residual_ms"],
            "R_vta_excess_ms": sum(d["run_delta_mean_ms"] for d in vta) if vta else None,
            "identity_verified": bool(identity and identity["compatible"]),
            "comparison_scope": "matched_identity" if identity else "metrics_only_unverified",
            "ddr_attribution": "unknown",
            "interpretation": "Stage excesses overlap across frames; they are not summed into an II correction."}


def _metrics(session):
    return session.get("metrics", session)


def select_cpu_contrast(sessions):
    """Choose a predeclared pilot contrast by CPU service change, not FPS labels.

    P0->P1/P2/P5 and P3->P4 are the only contrasts.  Each arm must have
    matched serial/pipeline observations in three common blocks on one boot.
    Thread/affinity interventions may change; compiled program identities may not.
    """
    sessions = list(sessions)
    candidates = []
    for control, contrast in (("P0", "P1"), ("P0", "P2"), ("P0", "P5"), ("P3", "P4")):
        arms = {}
        valid = True
        for name in (control, contrast):
            pipe = [s for s in sessions if s.get("condition") == name and _metrics(s).get("kind") == "pipeline"]
            blocks = {s.get("block") for s in pipe}
            if len(pipe) != 3 or len(blocks) != 3:
                valid = False
                break
            for p in pipe:
                serial = [s for s in sessions if s.get("condition") == name and s.get("block") == p.get("block")
                          and s.get("boot_id") == p.get("boot_id") and _metrics(s).get("kind") == "serial"]
                if len(serial) != 1 or not identity_compatibility(serial[0], p)["compatible"]:
                    valid = False
            arms[name] = pipe
        if not valid:
            continue
        a, b = arms[control], arms[contrast]
        if {s.get("block") for s in a} != {s.get("block") for s in b}:
            continue
        if not all(identity_compatibility(x, y, allow_thread_change=True)["compatible"]
                   for x in a for y in b):
            continue
        count = len(_metrics(a[0])["stages"])
        for index in (0, count - 1):
            baseline = statistics.median(_metrics(s)["stages"][index]["run_ms"]["mean"] for s in a)
            intervention = statistics.median(_metrics(s)["stages"][index]["run_ms"]["mean"] for s in b)
            effect = (baseline - intervention) / baseline if baseline else 0.0
            def on_critical_trajectory(arm):
                return all(_metrics(s)["stages"][index].get("busy_fraction", 0) >= 0.90
                           and abs(_metrics(s)["stages"][index]["completion_ii_ms"] -
                                   _metrics(s)["ii_ms"]) <= 0.05 * _metrics(s)["ii_ms"] for s in arm)
            trajectory = on_critical_trajectory(a) or on_critical_trajectory(b)
            if abs(effect) >= 0.05 and trajectory:
                serial_means = {}
                for name in (control, contrast):
                    serial_means[name] = statistics.median(
                        _metrics(s)["stages"][index]["run_ms"]["mean"] for s in sessions
                        if s.get("condition") == name and _metrics(s).get("kind") == "serial"
                        and s.get("boot_id") == a[0].get("boot_id")
                        and s.get("block") in {x.get("block") for x in a})
                if min(serial_means.values()) <= 0:
                    continue
                candidates.append({"baseline": control, "intervention": contrast,
                                   "stage_index": index, "cpu_stage": index,
                                   "baseline_run_ms": baseline, "intervention_run_ms": intervention,
                                   "effect_fraction": effect, "change_fraction": -effect,
                                   "eligible": True, "trajectory_consistent": trajectory,
                                   "baseline_serial_run_ms": serial_means[control],
                                   "intervention_serial_run_ms": serial_means[contrast],
                                   "baseline_parallel_amplification": baseline / serial_means[control],
                                   "intervention_parallel_amplification": intervention / serial_means[contrast],
                                   "selection_metric": "absolute change of median of three block mean CPU run services",
                                   "ambiguity": "Thread changes alter intrinsic serial service too; timing alone cannot separate CPU scheduling, cache and DDR."})
    return max(candidates, key=lambda c: (abs(c["effect_fraction"]),
                                          -int(c["intervention"][1:]), -c["stage_index"])) if candidates else None


def analyze_replicates(records, stage_index=None):
    """Infer stage service and II separately on three paired boot means.

    CPU stage run service is primary when stage_index is supplied globally or in
    each record.  Scalar II-only records remain supported for diagnostic replay.
    Exactly two common paired blocks are required on each independent boot.
    """
    records = list(records)
    choices = {r.get("stage_index", stage_index) for r in records}
    if len(choices) > 1:
        raise ValueError("Replicate stage index mismatch")
    grouped = {}
    seen = set()
    reference_identity = None
    selected_index = next(iter(choices), stage_index)
    for record in records:
        boot, block = record.get("boot_id"), record.get("block")
        if not boot or block is None or (boot, block) in seen:
            raise ValueError("Missing or duplicate boot/block pair")
        seen.add((boot, block))
        index = record.get("stage_index", stage_index)
        if selected_index is None:
            selected_index = index
        if index != selected_index:
            raise ValueError("Replicate stage index mismatch")
        stage_a = stage_b = None
        if "control" in record:
            compatibility = identity_compatibility(record["control"], record["contrast"], allow_thread_change=True)
            if not compatibility["compatible"]:
                raise ValueError("Replicate identity mismatch: " + ",".join(compatibility["reasons"]))
            a, b = _metrics(record["control"])["ii_ms"], _metrics(record["contrast"])["ii_ms"]
            if record["control"].get("boot_id") != boot or record["contrast"].get("boot_id") != boot:
                raise ValueError("Replicate record boot disagrees with sessions")
            for session in (record["control"], record["contrast"]):
                identity = {k: session.get(k) for k in ("topology", "runner_sha256", "runtime_bindings", "program_identity",
                                                         "numeric_input_sha256", "cpu_frequency_khz")}
                if reference_identity is None:
                    reference_identity = identity
                elif identity != reference_identity:
                    raise ValueError("Program/runtime/actual runner differs across replicate boots")
            if index is not None:
                if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                    raise ValueError("Invalid selected CPU stage")
                try:
                    stage_a = _metrics(record["control"])["stages"][index]["run_ms"]["mean"]
                    stage_b = _metrics(record["contrast"])["stages"][index]["run_ms"]["mean"]
                except (IndexError, KeyError):
                    raise ValueError("Selected CPU stage service unavailable") from None
        else:
            a, b = record.get("control_ii_ms"), record.get("contrast_ii_ms")
            if index is not None:
                stage_a, stage_b = record.get("control_stage_run_ms"), record.get("contrast_stage_run_ms")
        a, b = _number(a, "control II"), _number(b, "contrast II")
        if min(a, b) <= 0:
            raise ValueError("II must be positive")
        if index is not None:
            stage_a = _number(stage_a, "control CPU service")
            stage_b = _number(stage_b, "contrast CPU service")
            if min(stage_a, stage_b) <= 0:
                raise ValueError("CPU service must be positive")
        grouped.setdefault(boot, []).append((block, a, b, stage_a, stage_b))
    if len(grouped) != 3 or any(len(values) != 2 for values in grouped.values()):
        raise ValueError("Three independent boots and exactly two paired blocks per boot required")
    block_sets = [set(x[0] for x in v) for v in grouped.values()]
    if any(x != block_sets[0] for x in block_sets[1:]):
        raise ValueError("Replicate block support mismatch")
    boots = []
    for boot, values in sorted(grouped.items()):
        a = statistics.mean(v[1] for v in values)
        b = statistics.mean(v[2] for v in values)
        row = {"boot_id": boot, "control_ii_ms": a, "contrast_ii_ms": b,
               "delta_ii_ms": a - b, "ii_effect_fraction": (a - b) / a}
        if selected_index is not None:
            sa = statistics.mean(v[3] for v in values)
            sb = statistics.mean(v[4] for v in values)
            row.update(control_stage_run_ms=sa, contrast_stage_run_ms=sb,
                       delta_stage_run_ms=sa - sb, stage_effect_fraction=(sa - sb) / sa)
        boots.append(row)
    def inference(delta_field, effect_field):
        deltas = [b[delta_field] for b in boots]
        mean = statistics.mean(deltas)
        half = T95_DF2 * statistics.stdev(deltas) / math.sqrt(3)
        lower, upper = mean - half, mean + half
        effects = [b[effect_field] for b in boots]
        direction = "improvement" if all(e >= 0.05 for e in effects) else (
            "regression" if all(e <= -0.05 for e in effects) else "inconsistent_or_below_5_percent")
        excludes_zero = lower > 0 or upper < 0
        return {"delta_mean_ms": mean,
                "mean_ci95": {"lower": lower, "upper": upper, "degrees_of_freedom": 2,
                               "critical_t": T95_DF2, "method": "paired_boot_student_t_95"},
                "ci95_excludes_zero": excludes_zero,
                "same_direction_at_least_5_percent": direction != "inconsistent_or_below_5_percent",
                "direction": direction, "gate_passed": excludes_zero and direction != "inconsistent_or_below_5_percent"}
    ii = inference("delta_ii_ms", "ii_effect_fraction")
    stage = inference("delta_stage_run_ms", "stage_effect_fraction") if selected_index is not None else None
    primary = stage or ii
    return {"independent_boot_count": 3, "statistical_unit": "paired boot mean, not frames or blocks",
            "primary_metric": "cpu_stage_run_mean_ms" if stage else "diagnostic_ii_ms",
            "identity_verified": reference_identity is not None,
            "stage_index": selected_index, "boot_rows": boots,
            "stage_service": stage, "ii": ii,
            "stage_run_ms_mean_ci95": stage["mean_ci95"] if stage else None,
            "stage_gate_passed": stage["gate_passed"] if stage else None,
            "delta_ii_ms_mean": ii["delta_mean_ms"], "delta_ii_ms_mean_ci95": ii["mean_ci95"],
            "ii_ms_mean_ci95": ii["mean_ci95"], "ii_gate_passed": ii["gate_passed"],
            "ci95_excludes_zero": primary["ci95_excludes_zero"],
            "same_direction_at_least_5_percent": primary["same_direction_at_least_5_percent"],
            "direction": primary["direction"], "gate_passed": primary["gate_passed"],
            "ddr_attribution": "unknown"}


def _read_json(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _bindings(source):
    script = (source / "raw/c1_boot3_restore.sh").read_text()
    result = {}
    for name in ("libtvm_runtime.so", "libvta.so"):
        match = re.search(r"sha256sum " + re.escape(name) + r" .*?= ([0-9a-f]{64})", script)
        if not match:
            raise ValueError("Restore script lacks runtime binding: " + name)
        result[name] = match.group(1)
    if _sha(source / "raw/condition_packages/B/libtvm_runtime.so") != result["libtvm_runtime.so"]:
        raise ValueError("Packaged runtime binary differs from restore binding")
    result["evidence"] = "frozen successful restore script; not per-session /proc loader attestation"
    return result


def _prior_component_evidence():
    """Small archived controls, explicitly separate from the new campaign."""
    directory = Path(__file__).parent / "report_out/resource_aware_maxplus"
    result = {"scope": "archived controlled evidence; not new campaign measurements"}
    runtime = directory / "v1_p7b2_runtime_qualification_summary.json"
    boundary = directory / "v1_p8c_cross_boot_summary.json"
    if runtime.exists():
        data = _read_json(runtime)
        result["runtime"] = {"source": str(runtime.resolve()), "sha256": _sha(runtime),
                             "boot_id": data["board_boot_id"], "host_queue_floor": data["host_queue_floor"],
                             "poll_policy_deltas": data["poll_policy_deltas"],
                             "poll_owned_by_inclusive_run": data["poll_reason"]}
    if boundary.exists():
        data = _read_json(boundary)
        result["boundary"] = {"source": str(boundary.resolve()), "sha256": _sha(boundary),
                              "candidate_id": data["candidate_id"], "independent_boot_count": data["independent_boot_count"],
                              "aggregate": data["aggregate"], "gates": data["gates"]}
    return result


def _existing_session(source, name, runner, bindings):
    directory = source / "raw/native_sessions" / name
    contract = _read_json(directory / "session_contract.json")
    if contract.get("runs") != (307 if contract.get("mode") == "window" else 20):
        raise ValueError("Unexpected existing session run contract: " + name)
    expected_binary = source / "raw" / runner["name"]
    if _sha(expected_binary) != runner["sha256"]:
        raise ValueError("Actual runner binary drifted")
    if Path(contract["argv"][0]).name != runner["name"]:
        raise ValueError("Session actual runner differs from selected binary")
    program = []
    for stage in contract["stages"]:
        identity = {"name": stage["name"], "device": stage["device"], "files": {}}
        for file in ("graph.json", "module.so", "params.bin"):
            expected = stage["files"][file]["sha256"]
            if _sha(source / "raw" / stage["directory"] / file) != expected:
                raise ValueError("Program identity drift: " + stage["name"] + "/" + file)
            identity["files"][file] = expected
        program.append(identity)
    rows = _read_rows(directory / "result.jsonl")
    if len(rows) != contract["runs"] or contract.get("warmup_rows") != 5:
        raise ValueError("Existing rows/warmup contract mismatch")
    if contract.get("numeric_input_sha256") != _sha(source / "raw/inputs/input.npy"):
        raise ValueError("Existing numeric input identity mismatch")
    mode = "pipeline" if contract["mode"] == "window" else "serial"
    metrics = analyze_pipeline(rows) if mode == "pipeline" else analyze_serial(rows)
    parts = name.split("_")
    session = {"boot_id": contract["boot_id"], "boot_index": contract["boot_index"],
               "condition": parts[2], "mode": mode, "block": int(parts[1].removeprefix("block")),
               "topology": [s["name"] for s in contract["stages"]], "threads": contract["threads"],
               "runner_sha256": runner["sha256"], "runtime_bindings": bindings,
               "numeric_input_sha256": contract["numeric_input_sha256"],
               "program_identity": program, "metrics": metrics, "source_session": name}
    metrics["session_identity"] = {key: session[key] for key in (
        "boot_id", "topology", "threads", "runner_sha256", "runtime_bindings", "program_identity", "numeric_input_sha256")}
    return session


def analyze_existing(source_dir=DEFAULT_SOURCE):
    """Strictly replay existing final A/B data; never alter or re-label old runs."""
    source = Path(source_dir).resolve()
    summary = _read_json(source / "summary.json")
    if summary.get("status") != "SINGLE_IMAGE_SMOKE_PASS" or summary.get("A_B_same_program_identity") is not True:
        raise ValueError("Source audit is not a passed same-program single-image smoke")
    if summary.get("new_board_calls_in_this_directory") != 0 or summary.get("audit_only_recovery") is not True:
        raise ValueError("Expected sealed read-only recovery provenance")
    lineage = _read_json(source / "analysis/selection_deployment_lineage.json")
    runner = lineage["final_deployment_runner"]
    selection_runner = lineage["source_selection_runner"]
    if _sha(source / "raw" / selection_runner["name"]) != selection_runner["sha256"]:
        raise ValueError("Original selection runner binary drifted")
    if len(set(summary.get("C1_boot_ids", []))) != 3 or len(summary.get("final_ablation_boot_ids", [])) != 1:
        raise ValueError("Source C1/final boot evidence was pooled or malformed")
    bindings = _bindings(source)
    sessions = []
    pairs = []
    for condition in ("A", "B"):
        for block in (0, 1):
            a = _existing_session(source, f"b3_block{block}_{condition}_serial", runner, bindings)
            b = _existing_session(source, f"b3_block{block}_{condition}_window", runner, bindings)
            compatibility = identity_compatibility(a, b)
            if not compatibility["compatible"]:
                raise ValueError("Existing paired identity mismatch")
            if a["boot_id"] not in summary["final_ablation_boot_ids"]:
                raise ValueError("Final session boot identity disagrees with source summary")
            sessions.extend((a, b))
            pairs.append({"condition": condition, "block": block, "identity": compatibility,
                          "comparison": compare_services(a["metrics"], b["metrics"])})
            published = [p for p in summary.get("performance", []) if p.get("condition") == condition]
            if len(published) != 1 or len(published[0].get("ii_seconds_blocks", [])) != 2 or not math.isclose(
                    b["metrics"]["ii_ms"], published[0]["ii_seconds_blocks"][block] * 1000, abs_tol=1e-7):
                raise ValueError("Existing raw window disagrees with audited summary")
    if not identity_compatibility(sessions[0], sessions[4])["compatible"]:
        raise ValueError("Source A/B program/runtime identity mismatch")
    component_profile = _read_json(source / "analysis/c1_component_profile.json")
    for component in component_profile["observations"]:
        if _sha(source / component["raw_rows"]) != component["raw_rows_sha256"]:
            raise ValueError("Imported component rows drifted")
        metric = analyze_serial(_read_rows(source / component["raw_rows"]),
                                discard=component_profile["discard_initial_calls"])
        if (component.get("all_frontier_outputs_passed") is not True or
                metric["raw_row_count"] != component["numeric_calls"] or
                metric["scored_rows"] != component_profile["scored_calls_per_component"] or
                not math.isclose(metric["stages"][0]["busy_ms"]["median"], component["service_ms"], abs_tol=1e-7)):
            raise ValueError("Component preparation contract disagrees with raw timing")
    proxy = _read_json(source / "raw/c1_frozen/fair_support_contract.json")
    if proxy.get("label_free") is not True or proxy.get("actual_policy") != "ordinary_copy" or any(
            row["full_identity"]["runner_sha256"] != selection_runner["sha256"] for row in proxy["support"]):
        raise ValueError("Original label-free proxy identity mismatch")
    return {"source_run": str(source), "new_board_calls": 0, "sessions": sessions,
            "prior_evidence": _prior_component_evidence(),
            "frozen_proxy": {"support": proxy["support"], "runner": selection_runner,
                             "scope": "identity-bound historical component diagnostic; no fitting to new II labels"},
            "existing_replay": pairs,
            "identity_audit": {"final_runner": runner, "selection_runner": lineage["source_selection_runner"],
                               "runtime_bindings": bindings, "source_candidate_ids": [lineage["source_A_candidate_id"], lineage["source_B_candidate_id"]],
                               "C1_component_boot": component_profile["boot_id"],
                               "C1_selection_boots": summary["C1_boot_ids"],
                               "final_ablation_boots": summary["final_ablation_boot_ids"],
                               "selection_and_final_measurements_pooled": False},
            "existing_components": component_profile["observations"],
            "missing_measurements": ["pipeline per-thread CPU/PMU attribution", "CPU t2/t3 if those interventions are selected",
                                     "independent matched boots for the selected CPU contrast"],
            "recommendations": ["Reuse same-boot final serial/pipeline replay before new profiles.",
                                "Select a CPU-service contrast in the frozen pilot, then replicate only that contrast."],
            "ddr_attribution": "unknown", "physical_ddr_bandwidth": None,
            "claim_boundary": "Observation of service and overlap; no pure DDR or universal fixed-intercept attribution."}


def write_report(path, analysis):
    """Write a short Chinese report beside new analysis artifacts only."""
    lines = ["# CPU–VTA 推理残差验证", "", "以下忙碌时间、空洞和等待来自时间戳；共享 DDR 的物理归因仍为未知。", ""]
    historical = analysis.get("historical_e0", analysis if "existing_replay" in analysis else {})
    if historical.get("source_run"):
        lines.extend(["已有数据来源：`" + historical["source_run"] + "`。只读重放未新增板端调用。", ""])
    lines.extend(["| 判断 | 可能来源 | 支持范围 |", "|---|---|---|",
                  "| 已验证观察层 | 当前代表的 CPU 串行与流水服务差异 | 同启动、同实际 runner、同程序可直接比较；尚需干预启动复现形成因果证据 |",
                  "| 不支持全局常量 | 统一 +34 ms | 历史诊断不能迁移为新程序/线程配置的物理固定截距 |",
                  "| 已计入、需避免重复 | VTA LOAD/STORE、调用与等待 | inclusive run 已拥有这些服务；driver/poll 是分解证据 |",
                  "| 未闭合 | 满载队列/slot 背压 | 有时间戳才报告；可与其他工作重叠，不能直接相加 |",
                  "| 未闭合 | CPU 核调度、cache 与共享 DDR 各自贡献 | 流水阶段 process CPU time 不归属单个 stage；逻辑字节不等于物理 DDR 流量 |", ""])
    prior = historical.get("prior_evidence", analysis.get("prior_evidence", {}))
    if prior.get("runtime"):
        runtime = prior["runtime"]
        queue = runtime["host_queue_floor"]
        poll = "/".join(f"{p['run_ms_delta']:.3f}" for p in runtime["poll_policy_deltas"])
        lines.extend([f"已排除作为约 34 ms 的主要直接项：历史 P7B-2 空载 1/3 actor 链仅 {queue['one_actor_ms']:.3f}/{queue['three_actor_ms']:.3f} ms/帧；short/long 的 poll 策略增量为 {poll} ms，且已包含于 inclusive run。", "",
                      "该排除只针对冻结路径的空载 floor 与 poll 增量；不排除饱和队列调度反馈。来源：`" + runtime["source"] + "`。", ""])
    if prior.get("boundary"):
        evidence = prior["boundary"]
        a = evidence["aggregate"]
        interval = a["delta_ii_ms_mean_ci95"]
        lines.extend([f"已验证边界机制干预：历史 P8C 三启动代表的框架物化 {a['framework_materialization_bytes_B0_per_frame']}→0 B/帧，边界 API 减少 {a['boundary_api_service_reduction_ms_mean']:.6f} ms（{a['boundary_api_service_reduction_percent_mean']:.2f}%）；II 改善 95% CI [{interval['lower']:.6f}, {interval['upper']:.6f}] ms 跨零。", "",
                      "不支持该代表的框架物化是 34 ms 残差主要直接来源；小幅收益、协议代价抵消与其他拓扑影响仍未排除，VTA 内部 LOAD/STORE 保留。来源：`" + evidence["source"] + "`。", ""])
    if historical.get("identity_audit"):
        identities = historical["identity_audit"]
        lines.extend(["旧 C1 proxy runner：`" + identities["selection_runner"]["sha256"] + "`；旧最终代表 runner：`" + identities["final_runner"]["sha256"] + "`。", "",
                      "原组件/选择与最终代表分别保留启动和 runner 身份，不能合并成新 runner 的三次启动证据。", ""])
    telemetry = analysis.get("telemetry_gate")
    if telemetry:
        changes = "/".join(f"{v * 100:+.3f}%" for v in telemetry.get("paired_relative_changes", []))
        enabled = telemetry.get("primary_telemetry") is True
        lines.extend([f"附加遥测开关：主测量{'启用' if enabled else '关闭'} queue-timing 与 defer-result-write；同二进制平衡开/关 II 变化为 {changes or '未知'}，门槛为 ±2%。", "",
                      "遥测关闭时队列内部等待保持未知；当前附加 VTA profiler 是否触发：" + ("是" if analysis.get("vta_profile_triggered") else "否") + "。", ""])
    sessions = [s for s in analysis.get("sessions", []) if _metrics(s).get("kind") in ("serial", "pipeline")]
    if sessions:
        lines.extend(["| 条件/块 | 模式 | 样本 | II 或串行时延 (ms) |", "|---|---|---:|---:|"])
        for session in sessions:
            m = _metrics(session)
            value = m.get("ii_ms") if m.get("kind") == "pipeline" else m["total_latency_ms"]["mean"]
            lines.append(f"| {session.get('condition')}/{session.get('block')} | {m.get('kind')} | {m['scored_rows']} | {value:.6f} |")
        lines.append("")
    pipeline_sessions = [s for s in sessions if _metrics(s).get("kind") == "pipeline"]
    if pipeline_sessions:
        proxy = analysis.get("frozen_proxy", historical.get("frozen_proxy", {}))
        support = proxy.get("support", [])
        lines.extend(["| 条件/块 | 线程 | CPU0/VTA/CPU末 run 均值 (ms) | 旧 proxy (ms) | 同块 stage max (ms) | 同块资源 M1 (ms) | 实际 II (ms) | CPU0 忙碌占比 | 遥测 |",
                      "|---|---|---|---:|---:|---:|---:|---:|---|"])
        for p in pipeline_sessions:
            m = _metrics(p)
            serial = [s for s in sessions if s.get("condition") == p.get("condition") and s.get("block") == p.get("block")
                      and s.get("boot_id") == p.get("boot_id") and s.get("topology") == p.get("topology")
                      and _metrics(s).get("kind") == "serial"]
            m1 = _metrics(serial[0])["m1_ms"] if len(serial) == 1 and identity_compatibility(serial[0], p)["compatible"] else None
            stage_max = _metrics(serial[0])["stage_service_max_ms"] if m1 is not None else None
            topology = p.get("topology")
            topology_id = "vta_" + topology.replace(":", "_") if isinstance(topology, str) else next(
                (str(n) for n in topology or [] if str(n).startswith("vta_")), None)
            matches = [r for r in support if r.get("topology_id") == topology_id and r.get("thread_vector") == p.get("threads")]
            old = f"{matches[0]['m1_score_ms']:.6f}" if len(matches) == 1 else "未支持"
            service = "/".join(f"{stage['run_ms']['mean']:.3f}" for stage in m["stages"])
            thread = "/".join(map(str, p.get("threads", [])))
            lines.append(f"| {p.get('condition')}/{p.get('block')} | {thread} | {service} | {old} | " +
                         (f"{stage_max:.6f}" if stage_max is not None else "未匹配") + " | " +
                         (f"{m1:.6f}" if m1 is not None else "未匹配") +
                         f" | {m['ii_ms']:.6f} | {m['stages'][0]['busy_fraction'] * 100:.2f}% | " + ("开" if p.get("telemetry") else "关") + " |")
        lines.extend(["", "旧 proxy 仅列原 runner/线程域的诊断值；与新实际 runner 身份不兼容，不作为新测量的前瞻预测。新 M1 取同块 inclusive stage 均值最大值与独占 process CPU 总均值/4 的最大值；串行资源需求向并发迁移只是下界假设，affinity/Hall 与 DDR 下界仍未知。", ""])
    selection = analysis.get("selection")
    if selection:
        lines.extend([f"冻结 pilot 选择：{selection['baseline']}→{selection['intervention']}，CPU 阶段 {selection['stage_index']}，服务变化 {selection['change_fraction'] * 100:+.3f}%。", "",
                      f"串行归一化并发放大：{selection['baseline_parallel_amplification']:.4f}→{selection['intervention_parallel_amplification']:.4f}；线程改变也会改变自身串行服务。", ""])
    elif "selection" in analysis:
        lines.extend(["pilot 未找到同时满足 ≥5% 服务变化与关键轨迹一致性的 CPU 对照；不强行选择或扩大因果主张。", ""])
    for pair in historical.get("existing_replay", []):
        c = pair["comparison"]
        lines.extend([f"条件 {pair['condition']}、块 {pair['block']}：II 减去同块串行资源下界为 {c['ii_minus_m1_ms']:.6f} ms；减去最大阶段均值为 {c['ii_minus_stage_max_ms']:.6f} ms。", "",
                      "| 阶段 | 流水/串行 run 均值比 | run 均值增量 (ms) |", "|---|---:|---:|"])
        for stage in c["stage_deltas"]:
            ratio = stage["run_slowdown_mean"]
            lines.append(f"| {stage['index']} | {ratio:.4f} | {stage['run_delta_mean_ms']:.6f} |" if ratio is not None else f"| {stage['index']} | 未知 | {stage['run_delta_mean_ms']:.6f} |")
        lines.append("")
    replicate = analysis.get("replicates", analysis if "boot_rows" in analysis else None)
    replicates = analysis.get("replication", {})
    if replicate:
        replicates = dict(replicates, selected=replicate)
    if "replication" not in analysis and not replicate and "independent_boot_count" in analysis:
        lines.extend([f"当前独立启动数：{analysis['independent_boot_count']}；三启动因果门尚未闭合。", ""])
    for topology, replicate in replicates.items():
        lines.extend(["对照拓扑：`" + str(topology) + "`。", ""])
        interval = replicate["delta_ii_ms_mean_ci95"]
        if replicate.get("stage_service"):
            service = replicate["stage_service"]
            ci = service["mean_ci95"]
            lines.extend([f"CPU 阶段 {replicate['stage_index']} 的配对 run 服务差值为 {service['delta_mean_ms']:.6f} ms；95% CI [{ci['lower']:.6f}, {ci['upper']:.6f}] ms。", "",
                          "阶段服务验收门：" + ("通过" if service["gate_passed"] else "未通过") + "；II 是独立次指标，阶段效应不要求吞吐同时改善。", ""])
        lines.extend([f"三次独立启动配对 II 差值均值：{replicate['delta_ii_ms_mean']:.6f} ms；95% CI [{interval['lower']:.6f}, {interval['upper']:.6f}] ms。", "",
                      "II 验收门：" + ("通过" if replicate.get("ii_gate_passed", replicate["gate_passed"]) else "未通过") + "。统计单位是启动，不是帧或同次启动内的块。", ""])
    lines.extend(["阶段增量可能跨帧重叠，队列/slot 背压也可能重叠，不能将它们相加作为 II 修正。", "",
                  "C1 原选择、实际 runner 二进制、程序与 runtime 库分别保留身份；旧孤立数据不升级为新 runner 的多启动证据。", ""])
    destination = Path(path)
    if analysis.get("source_run"):
        source = Path(analysis["source_run"]).resolve()
        if destination.resolve() == source or source in destination.resolve().parents:
            raise ValueError("Report output must be outside the sealed source run")
    if historical.get("source_run"):
        source = Path(historical["source_run"]).resolve()
        if destination.resolve() == source or source in destination.resolve().parents:
            raise ValueError("Report output must be outside the sealed historical source run")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    return str(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    source, output = args.source_run.resolve(), args.output_dir.resolve()
    if output == source or source in output.parents:
        raise ValueError("Analysis output must be outside the sealed source run")
    if output.exists():
        raise ValueError("Output directory already exists")
    analysis = analyze_existing(source)
    output.mkdir(parents=True)
    (output / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n")
    write_report(output / "REPORT.md", analysis)
    print(json.dumps({"output_dir": str(output), "new_board_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
