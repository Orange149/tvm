"""Sparse, run-scoped per-TID schedstat diagnostics; no model fitting.

The native probe brackets each selected CPU run with per-thread /proc reads.
Every delta therefore has a measured read envelope: it is neither an exact
instruction counter nor a decomposition of wall time into additive causes.
"""
import json
from pathlib import Path
import statistics

from ..diagnosis.analysis import _read, _sha, _number, _csv, _bound_record

TARGETS = {"cpu_00_02", "cpu_05_07"}
CASE_NAMES = {"static_first": "静态第一名：两段四线程共享核心",
              "history_host2": "历史参考：三线程与单线程核组、主机固定核心2"}
COUNTERS = ("runtime_ns", "runqueue_wait_ns", "timeslices")
WARNINGS = [
    "CPU运行时间来自Linux每TID的schedstat，不是独立pthread CPU时钟或PMU。",
    "每个TID的前后读取时刻不同，增量覆盖run及读取包络内的少量外侧工作；原始读取上下界全部保留。",
    "Linux调度统计可能延迟记账；读取墙钟包络不是严格的调度计数误差上界，不进行精确误差修正。",
    "CPU运行时间包含计算、屏障／同步、自旋或yield调用、缓存／访存等待等在核活动，不等于有效算术工作。",
    "多个线程的运行时间可以超过run墙钟时间，禁止用wall减去线程CPU时间之和作为等待。",
    "运行队列等待仅表示可运行线程等待CPU；不代表全部同步、队列或访存等待。",
    "固定池中的TID及逻辑worker与资格记录绑定；缺失、回退或负增量记为不可用，不填零。",
    "流水探针仅采样帧8、12、16、20；其统计不代表全部16个计时帧的精确均值。",
    "每个配置只在本次启动运行，桥接差异和探针扰动均为观察值，不触发自动拒收或加测。",
    "本轮不改变原profile、公式、DP或Top-20，不把流水测量回填为新静态成本。",
]


def _finite(value, name):
    # Relative steady-clock times may have tiny negative origin rounding.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid probe timestamp: " + name)
    import math
    if not math.isfinite(value):
        raise ValueError("nonfinite probe timestamp: " + name)
    return value


def _sample(value, name):
    if not isinstance(value, dict) or type(value.get("available")) is not bool:
        raise ValueError("invalid per-TID sample availability: " + name)
    begin = _finite(value.get("read_start_ms"), name + " read start")
    end = _finite(value.get("read_end_ms"), name + " read end")
    if end < begin - 1e-6:
        raise ValueError("inverted per-TID read envelope")
    if value["available"]:
        for counter in COUNTERS:
            _number(value.get(counter), name + " " + counter, integer=True)
    elif any(value.get(counter) is not None for counter in COUNTERS):
        raise ValueError("unavailable per-TID counters must be null, not zero")
    reason = value.get("reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("invalid per-TID sample reason")
    return value


def validate_probe(trace, task, affinity, rows, sched_schedstats=None):
    """Validate bound native trace and normalize its nullable counter deltas.

    Schema failures, wrong TIDs/slots/frames and impossible timestamp ordering
    are invalid execution evidence. Explicitly unavailable reads or monotonicity
    failures remain diagnostic unknowns, preserving their raw reasons.
    """
    if task.get("run_probe") is not True:
        raise ValueError("run-scope trace belongs to a task without probe")
    if not isinstance(trace, dict) or trace.get("schema") != "vta_run_scope_v1":
        raise ValueError("invalid run-scope trace schema")
    if trace.get("clock") != "steady_clock_ms_relative_to_benchmark_origin":
        raise ValueError("run-scope trace clock differs")
    _number(trace.get("benchmark_origin_ms"), "benchmark origin")
    requested = task.get("probe_frames")
    if (not isinstance(requested, list) or any(type(frame) is not int or frame < 0 for frame in requested)
            or requested != sorted(set(requested)) or trace.get("requested_frames") != requested):
        raise ValueError("run-scope requested frame identity differs")
    by_frame = {row["frame_id"]: row for row in rows}
    if len(by_frame) != len(rows) or not set(requested) <= set(by_frame):
        raise ValueError("run-scope raw frames missing or duplicated")
    cfg = task["configuration"]
    expected = {(frame, i) for frame in requested for i, stage in enumerate(cfg["stages"])
                if stage["device"] == "cpu" and stage["program"] in TARGETS}
    records = trace.get("records")
    if not isinstance(records, list) or not expected:
        raise ValueError("run-scope target coverage is empty or malformed")
    if len(affinity.get("stages", [])) != len(cfg["stages"]):
        raise ValueError("run-scope affinity stage coverage differs")
    seen, sample_rows, thread_rows, warnings = set(), [], [], []
    kernel_enabled = sched_schedstats in (1, "1") and sched_schedstats is not True
    qualification = task["kind"].startswith("qualify")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid run-scope record")
        frame, index = record.get("frame_id"), record.get("stage_index")
        if type(frame) is not int or type(index) is not int or (frame, index) not in expected or (frame, index) in seen:
            raise ValueError("run-scope frame/stage duplicate or outside target domain")
        seen.add((frame, index))
        stage, raw = cfg["stages"][index], by_frame[frame]
        if record.get("program") != stage["program"]:
            raise ValueError("run-scope stage program differs")
        points = {key: _finite(record.get(key), key) for key in
                  ("before_start_ms", "before_end_ms", "run_start_ms", "run_end_ms", "after_start_ms", "after_end_ms")}
        ordered = [points[key] for key in ("before_start_ms", "before_end_ms", "run_start_ms", "run_end_ms", "after_start_ms", "after_end_ms")]
        if any(b < a - 1e-6 for a, b in zip(ordered, ordered[1:])):
            raise ValueError("run-scope boundary partial order differs")
        raw_start, raw_end = raw[f"stage{index}_run_start_ms"], raw[f"stage{index}_run_end_ms"]
        if points["before_start_ms"] < raw_start - 1e-5 or points["after_end_ms"] > raw_end + 1e-5:
            raise ValueError("diagnostic run and reads escaped legacy run boundaries")
        if abs((raw_end - raw_start) - raw[f"stage{index}_run_ms"]) > 1e-5:
            raise ValueError("legacy run wall-time scope differs")
        native = affinity["stages"][index]
        ownership = {entry["tid"]: entry for entry in native["effective_affinities"]}
        threads = record.get("threads")
        if not isinstance(threads, list) or len(threads) != stage["threads"]:
            raise ValueError("run-scope logical worker coverage differs")
        tids, slots, current = set(), set(), []
        for thread in threads:
            slot, tid = thread.get("logical_worker"), thread.get("tid")
            if (type(slot) is not int or not 0 <= slot < stage["threads"] or slot in slots
                    or type(tid) is not int or tid <= 0 or tid in tids or tid not in ownership):
                raise ValueError("run-scope worker slot/TID identity differs")
            slots.add(slot)
            tids.add(tid)
            core = stage["worker_core_map"][slot]
            role = "caller" if slot == 0 else "worker"
            if (thread.get("core") != core or type(thread.get("core")) is not int
                    or ownership[tid]["cpus"] != [core] or thread.get("role") != role
                    or ((tid == native["caller_tid"]) != (slot == 0))):
                raise ValueError("run-scope logical worker/core/caller mapping differs")
            before, after = (_sample(thread.get(key), key) for key in ("before", "after"))
            for value, side in ((before, "before"), (after, "after")):
                if (value["read_start_ms"] < points[side + "_start_ms"] - 1e-6
                        or value["read_end_ms"] > points[side + "_end_ms"] + 1e-6):
                    raise ValueError("per-TID read escaped stage probe envelope")
            available = before["available"] and after["available"]
            deltas = {key: after[key] - before[key] for key in COUNTERS} if available else None
            monotonic = all(delta >= 0 for delta in deltas.values()) if available else None
            if thread.get("monotonic") is not monotonic:
                raise ValueError("native counter monotonicity flag differs from raw values")
            for counter, field in (("runtime_ns", "runtime_delta_ns"), ("runqueue_wait_ns", "runqueue_wait_delta_ns"), ("timeslices", "timeslices_delta")):
                expected_delta = deltas[counter] if monotonic else None
                if thread.get(field) != expected_delta or (expected_delta is not None and type(thread.get(field)) is not int):
                    raise ValueError("run-scope delta differs from endpoint counters")
            fresh = thread.get("fresh_check")
            if not isinstance(fresh, dict) or fresh.get("performed") is not qualification:
                raise ValueError("qualification fresh-FD crosscheck coverage differs")
            if qualification:
                value = _sample(fresh.get("sample"), "fresh")
                if (value["read_start_ms"] < after["read_end_ms"] - 1e-6
                        or value["read_start_ms"] < points["after_start_ms"] - 1e-6
                        or value["read_end_ms"] > points["after_end_ms"] + 1e-6):
                    raise ValueError("fresh-FD read escaped post-run probe envelope")
                compatible = value["runtime_ns"] >= after["runtime_ns"] if value["available"] and after["available"] else None
                if fresh.get("cached_not_stale") is not compatible:
                    raise ValueError("fresh-FD crosscheck flag differs from endpoint counters")
                if compatible is False:
                    warnings.append("fresh_open_counter_precedes_cached_read")
            elif fresh.get("sample") is not None or fresh.get("cached_not_stale") is not None:
                raise ValueError("nonqualification task contains unexpected fresh-FD measurement")
            row = {"frame_id": frame, "stage_index": index, "program": stage["program"],
                   "threads": stage["threads"], "logical_worker": slot, "tid": tid, "core": core, "role": role,
                   "endpoint_reads_available": available, "monotonic": monotonic,
                   "runtime_delta_ns": thread.get("runtime_delta_ns"),
                   "runqueue_wait_delta_ns": thread.get("runqueue_wait_delta_ns"),
                   "timeslices_delta": thread.get("timeslices_delta"),
                   "runtime_ms": deltas["runtime_ns"] / 1e6 if monotonic else None,
                   "runqueue_wait_ms": deltas["runqueue_wait_ns"] / 1e6 if monotonic else None,
                   "before_read_start_ms": before["read_start_ms"], "before_read_end_ms": before["read_end_ms"],
                   "after_read_start_ms": after["read_start_ms"], "after_read_end_ms": after["read_end_ms"],
                   "outside_run_read_envelope_ms": (points["run_start_ms"] - before["read_start_ms"])
                        + (after["read_end_ms"] - points["run_end_ms"]),
                   "fresh_check_performed": qualification, "fresh_check_consistent": fresh.get("cached_not_stale"),
                   "before_unavailable_reason": before.get("reason"), "after_unavailable_reason": after.get("reason")}
            current.append(row)
        if tids != set(native["thread_ids"]):
            raise ValueError("run-scope threads differ from persistent stage pool")
        caller = next(thread for thread in threads if thread["role"] == "caller")
        helpers = [thread for thread in threads if thread["role"] == "worker"]
        if any(thread["before"]["read_end_ms"] > caller["before"]["read_start_ms"] + 1e-6
               or thread["after"]["read_start_ms"] < caller["after"]["read_end_ms"] - 1e-6 for thread in helpers):
            raise ValueError("caller must be read last before run and first after run")
        all_available = all(row["monotonic"] is True for row in current)
        growth = sum(row["runtime_delta_ns"] for row in current) > 0 if all_available else None
        if record.get("runtime_growth_observed") is not growth:
            raise ValueError("run-scope growth flag differs from per-TID deltas")
        if not all_available:
            warnings.append("unavailable_or_nonmonotonic_thread_counter")
        if growth is False:
            warnings.append("zero_runtime_growth_cannot_establish_absence_of_CPU_work")
        runtime_available = all_available and growth is True
        wait_available = all_available and (kernel_enabled or any(row["runqueue_wait_delta_ns"] > 0 for row in current))
        if not wait_available:
            warnings.append("zero_runqueue_wait_without_enabled_accounting_is_unknown")
        for row in current:
            if not runtime_available:
                row["runtime_ms"] = None
            if not wait_available:
                row["runqueue_wait_ms"] = None
            row["runtime_available"] = runtime_available
            row["runqueue_wait_available"] = wait_available
        thread_rows.extend(current)
        runtime = sum(row["runtime_delta_ns"] for row in current) / 1e6 if runtime_available else None
        diagnostic_run = points["run_end_ms"] - points["run_start_ms"]
        sample_rows.append({"frame_id": frame, "stage_index": index, "program": stage["program"],
                            "threads": stage["threads"], "mask": ",".join(map(str, stage["mask"])), **points,
                            "legacy_run_ms": raw[f"stage{index}_run_ms"], "diagnostic_run_ms": diagnostic_run,
                            "before_probe_ms": points["before_end_ms"] - points["before_start_ms"],
                            "after_probe_ms": points["after_end_ms"] - points["after_start_ms"],
                            "outside_diagnostic_run_ms": raw[f"stage{index}_run_ms"] - diagnostic_run,
                            "cpu_runtime_ms": runtime,
                            "runqueue_wait_ms": sum(row["runqueue_wait_delta_ns"] for row in current) / 1e6 if wait_available else None,
                            "runtime_available": runtime_available, "runqueue_wait_available": wait_available,
                            "runtime_growth_observed": growth,
                            "worker_runtime_min_ms": min(row["runtime_ms"] for row in current) if runtime_available else None,
                            "worker_runtime_max_ms": max(row["runtime_ms"] for row in current) if runtime_available else None,
                            "largest_worker_runtime_share": max(row["runtime_ms"] for row in current) / runtime if runtime_available else None,
                            "runtime_per_wall_ratio": runtime / diagnostic_run if runtime_available and diagnostic_run else None})
    if seen != expected:
        raise ValueError("run-scope target frame/stage coverage incomplete")
    return {"schema": "validated_vta_run_scope_v1", "sample_count": len(sample_rows),
            "thread_sample_count": len(thread_rows), "requested_frames": requested,
            "source": "linux_per_TID_schedstat_not_independent_CPU_clock",
            "samples": sample_rows, "threads": thread_rows,
            "all_runtime_samples_available": all(row["runtime_available"] for row in sample_rows),
            "all_wait_samples_available": all(row["runqueue_wait_available"] for row in sample_rows),
            "warnings": sorted(set(warnings)), "scope_warnings": WARNINGS}


def _bound_trace(root, record, rows, affinity, kernel):
    doc = record["document"]
    receipt = _read(root / record["summary"]["_receipt"]["path"])
    paths = [root / name for name in receipt["files"] if Path(name).name == "run_threads.json"]
    if len(paths) != 1:
        raise ValueError("required run_threads.json missing, duplicate or unbound")
    return validate_probe(_read(paths[0]), doc, affinity, rows, kernel)


def _mean(rows, key):
    values = [row[key] for row in rows]
    return statistics.mean(values) if values and all(value is not None for value in values) else None


def _format(value):
    return "不可用" if value is None else f"{value:.3f}"


def analyze(run_dir, bundle_dir=None):
    root = Path(run_dir).resolve()
    state = _read(root / "state.json")
    bundle = Path(bundle_dir or state["bundle"]).resolve()
    kernel = _read(root / "scheduler_capabilities.json").get("sched_schedstats") if (root / "scheduler_capabilities.json").exists() else None
    records, sources = [], {}
    for path in sorted((root / "records" / str(state["boot_id"])).rglob("*.json")):
        record = _read(path)
        if "diagnosis_case" not in record.get("document", {}):
            continue
        if record.get("boot_id") != state["boot_id"]:
            raise ValueError("run-scope record boot identity differs")
        rows, affinity, _, files = _bound_record(root, record)
        quality = _bound_trace(root, record, rows, affinity, kernel) if record["document"].get("run_probe") else None
        records.append((record, rows, quality))
        sources.update(files)
    tasks = [record["document"]["task_id"] for record, *_ in records]
    if len(tasks) != len(set(tasks)):
        raise ValueError("run-scope task counted twice")
    components, windows, stages, samples, threads, quality_notes = [], [], [], [], [], []
    for record, rows, quality in records:
        doc, summary = record["document"], record["summary"]
        label = {"case": doc["diagnosis_case"], "block": doc["diagnosis_block"], "executor": doc["executor"],
                 "task_id": doc["task_id"], "configuration": doc["configuration"]["id"]}
        kept = rows[doc["warmup"]:doc["warmup"] + doc["scored"]]
        if doc["kind"] == "calibration":
            stage = doc["configuration"]["stages"][0]
            values = {part: statistics.mean(row["stage0_" + ("ms" if part == "service" else part + "_ms")] for row in kept)
                      for part in ("set", "run", "get", "service")}
            components.append({**label, "program": stage["program"], "threads": stage["threads"],
                               "mask": ",".join(map(str, stage["mask"])),
                               **{part + "_ms": value for part, value in values.items()},
                               "run_core_ms": summary.get("run_core_ms")})
        if doc["kind"] == "performance":
            ii = (kept[-1]["completion_ms"] - kept[0]["completion_ms"]) / (len(kept) - 1)
            if abs(ii - summary["ii_ms"]) > 1e-6:
                raise ValueError("run-scope performance summary differs from raw frames")
            windows.append({**label, "ii_ms": ii, "fps": 1000 / ii,
                            "whole_process_core_ms_per_completion": summary.get("process_core_ms_per_completion"),
                            "kept_timestamps": len(kept), "completed_intervals": len(kept) - 1})
            for i, stage in enumerate(doc["configuration"]["stages"]):
                stages.append({**label, "stage_index": i, "program": stage["program"], "device": stage["device"],
                               "threads": stage["threads"], "mask": ",".join(map(str, stage["mask"])),
                               **{part + "_ms": statistics.mean(row[f"stage{i}_" + ("ms" if part == "service" else part + "_ms")] for row in kept)
                                  for part in ("set", "run", "get", "service")}})
        if quality:
            samples.extend({**label, "kind": doc["kind"], **sample} for sample in quality["samples"])
            threads.extend({**label, "kind": doc["kind"], **thread} for thread in quality["threads"])
            quality_notes.append({**label, **{key: value for key, value in quality.items() if key not in ("samples", "threads")}})
    bridges = []
    for case in CASE_NAMES:
        group = {row["block"]: row for row in windows if row["case"] == case}
        if not all(key in group for key in ("old_plain", "new_plain_before", "run_probe", "new_plain_after")):
            continue
        old, before, probe, after = (group[key] for key in ("old_plain", "new_plain_before", "run_probe", "new_plain_after"))
        mean = (before["ii_ms"] + after["ii_ms"]) / 2
        bridges.append({"case": case, "old_plain_fps": old["fps"], "new_plain_before_fps": before["fps"],
                        "probe_fps": probe["fps"], "new_plain_after_fps": after["fps"], "new_plain_mean_ii_ms": mean,
                        "new_plain_mean_fps": 1000 / mean,
                        "bridge_ii_change_percent": (mean / old["ii_ms"] - 1) * 100,
                        "probe_ii_change_percent": (probe["ii_ms"] / mean - 1) * 100,
                        "probe_difference_exceeds_2_percent_reference": abs(probe["ii_ms"] / mean - 1) > .02,
                        "new_plain_ii_range_percent": abs(before["ii_ms"] - after["ii_ms"]) / mean * 100})
    component_overhead = []
    for case in sorted({row["case"] for row in components}):
        group = {row["block"]: row for row in components if row["case"] == case}
        if "component_off" in group and "component_probe" in group:
            off, probe = group["component_off"], group["component_probe"]
            component_overhead.append({"case": case, "off_service_ms": off["service_ms"], "probe_service_ms": probe["service_ms"],
                                       "off_run_ms": off["run_ms"], "probe_legacy_run_ms": probe["run_ms"],
                                       "service_change_percent": (probe["service_ms"] / off["service_ms"] - 1) * 100,
                                       "run_change_percent": (probe["run_ms"] / off["run_ms"] - 1) * 100})
    grouped = {}
    for row in samples:
        if row["kind"] not in ("calibration", "performance"):
            continue
        grouped.setdefault((row["case"], row["program"], row["threads"], row["mask"], row["kind"]), []).append(row)
    aggregates = []
    for (case, program, count, mask, kind), values in grouped.items():
        aggregates.append({"case": case, "program": program, "threads": count, "mask": mask, "kind": kind,
                           "sample_count": len(values), **{key + "_mean": _mean(values, key) for key in
                           ("legacy_run_ms", "diagnostic_run_ms", "before_probe_ms", "after_probe_ms", "outside_diagnostic_run_ms",
                            "cpu_runtime_ms", "runqueue_wait_ms", "largest_worker_runtime_share", "runtime_per_wall_ratio")}})
    comparisons = []
    for pipeline in (row for row in aggregates if row["kind"] == "performance"):
        matching = [row for row in aggregates if row["kind"] == "calibration"
                    and all(row[key] == pipeline[key] for key in ("program", "threads", "mask"))]
        if len(matching) > 1:
            raise ValueError("multiple isolated profiles match the same run-scope resource setting")
        if matching:
            isolated = matching[0]
            values = {"case": pipeline["case"], "program": pipeline["program"],
                      "threads": pipeline["threads"], "mask": pipeline["mask"],
                      "isolated_samples": isolated["sample_count"], "pipeline_samples": pipeline["sample_count"]}
            for metric in ("diagnostic_run_ms_mean", "cpu_runtime_ms_mean", "runqueue_wait_ms_mean", "largest_worker_runtime_share_mean"):
                a, b = isolated[metric], pipeline[metric]
                values["isolated_" + metric] = a
                values["pipeline_" + metric] = b
                values["pipeline_minus_isolated_" + metric] = b - a if a is not None and b is not None else None
                values["pipeline_to_isolated_" + metric] = b / a if a and b is not None else None
            comparisons.append(values)
    thread_groups = {}
    sample_map = {(row["task_id"], row["frame_id"], row["stage_index"]): row for row in samples}
    for row in threads:
        if row["kind"] not in ("calibration", "performance"):
            continue
        key = (row["case"], row["kind"], row["program"], row["threads"], row["logical_worker"], row["core"])
        parent = sample_map[(row["task_id"], row["frame_id"], row["stage_index"])]
        thread_groups.setdefault(key, []).append({**row, "runtime_share": row["runtime_ms"] / parent["cpu_runtime_ms"]
                                                   if row["runtime_ms"] is not None and parent["cpu_runtime_ms"] else None})
    distributions = []
    for (case, kind, program, count, slot, core), values in thread_groups.items():
        distributions.append({"case": case, "kind": kind, "program": program, "threads": count,
                              "logical_worker": slot, "core": core, "sample_count": len(values),
                              "runtime_ms_mean": _mean(values, "runtime_ms"),
                              "runqueue_wait_ms_mean": _mean(values, "runqueue_wait_ms"),
                              "runtime_share_mean": _mean(values, "runtime_share"),
                              "equal_share_reference": 1 / count})
    report = {"experiment": state.get("experiment"), "bundle_id": state.get("bundle_id"), "boot_id": state["boot_id"],
              "accepted_tasks": len(records), "components": components, "performance_windows": windows,
              "bridge_and_probe_impact": bridges, "component_probe_impact": component_overhead,
              "sparse_run_aggregates": aggregates, "probe_quality": quality_notes,
              "isolated_to_pipeline": comparisons, "thread_distributions": distributions,
              "sample_count": len(samples), "thread_sample_count": len(threads),
              "source_raw_results": sources, "scope_warnings": WARNINGS,
              "model_changed": False, "rankings_changed": False, "automatic_additional_measurements": False}
    destination = root / "analysis"
    destination.mkdir(exist_ok=True)
    for name, values in (("components.csv", components), ("performance_windows.csv", windows), ("stage_comparison.csv", stages),
                         ("run_samples.csv", samples), ("run_threads.csv", threads), ("run_aggregates.csv", aggregates),
                         ("bridge_probe_impact.csv", bridges), ("component_probe_impact.csv", component_overhead),
                         ("isolated_to_pipeline.csv", comparisons), ("thread_distributions.csv", distributions)):
        _csv(destination / name, values)
    (destination / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    _write_report(destination, report)
    _plot(destination, aggregates, any(row["probe_difference_exceeds_2_percent_reference"] for row in bridges))
    return {"report": str(destination / "README.md"), "analysis": str(destination / "analysis.json"),
            "accepted_tasks": len(records), "sample_count": len(samples), "thread_sample_count": len(threads)}


def _write_report(destination, report):
    lines = ["# CPU run范围逐线程诊断", "",
             f"本次启动`{report['boot_id']}`，已接受{report['accepted_tasks']}个任务，"
             f"取得{report['sample_count']}个阶段／帧探针记录、{report['thread_sample_count']}个TID增量。",
             "原有静态成本、资源模型、DP和排名保持冻结。", "",
             "## 旧／新程序桥接与稀疏探针扰动", "",
             "无探针均值先平均完成间隔再换算FPS。各项均为本次观察，不作为拒收或追加测量门槛。", "",
             "|流水|旧程序FPS|新程序前／后FPS|新程序平均FPS|探针FPS|旧新间隔变化|探针间隔变化|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["bridge_and_probe_impact"]:
        lines.append(f"|{CASE_NAMES[row['case']]}|{row['old_plain_fps']:.3f}|{row['new_plain_before_fps']:.3f}／{row['new_plain_after_fps']:.3f}|"
                     f"{row['new_plain_mean_fps']:.3f}|{row['probe_fps']:.3f}|{row['bridge_ii_change_percent']:+.2f}%|{row['probe_ii_change_percent']:+.2f}%|")
    affected = [CASE_NAMES[row["case"]] for row in report["bridge_and_probe_impact"] if row["probe_difference_exceeds_2_percent_reference"]]
    if affected:
        lines += ["", "探针间隔差异超过2%参考值：" + "、".join(affected) + "。这些计数用于机制线索，不精确归因无探针性能；"
                  "2%只作为阅读提示，不拒收、不重测，也不承诺差异全部由探针造成。"]
    lines += ["", "## 隔离组件探针成本", "", "|组件|无探针service ms|有探针service ms|变化|", "|---|---:|---:|---:|"]
    for row in report["component_probe_impact"]:
        lines.append(f"|{row['case']}|{row['off_service_ms']:.3f}|{row['probe_service_ms']:.3f}|{row['service_change_percent']:+.2f}%|")
    lines += ["", "## run范围工作量与调度等待", "",
              "legacy run包含前后读数开销；diagnostic run只包住实际CPU模块run调用。CPU运行增量与等待增量"
              "各自对应每个TID的读取包络，不能相减求出同步等待。流水四个稀疏帧与隔离八帧分别报告。", "",
              "|场景／阶段|线程|样本|legacy run ms|diagnostic run ms|CPU运行核时 ms|队列等待核时 ms|最大线程份额|前／后探针 ms|",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["sparse_run_aggregates"]:
        lines.append(f"|{CASE_NAMES.get(row['case'], row['case'])}／{row['program']}|{row['threads']}|{row['sample_count']}|"
                     f"{_format(row['legacy_run_ms_mean'])}|{_format(row['diagnostic_run_ms_mean'])}|{_format(row['cpu_runtime_ms_mean'])}|"
                     f"{_format(row['runqueue_wait_ms_mean'])}|{_format(row['largest_worker_runtime_share_mean'])}|"
                     f"{_format(row['before_probe_ms_mean'])}／{_format(row['after_probe_ms_mean'])}|")
    lines += ["", "同程序、线程数和核心集合的隔离／流水探针配对：", "",
              "|流水／阶段|run墙时比值|CPU运行核时比值|隔离／流水队列等待 ms|隔离／流水最大线程份额|",
              "|---|---:|---:|---:|---:|"]
    for row in report["isolated_to_pipeline"]:
        lines.append(f"|{CASE_NAMES.get(row['case'], row['case'])}／{row['program']} t{row['threads']}|"
                     f"{_format(row['pipeline_to_isolated_diagnostic_run_ms_mean'])}|{_format(row['pipeline_to_isolated_cpu_runtime_ms_mean'])}|"
                     f"{_format(row['isolated_runqueue_wait_ms_mean'])}／{_format(row['pipeline_runqueue_wait_ms_mean'])}|"
                     f"{_format(row['isolated_largest_worker_runtime_share_mean'])}／{_format(row['pipeline_largest_worker_runtime_share_mean'])}|")
    lines += ["", "## 如何解释", "",
              "- 首先用旧／新无探针结果判断本次程序桥接差异，再结合探针扰动解释稀疏计数。扰动明显时，不用探针值精确分摊无探针损失。",
              "- 对同一程序、线程数、核心集合，比较隔离与流水的run墙时、CPU运行总量和运行队列等待。等待增加支持调度竞争；CPU总量变化仍混合有效计算与同步等在核开销。",
              "- 查看逻辑worker份额，检查均分核时假设是否与这些run样本一致；不把调用线程的交接工作混入run核时。",
              "- 前后读取包络和fresh-FD核验用于界定读数口径、检查缓存读取问题；它们不证明调度统计精确到每个指令。",
              "- 本轮不自动给出缓存、DDR、同步屏障各自占比，也不把wall−总CPU解释为任何一种等待。", "",
              "## 可用性", ""]
    for row in report["probe_quality"]:
        lines.append(f"- {row['case']}／{row['block']}：运行增量全部可用={row['all_runtime_samples_available']}，"
                     f"等待增量全部可用={row['all_wait_samples_available']}；警告：{';'.join(row['warnings']) or '无'}。")
    lines += ["", "## 数据口径", ""] + ["- " + warning for warning in WARNINGS]
    lines += ["", "完整原始增量、逻辑线程映射、读取上下界见run_threads.csv；"
              "逐帧run边界见run_samples.csv；原始证据与全部统计见analysis.json。", ""]
    (destination / "README.md").write_text("\n".join(lines))


def _plot(destination, aggregates, perturbation_warning=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not aggregates:
        return
    labels = [f"{row['case']}\n{row['program']} t{row['threads']}" for row in aggregates]
    fig, axes = plt.subplots(1, 2, figsize=(max(11, len(labels) * 1.15), 5))
    for ax, metric, title in ((axes[0], "diagnostic_run_ms_mean", "Module run wall time"),
                              (axes[1], "cpu_runtime_ms_mean", "Sum of per-TID scheduler runtime")):
        values = [row[metric] for row in aggregates]
        ax.bar(range(len(values)), [value or 0 for value in values], color="#2878b5")
        for i, value in enumerate(values):
            if value is None:
                ax.text(i, 0, "N/A", ha="center")
        ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("ms per sampled run (read envelope retained)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Warning: observed probe interval difference >2%; no causal apportionment"
                 if perturbation_warning else "Sparse probed runs; see bridge and probe overhead in report")
    fig.tight_layout()
    fig.savefig(destination / "run_comparison.png", dpi=150)
    fig.savefig(destination / "run_comparison.svg")
    plt.close(fig)
