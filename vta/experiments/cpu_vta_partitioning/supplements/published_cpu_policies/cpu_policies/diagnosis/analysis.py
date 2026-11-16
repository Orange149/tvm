"""Evidence-bound, descriptive diagnosis of isolated versus pipeline stage costs.

The native probe samples schedstat/PMU at collector callbacks, not precisely at
the two completion timestamps. These counters explain observed behavior but
must not be presented as per-frame, per-operator CPU or cache measurements.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


PMU_NAMES = ("cycles", "instructions", "cache_references", "cache_misses")
COUNTERS = ("runtime_ns", "runqueue_wait_ns", "timeslices", "voluntary_switches", "nonvoluntary_switches")
CASE_NAMES = {"static_first": "静态第一名：两段四线程计算共享核心",
              "history_host2": "历史参考：三线程／单线程核组复用、主机固定核心2",
              "history_shared": "匹配参考：相同切图与计算配置、主机共享四核"}
SCOPES = [
    "阶段均值取完成序号8至23所对应的16帧，不等于全局完成窗口内的资源利用率。",
    "资源覆盖使用全部28帧事件，裁剪到完成时刻C8至C23，并对重叠区间取并集。",
    "stage run是墙钟时间，包含抢占、内部屏障和同步；VTA run还包含主机提交与等待。",
    "阶段service不含CPU核组互斥等待、VTA互斥等待和Pop前的队列等待。",
    "VTA互斥锁持有覆盖不是FPGA纯计算利用率；CPU墙钟覆盖不是CPU在核运行率。",
    "schedstat/PMU在收集线程收到完成8/23后采样，原格式没有精确采样时间，不与C8/C23严格对齐。",
    "线程在开始或结束快照中缺失时不计入原生增量；生产线程可能提前退出，汇总并非完整进程核时账本。",
    "PMU计数汇总捕获的全部TID，不能将缓存缺失直接归给某个阶段。",
    "流水中阶段进程CPU时间应为null，禁止相加并发阶段的进程CPU时间。",
    "同一次启动的短窗口仅提供本次观察，不能推导跨启动稳定性或统计显著性。",
]


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _number(value, name, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (integer and not isinstance(value, int))):
        raise ValueError("invalid diagnostic metric: " + name)
    return value


def _csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def union_ms(intervals, begin, end):
    """Union of positive intervals clipped to one physical wall-clock window."""
    if end <= begin:
        raise ValueError("empty global completion window")
    clipped = sorted((max(a, begin), min(b, end)) for a, b in intervals if b > begin and a < end)
    result, previous = 0.0, None
    for a, b in clipped:
        if b < a:
            raise ValueError("inverted interval")
        if previous is None:
            previous = [a, b]
        elif a > previous[1]:
            result += previous[1] - previous[0]
            previous = [a, b]
        else:
            previous[1] = max(previous[1], b)
    return result + (previous[1] - previous[0] if previous else 0.0)


def _roles(affinity):
    result = {}
    for index, stage in enumerate(affinity.get("stages", [])):
        entries = stage.get("effective_affinities", [])
        for entry in entries:
            tid = entry["tid"]
            if tid in result:
                raise ValueError("duplicate affinity TID ownership")
            result[tid] = {"stage_index": index, "role": "caller" if tid == stage["caller_tid"] else "worker",
                           "actual_mask": entry["cpus"], "session_cpu_delta_ms": entry.get("cpu_time_delta_ms")}
    for entry in affinity.get("control_threads", []):
        tid = entry["tid"]
        if tid in result:
            raise ValueError("control and stage TID ownership overlaps")
        result[tid] = {"stage_index": None, "role": entry["role"], "actual_mask": entry["cpus"],
                       "session_cpu_delta_ms": None}
    return result


def profile_quality(profile, affinity, sched_schedstats=None):
    """Validate native JSON and preserve availability instead of inventing zeros.

    A valid native fallback with a fatal_reason is an unavailable probe, whereas
    malformed schema/counter values are invalid evidence. Runtime remains
    interpretable when positive scheduler runtime was actually observed; an
    entirely zero runtime ledger does not establish that no CPU work occurred.
    """
    if not isinstance(profile, dict) or profile.get("schema") != "vta_session_profile_v1":
        raise ValueError("invalid native session profile schema")
    fatal = profile.get("fatal_reason")
    if fatal is not None and not isinstance(fatal, str):
        raise ValueError("invalid native session profile fatal reason")
    warnings = []
    roles = _roles(affinity)
    if fatal:
        warnings.append("native_probe_unavailable:" + fatal)
        return {"schema": profile["schema"], "usable": False, "threads": [], "stage_totals": [],
                "pmu": {}, "coverage": {"expected_role_tids": sorted(roles), "missing_role_tids": sorted(roles)},
                "scheduler_runtime_available": False, "scheduler_wait_available": False,
                "sampling_scope": "collector_callbacks_after_completion_8_and_23_not_exact_C8_C23",
                "warnings": warnings}
    if profile.get("started") is not True or profile.get("scope") != "whole_process_existing_threads":
        raise ValueError("session profile was not started with expected scope")
    native_threads = profile.get("threads")
    if not isinstance(native_threads, list) or not isinstance(profile.get("aggregate"), dict):
        raise ValueError("missing native session profile counters")
    tids = set()
    for item in native_threads:
        tid = _number(item.get("tid"), "tid", integer=True)
        if not tid or tid in tids:
            raise ValueError("invalid or duplicate session profile TID")
        tids.add(tid)
        cpus = item.get("cpus")
        if (not isinstance(cpus, list) or cpus != sorted(set(cpus))
                or any(type(cpu) is not int or cpu < 0 for cpu in cpus)):
            raise ValueError("invalid session profile affinity")
        if tid in roles and cpus != roles[tid]["actual_mask"]:
            raise ValueError("profile and native affinity report differ")
        for name in COUNTERS:
            _number(item.get(name), name, integer=True)
    for name in COUNTERS:
        value = _number(profile["aggregate"].get(name), "aggregate " + name, integer=True)
        if value != sum(item[name] for item in native_threads):
            raise ValueError("profile aggregate does not equal captured thread sum")
    for name in ("start_tid_count", "end_tid_count"):
        _number(profile.get(name), name, integer=True)
    if len(tids) > profile["start_tid_count"]:
        raise ValueError("captured threads exceed start census")
    if isinstance(sched_schedstats, dict):
        sched_schedstats = sched_schedstats.get("value")
    if isinstance(sched_schedstats, str):
        sched_schedstats = sched_schedstats.strip()
    stats_enabled = sched_schedstats in (1, "1") and sched_schedstats is not True
    runtime_available = any(item["runtime_ns"] > 0 for item in native_threads)
    wait_available = any(item["runqueue_wait_ns"] > 0 for item in native_threads) or (stats_enabled and runtime_available)
    if not runtime_available:
        warnings.append("all_runtime_counters_zero_cannot_establish_absence_of_work")
    if not wait_available:
        warnings.append("all_runqueue_wait_zero_without_proven_enabled_accounting_is_unknown")
    threads, stages = [], {}
    for item in native_threads:
        role = roles.get(item["tid"], {"stage_index": None, "role": "unmapped_auxiliary",
                                       "actual_mask": item["cpus"], "session_cpu_delta_ms": None})
        row = {**item, **role, "actual_mask": ",".join(map(str, role["actual_mask"])),
               "runtime_available": runtime_available, "wait_available": wait_available,
               "runtime_ms": item["runtime_ns"] / 1e6 if runtime_available else None,
               "runqueue_wait_ms": item["runqueue_wait_ns"] / 1e6 if wait_available else None}
        if runtime_available and wait_available and item["runtime_ns"] + item["runqueue_wait_ns"] > 0:
            row["wait_fraction_of_runtime_plus_wait"] = item["runqueue_wait_ns"] / (item["runtime_ns"] + item["runqueue_wait_ns"])
        else:
            row["wait_fraction_of_runtime_plus_wait"] = None
        threads.append(row)
        if role["stage_index"] is not None:
            stages.setdefault(role["stage_index"], []).append(row)
    stage_rows = []
    for index in range(len(affinity.get("stages", []))):
        items = stages.get(index, [])
        expected = [tid for tid, role in roles.items() if role["stage_index"] == index]
        coverage = all(tid in tids for tid in expected)
        stage_rows.append({"stage_index": index, "expected_threads": len(expected), "observed_threads": len(items),
                           "complete_stage_thread_coverage": coverage,
                           "runtime_ms": sum(item["runtime_ns"] for item in items) / 1e6 if runtime_available else None,
                           "runqueue_wait_ms": sum(item["runqueue_wait_ns"] for item in items) / 1e6 if wait_available else None,
                           "thread_runtime_min_ms": min((item["runtime_ns"] / 1e6 for item in items), default=None) if runtime_available else None,
                           "thread_runtime_max_ms": max((item["runtime_ns"] / 1e6 for item in items), default=None) if runtime_available else None,
                           "voluntary_switches": sum(item["voluntary_switches"] for item in items),
                           "nonvoluntary_switches": sum(item["nonvoluntary_switches"] for item in items)})
    events = profile.get("pmu")
    if not isinstance(events, dict) or set(events) != set(PMU_NAMES):
        raise ValueError("session profile PMU event coverage differs")
    pmu = {}
    for name, event in events.items():
        if not isinstance(event, dict) or type(event.get("available")) is not bool:
            raise ValueError("invalid PMU availability")
        for key in ("raw", "scaled", "time_enabled", "time_running"):
            _number(event.get(key), "PMU " + name + " " + key, integer=key != "scaled")
        if event["time_running"] > event["time_enabled"]:
            raise ValueError("PMU running time exceeds enabled time")
        usable = event["available"] and event["time_running"] > 0
        reason = event.get("reason") or ("counter_never_scheduled" if event["available"] and not usable else None)
        pmu[name] = {**event, "usable": usable, "unavailable_reason": reason,
                     "interpreted_scaled": event["scaled"] if usable else None,
                     "running_fraction": event["time_running"] / event["time_enabled"] if usable else None}
        if not usable:
            warnings.append("PMU_" + name + "_unavailable:" + str(reason))
    missing = sorted(set(roles) - tids)
    if missing:
        warnings.append("role_threads_missing_from_both_endpoint_snapshot_intersection")
    if len(tids) < profile["start_tid_count"]:
        warnings.append("some_start_threads_missing_or_unreadable_at_end")
    return {"schema": profile["schema"], "usable": True, "threads": threads, "stage_totals": stage_rows,
            "pmu": pmu, "scheduler_runtime_available": runtime_available,
            "scheduler_wait_available": wait_available,
            "coverage": {"start_tid_count": profile["start_tid_count"], "end_tid_count": profile["end_tid_count"],
                         "returned_thread_count": len(tids), "expected_role_tids": sorted(roles),
                         "missing_role_tids": missing, "unmapped_auxiliary_tids": sorted(tids - set(roles))},
            "sampling_scope": "collector_callbacks_after_completion_8_and_23_not_exact_C8_C23",
            "system_start": profile.get("system_start"), "system_end": profile.get("system_end"),
            "warnings": warnings}


def _bound_record(root, record):
    doc, summary = record["document"], record["summary"]
    receipt_ref = summary["_receipt"]
    receipt_path = (root / receipt_ref["path"]).resolve()
    if not receipt_path.is_relative_to(root) or _sha(receipt_path) != receipt_ref["sha256"]:
        raise ValueError("diagnosis receipt missing, modified or outside run")
    receipt = _read(receipt_path)
    if receipt["task"] != doc or receipt["boot_id"] != record["boot_id"]:
        raise ValueError("diagnosis receipt task or boot differs")
    files = receipt["files"]
    for name, ref in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or _sha(path) != ref["sha256"] or path.stat().st_size != ref["bytes"]:
            raise ValueError("diagnosis raw evidence changed: " + name)
    def bound_leaf(leaf, required=True):
        found = [root / name for name in files if Path(name).name == leaf]
        if len(found) > 1 or (required and len(found) != 1):
            raise ValueError("missing or duplicate bound diagnosis artifact: " + leaf)
        return found[0] if found else None
    result = bound_leaf("result.jsonl")
    frames = [json.loads(line) for line in result.read_text().splitlines() if line.strip()]
    total = sum(doc[key] for key in ("warmup", "scored", "tail"))
    if (len(frames) != total or [f.get("completion_index") for f in frames] != list(range(total))
            or [f.get("frame_id") for f in frames] != list(range(total))
            or any(type(f.get("input_index")) is not int or f["input_index"] != 0 for f in frames)):
        raise ValueError("diagnosis frames incomplete, repeated or wrong input")
    affinity = _read(bound_leaf("affinity.json"))
    profile = bound_leaf("profile.json", required=bool(doc.get("diagnosis_detail")))
    return frames, affinity, _read(profile) if profile else None, {str(result.relative_to(root)): files[str(result.relative_to(root))]}


def _stage_rows(document, frames, baseline, components):
    cfg = document["configuration"]
    kept = frames[document["warmup"]:document["warmup"] + document["scored"]]
    begin, end = kept[0]["completion_ms"], kept[-1]["completion_ms"]
    rows, timeline, vta_intervals = [], [], []
    for index, stage in enumerate(cfg["stages"]):
        source = baseline.get(stage, cfg)
        component = components.get((stage["program"], stage["threads"], tuple(stage["mask"])))
        row = {"case": document["diagnosis_case"], "block": document["diagnosis_block"],
               "configuration": cfg["id"], "stage_index": index, "program": stage["program"],
               "device": stage["device"], "threads": stage["threads"],
               "mask": ",".join(map(str, stage["mask"])), "sample_frames": len(kept),
               "old_isolated_source": source["source_key"],
               "old_isolated_assumptions": ";".join(source["assumptions"])}
        for part in ("set", "run", "get", "service"):
            suffix = "ms" if part == "service" else part + "_ms"
            observed = [f[f"stage{index}_{suffix}"] for f in kept]
            for value in observed:
                _number(value, "stage " + part)
            row[f"old_isolated_{part}_ms"] = source[f"{part}_ms"]
            row[f"new_isolated_{part}_ms"] = component[f"{part}_ms"] if component else None
            row[f"pipeline_{part}_mean_ms"] = statistics.mean(observed)
            row[f"pipeline_{part}_min_ms"] = min(observed)
            row[f"pipeline_{part}_max_ms"] = max(observed)
            if component and component[f"{part}_ms"] > 0:
                row[f"pipeline_to_new_isolated_{part}_ratio"] = statistics.mean(observed) / component[f"{part}_ms"]
            else:
                row[f"pipeline_to_new_isolated_{part}_ratio"] = None
        row["pipeline_to_old_isolated_service_ratio"] = row["pipeline_service_mean_ms"] / source["service_ms"] if source["service_ms"] else None
        row["new_to_old_isolated_service_ratio"] = component["service_ms"] / source["service_ms"] if component and source["service_ms"] else None
        stage_intervals = []
        for frame in frames:
            start, stop = frame[f"stage{index}_start_ms"], frame[f"stage{index}_end_ms"]
            if stop < start or abs(stop - start - frame[f"stage{index}_ms"]) > 5e-6:
                raise ValueError("stage service boundary mismatch")
            if abs(frame[f"stage{index}_ms"] - sum(frame[f"stage{index}_{part}_ms"] for part in ("set", "run", "get"))) > 5e-6:
                raise ValueError("stage set/run/get scope mismatch")
            if frame.get(f"stage{index}_process_cpu_ms") is not None:
                raise ValueError("concurrent stage process CPU time must be null")
            stage_intervals.append((start, stop))
            timeline.append({"case": document["diagnosis_case"], "block": document["diagnosis_block"],
                             "frame_id": frame["frame_id"], "completion_index": frame["completion_index"],
                             "stage_index": index, "program": stage["program"], "device": stage["device"],
                             "start_ms": start, "run_start_ms": frame.get(f"stage{index}_run_start_ms"),
                             "run_end_ms": frame.get(f"stage{index}_run_end_ms"), "end_ms": stop,
                             "C8_ms": begin, "C23_ms": end,
                             "vta_wait_ms": frame.get(f"stage{index}_vta_mutex_wait_ms"),
                             "cpu_group_wait_ms": frame.get(f"stage{index}_cpu_gate_wait_ms")})
        row["wall_service_union_fraction_C8_C23"] = union_ms(stage_intervals, begin, end) / (end - begin)
        for name in ("vta_mutex_wait", "vta_mutex_held", "cpu_gate_wait"):
            values = [f.get(f"stage{index}_{name}_ms") for f in kept]
            row[name + "_mean_ms"] = statistics.mean(values) if all(x is not None for x in values) else None
        if stage["device"] == "vta":
            vta_intervals.extend((f[f"stage{index}_vta_mutex_acquire_ms"], f[f"stage{index}_vta_mutex_release_ms"]) for f in frames)
        if index:
            gaps = [f[f"stage{index}_queue_pop_ms"] - f[f"stage{index-1}_end_ms"] for f in kept]
            row["combined_handoff_residence_and_consumer_delay_mean_ms"] = statistics.mean(gaps)
        rows.append(row)
    return rows, timeline, union_ms(vta_intervals, begin, end) / (end - begin)


def _queue_rows(document, frames):
    result = []
    for frame in frames:
        if document.get("diagnosis_detail") and frame.get("queue_timing_enabled") is not True:
            raise ValueError("detailed diagnostic lacks queue timing")
        for event in frame.get("queue_timings", []):
            keys = ("push_request_ms", "enqueue_ms", "dequeue_ms", "push_return_ms")
            for key in keys:
                _number(event.get(key), "queue " + key)
            # A consumer may dequeue before the producer records Push's return.
            if (event["enqueue_ms"] < event["push_request_ms"] - 1e-6
                    or event["dequeue_ms"] < event["enqueue_ms"] - 1e-6
                    or event["push_return_ms"] < event["enqueue_ms"] - 1e-6):
                raise ValueError("queue causal event order differs")
            result.append({"case": document["diagnosis_case"], "block": document["diagnosis_block"],
                           "frame_id": frame["frame_id"], **event,
                           "producer_push_call_ms": event["push_return_ms"] - event["push_request_ms"],
                           "wait_before_enqueue_ms": event["enqueue_ms"] - event["push_request_ms"],
                           "queue_residence_ms": event["dequeue_ms"] - event["enqueue_ms"]})
    return result


def _component_cost(document, frames):
    kept = frames[document["warmup"]:document["warmup"] + document["scored"]]
    result = {}
    for part in ("set", "run", "get", "service"):
        suffix = "ms" if part == "service" else part + "_ms"
        result[part + "_ms"] = statistics.mean(_number(frame["stage0_" + suffix], part) for frame in kept)
    for part in ("set", "run", "get"):
        values = [frame.get("stage0_" + part + "_process_cpu_ms") for frame in kept]
        result[part + "_core_ms"] = statistics.mean(values) if all(v is not None for v in values) else None
    return result


def _environment_schedstat(root):
    # Root controller may persist the read-only sysctl result in either file.
    for path in (root / "scheduler_capabilities.json", root / "environment.json", root / "preflight.json"):
        if path.is_file():
            doc = _read(path)
            if "sched_schedstats" in doc:
                return doc["sched_schedstats"]
    return None


def analyze(run_dir, bundle_dir=None):
    """Write only this batch's analysis; never alter source costs or rankings."""
    root = Path(run_dir).resolve()
    state = _read(root / "state.json")
    bundle = Path(bundle_dir or state["bundle"]).resolve()
    from cpu_policies.open_resources.costs import Costs
    baseline = Costs(_read(bundle / "plans/baseline_costs.json"), _read(bundle / "plans/programs.json"))
    predictions_path = bundle / "plans/baseline_predictions.json"
    predictions = _read(predictions_path) if predictions_path.is_file() else {}
    records, sources = [], {}
    seen = set()
    for path in sorted((root / "records" / str(state["boot_id"])).rglob("*.json")):
        record = _read(path)
        doc = record.get("document", {})
        if "diagnosis_case" not in doc:
            continue
        if record.get("boot_id") != state["boot_id"]:
            raise ValueError("diagnosis record boot differs")
        if doc["task_id"] in seen:
            raise ValueError("same diagnostic task recorded twice")
        seen.add(doc["task_id"])
        frames, affinity, profile, files = _bound_record(root, record)
        sources.update(files)
        records.append((doc, frames, affinity, profile, record["summary"]))
    components, component_rows = {}, []
    for doc, frames, affinity, profile, summary in records:
        if doc["kind"] != "calibration":
            continue
        row = doc["configuration"]["stages"][0]
        key = (row["program"], row["threads"], tuple(row["mask"]))
        if key in components:
            raise ValueError("component profile duplicated")
        components[key] = _component_cost(doc, frames)
        component_rows.append({"case": doc["diagnosis_case"], "program": row["program"],
                               "threads": row["threads"], "mask": ",".join(map(str, row["mask"])),
                               "sample_frames": doc["scored"], **components[key]})
    stages, timelines, queues, windows, threads, stage_counters, probe = [], [], [], [], [], [], {}
    for doc, frames, affinity, profile, summary in records:
        if doc["kind"] != "performance":
            continue
        if (doc["warmup"], doc["scored"], doc["tail"]) != (8, 16, 4):
            raise ValueError("diagnostic performance window differs")
        kept = frames[8:24]
        gaps = [b["completion_ms"] - a["completion_ms"] for a, b in zip(kept, kept[1:])]
        if min(gaps) <= 0:
            raise ValueError("nonpositive completion gap")
        ii = statistics.mean(gaps)
        if abs(ii - summary["ii_ms"]) > 1e-6:
            raise ValueError("diagnosis summary and raw completion interval differ")
        stage_rows, timeline, coverage = _stage_rows(doc, frames, baseline, components)
        stages.extend(stage_rows)
        timelines.extend(timeline)
        queues.extend(_queue_rows(doc, frames))
        window = {"case": doc["diagnosis_case"], "label": CASE_NAMES.get(doc["diagnosis_case"], doc["diagnosis_case"]),
                  "block": doc["diagnosis_block"], "configuration": doc["configuration"]["id"],
                  "ii_ms": ii, "fps": 1000 / ii, "completion_interval_count": len(gaps),
                  "gap_min_ms": min(gaps), "gap_max_ms": max(gaps),
                  "gap_cv": statistics.pstdev(gaps) / ii, "C8_ms": kept[0]["completion_ms"],
                  "C23_ms": kept[-1]["completion_ms"], "vta_mutex_held_coverage": coverage,
                  "process_core_ms_per_completion": summary.get("process_core_ms_per_completion"),
                  "stage0_last_input_end_minus_C23_ms": frames[-1]["stage0_end_ms"] - kept[-1]["completion_ms"]}
        prediction = predictions.get(doc["configuration"]["id"], {}).get("score_ms")
        window["archived_predicted_ii_ms"] = prediction
        window["measured_to_archived_predicted_ii_ratio"] = ii / prediction if prediction else None
        windows.append(window)
        if profile is not None and doc.get("diagnosis_detail"):
            value = profile_quality(profile, affinity, _environment_schedstat(root))
            probe[doc["diagnosis_case"]] = value
            for row in value["threads"]:
                threads.append({"case": doc["diagnosis_case"], "block": doc["diagnosis_block"], **row})
            for row in value["stage_totals"]:
                stage = doc["configuration"]["stages"][row["stage_index"]]
                stage_counters.append({"case": doc["diagnosis_case"], "program": stage["program"],
                                       "device": stage["device"], **row})
    perturbations = []
    for case in CASE_NAMES:
        rows = {r["block"]: r for r in windows if r["case"] == case}
        if set(rows) != {"plain_before", "detailed", "plain_after"}:
            continue
        before, detail, after = (rows[k] for k in ("plain_before", "detailed", "plain_after"))
        mean_plain_ii = (before["ii_ms"] + after["ii_ms"]) / 2
        perturbations.append({"case": case, "plain_before_fps": before["fps"], "detailed_fps": detail["fps"],
                              "plain_after_fps": after["fps"], "mean_plain_ii_ms": mean_plain_ii,
                              "mean_plain_fps": 1000 / mean_plain_ii,
                              "plain_ii_range_percent_of_mean": abs(after["ii_ms"] - before["ii_ms"]) / mean_plain_ii * 100,
                              "detailed_ii_change_percent": (detail["ii_ms"] / mean_plain_ii - 1) * 100,
                              "detailed_outside_adjacent_plain_ii_range": not min(before["ii_ms"], after["ii_ms"]) <= detail["ii_ms"] <= max(before["ii_ms"], after["ii_ms"]),
                              "interpretation": "descriptive_probe_perturbation_not_a_rejection_or_repeat_gate"})
    growth = []
    for case in CASE_NAMES:
        plain = [row for row in stages if row["case"] == case and row["block"].startswith("plain") and row["device"] == "cpu"]
        for index in sorted({row["stage_index"] for row in plain}):
            rows = [row for row in plain if row["stage_index"] == index]
            current = rows[0]
            observed = statistics.mean(row["pipeline_service_mean_ms"] for row in rows)
            observed_run = statistics.mean(row["pipeline_run_mean_ms"] for row in rows)
            isolated = current["new_isolated_service_ms"]
            isolated_run = current["new_isolated_run_ms"]
            growth.append({"case": case, "stage_index": index, "program": current["program"],
                           "threads": current["threads"], "same_boot_isolated_service_ms": isolated,
                           "plain_mean_pipeline_service_ms": observed,
                           "pipeline_minus_same_boot_isolated_service_ms": observed - isolated if isolated is not None else None,
                           "pipeline_to_same_boot_isolated_service_ratio": observed / isolated if isolated else None,
                           "pipeline_minus_same_boot_isolated_run_ms": observed_run - isolated_run if isolated_run is not None else None,
                           "sample_scope": "mean_of_adjacent_plain_sessions_frame_conditioned_stage_service"})
    host_pair = None
    by_case = {r["case"]: r for r in perturbations}
    if all(key in by_case for key in ("history_host2", "history_shared")):
        host_pair = {"host2_mean_plain_fps": by_case["history_host2"]["mean_plain_fps"],
                     "shared_mean_plain_fps": by_case["history_shared"]["mean_plain_fps"],
                     "host2_vs_shared_throughput_percent": (by_case["history_host2"]["mean_plain_fps"] / by_case["history_shared"]["mean_plain_fps"] - 1) * 100,
                     "interpretation": "one_boot_host_placement_intervention_not_universal_best_core"}
    report = {"scope": "one_boot_three_configurations_existing_native_probe",
              "run_dir": str(root), "bundle_id": state.get("bundle_id"), "boot_id": state["boot_id"],
              "record_count": len(records), "component_profile_count": len(components),
              "performance_session_count": len(windows), "qualification_session_count": sum(doc["kind"].startswith("qualify") for doc, *_ in records),
              "performance_windows": windows, "component_profiles": component_rows,
              "stage_comparison": stages, "probe_perturbation": perturbations,
              "session_profiles": probe, "scope_warnings": SCOPES,
              "observations": {"plain_stage_growth": growth, "matched_host_pair": host_pair},
              "source_raw_results": sources, "baseline_cost_sha256": _sha(bundle / "plans/baseline_costs.json"),
              "model_modified": False, "rankings_modified": False, "automatic_extra_measurements": False}
    destination = root / "analysis"
    destination.mkdir(exist_ok=True)
    for name, rows in (("components.csv", component_rows), ("stage_comparison.csv", stages),
                       ("performance_windows.csv", windows), ("stage_timeline.csv", timelines),
                       ("queue_timeline.csv", queues), ("threads.csv", threads),
                       ("stage_scheduler.csv", stage_counters), ("probe_perturbation.csv", perturbations)):
        _csv(destination / name, rows)
    (destination / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    _write_report(destination, report)
    _plot(destination, timelines, windows)
    return {"report": str(destination / "README.md"), "analysis": str(destination / "analysis.json"),
            "component_profile_count": len(components), "performance_session_count": len(windows),
            "diagnostic_probe_count": len(probe)}


def _fmt(value, places=3):
    return "不可用" if value is None else f"{value:.{places}f}"


def _write_report(destination, report):
    lines = ["# CPU–VTA静态预测误差：同启动小规模诊断", "",
             f"测量启动ID：`{report['boot_id']}`。本轮包含{report['component_profile_count']}种隔离CPU配置、"
             f"{report['performance_session_count']}个流水性能会话及{report['qualification_session_count']}个正确性资格会话。",
             "原成本表、评分公式、DP和Top-20均未修改。以下为一次启动的短窗口观察，不作为跨启动统计结论。", "",
             "## 流水性能及探针扰动", "",
             "每个流水配置依次执行无详细探针、详细探针、无详细探针。正式观察以两侧无详细探针会话为准；"
             "中间会话用于调度与队列线索。均值先平均完成间隔再换算FPS。", "",
             "|配置|前侧FPS|详细探针FPS|后侧FPS|两侧平均FPS|详细探针完成间隔变化|", "|---|---:|---:|---:|---:|---:|"]
    for row in report["probe_perturbation"]:
        lines.append(f"|{CASE_NAMES[row['case']]}|{row['plain_before_fps']:.3f}|{row['detailed_fps']:.3f}|{row['plain_after_fps']:.3f}|{row['mean_plain_fps']:.3f}|{row['detailed_ii_change_percent']:+.2f}%|")
    if report["observations"]["matched_host_pair"]:
        pair = report["observations"]["matched_host_pair"]
        lines += ["", f"保持历史参考的切图和计算配置不变时，主机固定核心2相对共享四核的本次吞吐变化为"
                  f"{pair['host2_vs_shared_throughput_percent']:+.2f}%。该配对只改变主机放置，尚未消除会话顺序与时间波动的影响。"]
    lines += ["", "## 阶段耗时：旧隔离、同启动隔离、流水内", "",
              "表中流水耗时先平均同配置的两侧无详细探针会话，service包括输入／计算／输出，排除核组和VTA互斥等待。", "",
              "|流水配置|CPU阶段／线程|旧隔离service ms|新隔离service ms|流水service ms|流水／新隔离|流水run ms|", "|---|---|---:|---:|---:|---:|---:|"]
    for case in CASE_NAMES:
        plain = [row for row in report["stage_comparison"] if row["case"] == case and row["block"].startswith("plain") and row["device"] == "cpu"]
        indices = sorted({row["stage_index"] for row in plain})
        for index in indices:
            rows = [row for row in plain if row["stage_index"] == index]
            row = rows[0]
            service = statistics.mean(r["pipeline_service_mean_ms"] for r in rows)
            run = statistics.mean(r["pipeline_run_mean_ms"] for r in rows)
            isolated = row["new_isolated_service_ms"]
            ratio = service / isolated if isolated else None
            lines.append(f"|{CASE_NAMES[case]}|{row['program']}／{row['threads']}|{row['old_isolated_service_ms']:.3f}|{_fmt(isolated)}|{service:.3f}|{_fmt(ratio)}|{run:.3f}|")
    lines += ["", "## 调度、线程工作量与可用性", ""]
    for case, profile in report["session_profiles"].items():
        lines += [f"### {CASE_NAMES.get(case, case)}", "",
                  f"线程运行时间可解释：{profile['scheduler_runtime_available']}；运行队列等待可解释：{profile['scheduler_wait_available']}。",
                  f"缺失角色TID：`{profile['coverage'].get('missing_role_tids', [])}`。", "",
                  "|阶段索引|覆盖线程|CPU运行时间 ms|运行队列等待 ms|最少／最多线程运行时间 ms|", "|---|---:|---:|---:|---:|"]
        for row in profile["stage_totals"]:
            lines.append(f"|{row['stage_index']}|{row['observed_threads']}/{row['expected_threads']}|{_fmt(row['runtime_ms'])}|{_fmt(row['runqueue_wait_ms'])}|{_fmt(row['thread_runtime_min_ms'])}／{_fmt(row['thread_runtime_max_ms'])}|")
        usable = [key for key, value in profile["pmu"].items() if value["usable"]]
        lines += ["", "可用PMU事件：" + ("、".join(usable) if usable else "无") + "。原始值、运行比例和不可用原因见analysis.json。"]
        if profile["warnings"]:
            lines += ["", "计数质量：" + "；".join(profile["warnings"]) + "。"]
    lines += ["", "## 本轮能够支持的判断", "",
              "- **已确认的测量差异**：以上表格直接回答哪些CPU阶段在流水中变慢，以及差异主要发生在set、run还是get；完整拆分见stage_comparison.csv。",
              "- **受支持的机制线索**：有效且非零的运行队列等待证明该采样范围存在可运行线程争用；线程CPU时间差异可提示工作量不均。它们与阶段墙钟增量的时间口径不同，不能将聚合等待直接当成某一帧的慢化量。",
              "- **主机位置干预**：两个历史参考保持切图、CPU线程数和核组、门控、控制线程放置一致，仅改变VTA主机位置。它们的本次性能差异支持该配置上的放置影响，不推广为固定核心2普遍最优。",
              "- **仍未区分**：即使阶段run增加，现有探针也不能单独区分缓存／DDR竞争、同步屏障和调度抢占的全部贡献。聚合PMU不提供阶段级因果归属。",
              "- 本批次在这些证据处结束。若指标不可用或探针改变明显，先保留未知项；后续精确按阶段测线程CPU时间／同步等待的执行器方案单独制定，不用本轮性能回填成本或拟合常数。", "",
              "本次同启动对照中，CPU阶段service增量最大的项目如下（允许为负，表示流水更快；不预设存在慢化）：", ""]
    growth = [row for row in report["observations"]["plain_stage_growth"] if row["pipeline_minus_same_boot_isolated_service_ms"] is not None]
    for row in sorted(growth, key=lambda r: -r["pipeline_minus_same_boot_isolated_service_ms"])[:4]:
        lines.append(f"- {CASE_NAMES[row['case']]}，{row['program']}／{row['threads']}线程："
                     f"service变化{row['pipeline_minus_same_boot_isolated_service_ms']:+.3f} ms，"
                     f"run变化{row['pipeline_minus_same_boot_isolated_run_ms']:+.3f} ms，"
                     f"service比值{_fmt(row['pipeline_to_same_boot_isolated_service_ratio'])}。")
    lines += ["",
              "## 数据口径", ""]
    lines += ["- " + warning for warning in SCOPES]
    lines += ["", "## 产物", "",
              "- analysis.json：原始文件哈希、组件、流水、阶段及探针质量总表。",
              "- stage_comparison.csv／components.csv：局部耗时对照。",
              "- performance_windows.csv／probe_perturbation.csv：吞吐、完成间隔与扰动。",
              "- threads.csv／stage_scheduler.csv：TID角色、计数可用性与阶段汇总。",
              "- stage_timeline.csv／queue_timeline.csv：全部帧事件，保留C8/C23窗口界限。",
              "- timeline.png／timeline.svg：若matplotlib可用，显示三个详细会话的阶段时间线。", ""]
    (destination / "README.md").write_text("\n".join(lines))


def _plot(destination, timelines, windows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cases = [case for case in CASE_NAMES if any(r["case"] == case and r["block"] == "detailed" for r in timelines)]
    if not cases:
        return
    fig, axes = plt.subplots(len(cases), 1, figsize=(13, 3.1 * len(cases)), squeeze=False)
    for ax, case in zip(axes[:, 0], cases):
        rows = [r for r in timelines if r["case"] == case and r["block"] == "detailed"]
        window = next(r for r in windows if r["case"] == case and r["block"] == "detailed")
        begin, end = window["C8_ms"], window["C23_ms"]
        for row in rows:
            a, b = max(begin, row["start_ms"]), min(end, row["end_ms"])
            if b > a:
                ax.broken_barh([(a - begin, b - a)], (row["stage_index"] - .32, .64),
                               facecolors="#2878b5" if row["device"] == "cpu" else "#e1812c", alpha=.8)
        ticks = sorted({r["stage_index"] for r in rows})
        ax.set_yticks(ticks, [next(r["program"] for r in rows if r["stage_index"] == i) for i in ticks])
        ax.set_xlim(0, end - begin)
        ax.set_title(case + f"; detailed FPS={window['fps']:.3f}")
        ax.set_xlabel("ms since C8 (all frames clipped to C8..C23)")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
    fig.suptitle("Stage wall-service timeline: blue=CPU, orange=VTA; waiting excluded")
    fig.tight_layout()
    fig.savefig(destination / "timeline.png", dpi=150)
    fig.savefig(destination / "timeline.svg")
    plt.close(fig)
