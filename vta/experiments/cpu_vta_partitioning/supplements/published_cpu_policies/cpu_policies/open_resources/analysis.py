"""Report measured coverage without turning missing contexts into losers."""
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics

from .campaign import _common, load_costs, validate_state, validated_screening
from .domain import contexts


METHOD_NAMES = {"single_core": "Tarnawski式单线程物理核心映射适配",
                "prefix": "原前缀核组静态方法", "enhanced_pure": "细化CPU建模：纯静态前20",
                "enhanced_diverse": "细化CPU建模：差异化20"}


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _configuration(config):
    ctx = next(c for c in contexts() if c["id"] == config["context_id"])
    rows = config["stages"]
    cpu = [s for s in rows if s["device"] == "cpu"]
    return {"configuration": config["id"], "context_id": ctx["id"], "context": ctx["label"],
            "context_family": ctx["family"], "compute_pool": ",".join(map(str, config["compute_pool"])),
            "reserved_cores": ",".join(map(str, config["reserved_cores"])),
            "partition": " → ".join(f"{s['device'].upper()} {s['begin']}–{s['end'] - 1}" for s in rows),
            "cpu_threads": "/".join(str(s["threads"]) for s in cpu),
            "cpu_groups": "/".join("{" + ",".join(map(str, s["mask"])) + "}" for s in cpu),
            "worker_core_maps": "/".join(",".join(f"{i}→{c}" for i, c in enumerate(s["worker_core_map"])) for s in cpu),
            "caller_cores": ",".join(str(s["caller_core"]) for s in cpu),
            "host_cores": ",".join(map(str, config["host_mask"])),
            "producer_cores": ",".join(map(str, config["control_mask"])),
            "collector_main_cores": ",".join(map(str, config["control_mask"])),
            "affinity_mode": config["affinity_mode"], "execution_mode": config["execution_mode"]}


def _dispersion(root, item):
    api = _common()
    receipt = api.read(api.relative_file(root, item["summary"]["_receipt"]["path"]))
    files = [name for name in receipt["files"] if Path(name).name == "result.jsonl"]
    if len(files) != 1:
        raise ValueError("analysis requires exactly one raw completion record")
    from cpu_vta_partitioning.measurement import frames
    doc = item["document"]
    rows = frames(api.relative_file(root, files[0]), doc["warmup"] + doc["scored"] + doc["tail"])
    kept = rows[doc["warmup"]:doc["warmup"] + doc["scored"]]
    gaps = [b["completion_ms"] - a["completion_ms"] for a, b in zip(kept, kept[1:])]
    if len(gaps) != 15 or min(gaps) <= 0:
        raise ValueError("analysis completion interval count or ordering differs")
    half = len(gaps) // 2
    return {"gap_min_ms": min(gaps), "gap_max_ms": max(gaps),
            "gap_cv": statistics.pstdev(gaps) / statistics.mean(gaps),
            "first_part_ii_ms": statistics.mean(gaps[:half]),
            "last_part_ii_ms": statistics.mean(gaps[half:]),
            "performance_variation_is_rejection_gate": False}


def _series(rankings):
    methods = rankings.get("methods", {})
    pure = rankings.get("pure_static20", methods.get("enhanced", {}).get("rankings", []))
    diverse = rankings.get("diverse20", [])
    return {"single_core": methods.get("single_core", {}).get("rankings", []),
            "prefix": methods.get("prefix", {}).get("rankings", []),
            "enhanced_pure": pure, "enhanced_diverse": diverse}


def _spearman(rows):
    if len(rows) < 2:
        return None
    a, b = [r["static_rank"] for r in rows], [r["measured_rank"] for r in rows]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in a + b):
        return None
    av, bv = statistics.mean(a), statistics.mean(b)
    denominator = math.sqrt(sum((x - av) ** 2 for x in a) * sum((y - bv) ** 2 for y in b))
    return sum((x - av) * (y - bv) for x, y in zip(a, b)) / denominator if denominator else None


def _boot_accounting(state, records):
    source_boots, unknown_sources = set(), 0
    for record in records.values():
        if record.get("kind") != "imported":
            continue
        boot = (record.get("provenance", {}).get("source_boot_id")
                or record.get("measurement", {}).get("boot_id"))
        if boot:
            source_boots.add(boot)
        else:
            unknown_sources += 1
    boot = state.get("boot_id")
    shared = (boot in source_boots) if boot and not unknown_sources else None
    return {"measurement_boot_count": 1 if boot else 0,
            "imported_source_boot_ids": sorted(source_boots),
            "measurement_boot_shared_with_imports": shared,
            "new_boot_count_relative_to_imports": (0 if shared else 1) if shared is not None else None,
            "unknown_imported_source_boot_count": unknown_sources,
            "independent_confirmation_boots": 0,
            "boot_accounting_scope": "distinct_physical_boot_identity_not_statistical_independence"}


def _preparation_cost(root, state, rankings):
    """Count accepted tasks/frames separately from dated imported components."""
    api = _common(state["bundle"])
    kinds = {}
    for path in sorted((root / "accepted" / str(state.get("boot_id"))).glob("*.json")):
        item = api.read(path)
        doc = item["task"]
        if item["boot_id"] != state["boot_id"]:
            raise ValueError("cost accounting receipt belongs to another boot")
        row = kinds.setdefault(doc["kind"], {"accepted_tasks": 0, "warmup_frames": 0,
                                           "scored_frames": 0, "tail_frames": 0, "total_frames": 0})
        row["accepted_tasks"] += 1
        for source, target in (("warmup", "warmup_frames"), ("scored", "scored_frames"), ("tail", "tail_frames")):
            row[target] += doc[source]
            row["total_frames"] += doc[source]
    starts, accepted, seconds, unknown = {}, set(), {}, 0
    event_path = root / "events.jsonl"
    if event_path.is_file():
        for line in event_path.read_text().splitlines():
            try:
                event = json.loads(line)
                when = datetime.fromisoformat(event["utc"].replace("Z", "+00:00"))
            except (ValueError, KeyError, TypeError):
                unknown += 1
                continue
            key = event.get("task_id")
            if event.get("event") == "session_start":
                starts[key] = (when, event.get("task_kind", "unknown"))
            elif event.get("event") == "session_accepted" and key not in accepted:
                accepted.add(key)
                if key not in starts:
                    unknown += 1
                    continue
                began, kind = starts[key]
                duration = (when - began).total_seconds()
                if duration < 0:
                    unknown += 1
                    continue
                seconds[kind] = seconds.get(kind, 0.) + duration
    contexts = {key: {"dp_seconds": value.get("seconds"),
                       "oracle_seconds": value.get("verification", {}).get("seconds")}
                for key, value in rankings.get("per_context", {}).items()}
    payload = Path(state["bundle"]) / "plans/board_payload.json"
    reused_path = Path(state["bundle"]) / "plans/reused_calibration.json"
    reused = api.read(reused_path) if reused_path.is_file() else {}
    reused_components, reused_traffic = len(reused.get("costs", {})), len(reused.get("traffic", {}))
    return {"new_module_compilations": 0, "new_executor_compilations": 0,
            "planned_components": {"total": 775, "source_imports": 514 + reused_components,
                                   "source_pilot": 514, "reused_calibration": reused_components,
                                   "new": 261 - reused_components},
            "planned_traffic_tasks": 87, "reused_traffic_tasks": reused_traffic,
            "planned_new_traffic_tasks": 87 - reused_traffic,
            "reused_source_boot_ids": reused.get("source_boot_ids", []),
            "accepted_new_task_frame_counts": kinds,
            "qualification_is_counted_separately": True,
            "imported_frames_are_not_new_board_work": True,
            "new_acquisition_seconds_by_kind": seconds or None,
            "new_acquisition_seconds_total": sum(seconds.values()) if seconds else None,
            "acquisition_clock_scope": "controller_start_to_accept_including_execution_collection_and_verification",
            "interrupted_attempt_time_not_reconstructed": True,
            "unmatched_or_unreadable_timing_events": unknown,
            "search_seconds": rankings.get("seconds"), "search_contexts": contexts,
            "board_payload_unpacked_bytes": api.read(payload).get("unpacked_bytes") if payload.is_file() else None,
            "control_thread_cpu_delta_ms": None,
            "control_thread_cpu_delta_status": "not_sampled_not_zero",
            "complete_process_TID_census": None,
            "auxiliary_thread_census_scope": "reported_stage_pools_only_main_pool_idle_helpers_not_fully_enumerated"}


def analyze(root):
    root = Path(root).resolve()
    state = validate_state(root, validate_payload=False)
    api = _common(state["bundle"])
    records = load_costs(root, state)
    screen = validated_screening(root, state)
    rankings = api.read(root / "rankings.json") if (root / "rankings.json").exists() else {}
    pool = api.read(root / "search_pool.json") if (root / "search_pool.json").exists() else {}
    sensitivity_path = root / "cpu_weight_sensitivity.json"
    sensitivity = api.read(sensitivity_path) if sensitivity_path.is_file() else None
    destination = root / "analysis"
    destination.mkdir(exist_ok=True)
    if pool:
        from .explain import explain_pool
        api.write(destination / 'pool_model_breakdown.json',
                  explain_pool(pool, records, api.read(root / 'ddr.json'), rankings), replace=True)
    measurements, measured_rows, thread_rows = {}, [], []
    if screen is not None:
        for key, item in screen["measurements"].items():
            summary = item["summary"]
            measurements[key] = {"measured_ms": summary["ii_ms"], "fps": 1000.0 / summary["ii_ms"],
                                 "process_core_ms_per_completion": summary["process_core_ms_per_completion"],
                                 "average_cpu_cores": summary["average_cpu_cores"],
                                 "p95_gap_ms": summary["p95_gap_ms"], "kept_timestamps": summary["kept_timestamps"],
                                 "completion_intervals": summary["gaps"], **_dispersion(root, item)}
            measured_rows.append({**_configuration(pool[key]), **measurements[key]})
            qualification = root / "records" / state["boot_id"] / "pipeline" / key / "qualify.json"
            if qualification.is_file():
                affinity = api.read(qualification)["summary"]
                idle_tids = {r["tid"] for r in affinity.get("idle_auxiliary_threads", [])}
                for entry in affinity.get("thread_ledger", []):
                    row = {"configuration": key, "context_id": pool[key]["context_id"],
                           **entry, "idle_auxiliary_thread": entry.get("tid") in idle_tids}
                    for field in ("requested_mask", "actual_mask"):
                        if field in row:
                            row[field] = ",".join(map(str, row[field]))
                    thread_rows.append(row)
    series = _series(rankings)
    ranking_rows, selected_rows, method_results = [], [], {}
    for method, rows in series.items():
        present = [row for row in rows if row["id"] in measurements]
        actual_order = {row["id"]: i + 1 for i, row in enumerate(sorted(present, key=lambda r: (
                        measurements[r["id"]]["measured_ms"], r["id"])))}
        entries = []
        for position, row in enumerate(rows, 1):
            entry = {"method": METHOD_NAMES[method], "method_key": method, "selection_position": position,
                     "static_rank": row.get("static_rank", position if method != "enhanced_diverse" else None),
                     **_configuration(row),
                     "predicted_ms": row["score_ms"], "measured": row["id"] in measurements,
                     "selection_reason": row.get("selection_reason", row.get("selected_reason", "static_score"))}
            if row["id"] in measurements:
                entry.update(measurements[row["id"]])
                entry["measured_rank"] = actual_order[row["id"]]
                entry["relative_prediction_error"] = row["score_ms"] / entry["measured_ms"] - 1
            ranking_rows.append(entry)
            entries.append(entry)
        budgets = {}
        for budget in ((1, 5, 20) if method.startswith("enhanced") else (1, 5)):
            candidates = rows[:budget]
            if len(candidates) != budget or any(r["id"] not in measurements for r in candidates):
                continue
            best = min(candidates, key=lambda r: (measurements[r["id"]]["measured_ms"], r["id"]))
            entry = {"method": METHOD_NAMES[method], "method_key": method, "budget": budget,
                     "selected_position": next(i + 1 for i, r in enumerate(rows) if r["id"] == best["id"]),
                     "static_rank": best.get("static_rank"), **_configuration(best),
                     "predicted_ms": best["score_ms"], **measurements[best["id"]]}
            budgets[str(budget)] = entry
            selected_rows.append(entry)
        method_results[method] = {"budgets": budgets, "selected_count": len(rows),
                                  "measured_count": len(present), "all_selected_measured": len(rows) == len(present),
                                  "rank_correlation": (_spearman([r for r in entries if r["measured"]])
                                                       if method != "enhanced_diverse" else None),
                                  "rank_scope": "pure_static_order" if method != "enhanced_diverse"
                                                else "selection_order_is_not_static_rank"}
    comparisons = {}
    for budget in ("1", "5"):
        target = method_results.get("enhanced_pure", {}).get("budgets", {}).get(budget)
        if not target:
            continue
        for baseline in ("single_core", "prefix"):
            other = method_results.get(baseline, {}).get("budgets", {}).get(budget)
            if other:
                comparisons[f"{baseline}_budget{budget}"] = {
                    "budget": int(budget), "baseline": baseline,
                    "baseline_name": METHOD_NAMES[baseline], "enhanced": target["configuration"],
                    "baseline_configuration": other["configuration"],
                    "same_configuration": target["configuration"] == other["configuration"],
                    "throughput_gain": other["measured_ms"] / target["measured_ms"] - 1}
    pure_best = method_results["enhanced_pure"]["budgets"].get("20")
    diverse_best = method_results["enhanced_diverse"]["budgets"].get("20")
    diversity_comparison = None
    if pure_best and diverse_best:
        diversity_comparison = {"budget_each": 20, "pure_configuration": pure_best["configuration"],
                                "diverse_configuration": diverse_best["configuration"],
                                "same_configuration": pure_best["configuration"] == diverse_best["configuration"],
                                "throughput_gain": pure_best["measured_ms"] / diverse_best["measured_ms"] - 1}
    context_rows = []
    for ctx in contexts():
        measured = [r for r in measured_rows if r["context_id"] == ctx["id"]]
        best = min(measured, key=lambda r: (r["measured_ms"], r["configuration"])) if measured else None
        context_rows.append({"context_id": ctx["id"], "context": ctx["label"], "family": ctx["family"],
                             "compute_pool": ",".join(map(str, ctx["compute_pool"])),
                             "host_cores": ",".join(map(str, ctx["host_mask"])),
                             "producer_cores": ",".join(map(str, ctx["control_mask"])),
                             "collector_main_cores": ",".join(map(str, ctx["control_mask"])),
                             "pure_static20_count": sum(r["context_id"] == ctx["id"] for r in series["enhanced_pure"]),
                             "diverse20_count": sum(r["context_id"] == ctx["id"] for r in series["enhanced_diverse"]),
                             "pipeline_pool_count": sum(r["context_id"] == ctx["id"] for r in pool.values()),
                             "measured_count": len(measured), "best_fps": best["fps"] if best else None,
                             "best_configuration": best["configuration"] if best else None,
                             "physical_performance_conclusion": "observed_short_window_only" if best else "not_measured_no_performance_claim"})
    reference_path = Path(state["bundle"]) / "plans/historical_references.json"
    references = api.read(reference_path) if reference_path.exists() else []
    reference_rows = [{**_configuration(c), "historical_reference_only": True,
                       "measured": c["id"] in measurements, **measurements.get(c["id"], {})} for c in references]
    coverage_complete = bool(screen is not None and set(measurements) == set(pool) and pool)
    complete = state["status"] == "COMPLETE" or (state["phase"] == "ANALYZE" and coverage_complete)
    if complete and (not coverage_complete or not all(method_results[m]["all_selected_measured"]
                                                   for m in ("enhanced_pure", "enhanced_diverse"))):
        raise ValueError("complete campaign lacks the frozen candidate measurements")
    report = {"status": "COMPLETE" if complete else "INCOMPLETE", "experiment": state["experiment"],
              "scope": "single_boot_single_input_short_window_descriptive_results",
              "boot_id": state["boot_id"], **_boot_accounting(state, records),
              "imported_component_boots_are_not_new_boots": True, "input_indices": [0], "rounds": 1,
              "significance_or_equivalence_claim": False, "performance_rejection_gates": [],
              "component_count": len(records), "imported_components": sum(r["kind"] == "imported" for r in records.values()),
              "new_components": sum(r["kind"] == "measured" for r in records.values()),
              "traffic_tasks": state.get("completed_traffic", 0),
              "source_pilot_components": state.get("source_pilot_components", 0),
              "reused_calibration_components": state.get("reused_calibration_components", 0),
              "imported_traffic": state.get("imported_traffic", 0),
              "new_traffic": state.get("new_traffic", 0),
              "measured_pipeline_configurations": len(measurements), "methods": method_results,
              "comparisons_same_budget": comparisons, "diversity_comparison_budget20": diversity_comparison,
              "contexts": context_rows, "static_domain_coverage": rankings.get("per_context", rankings.get("counts", {})),
              "historical_references": reference_rows,
              "model_assumptions": ["CPU_same_core_count_permutation_is_an_approximation",
                                    "per_worker_compute_core_time_is_uniformly_distributed",
                                    "CPU_boundary_core_time_belongs_to_caller",
                                    "shared_host_capacity_is_a_migratable_lower_bound",
                                    "producer_collector_result_processing_cost_is_not_separately_profiled",
                                    "historical_hardware_DDR_parameters_are_not_current_formal_calibration"],
              "cpu_time_scope": "whole_process_between_16_scored_completions_divided_by_15_intervals",
              "control_thread_cpu_time_status": "not_separately_sampled_not_zero",
              "model_breakdown_file": "pool_model_breakdown.json" if pool else None,
              "qualified_thread_records": len(thread_rows),
              "preparation_cost": _preparation_cost(root, state, rankings),
              "cpu_weight_sensitivity": ({k: sensitivity.get(k) for k in
                   ("scope", "uses_pipeline_fps", "changes_frozen_selection", "pool_order_changed",
                    "maximum_absolute_relative_score_change", "limitations")} if sensitivity else None),
              "ranking_uses_pipeline_feedback": False}
    components = []
    for key, item in sorted(records.items()):
        stage = item["configuration"]["stages"][0]
        components.append({"component": key, "program": stage["program"], "device": stage["device"],
                           "threads": stage["threads"], "mask": ",".join(map(str, stage["mask"])),
                           "source": item["kind"], **item["cost"]})
    api.write(destination / "analysis.json", report, replace=True)
    for name, rows in (("rankings.csv", ranking_rows), ("measured_configurations.csv", measured_rows),
                       ("budgets.csv", selected_rows), ("contexts.csv", context_rows),
                       ("components.csv", components), ("historical_references.csv", reference_rows),
                       ("thread_affinity.csv", thread_rows)):
        _csv(destination / name, rows)
    lines = ["ResNet18 九种资源上下文与细化CPU模型：短窗口结果", "状态：" + report["status"],
             f"组件{len(records)}项（引用{report['imported_components']}，新增{report['new_components']}）；"
             f"流量{report['traffic_tasks']}项（引用{report['imported_traffic']}，新增{report['new_traffic']}）；"
             f"流水实测{len(measurements)}个配置。",
             "固定输入0，一次启动、一轮；组件2帧预热＋8帧计入，流水8帧预热＋16个时间戳＋4帧尾部。",
             "不使用漂移或波动拒收，不追加测到显著；15个完成间隔不当作15次独立启动。"]
    shared_boot = report["measurement_boot_shared_with_imports"]
    lines.append("本轮测量启动与引用标定共享boot：" +
                 ("是" if shared_boot is True else "否" if shared_boot is False else "来源身份不足，未知") +
                 "；未追加独立确认启动，启动身份不同也不直接等同统计独立。")
    for ctx in context_rows:
        if ctx["measured_count"]:
            lines.append(f"{ctx['context']}：静态20中{ctx['pure_static20_count']}项，差异化20中{ctx['diverse20_count']}项，"
                         f"实测{ctx['measured_count']}项，本轮最快{ctx['best_fps']:.3f} FPS。")
        else:
            lines.append(f"{ctx['context']}：未实测，不判断其实际性能优劣。")
    for method, value in method_results.items():
        for budget, best in value["budgets"].items():
            lines.append(f"{METHOD_NAMES[method]}，预算{budget}：{best['fps']:.3f} FPS；{best['context']}；"
                         f"CPU线程{best['cpu_threads']}，核组{best['cpu_groups']}，执行模式{best['execution_mode']}。")
    for comparison in comparisons.values():
        lines.append(f"预算{comparison['budget']}，细化模型相对{comparison['baseline_name']}："
                     f"吞吐变化{comparison['throughput_gain'] * 100:+.3f}%。")
    if diversity_comparison:
        lines.append(f"相同20次预算，差异化选点相对纯静态前20：吞吐变化{diversity_comparison['throughput_gain'] * 100:+.3f}%。")
    if sensitivity:
        lines.append("逐线程核时分配敏感性仅在冻结实测池内进行，不改变选点、不读取FPS；"
                     f"池内顺序改变：{sensitivity.get('pool_order_changed')}；"
                     f"最大绝对预测变化：{sensitivity.get('maximum_absolute_relative_score_change')}。")
    lines.extend(["CPU同核数置换及逐核核时分配使用显式近似；控制线程开销尚未单独标定，不能视作零。",
                  "历史配置单列复测；旧FPS属于原输入与窗口，不与本批新FPS混作同口径结果。"])
    (destination / "REPORT_zh.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
