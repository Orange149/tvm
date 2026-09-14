"""One bounded stage/tile iteration on a frozen candidate shortlist.

This pilot measures isolated VTA stages, tunes shared workloads, and re-ranks the
same candidate pool. CPU/boundary costs remain frozen. It does not claim full-space
search, a new pipeline FPS measurement, or a globally optimal schedule.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import tvm
from tvm import autotvm, relay, rpc
from tvm.contrib import graph_executor
from mxnet.gluon.model_zoo import vision
import vta

from profile_split_resnet18_stages import build_vta_stage, create_stage_module, export_and_upload
from split_resnet18_stages import (build_resnet18_unit_blocks, build_resnet18_unit_metadata,
                                  relay_inputs_for_stage, make_stage_block, lower_stage_to_relay)
from tune_resnet18_vta import register_vta_conv2d_template, tune_tasks
from vta_autotvm_measure import VTASequentialBuilder, VTADirectRunner


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def stage_key(stage):
    return "__".join(stage["unit_names"])


def inner_signature(entities):
    return tuple((name, value.size[-1] if hasattr(value, "size") else value.val)
                 for name, value in entities.items())


def local_candidates(task, incumbent, limit=8):
    """Keep the incumbent and choose diverse, one-axis schedule neighbours.

    One nearest candidate is selected for each of tile_h, tile_w, tile_ci,
    tile_co, oc_nthread and h_nthread before a second spatial/channel neighbour
    is considered.  This is a budget allocation rule, not a legality claim:
    the real builder and independent reference remain authoritative.
    """
    if limit < 1:
        raise ValueError("Candidate limit must include at least the incumbent")
    base = {name: value[-1] if kind == "sp" else value
            for name, kind, value in incumbent["entity"]}
    matched, pools = [], {name: [] for name in (
        "tile_h", "tile_w", "tile_ci", "tile_co", "oc_nthread", "h_nthread"
    )}
    for index in range(task.config_space.range_length):
        cfg = task.config_space.get(index)
        values = dict(inner_signature(cfg._entity_map))
        if values == base:
            matched.append((index, cfg))
            continue
        changed = [name for name in base if values[name] != base[name]]
        if len(changed) != 1 or changed[0] not in pools:
            continue
        axis = changed[0]
        old, new = base[axis], values[axis]
        if axis.startswith("tile_"):
            distance = abs(math.log(float(new) / float(old), 2))
            # At equal multiplicative distance, first test the coarser tile.
            direction = 0 if new > old else 1
        else:
            distance, direction = abs(new - old), 0
        pools[axis].append((distance, direction, index, cfg))
    if not matched:
        raise RuntimeError("Deployed incumbent cannot be represented in AutoTVM space")
    selected = matched[:1]
    axis_order = ("tile_w", "tile_h", "tile_co", "tile_ci", "oc_nthread", "h_nthread")
    for axis in axis_order:
        if len(selected) >= limit or not pools[axis]:
            continue
        _, _, index, cfg = min(pools[axis], key=lambda row: row[:3])
        selected.append((index, cfg))
    remaining = sorted(
        (distance, direction, axis_order.index(axis), index, cfg)
        for axis in axis_order for distance, direction, index, cfg in pools[axis]
        if index not in {chosen_index for chosen_index, _ in selected}
    )
    selected.extend((index, cfg) for _, _, _, index, cfg in remaining[:limit - len(selected)])
    allowed = {inner_signature(c._entity_map) for _, c in selected}
    task.config_space.multi_filter(lambda entities: inner_signature(entities) in allowed)
    return [i for i, _ in selected]


def candidate_manifest(task, indices, incumbent_index):
    """Describe the frozen bounded neighbourhood before board measurement."""
    base = dict(inner_signature(task.config_space.get(incumbent_index)._entity_map))
    rows = []
    for index in indices:
        config = task.config_space.get(index)
        values = dict(inner_signature(config._entity_map))
        rows.append({
            "config_index": index,
            "is_incumbent": index == incumbent_index,
            "changed_axes": [name for name in base if values[name] != base[name]],
            "inner_factors": values,
            "config": config.to_json_dict(),
        })
    return rows


def projection_repair_candidates(task, limit=2):
    """Bounded diagnostic alternatives, not a proven DMA pruning rule.

    For 1x1 stride-2 projections try full output width with shallow heights.
    The real compiler still checks SRAM capacity and the runner checks values.
    """
    data, weight, strides, padding = task.args[:4]
    if tuple(weight[1][2:4]) != (1, 1) or tuple(strides) != (2, 2) or any(padding):
        return []
    oh, ow = (int(data[1][2]) + 1) // 2, (int(data[1][3]) + 1) // 2
    heights = {1, max(h for h in range(1, min(7, oh) + 1) if oh % h == 0)}
    selected = []
    for index in range(task.config_space.range_length):
        cfg = task.config_space.get(index)
        values = dict(inner_signature(cfg._entity_map))
        if (values["tile_h"] in heights and values["tile_w"] == ow
                and all(values[k] == 1 for k in ("tile_b", "tile_ci", "tile_co", "h_nthread"))
                and values["oc_nthread"] == 2):
            selected.append((index, cfg))
    selected = selected[:limit]
    allowed = {inner_signature(c._entity_map) for _, c in selected}
    task.config_space.multi_filter(lambda entities: inner_signature(entities) in allowed)
    return [i for i, _ in selected]


def successful_record_map(paths):
    """Return successful records keyed by canonical workload and config index."""
    result = {}
    for path in paths:
        if not path or not Path(path).exists():
            continue
        for inp, measured in autotvm.record.load_from_file(str(path)):
            if measured.error_no == 0:
                key = (json.dumps(inp.task.workload, separators=(",", ":")), inp.config.index)
                result[key] = (inp, measured)
    return result


def write_validated_baseline(path, selected, repair_selection, record_paths):
    """Select one correctness-passing baseline record for every workload.

    Prefer the pre-existing dispatched incumbent when it passed.  If it failed, take the
    first correctness-passing repair in declared order.  Timing never decides
    this baseline, so it remains distinct from the tuned best log.
    """
    records = successful_record_map(record_paths)
    repairs = {json.dumps(item["workload"], separators=(",", ":")): item["selected_indices"]
               for item in repair_selection}
    chosen, manifest = [], []
    for item in selected:
        workload_key = json.dumps(item["workload"], separators=(",", ":"))
        candidates = [item["selected_indices"][0]] + repairs.get(workload_key, [])
        match = next((records[(workload_key, index)] for index in candidates
                      if (workload_key, index) in records), None)
        if match is None:
            raise RuntimeError("No validated baseline record for workload " + workload_key)
        inp, measured = match
        chosen.append(autotvm.record.encode(inp, measured))
        manifest.append({"workload": item["workload"], "config_index": inp.config.index,
                         "origin": "dispatched_incumbent" if inp.config.index == candidates[0]
                                   else "correctness_repair"})
    Path(path).write_text("\n".join(chosen) + "\n", encoding="utf-8")
    return manifest


def write_safe_overlay(path, selected, record_paths, min_improvement=0.02):
    """Write only candidates that beat a passing incumbent in the same runner.

    A failed direct-runner incumbent is not replaced: full Relay stages already
    establish TopHub as the deployable baseline, while the isolated template can
    differ in fusion and checking semantics.  Missing overlay records deliberately
    fall through to TopHub during stage compilation.
    """
    records = successful_record_map(record_paths)
    encoded, manifest = [], []
    for item in selected:
        workload_key = json.dumps(item["workload"], separators=(",", ":"))
        incumbent_index = item["incumbent_index"]
        incumbent = records.get((workload_key, incumbent_index))
        row = {"workload": item["workload"], "incumbent_index": incumbent_index,
               "selected_index": incumbent_index, "overridden": False}
        if incumbent is None:
            row["reason"] = "incumbent_failed_direct_runner_keep_tophub"
            manifest.append(row)
            continue
        incumbent_ms = float(np.mean(incumbent[1].costs)) * 1000
        passing = [records[(workload_key, index)] for index in item["selected_indices"]
                   if (workload_key, index) in records]
        best = min(passing, key=lambda pair: float(np.mean(pair[1].costs)))
        best_ms = float(np.mean(best[1].costs)) * 1000
        row.update({"incumbent_mean_ms": incumbent_ms, "best_mean_ms": best_ms,
                    "selected_index": best[0].config.index,
                    "relative_improvement": (incumbent_ms - best_ms) / incumbent_ms})
        if (best[0].config.index != incumbent_index
                and best_ms <= incumbent_ms * (1.0 - min_improvement)):
            encoded.append(autotvm.record.encode(*best))
            row.update({"overridden": True, "reason": "measured_candidate_improves_incumbent"})
        else:
            row.update({"selected_index": incumbent_index,
                        "reason": "no_candidate_cleared_improvement_gate"})
        manifest.append(row)
    if encoded:
        Path(path).write_text("\n".join(encoded) + "\n", encoding="utf-8")
    return manifest


def rerank(rows, measurements, phase):
    """Refresh VTA run and measured payload terms without double-counting DMA time."""
    result = []
    for row in rows:
        resource = row["resource_load_ms"]
        vta_ms, dma_ms = 0.0, 0.0
        for stage in row["scheme_cfg"]:
            if stage["device"] == "vta":
                measured = measurements[stage_key(stage)][phase]
                vta_ms += measured["median_ms"]
                profile = measured["runtime_profile"]
                rates = row["cost_provenance"]["vta_dma_qualification"]
                dma_ms += profile["load_buffer_2d_bytes"] / (rates["load_bandwidth_GBps"] * 1e6)
                dma_ms += profile["store_buffer_2d_bytes"] / (rates["store_bandwidth_GBps"] * 1e6)
        cpu_ddr = sum(s["shared_ddr_demand"]["service_ms"] for s in row["stages"] if s["device"] == "cpu")
        components = {"max_cpu_stage": resource["max_cpu_stage"],
                      "cpu_pool": max(resource["cpu_physical_pool_lower_bound"], resource["cpu_prefix_mask_lower_bound"]),
                      "vta_and_boundary": vta_ms + resource["vta_mutex_boundary_work_total"],
                      "ddr": cpu_ddr + dma_ms}
        result.append({"candidate_id": row["candidate_id"], "topology_id": row["topology_id"],
                       "original_rank": row["rank"], "components_ms": components,
                       "score_ms": max(components.values())})
    result.sort(key=lambda r: (r["score_ms"], r["candidate_id"]))
    for rank, row in enumerate(result, 1):
        row["rank"] = rank
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--ranking", default="vta/tutorials/frontend/report_out/resource_aware_maxplus/v1_p5b_iteration2_ranked_candidates.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--configs-per-task", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--reuse-candidate-dir", default="",
                        help="Reuse raw candidate measurements from a compatible prior run")
    parser.add_argument("--min-candidate-improvement", type=float, default=0.02)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("Choose a new output directory to preserve experiment evidence")
    if args.configs_per_task < 1 or args.repeat < 1:
        raise ValueError("Positive configuration and repeat budgets are required")
    output.mkdir(parents=True)
    rows = json.loads(Path(args.ranking).read_text())["rows"]
    stages = {stage_key(s): s for r in rows for s in r["scheme_cfg"] if s["device"] == "vta"}
    env = vta.get_env()
    full_model = vision.get_model("resnet18_v1", pretrained=True)
    features, head = list(full_model.features), full_model.output
    units = build_resnet18_unit_blocks(features, head)
    metadata = build_resnet18_unit_metadata(units, env.BATCH, 224)
    register_vta_conv2d_template()
    lowered, inputs, references, measurements, workload_rows = {}, {}, {}, {}, {}
    for key, stage in stages.items():
        names = relay_inputs_for_stage(stage, metadata)
        block = make_stage_block(features, head, stage, unit_blocks=units)
        lowered[key] = lower_stage_to_relay(block, names)
        rng = np.random.default_rng(0)
        inputs[key] = {name: rng.uniform(-1, 1, shape).astype("float32") for name, shape in names}

        mod, params = lowered[key]
        with tvm.transform.PassContext(opt_level=3):
            with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[]):
                quantized = relay.quantize.quantize(mod, params=params)
        with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            factory = relay.build(quantized, target="llvm", params=params)
        reference = graph_executor.GraphModule(factory["default"](tvm.cpu()))
        reference.set_input(**inputs[key])
        reference.run()
        references[key] = [reference.get_output(i).numpy()
                           for i in range(reference.get_num_outputs())]

    def measure_stage(key, phase, log=""):
        stage = stages[key]
        mod, params = lowered[key]
        audit = {}
        graph, lib, params = build_vta_stage(stage["name"], mod["main"], params, env,
            tune_log=log, tuning_audit=audit, require_tuned=bool(log))
        remote = rpc.connect(args.host, args.port, session_timeout=180)
        remote.get_function("runtime.config_threadpool")(1, 1)
        library = export_and_upload(lib, remote, env, "stage_tile_" + phase)
        executor, ctx = create_stage_module(stage["name"], "vta", graph, library, remote)
        executor.set_input(**params)
        for name, data in inputs[key].items():
            executor.set_input(name, data)
        executor.run()
        actual = [executor.get_output(i).numpy() for i in range(executor.get_num_outputs())]
        if len(actual) != len(references[key]):
            raise AssertionError("Reference output arity mismatch")
        comparisons = [{"elements": value.size,
            "mismatched": int(np.count_nonzero(value != expected)),
            "max_abs_error": float(np.max(np.abs(value.astype("float64") - expected.astype("float64"))))}
            for value, expected in zip(actual, references[key])]
        reference_correct = not any(c["mismatched"] for c in comparisons)
        if phase != "fallback_probe" and not reference_correct:
                write_json(output / "iteration_failure.json", {
                    "status": "failed_stage_reference", "phase": phase,
                    "stage": stage, "tuning": audit,
                    "comparisons": comparisons, "pipeline_fps_measured": False,
                    "ranking_after_valid": False})
                raise AssertionError("Validated stage differs from CPU reference; see iteration_failure.json")
        costs = executor.module.time_evaluator("run", ctx[0], number=1, repeat=args.repeat)().results
        remote.get_function("vta.runtime.profiler_clear")()
        executor.run()
        profile = json.loads(remote.get_function("vta.runtime.profiler_status")())
        item = {"median_ms": float(np.median(costs)) * 1000, "costs_s": list(costs),
                "runtime_profile": profile, "tuning": audit,
                "comparison": "same_quantized_relay_llvm_reference",
                "reference_correct": reference_correct, "reference_comparisons": comparisons,
                "reference_output_sha256": [hashlib.sha256(x.tobytes()).hexdigest()
                                            for x in references[key]],
                "output_sha256": [hashlib.sha256(x.tobytes()).hexdigest() for x in actual]}
        measurements.setdefault(key, {})[phase] = item
        write_json(output / "stage_measurements.json", measurements)
        for entry in audit["rows"]:
            workload_rows[json.dumps(entry["workload"])] = entry
        print("[STAGE] {} {} {:.3f} ms, tuned={}/{}".format(
            phase, stage["unit_names"][-1], item["median_ms"],
            audit["tuned_workloads"], audit["queried_workloads"]), flush=True)

    for key in stages:
        measure_stage(key, "incumbent_probe")
    tasks, selected = [], []
    for entry in workload_rows.values():
        workload = entry["workload"]
        if workload[0] != "conv2d_packed.vta":
            raise RuntimeError("Unsupported tuning task: " + workload[0])
        task = autotvm.task.create(workload[0], args=workload[1:], target=env.target, target_host=env.target_host)
        if entry.get("is_fallback"):
            raise RuntimeError("TopHub/history did not provide an incumbent for " + str(workload))
        indices = local_candidates(task, entry["config"], args.configs_per_task)
        tasks.append(task)
        selected.append({
            "workload": task.workload,
            "incumbent_index": indices[0],
            "incumbent_source": "tophub_or_history_dispatch",
            "selected_indices": indices,
            "candidates": candidate_manifest(task, indices, indices[0]),
        })
    write_json(output / "selected_tasks.json", selected)
    artifacts = output / "tuning_artifacts"
    initial_log = str(output / "tuning_initial.log")
    if args.reuse_candidate_dir:
        source = Path(args.reuse_candidate_dir)
        prior_selected = json.loads((source / "selected_tasks.json").read_text())
        canonical = lambda item: (json.dumps(item["workload"], separators=(",", ":")),
                                  item["selected_indices"])
        compatible = [canonical(x) for x in prior_selected] == [canonical(x) for x in selected]
        if not compatible:
            raise RuntimeError("Reused candidate manifest does not match this frozen search")
        shutil.copyfile(source / "tuning_initial.log.tmp", initial_log + ".tmp")
        shutil.copyfile(source / "tuning_initial.log", initial_log)
    else:
        options = SimpleNamespace(log_file=initial_log, trials=args.configs_per_task,
                                  tuner="gridsearch", early_stopping=0, rpc_host=args.host)
        tune_tasks(tasks, options, autotvm.measure_option(VTASequentialBuilder(artifacts),
            VTADirectRunner(args.host, args.port, number=1, repeat=args.repeat,
                            artifact_dir=artifacts)))
    overlay_log = str(output / "safe_overlay.log")
    overlay_manifest = write_safe_overlay(
        overlay_log, selected, [initial_log + ".tmp"], args.min_candidate_improvement)
    write_json(output / "safe_overlay_manifest.json", overlay_manifest)
    override_count = sum(item["overridden"] for item in overlay_manifest)
    for key in stages:
        measurements[key]["tophub_baseline"] = measurements[key]["incumbent_probe"]
        if override_count:
            measure_stage(key, "tuned_overlay", overlay_log)
        else:
            measurements[key]["tuned_overlay"] = measurements[key]["incumbent_probe"]
        baseline = measurements[key]["tophub_baseline"]
        tuned = measurements[key]["tuned_overlay"]
        accepted = tuned if tuned["reference_correct"] and tuned["median_ms"] <= baseline["median_ms"] else baseline
        measurements[key]["accepted"] = dict(accepted)
        measurements[key]["accepted"]["selection"] = (
            "tuned_overlay" if accepted is tuned else "tophub_baseline")
    write_json(output / "stage_measurements.json", measurements)
    before = rerank(rows, measurements, "tophub_baseline")
    after = rerank(rows, measurements, "accepted")
    write_json(output / "ranking_before.json", before)
    write_json(output / "ranking_after.json", after)
    write_json(output / "iteration_summary.json", {
        "status": "completed", "scope": "one_iteration_within_frozen_shortlist",
        "candidate_count": len(rows), "topology_count": len({r["topology_id"] for r in rows}),
        "unique_vta_stages": len(stages), "unique_tuning_workloads": len(tasks),
        "candidate_measurements_reused_from": args.reuse_candidate_dir or None,
        "safe_overlay_overrides": override_count,
        "min_candidate_improvement": args.min_candidate_improvement,
        "all_tophub_baseline_stages_match_cpu_reference": True,
        "all_accepted_stages_match_cpu_reference": True,
        "before_top1": before[0], "after_top1": after[0],
        "top1_changed": before[0]["candidate_id"] != after[0]["candidate_id"],
        "pipeline_fps_measured": False, "cpu_and_boundary_costs": "historical_frozen",
        "vta_host_core_work": "historical_frozen_not_reidentified",
        "new_stage_inputs": "deterministic_uniform_float32_not_end_to_end_activations",
        "board": args.host, "rpc_port": args.port,
        "ranking_sha256": hashlib.sha256(Path(args.ranking).read_bytes()).hexdigest(),
    })


if __name__ == "__main__":
    main()
