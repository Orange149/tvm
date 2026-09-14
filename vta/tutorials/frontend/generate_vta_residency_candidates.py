#!/usr/bin/env python3
"""Map the archived VTA neighborhood into the isolated residency template.

This tool is deliberately local-only.  It lowers static TIR and extracts logical
DMA requests; it never builds/runs FSim code, contacts RPC, or treats simulator
time as a performance measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import tvm
from tvm import autotvm
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from c3_candidate_identity import (
    candidate_identity_record,
    canonical_json_bytes,
    make_failure,
    normalize_config_entity,
)
from extract_static_vta_dma import SMALL_BYTES, extract_module_dma, tensor_bytes
from extract_vta_candidate_features import _descriptor_aggregates
from tune_resnet18_vta import register_vta_conv2d_template


POOL_SCHEMA = "c3_residency_candidate_pool_v1"
RESULT_SCHEMA = "c3_residency_static_feature_v1"
SUMMARY_SCHEMA = "c3_residency_static_summary_v1"
TEMPLATE_NAME = "conv2d_packed_residency.vta"
MODE_NUMBERS = {
    "original": 0,
    "input_stationary": 1,
    "weight_stationary": 2,
    "paper_inspired_hybrid": 3,
}
EXPERIMENTAL_MODES = tuple(name for name in MODE_NUMBERS if name != "original")
FORCED_KNOBS = {"oc_nthread": 1, "h_nthread": 1}


def load_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_config_key(config):
    """Return the complete semantic ConfigEntity key (debug index excluded)."""

    return canonical_json_bytes(normalize_config_entity(config)).decode("utf-8")


def force_experimental_knobs(config):
    """Copy a full serialized ConfigEntity and force only the experiment knobs."""

    copied = json.loads(json.dumps(config))
    seen = set()
    for entity in copied.get("entity", []):
        if len(entity) != 3:
            raise ValueError("malformed ConfigEntity row: {!r}".format(entity))
        name = entity[0]
        if name in FORCED_KNOBS:
            if entity[1] != "ot":
                raise ValueError("{} is not an option knob".format(name))
            entity[2] = FORCED_KNOBS[name]
            seen.add(name)
    missing = set(FORCED_KNOBS) - seen
    if missing:
        raise ValueError("ConfigEntity missing forced knobs: {}".format(sorted(missing)))
    copied.pop("index", None)
    return copied


def build_config_lookup(config_space):
    """Index a ConfigSpace by its complete semantic entity, rejecting ambiguity."""

    lookup = {}
    for index in range(len(config_space)):
        config = config_space.get(index).to_json_dict()
        key = semantic_config_key(config)
        if key in lookup:
            raise ValueError("duplicate semantic ConfigEntity at indices {} and {}".format(
                lookup[key]["index"], index
            ))
        lookup[key] = {"index": index, "config": config}
    return lookup


def map_selected_candidates(selected, config_lookup):
    """Force vthreads, exact-match into residency ConfigSpace, and deduplicate."""

    mapped = {}
    source_count = 0
    for source in selected["candidates"]:
        source_count += 1
        forced = force_experimental_knobs(source["config"])
        key = semantic_config_key(forced)
        if key not in config_lookup:
            raise ValueError(
                "no exact residency ConfigEntity for historical index {}".format(
                    source["config_index"]
                )
            )
        match = config_lookup[key]
        if semantic_config_key(match["config"]) != key:
            raise AssertionError("ConfigEntity lookup was not exact")
        record = mapped.setdefault(
            key,
            {
                "mapped_config_index": int(match["index"]),
                "complete_config_entity": match["config"],
                "sources": [],
            },
        )
        record["sources"].append(
            {
                "historical_config_index": int(source["config_index"]),
                "is_historical_incumbent": bool(source.get("is_incumbent", False)),
                "changed_axes": list(source.get("changed_axes", [])),
            }
        )
    records = sorted(mapped.values(), key=lambda row: row["mapped_config_index"])
    return {
        "source_count": source_count,
        "unique_count": len(records),
        "deduplicated_count": source_count - len(records),
        "records": records,
    }


def create_residency_task(workload, mode_number, env):
    return autotvm.task.create(
        TEMPLATE_NAME,
        args=tuple(workload[1:]) + (int(mode_number),),
        target=env.target,
        target_host=env.target_host,
    )


def create_original_task(workload, env):
    return autotvm.task.create(
        workload[0],
        args=workload[1:],
        target=env.target,
        target_host=env.target_host,
    )


def unique_tensor_bytes(workload):
    data, weight = workload[1], workload[2]
    data_shape, weight_shape = data[1], weight[1]
    stride, padding = workload[3], workload[4]
    output_h = (data_shape[2] + padding[0] + padding[2] - weight_shape[2]) // stride[0] + 1
    output_w = (data_shape[3] + padding[1] + padding[3] - weight_shape[3]) // stride[1] + 1
    output_bytes = (
        data_shape[0] * weight_shape[0] * output_h * output_w * data_shape[4] * weight_shape[4]
    )
    return {
        "input_bytes": tensor_bytes(data),
        "weight_bytes": tensor_bytes(weight),
        "output_bytes": int(output_bytes),
    }


def transfer_signature(dma, unique_bytes):
    totals = dma["totals"]
    return {
        "small_request_threshold_bytes": SMALL_BYTES,
        "totals": totals,
        "max_request_bytes_by_memory": dma.get("max_request_bytes_by_memory", {}),
        "unique_tensor_bytes": unique_bytes,
        "reload_ratio": {
            "input_load": totals.get("load_buffer_2d_inp_bytes", 0) / unique_bytes["input_bytes"],
            "weight_load": totals.get("load_buffer_2d_wgt_bytes", 0) / unique_bytes["weight_bytes"],
            "output_store": totals.get("store_buffer_2d_out_bytes", 0) / unique_bytes["output_bytes"],
        },
        "descriptor_aggregates": _descriptor_aggregates(dma.get("request_descriptors", [])),
    }


def classify_lower_failure(message, phase):
    text = message.lower()
    if "allocation exceed bound" in text or "allocated size" in text:
        return "allocation_capacity"
    if "pad on innermost block" in text or "pad on the innermost block" in text:
        return "dma_pad_innermost"
    if "cannot detect 2d pattern" in text:
        return "dma_2d_pattern"
    if "cannot prove compact" in text or "compact buffer" in text:
        return "dma_compact_buffer"
    if "kstorestage" in text or "storestage" in text and "loadstage" in text:
        return "dependency_store_load"
    if "dynamic" in text and "dma" in text:
        return "dynamic_dma_descriptor"
    return "schedule_instantiate" if phase == "instantiate" else "other_lower"


def lower_static(task, config, env, unique_bytes):
    phase = "instantiate"
    try:
        with task.target:
            schedule, tensors = task.instantiate(config)
        phase = "tir_lower"
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            module = tvm.lower(schedule, tensors, name="main")
        script = tvm.ir.save_json(module)
        phase = "static_dma_extract"
        dma = extract_module_dma(module, env)
        return {
            "status": "ok",
            "failure": None,
            "tir_sha256": hashlib.sha256(script.encode("utf-8")).hexdigest(),
            "tir_hash_encoding": "tvm.ir.save_json",
            "transfer_signature": transfer_signature(dma, unique_bytes),
        }
    except Exception as error:  # TVM raises several non-shared exception types.
        message = str(error)
        failure = make_failure("lower", message[:4000] or type(error).__name__, phase=phase)
        failure["subcategory"] = classify_lower_failure(message, phase)
        failure["exception_type"] = type(error).__name__
        return {
            "status": "failed",
            "failure": failure,
            "tir_sha256": None,
            "tir_hash_encoding": "tvm.ir.save_json",
            "transfer_signature": None,
        }


def dma_change(candidate, baseline):
    if candidate["status"] != "ok" or baseline["status"] != "ok":
        return None
    candidate_totals = candidate["transfer_signature"]["totals"]
    baseline_totals = baseline["transfer_signature"]["totals"]
    answer = {}
    for memory in ("inp", "wgt"):
        key = "load_buffer_2d_{}_bytes".format(memory)
        before, after = int(baseline_totals.get(key, 0)), int(candidate_totals.get(key, 0))
        answer[memory] = {
            "original_bytes": before,
            "candidate_bytes": after,
            "delta_bytes": after - before,
            "ratio_to_original": None if before == 0 else after / before,
            "reduction_fraction": None if before == 0 else (before - after) / before,
        }
    return answer


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values):
    return {
        "count": len(values),
        "min": percentile(values, 0.0),
        "p25": percentile(values, 0.25),
        "median": percentile(values, 0.5),
        "p75": percentile(values, 0.75),
        "max": percentile(values, 1.0),
        "positive": sum(value > 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "negative": sum(value < 0 for value in values),
    }


def summarize(results, mapping_diagnostics):
    modes = {}
    for mode in EXPERIMENTAL_MODES:
        rows = [row for row in results if row.get("residence_mode") == mode]
        failures = Counter(
            row["failure"]["subcategory"] for row in rows if row["status"] != "ok"
        )
        input_reductions = [
            row["relative_to_same_tile_original"]["inp"]["reduction_fraction"]
            for row in rows
            if row.get("relative_to_same_tile_original") is not None
        ]
        weight_reductions = [
            row["relative_to_same_tile_original"]["wgt"]["reduction_fraction"]
            for row in rows
            if row.get("relative_to_same_tile_original") is not None
        ]
        if mode == "input_stationary":
            target = input_reductions
            target_definition = "input_load_bytes"
        elif mode == "weight_stationary":
            target = weight_reductions
            target_definition = "weight_load_bytes"
        else:
            target = []
            for row in rows:
                change = row.get("relative_to_same_tile_original")
                if change is None:
                    continue
                before = change["inp"]["original_bytes"] + change["wgt"]["original_bytes"]
                after = change["inp"]["candidate_bytes"] + change["wgt"]["candidate_bytes"]
                if before:
                    target.append((before - after) / before)
            target_definition = "input_plus_weight_load_bytes"
        modes[mode] = {
            "attempted": len(rows),
            "lower_ok": sum(row["status"] == "ok" for row in rows),
            "lower_failed": sum(row["status"] != "ok" for row in rows),
            "legality_rate": (
                sum(row["status"] == "ok" for row in rows) / len(rows) if rows else None
            ),
            "failure_subcategories": dict(sorted(failures.items())),
            "input_reduction_fraction": distribution(input_reductions),
            "weight_reduction_fraction": distribution(weight_reductions),
            "target_dma_definition": target_definition,
            "target_dma_reduction_fraction": distribution(target),
        }
    workload_rows = {}
    for workload_id in sorted({row["workload_id"] for row in results}):
        workload_results = [row for row in results if row["workload_id"] == workload_id]
        workload_rows[workload_id] = {
            "protected_original_ok": any(
                row["candidate_role"] == "protected_original_incumbent" and row["status"] == "ok"
                for row in workload_results
            ),
            "mapped_tile_controls": sum(
                row["candidate_role"] == "same_tile_original_control" for row in workload_results
            ),
            "modes": {
                mode: {
                    "attempted": sum(row.get("residence_mode") == mode for row in workload_results),
                    "lower_ok": sum(
                        row.get("residence_mode") == mode and row["status"] == "ok"
                        for row in workload_results
                    ),
                }
                for mode in EXPERIMENTAL_MODES
            },
        }
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "scope": "local static TIR lowering and logical DMA requests; no FSim timing or board data",
        "mapping": mapping_diagnostics,
        "result_records": len(results),
        "modes": modes,
        "workloads": workload_rows,
    }


def parse_args():
    base = Path(__file__).resolve().parent / "report_out" / "stage_tile_cotuning"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-tasks",
        default=str(base / "iteration6_safe_overlay_final" / "selected_tasks.json"),
    )
    parser.add_argument(
        "--hardware-manifest",
        default=str(
            base
            / "c3_dma_residency_autotune"
            / "04_dma_command_signatures"
            / "20260910_p4_local_identity_adapter_run01"
            / "manifest.json"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = load_json(args.selected_tasks)
    hardware_manifest = load_json(args.hardware_manifest)
    hardware_fingerprint = hardware_manifest["hardware_fingerprint"]
    env = vta.get_env()
    register_vta_conv2d_template()

    source_dir = Path(__file__).resolve().parent
    vta_root = source_dir.parents[1]
    original_source = vta_root / "python" / "vta" / "top" / "vta_conv2d.py"
    residency_source = vta_root / "python" / "vta" / "top" / "vta_conv2d_residency.py"
    original_version = "vta_conv2d.py@{}".format(sha256_file(original_source))
    residency_version = "vta_residency@{}+{}".format(
        sha256_file(residency_source), sha256_file(original_source)
    )

    pool_workloads = []
    results = []
    mapping_totals = Counter()
    for workload_ordinal, selected_workload in enumerate(selected):
        workload_id = "W{:02d}".format(workload_ordinal)
        workload = selected_workload["workload"]
        unique_bytes = unique_tensor_bytes(workload)

        residency_tasks = {
            mode: create_residency_task(workload, number, env)
            for mode, number in MODE_NUMBERS.items()
        }
        lookup = build_config_lookup(residency_tasks["original"].config_space)
        mapping = map_selected_candidates(selected_workload, lookup)
        mapping_totals.update(
            {
                "historical_sources": mapping["source_count"],
                "mapped_unique": mapping["unique_count"],
                "deduplicated": mapping["deduplicated_count"],
            }
        )

        original_task = create_original_task(workload, env)
        incumbent_index = int(selected_workload["incumbent_index"])
        incumbent_config = original_task.config_space.get(incumbent_index)
        incumbent_json = incumbent_config.to_json_dict()
        protected_identity = candidate_identity_record(
            hardware_fingerprint,
            workload[0],
            original_version,
            workload,
            "original",
            incumbent_json,
            config_index=incumbent_index,
        )
        protected_result = {
            "schema": RESULT_SCHEMA,
            **protected_identity,
            "workload_id": workload_id,
            "candidate_role": "protected_original_incumbent",
            "protected": True,
            "residence_mode": "original",
            **lower_static(original_task, incumbent_config, env, unique_bytes),
            "relative_to_same_tile_original": None,
        }
        results.append(protected_result)

        mapped_pool = []
        for mapped in mapping["records"]:
            mapped_index = int(mapped["mapped_config_index"])
            config_json = mapped["complete_config_entity"]
            baseline_identity = candidate_identity_record(
                hardware_fingerprint,
                TEMPLATE_NAME,
                residency_version,
                workload,
                "original",
                config_json,
                config_index=mapped_index,
            )
            baseline = {
                "schema": RESULT_SCHEMA,
                **baseline_identity,
                "workload_id": workload_id,
                "candidate_role": "same_tile_original_control",
                "protected": False,
                "residence_mode": "original",
                "source_candidates": mapped["sources"],
                **lower_static(
                    residency_tasks["original"],
                    residency_tasks["original"].config_space.get(mapped_index),
                    env,
                    unique_bytes,
                ),
                "relative_to_same_tile_original": None,
            }
            results.append(baseline)

            mode_ids = {"original": baseline["candidate_id"]}
            for mode in EXPERIMENTAL_MODES:
                mode_config = residency_tasks[mode].config_space.get(mapped_index)
                if semantic_config_key(mode_config.to_json_dict()) != semantic_config_key(config_json):
                    raise ValueError(
                        "residency mode {} changed ConfigSpace mapping at index {}".format(
                            mode, mapped_index
                        )
                    )
                identity = candidate_identity_record(
                    hardware_fingerprint,
                    TEMPLATE_NAME,
                    residency_version,
                    workload,
                    mode,
                    config_json,
                    config_index=mapped_index,
                )
                lowered = lower_static(
                    residency_tasks[mode],
                    mode_config,
                    env,
                    unique_bytes,
                )
                record = {
                    "schema": RESULT_SCHEMA,
                    **identity,
                    "workload_id": workload_id,
                    "candidate_role": "residency_experiment",
                    "protected": False,
                    "residence_mode": mode,
                    "source_candidates": mapped["sources"],
                    **lowered,
                }
                record["relative_to_same_tile_original"] = dma_change(record, baseline)
                results.append(record)
                mode_ids[mode] = record["candidate_id"]
            mapped_pool.append(
                {
                    **mapped,
                    "candidate_ids": mode_ids,
                }
            )

        pool_workloads.append(
            {
                "workload_id": workload_id,
                "workload": workload,
                "protected_original_incumbent": {
                    "candidate_id": protected_identity["candidate_id"],
                    "config_index": incumbent_index,
                    "complete_config_entity": incumbent_json,
                },
                "mapping": {
                    "historical_source_count": mapping["source_count"],
                    "mapped_unique_count": mapping["unique_count"],
                    "deduplicated_count": mapping["deduplicated_count"],
                    "records": mapped_pool,
                },
            }
        )

    candidate_ids = [row["candidate_id"] for row in results]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate pool contains duplicate stable IDs")

    mapping_diagnostics = {
        "workload_count": len(pool_workloads),
        "historical_source_candidates": mapping_totals["historical_sources"],
        "mapped_unique_tile_entities": mapping_totals["mapped_unique"],
        "deduplicated_historical_sources": mapping_totals["deduplicated"],
        "protected_original_incumbents": len(pool_workloads),
        "experimental_modes_per_tile": len(EXPERIMENTAL_MODES),
        "experimental_candidates": mapping_totals["mapped_unique"] * len(EXPERIMENTAL_MODES),
    }
    pool = {
        "schema": POOL_SCHEMA,
        "template_name": TEMPLATE_NAME,
        "schedule_version": residency_version,
        "forced_experimental_knobs": FORCED_KNOBS,
        "hardware_fingerprint": hardware_fingerprint,
        "mapping_diagnostics": mapping_diagnostics,
        "workloads": pool_workloads,
    }
    summary = summarize(results, mapping_diagnostics)

    (output_dir / "candidate_pool.json").write_text(json.dumps(pool, indent=2) + "\n")
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results)
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    command = " ".join([sys.executable] + sys.argv)
    (output_dir / "command.txt").write_text(command + "\n")
    (output_dir / "stdout.log").write_text(
        "workloads={workload_count} historical={historical_source_candidates} "
        "mapped_unique={mapped_unique_tile_entities} experimental={experimental_candidates}\n".format(
            **mapping_diagnostics
        )
    )
    (output_dir / "stderr.log").write_text("")
    preregistered = {
        "schema": "c3_p4b_preregistered_v1",
        "modes": list(EXPERIMENTAL_MODES),
        "forced_knobs": FORCED_KNOBS,
        "mapping_rule": "complete semantic ConfigEntity exact match after forcing knobs; then deduplicate",
        "baseline_rule": "same mapped tile entity under residency mode original",
        "incumbent_protection": "one untouched original-template incumbent per workload",
        "performance_rule": "no FSim timing and no static-DMA-to-FPS claim",
        "target_dma": {
            "input_stationary": "input_load_bytes",
            "weight_stationary": "weight_load_bytes",
            "paper_inspired_hybrid": "input_plus_weight_load_bytes",
        },
    }
    (output_dir / "preregistered.json").write_text(json.dumps(preregistered, indent=2) + "\n")
    manifest = {
        "schema": "c3_p4b_local_manifest_v1",
        "date": "2026-09-10",
        "mode": "local_lower_and_static_dma_only",
        "environment": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tvm_version": tvm.__version__,
            "vta_target": env.TARGET,
            "board_contacted": False,
            "fsim_timing_used": False,
        },
        "inputs": {
            "selected_tasks": {"path": args.selected_tasks, "sha256": sha256_file(args.selected_tasks)},
            "hardware_manifest": {
                "path": args.hardware_manifest,
                "sha256": sha256_file(args.hardware_manifest),
            },
        },
        "sources": {
            "generator": str(Path(__file__)),
            "generator_sha256": sha256_file(Path(__file__)),
            "original_schedule": str(original_source),
            "residency_template": str(residency_source),
            "original_schedule_version": original_version,
            "residency_schedule_version": residency_version,
        },
        "results": mapping_diagnostics,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    status_lines = [
        "# C3-P4b local residency pool run 01",
        "",
        "- Status: `completed`",
        "- Scope: local TIR lowering and logical DMA extraction only",
        "- Board/RPC/SSH: not used",
        "- FSim timing: not used",
        "- Historical sources: {}".format(mapping_diagnostics["historical_source_candidates"]),
        "- Unique forced tile entities: {}".format(mapping_diagnostics["mapped_unique_tile_entities"]),
        "- Deduplicated sources: {}".format(mapping_diagnostics["deduplicated_historical_sources"]),
        "- Protected original incumbents: {}".format(mapping_diagnostics["protected_original_incumbents"]),
        "- Experimental residency candidates: {}".format(mapping_diagnostics["experimental_candidates"]),
        "",
        "Legality and DMA-reduction distributions are in `summary.json`. Static DMA is a "
        "compiler feature, not a latency/FPS result.",
    ]
    for mode in EXPERIMENTAL_MODES:
        mode_summary = summary["modes"][mode]
        status_lines.append(
            "- {}: lower {}/{}, target-DMA positive {}/{}".format(
                mode,
                mode_summary["lower_ok"],
                mode_summary["attempted"],
                mode_summary["target_dma_reduction_fraction"]["positive"],
                mode_summary["target_dma_reduction_fraction"]["count"],
            )
        )
    (output_dir / "STATUS.md").write_text("\n".join(status_lines) + "\n")
    handoff_lines = [
        "# HANDOFF",
        "",
        "`candidate_pool.json` records the exact 80-to-deduplicated mapping and preserves one "
        "original incumbent per workload. `results.jsonl` contains the protected originals, "
        "same-tile original controls, and every input/weight/hybrid static result.",
        "",
        "Use `summary.json` for lower legality and target-DMA distributions. Do not rank these "
        "records by FSim time: no simulator timing was collected. Candidates that lower are "
        "eligible only for later correctness and board measurement; they are not performance wins.",
    ]
    (output_dir / "HANDOFF.md").write_text("\n".join(handoff_lines) + "\n")

    artifacts = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_hashes.json":
            artifacts[path.name] = sha256_file(path)
    test_source = Path(__file__).with_name("test_generate_vta_residency_candidates.py")
    source_artifacts = {str(Path(__file__)): sha256_file(Path(__file__))}
    if test_source.is_file():
        source_artifacts[str(test_source)] = sha256_file(test_source)
    (output_dir / "artifact_hashes.json").write_text(
        json.dumps({"output_sha256": artifacts, "source_sha256": source_artifacts}, indent=2)
        + "\n"
    )
    print((output_dir / "stdout.log").read_text(), end="")


if __name__ == "__main__":
    main()
