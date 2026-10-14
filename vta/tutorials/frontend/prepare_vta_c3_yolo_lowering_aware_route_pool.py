#!/usr/bin/env python3
"""Freeze lowering-aware candidate pools for latency-unseen YOLO route tasks.

The older resolution preparation used an SRAM-capacity predicate only.  That
predicate is necessary but not sufficient: VTA's DMA injector must also be
able to express every tiled copy as a legal 2-D transfer.  This local-only
stage therefore treats successful real lowering as a hard compiler legality
gate, before FSim, FPGA contact, or latency observation.

For every selected 1x1 route task, ConfigEntities are visited in a frozen hash
order.  A family is admitted only when the same entity lowers in all declared
modes.  Three diverse admitted families are then selected, yielding at most
15 candidates per task.  Every attempted family and failure is retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from tvm import autotvm
import tvm
import vta

from c3_candidate_identity import (
    canonical_json_bytes,
    candidate_identity_record,
    normalize_config_entity,
)
from c3_reuse_semantics import capture_lower
from prepare_vta_c3_resnet18_literature_candidates import (
    KNOBS,
    analytic_applicability,
    hardware_fingerprint,
    knobs,
)
from prepare_vta_p7r115_yolo_confirmation import hardware_predicate
import run_vta_c3_resnet18_local_qualification as legacy


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_PREPARE = GROUPED / "20260924_c3_yolo_unseen_resolution_prepare_run03"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo576_lowering_aware_route_pools_run01"
TEMPLATE = "conv2d_packed_residency.vta"
SCHEMA = "c3_yolo_lowering_aware_route_pool_v1"
SELECTION_SALT = "c3-yolo-lowering-aware-route-pool-v1"
MODES = {
    "original": 0,
    "input_stationary": 1,
    "weight_resident_barrier": 4,
    "input_weight_resident_barrier": 5,
    "input_weight_resident_transition_safe": 11,
}
MODE_ORDER = tuple(MODES)
SCHEDULE_SOURCES = (
    REPO / "vta/python/vta/top/vta_conv2d.py",
    REPO / "vta/python/vta/top/residency_dispatch.py",
    REPO / "vta/python/vta/transform.py",
)


def digest(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def geometry(workload):
    data, weight = workload[1][1], workload[2][1]
    return {
        "ci": int(data[1] * data[5]),
        "co": int(weight[0] * weight[4]),
        "height": int(data[2]),
        "width": int(data[3]),
        "kernel": int(weight[2]),
        "stride": int(workload[3][0]),
    }


def select_route_tasks(tasks, env):
    """Select every full-weight-fit 1x1/CO256 route task without latency data."""

    selected = []
    for task in tasks:
        spec = geometry(task["workload"])
        full_weight_bytes = spec["ci"] * spec["co"] * spec["kernel"] * spec["kernel"]
        if spec["kernel"] == 1 and spec["co"] == 256 and full_weight_bytes <= env.WGT_BUFF_SIZE:
            selected.append((spec["ci"], spec["height"], spec["width"], task["task_id"], task, spec))
    return [(task, spec) for _, _, _, _, task, spec in sorted(selected)]


def make_candidate(workload_id, workload, spec, entity, index, mode, schedule_version, env):
    fingerprint = hardware_fingerprint(env)
    identity = candidate_identity_record(
        fingerprint,
        TEMPLATE,
        schedule_version,
        workload,
        mode,
        entity,
        config_index=index,
    )
    visible = knobs(entity)
    family_hash = digest(entity)[:16]
    return {
        "candidate_id": identity["candidate_id"],
        "workload_id": workload_id,
        "family_id": "{}-L{}".format(workload_id, family_hash),
        "model_hash": None,
        "hardware_fingerprint": fingerprint,
        "complete_config_entity": entity,
        "debug": {"config_index": int(index)},
        "residence_mode": mode,
        "implementation_mode": MODES[mode],
        "visible_features": visible,
        "workload_features": dict(spec),
        "static_features": {"status": "passed_lowering_generation_gate"},
        "applicability": analytic_applicability(spec, visible, env),
        "lowered_tir_sha256": None,
        "implementation_identity_status": "pending_full_local_qualification",
        "candidate_identity_fields": {
            "tile": visible,
            "residence_mode": mode,
            "implementation_mode": MODES[mode],
            "weight_load_policy": (
                "resident_with_entry_drain_scoped_force_serial_completion"
                if mode == "input_weight_resident_transition_safe"
                else "barrier_full_or_group_residency"
                if "resident" in mode
                else "template_default"
            ),
            "schedule_version": schedule_version,
        },
    }


def probe_lowering(candidate, workload):
    bound = json.loads(json.dumps(candidate))
    bound["identity"] = {
        "workload": workload,
        "complete_config_entity": candidate["complete_config_entity"],
    }
    started = time.perf_counter()
    try:
        schedule, tensors = legacy.instantiate(bound, vta.get_env())
        module, _, _ = capture_lower(schedule, tensors)
        return {
            "status": "passed",
            "lowered_tir_sha256": hashlib.sha256(tvm.ir.save_json(module).encode()).hexdigest(),
            "wall_ms": (time.perf_counter() - started) * 1000.0,
            "failure": None,
        }
    except Exception as error:  # TVM exposes several lowering exception classes.
        message = str(error) or type(error).__name__
        return {
            "status": "failed",
            "lowered_tir_sha256": None,
            "wall_ms": (time.perf_counter() - started) * 1000.0,
            "failure": {
                "phase": "real_vta_lowering",
                "exception_type": type(error).__name__,
                "message": message[-4000:],
                "dma_2d_pattern_failure": "cannot detect 2d pattern" in message,
            },
        }


def rank(workload_id, entity):
    return hashlib.sha256(
        "{}|{}|{}".format(SELECTION_SALT, workload_id, digest(entity)).encode()
    ).hexdigest()


def normalized_coordinates(entities):
    values = {name: sorted({knobs(entity)[name] for entity in entities}) for name in KNOBS}
    result = {}
    for entity in entities:
        row = knobs(entity)
        result[digest(entity)] = tuple(
            values[name].index(row[name]) / max(1, len(values[name]) - 1) for name in KNOBS
        )
    return result


def diverse_families(workload_id, rows, count):
    if len(rows) < count:
        return list(rows)
    entities = [row["entity"] for row in rows]
    coordinates = normalized_coordinates(entities)
    remaining = list(rows)
    first = min(remaining, key=lambda row: (row["selection_rank_sha256"], digest(row["entity"])))
    selected = [first]
    remaining.remove(first)
    while len(selected) < count:
        def score(row):
            coordinate = coordinates[digest(row["entity"])]
            distances = []
            for other in selected:
                other_coordinate = coordinates[digest(other["entity"])]
                distances.append(sum(abs(a - b) for a, b in zip(coordinate, other_coordinate)))
            # Maximize distance while using the frozen rank as a deterministic tie breaker.
            return min(distances), sum(distances), row["selection_rank_sha256"]

        choice = max(remaining, key=score)
        selected.append(choice)
        remaining.remove(choice)
    return selected


def verify_prepare(directory):
    directory = Path(directory).resolve()
    ledger = read_json(directory / "artifact_hashes.json")
    bindings = {
        "base_cfg": "yolov3-tiny.base.cfg",
        "attempts": "attempts.jsonl",
        "contract": "contract.json",
        "summary": "summary.json",
    }
    for key, name in bindings.items():
        expected = ledger.get(key)
        if expected is None or sha256_file(directory / name) != expected:
            raise ValueError("prepare artifact hash mismatch: {}".format(name))
    contract = read_json(directory / "contract.json")
    if contract.get("board_contacted") is not False or contract.get("latency_values_read") is not False:
        raise ValueError("prepare input is not latency isolated")
    resolution = contract.get("selected_resolution")
    exposure = (contract.get("latency_exposure_registry") or {}).get(str(resolution))
    if exposure is not None:
        raise ValueError("selected resolution is latency exposed")
    result_path = directory / "resolution_{}".format(resolution) / "result.json"
    return contract, read_json(result_path), result_path


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    prepare_contract, prepared, result_path = verify_prepare(args.prepare_dir)
    env = vta.get_env()
    selected_tasks = select_route_tasks(prepared["tasks"], env)
    if len(selected_tasks) < 3:
        raise RuntimeError("fewer than three full-weight-fit 1x1/CO256 route tasks")

    source_hashes = {str(path.relative_to(REPO)): sha256_file(path) for path in SCHEDULE_SOURCES}
    schedule_version = digest({
        "mechanism": "lowering-aware-five-mode-route-pool-v1",
        "source_hashes": source_hashes,
    })
    resolution = int(prepare_contract["selected_resolution"])
    model_hash = digest({
        "model": "yolov3-tiny",
        "resolution": resolution,
        "weights_sha256": prepare_contract["weights"]["sha256"],
    })
    output.mkdir(parents=True)
    scan_path = output / "lowering_scan.jsonl"
    scan_path.write_text("", encoding="utf-8")
    pool_files = []
    total_candidates = 0
    task_summaries = {}

    for task_row, spec in selected_tasks:
        workload = task_row["workload"]
        workload_id = "Y{}-{}".format(resolution, task_row["task_id"][:16])
        tasks = {
            mode: autotvm.task.create(
                TEMPLATE,
                args=tuple(workload[1:]) + (number,),
                target=env.target,
                target_host=env.target_host,
            )
            for mode, number in MODES.items()
        }
        domains = {
            mode: [normalize_config_entity(task.config_space.get(index).to_json_dict())
                   for index in range(len(task.config_space))]
            for mode, task in tasks.items()
        }
        reference = domains["original"]
        if any(domain != reference for domain in domains.values()):
            raise RuntimeError("{} mode ConfigSpaces differ".format(workload_id))
        eligible = []
        for index, entity in enumerate(reference):
            certificate = hardware_predicate(workload, entity, env)
            if certificate["passed"]:
                eligible.append((rank(workload_id, entity), digest(entity), index, entity, certificate))
        eligible.sort()

        admitted = []
        attempted = 0
        failure_counts = Counter()
        for selection_rank, entity_hash, index, entity, certificate in eligible[: args.scan_limit]:
            attempted += 1
            family_candidates = []
            mode_results = {}
            for mode in MODE_ORDER:
                candidate = make_candidate(
                    workload_id, workload, spec, entity, index, mode, schedule_version, env
                )
                candidate["model_hash"] = model_hash
                result = probe_lowering(candidate, workload)
                mode_results[mode] = result
                append_jsonl(scan_path, {
                    "schema": SCHEMA,
                    "workload_id": workload_id,
                    "gross_family_position": attempted,
                    "selection_rank_sha256": selection_rank,
                    "semantic_config_sha256": entity_hash,
                    "config_index_debug_only": index,
                    "mode": mode,
                    "candidate_id": candidate["candidate_id"],
                    "visible_features": candidate["visible_features"],
                    **result,
                })
                if result["status"] == "passed":
                    candidate["generation_lowered_tir_sha256"] = result["lowered_tir_sha256"]
                    family_candidates.append(candidate)
                else:
                    failure_counts[
                        "dma_2d_pattern" if result["failure"]["dma_2d_pattern_failure"] else "other"
                    ] += 1
            if len(family_candidates) == len(MODE_ORDER):
                admitted.append({
                    "entity": entity,
                    "selection_rank_sha256": selection_rank,
                    "semantic_config_sha256": entity_hash,
                    "config_index_debug_only": index,
                    "hardware_predicate": certificate,
                    "candidates": family_candidates,
                })
            if len(admitted) >= args.admitted_family_target:
                break

        chosen = diverse_families(workload_id, admitted, args.selected_families)
        candidates = []
        for ordinal, family in enumerate(chosen):
            family_id = "{}-F{}".format(workload_id, ordinal)
            for candidate in family["candidates"]:
                candidate["family_id"] = family_id
                candidate["generation_family_ordinal"] = ordinal
                candidates.append(candidate)
        incumbent_id = None
        if chosen:
            incumbent_id = next(
                row["candidate_id"]
                for row in chosen[0]["candidates"]
                if row["residence_mode"] == "original"
            )
        pool = {
            "schema": SCHEMA,
            "status": (
                "frozen_after_lowering_gate_before_fsim_fpga_or_latency"
                if len(chosen) == args.selected_families
                else "insufficient_fully_lowerable_families"
            ),
            "workload_id": workload_id,
            "execution_task_id": task_row["task_id"],
            "model": "yolov3_tiny_darknet",
            "input_resolution": resolution,
            "call_count": task_row["occurrences"],
            "workload": workload,
            "geometry": spec,
            "hardware_fingerprint": hardware_fingerprint(env),
            "candidate_modes": list(MODE_ORDER),
            "candidates": candidates,
            "incumbent": {
                "kind": "first_frozen_hash_rank_original_that_passes_all-mode_lowering_gate",
                "candidate_id": incumbent_id,
                "current_board_latency": None,
                "global_optimum_claimed": False,
            },
            "selection": {
                "task_rule": "all full-weight-fit K1 CO256 tasks, ordered by CI/H/W/task-id",
                "family_rule": (
                    "hardware-capacity predicate, then all-five-mode real VTA lowering; "
                    "deterministic max-min selection from admitted prefix"
                ),
                "eligible_config_count": len(eligible),
                "gross_families_attempted": attempted,
                "fully_lowerable_families": len(admitted),
                "selected_family_count": len(chosen),
                "scan_limit": args.scan_limit,
                "admitted_family_target": args.admitted_family_target,
                "performance_labels_used": False,
                "fsim_used": False,
            },
            "source_hashes": source_hashes,
            "board_contacted": False,
            "performance_labels_collected": False,
        }
        filename = workload_id + ".json"
        write_json(output / filename, pool)
        pool_files.append(filename)
        total_candidates += len(candidates)
        task_summaries[workload_id] = {
            "geometry": spec,
            "eligible_config_count": len(eligible),
            "gross_families_attempted": attempted,
            "fully_lowerable_families": len(admitted),
            "selected_family_count": len(chosen),
            "candidate_count": len(candidates),
            "failure_counts": dict(failure_counts),
            "status": pool["status"],
        }

    complete_tasks = sum(
        row["selected_family_count"] == args.selected_families for row in task_summaries.values()
    )
    contract = {
        "schema": SCHEMA + "_contract",
        "status": (
            "LOWERING_AWARE_POOLS_FROZEN" if complete_tasks == len(task_summaries)
            else "LOWERING_AWARE_POOL_INCOMPLETE"
        ),
        "input_resolution": resolution,
        "pool_files": pool_files,
        "execution_task_count": len(pool_files),
        "complete_execution_task_count": complete_tasks,
        "candidate_count": total_candidates,
        "candidate_modes": list(MODE_ORDER),
        "task_selection_rule": "all full-weight-fit K1 CO256 tasks, ordered by CI/H/W/task-id",
        "prepare_contract_sha256": sha256_file(Path(args.prepare_dir) / "contract.json"),
        "prepare_result_sha256": sha256_file(result_path),
        "schedule_version": schedule_version,
        "source_hashes": source_hashes,
        "task_summaries": task_summaries,
        "superseded_development_attempt": {
            "artifact": "20260924_c3_yolo576_route_local_qualification_run01",
            "reason": (
                "capacity-only family selection produced DMA-2D-lowering failures; its declared "
                "H=W=18 selection text also mismatched the included H=W=36/CI256 task"
            ),
            "performance_evidence_eligible": False,
        },
        "board_contacted": False,
        "performance_labels_read": False,
        "fsim_used_for_selection": False,
        "claim_boundary": (
            "real-lowering legality and frozen identities only; FSim, FPGA correctness, command "
            "equivalence, latency, oracle, and speed claims remain pending"
        ),
    }
    write_json(output / "contract.json", contract)
    write_json(output / "summary.json", contract)
    write_json(output / "artifact_hashes.json", {
        "artifacts": {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "artifact_hashes.json"
        },
        "source_sha256": sha256_file(Path(__file__).resolve()),
    })
    print(json.dumps(contract, indent=2, sort_keys=True), flush=True)
    if contract["status"] != "LOWERING_AWARE_POOLS_FROZEN":
        raise SystemExit(2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-dir", type=Path, default=DEFAULT_PREPARE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scan-limit", type=int, default=1024)
    parser.add_argument("--admitted-family-target", type=int, default=12)
    parser.add_argument("--selected-families", type=int, default=3)
    args = parser.parse_args()
    if args.selected_families * len(MODE_ORDER) > 15:
        parser.error("selected pool exceeds the frozen 15-candidate bound")
    if args.admitted_family_target < args.selected_families:
        parser.error("admitted-family-target must cover selected-families")
    return args


if __name__ == "__main__":
    run(parse_args())
