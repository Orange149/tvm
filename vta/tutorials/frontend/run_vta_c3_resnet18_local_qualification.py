#!/usr/bin/env python3
"""Locally qualify the frozen P7R470 ResNet18 four-mode candidate pools.

This runner is deliberately label blind.  It verifies the immutable P7R470
inputs, performs real VTA lowering, extracts audit-friendly ML2Tuner-style
hidden compiler features and logical DMA features, then correctness-checks each
lowering-success identity with three FSim seeds.  It never contacts the board
and never reads a performance or oracle label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tvm
from tvm import autotvm, tir
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from c3_candidate_identity import canonical_json_bytes, make_failure, normalize_config_entity
from collect_vta_fsim_command_signatures import command_signature, parse_queue_records
from extract_static_vta_dma import extract_module_dma_compact
from generate_vta_residency_candidates import classify_lower_failure
from qualify_vta_residency_fsim import array_sha256, reference_data_from_workload


SCHEMA = "c3_resnet18_local_qualification_v1"
SEEDS = (0, 20250901, 20260910)
MODES = {
    "original": 0,
    "input_stationary": 1,
    "weight_resident_barrier": 4,
    "input_weight_resident_barrier": 5,
}
WEIGHT_MODES = frozenset(("weight_resident_barrier", "input_weight_resident_barrier"))
DEFAULT_INPUT = Path(__file__).resolve().parent / (
    "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout/"
    "20260914_p7r470_resnet18_literature_candidates_run01"
)


def canonical_sha256(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def verify_p7r470(directory):
    directory = Path(directory).resolve()
    ledger = read_json(directory / "artifact_hashes.json")["artifacts"]
    checked = {}
    for name, expected in sorted(ledger.items()):
        path = directory / name
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError("P7R470 artifact hash mismatch: {}".format(name))
        checked[name] = observed
    pools = []
    for path in sorted(directory.glob("r18-h*_four_mode_96.json")):
        payload = read_json(path)
        if payload.get("board_contacted") is not False:
            raise ValueError("input pool is not label-isolated: {}".format(path))
        def reject_live_labels(value, location="$"):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key.lower() in ("latency_ms", "performance_label", "oracle_latency") and item is not None:
                        raise ValueError("future-label value at {}.{} in {}".format(location, key, path))
                    reject_live_labels(item, "{}.{}".format(location, key))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    reject_live_labels(item, "{}[{}]".format(location, index))

        reject_live_labels(payload)
        pools.append((path, payload))
    identity_count = sum(len(payload["candidates"]) for _, payload in pools)
    if not pools or identity_count == 0 or identity_count % 4:
        raise ValueError("expected one or more non-empty four-mode candidate pools")
    for path, payload in pools:
        families = {}
        for row in payload["candidates"]:
            families.setdefault(row["family_id"], set()).add(row["residence_mode"])
        if any(modes != set(MODES) for modes in families.values()):
            raise ValueError("pool does not contain exact four-mode families: {}".format(path))
    return checked, pools


def candidate_with_context(candidate, pool):
    row = json.loads(json.dumps(candidate))
    row["identity"] = {
        "workload": pool["workload"],
        "complete_config_entity": candidate["complete_config_entity"],
    }
    return row


def create_task(candidate, env):
    workload = candidate["identity"]["workload"]
    return autotvm.task.create(
        "conv2d_packed_residency.vta",
        args=tuple(workload[1:]) + (int(candidate["implementation_mode"]),),
        target=env.target,
        target_host=env.target_host,
    )


def instantiate(candidate, env):
    task = create_task(candidate, env)
    index = int(candidate["debug"]["config_index"])
    config = task.config_space.get(index)
    actual = normalize_config_entity(config.to_json_dict())
    expected = normalize_config_entity(candidate["complete_config_entity"])
    if canonical_json_bytes(actual) != canonical_json_bytes(expected):
        raise RuntimeError("ConfigEntity differs from frozen P7R470 identity")
    with task.target:
        schedule, tensors = task.instantiate(config)
    return schedule, tensors


def _constant_product(extents):
    result = 1
    for extent in extents:
        if not isinstance(extent, tir.IntImm):
            return None
        result *= int(extent)
    return result


def hidden_compiler_features(module, candidate):
    """Small, explicit counterpart of ML2Tuner's post-lowering feature layer."""

    counts = Counter()
    loop_extent_sum = 0
    loop_extent_log2_product = 0.0
    allocation_bytes = 0
    allocation_by_scope = Counter()
    externs = Counter()

    def visit(node):
        nonlocal loop_extent_sum, loop_extent_log2_product, allocation_bytes
        if isinstance(node, tir.For):
            counts["loop_count"] += 1
            if isinstance(node.extent, tir.IntImm):
                extent = int(node.extent)
                loop_extent_sum += extent
                loop_extent_log2_product += math.log2(max(1, extent))
        elif isinstance(node, tir.IfThenElse):
            counts["branch_count"] += 1
        elif isinstance(node, tir.Allocate):
            counts["allocation_count"] += 1
            elements = _constant_product(node.extents)
            if elements is not None:
                byte_count = elements * tvm.DataType(str(node.dtype)).bits // 8
                allocation_bytes += byte_count
                annotation = node.buffer_var.type_annotation
                scope = str(getattr(annotation, "storage_scope", "global") or "global")
                allocation_by_scope[scope] += byte_count
        elif (
            isinstance(node, tir.Call)
            and isinstance(node.op, tvm.ir.Op)
            and node.op.name == "tir.call_extern"
            and node.args
            and isinstance(node.args[0], tir.StringImm)
        ):
            externs[node.args[0].value] += 1

    for function in module.functions.values():
        tir.stmt_functor.post_order_visit(function.body, visit)

    visible = candidate["visible_features"]
    workload = candidate["workload_features"]
    out_h = (workload["height"] + 2 * (workload["kernel"] // 2) - workload["kernel"]) // workload["stride"] + 1
    out_w = (workload["width"] + 2 * (workload["kernel"] // 2) - workload["kernel"]) // workload["stride"] + 1
    dimensions = (
        (out_h, visible["tile_h"]),
        (out_w, visible["tile_w"]),
        (workload["ci"] // 16, visible["tile_ci"]),
        (workload["co"] // 16, visible["tile_co"]),
    )
    partial_tile_count = sum(extent % tile != 0 for extent, tile in dimensions)
    return {
        "loop_count": int(counts["loop_count"]),
        "loop_extent_sum": int(loop_extent_sum),
        "loop_extent_log2_product": float(loop_extent_log2_product),
        "loop_extent_product_log2": float(loop_extent_log2_product),
        "branch_count": int(counts["branch_count"]),
        "partial_tile_count": int(partial_tile_count),
        "allocation_count": int(counts["allocation_count"]),
        "allocation_bytes": int(allocation_bytes),
        "allocation_bytes_by_scope": dict(sorted(allocation_by_scope.items())),
        "tensorize_uop_push_sites": int(externs["VTAUopPush"]),
        "tensorize_uop_sites": int(externs["VTAUopPush"] + externs["VTAUopLoopBegin"]),
        "tensorize_count": int(externs["VTAUopPush"] + externs["VTAUopLoopBegin"]),
        "extern_call_site_histogram": dict(sorted(externs.items())),
        "scope": "deterministic features extracted after real lowering; no performance labels",
    }


def dependency_audit(module):
    pushes, pops = Counter(), Counter()

    def visit(node):
        if not isinstance(node, tir.Call) or not isinstance(node.op, tvm.ir.Op):
            return
        if node.op.name == "tir.vta.coproc_dep_push":
            pushes[(int(node.args[0]), int(node.args[1]))] += 1
        elif node.op.name == "tir.vta.coproc_dep_pop":
            pops[(int(node.args[0]), int(node.args[1]))] += 1

    tir.stmt_functor.post_order_visit(module["main"].body, visit)
    forbidden = sum(count for pair, count in pushes.items() if set(pair) == {1, 3})
    return {
        "balanced": pushes == pops,
        "forbidden_direct_1_3_count": int(forbidden),
        "push": {"{}->{}".format(*key): value for key, value in sorted(pushes.items())},
        "pop": {"{}->{}".format(*key): value for key, value in sorted(pops.items())},
    }


def sync_multiplicity(module):
    loops, values = [], []

    def preorder(node):
        if isinstance(node, tir.For):
            loops.append(int(node.extent))
        elif isinstance(node, tir.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tir.vta.coproc_sync":
            values.append(math.prod(loops))

    def postorder(node):
        if isinstance(node, tir.For):
            loops.pop()

    tir.stmt_functor.ir_transform(module["main"].body, preorder, postorder, None)
    return values


def dma_features(module, env):
    extracted = extract_module_dma_compact(module, env)
    totals = extracted["totals"]
    requests = extracted["request_descriptors"]
    return {
        "totals": totals,
        "max_request_bytes_by_memory": extracted["max_request_bytes_by_memory"],
        "request_site_count": len(requests),
        "padded_calls": int(sum(row["multiplicity"] for row in requests if row["padded"])),
        "logical_bytes": {
            name: int(totals.get("load_buffer_2d_{}_bytes".format(name), 0))
            for name in ("inp", "wgt", "acc", "uop")
        } | {"out": int(totals.get("store_buffer_2d_out_bytes", 0))},
        "logical_calls": {
            "load": int(totals.get("load_buffer_2d_calls", 0)),
            "store": int(totals.get("store_buffer_2d_calls", 0)),
        },
        "scope": "logical VTA LOAD/STORE after lowering; not physical AXI traffic",
    }


def fallback_candidate_id(candidate, family_rows):
    matches = [row for row in family_rows if row["residence_mode"] == "original"]
    if len(matches) != 1:
        raise ValueError("family lacks one exact original fallback")
    if normalize_config_entity(matches[0]["complete_config_entity"]) != normalize_config_entity(candidate["complete_config_entity"]):
        raise ValueError("fallback does not preserve the same tile")
    return matches[0]["candidate_id"]


def static_result(candidate, family_rows):
    start = time.monotonic()
    base = {
        "schema": SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "proposal_candidate_id": candidate["candidate_id"],
        "workload_id": candidate["workload_id"],
        "family_id": candidate["family_id"],
        "residence_mode": candidate["residence_mode"],
        "implementation_mode": candidate["implementation_mode"],
        "debug": candidate["debug"],
        "visible_features": candidate["visible_features"],
        "performance_label": None,
        "board_contacted": False,
        "phase": "static",
        "status": "failed",
        "failure": None,
    }
    if candidate["residence_mode"] in WEIGHT_MODES and not candidate["applicability"]["full_weight_fits_sram"]:
        base.update(
            status="not_applicable",
            applicability=candidate["applicability"],
            fallback={
                "policy": "same-tile original",
                "candidate_id": fallback_candidate_id(candidate, family_rows),
                "reason": "full layer weights exceed VTA weight SRAM",
            },
        )
        base["wall_seconds"] = time.monotonic() - start
        return base
    phase = "instantiate"
    try:
        env = vta.get_env()
        schedule, tensors = instantiate(candidate, env)
        phase = "lower"
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            module = tvm.lower(schedule, tensors, name="main")
        tir_hash = hashlib.sha256(tvm.ir.save_json(module).encode("utf-8")).hexdigest()
        phase = "static_features"
        dependencies = dependency_audit(module)
        if not dependencies["balanced"] or dependencies["forbidden_direct_1_3_count"]:
            raise RuntimeError("dependency audit rejected lowered program")
        sync = sync_multiplicity(module)
        dma = dma_features(module, env)
        implementation_payload = {
            "proposal_candidate_id": candidate["candidate_id"],
            "model_hash": candidate["model_hash"],
            "workload_id": candidate["workload_id"],
            "lowered_tir_sha256": tir_hash,
            "hardware_fingerprint": candidate["hardware_fingerprint"],
        }
        base.update(
            status="ok",
            lowered_tir_sha256=tir_hash,
            implementation_candidate_id=canonical_sha256(implementation_payload),
            implementation_identity=implementation_payload,
            hidden_compiler_features=hidden_compiler_features(module, candidate),
            dma_features=dma,
            dependency_audit=dependencies,
            sync={
                "static_execution_multiplicity": sync,
                "explicit_residency_drains": int(sum(sync[:-1])) if len(sync) > 1 else 0,
                "function_final_drain": int(sync[-1]) if sync else 0,
            },
        )
    except Exception as error:  # TVM deliberately exposes heterogeneous lowering exceptions.
        message = str(error) or type(error).__name__
        failure = make_failure("lower", message[:4000], phase=phase, retryable=False)
        failure["exception_type"] = type(error).__name__
        failure["subcategory"] = classify_lower_failure(message, phase)
        base["failure"] = failure
    base["wall_seconds"] = time.monotonic() - start
    return base


def base_fsim(candidate, static):
    return {
        "schema": SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "implementation_candidate_id": static.get("implementation_candidate_id"),
        "workload_id": candidate["workload_id"],
        "family_id": candidate["family_id"],
        "residence_mode": candidate["residence_mode"],
        "implementation_mode": candidate["implementation_mode"],
        "phase": "fsim",
        "status": "failed",
        "failure": None,
        "seeds": [],
        "performance_label": None,
        "board_contacted": False,
    }


def fsim_worker(candidate, static):
    start = time.monotonic()
    result = base_fsim(candidate, static)
    phase = "preflight"
    try:
        from vta.testing import simulator  # pylint: disable=import-outside-toplevel

        env = vta.get_env()
        if env.TARGET != "sim" or not simulator.enabled():
            raise RuntimeError("worker requires VTA TARGET=sim and enabled FSim runtime")
        schedule, tensors = instantiate(candidate, env)
        phase = "relower"
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            lowered = tvm.lower(schedule, tensors, name="main")
        tir_hash = hashlib.sha256(tvm.ir.save_json(lowered).encode("utf-8")).hexdigest()
        result["relowered_tir_sha256"] = tir_hash
        result["tir_hash_matches_static"] = tir_hash == static["lowered_tir_sha256"]
        if not result["tir_hash_matches_static"]:
            raise RuntimeError("FSim re-lowering differs from frozen static TIR")
        phase = "build"
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            module = vta.build(
                schedule,
                tensors,
                target=tvm.target.Target(env.target, host=env.target_host),
                name="main",
            )
        function = module["main"]
        device = tvm.device("ext_dev", 0)
        workload = candidate["identity"]["workload"]
        for seed in SEEDS:
            phase = "execute_seed_{}".format(seed)
            data, weight, expected = reference_data_from_workload(workload, seed)
            output = tvm.nd.empty(expected.shape, "int8", device=device)
            print("[C3_R18_SEED] {}".format(seed), file=sys.stderr, flush=True)
            function(tvm.nd.array(data, device), tvm.nd.array(weight, device), output)
            actual = output.numpy()
            mismatches = int(np.count_nonzero(actual != expected))
            result["seeds"].append(
                {
                    "seed": seed,
                    "correct": mismatches == 0,
                    "mismatch_count": mismatches,
                    "expected_sha256": array_sha256(expected),
                    "actual_sha256": array_sha256(actual),
                }
            )
        result["status"] = "passed" if all(row["correct"] for row in result["seeds"]) else "failed"
        if result["status"] != "passed":
            result["failure"] = make_failure("wrong_answer", "one or more FSim seeds failed", phase="execute")
    except Exception as error:
        category = "wrong_answer" if phase.startswith("execute_seed_") else ("compile" if phase == "build" else "environment")
        failure = make_failure(category, (str(error) or type(error).__name__)[:4000], phase=phase)
        failure["exception_type"] = type(error).__name__
        result["failure"] = failure
    result["wall_seconds"] = time.monotonic() - start
    return result


def parse_seed_queues(stderr):
    sections = {seed: [] for seed in SEEDS}
    current = None
    for line in stderr.splitlines():
        if "[C3_R18_SEED] " in line:
            try:
                current = int(line.split("[C3_R18_SEED] ", 1)[1].strip())
            except ValueError:
                current = None
        elif current in sections and "[VTA_QUEUE] " in line:
            sections[current].append(line)
    return {seed: parse_queue_records("\n".join(lines)) for seed, lines in sections.items()}


def normalize_command_signature(value):
    normalized = json.loads(json.dumps(value))
    for ordinal, row in enumerate(normalized.get("submit_sequence", []), 1):
        row.pop("submit", None)
        row["submit_ordinal_within_inference"] = ordinal
    return normalized


def attach_command_features(result, stderr, static):
    if result["status"] != "passed":
        result["command_features"] = {"status": "unavailable_due_to_fsim_failure", "values": None}
        return result
    parsed = parse_seed_queues(stderr)
    signatures = {}
    expected_drains = static["sync"]["explicit_residency_drains"]
    for seed in SEEDS:
        records, malformed = parsed[seed]
        if malformed or not records:
            result["status"] = "failed"
            result["failure"] = make_failure(
                "environment", "missing or malformed VTA_QUEUE diagnostics for seed {}".format(seed), phase="command_signature"
            )
            result["command_features"] = {"status": "unavailable_due_to_diagnostics", "values": None}
            return result
        signatures[seed] = normalize_command_signature(
            command_signature(records, result["implementation_mode"], expected_drains)["structural"]
        )
    hashes = {str(seed): canonical_sha256(signatures[seed]) for seed in SEEDS}
    if len(set(hashes.values())) != 1:
        result["status"] = "failed"
        result["failure"] = make_failure("environment", "command structure differs across seeds", phase="command_signature")
        result["command_features"] = {"status": "inconsistent_across_seeds", "values": None, "per_seed_sha256": hashes}
        return result
    result["command_features"] = {
        "status": "available_three_seed_consistent",
        "values": signatures[SEEDS[0]],
        "per_seed_sha256": hashes,
        "raw_timing_fields": "discarded",
    }
    return result


def run_fsim_subprocess(candidate, static, args):
    with tempfile.TemporaryDirectory(prefix="c3_r18_fsim_") as directory:
        request = Path(directory) / "request.json"
        response = Path(directory) / "response.json"
        write_json(request, {"candidate": candidate, "static": static})
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-request", str(request),
            "--worker-output", str(response),
        ]
        environment = os.environ.copy()
        environment["VTA_HW_PATH"] = str(Path(args.fsim_hw_path).resolve())
        environment["VTA_QUEUE_DIAGNOSTICS"] = "1"
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=args.fsim_timeout, env=environment, check=False)
        except subprocess.TimeoutExpired as error:
            result = base_fsim(candidate, static)
            result["failure"] = make_failure("timeout", "FSim exceeded {} seconds".format(args.fsim_timeout), phase="worker", retryable=True)
            result["wall_seconds"] = float(args.fsim_timeout)
            return result, error.stderr or ""
        if completed.returncode != 0 or not response.is_file():
            result = base_fsim(candidate, static)
            result["failure"] = make_failure(
                "environment",
                "worker exit={} output_present={}".format(completed.returncode, response.is_file()),
                phase="worker_process",
            )
            result["wall_seconds"] = None
        else:
            result = read_json(response)
        return attach_command_features(result, completed.stderr, static), completed.stderr


def summary_record(static_rows, fsim_rows, elapsed, input_hashes):
    geometries = {}
    for workload_id in sorted({row["workload_id"] for row in static_rows}):
        srows = [row for row in static_rows if row["workload_id"] == workload_id]
        frows = [row for row in fsim_rows if row["workload_id"] == workload_id]
        modes = {}
        for mode in MODES:
            smode = [row for row in srows if row["residence_mode"] == mode]
            fmode = [row for row in frows if row["residence_mode"] == mode]
            modes[mode] = {
                "proposed": len(smode),
                "static_ok": sum(row["status"] == "ok" for row in smode),
                "not_applicable": sum(row["status"] == "not_applicable" for row in smode),
                "static_failed": sum(row["status"] == "failed" for row in smode),
                "fsim_passed": sum(row["status"] == "passed" for row in fmode),
                "fsim_failed": sum(row["status"] != "passed" for row in fmode),
            }
        local_legal = sum(row["status"] == "passed" for row in frows)
        geometries[workload_id] = {
            "modes": modes,
            "local_fsim_legal_identities": local_legal,
            "expansion_triggered": local_legal < 12,
            "expansion_rule": "append next 24 deterministic max-min tiles iff local_fsim_legal_identities < 12",
        }
    return {
        "schema": SCHEMA,
        "status": "complete_local_qualification" if fsim_rows else "complete_static_qualification_only",
        "scope": "real lowering and three-seed local FSim; no board, latency, oracle, or TopHub label",
        "input_artifact_hashes": input_hashes,
        "seeds": list(SEEDS),
        "proposed_identities": len(static_rows),
        "static_ok": sum(row["status"] == "ok" for row in static_rows),
        "not_applicable": sum(row["status"] == "not_applicable" for row in static_rows),
        "static_failed": sum(row["status"] == "failed" for row in static_rows),
        "fsim_attempted": len(fsim_rows),
        "fsim_passed": sum(row["status"] == "passed" for row in fsim_rows),
        "fsim_failed": sum(row["status"] != "passed" for row in fsim_rows),
        "wall_seconds": elapsed,
        "geometries": geometries,
        "board_contacted": False,
        "performance_labels_collected": False,
    }


def run_worker(args):
    request = read_json(args.worker_request)
    result = fsim_worker(request["candidate"], request["static"])
    write_json(args.worker_output, result)


def run_main(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    input_hashes, pools = verify_p7r470(args.input_dir)
    if args.workload_id:
        pools = [
            (path, pool)
            for path, pool in pools
            if pool["workload_id"] == args.workload_id
        ]
        if len(pools) != 1:
            raise ValueError("input does not contain exactly one {} pool".format(args.workload_id))
    output.mkdir(parents=True)
    static_path, fsim_path, timeline_path = output / "static_results.jsonl", output / "fsim_results.jsonl", output / "timeline.jsonl"
    for path in (static_path, fsim_path, timeline_path, output / "results.jsonl"):
        path.write_text("", encoding="utf-8")
    candidates = []
    family_rows = {}
    for _, pool in pools:
        for raw in pool["candidates"]:
            candidate = candidate_with_context(raw, pool)
            candidates.append(candidate)
            family_rows.setdefault(candidate["family_id"], []).append(candidate)
    if args.limit is not None:
        candidates = candidates[:args.limit]
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_local_qualification",
        "input_dir": str(Path(args.input_dir).resolve()),
        "input_artifact_hashes": input_hashes,
        "candidate_order": [row["candidate_id"] for row in candidates],
        "seeds": list(SEEDS),
        "failure_accounting": "all proposed identities remain in gross results; not_applicable uses exact same-tile original fallback",
        "forbidden_inputs": ["board timing", "future correctness", "pool oracle", "TopHub cost"],
        "board_contacted": False,
    }
    write_json(output / "contract.json", contract)
    start = time.monotonic()
    static_rows = []
    for position, candidate in enumerate(candidates, 1):
        row = static_result(candidate, family_rows[candidate["family_id"]])
        row["gross_position"] = position
        static_rows.append(row)
        append_jsonl(static_path, row)
        append_jsonl(output / "results.jsonl", row)
        append_jsonl(timeline_path, {"event": "static_complete", "position": position, "candidate_id": candidate["candidate_id"], "status": row["status"], "elapsed_seconds": time.monotonic() - start})
        if position % 12 == 0 or position == len(candidates):
            print("static {}/{} ok={} failed={} not_applicable={}".format(position, len(candidates), sum(x["status"] == "ok" for x in static_rows), sum(x["status"] == "failed" for x in static_rows), sum(x["status"] == "not_applicable" for x in static_rows)), flush=True)
    fsim_rows = []
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    eligible = [row for row in static_rows if row["status"] == "ok"] if args.phase == "all" else []
    for position, static in enumerate(eligible, 1):
        candidate = candidate_by_id[static["candidate_id"]]
        row, stderr = run_fsim_subprocess(candidate, static, args)
        row["fsim_position"] = position
        fsim_rows.append(row)
        append_jsonl(fsim_path, row)
        append_jsonl(output / "results.jsonl", row)
        append_jsonl(timeline_path, {"event": "fsim_complete", "position": position, "candidate_id": candidate["candidate_id"], "status": row["status"], "elapsed_seconds": time.monotonic() - start})
        if stderr.strip() and row["status"] != "passed":
            append_jsonl(output / "worker_failures.jsonl", {"candidate_id": candidate["candidate_id"], "stderr_tail": stderr[-4000:]})
        if position % 4 == 0 or position == len(eligible):
            print("fsim {}/{} passed={} failed={}".format(position, len(eligible), sum(x["status"] == "passed" for x in fsim_rows), sum(x["status"] != "passed" for x in fsim_rows)), flush=True)
    summary = summary_record(static_rows, fsim_rows, time.monotonic() - start, input_hashes)
    write_json(output / "summary.json", summary)
    write_json(output / "expansion_decision.json", {
        "schema": SCHEMA,
        "decision_basis": "local FSim legal identity count only; no board performance label",
        "geometries": {key: {"local_fsim_legal_identities": value["local_fsim_legal_identities"], "expand": value["expansion_triggered"]} for key, value in summary["geometries"].items()},
    })
    (output / "command.txt").write_text(" ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n", encoding="utf-8")
    artifacts = {path.name: sha256_file(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "artifact_hashes.json"}
    sources = {str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve())}
    write_json(output / "artifact_hashes.json", {"artifacts": artifacts, "source_hashes": sources})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir")
    parser.add_argument("--fsim-hw-path", default="/tmp/vta-hw-fsim")
    parser.add_argument("--fsim-timeout", type=int, default=300)
    parser.add_argument("--phase", choices=("static", "all"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workload-id", choices=("R18-H1", "R18-H2", "R18-H3"))
    parser.add_argument("--worker-request")
    parser.add_argument("--worker-output")
    args = parser.parse_args()
    if bool(args.worker_request) != bool(args.worker_output):
        parser.error("worker request/output must be provided together")
    if not args.worker_request and not args.output_dir:
        parser.error("--output-dir is required")
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.worker_request:
        run_worker(parsed_args)
    else:
        run_main(parsed_args)
