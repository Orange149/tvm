#!/usr/bin/env python3
"""Lower and locally qualify the frozen P7R115 YOLO pilot without TopHub labels.

The parent process isolates every lower/FSim attempt in a bounded subprocess and
appends one record immediately.  Compiler/FSim wall times are diagnostic resource
costs only; they are never candidate performance labels or ranking features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tvm
from tvm import autotvm
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from c3_candidate_identity import canonical_json_bytes, make_failure, normalize_config_entity
from collect_vta_fsim_command_signatures import (
    array_sha256,
    command_signature,
    parse_queue_records,
)
from extract_vta_candidate_features import _descriptor_aggregates
from extract_static_vta_dma import extract_module_dma
from generate_vta_residency_candidates import (
    classify_lower_failure,
    transfer_signature,
    unique_tensor_bytes,
)
from qualify_vta_residency_fsim import reference_data_from_workload


STATIC_SCHEMA = "c3_p7r117_static_candidate_v1"
FSIM_SCHEMA = "c3_p7r117_fsim_candidate_v1"
SEARCH_SCHEMA = "c3_no_leak_search_pool_v1"
SEEDS = (0, 20250901, 20260910)
MODE_NUMBERS = {
    "original": 0,
    "input_stationary": 1,
    "weight_stationary": 2,
    "paper_inspired_hybrid": 3,
}
TIMING_KEYS = {
    "finalize_us",
    "uop_pack_us",
    "uop_copy_us",
    "insn_copy_us",
    "queue_cache_us",
    "device_run_us",
    "submit_mmio_us",
    "poll_wait_us",
}

HERE = Path(__file__).resolve().parent
C3 = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P7R115 = C3 / "07_grouped_holdout" / "20260911_p7r115_yolo_confirmation_protocol_run01"
DEFAULT_CONTRACT = P7R115 / "contract.json"
DEFAULT_CANDIDATES = P7R115 / "candidates.jsonl"
DEFAULT_OUTPUT = C3 / "07_grouped_holdout" / "20260911_p7r117_yolo_no_leak_local_run01"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def semantic_key(config):
    return canonical_json_bytes(normalize_config_entity(config))


def make_task(candidate, env):
    workload = candidate["identity"]["workload"]
    mode = candidate["residence_mode"]
    return autotvm.task.create(
        "conv2d_packed_residency.vta",
        args=tuple(workload[1:]) + (MODE_NUMBERS[mode],),
        target=env.target,
        target_host=env.target_host,
    )


def sram_features(candidate):
    certificate = candidate["hardware_predicate"]
    required, limits = certificate["required"], certificate["limits"]
    return {
        "certificate_passed": bool(certificate["passed"]),
        "required_vectors": dict(required),
        "capacity_vectors": dict(limits),
        "headroom_vectors": {key: int(limits[key]) - int(required[key]) for key in required},
        "scope": certificate["formula_scope"],
    }


def static_feature_vector(signature, sram):
    totals = signature["totals"]
    descriptors = signature["descriptor_aggregates"]
    by_memory = descriptors.get("by_memory", {})
    reloads = signature["reload_ratio"]
    return {
        "input_dma_bytes": int(totals.get("load_buffer_2d_inp_bytes", 0)),
        "weight_dma_bytes": int(totals.get("load_buffer_2d_wgt_bytes", 0)),
        "output_dma_bytes": int(totals.get("store_buffer_2d_out_bytes", 0)),
        "input_dma_calls": int(totals.get("load_buffer_2d_inp_calls", 0)),
        "weight_dma_calls": int(totals.get("load_buffer_2d_wgt_calls", 0)),
        "output_dma_calls": int(totals.get("store_buffer_2d_out_calls", 0)),
        "load_dma_bytes": int(totals.get("load_buffer_2d_bytes", 0)),
        "store_dma_bytes": int(totals.get("store_buffer_2d_bytes", 0)),
        "load_dma_calls": int(totals.get("load_buffer_2d_calls", 0)),
        "store_dma_calls": int(totals.get("store_buffer_2d_calls", 0)),
        "small_dma_calls": int(
            totals.get("load_buffer_2d_small_calls", 0)
            + totals.get("store_buffer_2d_small_calls", 0)
        ),
        "strided_dma_calls": int(
            totals.get("load_buffer_2d_strided_calls", 0)
            + totals.get("store_buffer_2d_strided_calls", 0)
        ),
        "padded_dma_calls": int(
            descriptors.get("padded_calls", {}).get("load", 0)
            + descriptors.get("padded_calls", {}).get("store", 0)
        ),
        "input_reload": float(reloads.get("input_load", 0.0)),
        "weight_reload": float(reloads.get("weight_load", 0.0)),
        "output_reload": float(reloads.get("output_store", 0.0)),
        "input_max_request_bytes": int(signature.get("max_request_bytes_by_memory", {}).get("inp", 0)),
        "weight_max_request_bytes": int(signature.get("max_request_bytes_by_memory", {}).get("wgt", 0)),
        "output_max_request_bytes": int(signature.get("max_request_bytes_by_memory", {}).get("out", 0)),
        "input_sram_required_vectors": int(sram["required_vectors"]["input_vectors"]),
        "weight_sram_required_vectors": int(sram["required_vectors"]["weight_vectors"]),
        "accumulator_sram_required_vectors": int(
            sram["required_vectors"]["accumulator_vectors"]
        ),
        "input_sram_headroom_vectors": int(sram["headroom_vectors"]["input_vectors"]),
        "weight_sram_headroom_vectors": int(sram["headroom_vectors"]["weight_vectors"]),
        "accumulator_sram_headroom_vectors": int(
            sram["headroom_vectors"]["accumulator_vectors"]
        ),
        "descriptor_rows": int(descriptors.get("descriptor_rows", 0)),
        "expanded_dma_calls": int(descriptors.get("expanded_calls", 0)),
        "input_descriptor_calls": int(by_memory.get("inp", {}).get("calls", 0)),
        "weight_descriptor_calls": int(by_memory.get("wgt", {}).get("calls", 0)),
        "output_descriptor_calls": int(by_memory.get("out", {}).get("calls", 0)),
    }


def static_worker_result(candidate):
    start = time.monotonic()
    result = {
        "schema": STATIC_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "workload_id": candidate["workload_id"],
        "family_id": candidate["family_id"],
        "residence_mode": candidate["residence_mode"],
        "debug": candidate["debug"],
        "status": "failed",
        "failure": None,
        "phase_wall_seconds": {},
        "performance_label": None,
        "wall_time_semantics": "diagnostic compiler resource cost; forbidden as runtime performance label",
    }
    env = vta.get_env()
    phase = "task_create"
    try:
        mark = time.monotonic()
        task = make_task(candidate, env)
        result["phase_wall_seconds"][phase] = time.monotonic() - mark
        index = int(candidate["debug"]["config_index"])
        config = task.config_space.get(index)
        if semantic_key(config.to_json_dict()) != semantic_key(
            candidate["identity"]["complete_config_entity"]
        ):
            raise RuntimeError("ConfigEntity differs from frozen P7R115 identity")

        phase = "instantiate"
        mark = time.monotonic()
        with task.target:
            schedule, tensors = task.instantiate(config)
        result["phase_wall_seconds"][phase] = time.monotonic() - mark

        phase = "tir_lower"
        mark = time.monotonic()
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            lowered = tvm.lower(schedule, tensors, name="main")
        result["phase_wall_seconds"][phase] = time.monotonic() - mark
        result["tir_sha256"] = hashlib.sha256(
            tvm.ir.save_json(lowered).encode("utf-8")
        ).hexdigest()

        phase = "static_dma_extract"
        mark = time.monotonic()
        dma = extract_module_dma(lowered, env)
        result["phase_wall_seconds"][phase] = time.monotonic() - mark
        signature = transfer_signature(
            dma,
            unique_tensor_bytes(candidate["identity"]["workload"]),
        )
        sram = sram_features(candidate)
        result.update(
            status="ok",
            transfer_signature=signature,
            static_feature_vector=static_feature_vector(signature, sram),
            sram_features=sram,
            command_features={
                "status": "requires_local_fsim_execution",
                "values": None,
            },
        )
    except Exception as error:  # TVM exposes several exception types.
        category = "environment" if phase in ("task_create", "identity_check") else "lower"
        failure = make_failure(
            category,
            (str(error) or type(error).__name__)[:4000],
            phase=phase,
            retryable=False,
        )
        failure["exception_type"] = type(error).__name__
        if category == "lower":
            failure["subcategory"] = classify_lower_failure(str(error), phase)
        result["failure"] = failure
    result["worker_wall_seconds"] = time.monotonic() - start
    return result


def numeric_delta(candidate, control):
    if candidate.get("status") != "ok" or control.get("status") != "ok":
        return None
    candidate_features = candidate["static_feature_vector"]
    control_features = control["static_feature_vector"]
    if set(candidate_features) != set(control_features):
        raise ValueError("same-tile feature schemas differ")
    return {
        key: float(candidate_features[key]) - float(control_features[key])
        for key in sorted(candidate_features)
    }


def build_no_leak_skeleton(candidates, static_results):
    if len(candidates) != len(static_results):
        raise ValueError("gross candidate/static result count mismatch")
    by_id = {row["candidate_id"]: row for row in static_results}
    if len(by_id) != len(static_results):
        raise ValueError("duplicate static candidate ID")
    controls = {}
    for candidate in candidates:
        if candidate["residence_mode"] == "original":
            key = (candidate["workload_id"], candidate["family_id"])
            if key in controls:
                raise ValueError("duplicate same-tile original for {}".format(key))
            controls[key] = by_id[candidate["candidate_id"]]
    rows = []
    for candidate in candidates:
        result = by_id[candidate["candidate_id"]]
        key = (candidate["workload_id"], candidate["family_id"])
        if key not in controls:
            raise ValueError("missing same-tile original for {}".format(key))
        control = controls[key]
        rows.append(
            {
                "schema": SEARCH_SCHEMA,
                "candidate_id": candidate["candidate_id"],
                "workload_id": candidate["workload_id"],
                "family_id": candidate["family_id"],
                "residence_mode": candidate["residence_mode"],
                "complete_config_entity": candidate["identity"]["complete_config_entity"],
                "debug": candidate["debug"],
                "gross_dispatch_position": len(rows),
                "lowering_status": result["status"],
                "failure": result.get("failure"),
                "absolute_static_features": result.get("static_feature_vector"),
                "same_tile_original_candidate_id": control["candidate_id"],
                "same_tile_delta_static_features": numeric_delta(result, control),
                "command_features": result.get(
                    "command_features", {"status": "unavailable_due_to_lower_failure", "values": None}
                ),
                "performance_label": None,
                "selection_eligibility": "lower_success" if result["status"] == "ok" else "failed_charged_no_replacement",
                "forbidden_feature_sources": ["TopHub", "board timing", "FSim wall time"],
            }
        )
    return rows


def fsim_worker_result(candidate, expected_tir_sha256):
    result = {
        "schema": FSIM_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "workload_id": candidate["workload_id"],
        "family_id": candidate["family_id"],
        "residence_mode": candidate["residence_mode"],
        "status": "failed",
        "failure": None,
        "seeds": [],
        "diagnostic_wall_seconds": {},
        "performance_measurement": "not_collected",
        "wall_time_semantics": "diagnostic feasibility cost; forbidden as search or latency feature",
    }
    start = time.monotonic()
    phase = "preflight"
    try:
        from vta.testing import simulator  # pylint: disable=import-outside-toplevel

        env = vta.get_env()
        if env.TARGET != "sim" or not simulator.enabled():
            raise RuntimeError("P7R117 FSim worker requires TARGET=sim and enabled FSim runtime")
        task = make_task(candidate, env)
        config = task.config_space.get(int(candidate["debug"]["config_index"]))
        if semantic_key(config.to_json_dict()) != semantic_key(
            candidate["identity"]["complete_config_entity"]
        ):
            raise RuntimeError("FSim ConfigEntity differs from frozen P7R115 identity")

        phase = "build"
        mark = time.monotonic()
        with task.target:
            schedule, tensors = task.instantiate(config)
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            lowered = tvm.lower(schedule, tensors, name="main")
        result["tir_sha256"] = hashlib.sha256(
            tvm.ir.save_json(lowered).encode("utf-8")
        ).hexdigest()
        result["tir_matches_static"] = result["tir_sha256"] == expected_tir_sha256
        if not result["tir_matches_static"]:
            raise RuntimeError("FSim re-lowered TIR differs from frozen static result")
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            module = vta.build(
                schedule,
                tensors,
                target=tvm.target.Target(env.target, host=env.target_host),
                name="main",
            )
        function = module["main"]
        device = tvm.device("ext_dev", 0)
        result["diagnostic_wall_seconds"][phase] = time.monotonic() - mark

        workload = candidate["identity"]["workload"]
        # The local runtime's first submission can differ by one 16-byte command
        # because queue initialization is cold.  Execute and verify one declared
        # warmup before emitting seed markers; its queue record is deliberately
        # excluded from command-signature extraction.
        phase = "correctness_warmup"
        mark = time.monotonic()
        warm_data, warm_weight, warm_expected = reference_data_from_workload(workload, SEEDS[0])
        warm_output = tvm.nd.empty(warm_expected.shape, "int8", device=device)
        function(
            tvm.nd.array(warm_data, device),
            tvm.nd.array(warm_weight, device),
            warm_output,
        )
        warm_actual = warm_output.numpy()
        warm_mismatch = int(np.count_nonzero(warm_actual != warm_expected))
        result["correctness_warmup"] = {
            "seed": SEEDS[0],
            "correct": warm_mismatch == 0,
            "mismatch_count": warm_mismatch,
            "excluded_from_command_signature": True,
            "wall_seconds": time.monotonic() - mark,
        }
        if warm_mismatch:
            raise RuntimeError("correctness warmup differs in {} elements".format(warm_mismatch))
        for seed in SEEDS:
            seed_row = {"seed": seed, "status": "failed", "correct": False}
            phase = "reference_seed_{}".format(seed)
            mark = time.monotonic()
            data, weight, expected = reference_data_from_workload(workload, seed)
            seed_row["reference_wall_seconds"] = time.monotonic() - mark
            output = tvm.nd.empty(expected.shape, "int8", device=device)
            print("[P7R117_SEED] {}".format(seed), file=sys.stderr, flush=True)
            phase = "execute_seed_{}".format(seed)
            mark = time.monotonic()
            function(tvm.nd.array(data, device), tvm.nd.array(weight, device), output)
            seed_row["execution_wall_seconds"] = time.monotonic() - mark
            actual = output.numpy()
            mismatch = int(np.count_nonzero(actual != expected))
            seed_row.update(
                status="passed" if mismatch == 0 else "failed",
                correct=mismatch == 0,
                mismatch_count=mismatch,
                expected_sha256=array_sha256(expected),
                actual_sha256=array_sha256(actual),
            )
            if mismatch:
                seed_row["failure"] = make_failure(
                    "wrong_answer",
                    "{} output elements differ".format(mismatch),
                    phase=phase,
                )
            result["seeds"].append(seed_row)
        result["status"] = (
            "passed" if len(result["seeds"]) == len(SEEDS) and all(row["correct"] for row in result["seeds"])
            else "failed"
        )
        if result["status"] == "failed":
            result["failure"] = next(
                (row.get("failure") for row in result["seeds"] if row.get("failure")),
                make_failure("wrong_answer", "one or more FSim seeds failed", phase="execute"),
            )
    except Exception as error:
        category = "wrong_answer" if phase.startswith("execute") else (
            "compile" if phase == "build" else "environment"
        )
        failure = make_failure(
            category,
            (str(error) or type(error).__name__)[:4000],
            phase=phase,
        )
        failure["exception_type"] = type(error).__name__
        result["failure"] = failure
    result["worker_wall_seconds"] = time.monotonic() - start
    return result


def parse_seed_queue_records(stderr):
    sections = {seed: [] for seed in SEEDS}
    current = None
    for line in stderr.splitlines():
        if "[P7R117_SEED] " in line:
            try:
                current = int(line.split("[P7R117_SEED] ", 1)[1].strip())
            except ValueError:
                current = None
            continue
        if current in sections and "[VTA_QUEUE] " in line:
            sections[current].append(line)
    parsed = {}
    malformed = {}
    for seed, lines in sections.items():
        parsed[seed], malformed[seed] = parse_queue_records("\n".join(lines))
    return parsed, malformed


def normalize_command_structural(structural):
    """Remove process-global submit numbers while preserving within-call order."""

    normalized = json.loads(json.dumps(structural))
    for ordinal, row in enumerate(normalized.get("submit_sequence", []), 1):
        row.pop("submit", None)
        row["submit_ordinal_within_inference"] = ordinal
    return normalized


def attach_command_features(result, stderr):
    if result.get("status") != "passed":
        result["command_features"] = {
            "status": "unavailable_due_to_fsim_failure",
            "values": None,
        }
        return result
    records, malformed = parse_seed_queue_records(stderr)
    signatures = {}
    for seed in SEEDS:
        if malformed[seed] or not records[seed]:
            result["failure"] = make_failure(
                "environment",
                "missing or malformed VTA_QUEUE diagnostics for seed {}".format(seed),
                phase="command_signature",
            )
            result["status"] = "failed"
            result["command_features"] = {
                "status": "unavailable_due_to_diagnostics",
                "values": None,
            }
            return result
        signatures[seed] = normalize_command_structural(
            command_signature(
                records[seed], MODE_NUMBERS[result["residence_mode"]], 0
            )["structural"]
        )
    hashes = {seed: canonical_sha256(signatures[seed]) for seed in SEEDS}
    if len(set(hashes.values())) != 1:
        result["failure"] = make_failure(
            "environment",
            "three FSim seeds produced different structural command signatures",
            phase="command_signature",
        )
        result["status"] = "failed"
        result["command_features"] = {
            "status": "inconsistent_across_seeds",
            "values": None,
            "per_seed_sha256": hashes,
        }
        return result
    structural = signatures[SEEDS[0]]
    result["command_features"] = {
        "status": "available_three_seed_consistent",
        "values": structural,
        "per_seed_sha256": hashes,
        "raw_queue_timing_fields": "discarded",
        "process_global_submit_numbers": "normalized_to_per-inference_ordinals",
    }
    return result


def timeout_static(candidate, seconds):
    failure = make_failure(
        "timeout",
        "static worker exceeded {} seconds".format(seconds),
        phase="static_worker",
        retryable=True,
    )
    return {
        "schema": STATIC_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "workload_id": candidate["workload_id"],
        "family_id": candidate["family_id"],
        "residence_mode": candidate["residence_mode"],
        "debug": candidate["debug"],
        "status": "failed",
        "failure": failure,
        "performance_label": None,
        "wall_time_semantics": "bounded diagnostic compiler resource cost",
    }


def timeout_fsim(candidate, seconds):
    return {
        "schema": FSIM_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "workload_id": candidate["workload_id"],
        "family_id": candidate["family_id"],
        "residence_mode": candidate["residence_mode"],
        "status": "failed",
        "failure": make_failure(
            "timeout",
            "FSim worker exceeded {} seconds".format(seconds),
            phase="fsim_worker",
            retryable=True,
        ),
        "seeds": [],
        "performance_measurement": "not_collected",
    }


def run_isolated_worker(candidate, phase, timeout_seconds, fsim_hw_path=None, expected_tir=None):
    with tempfile.TemporaryDirectory(prefix="c3_p7r117_{}_".format(phase)) as directory:
        candidate_path = Path(directory) / "candidate.json"
        output_path = Path(directory) / "result.json"
        write_json(candidate_path, candidate)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-phase",
            phase,
            "--worker-candidate",
            str(candidate_path),
            "--worker-output",
            str(output_path),
        ]
        if expected_tir:
            command.extend(["--worker-expected-tir", expected_tir])
        environment = os.environ.copy()
        if phase == "fsim":
            environment["VTA_HW_PATH"] = str(Path(fsim_hw_path).resolve())
            environment["VTA_QUEUE_DIAGNOSTICS"] = "1"
        start = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=environment,
            )
            wall = time.monotonic() - start
        except subprocess.TimeoutExpired as error:
            result = (
                timeout_static(candidate, timeout_seconds)
                if phase == "static"
                else timeout_fsim(candidate, timeout_seconds)
            )
            result["parent_observed_wall_seconds"] = time.monotonic() - start
            return result, (error.stderr or "")
        if completed.returncode != 0 or not output_path.is_file():
            result = (
                timeout_static(candidate, timeout_seconds)
                if phase == "static"
                else timeout_fsim(candidate, timeout_seconds)
            )
            result["failure"] = make_failure(
                "environment",
                "worker exit={} output_present={}".format(completed.returncode, output_path.is_file()),
                phase="{}_worker_process".format(phase),
            )
        else:
            result = load_json(output_path)
        result["parent_observed_wall_seconds"] = wall
        return result, completed.stderr


def verify_inputs(contract_path, candidates_path):
    ledger = load_json(Path(contract_path).parent / "artifact_hashes.json")["artifacts"]
    for path in (Path(contract_path), Path(candidates_path)):
        if ledger.get(path.name) != sha256_file(path):
            raise ValueError("P7R115 frozen input hash mismatch: {}".format(path))
    contract = load_json(contract_path)
    candidates = load_jsonl(candidates_path)
    if contract.get("schema") != "c3_p7r115_yolo_confirmation_contract_v1":
        raise ValueError("unexpected P7R115 contract schema")
    if contract.get("candidate_pool_commitment_sha256") != canonical_sha256(candidates):
        raise ValueError("P7R115 candidate commitment mismatch")
    if len(candidates) != int(contract["candidate_count"]):
        raise ValueError("P7R115 candidate count mismatch")
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise ValueError("duplicate P7R115 candidate ID")
    return contract, candidates


def parse_workloads(text):
    values = tuple(item.strip().upper() for item in str(text).split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("--workloads must be a nonempty unique CSV")
    return values


def run_worker(args):
    candidate = load_json(args.worker_candidate)
    if args.worker_phase == "static":
        result = static_worker_result(candidate)
    else:
        if not args.worker_expected_tir:
            raise ValueError("FSim worker requires --worker-expected-tir")
        result = fsim_worker_result(candidate, args.worker_expected_tir)
    write_json(args.worker_output, result)


def run_main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    contract, all_candidates = verify_inputs(args.contract, args.candidates)
    known_workloads = {row["workload_id"] for row in all_candidates}
    if not set(args.workloads) <= known_workloads:
        raise ValueError("unknown workload selection: {}".format(sorted(set(args.workloads) - known_workloads)))
    candidates = [row for row in all_candidates if row["workload_id"] in args.workloads]
    output.mkdir(parents=True)
    preregistered = {
        "schema": "c3_p7r117_no_leak_local_protocol_v1",
        "status": "frozen_before_static_lowering",
        "source_contract_sha256": sha256_file(args.contract),
        "source_candidates_sha256": sha256_file(args.candidates),
        "source_pool_commitment_sha256": contract["candidate_pool_commitment_sha256"],
        "selected_workloads": list(args.workloads),
        "gross_candidates": len(candidates),
        "gross_candidate_order": [row["candidate_id"] for row in candidates],
        "lower_timeout_seconds_per_candidate": args.lower_timeout_seconds,
        "fsim_timeout_seconds_per_candidate": args.fsim_timeout_seconds,
        "fsim_seeds": list(SEEDS),
        "fsim_correctness_warmup": {
            "count_per_candidate": 1,
            "seed": SEEDS[0],
            "must_be_correct": True,
            "command_record_excluded": True,
            "reason": "remove the known first-submit queue-initialization command difference",
        },
        "failure_policy": "every timeout/failure is charged; no replacement or adaptive candidate",
        "feature_policy": "static TIR/DMA/SRAM and later structural command features only",
        "forbidden_selection_or_feature_sources": [
            "TopHub ConfigEntity/index/cost",
            "board timing",
            "FSim or compiler wall time",
        ],
        "wall_time_policy": "recorded only as feasibility/resource cost",
        "board_contacted": False,
    }
    write_json(output / "preregistered.json", preregistered)
    write_json(
        output / "pre_static_hashes.json",
        {"artifacts": {"preregistered.json": sha256_file(output / "preregistered.json")}},
    )

    static_path = output / "static_results.jsonl"
    stderr_path = output / "stderr.log"
    static_path.touch(exist_ok=False)
    stderr_path.touch(exist_ok=False)
    static_results = []
    start = time.monotonic()
    for position, candidate in enumerate(candidates):
        result, stderr = run_isolated_worker(
            candidate, "static", args.lower_timeout_seconds
        )
        result["gross_dispatch_position"] = position
        static_results.append(result)
        with static_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
        if stderr.strip():
            with stderr_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    "[static] {}/{} {} {}\n{}\n".format(
                        position + 1,
                        len(candidates),
                        candidate["candidate_id"],
                        candidate["workload_id"],
                        stderr[-4000:],
                    )
                )
        print(
            "static {}/{} {} {} {} {:.3f}s".format(
                position + 1,
                len(candidates),
                candidate["workload_id"],
                candidate["residence_mode"],
                result["status"],
                result["parent_observed_wall_seconds"],
            ),
            flush=True,
        )

    skeleton = build_no_leak_skeleton(candidates, static_results)
    skeleton_path = output / "predispatch_skeleton.jsonl"
    skeleton_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in skeleton), encoding="utf-8"
    )
    pre_fsim = {
        "static_results.jsonl": sha256_file(static_path),
        "predispatch_skeleton.jsonl": sha256_file(skeleton_path),
    }
    write_json(output / "pre_fsim_hashes.json", {"artifacts": pre_fsim})

    fsim_results = []
    fsim_path = output / "fsim_results.jsonl"
    fsim_path.touch(exist_ok=False)
    static_by_id = {row["candidate_id"]: row for row in static_results}
    if not args.skip_fsim:
        for position, candidate in enumerate(candidates):
            static = static_by_id[candidate["candidate_id"]]
            if static["status"] != "ok":
                continue
            result, stderr = run_isolated_worker(
                candidate,
                "fsim",
                args.fsim_timeout_seconds,
                fsim_hw_path=args.fsim_hw_path,
                expected_tir=static["tir_sha256"],
            )
            result = attach_command_features(result, stderr)
            result["gross_dispatch_position"] = position
            fsim_results.append(result)
            with fsim_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, sort_keys=True) + "\n")
            if stderr.strip():
                safe_stderr = "\n".join(
                    "[VTA_QUEUE] <parsed; timing fields discarded>"
                    if "[VTA_QUEUE] " in line
                    else line
                    for line in stderr.splitlines()
                )
                with stderr_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        "[fsim] {}/{} {} {}\n{}\n".format(
                            position + 1,
                            len(candidates),
                            candidate["candidate_id"],
                            candidate["workload_id"],
                            safe_stderr[-4000:],
                        )
                    )
            print(
                "fsim {}/{} {} {} {} {:.3f}s".format(
                    position + 1,
                    len(candidates),
                    candidate["workload_id"],
                    candidate["residence_mode"],
                    result["status"],
                    result["parent_observed_wall_seconds"],
                ),
                flush=True,
            )

    lower_failures = Counter(
        row["failure"]["category"] for row in static_results if row.get("failure")
    )
    fsim_failures = Counter(
        row["failure"]["category"] for row in fsim_results if row.get("failure")
    )
    summary = {
        "schema": "c3_p7r117_no_leak_local_summary_v1",
        "status": "completed_with_recorded_outcomes",
        "scope": "local lowering and three-seed FSim feasibility; no board/performance result",
        "selected_workloads": list(args.workloads),
        "gross_candidates": len(candidates),
        "static_records": len(static_results),
        "static_ok": sum(row["status"] == "ok" for row in static_results),
        "static_failed": sum(row["status"] != "ok" for row in static_results),
        "static_failure_categories": dict(sorted(lower_failures.items())),
        "fsim_attempted": len(fsim_results),
        "fsim_passed": sum(row["status"] == "passed" for row in fsim_results),
        "fsim_failed": sum(row["status"] != "passed" for row in fsim_results),
        "fsim_failure_categories": dict(sorted(fsim_failures.items())),
        "command_features_available": sum(
            row.get("command_features", {}).get("status") == "available_three_seed_consistent"
            for row in fsim_results
        ),
        "skip_fsim": bool(args.skip_fsim),
        "elapsed_wall_seconds": time.monotonic() - start,
        "elapsed_wall_semantics": "workflow feasibility/resource cost only",
        "board_contacted": False,
        "tophub_used_for_selection_or_features": False,
    }
    write_json(output / "summary.json", summary)
    (output / "STATUS.md").write_text(
        "# P7R117 YOLO no-leak local pilot\n\n"
        "- Status: `completed_with_recorded_outcomes`\n"
        "- Static lower: {}/{} successful\n"
        "- Three-seed FSim: {}/{} successful\n"
        "- Command features: {}/{} available\n"
        "- TopHub used for selection/features: no\n"
        "- Board contacted: no\n"
        "- Claim: pilot feasibility only, not latency/FPS or final CCF-B evidence\n".format(
            summary["static_ok"],
            summary["gross_candidates"],
            summary["fsim_passed"],
            summary["fsim_attempted"],
            summary["command_features_available"],
            summary["fsim_attempted"],
        ),
        encoding="utf-8",
    )
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    hashes = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(output / "artifact_hashes.json", {"artifacts": hashes})
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workloads", type=parse_workloads, default=("Y00", "Y01", "Y02"))
    parser.add_argument("--lower-timeout-seconds", type=int, default=300)
    parser.add_argument("--fsim-timeout-seconds", type=int, default=120)
    parser.add_argument("--fsim-hw-path", type=Path, default=Path("/tmp/vta-hw-fsim"))
    parser.add_argument("--skip-fsim", action="store_true")
    parser.add_argument("--worker-phase", choices=("static", "fsim"))
    parser.add_argument("--worker-candidate", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--worker-expected-tir")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker_phase:
        if not args.worker_candidate or not args.worker_output:
            raise ValueError("worker mode requires candidate and output paths")
        run_worker(args)
    else:
        if min(args.lower_timeout_seconds, args.fsim_timeout_seconds) <= 0:
            raise ValueError("timeouts must be positive")
        run_main(args)


if __name__ == "__main__":
    main()
