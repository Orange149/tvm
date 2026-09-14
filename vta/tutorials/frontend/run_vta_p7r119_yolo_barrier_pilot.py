#!/usr/bin/env python3
"""Audit an explicit-barrier weight-residency mode on label-isolated Y02 tiles.

This is a local pilot, not a board benchmark.  It uses three full-height Y02
ConfigEntities selected from the frozen P7R118 legal set by a structural rule,
and compares the same tile under original, input-stationary, and implementation
mode 4.  Mode 4 is published here only as ``weight_resident_barrier``: it keeps
weights across a spatial region and explicitly drains the command pipeline at
the residency boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from collect_vta_fsim_command_signatures import (
    array_sha256,
    command_signature,
    parse_queue_records,
)
from extract_static_vta_dma import extract_module_dma
from generate_vta_residency_candidates import (
    classify_lower_failure,
    transfer_signature,
    unique_tensor_bytes,
)
from qualify_vta_residency_fsim import reference_data_from_workload


SCHEMA = "c3_p7r119_yolo_barrier_pilot_v2"
IDENTITY_SCHEMA = "c3_candidate_id_v2"
MODES = {
    "original": 0,
    "input_stationary": 1,
    "weight_resident_barrier": 4,
}
LEGACY_ABLATIONS = {
    "weight_stationary": 2,
    "paper_inspired_hybrid": 3,
}
SEEDS = (0, 20250901, 20260910)
SELECTED_TILE_CO = (1, 2, 8)

HERE = Path(__file__).resolve().parent
C3 = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P7R115 = C3 / "07_grouped_holdout" / "20260911_p7r115_yolo_confirmation_protocol_run01"
P7R118 = C3 / "07_grouped_holdout" / "20260911_p7r118_y02_legality_run01"
P3E = C3 / "03_residency_schedules" / "20260910_p3e_weight_barrier_legalizer_run01"
DEFAULT_CANDIDATES = P7R115 / "candidates.jsonl"
DEFAULT_DOMAIN = P7R115 / "complete_domain.jsonl"
DEFAULT_LEGALITY = P7R118 / "four_mode_scan.jsonl"
DEFAULT_P3E = P3E / "results.jsonl"
DEFAULT_OUTPUT = (
    C3
    / "07_grouped_holdout"
    / "20260911_p7r119_y02_weight_resident_barrier_pilot_v2_run01"
)


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


def verify_frozen(path):
    path = Path(path)
    ledger = load_json(path.parent / "artifact_hashes.json")["artifacts"]
    observed = sha256_file(path)
    if ledger.get(path.name) != observed:
        raise ValueError("frozen input hash mismatch: {}".format(path))
    return {"path": str(path.resolve()), "sha256": observed}


def config_knobs(entity):
    return {
        name: int(value[-1] if kind == "sp" else value)
        for name, kind, value in entity["entity"]
    }


def select_structural_probe_rows(rows):
    """Select a full-height tile_co sweep without DMA or timing labels."""

    legal = [row for row in rows if row.get("all_four_modes_legal")]
    if not legal:
        raise ValueError("P7R118 contains no four-mode-legal Y02 rows")
    output_height = 26
    selected = []
    for tile_co in SELECTED_TILE_CO:
        matches = [
            row
            for row in legal
            if row["knobs"]["tile_h"] == output_height
            and row["knobs"]["tile_w"] == 2
            and row["knobs"]["tile_ci"] == 1
            and row["knobs"]["tile_co"] == tile_co
            and row["knobs"]["oc_nthread"] == 1
            and row["knobs"]["h_nthread"] == 1
        ]
        if len(matches) != 1:
            raise ValueError("expected one structural anchor for tile_co={}".format(tile_co))
        selected.append(matches[0])
    return selected


def candidate_identity(base, mode, mode_number):
    payload = {
        "schema": IDENTITY_SCHEMA,
        "hardware_fingerprint": base["identity"]["hardware_fingerprint"],
        "template_name": base["identity"]["template_name"],
        "schedule_version": base["identity"]["schedule_version"],
        "workload": base["identity"]["workload"],
        "public_mode": mode,
        "implementation_mode": int(mode_number),
        "complete_config_entity": normalize_config_entity(
            base["complete_config_entity"]
        ),
    }
    return {
        "candidate_id": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        "identity": payload,
        "debug": {"config_index": int(base["debug"]["config_index"])},
    }


def build_candidates(candidate_rows, domain_rows, legality_rows):
    y02_templates = [row for row in candidate_rows if row["workload_id"] == "Y02"]
    if not y02_templates:
        raise ValueError("missing Y02 candidate template")
    template = y02_templates[0]
    domains = {
        int(row["debug"]["config_index"]): row
        for row in domain_rows
        if row["workload_id"] == "Y02"
    }
    selected = select_structural_probe_rows(legality_rows)
    candidates = []
    for family_ordinal, legal in enumerate(selected):
        index = int(legal["debug"]["config_index"])
        domain = domains[index]
        base = {
            "identity": template["identity"],
            "complete_config_entity": domain["complete_config_entity"],
            "debug": {"config_index": index},
        }
        family_id = "Y02B{:02d}".format(family_ordinal)
        for mode, mode_number in MODES.items():
            identity = candidate_identity(base, mode, mode_number)
            candidates.append(
                {
                    **identity,
                    "schema": SCHEMA,
                    "workload_id": "Y02",
                    "family_id": family_id,
                    "family_ordinal": family_ordinal,
                    "public_mode": mode,
                    "implementation_mode": mode_number,
                    "knobs": legal["knobs"],
                    "selection_basis": (
                        "P7R118 four-mode legal; full-height, tile_w=2, tile_ci=1; "
                        "tile_co sweep 1/2/8; no DMA, FSim, board, or TopHub label"
                    ),
                    "performance_label": None,
                }
            )
    return candidates


def create_task(candidate, env):
    return autotvm.task.create(
        "conv2d_packed_residency.vta",
        args=tuple(candidate["identity"]["workload"][1:])
        + (int(candidate["implementation_mode"]),),
        target=env.target,
        target_host=env.target_host,
    )


def instantiate(candidate, env):
    task = create_task(candidate, env)
    config = task.config_space.get(int(candidate["debug"]["config_index"]))
    actual = normalize_config_entity(config.to_json_dict())
    expected = candidate["identity"]["complete_config_entity"]
    if canonical_json_bytes(actual) != canonical_json_bytes(expected):
        raise RuntimeError("ConfigEntity differs from frozen semantic identity")
    with task.target:
        schedule, tensors = task.instantiate(config)
    return schedule, tensors


def dependency_audit(module):
    pairs = []

    def visit(node):
        if not isinstance(node, tir.Call) or not isinstance(node.op, tvm.ir.Op):
            return
        if node.op.name in ("tir.vta.coproc_dep_push", "tir.vta.coproc_dep_pop"):
            pairs.append(
                (node.op.name.rsplit("_", 1)[-1], int(node.args[0]), int(node.args[1]))
            )

    tir.stmt_functor.post_order_visit(module["main"].body, visit)
    pushes = Counter((src, dst) for kind, src, dst in pairs if kind == "push")
    pops = Counter((src, dst) for kind, src, dst in pairs if kind == "pop")
    forbidden = sum(
        count for (src, dst), count in pushes.items() if {src, dst} == {1, 3}
    )
    return {
        "balanced": pushes == pops,
        "forbidden_direct_1_3_count": forbidden,
        "push": {"{}->{}".format(*key): value for key, value in sorted(pushes.items())},
        "pop": {"{}->{}".format(*key): value for key, value in sorted(pops.items())},
    }


def sync_call_multiplicity(module):
    loop_extents = []
    multiplicities = []

    def preorder(node):
        if isinstance(node, tir.For):
            loop_extents.append(int(node.extent))
        if (
            isinstance(node, tir.Call)
            and isinstance(node.op, tvm.ir.Op)
            and node.op.name == "tir.vta.coproc_sync"
        ):
            product = 1
            for extent in loop_extents:
                product *= extent
            multiplicities.append(product)

    def postorder(node):
        if isinstance(node, tir.For):
            loop_extents.pop()

    tir.stmt_functor.ir_transform(module["main"].body, preorder, postorder, None)
    return multiplicities


def static_result(candidate):
    result = {
        "schema": SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "workload_id": candidate["workload_id"],
        "family_id": candidate["family_id"],
        "public_mode": candidate["public_mode"],
        "implementation_mode": candidate["implementation_mode"],
        "debug": candidate["debug"],
        "knobs": candidate["knobs"],
        "status": "failed",
        "failure": None,
        "performance_label": None,
    }
    phase = "instantiate"
    start = time.monotonic()
    try:
        env = vta.get_env()
        schedule, tensors = instantiate(candidate, env)
        phase = "tir_lower"
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            module = tvm.lower(schedule, tensors, name="main")
        phase = "static_dma_extract"
        dma = extract_module_dma(module, env)
        signature = transfer_signature(
            dma, unique_tensor_bytes(candidate["identity"]["workload"])
        )
        totals = signature["totals"]
        sync = sync_call_multiplicity(module)
        dependencies = dependency_audit(module)
        if not dependencies["balanced"] or dependencies["forbidden_direct_1_3_count"]:
            raise RuntimeError("dependency audit rejected the lowered module")
        if candidate["implementation_mode"] == 4 and (
            len(sync) < 2 or int(sync[-1]) != 1
        ):
            raise RuntimeError(
                "mode4 must contain residency drains followed by one function-final drain"
            )
        result.update(
            status="ok",
            tir_sha256=hashlib.sha256(tvm.ir.save_json(module).encode("utf-8")).hexdigest(),
            transfer_signature=signature,
            dma_summary={
                "input_bytes": int(totals.get("load_buffer_2d_inp_bytes", 0)),
                "input_calls": int(totals.get("load_buffer_2d_inp_calls", 0)),
                "weight_bytes": int(totals.get("load_buffer_2d_wgt_bytes", 0)),
                "weight_calls": int(totals.get("load_buffer_2d_wgt_calls", 0)),
                "load_bytes": int(totals.get("load_buffer_2d_bytes", 0)),
                "load_calls": int(totals.get("load_buffer_2d_calls", 0)),
                "output_bytes": int(totals.get("store_buffer_2d_out_bytes", 0)),
                "output_calls": int(totals.get("store_buffer_2d_out_calls", 0)),
            },
            sync={
                "static_execution_multiplicity": sync,
                "residency_drains": int(sync[0])
                if candidate["implementation_mode"] == 4 and len(sync) >= 2
                else 0,
                "function_final_drain": int(sync[-1]) if sync else 0,
            },
            dependency_audit=dependencies,
        )
    except Exception as error:  # TVM exposes several lowering exception types.
        message = str(error) or type(error).__name__
        failure = make_failure(
            "lower", message[:4000], phase=phase, retryable=False
        )
        failure["exception_type"] = type(error).__name__
        failure["subcategory"] = classify_lower_failure(message, phase)
        result["failure"] = failure
    result["diagnostic_wall_seconds"] = time.monotonic() - start
    result["wall_time_semantics"] = "compiler resource cost only; not latency"
    return result


def fsim_worker(candidate, expected_tir):
    result = {
        "schema": SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "workload_id": candidate["workload_id"],
        "family_id": candidate["family_id"],
        "public_mode": candidate["public_mode"],
        "implementation_mode": candidate["implementation_mode"],
        "status": "failed",
        "failure": None,
        "seeds": [],
        "performance_measurement": "not_collected",
    }
    phase = "environment"
    start = time.monotonic()
    try:
        from vta.testing import simulator  # pylint: disable=import-outside-toplevel

        env = vta.get_env()
        if env.TARGET != "sim" or not simulator.enabled():
            raise RuntimeError("worker requires local FSim")
        phase = "build"
        schedule, tensors = instantiate(candidate, env)
        with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
            lowered = tvm.lower(schedule, tensors, name="main")
        tir_hash = hashlib.sha256(tvm.ir.save_json(lowered).encode("utf-8")).hexdigest()
        result["tir_sha256"] = tir_hash
        result["tir_matches_static"] = tir_hash == expected_tir
        if not result["tir_matches_static"]:
            raise RuntimeError("FSim TIR differs from frozen static TIR")
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

        phase = "warmup"
        data, weight, expected = reference_data_from_workload(workload, SEEDS[0])
        output = tvm.nd.empty(expected.shape, "int8", device=device)
        function(tvm.nd.array(data, device), tvm.nd.array(weight, device), output)
        mismatch = int(np.count_nonzero(output.numpy() != expected))
        result["correctness_warmup"] = {
            "seed": SEEDS[0],
            "correct": mismatch == 0,
            "mismatch_count": mismatch,
            "excluded_from_command_signature": True,
        }
        if mismatch:
            raise RuntimeError("warmup wrong answer: {} mismatches".format(mismatch))

        for seed in SEEDS:
            phase = "execute_seed_{}".format(seed)
            data, weight, expected = reference_data_from_workload(workload, seed)
            output = tvm.nd.empty(expected.shape, "int8", device=device)
            print("[P7R119_SEED] {}".format(seed), file=sys.stderr, flush=True)
            function(tvm.nd.array(data, device), tvm.nd.array(weight, device), output)
            actual = output.numpy()
            mismatch = int(np.count_nonzero(actual != expected))
            result["seeds"].append(
                {
                    "seed": seed,
                    "correct": mismatch == 0,
                    "mismatch_count": mismatch,
                    "expected_sha256": array_sha256(expected),
                    "actual_sha256": array_sha256(actual),
                }
            )
        result["status"] = (
            "passed" if all(row["correct"] for row in result["seeds"]) else "failed"
        )
        if result["status"] != "passed":
            result["failure"] = make_failure(
                "wrong_answer", "one or more seeds failed", phase="execute"
            )
    except Exception as error:
        category = (
            "wrong_answer"
            if phase == "warmup" or phase.startswith("execute_seed_")
            else ("compile" if phase == "build" else "environment")
        )
        result["failure"] = make_failure(
            category, (str(error) or type(error).__name__)[:4000], phase=phase
        )
        result["failure"]["exception_type"] = type(error).__name__
    result["diagnostic_wall_seconds"] = time.monotonic() - start
    result["wall_time_semantics"] = "FSim feasibility cost only; not latency"
    return result


def parse_seed_records(stderr):
    sections = {seed: [] for seed in SEEDS}
    current = None
    for line in stderr.splitlines():
        if "[P7R119_SEED] " in line:
            try:
                current = int(line.split("[P7R119_SEED] ", 1)[1].strip())
            except ValueError:
                current = None
        elif current in sections and "[VTA_QUEUE] " in line:
            sections[current].append(line)
    parsed = {}
    malformed = {}
    for seed, lines in sections.items():
        parsed[seed], malformed[seed] = parse_queue_records("\n".join(lines))
    return parsed, malformed


def normalize_structural(structural):
    normalized = json.loads(json.dumps(structural))
    for ordinal, row in enumerate(normalized.get("submit_sequence", []), 1):
        row.pop("submit", None)
        row["submit_ordinal_within_inference"] = ordinal
    return normalized


def attach_command_evidence(result, stderr, expected_drains):
    if result.get("status") != "passed":
        result["command_evidence"] = {"status": "unavailable", "values": None}
        return result
    records, malformed = parse_seed_records(stderr)
    signatures = {}
    for seed in SEEDS:
        if malformed[seed] or not records[seed]:
            result["status"] = "failed"
            result["failure"] = make_failure(
                "environment",
                "missing or malformed command diagnostics for seed {}".format(seed),
                phase="command_signature",
            )
            result["command_evidence"] = {"status": "unavailable", "values": None}
            return result
        signatures[seed] = normalize_structural(
            command_signature(
                records[seed], result["implementation_mode"], expected_drains
            )["structural"]
        )
    hashes = {str(seed): canonical_sha256(signatures[seed]) for seed in SEEDS}
    if len(set(hashes.values())) != 1:
        result["status"] = "failed"
        result["failure"] = make_failure(
            "environment", "command structure differs across seeds", phase="command_signature"
        )
        result["command_evidence"] = {
            "status": "inconsistent_across_seeds",
            "values": None,
            "per_seed_sha256": hashes,
        }
        return result
    result["command_evidence"] = {
        "status": "available_three_seed_consistent",
        "values": signatures[SEEDS[0]],
        "per_seed_sha256": hashes,
        "raw_timing_fields": "discarded",
    }
    return result


def run_fsim(candidate, expected_tir, expected_drains, timeout, fsim_hw_path):
    with tempfile.TemporaryDirectory(prefix="c3_p7r119_fsim_") as directory:
        directory = Path(directory)
        candidate_path = directory / "candidate.json"
        result_path = directory / "result.json"
        write_json(candidate_path, candidate)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-candidate",
            str(candidate_path),
            "--worker-expected-tir",
            expected_tir,
            "--worker-output",
            str(result_path),
        ]
        environment = os.environ.copy()
        environment["VTA_HW_PATH"] = str(Path(fsim_hw_path).resolve())
        environment["VTA_QUEUE_DIAGNOSTICS"] = "1"
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=environment,
            )
        except subprocess.TimeoutExpired as error:
            result = {
                "schema": SCHEMA,
                "candidate_id": candidate["candidate_id"],
                "workload_id": candidate["workload_id"],
                "family_id": candidate["family_id"],
                "public_mode": candidate["public_mode"],
                "implementation_mode": candidate["implementation_mode"],
                "status": "failed",
                "failure": make_failure(
                    "timeout", "FSim exceeded {} seconds".format(timeout), phase="fsim"
                ),
                "performance_measurement": "not_collected",
            }
            return result, error.stderr or ""
        if completed.returncode != 0 or not result_path.is_file():
            result = {
                "schema": SCHEMA,
                "candidate_id": candidate["candidate_id"],
                "workload_id": candidate["workload_id"],
                "family_id": candidate["family_id"],
                "public_mode": candidate["public_mode"],
                "implementation_mode": candidate["implementation_mode"],
                "status": "failed",
                "failure": make_failure(
                    "environment",
                    "worker exit={} output_present={}".format(
                        completed.returncode, result_path.is_file()
                    ),
                    phase="fsim_worker",
                ),
                "performance_measurement": "not_collected",
            }
        else:
            result = load_json(result_path)
        return attach_command_evidence(result, completed.stderr, expected_drains), completed.stderr


def minimum_access_policy(static_rows):
    """Return a parameterized real-mechanism policy, never a hybrid candidate."""

    by_family = {}
    for row in static_rows:
        by_family.setdefault(row["family_id"], {})[row["public_mode"]] = row
    policies = []
    for family_id in sorted(by_family):
        modes = by_family[family_id]
        choices = {}
        for mode in ("input_stationary", "weight_resident_barrier"):
            row = modes[mode]
            choices[mode] = {
                "candidate_id": row["candidate_id"],
                "lowering_status": row["status"],
                "load_dma_bytes": row.get("dma_summary", {}).get("load_bytes"),
                "load_dma_calls": row.get("dma_summary", {}).get("load_calls"),
                "explicit_residency_drains": row.get("sync", {}).get(
                    "residency_drains"
                ),
            }
        policies.append(
            {
                "schema": "c3_minimum_access_policy_v2",
                "family_id": family_id,
                "choices": choices,
                "cost_model": "C(mode, lambda_sync) = load_dma_bytes + lambda_sync * explicit_residency_drains",
                "lambda_sync_units": "bytes-equivalent per explicit drain",
                "lambda_sync_value": None,
                "selection": "deferred_until_lambda_is_preregistered_on_development_data",
                "is_schedule_candidate": False,
                "claim": "policy chooses one real mechanism; it is not simultaneous input/weight residency",
            }
        )
    return policies


def report(static_rows, fsim_rows, legacy_rows, policies):
    by_family = {}
    for row in static_rows:
        by_family.setdefault(row["family_id"], {})[row["public_mode"]] = row
    comparisons = []
    for family_id in sorted(by_family):
        modes = by_family[family_id]
        control = modes["original"]
        item = {"family_id": family_id, "knobs": control.get("knobs"), "modes": {}}
        for mode in MODES:
            row = modes[mode]
            dma = row.get("dma_summary")
            delta = None
            if dma and control.get("dma_summary"):
                delta = {
                    key: dma[key] - control["dma_summary"][key]
                    for key in sorted(dma)
                }
            item["modes"][mode] = {
                "status": row["status"],
                "dma": dma,
                "same_tile_delta_from_original": delta,
                "sync": row.get("sync"),
                "dependency_audit": row.get("dependency_audit"),
            }
        comparisons.append(item)
    return {
        "schema": SCHEMA,
        "status": "completed_local_pilot_with_recorded_outcomes",
        "scope": "Y02 local lowering/FSim only; no board latency or FPS",
        "formal_public_modes_v2": MODES,
        "historical_supplemental_ablations": LEGACY_ABLATIONS,
        "mode4_claim_name": "weight_resident_barrier",
        "mode4_claim_boundary": (
            "explicit full-drain schedule probe; not barrier-free reuse, hardware prefetch, "
            "or an exact reproduction of another paper"
        ),
        "static": {
            "attempted": len(static_rows),
            "ok": sum(row["status"] == "ok" for row in static_rows),
            "failed": sum(row["status"] != "ok" for row in static_rows),
        },
        "fsim": {
            "attempted": len(fsim_rows),
            "passed": sum(row["status"] == "passed" for row in fsim_rows),
            "failed": sum(row["status"] != "passed" for row in fsim_rows),
            "seeds": list(SEEDS),
            "performance_measurement": "not_collected",
        },
        "same_tile_comparisons": comparisons,
        "prior_mode4_correctness_evidence": {
            row["workload_id"]: {
                "config_index": row["config_index"],
                "original_weight_bytes": row["original"]["weight_bytes"],
                "mode4_weight_bytes": row["mode4"]["weight_bytes"],
                "mode4_weight_calls": row["mode4"]["weight_calls"],
                "sync_static_execution_multiplicity": row["mode4"][
                    "sync_static_execution_multiplicity"
                ],
                "dependency_balanced": row["dependency_balanced"],
                "forbidden_1_3_count": row["forbidden_1_3_count"],
                "three_seed_correct": all(seed["correct"] for seed in row["frozen_seeds"]),
            }
            for row in legacy_rows
        },
        "minimum_access_strategy": {
            "schema": "c3_minimum_access_policy_v2",
            "family_count": len(policies),
            "description": (
                "choose between input_stationary and weight_resident_barrier after a "
                "development-only synchronization penalty is preregistered"
            ),
            "simultaneous_hybrid_claim": False,
        },
        "tophub_used": False,
        "board_contacted": False,
        "sufficient_for_final_ccf_b_claim": False,
    }


def run_main(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    inputs = {
        "p7r115_candidates": verify_frozen(args.candidates),
        "p7r115_domain": verify_frozen(args.complete_domain),
        "p7r118_legality": verify_frozen(args.legality),
        "p3e_mode4_evidence": verify_frozen(args.p3e_results),
    }
    for path in (args.candidates, args.complete_domain, args.legality):
        if b"tophub" in Path(path).read_bytes().lower():
            raise ValueError("unexpected TopHub material in selection input {}".format(path))
    candidates = build_candidates(
        load_jsonl(args.candidates),
        load_jsonl(args.complete_domain),
        load_jsonl(args.legality),
    )
    output.mkdir(parents=True)
    preregistered = {
        "schema": "c3_p7r119_yolo_barrier_pilot_protocol_v2",
        "status": "frozen_before_static_or_fsim",
        "inputs": inputs,
        "candidate_identity_schema": IDENTITY_SCHEMA,
        "public_modes": MODES,
        "legacy_ablation_modes_excluded_from_formal_pool": LEGACY_ABLATIONS,
        "candidate_count": len(candidates),
        "candidate_order": [row["candidate_id"] for row in candidates],
        "selection_rule": candidates[0]["selection_basis"],
        "gross_budget_policy": "all nine candidates are charged; no replacement",
        "correctness_policy": "three FSim seeds before any future board timing",
        "fsim_timeout_seconds_per_candidate": args.fsim_timeout_seconds,
        "forbidden_sources": ["TopHub entity/index/cost", "board timing", "FSim wall time"],
        "board_contacted": False,
    }
    write_json(output / "preregistered.json", preregistered)
    (output / "candidates_v2.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates),
        encoding="utf-8",
    )
    write_json(
        output / "pre_run_hashes.json",
        {
            "artifacts": {
                "preregistered.json": sha256_file(output / "preregistered.json"),
                "candidates_v2.jsonl": sha256_file(output / "candidates_v2.jsonl"),
            }
        },
    )

    static_rows = []
    with (output / "static_results.jsonl").open("x", encoding="utf-8") as stream:
        for candidate in candidates:
            row = static_result(candidate)
            static_rows.append(row)
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            print(
                "static {} {} {}".format(
                    candidate["family_id"], candidate["public_mode"], row["status"]
                ),
                flush=True,
            )

    policies = minimum_access_policy(static_rows)
    (output / "minimum_access_policy_v2.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in policies),
        encoding="utf-8",
    )
    write_json(
        output / "pre_fsim_hashes.json",
        {
            "artifacts": {
                "static_results.jsonl": sha256_file(output / "static_results.jsonl"),
                "minimum_access_policy_v2.jsonl": sha256_file(
                    output / "minimum_access_policy_v2.jsonl"
                ),
            }
        },
    )

    fsim_rows = []
    stderr_path = output / "stderr.log"
    stderr_path.touch(exist_ok=False)
    static_by_id = {row["candidate_id"]: row for row in static_rows}
    with (output / "fsim_results.jsonl").open("x", encoding="utf-8") as stream:
        if not args.skip_fsim:
            for candidate in candidates:
                static = static_by_id[candidate["candidate_id"]]
                if static["status"] != "ok":
                    continue
                expected_drains = static["sync"]["residency_drains"]
                row, stderr = run_fsim(
                    candidate,
                    static["tir_sha256"],
                    expected_drains,
                    args.fsim_timeout_seconds,
                    args.fsim_hw_path,
                )
                fsim_rows.append(row)
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                safe_stderr = "\n".join(
                    "[VTA_QUEUE] <parsed; timing discarded>"
                    if "[VTA_QUEUE] " in line
                    else line
                    for line in stderr.splitlines()
                )
                if safe_stderr.strip():
                    with stderr_path.open("a", encoding="utf-8") as errstream:
                        errstream.write(
                            "[{} {}]\n{}\n".format(
                                candidate["family_id"], candidate["public_mode"], safe_stderr[-4000:]
                            )
                        )
                print(
                    "fsim {} {} {}".format(
                        candidate["family_id"], candidate["public_mode"], row["status"]
                    ),
                    flush=True,
                )

    legacy_rows = load_jsonl(args.p3e_results)
    final_report = report(static_rows, fsim_rows, legacy_rows, policies)
    write_json(output / "report.json", final_report)
    (output / "STATUS.md").write_text(
        "# P7R119 Y02 explicit-barrier pilot\n\n"
        "- Status: `completed_local_pilot_with_recorded_outcomes`\n"
        "- Formal schema-v2 modes: original, input_stationary, weight_resident_barrier\n"
        "- Static lowering: {}/{} successful\n"
        "- Three-seed local FSim: {}/{} passed\n"
        "- Minimum-access: parameterized policy over two real mechanisms; not a hybrid schedule\n"
        "- TopHub used: no\n"
        "- Board contacted: no\n"
        "- Claim limit: pilot only; no latency/FPS or exact-paper-reproduction claim\n".format(
            final_report["static"]["ok"],
            final_report["static"]["attempted"],
            final_report["fsim"]["passed"],
            final_report["fsim"]["attempted"],
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
    print(json.dumps(final_report, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--complete-domain", type=Path, default=DEFAULT_DOMAIN)
    parser.add_argument("--legality", type=Path, default=DEFAULT_LEGALITY)
    parser.add_argument("--p3e-results", type=Path, default=DEFAULT_P3E)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fsim-hw-path", type=Path, default=Path("/tmp/vta-hw-fsim"))
    parser.add_argument("--fsim-timeout-seconds", type=int, default=300)
    parser.add_argument("--skip-fsim", action="store_true")
    parser.add_argument("--worker-candidate", type=Path)
    parser.add_argument("--worker-expected-tir")
    parser.add_argument("--worker-output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker_candidate:
        if not args.worker_expected_tir or not args.worker_output:
            raise ValueError("worker requires expected TIR and output")
        result = fsim_worker(load_json(args.worker_candidate), args.worker_expected_tir)
        write_json(args.worker_output, result)
    else:
        if args.fsim_timeout_seconds <= 0:
            raise ValueError("timeout must be positive")
        run_main(args)


if __name__ == "__main__":
    main()
