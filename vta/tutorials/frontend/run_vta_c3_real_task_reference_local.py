#!/usr/bin/env python3
"""Locally qualify the frozen C3 real-task reference pools.

The input pools were frozen before current-board latency was observed.  This
runner performs real lowering, three-seed FSim correctness, command-signature
capture, exact-program deduplication, and a genuine-source-TopHub versus mode-0
adapter TIR check.  It never contacts the FPGA board.
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
from collections import defaultdict
from pathlib import Path

import numpy as np
import tvm
from tvm import autotvm
from tvm.autotvm.task.space import ConfigEntity
import vta

from c3_candidate_identity import canonical_json_bytes, normalize_config_entity
from qualify_vta_residency_fsim import array_sha256
from run_vta_p7r132_y00_search_confirmation import (
    allocate_reusable_buffers,
    prepare_reused_buffers,
)
from tune_resnet18_vta import register_vta_conv2d_template
import run_vta_c3_resnet18_local_qualification as legacy


HERE = Path(__file__).resolve().parent
C3 = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune"
GROUPED = C3 / "07_grouped_holdout"
DEFAULT_INPUT = GROUPED / "20260924_c3_real_task_reference_prepare_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_real_task_reference_local_run01"
SCHEMA = "c3_real_task_reference_local_v1"
MIN_DISTINCT_PROGRAMS = 8


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def verify_input(directory):
    directory = Path(directory).resolve()
    ledger_path = directory / "artifact_hashes.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    checked = {}
    for name, expected in sorted(ledger["artifacts"].items()):
        path = directory / name
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError("input artifact hash mismatch: {}".format(name))
        checked[name] = observed
    contract = json.loads((directory / "contract.json").read_text(encoding="utf-8"))
    if contract.get("current_board_performance_labels_read") is not False:
        raise ValueError("input preparation was not current-board-label blind")
    if contract.get("external_tophub_source_cost_read_for_incumbent_only") is not True:
        raise ValueError("input does not disclose external TopHub incumbent binding")
    checked["artifact_hashes.json"] = sha256_file(ledger_path)
    input_hashes, pools = legacy.verify_p7r470(directory)
    checked.update(input_hashes)
    return checked, pools


def genuine_tophub_tir(pool):
    register_vta_conv2d_template()
    env = vta.get_env()
    task = autotvm.task.create(
        "conv2d_packed.vta",
        args=tuple(pool["workload"][1:]),
        target=env.target,
        target_host=env.target_host,
    )
    source = pool["incumbent"]["source_complete_config_entity"]
    config = ConfigEntity.from_json_dict(source)
    if normalize_config_entity(config.to_json_dict()) != normalize_config_entity(source):
        raise RuntimeError("genuine TopHub ConfigEntity changed during reconstruction")
    with task.target:
        schedule, tensors = task.instantiate(config)
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        module = tvm.lower(schedule, tensors, name="main")
    return hashlib.sha256(tvm.ir.save_json(module).encode("utf-8")).hexdigest()


def source_adapter_equivalence(pools, static_rows):
    by_id = {row["candidate_id"]: row for row in static_rows}
    results = []
    for _, pool in pools:
        candidate_id = pool["incumbent"]["candidate_id"]
        adapter = by_id[candidate_id]
        row = {
            "workload_id": pool["workload_id"],
            "source_candidate_id": candidate_id,
            "source_template": "conv2d_packed.vta",
            "adapter_template": "conv2d_packed_residency.vta(mode=0)",
            "source_tir_sha256": None,
            "adapter_tir_sha256": adapter.get("lowered_tir_sha256"),
            "status": "failed",
        }
        if adapter["status"] != "ok":
            row["reason"] = "mode0_adapter_did_not_lower"
        else:
            try:
                row["source_tir_sha256"] = genuine_tophub_tir(pool)
                row["status"] = (
                    "equivalent"
                    if row["source_tir_sha256"] == row["adapter_tir_sha256"]
                    else "not_equivalent"
                )
                if row["status"] != "equivalent":
                    row["reason"] = "lowered_tir_hash_differs"
            except Exception as error:  # TVM deliberately exposes heterogeneous errors.
                row["reason"] = "{}: {}".format(type(error).__name__, str(error)[:2000])
        results.append(row)
    return results


def fsim_worker_reused_buffers(candidate, static):
    """Build once and validate with stable handles after an untimed warm-up."""
    start = time.monotonic()
    result = legacy.base_fsim(candidate, static)
    phase = "preflight"
    try:
        from vta.testing import simulator  # pylint: disable=import-outside-toplevel

        env = vta.get_env()
        if env.TARGET != "sim" or not simulator.enabled():
            raise RuntimeError("worker requires VTA TARGET=sim and enabled FSim runtime")
        schedule, tensors = legacy.instantiate(candidate, env)
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
        workload = candidate["identity"]["workload"]
        buffers = allocate_reusable_buffers(tvm.device("ext_dev", 0), workload)
        for seed in legacy.SEEDS:
            phase = "warmup_seed_{}".format(seed)
            prepare_reused_buffers(buffers, workload, seed)
            function(buffers["data"], buffers["weight"], buffers["output"])
        for seed in legacy.SEEDS:
            phase = "execute_seed_{}".format(seed)
            expected = prepare_reused_buffers(buffers, workload, seed)
            print("[C3_R18_SEED] {}".format(seed), file=sys.stderr, flush=True)
            function(buffers["data"], buffers["weight"], buffers["output"])
            actual = buffers["output"].numpy()
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
            result["failure"] = legacy.make_failure(
                "wrong_answer", "one or more FSim seeds failed", phase="execute"
            )
        result["buffer_protocol"] = {
            "allocation_order": buffers["allocation_order"],
            "handles_reused_across_seeds": True,
            "warmup_once_per_seed_before_diagnostics": True,
            "warmup_is_not_performance_measurement": True,
        }
    except Exception as error:  # TVM deliberately exposes heterogeneous errors.
        category = (
            "wrong_answer"
            if phase.startswith("execute_seed_") or phase.startswith("warmup_seed_")
            else ("compile" if phase == "build" else "environment")
        )
        failure = legacy.make_failure(
            category, (str(error) or type(error).__name__)[:4000], phase=phase
        )
        failure["exception_type"] = type(error).__name__
        result["failure"] = failure
    result["wall_seconds"] = time.monotonic() - start
    return result


def run_fsim_subprocess(candidate, static, args):
    with tempfile.TemporaryDirectory(prefix="c3_real_task_fsim_") as directory:
        request = Path(directory) / "request.json"
        response = Path(directory) / "response.json"
        write_json(request, {"candidate": candidate, "static": static})
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker-request",
            str(request),
            "--worker-output",
            str(response),
        ]
        environment = os.environ.copy()
        environment["VTA_HW_PATH"] = str(Path(args.fsim_hw_path).resolve())
        environment["VTA_QUEUE_DIAGNOSTICS"] = "1"
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=args.fsim_timeout,
                env=environment,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            result = legacy.base_fsim(candidate, static)
            result["failure"] = legacy.make_failure(
                "timeout",
                "FSim exceeded {} seconds".format(args.fsim_timeout),
                phase="worker",
                retryable=True,
            )
            result["wall_seconds"] = float(args.fsim_timeout)
            return result, error.stderr or ""
        if completed.returncode != 0 or not response.is_file():
            result = legacy.base_fsim(candidate, static)
            result["failure"] = legacy.make_failure(
                "environment",
                "worker exit={} output_present={}".format(
                    completed.returncode, response.is_file()
                ),
                phase="worker_process",
            )
            result["wall_seconds"] = None
        else:
            result = json.loads(response.read_text(encoding="utf-8"))
        return legacy.attach_command_features(result, completed.stderr, static), completed.stderr


def distinct_programs(candidates, static_rows, fsim_rows):
    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    static_by_id = {row["candidate_id"]: row for row in static_rows}
    groups = defaultdict(list)
    for fsim in fsim_rows:
        if fsim["status"] != "passed":
            continue
        values = fsim.get("command_features", {}).get("values")
        if values is None:
            continue
        static = static_by_id[fsim["candidate_id"]]
        key = (static["lowered_tir_sha256"], canonical_sha256(values))
        groups[(fsim["workload_id"],) + key].append(fsim["candidate_id"])
    records = []
    for ordinal, (key, ids) in enumerate(sorted(groups.items()), 1):
        workload_id, tir_hash, command_hash = key
        representative = sorted(ids)[0]
        records.append(
            {
                "schema": SCHEMA,
                "program_id": canonical_sha256(
                    {"workload_id": workload_id, "tir": tir_hash, "commands": command_hash}
                ),
                "workload_id": workload_id,
                "ordinal": ordinal,
                "lowered_tir_sha256": tir_hash,
                "command_signature_sha256": command_hash,
                "representative_candidate_id": representative,
                "equivalent_candidate_ids": sorted(ids),
                "representative_identity": candidate_by_id[representative]["candidate_identity_fields"],
            }
        )
    return records


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    input_hashes, pools = verify_input(args.input_dir)
    output.mkdir(parents=True)
    paths = {
        name: output / name
        for name in (
            "static_results.jsonl",
            "fsim_results.jsonl",
            "results.jsonl",
            "timeline.jsonl",
            "qualified_programs.jsonl",
        )
    }
    for path in paths.values():
        path.write_text("", encoding="utf-8")

    candidates, families = [], defaultdict(list)
    for _, pool in pools:
        for raw in pool["candidates"]:
            candidate = legacy.candidate_with_context(raw, pool)
            candidates.append(candidate)
            families[candidate["family_id"]].append(candidate)

    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_local_qualification",
        "input_dir": str(Path(args.input_dir).resolve()),
        "input_artifact_hashes": input_hashes,
        "candidate_order": [row["candidate_id"] for row in candidates],
        "seeds": list(legacy.SEEDS),
        "reference_eligibility": {
            "minimum_distinct_fsim_correct_programs_per_task": MIN_DISTINCT_PROGRAMS,
            "maximum_programs_per_task": 15,
            "incumbent_mode0_must_pass": True,
            "genuine_tophub_and_mode0_adapter_tir_must_match": True,
        },
        "label_boundary": {
            "current_board_performance_forbidden": True,
            "future_correctness_forbidden_at_prepare_time_but_collected_here": True,
            "pool_oracle_forbidden": True,
            "external_tophub_source_cost_read_for_incumbent_binding_only": True,
        },
        "failure_accounting": "retain every proposed identity; no replacement or performance-driven expansion",
        "board_contacted": False,
    }
    write_json(output / "contract.json", contract)

    start = time.monotonic()
    static_rows = []
    for position, candidate in enumerate(candidates, 1):
        row = legacy.static_result(candidate, families[candidate["family_id"]])
        row["gross_position"] = position
        static_rows.append(row)
        append_jsonl(paths["static_results.jsonl"], row)
        append_jsonl(paths["results.jsonl"], row)
        append_jsonl(
            paths["timeline.jsonl"],
            {
                "event": "static_complete",
                "position": position,
                "candidate_id": candidate["candidate_id"],
                "status": row["status"],
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        print("static {}/{} {} {}".format(position, len(candidates), candidate["workload_id"], row["status"]), flush=True)

    equivalence = source_adapter_equivalence(pools, static_rows)
    write_json(output / "source_adapter_equivalence.json", equivalence)

    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    eligible = [row for row in static_rows if row["status"] == "ok"]
    fsim_rows = []
    for position, static in enumerate(eligible, 1):
        candidate = candidate_by_id[static["candidate_id"]]
        row, stderr = run_fsim_subprocess(candidate, static, args)
        row["fsim_position"] = position
        fsim_rows.append(row)
        append_jsonl(paths["fsim_results.jsonl"], row)
        append_jsonl(paths["results.jsonl"], row)
        append_jsonl(
            paths["timeline.jsonl"],
            {
                "event": "fsim_complete",
                "position": position,
                "candidate_id": candidate["candidate_id"],
                "status": row["status"],
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        if stderr.strip() and row["status"] != "passed":
            append_jsonl(
                output / "worker_failures.jsonl",
                {"candidate_id": candidate["candidate_id"], "stderr_tail": stderr[-4000:]},
            )
        print("fsim {}/{} {} {}".format(position, len(eligible), candidate["workload_id"], row["status"]), flush=True)

    programs = distinct_programs(candidates, static_rows, fsim_rows)
    for row in programs:
        append_jsonl(paths["qualified_programs.jsonl"], row)

    workloads = {}
    for _, pool in pools:
        workload_id = pool["workload_id"]
        srows = [row for row in static_rows if row["workload_id"] == workload_id]
        frows = [row for row in fsim_rows if row["workload_id"] == workload_id]
        prows = [row for row in programs if row["workload_id"] == workload_id]
        eq = next(row for row in equivalence if row["workload_id"] == workload_id)
        incumbent_id = pool["incumbent"]["candidate_id"]
        incumbent_fsim = next((row for row in frows if row["candidate_id"] == incumbent_id), None)
        eligible_reference = (
            len(prows) >= MIN_DISTINCT_PROGRAMS
            and len(prows) <= 15
            and incumbent_fsim is not None
            and incumbent_fsim["status"] == "passed"
            and eq["status"] == "equivalent"
        )
        workloads[workload_id] = {
            "proposed_identities": len(srows),
            "static_ok": sum(row["status"] == "ok" for row in srows),
            "not_applicable": sum(row["status"] == "not_applicable" for row in srows),
            "static_failed": sum(row["status"] == "failed" for row in srows),
            "fsim_passed_identities": sum(row["status"] == "passed" for row in frows),
            "fsim_failed_identities": sum(row["status"] != "passed" for row in frows),
            "distinct_fsim_correct_programs": len(prows),
            "incumbent_candidate_id": incumbent_id,
            "incumbent_fsim_passed": bool(incumbent_fsim and incumbent_fsim["status"] == "passed"),
            "source_adapter_equivalence": eq["status"],
            "reference_eligible": eligible_reference,
        }

    summary = {
        "schema": SCHEMA,
        "status": (
            "ready_for_rule_freeze" if all(row["reference_eligible"] for row in workloads.values())
            else "local_qualification_incomplete_or_insufficient"
        ),
        "workloads": workloads,
        "proposed_identities": len(static_rows),
        "static_ok": sum(row["status"] == "ok" for row in static_rows),
        "not_applicable": sum(row["status"] == "not_applicable" for row in static_rows),
        "static_failed": sum(row["status"] == "failed" for row in static_rows),
        "fsim_attempted": len(fsim_rows),
        "fsim_passed": sum(row["status"] == "passed" for row in fsim_rows),
        "distinct_fsim_correct_programs": len(programs),
        "wall_seconds": time.monotonic() - start,
        "board_contacted": False,
        "current_board_performance_labels_collected": False,
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    artifacts = {
        path.name: sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": artifacts,
            "source_hashes": {
                str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
                str(Path(legacy.__file__).resolve()): sha256_file(Path(legacy.__file__).resolve()),
            },
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--fsim-hw-path", default=str(HERE.parents[2] / "3rdparty/vta-hw"))
    parser.add_argument("--fsim-timeout", type=int, default=300)
    parser.add_argument("--worker-request")
    parser.add_argument("--worker-output")
    args = parser.parse_args()
    if bool(args.worker_request) != bool(args.worker_output):
        parser.error("worker request/output must be provided together")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker_request:
        request = json.loads(Path(parsed.worker_request).read_text(encoding="utf-8"))
        write_json(
            parsed.worker_output,
            fsim_worker_reused_buffers(request["candidate"], request["static"]),
        )
    else:
        run(parsed)
