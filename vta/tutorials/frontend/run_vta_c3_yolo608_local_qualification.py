#!/usr/bin/env python3
"""Locally qualify the frozen, latency-blind YOLO-608 C3 candidate pools.

This program never contacts the FPGA board and never reads latency.  It binds
each frozen candidate identity to real lowered TIR, proves the statically
enumerable SRAM/DDR/UOP address ranges, validates three deterministic seeds in
FSim, captures a structural command signature, and removes only programs whose
lowered TIR *and* command signatures are identical.
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

import tvm
import vta

from c3_candidate_identity import canonical_json_bytes
from c3_reuse_semantics import capture_lower, extract_requests
from c3_thread_sync_audit import final_tir_audit
import run_vta_c3_resnet18_local_qualification as legacy
from run_vta_c3_real_task_reference_local import fsim_worker_reused_buffers


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_INPUT = GROUPED / "20260924_c3_yolo608_candidate_pools_run01"
DEFAULT_OUTPUT = GROUPED / "20260924_c3_yolo608_local_qualification_run01"
SCHEMA = "c3_yolo608_local_qualification_v1"
MIN_DISTINCT_PROGRAMS = 8
DEFAULT_EXPECTED_MODES = frozenset(legacy.MODES)
ARENA_CAPACITY_KEYS = {
    "local.inp_buffer": "inp_buff_size",
    "local.wgt_buffer": "wgt_buff_size",
    "local.acc_buffer": "acc_buff_size",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def reject_latency_labels(value, location="$"):
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = key.lower()
            if ("latency" in lowered or "oracle" in lowered or lowered == "performance_label") and item is not None:
                raise ValueError("forbidden performance label at {}.{}".format(location, key))
            reject_latency_labels(item, "{}.{}".format(location, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_latency_labels(item, "{}[{}]".format(location, index))


def verify_input(directory):
    directory = Path(directory).resolve()
    ledger_path = directory / "artifact_hashes.json"
    ledger = read_json(ledger_path)
    checked = {}
    for name, expected in sorted(ledger["artifacts"].items()):
        path = directory / name
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError("input artifact hash mismatch: {}".format(name))
        checked[name] = observed
    contract = read_json(directory / "contract.json")
    if contract.get("board_contacted") is not False or contract.get("performance_labels_read") is not False:
        raise ValueError("input candidate freeze is not latency blind")
    pool_files = contract.get("pool_files", [])
    if len(pool_files) != contract.get("execution_task_count") or not pool_files:
        raise ValueError("frozen pool manifest is incomplete")
    pools = []
    candidate_ids = []
    declared_modes = contract.get("candidate_modes")
    expected_modes = (
        frozenset(str(value) for value in declared_modes)
        if declared_modes is not None
        else DEFAULT_EXPECTED_MODES
    )
    supported_modes = legacy.WEIGHT_MODES | {"original", "input_stationary"}
    if not expected_modes or not expected_modes.issubset(supported_modes):
        raise ValueError("unsupported or empty declared candidate mode set")
    for name in pool_files:
        pool = read_json(directory / name)
        reject_latency_labels(pool)
        if pool.get("board_contacted") is not False or pool.get("performance_labels_collected") is not False:
            raise ValueError("pool is not label isolated: {}".format(name))
        families = defaultdict(set)
        for candidate in pool["candidates"]:
            candidate_ids.append(candidate["candidate_id"])
            families[candidate["family_id"]].add(candidate["residence_mode"])
        pool_modes = frozenset(str(value) for value in pool.get("candidate_modes", expected_modes))
        if pool_modes != expected_modes:
            raise ValueError("pool and contract candidate mode sets differ: {}".format(name))
        if any(modes != expected_modes for modes in families.values()):
            raise ValueError("pool lacks an exact declared-mode family: {}".format(name))
        pools.append((directory / name, pool))
    if len(candidate_ids) != contract.get("candidate_count") or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate count or global uniqueness mismatch")
    checked["artifact_hashes.json"] = sha256_file(ledger_path)
    return checked, pools


def select_pools(pools, workload_ids):
    requested = [str(value) for value in workload_ids or []]
    if not requested:
        return list(pools)
    if len(requested) != len(set(requested)):
        raise ValueError("workload allowlist contains duplicates")
    by_id = {pool["workload_id"]: (path, pool) for path, pool in pools}
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise ValueError("unknown workload IDs: {}".format(missing))
    return [by_id[workload_id] for workload_id in requested]


def arena_certificate(arenas, fingerprint):
    spans = arenas["scope_arena_span_sum_upper_bound_bytes"]
    domains = {}
    for scope, capacity_key in ARENA_CAPACITY_KEYS.items():
        required = int(spans.get(scope, 0))
        capacity = int(fingerprint[capacity_key])
        domains[scope] = {
            "required_upper_bound_bytes": required,
            "capacity_bytes": capacity,
            "headroom_bytes": capacity - required,
            "passed": required <= capacity,
        }
    unexpected = sorted(scope for scope in spans if scope.startswith("local.") and scope not in ARENA_CAPACITY_KEYS)
    passed = all(row["passed"] for row in domains.values()) and not unexpected
    return {
        "status": "passed" if passed else "failed",
        "domains": domains,
        "unexpected_local_scopes": unexpected,
        "measure": "sum of StorageRewrite arena spans; conservative upper bound, not temporal occupancy",
        "effective_peak_occupancy": None,
    }


def static_result(candidate, family_rows, tir_dir):
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
    if candidate["residence_mode"] in legacy.WEIGHT_MODES and not candidate["applicability"]["full_weight_fits_sram"]:
        base.update(
            status="not_applicable",
            applicability=candidate["applicability"],
            fallback={
                "policy": "same-tile original",
                "candidate_id": legacy.fallback_candidate_id(candidate, family_rows),
                "reason": "the frozen full-residency implementation requires the full layer weights to fit",
            },
        )
        base["wall_seconds"] = time.monotonic() - start
        return base
    phase = "instantiate"
    try:
        env = vta.get_env()
        schedule, tensors = legacy.instantiate(candidate, env)
        phase = "storage_rewrite_and_lower"
        module, storage_module, arenas = capture_lower(schedule, tensors)
        tir_hash = hashlib.sha256(tvm.ir.save_json(module).encode("utf-8")).hexdigest()
        storage_hash = hashlib.sha256(tvm.ir.save_json(storage_module).encode("utf-8")).hexdigest()
        phase = "dependency"
        dependencies = legacy.dependency_audit(module)
        if not dependencies["balanced"] or dependencies["forbidden_direct_1_3_count"]:
            raise ValueError("dependency audit rejected lowered program")
        phase = "request_semantics"
        semantics = extract_requests(module, env, max_expansions=2_000_000)
        phase = "final_address_audit"
        address_audit = final_tir_audit(module, semantics)
        phase = "arena_capacity"
        arena = arena_certificate(arenas, candidate["hardware_fingerprint"])
        if arena["status"] != "passed":
            raise ValueError("StorageRewrite arena capacity certificate failed")
        sync = legacy.sync_multiplicity(module)
        tir_path = tir_dir / (candidate["candidate_id"] + ".json")
        storage_path = tir_dir / (candidate["candidate_id"] + ".storage.json")
        tir_path.write_text(tvm.ir.save_json(module), encoding="utf-8")
        storage_path.write_text(tvm.ir.save_json(storage_module), encoding="utf-8")
        identity = {
            "proposal_candidate_id": candidate["candidate_id"],
            "model_hash": candidate["model_hash"],
            "workload_id": candidate["workload_id"],
            "lowered_tir_sha256": tir_hash,
            "hardware_fingerprint": candidate["hardware_fingerprint"],
        }
        base.update(
            status="ok",
            lowered_tir_sha256=tir_hash,
            storage_tir_sha256=storage_hash,
            lowered_tir_file=str(tir_path.name),
            storage_tir_file=str(storage_path.name),
            implementation_candidate_id=canonical_sha256(identity),
            implementation_identity=identity,
            storage_arenas=arenas,
            arena_capacity_certificate=arena,
            request_semantics=semantics,
            final_address_audit=address_audit,
            dependency_audit=dependencies,
            dma_features={
                "totals": semantics["totals"],
                "unique_source_region_bytes": semantics["unique_source_region_bytes"],
                "reload_ratios": semantics["reload_ratios"],
                "scope": semantics["scope"],
            },
            sync={
                "static_execution_multiplicity": sync,
                "explicit_residency_drains": int(sum(sync[:-1])) if len(sync) > 1 else 0,
                "function_final_drain": int(sync[-1]) if sync else 0,
            },
        )
    except Exception as error:  # TVM intentionally exposes heterogeneous errors.
        base["failure"] = {
            "category": "static_qualification",
            "phase": phase,
            "exception_type": type(error).__name__,
            "message": (str(error) or type(error).__name__)[:4000],
            "retryable": False,
        }
    base["wall_seconds"] = time.monotonic() - start
    return base


def run_fsim_subprocess(candidate, static, args):
    with tempfile.TemporaryDirectory(prefix="c3_y608_fsim_") as directory:
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
                "worker exit={} output_present={}".format(completed.returncode, response.is_file()),
                phase="worker_process",
            )
            result["wall_seconds"] = None
        else:
            result = read_json(response)
        return legacy.attach_command_features(result, completed.stderr, static), completed.stderr


def distinct_programs(candidates, static_rows, fsim_rows):
    candidates_by_id = {row["candidate_id"]: row for row in candidates}
    static_by_id = {row["candidate_id"]: row for row in static_rows}
    groups = defaultdict(list)
    for fsim in fsim_rows:
        values = fsim.get("command_features", {}).get("values")
        if fsim["status"] != "passed" or values is None:
            continue
        static = static_by_id[fsim["candidate_id"]]
        command_hash = canonical_sha256(values)
        groups[(fsim["workload_id"], static["lowered_tir_sha256"], command_hash)].append(fsim["candidate_id"])
    rows = []
    for ordinal, (key, ids) in enumerate(sorted(groups.items()), 1):
        workload_id, tir_hash, command_hash = key
        representative = sorted(ids)[0]
        rows.append({
            "schema": SCHEMA,
            "program_id": canonical_sha256({"workload_id": workload_id, "tir": tir_hash, "commands": command_hash}),
            "workload_id": workload_id,
            "ordinal": ordinal,
            "lowered_tir_sha256": tir_hash,
            "command_signature_sha256": command_hash,
            "representative_candidate_id": representative,
            "equivalent_candidate_ids": sorted(ids),
            "representative_identity": candidates_by_id[representative]["candidate_identity_fields"],
        })
    return rows


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    input_hashes, pools = verify_input(args.input_dir)
    pools = select_pools(pools, args.workload_id)
    output.mkdir(parents=True)
    tir_dir = output / "tir"
    tir_dir.mkdir()
    paths = {name: output / name for name in (
        "static_results.jsonl", "fsim_results.jsonl", "results.jsonl",
        "timeline.jsonl", "qualified_programs.jsonl",
    )}
    for path in paths.values():
        path.write_text("", encoding="utf-8")

    candidates, families = [], defaultdict(list)
    pool_by_workload = {}
    for _, pool in pools:
        pool_by_workload[pool["workload_id"]] = pool
        for raw in pool["candidates"]:
            candidate = legacy.candidate_with_context(raw, pool)
            candidates.append(candidate)
            families[candidate["family_id"]].append(candidate)
    if args.limit is not None:
        candidates = candidates[:args.limit]
    contract = {
        "schema": SCHEMA,
        "status": "frozen_before_local_qualification",
        "input_dir": str(Path(args.input_dir).resolve()),
        "input_artifact_hashes": input_hashes,
        "candidate_order": [row["candidate_id"] for row in candidates],
        "workload_allowlist": [pool["workload_id"] for _, pool in pools],
        "workload_selection_rule": args.workload_selection_rule,
        "seeds": list(legacy.SEEDS),
        "reference_eligibility": {
            "minimum_distinct_fsim_correct_programs_per_task": MIN_DISTINCT_PROGRAMS,
            "maximum_programs_per_task": 15,
            "frozen_incumbent_must_pass": True,
        },
        "hard_qualification": [
            "lowering", "dependency tokens", "StorageRewrite arena capacities",
            "DDR DMA bounds", "SRAM DMA bounds", "UOP/GEMM/ALU address bounds",
            "three-seed FSim exact equality", "three-seed command-signature consistency",
        ],
        "label_boundary": {
            "board_contact_forbidden": True,
            "latency_read_forbidden": True,
            "pool_oracle_forbidden": True,
        },
        "failure_accounting": "retain every frozen identity; invalid attempts are not replaced",
        "board_contacted": False,
    }
    write_json(output / "contract.json", contract)

    start = time.monotonic()
    static_rows = []
    for position, candidate in enumerate(candidates, 1):
        row = static_result(candidate, families[candidate["family_id"]], tir_dir)
        row["gross_position"] = position
        static_rows.append(row)
        append_jsonl(paths["static_results.jsonl"], row)
        append_jsonl(paths["results.jsonl"], row)
        append_jsonl(paths["timeline.jsonl"], {
            "event": "static_complete", "position": position,
            "candidate_id": candidate["candidate_id"], "status": row["status"],
            "elapsed_seconds": time.monotonic() - start,
        })
        print("static {}/{} {} {}".format(position, len(candidates), candidate["workload_id"], row["status"]), flush=True)

    candidate_by_id = {row["candidate_id"]: row for row in candidates}
    eligible = [row for row in static_rows if row["status"] == "ok"]
    fsim_rows = []
    if args.phase == "all":
        for position, static in enumerate(eligible, 1):
            candidate = candidate_by_id[static["candidate_id"]]
            row, stderr = run_fsim_subprocess(candidate, static, args)
            row["fsim_position"] = position
            fsim_rows.append(row)
            append_jsonl(paths["fsim_results.jsonl"], row)
            append_jsonl(paths["results.jsonl"], row)
            append_jsonl(paths["timeline.jsonl"], {
                "event": "fsim_complete", "position": position,
                "candidate_id": candidate["candidate_id"], "status": row["status"],
                "elapsed_seconds": time.monotonic() - start,
            })
            if stderr.strip() and row["status"] != "passed":
                append_jsonl(output / "worker_failures.jsonl", {
                    "candidate_id": candidate["candidate_id"], "stderr_tail": stderr[-4000:],
                })
            print("fsim {}/{} {} {}".format(position, len(eligible), candidate["workload_id"], row["status"]), flush=True)

    programs = distinct_programs(candidates, static_rows, fsim_rows)
    for row in programs:
        append_jsonl(paths["qualified_programs.jsonl"], row)

    workloads = {}
    for workload_id, pool in sorted(pool_by_workload.items()):
        srows = [row for row in static_rows if row["workload_id"] == workload_id]
        frows = [row for row in fsim_rows if row["workload_id"] == workload_id]
        prows = [row for row in programs if row["workload_id"] == workload_id]
        incumbent_id = pool["incumbent"]["candidate_id"]
        incumbent_fsim = next((row for row in frows if row["candidate_id"] == incumbent_id), None)
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
            "reference_eligible": (
                len(prows) >= MIN_DISTINCT_PROGRAMS
                and len(prows) <= 15
                and incumbent_fsim is not None
                and incumbent_fsim["status"] == "passed"
            ),
        }
    summary = {
        "schema": SCHEMA,
        "status": (
            "yolo_resolution_eligible" if sum(row["reference_eligible"] for row in workloads.values()) >= 2
            else "insufficient_tasks_with_eight_distinct_programs"
        ),
        "workloads": workloads,
        "reference_eligible_workload_count": sum(row["reference_eligible"] for row in workloads.values()),
        "proposed_identities": len(static_rows),
        "static_ok": sum(row["status"] == "ok" for row in static_rows),
        "not_applicable": sum(row["status"] == "not_applicable" for row in static_rows),
        "static_failed": sum(row["status"] == "failed" for row in static_rows),
        "fsim_attempted": len(fsim_rows),
        "fsim_passed": sum(row["status"] == "passed" for row in fsim_rows),
        "distinct_fsim_correct_programs": len(programs),
        "wall_seconds": time.monotonic() - start,
        "board_contacted": False,
        "latency_values_read": False,
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(
        " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
        encoding="utf-8",
    )
    artifacts = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    write_json(output / "artifact_hashes.json", {
        "artifacts": artifacts,
        "source_hashes": {
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
            str(Path(legacy.__file__).resolve()): sha256_file(Path(legacy.__file__).resolve()),
        },
    })
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--fsim-hw-path", default="/tmp/vta-hw-fsim")
    parser.add_argument("--fsim-timeout", type=int, default=300)
    parser.add_argument("--phase", choices=("static", "all"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workload-id", action="append", default=[])
    parser.add_argument(
        "--workload-selection-rule",
        default="all frozen workloads",
        help="Label-blind rule frozen by the caller for the workload allowlist.",
    )
    parser.add_argument("--worker-request")
    parser.add_argument("--worker-output")
    args = parser.parse_args()
    if bool(args.worker_request) != bool(args.worker_output):
        parser.error("worker request/output must be provided together")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker_request:
        request = read_json(parsed.worker_request)
        write_json(parsed.worker_output, fsim_worker_reused_buffers(request["candidate"], request["static"]))
    else:
        run(parsed)
