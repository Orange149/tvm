#!/usr/bin/env python3
"""Prepare/freeze/run/analyze T8 O2, eager full residency, and staged residency."""

from __future__ import annotations

import argparse
import contextlib
import copy
import itertools
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import c3_thread_sync_audit as audit
import c3_vthread_staged_protocol as protocol
import run_vta_c3_reuse_supplement as s
import run_vta_c3_thread_sync_ablation as diagnostic_common


HERE, ROOT = s.HERE, s.ROOT
BASE = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune"
PARENT = BASE / "07_grouped_holdout/20260922_c3_thread_sync_prepare_run07"
SCHEMA = "c3_vthread_staged_residency_v1"
SCOPE = "R18-H1 T8: O2 versus eager/staged two-vthread full WGT residency"
SDK = "/home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux"


def instantiate(row):
    """Instantiate the frozen complete ConfigEntity, never a historical index."""
    import vta
    from tvm import autotvm
    from tvm.autotvm.task.space import ConfigEntity
    import vta.top.vta_conv2d_residency  # noqa: F401
    from tune_resnet18_vta import register_vta_conv2d_template

    env = vta.get_env()
    workload = row["identity"]["workload"]
    if row.get("source_tophub"):
        register_vta_conv2d_template()
        name, args = workload[0], tuple(workload[1:])
    else:
        name = "conv2d_packed_residency.vta"
        args = tuple(workload[1:]) + (row["implementation_mode"],)
    task = autotvm.task.create(
        name, args=args, target=env.target, target_host=env.target_host
    )
    config = ConfigEntity.from_json_dict(
        {"index": -1, "code_hash": None, **row["complete_config_entity"]}
    )
    with task.target:
        return task.instantiate(config)


@contextlib.contextmanager
def adapter():
    old_file, old_instantiate = s.__file__, s.instantiate
    s.__file__, s.instantiate = __file__, instantiate
    previous_ld = os.environ.pop("LD_LIBRARY_PATH", None)
    previous_queue = {key: os.environ.get(key) for key in protocol.QUEUE_ENV}
    os.environ.update(protocol.QUEUE_ENV)
    try:
        yield
    finally:
        s.__file__, s.instantiate = old_file, old_instantiate
        if previous_ld is not None:
            os.environ["LD_LIBRARY_PATH"] = previous_ld
        for key, value in previous_queue.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def bindings():
    result = diagnostic_common.bindings()
    names = (
        Path(__file__),
        ROOT / "vta/python/vta/top/vta_staged_weight.py",
        HERE / "c3_vthread_staged_protocol.py",
        HERE / "c3_vthread_staged_board.py",
        HERE / "c3_vthread_staged_results.py",
        HERE / "test_vta_vthread_weight_residency.py",
    )
    return {**result, **{str(path.resolve()): s.file_hash(path) for path in names}}


def make_candidates(fingerprint, schedule_version):
    rows = s.read(PARENT / "candidates.json")
    baseline = copy.deepcopy(next(row for row in rows if row["candidate_id"] == "T8_A"))
    expected = {
        "tile_b": 1,
        "tile_h": 8,
        "tile_w": 56,
        "tile_ci": 1,
        "tile_co": 2,
        "oc_nthread": 2,
        "h_nthread": 1,
    }
    if baseline["status"] != "passed" or baseline["visible_features"] != expected:
        raise ValueError("sealed T8 genuine TopHub baseline identity changed")
    shared = {
        "tile": "T8",
        "family_id": "R18-H1-T8-O2-staged-full-residency",
        "workload_id": "R18-H1",
        "complete_config_entity": baseline["complete_config_entity"],
        "visible_features": baseline["visible_features"],
        "hardware_fingerprint": fingerprint,
        "applicability": {"full_weight_fits_sram": True},
        "schedule_version": schedule_version,
    }
    result = []
    for variant, mode, name, source_tophub, build_variant in (
        ("O2", 0, "original_o2", True, "stock"),
        ("PE", 7, "weight_vthread_resident_eager", False, "stock"),
        (
            "PS",
            8,
            "weight_vthread_resident_staged",
            False,
            "staged_weight_residency",
        ),
    ):
        row = {
            **shared,
            "candidate_id": "T8_" + variant,
            "variant": variant,
            "implementation_mode": mode,
            "residence_mode": name,
            "source_tophub": source_tophub,
            "build_config_variant": build_variant,
            "identity": {
                "workload": baseline["identity"]["workload"],
                "complete_config_entity": baseline["complete_config_entity"],
            },
        }
        row["identity_sha256"] = s.digest(
            {key: value for key, value in row.items() if key != "candidate_id"}
        )
        result.append(row)
    return result


def _bytes_before_first_mac(address):
    transferred = 0
    for event in address["ordered_events"]:
        if event[0] == "VTALoadBuffer2D" and event[1][-1] == 1:
            transferred += event[1][1] * event[1][2] * 256
        if event[0] == "kernel" and any(
            uop["args"][0] == 0 and uop["args"][1] == 0 for uop in event[1]
        ):
            return transferred
    raise ValueError("no effective MAC in final TIR")


def pair_audit(rows, addresses):
    by_id = {row["candidate_id"]: row for row in rows}
    eager, staged = by_id["T8_PE"], by_id["T8_PS"]
    if eager["status"] != "passed" or staged["status"] != "passed":
        return {"eligible_program_comparison": False, "reason": "candidate unqualified"}
    eager_local, staged_local = eager["local_qualification"], staged["local_qualification"]
    eager_totals = eager_local["semantics"]["totals"]
    staged_totals = staged_local["semantics"]["totals"]
    structural = audit.control_match(
        addresses["T8_PE"],
        addresses["T8_PS"],
        None,
        None,
        "residency",
    )
    eager_prefix = _bytes_before_first_mac(addresses["T8_PE"])
    staged_prefix = _bytes_before_first_mac(addresses["T8_PS"])
    weight_groups = [
        row
        for row in staged_local["semantics"]["request_groups"]
        if row["direction"] == "load" and row["memory_name"] == "wgt"
    ]
    checks = {
        "same_complete_config": eager["complete_config_entity"]
        == staged["complete_config_entity"],
        "same_two_vthread_knobs": eager["visible_features"]["oc_nthread"]
        == staged["visible_features"]["oc_nthread"]
        == 2,
        "same_compute_context": structural["checks"]["compute_context"],
        "same_nonweight_requests": structural["checks"]["nonweight_data"],
        "same_sync_count": structural["checks"]["sync_count"],
        "same_storage_arenas": eager_local["storage_arenas"]
        ["scope_arena_span_sum_upper_bound_bytes"]
        == staged_local["storage_arenas"]["scope_arena_span_sum_upper_bound_bytes"],
        "same_total_weight_bytes": eager_totals["load_buffer_2d_wgt_bytes"]
        == staged_totals["load_buffer_2d_wgt_bytes"]
        == 36864,
        "staged_is_eight_equal_groups": len(weight_groups) == 8
        and all(row["bytes_per_request"] == 4608 for row in weight_groups),
        "staged_reduces_pre_first_mac_weight_bytes": eager_prefix == 36864
        and staged_prefix == 9216,
    }
    return {
        "eligible_program_comparison": all(checks.values()),
        "checks": checks,
        "eager_pre_first_mac_weight_bytes": eager_prefix,
        "staged_pre_first_mac_weight_bytes": staged_prefix,
        "declared_mechanism_difference": (
            "same full-resident WGT content and compute context; only WGT request "
            "granularity/placement and derived instruction/UOP organization may differ"
        ),
        "interpretation": "program-level first-use staging effect, not isolated DMA cycles",
    }


def prepare(args, out):
    import numpy as np
    import tvm
    from audit_resnet18_sram_accounting import domain_metrics
    from qualify_vta_residency_fsim import array_sha256, reference_data_from_workload

    parent_contract = s.verify(PARENT)
    for name, path in parent_contract["runtime_paths"].items():
        if s.file_hash(path) != parent_contract["runtime_sha256"][name]:
            raise ValueError("qualified runtime changed: " + name)
    _, fingerprint = s.hardware()
    source = bindings()
    rows = make_candidates(fingerprint, s.digest(source))
    for folder in ("logs", "tir", "binaries", "address_audit"):
        (out / folder).mkdir()

    for label, test in (
        ("timer", HERE / "test_c3_reuse_timed_once.py"),
        ("mechanism", HERE / "test_vta_vthread_weight_residency.py"),
    ):
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(test)],
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
        (out / (label + "_test.log")).write_text(completed.stdout + completed.stderr)
        if completed.returncode:
            raise ValueError(label + " regression failed")
    helper_sha256 = diagnostic_common.build_helper(args, out)

    expected_outputs = [
        reference_data_from_workload(rows[0]["identity"]["workload"], seed)[2]
        for seed in protocol.SEEDS
    ]
    mismatch_counts = {
        "{}/{}".format(protocol.SEEDS[left], protocol.SEEDS[right]): int(
            np.count_nonzero(expected_outputs[left] != expected_outputs[right])
        )
        for left, right in itertools.combinations(range(3), 2)
    }
    if not all(mismatch_counts.values()):
        raise ValueError("reference seeds do not discriminate outputs")

    contract = {
        "schema": SCHEMA,
        "phase": "prepare",
        "scope": SCOPE,
        "parent_directory": str(PARENT),
        "parent_manifest_sha256": s.file_hash(PARENT / "artifact_hashes.json"),
        "source_bindings": source,
        "runtime_paths": parent_contract["runtime_paths"],
        "runtime_sha256": parent_contract["runtime_sha256"],
        "helper_sha256": helper_sha256,
        "bitstream_sha256": s.BITSTREAM,
        "queue_environment": protocol.QUEUE_ENV,
        "seeds": list(protocol.SEEDS),
        "sequences": protocol.SEQUENCES,
        "balance": protocol.balance(),
        "reference_hashes": {
            str(seed): array_sha256(value)
            for seed, value in zip(protocol.SEEDS, expected_outputs)
        },
        "seed_pair_mismatches": mismatch_counts,
        "affinity_cpu": 3,
        "warmup_per_candidate": 3,
        "timing_invocations_per_observation": 1,
        "timing_rounds": len(protocol.SEQUENCES),
        "development_session_only": True,
        "no_cross_boot_stability_claim": True,
        "timer_test_passed": True,
        "mechanism_test_passed": True,
        "board_contacted": False,
    }
    s.write(out / "contract.json", contract)

    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for path in source:
        original = Path(path)
        if original.suffix in (".py", ".cc", ".h", ".json") and original.is_relative_to(ROOT):
            target = snapshot / original.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)

    addresses = {}
    with adapter():
        for row in rows:
            candidate_id = row["candidate_id"]
            row["local_qualification"] = s.lower_candidate(row, out)
            try:
                if row["local_qualification"]["status"] != "passed":
                    raise ValueError("lowering rejected")
                module = tvm.ir.load_json((out / "tir" / (candidate_id + ".json")).read_text())
                address = audit.final_tir_audit(
                    module, row["local_qualification"]["semantics"]
                )
                addresses[candidate_id] = address
                s.write(out / "address_audit" / (candidate_id + ".json"), address)
                row["address_status"] = "passed"
                row["arenas"] = domain_metrics(
                    row["local_qualification"]["storage_arenas"]["allocations"],
                    fingerprint,
                )
                if any(
                    value is None or value > 100
                    for key, value in row["arenas"].items()
                    if key.endswith("_arena_capacity_pct")
                ):
                    raise ValueError("arena legality ambiguous or exceeds capacity")
                row["fsim_qualification"] = s.fsim(
                    row,
                    row["local_qualification"],
                    out,
                    str(ROOT / "build/libvta_fsim.so"),
                    args.timeout,
                )
                if row["fsim_qualification"]["status"] != "passed":
                    raise ValueError("FSim rejected")
                row["command_qualification"] = diagnostic_common.diagnostic(row, out, args)
                with adapter():
                    row["cross_qualification"] = s.cross(row, out, args)
                if row["cross_qualification"]["status"] != "passed":
                    raise ValueError("cross compilation rejected")
                row["status"] = "passed"
            except Exception as error:  # preserve every attempted identity
                row.update(status="rejected", rejection=str(error))
            s.append(out / "qualification.jsonl", row)
            print(candidate_id, row["status"], row.get("rejection", ""), flush=True)

    pair = pair_audit(rows, addresses)
    s.write(out / "pair_audit.json", pair)
    s.write(out / "candidates.json", rows)
    expected_ids = [row["candidate_id"] for row in rows]
    passed = [row["candidate_id"] for row in rows if row["status"] == "passed"]
    s.write(
        out / "summary.json",
        {
            "status": "PREPARED"
            if passed == expected_ids
            and pair["eligible_program_comparison"]
            else "LOCAL_GATE_REJECTED",
            "passed": passed,
            "pair_eligible": pair["eligible_program_comparison"],
            "board_contacted": False,
        },
    )
    s.check_bindings(source)


def freeze(args, out):
    prepared = Path(args.contract).resolve().parent
    contract = s.verify(prepared)
    if contract.get("schema") != SCHEMA or contract.get("phase") != "prepare":
        raise ValueError("freeze requires this experiment's prepare contract")
    s.check_bindings(contract["source_bindings"])
    rows = s.read(prepared / "candidates.json")
    expected = [row["candidate_id"] for row in rows]
    accepted = [row["candidate_id"] for row in rows if row["status"] == "passed"]
    if accepted != expected:
        raise ValueError("all preregistered candidates must qualify")
    if not s.read(prepared / "pair_audit.json")["eligible_program_comparison"]:
        raise ValueError("eager/staged comparison is not qualified")
    for row in rows:
        if row["address_status"] != "passed" or any(
            row[key]["status"] != "passed"
            for key in (
                "local_qualification",
                "fsim_qualification",
                "command_qualification",
                "cross_qualification",
            )
        ):
            raise ValueError("candidate qualification incomplete")
    frozen = {
        **contract,
        "phase": "freeze",
        "prepared_directory": str(prepared),
        "prepared_manifest_sha256": s.file_hash(prepared / "artifact_hashes.json"),
        "accepted": accepted,
        "timing_schedule": protocol.schedule(accepted),
        "balance": protocol.balance(),
        "analysis_comparisons": protocol.COMPARISONS,
        "performance_rule": (
            "single-boot development signal: paired median >=2% and wins >=4/6; "
            "not a cross-boot stable result"
        ),
    }
    s.write(out / "contract.json", frozen)
    s.write(
        out / "summary.json",
        {
            "status": "FROZEN_STAGED_RESIDENCY_DEVELOPMENT",
            "accepted": accepted,
            "timing_samples": len(frozen["timing_schedule"]),
            "board_contacted": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "freeze", "board", "analyze"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--contract")
    parser.add_argument("--board-dir", type=Path)
    parser.add_argument("--execute-board", action="store_true")
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    parser.add_argument("--sdk-environment", default=SDK)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--worker-request")
    parser.add_argument("--cross-request")
    parser.add_argument("--diagnostic-request")
    parser.add_argument("--worker-response")
    args = parser.parse_args()
    with adapter():
        if args.worker_request:
            s.worker(s.read(args.worker_request), Path(args.worker_response))
            return
        if args.cross_request:
            s.cross_worker(s.read(args.cross_request), Path(args.worker_response))
            return
        if args.diagnostic_request:
            diagnostic_common.diagnostic_worker(
                s.read(args.diagnostic_request), Path(args.worker_response)
            )
            return
    if not args.phase or not args.output_dir:
        parser.error("phase and output-dir required")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        if args.phase == "prepare":
            prepare(args, out)
        elif args.phase == "freeze":
            freeze(args, out)
        elif args.phase == "board":
            from c3_vthread_staged_board import run

            run(args, out)
        else:
            from c3_vthread_staged_results import analyze

            analyze(args, out)
    except BaseException as error:
        s.write(
            out / "invalid_session.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "all_timing_invalid": True,
                "wall_seconds": time.monotonic() - started,
            },
        )
        raise
    finally:
        s.write(out / "phase_cost.json", {"wall_seconds": time.monotonic() - started})
        s.seal(out)


if __name__ == "__main__":
    main()
