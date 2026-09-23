#!/usr/bin/env python3
"""T7 prepare/freeze/run/analyze entry for the two-vthread residency pilot."""

from __future__ import annotations

import argparse
import contextlib
import copy
from pathlib import Path
import time

import c3_vthread_residency_t7_protocol as protocol
import run_vta_c3_reuse_supplement as s
import run_vta_c3_vthread_residency as base


HERE, ROOT = s.HERE, s.ROOT
PARENT = base.PARENT
SCHEMA = "c3_vthread_weight_residency_t7_v1"
SDK = base.SDK
_BASE_BINDINGS = base.bindings


def bindings():
    result = _BASE_BINDINGS()
    names = (
        Path(__file__),
        HERE / "c3_vthread_residency_t7_protocol.py",
        HERE / "c3_vthread_residency_t7_board.py",
        HERE / "c3_vthread_residency_t7_results.py",
    )
    return {**result, **{str(path.resolve()): s.file_hash(path) for path in names}}


def make_candidates(fingerprint, schedule_version):
    rows = s.read(PARENT / "candidates.json")
    baseline = copy.deepcopy(next(row for row in rows if row["candidate_id"] == "T7_A"))
    expected = {
        "tile_b": 1,
        "tile_h": 7,
        "tile_w": 56,
        "tile_ci": 1,
        "tile_co": 2,
        "oc_nthread": 2,
        "h_nthread": 1,
    }
    if baseline["status"] != "passed" or baseline["visible_features"] != expected:
        raise ValueError("sealed T7 mode-0 baseline identity changed")
    shared = {
        "tile": "T7",
        "family_id": "R18-H1-T7-O2-vthread-residency",
        "workload_id": "R18-H1",
        "complete_config_entity": baseline["complete_config_entity"],
        "visible_features": baseline["visible_features"],
        "hardware_fingerprint": fingerprint,
        "applicability": {"full_weight_fits_sram": True},
        "schedule_version": schedule_version,
        "baseline_provenance": "production mode-0 schedule with historical ALT_251 T7 entity; not genuine TopHub",
    }
    result = []
    for variant, mode, name, production_original in (
        ("O2", 0, "original_o2", True),
        ("VR", 6, "weight_vthread_reload", False),
        ("VS", 7, "weight_vthread_resident", False),
    ):
        row = {
            **shared,
            "candidate_id": "T7_" + variant,
            "variant": variant,
            "implementation_mode": mode,
            "residence_mode": name,
            # This flag selects the production mode-0 template.  T7 is not
            # presented as an official/genuine TopHub entity.
            "source_tophub": production_original,
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


def pair_audit(rows, addresses):
    by_id = {row["candidate_id"]: row for row in rows}
    reload, resident = by_id["T7_VR"], by_id["T7_VS"]
    if reload["status"] != "passed" or resident["status"] != "passed":
        return {"eligible_program_comparison": False, "reason": "candidate unqualified"}
    left, right = reload["local_qualification"], resident["local_qualification"]
    left_totals, right_totals = left["semantics"]["totals"], right["semantics"]["totals"]
    checks = {
        "same_complete_config": reload["complete_config_entity"]
        == resident["complete_config_entity"],
        "same_two_vthread_knobs": reload["visible_features"]["oc_nthread"]
        == resident["visible_features"]["oc_nthread"]
        == 2,
        "same_compute_program": addresses["T7_VR"]["compute_sha256"]
        == addresses["T7_VS"]["compute_sha256"],
        "same_inp_requests": all(
            left_totals[key] == right_totals[key]
            for key in ("load_buffer_2d_inp_bytes", "load_buffer_2d_inp_calls")
        ),
        "same_out_requests": all(
            left_totals[key] == right_totals[key]
            for key in ("store_buffer_2d_out_bytes", "store_buffer_2d_out_calls")
        ),
        "same_storage_arenas": left["storage_arenas"]
        ["scope_arena_span_sum_upper_bound_bytes"]
        == right["storage_arenas"]["scope_arena_span_sum_upper_bound_bytes"],
        "resident_reduces_weight_bytes": right_totals["load_buffer_2d_wgt_bytes"]
        < left_totals["load_buffer_2d_wgt_bytes"],
    }
    return {
        "eligible_program_comparison": all(checks.values()),
        "checks": checks,
        "declared_mechanism_differences": {
            "reload_sync_multiplicity": left["sync_multiplicity"],
            "resident_sync_multiplicity": right["sync_multiplicity"],
            "reload_weight_bytes": left_totals["load_buffer_2d_wgt_bytes"],
            "resident_weight_bytes": right_totals["load_buffer_2d_wgt_bytes"],
            "reload_weight_calls": left_totals["load_buffer_2d_wgt_calls"],
            "resident_weight_calls": right_totals["load_buffer_2d_wgt_calls"],
        },
        "interpretation": (
            "program-level residency effect including dependency, submission, "
            "instruction, and uop changes; not isolated DMA time"
        ),
    }


@contextlib.contextmanager
def configured_base():
    previous = (
        base.SCHEMA,
        base.protocol,
        base.bindings,
        base.make_candidates,
        base.pair_audit,
    )
    base.SCHEMA = SCHEMA
    base.protocol = protocol
    base.bindings = bindings
    base.make_candidates = make_candidates
    base.pair_audit = pair_audit
    try:
        yield
    finally:
        (
            base.SCHEMA,
            base.protocol,
            base.bindings,
            base.make_candidates,
            base.pair_audit,
        ) = previous


def prepare(args, out):
    base.prepare(args, out)
    contract = s.read(out / "contract.json")
    contract.update(
        scope="R18-H1, T7, oc_nthread=2, h_nthread=1, weight-only residency pilot",
        baseline_provenance=(
            "production mode-0 schedule with historical ALT_251 T7 entity; not genuine TopHub"
        ),
    )
    s.write(out / "contract.json", contract)


def freeze(args, out):
    prepared = Path(args.contract).resolve().parent
    contract = s.verify(prepared)
    if contract.get("schema") != SCHEMA or contract.get("phase") != "prepare":
        raise ValueError("freeze requires the T7 prepare contract")
    s.check_bindings(contract["source_bindings"])
    rows = s.read(prepared / "candidates.json")
    accepted = [row["candidate_id"] for row in rows if row["status"] == "passed"]
    expected = ["T7_O2", "T7_VR", "T7_VS"]
    if accepted != expected:
        raise ValueError("all three preregistered T7 candidates must qualify")
    if not s.read(prepared / "pair_audit.json")["eligible_program_comparison"]:
        raise ValueError("T7 reload/resident program comparison is not qualified")
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
            raise ValueError("T7 candidate qualification incomplete")
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
            "status": "FROZEN_T7_DEVELOPMENT_PILOT",
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
    with configured_base(), base.adapter():
        if args.worker_request:
            s.worker(s.read(args.worker_request), Path(args.worker_response))
            return
        if args.cross_request:
            s.cross_worker(s.read(args.cross_request), Path(args.worker_response))
            return
        if args.diagnostic_request:
            base.old.diagnostic_worker(s.read(args.diagnostic_request), Path(args.worker_response))
            return
    if not args.phase or not args.output_dir:
        parser.error("phase and output-dir required")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        with configured_base():
            if args.phase == "prepare":
                prepare(args, out)
            elif args.phase == "freeze":
                freeze(args, out)
            elif args.phase == "board":
                from c3_vthread_residency_t7_board import run

                run(args, out)
            else:
                from c3_vthread_residency_t7_results import analyze

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

