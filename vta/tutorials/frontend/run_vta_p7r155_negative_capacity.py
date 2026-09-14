#!/usr/bin/env python3
"""Complete P7R155 with isolated accept/reject RPC processes.

The positive 16 KiB/4 KiB evidence is imported from the immutable P7R155
run02 artifact.  Under the one-page-smaller 12 KiB/4 KiB contract, an
in-capacity identity and the 13,488-byte identity are then exercised in two
fresh RPC processes.  This prevents the expected capacity exception from
contaminating the accepted control's RPC session or queue counters.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import traceback
from pathlib import Path

from run_vta_p7r155_selected_allowlist_capacity import (
    Board,
    EXPECTED_CAPACITY_RUNTIME_HASHES,
    EXPECTED_DEFAULT_RUNTIME_HASHES,
    build_all,
    canonical_sha256,
    capacity_environment,
    check_health,
    finalize,
    read_json,
    run_entries,
    sha256_file,
    three_seeds,
    validate_environment,
    verify_run,
    write_json,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P7 = ROOT / "07_grouped_holdout"
DEFAULT_POSITIVE_RUN = P7 / "20260912_p7r155_selected_allowlist_capacity_run02"
DEFAULT_OUTPUT = P7 / "20260912_p7r155_negative_capacity_run04"


def single_seed(_entry):
    return (0,)


def validate_positive_run(directory):
    ledger_hash = verify_run(directory)
    contract = read_json(directory / "contract.json")
    positive = read_json(directory / "positive_capacity.json")
    rows = positive["rows"]
    queue = positive["queue_status"]
    expected = contract["capacity_derivation"]["positive"]
    if len(rows) != 9 or not all(row.get("correct") for row in rows):
        raise ValueError("imported positive evidence is not 9/9 correct")
    if (
        int(queue["insn_capacity_bytes"]) != int(expected["insn_bytes"])
        or int(queue["uop_capacity_bytes"]) != int(expected["uop_bytes"])
        or int(queue["submissions"]) != 9
    ):
        raise ValueError("imported positive queue evidence differs from its contract")
    return ledger_hash, contract, positive


def capacity_row_by_id(contract, candidate_id):
    matches = [row for row in contract["allowlist"] if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError("candidate is not unique in imported allowlist: " + candidate_id)
    return matches[0]


def validate_accepted(result, entry, capacity, manifest_id):
    rows = result["rows"]
    queue = result["queue_status"]
    if len(rows) != 1 or not rows[0].get("correct"):
        raise RuntimeError("in-capacity control was not accepted correctly")
    if int((rows[0].get("runtime_profile_complete") or {}).get("driver_run_calls", -1)) != 1:
        raise RuntimeError("in-capacity control did not reach exactly one device run")
    if (
        int(queue.get("insn_capacity_bytes", -1)) != int(capacity["insn_bytes"])
        or int(queue.get("uop_capacity_bytes", -1)) != int(capacity["uop_bytes"])
        or int(queue.get("submissions", -1)) != 1
        or int(queue.get("insn_peak_bytes", -1)) > int(capacity["insn_bytes"])
        or int(queue.get("uop_peak_bytes", -1)) > int(capacity["uop_bytes"])
        or queue.get("command_manifest_id") != manifest_id
        or queue.get("replay_policy") != "disabled"
    ):
        raise RuntimeError("in-capacity control queue status violates the frozen contract")
    if int(queue["insn_peak_bytes"]) > int(entry["command_peak"]["insn_bytes"]):
        raise RuntimeError("observed in-capacity instruction peak exceeds the static certificate")


def validate_rejected(result, capacity, manifest_id):
    rows = result["rows"]
    queue = result["queue_status"]
    if len(rows) != 1:
        raise RuntimeError("expected exactly one out-of-capacity observation")
    row = rows[0]
    profile = row.get("runtime_profile_complete") or {}
    if (
        row.get("status") != "execution_failed"
        or "queue backing capacity exceeded before submission" not in row.get("message", "")
        or int(profile.get("driver_run_calls", -1)) != 0
    ):
        raise RuntimeError("out-of-capacity identity was not rejected before device run")
    if (
        int(queue.get("insn_capacity_bytes", -1)) != int(capacity["insn_bytes"])
        or int(queue.get("uop_capacity_bytes", -1)) != int(capacity["uop_bytes"])
        or int(queue.get("submissions", -1)) != 0
        or int(queue.get("insn_peak_bytes", -1)) != 0
        or int(queue.get("uop_peak_bytes", -1)) != 0
        or queue.get("command_manifest_id") != manifest_id
        or queue.get("replay_policy") != "disabled"
    ):
        raise RuntimeError("rejected process queue counters are not fail-closed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-run", type=Path, default=DEFAULT_POSITIVE_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--ssh-known-hosts", type=Path, required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    parser.add_argument("--capacity-runtime", default="/var/volatile/vta_c3_ram/capacity_runtime")
    args = parser.parse_args()

    output = args.output_dir
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    output.mkdir(parents=True)
    board = Board(args.host, args.ssh_known_hosts)
    active_rpc = None
    error = None
    before = None
    health_before = None
    health_after = None
    accepted = None
    rejected = None
    try:
        ledger_hash, source_contract, source_positive = validate_positive_run(args.positive_run)
        derivation = source_contract["capacity_derivation"]
        negative_capacity = derivation["negative_one_insn_page"]
        target = capacity_row_by_id(source_contract, derivation["negative_target_candidate_id"])
        eligible = [
            row for row in source_contract["allowlist"]
            if int(row["command_peak"]["insn_bytes"]) <= int(negative_capacity["insn_bytes"])
            and int(row["command_peak"]["uop_bytes"]) <= int(negative_capacity["uop_bytes"])
        ]
        if not eligible:
            raise ValueError("no in-capacity control exists in the imported allowlist")
        # Keep this a same-workload comparison.  In the positive run, Y00 is
        # allocated before Y03; selecting Y03 alone would itself change the
        # tensor address layout and confound the queue-capacity experiment.
        y00_selected = [
            row for row in eligible
            if row.get("workload_id") == "Y00" and row.get("allowlist_role") == "selected"
        ]
        if len(y00_selected) != 1 or target.get("workload_id") != "Y00":
            raise ValueError("expected a unique Y00 selected control and Y00 largest target")
        control = y00_selected[0]
        layout_padding_bytes = int(derivation["positive"]["insn_bytes"]) - int(
            negative_capacity["insn_bytes"]
        )
        if layout_padding_bytes != 4096:
            raise ValueError("expected exactly one page of address-preserving padding")
        if not (
            int(control["command_peak"]["insn_bytes"]) <= int(negative_capacity["insn_bytes"])
            < int(target["command_peak"]["insn_bytes"])
        ):
            raise ValueError("negative capacity does not straddle control and target")

        before = board.state()
        if before.get("FPGA") != "operating" or before.get("UDMABUF") != "201326592":
            raise RuntimeError("FPGA/u-dma-buf preflight failed")
        if before.get("STORAGE_ERRORS"):
            raise RuntimeError("current boot has EXT4/mmc errors: {}".format(before["STORAGE_ERRORS"]))
        if board.runtime_hashes(args.default_runtime) != EXPECTED_DEFAULT_RUNTIME_HASHES:
            raise RuntimeError("default runtime hash mismatch")
        if board.runtime_hashes(args.capacity_runtime) != EXPECTED_CAPACITY_RUNTIME_HASHES:
            raise RuntimeError("capacity runtime hash mismatch")
        original_rpc = board.rpc_state()
        if original_rpc.get("CWD") != args.default_runtime:
            raise RuntimeError("default RPC is not in the frozen tmpfs directory")

        manifest_id = canonical_sha256({
            "capacity": negative_capacity,
            "accepted_control": control["candidate_id"],
            "rejected_target": target["candidate_id"],
            "source_positive_ledger": ledger_hash,
        })
        environment = capacity_environment(negative_capacity, manifest_id)
        contract = {
            "schema": "c3_p7r155_isolated_negative_capacity_contract_v1",
            "status": "frozen_before_isolated_negative_fpga_execution",
            "source_positive_run": str(args.positive_run.resolve()),
            "source_positive_artifact_ledger_sha256": ledger_hash,
            "source_positive_capacity": derivation["positive"],
            "source_positive_correctness": "9/9",
            "negative_capacity": negative_capacity,
            "accepted_control": control,
            "rejected_target": target,
            "manifest_id": manifest_id,
            "isolation": "fresh RPC process for accept and fresh RPC process for reject",
            "layout_control": {
                "reason": "queue backing is bump-allocated before tensor buffers",
                "padding_bytes": layout_padding_bytes,
                "effect": "negative queue backing plus padding equals positive queue backing before Y00 tensors",
            },
            "board": before,
            "runtime_hashes": {
                "default": EXPECTED_DEFAULT_RUNTIME_HASHES,
                "capacity": EXPECTED_CAPACITY_RUNTIME_HASHES,
            },
            "seed": 0,
            "failure_policy": "persist raw observations before validation; always restore default RPC",
            "performance_measurement": "not_collected",
            "persistent_board_writes": False,
        }
        write_json(output / "contract.json", contract)
        write_json(output / "pre_execution_hashes.json", {
            "contract.json": sha256_file(output / "contract.json")
        })

        with tempfile.TemporaryDirectory(prefix="c3_p7r155_negative_cross_") as temporary:
            directory = Path(temporary)
            health = source_contract["health_canary"]
            certificates = build_all([control, target], health, directory)
            write_json(output / "cross_compile_certificates.json", certificates)
            health_entry = dict(health)
            health_entry["allowlist_role"] = "health_canary"

            health_before = run_entries(
                args.host, args.port, directory, [health_entry], three_seeds
            )
            write_json(output / "health_before.json", health_before)
            check_health(health_before)
            board.stop_rpc(original_rpc)

            active_rpc = board.start_rpc(
                args.capacity_runtime, "p7r155_negative_accept.log", environment
            )
            validate_environment(active_rpc, environment)
            accepted = run_entries(
                args.host, args.port, directory, [control], single_seed,
                require_queue_status=True,
                preallocate_bytes=layout_padding_bytes,
            )
            write_json(output / "accepted_control_raw.json", accepted)
            validate_accepted(accepted, control, negative_capacity, manifest_id)
            board.stop_rpc(active_rpc)
            active_rpc = None

            active_rpc = board.start_rpc(
                args.capacity_runtime, "p7r155_negative_reject.log", environment
            )
            validate_environment(active_rpc, environment)
            rejected = run_entries(
                args.host, args.port, directory, [target], single_seed,
                require_queue_status=True,
                preallocate_bytes=layout_padding_bytes,
            )
            write_json(output / "rejected_target_raw.json", rejected)
            validate_rejected(rejected, negative_capacity, manifest_id)
            board.stop_rpc(active_rpc)
            active_rpc = None

            restored = board.start_rpc(args.default_runtime, "p7r155_negative_restored.log", None)
            forbidden = set(environment)
            if forbidden.intersection(restored):
                raise RuntimeError("restored default RPC retained capacity environment")
            health_after = run_entries(
                args.host, args.port, directory, [health_entry], three_seeds
            )
            write_json(output / "health_after.json", health_after)
            check_health(health_after)

        after = board.state()
        if after.get("BOOT") != before.get("BOOT") or after.get("STORAGE_ERRORS"):
            raise RuntimeError("postflight board/storage state changed")
        summary = {
            "schema": "c3_p7r155_selected_allowlist_capacity_summary_v2",
            "status": "passed",
            "boot_id": before["BOOT"],
            "positive_source_run": str(args.positive_run.resolve()),
            "positive_capacity": derivation["positive"],
            "positive_exact_correctness": "9/9",
            "positive_queue_status": source_positive["queue_status"],
            "undersized_capacity": negative_capacity,
            "accepted_control_candidate_id": control["candidate_id"],
            "accepted_control_correctness": "1/1",
            "accepted_control_queue_status": accepted["queue_status"],
            "rejected_target_candidate_id": target["candidate_id"],
            "rejected_target_status": rejected["rows"][0]["status"],
            "rejected_target_driver_run_calls": 0,
            "rejected_target_queue_status": rejected["queue_status"],
            "negative_processes_isolated": True,
            "address_preserving_padding_bytes": layout_padding_bytes,
            "health_before": "3/3",
            "health_after_restore": "3/3",
            "default_rpc_restored": True,
            "performance_measurement": "not_collected",
            "sd_experiment_writes": "none; runtime/log/upload paths are tmpfs",
            "claim_boundary": "three exact selected/fallback identities; negative boundary on one boot",
        }
        write_json(output / "board_postflight.json", after)
        write_json(output / "summary.json", summary)
        (output / "STATUS.md").write_text(
            "# P7R155 isolated command-capacity validation\n\n"
            "- Status: `passed`\n"
            "- Imported positive gate: instruction 16 KiB + UOP 4 KiB, 9/9 correct\n"
            "- One-page-smaller gate: instruction 12 KiB + UOP 4 KiB\n"
            "- A certified smaller identity was accepted and correct in a fresh RPC process\n"
            "- The 13,488-byte identity was rejected before submission in another fresh RPC process\n"
            "- Default tmpfs RPC restored; W05 health passed 3/3 before and after\n"
            "- No latency measurement and no persistent SD experiment write\n",
            encoding="utf-8",
        )
    except Exception as caught:
        error = caught
        write_json(output / "failure.json", {
            "status": "failed_closed",
            "exception_type": type(caught).__name__,
            "message": (str(caught) or type(caught).__name__)[:4000],
            "traceback": traceback.format_exc()[-16000:],
        })
    finally:
        try:
            if active_rpc is not None:
                current = board.rpc_state()
                if current.get("PID") == active_rpc.get("PID"):
                    board.stop_rpc(active_rpc)
            current = board.rpc_state()
            if current.get("CWD") != args.default_runtime:
                if current.get("PID") and current.get("CWD"):
                    board.stop_rpc(current)
                board.start_rpc(args.default_runtime, "p7r155_negative_emergency_restore.log", None)
        except Exception as restore_error:
            write_json(output / "restoration_failure.json", {
                "exception_type": type(restore_error).__name__,
                "message": (str(restore_error) or type(restore_error).__name__)[:4000],
            })
            if error is None:
                error = restore_error
        finalize(output)
    if error is not None:
        raise error
    print(json.dumps(read_json(output / "summary.json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
