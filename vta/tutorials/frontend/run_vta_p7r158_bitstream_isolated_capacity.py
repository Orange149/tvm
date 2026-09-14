#!/usr/bin/env python3
"""Diagnose command-capacity correctness with a bitstream-isolated A/B/A sequence.

A is the frozen 16 KiB instruction / 4 KiB UOP deployment capacity.  B removes
one instruction page and inserts a same-sized VTA allocation before tensors to
hold their intended bump-allocator offset constant.  The same Y00 selected
identity is run as A1/B/A2 in fresh RPC processes.  The Y00 13,488-byte
fallback is separately required to fail before submission under B.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import traceback
from pathlib import Path

from run_vta_p7r155_negative_capacity import (
    DEFAULT_POSITIVE_RUN,
    capacity_row_by_id,
    single_seed,
    validate_positive_run,
)
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
    write_json,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P7 = ROOT / "07_grouped_holdout"
DEFAULT_OUTPUT = P7 / "20260912_p7r158_bitstream_isolated_capacity_run01"


def reload_frozen_bitstream(board):
    command = (
        "test $(sha256sum /lib/firmware/vta_hpc.bit | cut -d' ' -f1) = "
        "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6; "
        "echo vta_hpc.bit > /sys/class/fpga_manager/fpga0/firmware; "
        "test $(cat /sys/class/fpga_manager/fpga0/state) = operating; "
        "cat /sys/class/fpga_manager/fpga0/state"
    )
    observed = board.run(command, timeout=60).stdout.strip()
    if observed != "operating":
        raise RuntimeError("frozen FPGA reload did not reach operating")
    return {
        "firmware": "vta_hpc.bit",
        "sha256": "7bf1ac95b1182c670cd25241111f33725b4ea37ed6d66f2df37b0b2e7df528d6",
        "state": observed,
    }


def assert_success(result, capacity, manifest_id, label):
    rows = result["rows"]
    queue = result["queue_status"]
    if len(rows) != 1 or not rows[0].get("correct"):
        raise RuntimeError(label + " was not correct")
    if int((rows[0].get("runtime_profile_complete") or {}).get("driver_run_calls", -1)) != 1:
        raise RuntimeError(label + " did not execute exactly once")
    if (
        int(queue.get("insn_capacity_bytes", -1)) != int(capacity["insn_bytes"])
        or int(queue.get("uop_capacity_bytes", -1)) != int(capacity["uop_bytes"])
        or int(queue.get("submissions", -1)) != 1
        or queue.get("command_manifest_id") != manifest_id
        or queue.get("replay_policy") != "disabled"
    ):
        raise RuntimeError(label + " queue status mismatch")


def assert_pre_submit_rejection(result, capacity, manifest_id):
    rows = result["rows"]
    queue = result["queue_status"]
    if len(rows) != 1:
        raise RuntimeError("rejection phase did not produce exactly one row")
    row = rows[0]
    profile = row.get("runtime_profile_complete") or {}
    if (
        row.get("status") != "execution_failed"
        or "queue backing capacity exceeded before submission" not in row.get("message", "")
        or int(profile.get("driver_run_calls", -1)) != 0
    ):
        raise RuntimeError("oversized identity was not rejected before device run")
    if (
        int(queue.get("insn_capacity_bytes", -1)) != int(capacity["insn_bytes"])
        or int(queue.get("uop_capacity_bytes", -1)) != int(capacity["uop_bytes"])
        or int(queue.get("submissions", -1)) != 0
        or int(queue.get("insn_peak_bytes", -1)) != 0
        or int(queue.get("uop_peak_bytes", -1)) != 0
        or queue.get("command_manifest_id") != manifest_id
        or queue.get("replay_policy") != "disabled"
    ):
        raise RuntimeError("rejection queue counters are not fail-closed")


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
    try:
        ledger_hash, source_contract, _ = validate_positive_run(args.positive_run)
        derivation = source_contract["capacity_derivation"]
        positive_capacity = derivation["positive"]
        negative_capacity = derivation["negative_one_insn_page"]
        target = capacity_row_by_id(source_contract, derivation["negative_target_candidate_id"])
        controls = [
            row for row in source_contract["allowlist"]
            if row.get("workload_id") == "Y00" and row.get("allowlist_role") == "selected"
        ]
        if len(controls) != 1 or target.get("workload_id") != "Y00":
            raise ValueError("expected one Y00 selected control and Y00 rejection target")
        control = controls[0]
        module_entries = list(source_contract["allowlist"])
        if len(module_entries) != 3 or len({row["candidate_id"] for row in module_entries}) != 3:
            raise ValueError("source positive allowlist must contain three unique modules")
        padding = int(positive_capacity["insn_bytes"]) - int(negative_capacity["insn_bytes"])
        if padding != 4096:
            raise ValueError("A/B capacity delta must be one 4 KiB page")

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
            raise RuntimeError("default RPC is not active at preflight")

        phase_specs = {
            "A1": {"capacity": positive_capacity, "padding": 0, "entry": control},
            "B_control": {"capacity": negative_capacity, "padding": padding, "entry": control},
            "B_reject": {"capacity": negative_capacity, "padding": padding, "entry": target},
            "A2": {"capacity": positive_capacity, "padding": 0, "entry": control},
            "A3_after_reload": {
                "capacity": positive_capacity, "padding": 0, "entry": control
            },
        }
        for name, spec in phase_specs.items():
            spec["manifest_id"] = canonical_sha256({
                "phase": name,
                "capacity": spec["capacity"],
                "padding": spec["padding"],
                "candidate_id": spec["entry"]["candidate_id"],
                "source_positive_ledger": ledger_hash,
            })
        contract = {
            "schema": "c3_p7r158_bitstream_isolated_capacity_contract_v1",
            "status": "frozen_before_p7r158_fpga_execution",
            "source_positive_run": str(args.positive_run.resolve()),
            "source_positive_artifact_ledger_sha256": ledger_hash,
            "board": before,
            "control": control,
            "rejection_target": target,
            "loaded_module_set": [row["candidate_id"] for row in module_entries],
            "phase_specs": phase_specs,
            "runtime_hashes": {
                "default": EXPECTED_DEFAULT_RUNTIME_HASHES,
                "capacity": EXPECTED_CAPACITY_RUNTIME_HASHES,
            },
            "seed": 0,
            "process_policy": "every A/B phase uses a fresh RPC process",
            "fpga_reset_policy": (
                "reload exact frozen bitstream before A1 and before A3; do not reload between "
                "A1/B/A2 so persistent state is observable"
            ),
            "observation_policy": "write raw result before semantic validation",
            "performance_measurement": "not_collected",
            "persistent_board_writes": False,
        }
        write_json(output / "contract.json", contract)
        write_json(output / "pre_execution_hashes.json", {
            "contract.json": sha256_file(output / "contract.json")
        })

        with tempfile.TemporaryDirectory(prefix="c3_p7r156_cross_") as temporary:
            directory = Path(temporary)
            health = source_contract["health_canary"]
            write_json(
                output / "cross_compile_certificates.json",
                build_all(module_entries, health, directory),
            )
            health_entry = dict(health)
            health_entry["allowlist_role"] = "health_canary"
            health_before = run_entries(
                args.host, args.port, directory, [health_entry], three_seeds
            )
            write_json(output / "health_before.json", health_before)
            check_health(health_before)
            board.stop_rpc(original_rpc)

            reload_before = reload_frozen_bitstream(board)
            write_json(output / "reload_before_A1.json", reload_before)
            reloaded_default = board.start_rpc(args.default_runtime, "p7r158_reloaded_health.log", None)
            health_after_reload = run_entries(
                args.host, args.port, directory, [health_entry], three_seeds
            )
            write_json(output / "health_after_reload_before_A1.json", health_after_reload)
            check_health(health_after_reload)
            board.stop_rpc(reloaded_default)

            results = {}
            for name in ("A1", "B_control", "B_reject", "A2"):
                spec = phase_specs[name]
                environment = capacity_environment(spec["capacity"], spec["manifest_id"])
                active_rpc = board.start_rpc(
                    args.capacity_runtime, "p7r156_{}.log".format(name.lower()), environment
                )
                validate_environment(active_rpc, environment)
                result = run_entries(
                    args.host, args.port, directory, [spec["entry"]], single_seed,
                    require_queue_status=True, preallocate_bytes=spec["padding"],
                    module_entries=module_entries,
                )
                results[name] = result
                write_json(output / (name + "_raw.json"), result)
                board.stop_rpc(active_rpc)
                active_rpc = None

            reload_after = reload_frozen_bitstream(board)
            write_json(output / "reload_before_A3.json", reload_after)
            spec = phase_specs["A3_after_reload"]
            environment = capacity_environment(spec["capacity"], spec["manifest_id"])
            active_rpc = board.start_rpc(
                args.capacity_runtime, "p7r158_a3_after_reload.log", environment
            )
            validate_environment(active_rpc, environment)
            results["A3_after_reload"] = run_entries(
                args.host, args.port, directory, [spec["entry"]], single_seed,
                require_queue_status=True, module_entries=module_entries,
            )
            write_json(output / "A3_after_reload_raw.json", results["A3_after_reload"])
            board.stop_rpc(active_rpc)
            active_rpc = None

            # Restore and prove board health before interpreting the diagnostic.
            restored = board.start_rpc(args.default_runtime, "p7r158_restored.log", None)
            forbidden = set(capacity_environment(positive_capacity, phase_specs["A2"]["manifest_id"]))
            if forbidden.intersection(restored):
                raise RuntimeError("restored default RPC retained capacity environment")
            health_after = run_entries(
                args.host, args.port, directory, [health_entry], three_seeds
            )
            write_json(output / "health_after.json", health_after)
            check_health(health_after)

            # Validate only after every raw phase and restoration evidence exist.
            assert_success(results["A1"], positive_capacity, phase_specs["A1"]["manifest_id"], "A1")
            assert_success(
                results["A3_after_reload"], positive_capacity,
                phase_specs["A3_after_reload"]["manifest_id"], "A3_after_reload",
            )
            assert_pre_submit_rejection(
                results["B_reject"], negative_capacity, phase_specs["B_reject"]["manifest_id"]
            )
            b_row = results["B_control"]["rows"][0]
            b_queue = results["B_control"]["queue_status"]
            b_correct = bool(b_row.get("correct"))
            a2_row = results["A2"]["rows"][0]
            a2_correct = bool(a2_row.get("correct"))

        after = board.state()
        if after.get("BOOT") != before.get("BOOT") or after.get("STORAGE_ERRORS"):
            raise RuntimeError("postflight board/storage state changed")
        summary = {
            "schema": "c3_p7r158_bitstream_isolated_capacity_summary_v1",
            "status": (
                "passed" if b_correct and a2_correct
                else "passed_with_capacity_induced_persistent_correctness_anomaly"
            ),
            "boot_id": before["BOOT"],
            "A1_correct": True,
            "A2_correct_without_reload": a2_correct,
            "A2_status": a2_row.get("status"),
            "A2_mismatch_count": a2_row.get("mismatch_count"),
            "A3_correct_after_reload": True,
            "A_capacity": positive_capacity,
            "B_control_correct": b_correct,
            "B_control_status": b_row.get("status"),
            "B_control_mismatch_count": b_row.get("mismatch_count"),
            "B_control_queue_status": b_queue,
            "B_capacity": negative_capacity,
            "B_address_offset_compensation_bytes": padding,
            "B_target_rejected_before_device_run": True,
            "B_target_queue_status": results["B_reject"]["queue_status"],
            "loaded_module_count_each_phase": len(module_entries),
            "frozen_bitstream_reloaded_before_A1_and_A3": True,
            "health_before": "3/3",
            "health_after_restore": "3/3",
            "default_rpc_restored": True,
            "deployment_decision": (
                "retain 16 KiB/4 KiB; 12 KiB/4 KiB is invalid for this exact allowlist "
                "whether by capacity rejection or hardware correctness"
            ),
            "claim_boundary": "same Y00 control A/B/A and exact Y00 fallback rejection on one boot",
        }
        write_json(output / "board_postflight.json", after)
        write_json(output / "summary.json", summary)
        (output / "STATUS.md").write_text(
            "# P7R158 bitstream-isolated command-capacity diagnostic\n\n"
            "- Status: `{}`\n"
            "- A1 and A3 after exact bitstream reload: 16 KiB + 4 KiB, same Y00 control correct\n"
            "- A2 without reload after B: `{}`\n"
            "- B control: 12 KiB instruction + 4 KiB UOP with 4 KiB offset compensation, `{}`\n"
            "- B rejection: exact 13,488-byte Y00 fallback rejected before device run\n"
            "- Deployment decision: retain 16 KiB + 4 KiB for this exact allowlist\n"
            "- Default RPC restored; W05 health passed before and after\n".format(
                summary["status"], a2_row.get("status"), b_row.get("status")
            ),
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
                board.start_rpc(args.default_runtime, "p7r158_emergency_restore.log", None)
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
