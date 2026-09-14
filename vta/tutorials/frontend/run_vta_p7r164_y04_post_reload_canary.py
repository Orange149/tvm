#!/usr/bin/env python3
"""Retest a label-independent Y04 original after exact bitstream reload."""

from __future__ import annotations

import argparse
import gc
import json
import tempfile
import traceback
from pathlib import Path

from tvm import rpc
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from prepare_vta_p7r120_yolo_rpc_contract import build_and_export
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory
from run_vta_p7r132_y00_search_confirmation import (
    allocate_reusable_buffers,
    profiled_reused_call,
)
from run_vta_p7r155_selected_allowlist_capacity import (
    Board,
    EXPECTED_DEFAULT_RUNTIME_HASHES,
    finalize,
    read_json,
    sha256_file,
    verify_run,
    write_json,
)
from run_vta_p7r158_bitstream_isolated_capacity import reload_frozen_bitstream


HERE = Path(__file__).resolve().parent
ROOT = HERE / "report_out" / "stage_tile_cotuning" / "c3_dma_residency_autotune"
P7 = ROOT / "07_grouped_holdout"
DEFAULT_CONTRACT = P7 / "20260912_p7r162_y04_search_confirmation_contract_run01"
DEFAULT_FAILED_POOL = P7 / "20260912_p7r163_y04_search_confirmation_board_run01"
DEFAULT_OUTPUT = P7 / "20260912_p7r164_y04_post_reload_canary_run01"
SEEDS = (0, 20250901, 20260910)


def read_jsonl(path):
    return [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run_loaded_session(host, port, directory, candidates, target, health):
    remote = rpc.connect(host, port, session_timeout=120)
    inventory, functions = runtime_inventory(remote)
    device = remote.ext_dev(0)
    health_module = load_ephemeral(remote, directory / (health["candidate_id"] + ".so"))
    loaded = {
        row["candidate_id"]: load_ephemeral(
            remote, directory / (row["candidate_id"] + ".so")
        )
        for row in candidates
    }
    # Match P7R163 exactly: health buffers first, then one reusable Y04 set.
    health_buffers = allocate_reusable_buffers(device, health["identity"]["workload"])
    target_buffers = allocate_reusable_buffers(device, target["identity"]["workload"])
    health_rows = []
    for seed in SEEDS:
        row = profiled_reused_call(
            health_module["main"], health_buffers, health["identity"]["workload"],
            seed, functions,
        )
        health_rows.append(row)
        if not row.get("correct"):
            break
    target_rows = []
    if len(health_rows) == 3 and all(row.get("correct") for row in health_rows):
        for seed in SEEDS:
            row = profiled_reused_call(
                loaded[target["candidate_id"]]["main"], target_buffers,
                target["identity"]["workload"], seed, functions,
            )
            target_rows.append(row)
            if not row.get("correct"):
                break
    del loaded, target_buffers, health_buffers, health_module, functions, device, remote
    gc.collect()
    return {"inventory": inventory, "health_rows": health_rows, "target_rows": target_rows}


def run_health_only(host, port, directory, health):
    remote = rpc.connect(host, port, session_timeout=120)
    inventory, functions = runtime_inventory(remote)
    device = remote.ext_dev(0)
    module = load_ephemeral(remote, directory / (health["candidate_id"] + ".so"))
    buffers = allocate_reusable_buffers(device, health["identity"]["workload"])
    rows = []
    for seed in SEEDS:
        row = profiled_reused_call(
            module["main"], buffers, health["identity"]["workload"], seed, functions,
        )
        rows.append(row)
        if not row.get("correct"):
            break
    del buffers, module, functions, device, remote
    gc.collect()
    return {"inventory": inventory, "rows": rows}


def health_pass(rows):
    return len(rows) == 3 and all(row.get("correct") for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--failed-pool-dir", type=Path, default=DEFAULT_FAILED_POOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--ssh-known-hosts", type=Path, required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    args = parser.parse_args()

    output = args.output_dir
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable output {}".format(output))
    output.mkdir(parents=True)
    board = Board(args.host, args.ssh_known_hosts)
    active_rpc = None
    error = None
    try:
        contract_ledger = verify_run(args.contract_dir)
        failed_ledger = verify_run(args.failed_pool_dir)
        board_contract = read_json(args.contract_dir / "board_collection_contract.json")
        prior = read_jsonl(args.failed_pool_dir / "correctness.jsonl")
        prior_failure = read_json(args.failed_pool_dir / "failure.json")
        if len(prior) != 24 or any(row["status"] != "failed" for row in prior):
            raise ValueError("P7R163 is not the complete 24/24 failed correctness pool")
        if prior_failure.get("timing_samples_completed") != 0:
            raise ValueError("P7R163 unexpectedly contains timing labels")
        candidates = board_contract["candidates"]
        by_id = {row["candidate_id"]: row for row in candidates}
        target = next(
            by_id[candidate_id]
            for candidate_id in board_contract["candidate_order_for_unbiased_full_label_collection"]
            if by_id[candidate_id]["public_mode"] == "original"
        )
        health = board_contract["health_canary"]
        target_prior = next(row for row in prior if row["candidate_id"] == target["candidate_id"])
        before = board.state()
        if before.get("FPGA") != "operating" or before.get("UDMABUF") != "201326592":
            raise RuntimeError("FPGA/u-dma-buf preflight failed")
        if before.get("STORAGE_ERRORS"):
            raise RuntimeError("current boot has EXT4/mmc errors")
        if board.runtime_hashes(args.default_runtime) != EXPECTED_DEFAULT_RUNTIME_HASHES:
            raise RuntimeError("default runtime hash mismatch")
        original_rpc = board.rpc_state()
        if original_rpc.get("CWD") != args.default_runtime:
            raise RuntimeError("default RPC is not active")

        contract = {
            "schema": "c3_p7r164_y04_post_reload_canary_contract_v1",
            "status": "frozen_before_post_reload_retest",
            "source_contract": {
                "path": str(args.contract_dir.resolve()),
                "artifact_ledger_sha256": contract_ledger,
            },
            "source_failed_pool": {
                "path": str(args.failed_pool_dir.resolve()),
                "artifact_ledger_sha256": failed_ledger,
                "result": "24/24 identities failed seed0 FPGA correctness; zero timing labels",
            },
            "selection_rule": "first original identity in the pre-registered unbiased board order",
            "target": target,
            "target_prior_observation": target_prior,
            "health_canary": health,
            "module_load_policy": "load all 24 frozen Y04 modules before executing target",
            "buffer_policy": "P7R163 layout: health output/data/weight then Y04 output/data/weight",
            "bitstream_policy": "reload exact frozen bitstream before target and again before restoration",
            "seeds": list(SEEDS),
            "performance_measurement": "forbidden",
            "board": before,
        }
        write_json(output / "contract.json", contract)
        write_json(output / "pre_observation_hashes.json", {
            "contract.json": sha256_file(output / "contract.json")
        })

        with tempfile.TemporaryDirectory(prefix="c3_p7r164_cross_") as temporary:
            directory = Path(temporary)
            certificates = {"health": build_and_export(health, vta.get_env(), directory)}
            bindings = board_contract["qualification_bindings"]
            for candidate in candidates:
                certificates[candidate["candidate_id"]] = build_and_export(
                    candidate, vta.get_env(), directory,
                    bindings[candidate["candidate_id"]]["tir_sha256"],
                )
            write_json(output / "cross_compile_certificates.json", certificates)

            board.stop_rpc(original_rpc)
            write_json(output / "reload_before_target.json", reload_frozen_bitstream(board))
            active_rpc = board.start_rpc(args.default_runtime, "p7r164_target.log", None)
            observation = run_loaded_session(
                args.host, args.port, directory, candidates, target, health
            )
            write_json(output / "post_reload_observation.json", observation)
            if not health_pass(observation["health_rows"]):
                raise RuntimeError("post-reload pre-target W05 health failed")
            board.stop_rpc(active_rpc)
            active_rpc = None

            write_json(output / "reload_before_restore.json", reload_frozen_bitstream(board))
            active_rpc = board.start_rpc(args.default_runtime, "p7r164_restored.log", None)
            restored_health = run_health_only(args.host, args.port, directory, health)
            write_json(output / "restored_health.json", restored_health)
            if not health_pass(restored_health["rows"]):
                raise RuntimeError("restored W05 health failed")

        target_rows = observation["target_rows"]
        passed = len(target_rows) == 3 and all(row.get("correct") for row in target_rows)
        after = board.state()
        if after.get("BOOT") != before.get("BOOT") or after.get("STORAGE_ERRORS"):
            raise RuntimeError("board postflight changed")
        summary = {
            "schema": "c3_p7r164_y04_post_reload_canary_summary_v1",
            "status": "passed_after_reload" if passed else "failed_after_reload",
            "target_candidate_id": target["candidate_id"],
            "target_family_id": target["family_id"],
            "target_mode": target["public_mode"],
            "selection_used_post_failure_magnitude": False,
            "prior_status": target_prior["status"],
            "post_reload_seed_checks": len(target_rows),
            "post_reload_correct": passed,
            "post_reload_first_seed": target_rows[0] if target_rows else None,
            "health_before_target": "3/3",
            "health_after_second_reload": "3/3",
            "bitstream_reloaded_twice": True,
            "performance_labels_collected": False,
            "default_rpc_restored": True,
            "claim_boundary": "one pre-registered Y04 original diagnostic; not a complete-pool rerun",
        }
        write_json(output / "board_postflight.json", after)
        write_json(output / "summary.json", summary)
        (output / "STATUS.md").write_text(
            "# P7R164 Y04 post-reload canary\n\n"
            "- Target: first original in the frozen unbiased board order.\n"
            "- Post-reload result: `{}`.\n"
            "- W05 health: 3/3 before target and 3/3 after the second bitstream reload.\n"
            "- No latency label collected.\n".format(summary["status"]),
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
                board.start_rpc(args.default_runtime, "p7r164_emergency_restore.log", None)
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
