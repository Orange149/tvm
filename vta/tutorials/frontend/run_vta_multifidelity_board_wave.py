#!/usr/bin/env python3
"""Cross-compile and clean-start execute one frozen multi-fidelity board wave."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

from tvm import rpc
import vta

from prepare_vta_p7r120_yolo_rpc_contract import build_and_export, write_json
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory
import run_vta_p7r132_y00_search_confirmation as board


SCHEMA = "c3_vta_multifidelity_live_board_wave_v1"
SEEDS = board.SEEDS
ROUNDS = board.ROUNDS
HERE = Path(__file__).resolve().parent
P7 = (HERE / "report_out" / "stage_tile_cotuning" /
      "c3_dma_residency_autotune" / "07_grouped_holdout")
DEFAULT_HEALTH_CONTRACT = (
    P7 / "20260912_p7r162_y04_search_confirmation_contract_run01" /
    "board_collection_contract.json"
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def verify(directory):
    directory = Path(directory)
    ledger = read_json(directory / "artifact_hashes.json")
    for name, expected in ledger["artifacts"].items():
        if board.sha256(directory / name) != expected:
            raise ValueError("artifact hash mismatch: " + str(directory / name))
    return board.sha256(directory / "artifact_hashes.json")


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite immutable board wave " + str(output))
    prefix_dir = Path(args.local_prefix_dir or args.full_local_dir)
    prefix_ledger = verify(prefix_dir)
    if args.full_local_dir:
        if args.prior_board_wave is None:
            raise ValueError("--prior-board-wave is required for oracle completion")
        prior_board_dir = Path(args.prior_board_wave)
        prior_board_binding = {
            "path": str(prior_board_dir.resolve()),
            "ledger_sha256": verify(prior_board_dir),
        }
        all_candidates = read_jsonl(prefix_dir / "candidates_v2.jsonl")
        fsim = {row["candidate_id"]: row
                for row in read_jsonl(prefix_dir / "fsim_results.jsonl")}
        candidates = [row for row in all_candidates if fsim[row["candidate_id"]]["status"] == "passed"]
        order = [row["candidate_id"] for row in sorted(candidates, key=lambda row: hashlib.sha256(
            ("y05-oracle-completion:" + row["candidate_id"]).encode()
        ).hexdigest())]
        wave = {
            "history_head_sha256": None,
            "service_scores": {},
            "status": "full_local_fsim_pass_pool_for_oracle_completion",
        }
        collection_status = "oracle_completion_after_first_wave_labels_exposed"
    else:
        if args.prior_board_wave is not None:
            raise ValueError("--prior-board-wave applies only to --full-local-dir")
        prior_board_binding = None
        wave = read_json(prefix_dir / "board_wave.json")
        if wave["status"] not in (
            "frozen_after_live_local_prefix_before_cross_or_board",
            "frozen_after_adaptive_live_local_prefix_before_cross_or_board",
        ):
            raise ValueError("board wave is not pristine")
        candidates = wave["candidate_records"]
        order = wave["candidate_order"]
        collection_status = "prospective_first_online_board_wave"
    by_id = {row["candidate_id"]: row for row in candidates}
    static = {row["candidate_id"]: row
              for row in read_jsonl(prefix_dir / "static_results.jsonl")}
    health_source = Path(args.health_contract)
    health = read_json(health_source)["health_canary"]
    execution = {
        "schema": SCHEMA,
        "status": "frozen_before_cross_compile_or_board_contact",
        "local_prefix": {"path": str(prefix_dir.resolve()), "ledger_sha256": prefix_ledger,
                         "history_head_sha256": wave["history_head_sha256"]},
        "health_contract": {"path": str(health_source.resolve()),
                            "sha256": board.sha256(health_source)},
        "candidate_order": order,
        "candidate_service_scores": wave["service_scores"],
        "collection_status": collection_status,
        "prior_board_wave_with_exposed_labels": prior_board_binding,
        "seeds": list(SEEDS),
        "timing_rounds": ROUNDS,
        "clean_start": True,
        "buffer_allocation_order": ["output", "data", "weight"],
        "failure_policy": "health failure stops; candidate first error stops that identity; timed mismatch stops",
        "source_sha256": {str(Path(__file__).resolve()): board.sha256(__file__),
                          str(Path(board.__file__).resolve()): board.sha256(board.__file__)},
    }
    output.mkdir(parents=True)
    write_json(output / "execution_contract.json", execution)
    write_json(output / "pre_observation_hashes.json", {"artifacts": {
        "execution_contract.json": board.sha256(output / "execution_contract.json")
    }})
    temporary = None
    correctness_rows = []
    timing_rows = []
    health_rows = []
    try:
        env = vta.get_env()
        if env.TARGET != "axu5evb":
            raise RuntimeError("board wave requires TARGET=axu5evb")
        temporary = tempfile.TemporaryDirectory(prefix="c3_multifidelity_wave_")
        directory = Path(temporary.name)
        started = time.perf_counter()
        health_certificate = build_and_export(health, env, directory)
        health_certificate["cross_compile_wall_ms"] = (time.perf_counter() - started) * 1000
        certificates = {"health_canary": health_certificate}
        for cid in order:
            started = time.perf_counter()
            certificate = build_and_export(by_id[cid], env, directory, static[cid]["tir_sha256"])
            certificate["cross_compile_wall_ms"] = (time.perf_counter() - started) * 1000
            certificates[cid] = certificate
        write_json(output / "cross_certificates.json", certificates)
        write_json(output / "pre_rpc_hashes.json", {"artifacts": {
            "execution_contract.json": board.sha256(output / "execution_contract.json"),
            "cross_certificates.json": board.sha256(output / "cross_certificates.json"),
        }})

        control = board.CleanStartBoard(args.host, args.ssh_known_hosts)
        before = control.state()
        if before.get("FPGA") != "operating" or before.get("UDMABUF") != "201326592":
            raise RuntimeError("FPGA/u-dma-buf preflight failed")
        if before.get("STORAGE_ERRORS"):
            raise RuntimeError("storage errors found before board wave")
        if control.runtime_hashes(args.default_runtime) != board.EXPECTED_DEFAULT_RUNTIME_HASHES:
            raise RuntimeError("default runtime hash mismatch")
        prior_rpc = control.rpc_state()
        if prior_rpc.get("CWD") != args.default_runtime:
            raise RuntimeError("default RPC is not active")
        control.stop_rpc(prior_rpc)
        reload_result = control.reload_frozen_bitstream()
        fresh_rpc = control.start_rpc(args.default_runtime, "p7r_multifidelity_wave.log")
        write_json(output / "clean_board_start.json", {
            "status": "passed", "before": before, "stopped_rpc": prior_rpc,
            "bitstream_reload": reload_result, "fresh_rpc": fresh_rpc,
            "target_allocation_before_fresh_rpc": False,
        })

        remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
        inventory, functions = runtime_inventory(remote)
        write_json(output / "rpc_inventory.json", inventory)
        device = remote.ext_dev(0)
        health_module = load_ephemeral(remote, directory / (health["candidate_id"] + ".so"))
        modules = {cid: load_ephemeral(remote, directory / (cid + ".so")) for cid in order}
        health_buffers = board.allocate_reusable_buffers(device, health["identity"]["workload"])
        target_buffers = board.allocate_reusable_buffers(device, candidates[0]["identity"]["workload"])
        for seed in SEEDS:
            started = time.perf_counter()
            row = board.profiled_reused_call(
                health_module["main"], health_buffers, health["identity"]["workload"],
                seed, functions,
            )
            row["host_wall_ms"] = (time.perf_counter() - started) * 1000
            health_rows.append(row)
            if not row.get("correct"):
                break
        health_pass = len(health_rows) == 3 and all(row.get("correct") for row in health_rows)
        write_json(output / "health_gate.json", {
            "status": "passed" if health_pass else "failed_stop", "seeds": health_rows
        })
        if not health_pass:
            raise RuntimeError("health gate failed before Y05 dispatch")

        with (output / "correctness.jsonl").open("x", encoding="utf-8") as stream:
            for position, cid in enumerate(order):
                rows = []
                for seed in SEEDS:
                    started = time.perf_counter()
                    row = board.profiled_reused_call(
                        modules[cid]["main"], target_buffers,
                        by_id[cid]["identity"]["workload"], seed, functions,
                    )
                    row["host_wall_ms"] = (time.perf_counter() - started) * 1000
                    rows.append(row)
                    if not row.get("correct"):
                        break
                result = {
                    "candidate_id": cid, "position": position,
                    "family_id": by_id[cid]["family_id"],
                    "public_mode": by_id[cid]["public_mode"],
                    "status": "passed" if len(rows) == 3 and all(x.get("correct") for x in rows)
                    else "failed",
                    "seeds": rows,
                }
                correctness_rows.append(result)
                stream.write(json.dumps(result, sort_keys=True) + "\n"); stream.flush()
                print("correctness {}/{} {}".format(position + 1, len(order), result["status"]),
                      flush=True)
        correct_ids = [row["candidate_id"] for row in correctness_rows
                       if row["status"] == "passed"]
        if not correct_ids:
            raise RuntimeError("board wave has no FPGA-correct candidate")
        orders = board.timing_orders(correct_ids)
        write_json(output / "timing_orders.json", orders)
        with (output / "timing.jsonl").open("x", encoding="utf-8") as stream:
            for round_index, round_order in enumerate(orders):
                for position, cid in enumerate(round_order):
                    started = time.perf_counter()
                    row = board.timed_reused_call(
                        modules[cid], device, target_buffers,
                        by_id[cid]["identity"]["workload"], 0, functions,
                    )
                    row["host_wall_ms"] = (time.perf_counter() - started) * 1000
                    row.update(candidate_id=cid, round=round_index, position=position)
                    timing_rows.append(row)
                    stream.write(json.dumps(row, sort_keys=True) + "\n"); stream.flush()
                print("timing round {}/{}".format(round_index + 1, ROUNDS), flush=True)
        timing_summary = board.summarize_timing(timing_rows, correct_ids)
        write_json(output / "timing_summary.json", timing_summary)
        write_json(output / "summary.json", {
            "schema": SCHEMA,
            "status": "live_board_wave_complete",
            "candidate_count": len(order),
            "correct_candidate_count": len(correct_ids),
            "correctness_invocations": sum(len(row["seeds"]) for row in correctness_rows),
            "timing_samples": len(timing_rows),
            "all_timed_calls_correct": True,
            "wave_best_candidate_id": timing_summary["pool_oracle_candidate_id"],
            "wave_best_latency_ms": timing_summary["pool_oracle_latency_ms"],
            "full_pool_oracle_known": bool(args.full_local_dir),
            "claim_boundary": (
                "full FSim-pass oracle completion after first-wave labels; use only to reveal the "
                "pool oracle and never as prospective ordering evidence"
                if args.full_local_dir else
                "prospective first board wave only; not yet a full-pool oracle claim"
            ),
        })
        (output / "command.txt").write_text(
            " ".join([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]) + "\n",
            encoding="utf-8",
        )
        board.finalize(output)
    except Exception as error:
        write_json(output / "failure.json", {
            "status": "failed_closed", "exception_type": type(error).__name__,
            "message": (str(error) or type(error).__name__)[:4000],
            "traceback": traceback.format_exc()[-12000:],
            "health_seed_checks": len(health_rows),
            "candidate_identities_completed": len(correctness_rows),
            "timing_samples_completed": len(timing_rows),
        })
        board.finalize(output)
        raise
    finally:
        if temporary is not None:
            temporary.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--local-prefix-dir", type=Path)
    source.add_argument("--full-local-dir", type=Path)
    parser.add_argument("--prior-board-wave", type=Path)
    parser.add_argument("--health-contract", type=Path, default=DEFAULT_HEALTH_CONTRACT)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=120)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    parser.add_argument("--ssh-known-hosts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
