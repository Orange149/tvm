#!/usr/bin/env python3
"""Run the label-free H1 DMA-multifidelity path from clean T0 through full graph."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from types import SimpleNamespace

from tvm import rpc
import vta

from analyze_vta_fused_program_service_proxy_norefit import verify_artifacts_compatible
from build_vta_c3_resnet18_fullgraph_programs import build_routes
from build_vta_resnet50_fused_program_pool import build_stock, fused_features
from build_vta_resnet50_relay_residency_dispatch_pair import (
    make_relay_testing_resnet_program,
)
from run_vta_p7r120_yolo_rpc_board import load_ephemeral, runtime_inventory
from run_vta_p7r132_y00_search_confirmation import (
    CleanStartBoard,
    EXPECTED_DEFAULT_RUNTIME_HASHES,
    allocate_reusable_buffers,
    profiled_reused_call,
    timed_reused_call,
)
from run_vta_resnet50_fused_program_pareto_board import assert_board_state, run_one
from run_vta_resnet50_relay_residency_dispatch_pair_board import (
    semantic_params_hash,
    sha256,
    write_json,
)


PROCESS_STARTED = time.perf_counter()
PROCESS_STARTED_WALL = time.time()
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SEEDS = (0, 20250901, 20260910)
RPC_IDLE_TIMEOUT_SECONDS = 15.0


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")
        stream.flush()


def parse_rpc_processes(stdout):
    """Parse PID/PPID/CWD rows emitted by the narrow board probe."""
    rows = []
    for line in stdout.splitlines():
        if not line.startswith("RPC_PROCESS="):
            continue
        fields = line.split("=", 1)[1].split("|", 2)
        if len(fields) != 3 or not all(fields[:2]):
            raise RuntimeError("malformed RPC process probe row")
        rows.append({"PID": fields[0], "PPID": fields[1], "CWD": fields[2]})
    return rows


class ParentAwareCleanStartBoard(CleanStartBoard):
    """Identify the persistent RPC parent, never its per-connection child."""

    def rpc_processes(self):
        command = r'''
for pid in $(pidof tvm_rpc 2>/dev/null || true); do
  ppid=$(awk '/^PPid:/{print $2}' /proc/$pid/status)
  cwd=$(readlink /proc/$pid/cwd)
  echo RPC_PROCESS=$pid\|$ppid\|$cwd
done
'''
        return parse_rpc_processes(self.run(command).stdout)

    def rpc_state(self):
        processes = self.rpc_processes()
        if not processes:
            return {"PID": "", "CWD": ""}
        parents = [row for row in processes if row["PPID"] == "1"]
        if len(parents) != 1:
            raise RuntimeError("expected exactly one persistent RPC parent")
        return {"PID": parents[0]["PID"], "CWD": parents[0]["CWD"]}

    def wait_idle_rpc(self, directory, timeout_seconds=RPC_IDLE_TIMEOUT_SECONDS):
        deadline = time.monotonic() + timeout_seconds
        last = []
        while time.monotonic() < deadline:
            last = self.rpc_processes()
            parents = [
                row for row in last
                if row["PPID"] == "1" and row["CWD"] == directory
            ]
            if len(last) == 1 and len(parents) == 1:
                return {"PID": parents[0]["PID"], "CWD": parents[0]["CWD"]}
            time.sleep(0.1)
        raise RuntimeError(
            "RPC did not become an idle single-parent server: {}".format(last)
        )


def run_phase(name, command, output, phases, t0):
    started = time.perf_counter()
    log = output / (name + ".log")
    with log.open("x", encoding="utf-8") as stream:
        stream.write("COMMAND " + " ".join(str(value) for value in command) + "\n")
        stream.flush()
        completed = subprocess.run(
            [str(value) for value in command],
            cwd=REPO,
            env=os.environ.copy(),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    row = {
        "phase": name,
        "seconds": time.perf_counter() - started,
        "finished_t0_seconds": time.perf_counter() - t0,
        "returncode": completed.returncode,
        "log": log.name,
    }
    phases.append(row)
    write_json(output / "phase_timeline.json", phases)
    if completed.returncode:
        raise RuntimeError("{} failed; see {}".format(name, log))


def clean_start(args, log_name):
    board = ParentAwareCleanStartBoard(args.host, args.ssh_known_hosts)
    before = board.state()
    assert_board_state(before)
    if board.runtime_hashes(args.default_runtime) != EXPECTED_DEFAULT_RUNTIME_HASHES:
        raise RuntimeError("default runtime hash mismatch")
    prior = board.wait_idle_rpc(args.default_runtime)
    if not prior.get("PID") or prior.get("CWD") != args.default_runtime:
        raise RuntimeError("healthy default RPC is required")
    board.stop_rpc(prior)
    bitstream = board.reload_frozen_bitstream()
    fresh = board.start_rpc(args.default_runtime, log_name)
    return board, {
        "before": before,
        "stopped_rpc": prior,
        "bitstream": bitstream,
        "fresh_rpc": fresh,
    }


def dma_key(candidate):
    totals = candidate["dma_features"]["totals"]
    return (
        int(totals.get("load_buffer_2d_bytes", 0))
        + int(totals.get("store_buffer_2d_bytes", 0)),
        int(totals.get("load_buffer_2d_calls", 0))
        + int(totals.get("store_buffer_2d_calls", 0)),
        candidate["candidate_id"],
    )


def profile_cost(rows):
    total = Counter()
    for row in rows:
        profile = row.get("runtime_profile_complete") or {}
        total["logical_load_bytes"] += int(profile.get("load_buffer_2d_bytes", 0))
        total["logical_store_bytes"] += int(profile.get("store_buffer_2d_bytes", 0))
        total["logical_dma_calls"] += int(profile.get("load_buffer_2d_calls", 0))
        total["logical_dma_calls"] += int(profile.get("store_buffer_2d_calls", 0))
        total["fpga_kernel_invocations"] += int(profile.get("driver_run_calls", 0))
    return dict(total)


def candidate_reset(board, args, position):
    prior = board.wait_idle_rpc(args.default_runtime)
    if prior.get("PID"):
        board.stop_rpc(prior)
    bitstream = board.reload_frozen_bitstream()
    fresh = board.start_rpc(
        args.default_runtime, "r18_h1_ours_candidate_{:03d}.log".format(position)
    )
    return {"stopped_rpc": prior, "bitstream": bitstream, "fresh_rpc": fresh}


def online_dma_search(args, board, pool_dir, cross_dir, output, t0):
    candidates = sorted(read_jsonl(pool_dir / "candidates.jsonl"), key=dma_key)
    candidates = candidates[: min(args.candidate_budget, len(candidates))]
    binary_dir = cross_dir / "binaries"
    timeline_path = output / "online_timeline.jsonl"
    timeline_path.write_text("", encoding="utf-8")
    results = []
    all_calls = []
    for position, candidate in enumerate(candidates, 1):
        started = time.perf_counter()
        reset = candidate_reset(board, args, position)
        remote = None
        functions = None
        device = None
        module = None
        buffers = None
        correctness = []
        timings = []
        try:
            remote = rpc.connect(args.host, args.port, session_timeout=args.session_timeout)
            _, functions = runtime_inventory(remote)
            device = remote.ext_dev(0)
            module = load_ephemeral(
                remote, binary_dir / (candidate["candidate_id"] + ".so")
            )
            buffers = allocate_reusable_buffers(device, candidate["identity"]["workload"])
            for seed in SEEDS:
                call = profiled_reused_call(
                    module["main"],
                    buffers,
                    candidate["identity"]["workload"],
                    seed,
                    functions,
                )
                correctness.append(call)
                all_calls.append(call)
                if call.get("status") == "execution_failed":
                    raise RuntimeError("RPC/device interruption during candidate correctness")
                if not call.get("correct"):
                    break
            passed = len(correctness) == len(SEEDS) and all(
                row.get("correct") for row in correctness
            )
            if passed:
                for round_index in range(5):
                    call = timed_reused_call(
                        module,
                        device,
                        buffers,
                        candidate["identity"]["workload"],
                        SEEDS[round_index % len(SEEDS)],
                        functions,
                    )
                    call["round"] = round_index
                    timings.append(call)
                    all_calls.append(call)
            row = {
                "dispatch": position,
                "candidate_id": candidate["candidate_id"],
                "family_id": candidate["family_id"],
                "public_mode": candidate["residence_mode"],
                "dma_key": list(dma_key(candidate)[:2]),
                "status": "passed" if passed else "rejected_fail_closed",
                "correctness": correctness,
                "timings": timings,
                "median_latency_ms": (
                    statistics.median(item["latency_ms"] for item in timings)
                    if timings else None
                ),
                "candidate_wall_seconds": time.perf_counter() - started,
                "cumulative_t0_seconds": time.perf_counter() - t0,
                "clean_reset": reset,
            }
            results.append(row)
            append_jsonl(timeline_path, row)
            append_jsonl(
                output / "results.jsonl",
                {"record_type": "operator_candidate", **row},
            )
        except Exception as error:
            row = {
                "dispatch": position,
                "candidate_id": candidate["candidate_id"],
                "family_id": candidate["family_id"],
                "public_mode": candidate["residence_mode"],
                "dma_key": list(dma_key(candidate)[:2]),
                "status": "session_invalidated",
                "correctness": correctness,
                "timings": timings,
                "median_latency_ms": None,
                "candidate_wall_seconds": time.perf_counter() - started,
                "cumulative_t0_seconds": time.perf_counter() - t0,
                "clean_reset": reset,
                "failure": {
                    "exception_type": type(error).__name__,
                    "message": (str(error) or type(error).__name__)[:4000],
                },
            }
            results.append(row)
            append_jsonl(timeline_path, row)
            append_jsonl(
                output / "results.jsonl",
                {"record_type": "operator_candidate", **row},
            )
            raise
        finally:
            # Device, module, packed functions and NDArrays all retain the RPC
            # session.  Dropping only ``remote`` leaves connection children
            # alive and makes the next clean restart target the wrong process.
            buffers = None
            module = None
            functions = None
            device = None
            remote = None
            gc.collect()
        print(
            "online {}/{} {}".format(position, len(candidates), results[-1]["status"]),
            flush=True,
        )
    passed = [row for row in results if row["status"] == "passed"]
    if not passed:
        raise RuntimeError("DMA-multifidelity search produced no FPGA-correct candidate")
    selected_result = min(
        passed, key=lambda row: (row["median_latency_ms"], row["candidate_id"])
    )
    selected = next(
        row for row in candidates if row["candidate_id"] == selected_result["candidate_id"]
    )
    write_json(
        output / "online_summary.json",
        {
            "candidate_budget": args.candidate_budget,
            "effective_dispatches": len(results),
            "fpga_correct": len(passed),
            "selected_candidate_id": selected["candidate_id"],
            "selected_latency_ms": selected_result["median_latency_ms"],
            "resource_cost": profile_cost(all_calls),
            "results": results,
        },
    )
    return selected, results, all_calls


def finish(output):
    write_json(
        output / "artifact_hashes.json",
        {
            "artifacts": {
                str(path.relative_to(output)): sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "artifact_hashes.json"
            },
            "source_sha256": sha256(Path(__file__).resolve()),
        },
    )


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    source_dir = Path(args.source_contract).resolve()
    verify_artifacts_compatible(source_dir)
    source = read_json(source_dir / "contract.json")
    if source.get("status") != "frozen_before_resnet18_fullgraph_build_fpga_or_latency":
        raise RuntimeError("R18 source contract is not pristine")
    contract = {
        "schema": "c3_resnet18_h1_from_scratch_ours_v1",
        "status": "frozen_before_W0_board_observation",
        "strategy": "label_free_maxmin_then_real_lower_fsim_dma_order_fpga",
        "workload_id": "R18-H1",
        "candidate_budget": args.candidate_budget,
        "correctness_seeds": list(SEEDS),
        "local_expansion_threshold": 12,
        "old_target_performance_labels_used": False,
        "W0_definition": "host process begins before board preflight",
        "T0_definition": "clean bitstream and fresh default RPC ready before candidate generation",
        "T1_definition": "selected route completes full-graph three-input correctness and seven paired rounds",
        "source_contract_manifest_sha256": sha256(source_dir / "artifact_hashes.json"),
    }
    write_json(output / "contract.json", contract)
    (output / "timeline.jsonl").write_text("", encoding="utf-8")
    (output / "results.jsonl").write_text("", encoding="utf-8")
    board = None
    phases = []
    t0 = None
    try:
        board, clean = clean_start(args, "r18_h1_ours_t0.log")
        write_json(output / "clean_start.json", clean)
        t0 = time.perf_counter()
        python = sys.executable
        initial = output / "candidate_batch0"
        run_phase(
            "candidate_generation_batch0",
            [
                python,
                HERE / "prepare_vta_c3_resnet18_literature_candidates.py",
                "--workload-id",
                "R18-H1",
                "--output-dir",
                initial,
            ],
            output,
            phases,
            t0,
        )
        pools = [initial]
        qualifications = []
        cumulative = 0
        batch = 0
        while cumulative < 12:
            pool = pools[-1]
            qualification = output / "local_qualification_batch{}".format(batch)
            run_phase(
                "local_qualification_batch{}".format(batch),
                [
                    python,
                    HERE / "run_vta_c3_resnet18_local_qualification.py",
                    "--input-dir",
                    pool,
                    "--workload-id",
                    "R18-H1",
                    "--phase",
                    "all",
                    "--fsim-timeout",
                    "300",
                    "--output-dir",
                    qualification,
                ],
                output,
                phases,
                t0,
            )
            qualifications.append(qualification)
            cumulative += int(
                read_json(qualification / "summary.json")["geometries"]["R18-H1"][
                    "local_fsim_legal_identities"
                ]
            )
            if cumulative >= 12:
                break
            batch += 1
            if batch > 5:
                raise RuntimeError("H1 local expansion exceeded preregistered safety cap")
            extension = output / "candidate_batch{}".format(batch)
            command = [
                python,
                HERE / "extend_vta_c3_resnet18_literature_candidates.py",
                "--preregistered-dir",
                initial,
                "--workload-id",
                "R18-H1",
                "--prior-tiles",
                str(24 * batch),
            ]
            for path in qualifications:
                command.extend(("--qualification-dir", path))
            for path in pools:
                command.extend(("--prior-pool-dir", path))
            command.extend(("--output-dir", extension))
            run_phase(
                "candidate_generation_batch{}".format(batch),
                command,
                output,
                phases,
                t0,
            )
            pools.append(extension)

        pool = output / "h1_board_pool"
        freeze_command = [
            python,
            HERE / "freeze_vta_c3_resnet18_board_pool.py",
            "--workload-id",
            "R18-H1",
        ]
        for path in pools:
            freeze_command.extend(("--pool-dir", path))
        for path in qualifications:
            freeze_command.extend(("--qualification-dir", path))
        freeze_command.extend(("--output-dir", pool))
        run_phase("freeze_h1_pool", freeze_command, output, phases, t0)
        cross = output / "h1_cross_compile"
        run_phase(
            "cross_compile_h1_pool",
            [
                python,
                HERE / "cross_compile_vta_c3_resnet18_board_pool.py",
                "--board-pool-dir",
                pool,
                "--output-dir",
                cross,
                "--timeout",
                "300",
            ],
            output,
            phases,
            t0,
        )
        online_started = time.perf_counter()
        selected, online_results, online_calls = online_dma_search(
            args, board, pool, cross, output, t0
        )
        phases.append(
            {
                "phase": "online_dma_search",
                "seconds": time.perf_counter() - online_started,
                "finished_t0_seconds": time.perf_counter() - t0,
                "returncode": 0,
            }
        )
        write_json(output / "phase_timeline.json", phases)

        env = vta.get_env()
        relay_program, params = make_relay_testing_resnet_program(env, 18, device_annot=True)
        fullgraph = output / "selected_fullgraph_build"
        fullgraph.mkdir()
        build_started = time.perf_counter()
        stock = build_stock(relay_program, params, fullgraph, env, target_mode="heterogeneous")
        stock_graph = read_json(fullgraph / "stock_reference" / "graph.json")
        stock_features = fused_features(stock, stock_graph, source["workload"])
        stock["fused_features"] = stock_features
        write_json(fullgraph / "stock_reference" / "build.json", stock)
        route = {
            "candidate_id": selected["candidate_id"],
            "family_id": selected["family_id"],
            "public_mode": selected["residence_mode"],
            "implementation_mode": selected["implementation_mode"],
            "identity": {
                "workload": selected["identity"]["workload"],
                "complete_config_entity": selected["complete_config_entity"],
                "public_mode": selected["residence_mode"],
                "implementation_mode": selected["implementation_mode"],
            },
        }
        selected_dir = fullgraph / "selected_candidate"
        built_route = build_routes([route], relay_program, params, selected_dir, env)
        if read_json(selected_dir / "graph.json") != stock_graph:
            raise RuntimeError("selected graph structure differs from stock")
        if semantic_params_hash(selected_dir / "params.bin") != semantic_params_hash(
            fullgraph / "stock_reference" / "params.bin"
        ):
            raise RuntimeError("selected graph parameter semantics differ from stock")
        features = fused_features(built_route, stock_graph, source["workload"])
        program = {
            "candidate_id": selected["candidate_id"],
            "family_id": selected["family_id"],
            "public_mode": selected["residence_mode"],
            "relative_dir": "selected_candidate",
            **features,
        }
        write_json(selected_dir / "program.json", program)
        phases.append(
            {
                "phase": "selected_fullgraph_build",
                "seconds": time.perf_counter() - build_started,
                "finished_t0_seconds": time.perf_counter() - t0,
                "returncode": 0,
            }
        )
        board_args = SimpleNamespace(
            host=args.host,
            port=args.port,
            session_timeout=args.session_timeout,
            default_runtime=args.default_runtime,
        )
        graph_started = time.perf_counter()
        graph_result = run_one(
            board,
            board_args,
            output,
            fullgraph,
            program,
            stock_features,
            0,
            input_spec={
                "shape": [env.BATCH, 3, 224, 224],
                "uniform_range": [0.0, 1.0],
                "all_outputs": True,
            },
        )
        phases.append(
            {
                "phase": "selected_fullgraph_correctness_and_timing",
                "seconds": time.perf_counter() - graph_started,
                "finished_t0_seconds": time.perf_counter() - t0,
                "returncode": 0,
            }
        )
        write_json(output / "phase_timeline.json", phases)
        if graph_result.get("status") != "passed":
            raise RuntimeError("selected full graph rejected fail closed")
        write_json(output / "fullgraph_result.json", graph_result)
        summary = {
            "schema": "c3_resnet18_h1_from_scratch_ours_result_v1",
            "status": "completed_W0_and_T0_to_T1",
            "workload_id": "R18-H1",
            "generated_batches": len(pools),
            "locally_qualified_candidates": cumulative,
            "candidate_budget": args.candidate_budget,
            "fpga_dispatches": len(online_results),
            "selected_candidate_id": selected["candidate_id"],
            "selected_mode": selected["residence_mode"],
            "operator_search_resource_cost": profile_cost(online_calls),
            "phase_timeline": phases,
            "W0_to_T1_seconds": time.perf_counter() - PROCESS_STARTED,
            "T0_to_T1_seconds": time.perf_counter() - t0,
            "W0_wall_unix": PROCESS_STARTED_WALL,
            "fullgraph_result": graph_result,
            "target_performance_labels_reused": False,
            "claim_boundary": (
                "One clean-start execution of the frozen method. Three independent runs per "
                "method are required before aggregate comparison; random-input equality is not accuracy."
            ),
        }
        write_json(output / "summary.json", summary)
        (output / "results.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in online_results),
            encoding="utf-8",
        )
        (output / "timeline.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in phases),
            encoding="utf-8",
        )
        (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
        finish(output)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    except Exception as error:
        invalid = {
            "schema": "c3_resnet18_h1_from_scratch_ours_failure_v1",
            "status": "invalid_entire_session_fail_closed",
            "exception_type": type(error).__name__,
            "message": (str(error) or type(error).__name__)[:4000],
            "traceback": traceback.format_exc()[-12000:],
            "W0_elapsed_seconds": time.perf_counter() - PROCESS_STARTED,
            "T0_elapsed_seconds": None if t0 is None else time.perf_counter() - t0,
            "may_splice_with_other_session": False,
        }
        write_json(
            output / "invalid_session.json",
            invalid,
        )
        write_json(
            output / "summary.json",
            {
                "schema": "c3_resnet18_h1_from_scratch_ours_result_v1",
                "status": invalid["status"],
                "session_spliced": False,
                "W0_to_failure_seconds": invalid["W0_elapsed_seconds"],
                "T0_to_failure_seconds": invalid["T0_elapsed_seconds"],
                "completed_phases": phases,
                "failure": {
                    "exception_type": invalid["exception_type"],
                    "message": invalid["message"],
                },
            },
        )
        timeline_rows = list(phases)
        timeline_rows.append(
            {
                "phase": "session_invalidated",
                "W0_elapsed_seconds": invalid["W0_elapsed_seconds"],
                "T0_elapsed_seconds": invalid["T0_elapsed_seconds"],
                "exception_type": invalid["exception_type"],
                "message": invalid["message"],
                "may_splice_with_other_session": False,
            }
        )
        (output / "timeline.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in timeline_rows),
            encoding="utf-8",
        )
        (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
        finish(output)
        raise


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-budget", type=int, default=13)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--session-timeout", type=int, default=600)
    parser.add_argument("--ssh-known-hosts", required=True)
    parser.add_argument("--default-runtime", default="/var/volatile/vta_c3_ram/runtime")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
