#!/usr/bin/env python3
"""New C1/C2 harness; leaves the sealed Node-B implementation unchanged.

prepare closes timing/runtime bindings without target labels. board executes
only an explicit frozen preparation, on an exclusive tmpfs RPC session.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import time

from run_vta_c3_reuse_supplement import (
    HERE, ROOT, SEEDS, MODES, BITSTREAM, read, write, append, seal, verify,
    file_hash, check_bindings, lower_candidate, cross, fsim, hardware,
)
from recover_vta_c3_reuse_cross import sdk_environment

RUNTIME = {
    "tvm_rpc": "20bbd3d896d124dd1127343f020c957fbb6034423af83433c2429955b6b316fa",
    "libtvm_runtime.so": "04e894baf305311315fd0c79d03ec2a6375b9591916e9c6f429cf3697c4457b2",
    "libvta.so": "eedfabb0630d58bf2eabaef9b4f2d2e4bfcdfa52a70503090f5608e79404ee5d",
}
HEALTH_DIR = ROOT / "vta/tutorials/frontend/report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout/20260912_p7r144_y00_layout_preserving_search_board_run01"


def bind_sources(parent):
    paths = [Path(__file__), HERE / "c3_reuse_timed_once.cc",
             HERE / "run_vta_p7r132_y00_search_confirmation.py",
             HERE / "run_vta_p7r120_yolo_rpc_board.py"]
    return {**parent["source_bindings"], **{str(p.resolve()): file_hash(p) for p in paths}}


def prepare(args, output, started):
    parent = Path(args.contract).resolve().parent
    frozen = verify(parent)
    check_bindings(frozen["source_bindings"])
    if frozen["phase"] != "freeze" or len(frozen["frozen_mode_orders"]) != 8:
        raise ValueError("requires final eight-round Node-B freeze")
    runtime_paths = {name: ROOT / "build_axu_aarch64" / name for name in RUNTIME}
    for name, path in runtime_paths.items():
        if file_hash(path) != RUNTIME[name]:
            raise ValueError("runtime binary changed: " + name)
    from prepare_vta_c3_resnet18_literature_candidates import candidate_row
    from run_vta_p7r132_y00_search_confirmation import verify_run
    health_ledger = verify_run(HEALTH_DIR)
    historical = read(HEALTH_DIR / "contract.json")["health_canary"]
    env, fingerprint = hardware()
    spec = dict(ci=256, co=256, height=14, width=14, kernel=3, stride=1)
    identity = historical["identity"]
    health = candidate_row("HEALTH", spec, identity["workload"], identity["complete_config_entity"],
                           575, "W05_HEALTH_575", "original", frozen["schedule_version"], fingerprint, env)
    health["identity"] = {"workload": identity["workload"],
                          "complete_config_entity": identity["complete_config_entity"]}
    health["local_qualification"] = lower_candidate(health, output)
    if health["local_qualification"]["status"] != "passed":
        raise ValueError("health probe lowering failed")
    health["fsim_qualification"] = fsim(health, health["local_qualification"], output,
                                          str(ROOT / "build/libvta_fsim.so"), args.timeout)
    if health["fsim_qualification"]["status"] != "passed":
        raise ValueError("health probe FSim failed")
    previous = os.environ.get("LD_LIBRARY_PATH")
    try:
        os.environ.pop("LD_LIBRARY_PATH", None)
        health["cross_qualification"] = cross(health, output, args)
    finally:
        if previous is not None:
            os.environ["LD_LIBRARY_PATH"] = previous
    if health["cross_qualification"]["status"] != "passed":
        raise ValueError("health probe cross failed")
    helper = output / "binaries/c3_reuse_timed_once.so"
    command = ["bash", "-c",
               'source "$1" >/dev/null && shift && exec $CXX "$@"',
               "c3-timer-build", args.sdk_environment, "-std=c++17", "-shared", "-fPIC",
               str(HERE / "c3_reuse_timed_once.cc"), "-I" + str(ROOT / "include"),
               "-I" + str(ROOT / "3rdparty/dlpack/include"),
               "-I" + str(ROOT / "3rdparty/dmlc-core/include"),
               "-L" + str(ROOT / "build_axu_aarch64"), "-ltvm_runtime", "-Wl,-rpath,$ORIGIN",
               "-o", str(helper)]
    compilation = subprocess.run(command, env=sdk_environment(), capture_output=True, text=True,
                                 check=False, timeout=args.timeout)
    (output / "logs/timer_compile.stderr.log").write_text(compilation.stderr)
    if compilation.returncode:
        raise RuntimeError("timer compiler failed: " + compilation.stderr[-4000:])
    bindings = bind_sources(frozen)
    contract = {
        "schema": "c3_reuse_board_v1", "phase": "board_prepared",
        "parent_directory": str(parent), "parent_manifest_sha256": file_hash(parent / "artifact_hashes.json"),
        "source_bindings": bindings,
        "runtime_paths": {n: str(p) for n, p in runtime_paths.items()}, "runtime_sha256": RUNTIME,
        "helper_sha256": file_hash(helper), "health": health,
        "health_historical_manifest_sha256": health_ledger,
        "board_contacted": False, "performance_labels_collected": False,
        "timing_method": "board steady_clock around one synchronous main call; module lookup excluded; output checked after each call",
        "timing_includes": "operator host instruction generation, VTA submission and waits; excludes RPC/network/input copies/output checks",
        "hidden_warmup_invocations": 0, "timing_invocations_per_observation": 1,
        "explicit_warmup_calls_per_correct_identity": 1,
        "profile_extra_invocations_per_correct_identity": 1,
        "correctness_seeds": list(SEEDS), "rounds": 8, "session_timeout_seconds": 7200,
        "controls": "independent W/resident,W/reload,IW/resident,IW/reload four-way Latin block per control family",
        "buffer_layout": ["output", "data", "weight"],
        "interruption_invalidates_whole_session": True,
        "reload_bitstream_before_T0": True,
    }
    write(output / "contract.json", contract)
    write(output / "summary.json", {"status": "C1_READY", "board_contacted": False,
                                    "main_identities": 44, "reload_controls": 4,
                                    "wall_seconds": time.monotonic() - started})
    check_bindings(bindings)


class Board:
    def __init__(self, args, runtime):
        self.args, self.runtime, self.pid = args, runtime, None
        self.opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=yes",
                     "-o", "UserKnownHostsFile=" + args.known_hosts,
                     "-o", "HostKeyAlgorithms=+ssh-rsa", "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa"]

    def ssh(self, command, timeout=25):
        return subprocess.run(["ssh", *self.opts, "root@" + self.args.host, command],
                              check=True, capture_output=True, text=True, timeout=timeout).stdout

    def state(self):
        script = (
            "cat /proc/sys/kernel/random/boot_id && "
            "cat /sys/class/fpga_manager/fpga0/state && "
            "cat /sys/class/u-dma-buf/udmabuf0/size && "
            "sha256sum /lib/firmware/vta_hpc.bit && "
            "ps && mount && dmesg"
        )
        raw = self.ssh(script)
        lines = raw.splitlines()
        return {"boot_id": lines[0], "fpga_state": lines[1], "udmabuf_size": int(lines[2]),
                "bitstream_sha256": lines[3].split()[0], "raw": raw}

    def deploy(self, paths):
        destination = shlex.quote(self.runtime)
        self.ssh("test ! -e " + destination + " && mkdir -p " + destination + " && test -d " + destination)
        subprocess.run(["scp", "-O", *self.opts, *[str(p) for p in paths.values()],
                        "root@" + self.args.host + ":" + self.runtime + "/"],
                       capture_output=True, text=True, check=True, timeout=60)
        observed = self.ssh("cd " + destination + " && sha256sum tvm_rpc libtvm_runtime.so libvta.so")
        hashes = {line.split()[1]: line.split()[0] for line in observed.splitlines()}
        if hashes != RUNTIME:
            raise ValueError("remote runtime attestation failed")
        return hashes

    def start(self):
        path = shlex.quote(self.runtime)
        script = (
            "(cd " + path + " && exec env LD_LIBRARY_PATH=" + path + " LD_PRELOAD=" + path + "/libvta.so "
            "TVM_NUM_THREADS=4 VTA_REPLAY_POLICY=disabled VTA_QUEUE_DIAGNOSTICS=0 VTA_RUNTIME_TRACE=0 "
            "./tvm_rpc server --host=0.0.0.0 --port=9090 --port-end=9091) >" + path + "/rpc.log 2>&1 < /dev/null & echo $!"
        )
        self.pid = int(self.ssh(script).strip())
        return self.pid

    def reload(self):
        result = self.ssh("test \"$(sha256sum /lib/firmware/vta_hpc.bit | cut -d' ' -f1)\" = " + BITSTREAM
                          + " && echo vta_hpc.bit > /sys/class/fpga_manager/fpga0/firmware"
                          + " && test \"$(cat /sys/class/fpga_manager/fpga0/state)\" = operating")
        return {"firmware": "vta_hpc.bit", "sha256": BITSTREAM, "state": "operating"}

    def stop(self):
        if self.pid is None:
            return
        # Kill only our recorded server, and only if its cwd is still ours.
        self.ssh("test \"$(readlink /proc/" + str(self.pid) + "/cwd)\" = " + shlex.quote(self.runtime)
                 + " && kill " + str(self.pid))


def validate_state(state, exclusive=False):
    import re
    if state["fpga_state"] != "operating" or state["udmabuf_size"] != 201326592 or state["bitstream_sha256"] != BITSTREAM:
        raise ValueError("board hardware/bitstream does not match frozen pool")
    if re.search(r"EXT4-fs.*(?:error|abort)|mmc.*(?:I/O error|timeout|error -)", state["raw"], re.I):
        raise ValueError("filesystem/MMC errors require diagnosis before execution")
    if "tmpfs on /var/volatile type tmpfs" not in state["raw"]:
        raise ValueError("runtime destination is not confirmed tmpfs")
    if exclusive and re.search(r"^\s*\d+.*(?:/|\s)tvm_rpc(?:\s|$)", state["raw"], re.M):
        raise ValueError("another RPC already owns the board; do not interrupt it")


def invoke(module, buffers, workload, seed, functions, timer, timed=False):
    import numpy as np
    from qualify_vta_residency_fsim import array_sha256
    from run_vta_p7r132_y00_search_confirmation import prepare_reused_buffers
    expected = prepare_reused_buffers(buffers, workload, seed)
    functions["vta.runtime.profiler_clear"]()
    started = time.monotonic()
    if timed:
        latency = float(timer(module, buffers["data"], buffers["weight"], buffers["output"]))
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError("invalid board elapsed time")
    else:
        module["main"](buffers["data"], buffers["weight"], buffers["output"])
        latency = None
    execution_wall = time.monotonic() - started
    actual = buffers["output"].numpy()
    mismatch = int(np.count_nonzero(actual != expected))
    profile = json.loads(functions["vta.runtime.profiler_status"]())
    return {"seed": seed, "correct": mismatch == 0, "mismatch_count": mismatch,
            "latency_ms": latency, "expected_sha256": array_sha256(expected),
            "actual_sha256": array_sha256(actual), "runtime_profile_complete": profile,
            "complete_operator_calls": 1, "hidden_warmup_calls": 0,
            "rpc_call_wall_seconds": execution_wall}


def control_blocks(candidates, controls, passed):
    blocks = []
    for family in dict.fromkeys(row["family_id"] for row in controls):
        resident = {r["residence_mode"]: r for r in candidates if r["family_id"] == family}
        reload = {r["residence_mode"]: r for r in controls if r["family_id"] == family}
        modes = ("weight_resident_barrier", "input_weight_resident_barrier")
        block = [r for mode in modes for r in (resident[mode], reload[mode])]
        if all(r["candidate_id"] in passed for r in block):
            blocks.append((family, block))
    return blocks


def run_board(args, output, started):
    if not args.execute_board:
        raise ValueError("board execution requires --execute-board")
    prepared_dir = Path(args.contract).resolve().parent
    prepared = verify(prepared_dir)
    if prepared["phase"] != "board_prepared":
        raise ValueError("requires sealed board preparation")
    check_bindings(prepared["source_bindings"])
    parent = Path(prepared["parent_directory"])
    frozen = verify(parent)
    if file_hash(parent / "artifact_hashes.json") != prepared["parent_manifest_sha256"]:
        raise ValueError("Node-B manifest changed")
    candidates, controls = read(parent / "candidates.json"), read(parent / "controls.json")
    runtime = "/var/volatile/" + output.name + "/runtime"
    board = Board(args, runtime)
    initial = board.state()
    validate_state(initial, exclusive=True)
    if args.previous_boot and initial["boot_id"] == args.previous_boot:
        raise ValueError("C2 requires actual reboot; same boot cannot be a second confirmation")
    write(output / "preflight.json", initial)
    contract = {**prepared, "phase": "board", "board_contacted": True,
                "prepared_manifest_sha256": file_hash(prepared_dir / "artifact_hashes.json"),
                "boot_id": initial["boot_id"], "runtime_directory": runtime,
                "known_hosts_sha256": file_hash(args.known_hosts), "host": args.host}
    write(output / "contract.json", contract)
    remote = None
    completed_calls = 0
    timing_rows, correctness = [], []
    T0 = None
    try:
        import tvm
        from tvm import rpc
        from run_vta_p7r132_y00_search_confirmation import allocate_reusable_buffers
        from run_vta_p7r120_yolo_rpc_board import runtime_inventory, load_ephemeral
        write(output / "runtime_attestation.json", board.deploy(prepared["runtime_paths"]))
        write(output / "bitstream_reload.json", board.reload())
        pid = board.start()
        append(output / "timeline.jsonl", {"event": "rpc_start", "pid": pid, "wall_unix": time.time()})
        # Connection readiness polling does not dispatch any FPGA operator.
        connection_error = None
        for _ in range(20):
            try:
                remote = rpc.connect(args.host, 9090, session_timeout=7200)
                break
            except Exception as error:
                connection_error = error
                time.sleep(0.25)
        if remote is None:
            raise RuntimeError("RPC did not become ready") from connection_error
        T0 = time.monotonic()
        append(output / "timeline.jsonl", {"event": "T0", "elapsed_seconds": T0 - started, "wall_unix": time.time()})
        inventory, functions = runtime_inventory(remote)
        write(output / "runtime_inventory.json", inventory)
        helper_module = load_ephemeral(remote, prepared_dir / "binaries/c3_reuse_timed_once.so")
        timer = remote.get_function("c3.reuse.timed_once")
        device = remote.ext_dev(0)
        buffers, modules = {}, {}
        all_rows = [prepared["health"]] + candidates + controls

        def call(row, phase, seed, timed=False, **extra):
            nonlocal completed_calls
            if time.monotonic() - started > 7200:
                raise TimeoutError("whole board session exceeded 7200 seconds")
            workload = row["identity"]["workload"]
            key = row["workload_id"]
            if key not in buffers:
                buffers[key] = allocate_reusable_buffers(device, workload)
                append(output / "allocations.jsonl", {"workload_id": key,
                       **{k: v for k, v in buffers[key].items() if k not in {"data", "weight", "output"}},
                       "allocation_events": 3, "lifetime": "retained to session end"})
            cid = row["candidate_id"]
            if cid not in modules:
                directory = prepared_dir if key == "HEALTH" else parent
                binary = directory / "binaries" / (cid + ".so")
                if file_hash(binary) != row["cross_qualification"]["binary_sha256"]:
                    raise ValueError("candidate binary changed")
                modules[cid] = load_ephemeral(remote, binary)
            begin = time.monotonic()
            append(output / "timeline.jsonl", {"event": "dispatch_begin", "phase": phase,
                   "candidate_id": cid, "seed": seed, "elapsed_seconds": begin - started, **extra})
            result = invoke(modules[cid], buffers[key], workload, seed, functions, timer, timed)
            completed_calls += 1
            result.update(phase=phase, candidate_id=cid, family_id=row["family_id"],
                          residence_mode=row["residence_mode"], control=row.get("experimental_reload_control", False),
                          host_wall_seconds=time.monotonic() - begin, **extra)
            append(output / "results.jsonl", result)
            append(output / "timeline.jsonl", {"event": "dispatch_complete", "phase": phase,
                   "candidate_id": cid, "correct": result["correct"], "elapsed_seconds": time.monotonic() - started})
            return result

        # Health must precede every target FPGA label.
        health = all_rows[0]
        for seed in SEEDS:
            if not call(health, "health", seed)["correct"]:
                raise RuntimeError("historically qualified original health probe failed")
        passed = set()
        for position, row in enumerate(candidates + controls):
            observations = []
            for seed in SEEDS:
                result = call(row, "correctness", seed)
                observations.append(result)
                if not result["correct"]:
                    break
            ok = len(observations) == 3 and all(r["correct"] for r in observations)
            if ok:
                passed.add(row["candidate_id"])
            correctness.append({"candidate_id": row["candidate_id"], "correct": ok,
                                "seed_calls": len(observations)})
            append(output / "correctness.jsonl", correctness[-1])
            print("FPGA correctness {}/48 {} {} {}".format(position + 1, row["family_id"], row["residence_mode"], ok), flush=True)
        for row in candidates + controls:
            if row["candidate_id"] in passed and not call(row, "warmup", SEEDS[0])["correct"]:
                raise RuntimeError("checked explicit warmup failed; invalidate session")
        completed_families = []
        for family in frozen["frozen_family_order"]:
            by_mode = {r["residence_mode"]: r for r in candidates if r["family_id"] == family}
            if not all(r["candidate_id"] in passed for r in by_mode.values()):
                continue
            for round_index, order in enumerate(frozen["frozen_mode_orders"]):
                for mode in order:
                    result = call(by_mode[mode], "timing", SEEDS[round_index % 3], True,
                                  round=round_index, block="main")
                    if not result["correct"]:
                        raise RuntimeError("timed output wrong; invalidate performance session")
                    timing_rows.append(result)
            completed_families.append(family)
            print("eight-round family complete " + family, flush=True)
        for family, block in control_blocks(candidates, controls, passed):
            for round_index in range(8):
                offset = round_index % 4
                for row in block[offset:] + block[:offset]:
                    result = call(row, "timing", SEEDS[round_index % 3], True,
                                  round=round_index, block="reload_control")
                    if not result["correct"]:
                        raise RuntimeError("control timing output wrong; invalidate session")
                    timing_rows.append(result)
        for row in candidates + controls:
            if row["candidate_id"] in passed and not call(row, "profile", SEEDS[0])["correct"]:
                raise RuntimeError("independent profile wrong; invalidate session")
        final_state = board.state()
        validate_state(final_state)
        if final_state["boot_id"] != initial["boot_id"]:
            raise RuntimeError("boot changed during session")
        write(output / "final_state.json", final_state)
        stats = {}
        for row in timing_rows:
            if row["block"] == "main":
                stats.setdefault(row["candidate_id"], []).append(row["latency_ms"])
        if any(len(values) != 8 for values in stats.values()):
            raise ValueError("incomplete eight-round measurement")
        summary = {"status": "C1_COMPLETE_PAUSED" if not args.previous_boot else "C2_COMPLETE_READY_FOR_ANALYSIS",
                   "boot_id": initial["boot_id"], "correct_identities": len(passed),
                   "rejected_identities": 48 - len(passed), "complete_four_mode_families": completed_families,
                   "timed_complete_operator_calls": len(timing_rows), "all_completed_operator_calls": completed_calls,
                   "candidate_timing": {cid: {"values_ms": values, "median_ms": statistics.median(values)} for cid, values in stats.items()},
                   "W0_to_T1_seconds": time.monotonic() - started,
                   "T0_to_T1_seconds": time.monotonic() - T0,
                   "pure_instruction_claim": False, "physical_axi_claim": False,
                   "cost_scope": "this continuous valid board session; prior Node-B qualification remains separate"}
        write(output / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    except BaseException as error:
        write(output / "invalid_session.json", {"type": type(error).__name__, "message": str(error)[-5000:],
              "all_timing_invalid": True, "completed_operator_calls": completed_calls,
              "W0_to_failure_seconds": time.monotonic() - started,
              "T0_to_failure_seconds": None if T0 is None else time.monotonic() - T0})
        raise
    finally:
        if remote is not None:
            del remote
        try:
            board.stop()
        except Exception as error:
            write(output / "cleanup_warning.json", {"message": str(error)[-2000:], "owned_rpc_pid": board.pid})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "board"))
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execute-board", action="store_true")
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
    parser.add_argument("--previous-boot")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sdk-environment", default="/home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux")
    args = parser.parse_args()
    started = time.monotonic()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    append(output / "timeline.jsonl", {"event": "W0", "wall_unix": time.time()})
    try:
        (prepare if args.phase == "prepare" else run_board)(args, output, started)
        append(output / "timeline.jsonl", {"event": "T1", "elapsed_seconds": time.monotonic() - started})
    except BaseException as error:
        if not (output / "invalid_session.json").exists():
            write(output / "invalid_session.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        seal(output)


if __name__ == "__main__":
    main()
