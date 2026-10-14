"""Fail-closed FPGA qualification and six-program timing for the frozen pilot."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import ctypes
import gc
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import time
import traceback

import numpy as np
import tvm

import c3_residency_tradeoff as p
from c3_residency_tradeoff_evidence import check_runtime, OLD, GEMM, function
from c3_thread_sync_audit import instruction_dump
from c3_thread_sync_board import ExperimentBoard
import run_vta_c3_reuse_board as reuse
from run_vta_c3_mode11_graph_context_diagnostic import connect, upload_module
from run_vta_c3_node24_partial_board import RUNTIME, SYMBOL, same, sentinel
from run_vta_c3_node24_partial_local import _reference, frozen_fusion
from run_vta_p7r120_yolo_rpc_board import runtime_inventory, load_ephemeral


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TIMER = OLD.parent / "20261008_c3_node34_timer_build_run01/c3_node34_fused_timed_once.so"
DIAGNOSTIC = GEMM / "c3_node24_fused_diagnostic_once.so"
COUNTER_SOURCE = HERE / "c3_residency_tradeoff_counter.cc"


def array_sha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def buffer_handles(buffers):
    """Stable client-side NDArray handles, not physical SRAM/DDR addresses."""
    return [int(ctypes.cast(b.handle, ctypes.c_void_p).value) for b in buffers]


def append(path, row):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def dump_signature(parsed):
    semantic = Counter()
    payload = Counter()
    for submission in parsed["submissions"]:
        semantic.update(submission["semantic"])
        payload.update(submission["load_bytes"])
    return {"reset": semantic["RESET_GEMM"], "mac": semantic["MAC_GEMM"],
            "wgt_loads": semantic["LOAD_WGT"], "wgt_payload_bytes": payload["WGT"],
            "uop_loads": semantic["LOAD_UOP"], "uop_payload_bytes": payload["UOP"],
            "instructions": sum(x["instructions"] for x in parsed["submissions"]),
            "submissions": parsed["submission_count"], "semantic": dict(semantic)}


def validate_request_profile(profile, totals):
    for key, value in totals.items():
        if key.startswith(("load_buffer_2d_", "store_buffer_2d_")) and \
                key.endswith(("_calls", "_bytes")):
            if profile.get(key) != value:
                raise ValueError("static/actual data request mismatch: " + key)
    if profile.get("template_replay_hits", 0) or profile.get("template_capture_calls", 0):
        raise ValueError("unexpected replay/capture is outside the frozen call protocol")
    if profile.get("mem_copy_to_host_calls") != 0:
        raise ValueError("profile was contaminated by output reads before its snapshot")


class Runner:
    def __init__(self, args):
        self.args = args
        self.root = args.output_dir.resolve()
        self.prepared = p.read(self.root / "prepared.json")
        self.protocol = p.read(self.root / "protocol.json")
        if (self.prepared["status"] != "ALL_LOCAL_GATES_PASSED_BOARD_NOT_QUALIFIED"
                or self.prepared["schema"] != p.SCHEMA
                or self.prepared["protocol_sha256"] != p.digest(self.root / "protocol.json")):
            raise ValueError("local preparation or protocol identity is invalid")
        for path, expected in self.protocol["source_sha256"].items():
            if p.digest(path) != expected:
                raise ValueError("prepared source changed: " + path)
        if p.digest(self.protocol["document"]) != self.protocol["document_sha256"]:
            raise ValueError("execution protocol document changed")
        self.runtime = p.read(RUNTIME)
        reuse.RUNTIME = self.runtime["runtime_sha256"]
        check_runtime()
        self.libraries = {}
        self.totals = {}
        for name in p.PROGRAMS:
            record = self.prepared["programs"][name]
            directory, binary = Path(record["directory"]), Path(record["binary"])
            if (p.digest(binary) != record["binary_sha256"] or
                    p.digest(directory / "node24_final.tir.json") != record["tir_sha256"]):
                raise ValueError("prepared binary or TIR changed: " + name)
            self.libraries[name] = binary
            self.totals[name] = p.read(directory / "result.json")["request_totals"]
        self.x = {s: np.load(OLD / "board_qualify03" / ("X_%d.npy" % s)) for s in p.SEEDS}
        self.y = {s: np.load(OLD / "board_qualify03" / ("Y24_%d.npy" % s)) for s in p.SEEDS}
        self.weight = np.load(OLD / "board_qualify03/W_A.npy")
        self.weight_b = np.load(OLD / "board_qualify03/W_B.npy")
        self.add = np.load(OLD / "board_qualify03/ADD.npy")
        self.y_b = np.load(OLD / "board_qualify03/Y_B.npy")
        # Revalidate the exact real-data reference against the complete math.
        if any(not same(_reference(self.x[s], self.weight, self.add), self.y[s]) for s in p.SEEDS) or \
                not same(_reference(self.x[0], self.weight_b, self.add), self.y_b) or same(self.y[0], self.y_b):
            raise ValueError("real fused reference contract or A/B distinguishing power failed")
        reference = function(Path(self.prepared["programs"]["R13-G"]["directory"]))
        abi = [{"shape": [int(x) for x in reference.buffer_map[param].shape],
                "dtype": str(reference.buffer_map[param].dtype),
                "strides": [str(x) for x in reference.buffer_map[param].strides],
                "element_offset": str(reference.buffer_map[param].elem_offset)}
               for param in reference.params]
        for name in p.PROGRAMS:
            actual = function(Path(self.prepared["programs"][name]["directory"]))
            other = [{"shape": [int(x) for x in actual.buffer_map[param].shape],
                      "dtype": str(actual.buffer_map[param].dtype),
                      "strides": [str(x) for x in actual.buffer_map[param].strides],
                      "element_offset": str(actual.buffer_map[param].elem_offset)}
                     for param in actual.params]
            if abi != other:
                raise ValueError("four-parameter buffer ABI differs: " + name)
        self.data_contract = {
            "abi_order": ["X", "W", "ADD", "Y"], "buffers": abi,
            "fused_relay_sha256": hashlib.sha256(tvm.ir.save_json(frozen_fusion()).encode()).hexdigest(),
            "arithmetic": "int32 conv; ADD[g,0,0,0,co_inner] broadcast before arithmetic >>7; "
                          "clip(-127,127); int8 cast",
            "independent_buffers_no_alias": True,
            "input_data_sha256": {str(s): array_sha(self.x[s]) for s in p.SEEDS},
            "reference_sha256": {str(s): array_sha(self.y[s]) for s in p.SEEDS},
            "weight_A_sha256": array_sha(self.weight), "weight_B_sha256": array_sha(self.weight_b),
            "add_sha256": array_sha(self.add), "reference_AB_mismatches": int(np.count_nonzero(self.y[0] != self.y_b)),
        }
        probe = ExperimentBoard(args, "/var/volatile/c3_tradeoff_readonly_preflight", args.port)
        state = probe.state()
        reuse.validate_state(state, exclusive=True)
        self.boot = state["boot_id"]
        self.initial = state
        self.sessions = []
        self.rows = []
        self.out = None

    def start_phase(self, phase):
        self.out = self.root / phase
        self.out.mkdir(exist_ok=False)
        (self.out / "programs").mkdir()
        p.write_new(self.out / "preflight.json", self.initial)

    def record(self, row):
        row.setdefault("latency_ms", None)
        row["boot_id"] = self.boot
        row["invocation_id"] = self.out.name + ":%06d" % len(self.rows)
        row["status"] = "PASS" if row.get("correct") and row.get("arguments_unchanged") else "FAIL"
        append(self.out / "calls.jsonl", row)
        if row.get("program") in p.PROGRAMS:
            append(self.out / "programs" / (row["program"] + ".jsonl"), row)
        self.rows.append(row)
        return row

    @contextmanager
    def session(self, name, binary):
        sequence = len(self.sessions)
        runtime_dir = "/var/volatile/c3_tradeoff_%s_%s_%03d" % (self.root.name, self.out.name, sequence)
        board = ExperimentBoard(self.args, runtime_dir, self.args.port + 2 * sequence)
        state = board.state()
        reuse.validate_state(state, exclusive=True)
        if state["boot_id"] != self.boot:
            raise ValueError("Linux boot changed before session")
        remote, refs = None, []
        data = {"session_id": "%s:%03d" % (self.out.name, sequence), "program": name,
                "runtime_directory": runtime_dir, "start_state": state}
        try:
            data["runtime_attestation"] = board.deploy(self.runtime["runtime_paths"])
            data["bitstream_load_record"] = board.reload()
            board.start()
            remote = connect(self.args.host, board.port, self.args.timeout)
            inventory, functions = runtime_inventory(remote)
            if not all(key in functions for key in ("vta.runtime.profiler_clear", "vta.runtime.profiler_status")):
                raise ValueError("required lightweight profiler functions unavailable")
            environment = board.environment(require_affinity=True)
            if environment["boot_id"] != self.boot:
                raise ValueError("RPC worker boot identity changed")
            data.update(inventory=inventory, environment=environment)
            data["loaded_maps"] = board.ssh(
                "for c3_pid in $(pidof tvm_rpc); do "
                "test \"$(readlink /proc/$c3_pid/cwd)\" = " + shlex.quote(runtime_dir) + " || continue; "
                "awk '/libvta.so|libtvm_runtime.so/ {print}' /proc/$c3_pid/maps; done")
            lib = load_ephemeral(remote, Path(binary))
            refs.append(lib)
            yield remote, lib, functions, refs, data["session_id"]
            data["end_environment"] = board.environment(require_affinity=True)
            if data["end_environment"]["boot_id"] != self.boot:
                raise ValueError("RPC worker boot changed after calls")
            data["status"] = "PASS"
        except BaseException as error:
            data.update(status="FAIL", error=str(error))
            raise
        finally:
            refs.clear()
            remote = None
            gc.collect()
            try:
                board.stop()
                # Retain all logs before removing only this created runtime copy.
                data["rpc_log"] = board.ssh("test ! -f " + shlex.quote(runtime_dir + "/rpc.log") +
                                           " || cat " + shlex.quote(runtime_dir + "/rpc.log"))
                if not runtime_dir.startswith("/var/volatile/c3_tradeoff_" + self.root.name + "_"):
                    raise ValueError("unvalidated cleanup target")
                target = shlex.quote(runtime_dir)
                board.ssh("test -z \"$(pidof tvm_rpc 2>/dev/null || true)\" && test -d " + target +
                          " && test ! -L " + target + " && rm -r " + target)
            except Exception as cleanup:
                data["cleanup_error"] = str(cleanup)
                append(self.out / "sessions.jsonl", data)
                self.sessions.append(data)
                raise
            append(self.out / "sessions.jsonl", data)
            self.sessions.append(data)

    def buffers(self, remote, seed):
        dev = remote.ext_dev(0)
        return (tvm.nd.array(self.x[seed], dev), tvm.nd.array(self.weight, dev),
                tvm.nd.array(self.add, dev), tvm.nd.array(sentinel(self.y[seed]), dev))

    def call(self, lib, functions, name, buffers, seed, kind, session_id,
             weight=None, expected=None, timer=None, diagnostic=None, signature=None, **extra):
        weight = self.weight if weight is None else weight
        expected = self.y[seed] if expected is None else expected
        xb, wb, ab, yb = buffers
        xb.copyfrom(self.x[seed])
        wb.copyfrom(weight)
        ab.copyfrom(self.add)
        yb.copyfrom(sentinel(expected))
        functions["vta.runtime.profiler_clear"]()
        started = time.monotonic()
        dispatch_id = self.out.name + ":%06d" % len(self.rows)
        append(self.out / "dispatch.jsonl", {"invocation_id": dispatch_id, "event": "begin",
                                           "program": name, "call_kind": kind})
        latency, parsed, dump = None, None, None
        if timer is not None:
            latency = float(timer(lib, SYMBOL, *buffers))
        elif diagnostic is not None:
            dump = str(diagnostic(lib, SYMBOL, *buffers))
        else:
            lib[SYMBOL](*buffers)
        # Must precede output/argument reads: copies are outside the timer and profile.
        profile = json.loads(str(functions["vta.runtime.profiler_status"]()))
        actual = np.array(yb.numpy(), copy=True)
        unchanged = all(same(got, want) for got, want in
                        zip((xb.numpy(), wb.numpy(), ab.numpy()), (self.x[seed], weight, self.add)))
        correct = same(actual, expected)
        row = {"program": name, "seed": seed, "call_kind": kind, "session_id": session_id,
               "correct": correct, "arguments_unchanged": unchanged, "latency_ms": latency,
               "runtime_profile": profile, "profile_before_output_reads": True,
               "output_sha256": array_sha(actual), "expected_sha256": array_sha(expected),
               "buffer_handles": buffer_handles(buffers),
               "handle_scope": "client NDArray identity; not physical device address",
               "weight_payload_sha256": array_sha(weight),
               "complete_operator_calls": 1, "hidden_warmup_calls": 0, **extra}
        if dump is not None:
            dump_name = dispatch_id.replace(":", "_") + ".txt"
            (self.out / dump_name).write_text(dump)
            parsed = instruction_dump(dump)
            observed = dump_signature(parsed)
            row.update(dump_file=dump_name, dump_sha256=p.digest(self.out / dump_name),
                       instruction_audit=parsed, command_signature=observed)
            exp = p.expectations(name)
            if any(observed[key] != exp[key] for key in ("reset", "mac", "wgt_loads", "wgt_payload_bytes")):
                self.record(row)
                raise ValueError("actual instruction signature violates compiled target")
            if observed["submissions"] != profile["driver_run_calls"] or \
                    observed["instructions"] != profile["driver_run_insns"]:
                self.record(row)
                raise ValueError("dump is not associated with the same invocation profile")
            signature = observed
        validate_request_profile(profile, self.totals[name])
        if signature is None:
            exp = p.expectations(name)
            if profile["push_gemm_op_calls"] != exp["total_gemm_push"] or \
                    profile.get("driver_timeout_calls", 0) or profile.get("driver_run_calls", 0) <= 0:
                self.record(row)
                raise ValueError("unexpected target commands or incomplete device work")
        else:
            row["model_features"] = p.verify_call_profile(name, profile, signature)
            if (profile["driver_run_insns"] != signature["instructions"]
                    or profile["driver_run_calls"] != signature["submissions"]):
                row["diagnostic_signature_matched"] = False
                self.record(row)
                raise ValueError("formal call and state-matched diagnostic differ")
            row["diagnostic_signature_matched"] = True
        if timer is not None and (not math.isfinite(latency) or latency <= 0):
            self.record(row)
            raise ValueError("invalid timer result")
        row["host_call_and_validation_seconds"] = time.monotonic() - started
        self.record(row)
        append(self.out / "dispatch.jsonl", {"invocation_id": row["invocation_id"], "event": "end"})
        if not correct or not unchanged:
            np.save(self.out / ("failure_" + dispatch_id.replace(":", "_") + ".npy"), actual)
            raise ValueError("FPGA output/argument error; stop all subsequent board work")
        return row

    def qualify(self):
        self.start_phase("qualification")
        prepared_dir = self.root / "infrastructure"
        prepared_dir.mkdir(exist_ok=False)
        counter = compile_counter(prepared_dir)
        helper_manifest = helper_certificate()
        p.write_new(self.out / "contract.json", {
            "schema": p.SCHEMA, "boot_id": self.boot, "data_contract": self.data_contract,
            "prepared_sha256": p.digest(self.root / "prepared.json"), "helpers": helper_manifest,
            "sources": execution_sources(), "no_reboot_or_SD_writes": True,
        })
        # CPU-only helper regression: 3 requests must produce exactly 3 increments.
        with self.session("helper_regression", counter) as (remote, lib, funcs, refs, sid):
            refs.append(load_ephemeral(remote, TIMER))
            reset = remote.get_function("c3.node34.reset_counts")
            counts = remote.get_function("c3.node34.counts")
            timer = remote.get_function("c3.node34.fused_timed_once")
            buffers = tuple(tvm.nd.array(np.array([v], dtype="int32"), remote.cpu(0)) for v in (1, 2, 3, 0))
            reset()
            for index in range(3):
                funcs["vta.runtime.profiler_clear"]()
                timer(lib, "c3_four_arg_counter", *buffers)
                profile = json.loads(str(funcs["vta.runtime.profiler_status"]()))
                result = int(buffers[-1].numpy()[0])
                if result != 123 * (index + 1) or int(counts()) != index + 1 or profile["driver_run_calls"]:
                    raise ValueError("timer ABI/hidden invocation/no-extra-sync regression failed")
                self.record({"program": "helper_regression", "call_kind": "helper_regression",
                             "session_id": sid, "seed": None, "correct": True,
                             "arguments_unchanged": all(int(b.numpy()[0]) == v
                                                        for b, v in zip(buffers[:3], (1, 2, 3))),
                             "counter_value": result, "helper_calls": int(counts()), "runtime_profile": profile})
        for name in p.PROGRAMS:
            with self.session(name, self.libraries[name]) as (remote, lib, funcs, refs, sid):
                buffers = self.buffers(remote, p.SEEDS[0])
                for seed in p.SEEDS:
                    self.call(lib, funcs, name, buffers, seed, "correctness", sid)
                handles = buffer_handles(buffers)
                for label, weight, ref in (("A", self.weight, self.y[0]),
                                           ("B", self.weight_b, self.y_b),
                                           ("A_return", self.weight, self.y[0])):
                    self.call(lib, funcs, name, buffers, 0, "weight_ABA", sid,
                              weight=weight, expected=ref, aba_label=label)
                    if buffer_handles(buffers) != handles:
                        raise ValueError("A/B/A changed parameter buffer handles")
            print("FPGA correct and same-handle A/B/A: " + name, flush=True)
        diagnostic_data = {}
        for name in p.PROGRAMS:
            # Cold command state is independent from the warmed 4/5/6 states.
            with self.session(name, self.libraries[name]) as (remote, lib, funcs, refs, sid):
                refs.append(load_ephemeral(remote, DIAGNOSTIC))
                diag = remote.get_function("c3.node24.fused_diagnostic_once")
                row = self.call(lib, funcs, name, self.buffers(remote, 0), 0, "first_call_diagnostic",
                                sid, diagnostic=diag, call_position=1)
                diagnostic_data[name] = {"first_call": row["command_signature"], "rotations": {}}
            for rotation in range(3):
                seeds = p.seed_order(rotation)
                with self.session(name, self.libraries[name]) as (remote, lib, funcs, refs, sid):
                    refs.append(load_ephemeral(remote, DIAGNOSTIC))
                    diag = remote.get_function("c3.node24.fused_diagnostic_once")
                    buffers = self.buffers(remote, seeds[0])
                    for position, seed in enumerate(seeds):
                        self.call(lib, funcs, name, buffers, seed, "diagnostic_warmup", sid,
                                  rotation=rotation, call_position=position + 1)
                    diagnostic_data[name]["rotations"][str(rotation)] = []
                    for position, seed in enumerate(seeds):
                        row = self.call(lib, funcs, name, buffers, seed, "warmed_diagnostic", sid,
                                        diagnostic=diag, rotation=rotation, call_position=position + 4)
                        diagnostic_data[name]["rotations"][str(rotation)].append(row["command_signature"])
            print("Cold and calls 4/5/6 diagnostic qualified: " + name, flush=True)
        kinds = Counter(r["call_kind"] for r in self.rows)
        if kinds["correctness"] != 18 or kinds["weight_ABA"] != 18 or \
                kinds["warmed_diagnostic"] != 54 or kinds["diagnostic_warmup"] != 54 or \
                kinds["first_call_diagnostic"] != 6 or kinds["helper_regression"] != 3:
            raise ValueError("qualification/diagnostic call ledger mismatch")
        state = self.final_state()
        p.write_new(self.out / "diagnostic_signatures.json", diagnostic_data)
        p.write_new(self.out / "summary.json", {
            "schema": p.SCHEMA, "status": "SIX_PROGRAMS_FPGA_AND_COMMAND_QUALIFIED",
            "boot_id": self.boot, "call_counts": dict(kinds), "target_fpga_calls": len(self.rows) - 3,
            "helper_regression_calls": 3, "all_correct": True, "final_state": state,
            "prepared_sha256": p.digest(self.root / "prepared.json"),
            "contract_sha256": p.digest(self.out / "contract.json"),
            "diagnostic_signatures_sha256": p.digest(self.out / "diagnostic_signatures.json"),
        })

    def formal(self):
        qualified = p.read(self.root / "qualification/summary.json")
        certificate = p.read(self.root / "qualification/contract.json")
        if (qualified["status"] != "SIX_PROGRAMS_FPGA_AND_COMMAND_QUALIFIED"
                or qualified["boot_id"] != self.boot
                or qualified["prepared_sha256"] != p.digest(self.root / "prepared.json")
                or qualified["contract_sha256"] != p.digest(self.root / "qualification/contract.json")
                or certificate["sources"] != execution_sources()
                or certificate["helpers"] != helper_certificate()):
            raise ValueError("FPGA prerequisite is stale, incomplete, or from another boot")
        diag_path = self.root / "qualification/diagnostic_signatures.json"
        if p.digest(diag_path) != qualified["diagnostic_signatures_sha256"]:
            raise ValueError("diagnostic signature identity changed")
        signatures = p.read(diag_path)
        self.start_phase("board")
        p.write_new(self.out / "contract.json", {
            "status": "PRE_LATENCY_FROZEN", "schema": p.SCHEMA, "boot_id": self.boot,
            "programs": self.prepared["programs"], "orders": p.ORDERS, "seeds": p.SEEDS,
            "balance": p.balance(), "qualification_sha256": p.digest(self.root / "qualification/summary.json"),
            "sources": execution_sources(), "helpers": helper_certificate(), "data_contract": self.data_contract,
            "model": {"calibration": p.CALIBRATION, "validation": p.VALIDATION,
                      "nonnegative": True, "threshold_absolute_relative_pct": 10},
            "timer_boundary": "one packed fused call, host organization/submits/completion inside; "
                              "lookup/RPC/upload/readback/checks outside; no additional synchronization",
        })
        for round_id, order in enumerate(p.ORDERS):
            seeds = p.seed_order(round_id)
            for name in order:
                with self.session(name, self.libraries[name]) as (remote, lib, funcs, refs, sid):
                    refs.append(load_ephemeral(remote, TIMER))
                    reset = remote.get_function("c3.node34.reset_counts")
                    counts = remote.get_function("c3.node34.counts")
                    timer = remote.get_function("c3.node34.fused_timed_once")
                    reset()
                    buffers = self.buffers(remote, seeds[0])
                    for position, seed in enumerate(seeds):
                        self.call(lib, funcs, name, buffers, seed, "warmup", sid,
                                  round=round_id, seed_position=position, call_position=position + 1)
                    for position, seed in enumerate(seeds):
                        signature = signatures[name]["rotations"][str(round_id % 3)][position]
                        row = self.call(lib, funcs, name, buffers, seed, "timing", sid,
                                        timer=timer, signature=signature, round=round_id,
                                        seed_position=position, call_position=position + 4)
                        if int(counts()) != position + 1:
                            raise ValueError("unexpected extra/missing full timed invocation")
                        append(self.out / "helper_counts.jsonl", {
                            "invocation_id": row["invocation_id"], "count": int(counts()),
                            "expected": position + 1})
                # No latency or validation ranking is printed before model sealing.
                print("Formal cell checked: round %d %s" % (round_id + 1, name), flush=True)
        kinds = Counter(r["call_kind"] for r in self.rows)
        if kinds != {"warmup": 108, "timing": 108}:
            raise ValueError("formal call ledger is not 108+108")
        state = self.final_state()
        p.write_new(self.out / "summary.json", {
            "status": "FORMAL_ACQUISITION_COMPLETE", "schema": p.SCHEMA, "boot_id": self.boot,
            "timed_calls": 108, "warmup_calls": 108, "call_counts": dict(kinds),
            "all_correct": True, "final_state": state,
            "no_reboot_second_boot_graph_or_transparent_retry": True,
        })

    def final_state(self):
        board = ExperimentBoard(self.args, "/var/volatile/c3_tradeoff_readonly_final", self.args.port)
        state = board.state()
        reuse.validate_state(state, exclusive=True)
        if state["boot_id"] != self.boot:
            raise ValueError("boot changed during the phase")
        p.write_new(self.out / "final_state.json", state)
        return {key: value for key, value in state.items() if key != "raw"}


def helper_certificate():
    if not TIMER.is_file() or not DIAGNOSTIC.is_file():
        raise ValueError("frozen helper binary unavailable")
    result = {}
    for name, binary, source in (
            ("timer", TIMER, HERE / "c3_node34_fused_timed_once.cc"),
            ("diagnostic", DIAGNOSTIC, HERE / "c3_node24_fused_diagnostic_once.cc")):
        deps = subprocess.run(["readelf", "-d", str(binary)], capture_output=True,
                              text=True, check=True).stdout
        defined = subprocess.run(["nm", "-D", "--defined-only", str(binary)], capture_output=True,
                                 text=True, check=True).stdout
        if "libvta" in deps or "VTASynchronize" in defined or "VTATLSCommandHandle" in defined:
            raise ValueError("helper contains a second VTA runtime")
        result[name] = {"binary": str(binary), "binary_sha256": p.digest(binary),
                        "source": str(source), "source_sha256": p.digest(source),
                        "elf_dependencies": deps, "defines_VTA_runtime": False}
    return result


def execution_sources():
    names = ("c3_residency_tradeoff.py", "c3_residency_tradeoff_local.py",
             "c3_residency_tradeoff_evidence.py", "c3_residency_tradeoff_analysis.py",
             "c3_residency_tradeoff_board.py", "c3_residency_tradeoff_counter.cc",
             "run_vta_c3_residency_tradeoff.py")
    return {str(HERE / name): p.digest(HERE / name) for name in names}


def compile_counter(output):
    binary = output / "four_arg_counter.so"
    sdk = "/home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux"
    command = ["bash", "-c", 'unset LD_LIBRARY_PATH; source "$1" && shift && exec $CXX "$@"',
               "c3-counter-build", sdk, "-std=c++14", "-shared", "-fPIC", str(COUNTER_SOURCE),
               "-I" + str(ROOT / "include"), "-I" + str(ROOT / "3rdparty/dlpack/include"),
               "-o", str(binary)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1200)
    p.write_new(output / "counter_build.json", {"command": command, "returncode": result.returncode,
                                              "stdout": result.stdout, "stderr": result.stderr,
                                              "source_sha256": p.digest(COUNTER_SOURCE)})
    if result.returncode:
        raise ValueError("four-argument helper regression module failed to compile")
    return binary


def run(args):
    runner = None
    try:
        runner = Runner(args)
        (runner.qualify if args.phase == "qualify" else runner.formal)()
    except BaseException as error:
        if runner is not None and runner.out is not None:
            p.write_new(runner.out / "invalid_attempt.json", {
                "status": "INVALID_NON_SPLICEABLE_STOP_ALL_BOARD_WORK",
                "boot_id": runner.boot, "error": str(error), "traceback": traceback.format_exc(),
                "completed_calls": len(runner.rows), "all_timing_invalid": True,
            })
        raise
