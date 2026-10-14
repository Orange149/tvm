"""Four-program prefetch validation using immutable historical timer/session code."""

from __future__ import annotations

from collections import Counter
import inspect
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace

import numpy as np
import tvm

import c3_residency_tradeoff as p
import c3_residency_tradeoff_board as base
from c3_five_tradeoff_policy import PROGRAMS, SEEDS, INCUMBENT, NEW, choose_visible
from c3_five_tradeoff_commands import certify_prefetch_dump

SCHEMA = "c3_five_tradeoff_single_candidate_v1"
ROTATIONS = tuple(PROGRAMS[i:] + PROGRAMS[:i] for i in range(4))
ORDERS = ROTATIONS + (tuple(reversed(ROTATIONS[0])), tuple(reversed(ROTATIONS[1])))
HERE = Path(__file__).resolve().parent
HISTORY = base.OLD.parent / "20261012_c3_residency_instruction_tradeoff_run01"


def expectations(name):
    if name not in PROGRAMS:
        raise ValueError("program outside frozen pool")
    return p.expectations("R12-G" if name == NEW else name)


def verify_profile(name, profile, signature):
    # Only command COUNT expectations are shared; identity and program label
    # remain NEW throughout dispatch, libraries, actual dump and call journals.
    return p.verify_call_profile("R12-G" if name == NEW else name, profile, signature)


def adapted_method(method):
    proxy = SimpleNamespace(**vars(p))
    proxy.PROGRAMS, proxy.SEEDS, proxy.SCHEMA = PROGRAMS, SEEDS, SCHEMA
    proxy.expectations, proxy.verify_call_profile = expectations, verify_profile
    namespace = dict(base.__dict__, p=proxy)
    local = {}
    exec(compile(textwrap.dedent(inspect.getsource(method)), __file__ + ":" + method.__name__, "exec"),
         namespace, local)
    return local[method.__name__]


def execution_sources():
    paths = [Path(base.__file__), Path(p.__file__), HERE / "c3_five_tradeoff_board.py",
             HERE / "c3_five_tradeoff_policy.py", HERE / "c3_five_tradeoff_commands.py",
             HERE / "run_vta_c3_five_tradeoff_board.py"]
    return {str(path): p.digest(path) for path in paths}


def execution_freeze(root):
    """Local-only: supplement session ordering/cost before any new FPGA timing."""
    root = Path(root).resolve()
    rule = p.read(root / "rule_freeze.json")
    for path, sha in rule["source_bindings"].items():
        if p.digest(path) != sha:
            raise ValueError("rule source identity drift")
    if p.digest(root / "PROTOCOL.md") != rule["protocol_sha256"]:
        raise ValueError("pre-latency protocol changed")
    old = p.read(HISTORY / "closeout_manifest.json")
    for path in (str(Path(base.__file__)), str(Path(p.__file__))):
        if p.digest(path) != old["source_files"][path]:
            raise ValueError("historical execution source changed")
    protocol = {"schema": SCHEMA, "source_sha256": execution_sources(),
                "document": str(root / "PROTOCOL.md"), "document_sha256": p.digest(root / "PROTOCOL.md"),
                "rule_sha256": p.digest(root / "rule_freeze.json"),
                "orders": ORDERS, "selection_orders": ROTATIONS[:3],
                "selection_cell": "3 native explicit seed warmups then 1 timed rotated seed, helper count=1",
                "confirmation_cell": "3 native warmups then 3 single timers, helper counts=1/2/3",
                "calls": {"qualification_target": 100, "CPU_helper_regression": 3,
                          "selection_warmup": 36, "selection": 12,
                          "confirmation_warmup": 72, "confirmation": 72},
                "rule_and_bytes_visible_calls_each": 6, "helper_certificate": base.helper_certificate(),
                "no_new_FPGA_latency_observed_at_freeze": True}
    p.write_new(root / "execution_protocol.json", protocol)
    p.write_new(root / "protocol.json", protocol)
    p.write_new(root / "prepared.json", {
        "schema": SCHEMA, "status": "ALL_LOCAL_GATES_PASSED_BOARD_NOT_QUALIFIED",
        "programs": rule["programs"], "runtime": rule["hardware_runtime"],
        "protocol_sha256": p.digest(root / "protocol.json")})
    print("EXECUTION_FROZEN_BEFORE_NEW_FPGA_LATENCY", flush=True)


class Runner(base.Runner):
    __init__ = adapted_method(base.Runner.__init__)
    _native_call = adapted_method(base.Runner.call)

    def start_phase(self, phase):
        self.rows, self.sessions = [], []
        super().start_phase(phase)

    def record(self, row):
        row.setdefault("latency_ms", None)
        row.update(boot_id=self.boot, invocation_id=self.out.name + ":%06d" % len(self.rows))
        row["status"] = "PASS" if row.get("correct") and row.get("arguments_unchanged") else "FAIL"
        base.append(self.out / "calls.jsonl", row)
        if row.get("program") in PROGRAMS:
            base.append(self.out / "programs" / (row["program"] + ".jsonl"), row)
        self.rows.append(row)
        return row

    def call(self, *args, **kwargs):
        row = self._native_call(*args, **kwargs)
        if row["program"] == NEW and row.get("dump_file"):
            proof = certify_prefetch_dump((self.out / row["dump_file"]).read_text())
            base.append(self.out / "prefetch_native_certificates.jsonl", {
                "invocation_id": row["invocation_id"], "dump_sha256": row["dump_sha256"],
                "actual_FPGA_runtime_encoding": True, "certificate": proof})
        return row

    def contract(self, **extra):
        p.write_new(self.out / "contract.json", {
            "schema": SCHEMA, "boot_id": self.boot, "data_contract": self.data_contract,
            "rule_sha256": p.digest(self.root / "rule_freeze.json"),
            "execution_protocol_sha256": p.digest(self.root / "execution_protocol.json"),
            "prepared_sha256": p.digest(self.root / "prepared.json"),
            "sources": execution_sources(), "helpers": base.helper_certificate(),
            "no_reboot_or_SD_writes": True, **extra})

    def qualify(self):
        self.start_phase("qualification")
        self.contract()
        counter = HISTORY / "infrastructure/four_arg_counter.so"
        historical = p.read(HISTORY / "closeout_manifest.json")
        if p.digest(counter) != historical["artifacts"]["infrastructure/four_arg_counter.so"]:
            raise ValueError("four-argument helper regression module drift")
        with self.session("helper_regression", counter) as (remote, lib, funcs, refs, sid):
            refs.append(base.load_ephemeral(remote, base.TIMER))
            reset, count = remote.get_function("c3.node34.reset_counts"), remote.get_function("c3.node34.counts")
            timer = remote.get_function("c3.node34.fused_timed_once")
            buffers = tuple(tvm.nd.array(np.array([x], dtype="int32"), remote.cpu(0)) for x in (1, 2, 3, 0))
            reset()
            for i in range(3):
                funcs["vta.runtime.profiler_clear"]()
                timer(lib, "c3_four_arg_counter", *buffers)
                profile = json.loads(str(funcs["vta.runtime.profiler_status"]()))
                actual = int(buffers[-1].numpy()[0])
                if actual != 123 * (i + 1) or int(count()) != i + 1 or profile["driver_run_calls"]:
                    raise ValueError("timer hidden invocation/extra sync/ABI failure")
                self.record({"program": "helper_regression", "call_kind": "helper_regression", "session_id": sid,
                             "seed": None, "correct": True, "arguments_unchanged": all(
                                 int(b.numpy()[0]) == x for b, x in zip(buffers[:3], (1, 2, 3))),
                             "counter_value": actual, "helper_calls": int(count()), "runtime_profile": profile})
        for name in PROGRAMS:
            with self.session(name, self.libraries[name]) as (remote, lib, funcs, refs, sid):
                buffers = self.buffers(remote, SEEDS[0])
                for seed in SEEDS:
                    self.call(lib, funcs, name, buffers, seed, "correctness", sid)
                handles = base.buffer_handles(buffers)
                for label, weight, expected in (("A", self.weight, self.y[0]),
                                                 ("B", self.weight_b, self.y_b),
                                                 ("A_return", self.weight, self.y[0])):
                    self.call(lib, funcs, name, buffers, 0, "weight_ABA", sid,
                              weight=weight, expected=expected, aba_label=label)
                    if base.buffer_handles(buffers) != handles:
                        raise ValueError("same-handle ABA object replaced")
            print("FPGA three-seed and fixed-input same-handle ABA passed: " + name, flush=True)
        signatures = {}
        for name in PROGRAMS:
            with self.session(name, self.libraries[name]) as (remote, lib, funcs, refs, sid):
                refs.append(base.load_ephemeral(remote, base.DIAGNOSTIC))
                diag = remote.get_function("c3.node24.fused_diagnostic_once")
                row = self.call(lib, funcs, name, self.buffers(remote, 0), 0, "first_call_diagnostic", sid,
                                diagnostic=diag, call_position=1)
                signatures[name] = {"first_call": row["command_signature"], "rotations": {}}
            for rotation in range(3):
                seeds = p.seed_order(rotation)
                with self.session(name, self.libraries[name]) as (remote, lib, funcs, refs, sid):
                    refs.append(base.load_ephemeral(remote, base.DIAGNOSTIC))
                    diag = remote.get_function("c3.node24.fused_diagnostic_once")
                    buffers = self.buffers(remote, seeds[0])
                    for i, seed in enumerate(seeds):
                        self.call(lib, funcs, name, buffers, seed, "diagnostic_warmup", sid,
                                  rotation=rotation, call_position=i + 1)
                    signatures[name]["rotations"][str(rotation)] = []
                    for i, seed in enumerate(seeds):
                        row = self.call(lib, funcs, name, buffers, seed, "warmed_diagnostic", sid,
                                        diagnostic=diag, rotation=rotation, call_position=i + 4)
                        signatures[name]["rotations"][str(rotation)].append(row["command_signature"])
            print("FPGA cold and matching warmed command states passed: " + name, flush=True)
        expected = {"helper_regression": 3, "correctness": 12, "weight_ABA": 12,
                    "first_call_diagnostic": 4, "diagnostic_warmup": 36, "warmed_diagnostic": 36}
        if dict(Counter(r["call_kind"] for r in self.rows)) != expected:
            raise ValueError("qualification call ledger differs")
        p.write_new(self.out / "diagnostic_signatures.json", signatures)
        p.write_new(self.out / "summary.json", {"status": "FOUR_PROGRAMS_FPGA_AND_COMMAND_QUALIFIED",
                    "boot_id": self.boot, "call_counts": expected, "all_correct": True,
                    "final_state": self.final_state(), "sources": execution_sources()})

    def require_qualification(self):
        for phase in ("qualification",):
            summary = p.read(self.root / phase / "summary.json")
            if summary["status"] != "FOUR_PROGRAMS_FPGA_AND_COMMAND_QUALIFIED" or summary["boot_id"] != self.boot \
                    or summary["sources"] != execution_sources():
                raise ValueError("qualification absent, changed or from another boot")
        return p.read(self.root / "qualification/diagnostic_signatures.json")

    def selection(self):
        signatures = self.require_qualification()
        self.start_phase("selection")
        self.contract(orders=ROTATIONS[:3], warmup_calls=36, timed_calls=12)
        for round_id, order in enumerate(ROTATIONS[:3]):
            seeds = p.seed_order(round_id)
            for name in order:
                with self.session(name, self.libraries[name]) as (remote, lib, funcs, refs, sid):
                    refs.append(base.load_ephemeral(remote, base.TIMER))
                    reset, count = remote.get_function("c3.node34.reset_counts"), remote.get_function("c3.node34.counts")
                    timer = remote.get_function("c3.node34.fused_timed_once")
                    reset()
                    buffers = self.buffers(remote, seeds[0])
                    for i, seed in enumerate(seeds):
                        self.call(lib, funcs, name, buffers, seed, "selection_warmup", sid,
                                  round=round_id, call_position=i + 1)
                    row = self.call(lib, funcs, name, buffers, seeds[0], "selection", sid, timer=timer,
                                    signature=signatures[name]["rotations"][str(round_id)][0],
                                    round=round_id, call_position=4)
                    if int(count()) != 1:
                        raise ValueError("extra complete selection invocation")
                    base.append(self.out / "helper_counts.jsonl", {"invocation_id": row["invocation_id"],
                                                                  "count": int(count()), "expected": 1})
                print("Shared selection cell complete: %d %s" % (round_id + 1, name), flush=True)
        if dict(Counter(r["call_kind"] for r in self.rows)) != {"selection": 12, "selection_warmup": 36}:
            raise ValueError("selection ledger differs")
        state = self.final_state()
        prefixes = p.read(self.root / "rule_freeze.json")["prefixes"]
        panel = [r for r in self.rows if r["call_kind"] == "selection"]
        winners = {name: choose_visible(panel, prefixes[name]) for name in ("rule", "bytes_only", "incumbent_only")}
        p.write_new(self.root / "winners.json", {
            "status": "WINNERS_SEALED_BEFORE_CONFIRMATION", "boot_id": self.boot, "winners": winners,
            "selection_journal_sha256": p.digest(self.out / "calls.jsonl"),
            "rule_sha256": p.digest(self.root / "rule_freeze.json"),
            "program_binary_sha256": {name: p.digest(self.libraries[name]) for name in PROGRAMS},
            "prefixes": {name: prefixes[name] for name in winners},
            "selection_ms_not_confirmation": True})
        p.write_new(self.out / "summary.json", {"status": "SELECTION_COMPLETE_WINNERS_SEALED",
                    "boot_id": self.boot, "call_counts": {"selection": 12, "selection_warmup": 36},
                    "winners_sha256": p.digest(self.root / "winners.json"), "final_state": state})
        print("WINNERS_SEALED_BEFORE_CONFIRMATION", flush=True)

    def confirmation(self):
        signatures = self.require_qualification()
        sealed = p.read(self.root / "winners.json")
        if sealed["boot_id"] != self.boot or sealed["rule_sha256"] != p.digest(self.root / "rule_freeze.json") \
                or sealed["selection_journal_sha256"] != p.digest(self.root / "selection/calls.jsonl"):
            raise ValueError("winner seal or selection identity invalid")
        winner_sha = p.digest(self.root / "winners.json")
        self.start_phase("confirmation")
        self.contract(orders=ORDERS, winners_sha256=winner_sha, warmup_calls=72, timed_calls=72)
        for round_id, order in enumerate(ORDERS):
            seeds = p.seed_order(round_id)
            for name in order:
                with self.session(name, self.libraries[name]) as (remote, lib, funcs, refs, sid):
                    refs.append(base.load_ephemeral(remote, base.TIMER))
                    reset, count = remote.get_function("c3.node34.reset_counts"), remote.get_function("c3.node34.counts")
                    timer = remote.get_function("c3.node34.fused_timed_once")
                    reset()
                    buffers = self.buffers(remote, seeds[0])
                    for i, seed in enumerate(seeds):
                        self.call(lib, funcs, name, buffers, seed, "warmup", sid,
                                  round=round_id, seed_position=i, call_position=i + 1)
                    for i, seed in enumerate(seeds):
                        row = self.call(lib, funcs, name, buffers, seed, "timing", sid, timer=timer,
                                        signature=signatures[name]["rotations"][str(round_id % 3)][i],
                                        round=round_id, seed_position=i, call_position=i + 4)
                        if int(count()) != i + 1:
                            raise ValueError("extra complete confirmation invocation")
                        base.append(self.out / "helper_counts.jsonl", {"invocation_id": row["invocation_id"],
                                                                      "count": int(count()), "expected": i + 1})
                print("Independent confirmation cell complete: %d %s" % (round_id + 1, name), flush=True)
        if dict(Counter(r["call_kind"] for r in self.rows)) != {"timing": 72, "warmup": 72} \
                or p.digest(self.root / "winners.json") != winner_sha:
            raise ValueError("confirmation ledger/winner identity differs")
        p.write_new(self.out / "summary.json", {"status": "CONFIRMATION_COMPLETE_NO_RESELECTION",
                    "boot_id": self.boot, "call_counts": {"timing": 72, "warmup": 72},
                    "winners_sha256": winner_sha, "all_correct": True, "final_state": self.final_state()})
