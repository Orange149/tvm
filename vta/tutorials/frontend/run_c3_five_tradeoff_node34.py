#!/usr/bin/env python3
"""Confirm the already nominated node34 Input, without retuning or reselection."""

import argparse
from collections import Counter
import inspect
import json
from pathlib import Path
from statistics import median
import textwrap
import traceback
from types import MethodType, SimpleNamespace

import numpy as np
import tvm

import c3_residency_tradeoff as p
import c3_residency_tradeoff_board as base
from c3_five_tradeoff_board import Runner as SessionRunner
from c3_five_tradeoff_node34 import PROGRAMS, SEEDS, SYMBOL, function


class Runner(SessionRunner):
    def __init__(self, args):
        for phase in ("qualification", "selection", "confirmation", "graph_node24"):
            if (args.output_dir / phase / "invalid_attempt.json").exists():
                raise ValueError("global stop from earlier board phase")
        super().__init__(args)
        self.nomination = p.read(self.root / "node34_nomination.json")
        if self.nomination["selected_program"] != "N34-Input" or \
                self.nomination["reference_source_sha256"] != p.digest(Path(__file__).parent / "c3_five_tradeoff_node34.py"):
            raise ValueError("known Input nomination changed")
        if self.boot != p.read(self.root / "confirmation/summary.json")["boot_id"]:
            raise ValueError("node34 must use the same study boot")
        self.libraries, self.totals, self.expected = {}, {}, {}
        for name, rec in self.nomination["programs"].items():
            if p.digest(rec["binary"]) != rec["binary_sha256"] or p.digest(rec["tir_path"]) != rec["tir_sha256"]:
                raise ValueError("historical node34 program changed")
            self.libraries[name] = Path(rec["binary"])
            self.totals[name], self.expected[name] = rec["request_totals"], rec["expected"]
        arrays = {}
        for name, rec in self.nomination["data_files"].items():
            if p.digest(rec["path"]) != rec["sha256"]:
                raise ValueError("full fusion snapshot/reference changed")
            arrays[name] = np.load(rec["path"])
        self.x, self.y = {s: arrays["X_%d" % s] for s in SEEDS}, {s: arrays["Y_%d" % s] for s in SEEDS}
        self.weight, self.weight_b, self.add, self.y_b = (arrays[k] for k in ("weight_A", "weight_B", "add", "Y_B"))
        if np.array_equal(self.y[0], self.y_b):
            raise ValueError("ABA reference does not distinguish weights")
        f, _ = function(Path(self.nomination["programs"]["N34-G2"]["directory"]), "N34-G2")
        abi = [{"shape": list(map(int, f.buffer_map[param].shape)), "dtype": str(f.buffer_map[param].dtype),
                "strides": list(map(str, f.buffer_map[param].strides)), "element_offset": str(f.buffer_map[param].elem_offset)}
               for param in f.params]
        other, _ = function(Path(self.nomination["programs"]["N34-Input"]["directory"]), "N34-Input")
        if [(list(map(int, other.buffer_map[x].shape)), str(other.buffer_map[x].dtype)) for x in other.params] != \
                [(b["shape"], b["dtype"]) for b in abi]:
            raise ValueError("node34 actual four-buffer ABI differs")
        self.data_contract = {"abi_order": ["X", "W", "ADD", "Y"], "buffers": abi,
                              "arithmetic": "int32 conv + broadcast ADD; arithmetic >>6; clip(-127,127); int8",
                              "independent_buffers_no_alias": True,
                              "reference_AB_mismatches": int(np.count_nonzero(self.y[0] != self.y_b)),
                              "nomination_sha256": p.digest(self.root / "node34_nomination.json")}
        def verify(name, profile, signature):
            exp = self.expected[name]
            if any(signature[k] != exp[k] for k in ("reset", "mac", "wgt_loads", "wgt_payload_bytes")) or \
                    profile["push_gemm_op_calls"] != exp["total_gemm_push"] or profile.get("driver_timeout_calls", 0):
                raise ValueError("actual node34 command signature differs")
            return {"mac": signature["mac"], "wgt_loads": signature["wgt_loads"]}
        proxy = SimpleNamespace(**vars(p))
        proxy.expectations, proxy.verify_call_profile = lambda name: self.expected[name], verify
        namespace = dict(base.__dict__, p=proxy, SYMBOL=SYMBOL)
        local = {}
        exec(compile(textwrap.dedent(inspect.getsource(base.Runner.call)), __file__ + ":native_call", "exec"), namespace, local)
        self.call = MethodType(local["call"], self)

    def record(self, row):
        row.setdefault("latency_ms", None)
        row.update(boot_id=self.boot, invocation_id=self.out.name + ":%06d" % len(self.rows))
        row["status"] = "PASS" if row.get("correct") and row.get("arguments_unchanged") else "FAIL"
        base.append(self.out / "calls.jsonl", row)
        base.append(self.out / "programs" / (row["program"] + ".jsonl"), row)
        self.rows.append(row)
        return row

    def contract(self, **extra):
        p.write_new(self.out / "contract.json", {"nomination_sha256": p.digest(self.root / "node34_nomination.json"),
                    "boot_id": self.boot, "data_contract": self.data_contract,
                    "sources": {__file__: p.digest(__file__), base.__file__: p.digest(base.__file__)},
                    "helpers": base.helper_certificate(), "programs": self.nomination["programs"], **extra})

    def qualify(self):
        self.start_phase("node34_qualification")
        self.contract(expected_calls={"correctness": 6, "weight_ABA": 6, "first_call_diagnostic": 2,
                                      "diagnostic_warmup": 18, "warmed_diagnostic": 18})
        for name in PROGRAMS:
            with self.session(name, self.libraries[name]) as (remote, lib, funcs, refs, sid):
                buffers = self.buffers(remote, 0)
                handles = base.buffer_handles(buffers)
                for seed in SEEDS:
                    self.call(lib, funcs, name, buffers, seed, "correctness", sid)
                for label, weight, output in (("A", self.weight, self.y[0]), ("B", self.weight_b, self.y_b),
                                               ("A_return", self.weight, self.y[0])):
                    self.call(lib, funcs, name, buffers, 0, "weight_ABA", sid, weight=weight,
                              expected=output, aba_label=label)
                    if base.buffer_handles(buffers) != handles:
                        raise ValueError("node34 ABA handles changed")
        signatures = {}
        for name in PROGRAMS:
            with self.session(name, self.libraries[name]) as (remote, lib, funcs, refs, sid):
                refs.append(base.load_ephemeral(remote, base.DIAGNOSTIC))
                diag = remote.get_function("c3.node24.fused_diagnostic_once")
                row = self.call(lib, funcs, name, self.buffers(remote, 0), 0, "first_call_diagnostic", sid,
                                diagnostic=lambda module, _symbol, *buffers: diag(module, SYMBOL, *buffers), call_position=1)
                signatures[name] = {"first_call": row["command_signature"], "rotations": {}}
            for rotation in range(3):
                seeds = p.seed_order(rotation)
                with self.session(name, self.libraries[name]) as (remote, lib, funcs, refs, sid):
                    refs.append(base.load_ephemeral(remote, base.DIAGNOSTIC))
                    diag = remote.get_function("c3.node24.fused_diagnostic_once")
                    buffers = self.buffers(remote, seeds[0])
                    for i, seed in enumerate(seeds):
                        self.call(lib, funcs, name, buffers, seed, "diagnostic_warmup", sid, rotation=rotation, call_position=i + 1)
                    values = []
                    for i, seed in enumerate(seeds):
                        row = self.call(lib, funcs, name, buffers, seed, "warmed_diagnostic", sid,
                                        diagnostic=lambda module, _symbol, *buf: diag(module, SYMBOL, *buf),
                                        rotation=rotation, call_position=i + 4)
                        values.append(row["command_signature"])
                    signatures[name]["rotations"][str(rotation)] = values
        expected = {"correctness": 6, "weight_ABA": 6, "first_call_diagnostic": 2,
                    "diagnostic_warmup": 18, "warmed_diagnostic": 18}
        if dict(Counter(r["call_kind"] for r in self.rows)) != expected:
            raise ValueError("node34 qualification ledger differs")
        p.write_new(self.out / "diagnostic_signatures.json", signatures)
        p.write_new(self.out / "summary.json", {"status": "NODE34_KNOWN_INPUT_QUALIFIED", "boot_id": self.boot,
                    "call_counts": expected, "all_correct": True, "final_state": self.final_state()})
        print("NODE34_KNOWN_INPUT_QUALIFIED", flush=True)

    def confirmation(self):
        summary = p.read(self.root / "node34_qualification/summary.json")
        if summary["status"] != "NODE34_KNOWN_INPUT_QUALIFIED" or summary["boot_id"] != self.boot:
            raise ValueError("node34 qualification absent/different boot")
        signatures = p.read(self.root / "node34_qualification/diagnostic_signatures.json")
        self.start_phase("node34_confirmation")
        orders = self.nomination["confirmation_orders"]
        self.contract(orders=orders, warmup_calls=18, timing_calls=18, selected_before_timing="N34-Input")
        for r, order in enumerate(orders):
            seeds = p.seed_order(r)
            for name in order:
                with self.session(name, self.libraries[name]) as (remote, lib, funcs, refs, sid):
                    refs.append(base.load_ephemeral(remote, base.TIMER))
                    reset, count = remote.get_function("c3.node34.reset_counts"), remote.get_function("c3.node34.counts")
                    timer = remote.get_function("c3.node34.fused_timed_once")
                    reset()
                    buffers = self.buffers(remote, seeds[0])
                    for i, seed in enumerate(seeds):
                        self.call(lib, funcs, name, buffers, seed, "warmup", sid, round=r, call_position=i + 1)
                    for i, seed in enumerate(seeds):
                        row = self.call(lib, funcs, name, buffers, seed, "timing", sid,
                                        timer=lambda module, _symbol, *buf: timer(module, SYMBOL, *buf),
                                        signature=signatures[name]["rotations"][str(r)][i], round=r, call_position=i + 4)
                        if int(count()) != i + 1:
                            raise ValueError("node34 hidden target call")
                        base.append(self.out / "helper_counts.jsonl", {"invocation_id": row["invocation_id"],
                                                                      "expected": i + 1, "count": int(count())})
        if dict(Counter(r["call_kind"] for r in self.rows)) != {"warmup": 18, "timing": 18}:
            raise ValueError("node34 confirmation incomplete")
        rounds = {r: {name: median(x["latency_ms"] for x in self.rows if x["call_kind"] == "timing"
                                  and x["round"] == r and x["program"] == name) for name in PROGRAMS} for r in range(3)}
        gains = [100 * (1 - rounds[r]["N34-Input"] / rounds[r]["N34-G2"]) for r in range(3)]
        result = {"status": "NODE34_KNOWN_INPUT_INDEPENDENTLY_CONFIRMED", "round_medians_ms": rounds,
                  "median_ms": {n: median(rounds[r][n] for r in range(3)) for n in PROGRAMS},
                  "paired_improvements_pct": gains, "paired_median_improvement_pct": median(gains),
                  "wins": sum(g > 0 for g in gains), "graph_eligible": median(gains) >= 2 and all(g > 0 for g in gains),
                  "not_latency_unseen_or_new_residency_mechanism": True}
        p.write_new(self.out / "results.json", result)
        p.write_new(self.out / "summary.json", {"status": result["status"], "boot_id": self.boot,
                    "graph_eligible": result["graph_eligible"], "call_counts": {"warmup": 18, "timing": 18},
                    "all_correct": True, "final_state": self.final_state()})
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("qualify", "confirmation"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="192.168.1.247")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--known-hosts", default=str(Path.home() / ".ssh/known_hosts"))
    parser.add_argument("--execute-board", action="store_true")
    args = parser.parse_args()
    if not args.execute_board:
        raise SystemExit("explicit --execute-board required")
    runner = None
    try:
        runner = Runner(args)
        getattr(runner, args.phase)()
    except BaseException as error:
        out = runner.out if runner is not None and runner.out is not None else args.output_dir / "node34_preflight"
        out.mkdir(exist_ok=True)
        p.write_new(out / "invalid_attempt.json", {"status": "STOP_SUBSEQUENT_BOARD_WORK", "error": str(error),
                    "traceback": traceback.format_exc(), "no_retry": True})
        raise
