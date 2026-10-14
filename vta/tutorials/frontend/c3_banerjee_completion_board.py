"""One-boot, common-runtime comparison of analytical ties and sealed programs."""
import inspect
import json
from collections import Counter
from pathlib import Path
import random
from statistics import median
import textwrap
from types import SimpleNamespace
import numpy as np
import tvm

import c3_residency_tradeoff as p
import c3_residency_tradeoff_board as base
from c3_banerjee_baseline import HERE, GROUPED, digest, write_new
from prepare_c3_banerjee_completion import TPS
from audit_c3_banerjee_completion import native_certificate
from c3_five_tradeoff_commands import certify_prefetch_dump, decode_submissions, completion_dag

PROGRAMS = TPS + ("R13-G", "N24-R12-G2-PREFETCH", "Input-7x7")
SCHEMA = "c3_banerjee_software_baseline_completion_v2"
_rng = random.Random(20261013)
_rows = []
for _ in range(3):
    row = list(PROGRAMS)
    _rng.shuffle(row)
    _rows.append(tuple(row))
ORDERS = tuple(_rows + [tuple(reversed(row)) for row in _rows])
_EXPECTATIONS = {}
_PROXIES = []


def balance():
    pair = Counter((a, b) for row in ORDERS for i, a in enumerate(row) for b in row[i+1:])
    if set(pair.values()) != {3}:
        raise ValueError("pairwise before/after balance failed")
    pos = Counter((name, i) for row in ORDERS for i, name in enumerate(row))
    return {"every_pair_each_direction_three_rounds": True,
            "positions": {"%s@%d" % key: val for key, val in pos.items()},
            "not_full_position_or_predecessor_balance": True}


def expectations(name):
    return _EXPECTATIONS[name]


def verify_profile(name, profile, signature):
    exp = expectations(name)
    if any(signature[k] != exp[k] for k in ("mac", "reset", "wgt_loads", "wgt_payload_bytes")):
        raise ValueError("actual commands differ from qualified program")
    if profile.get("push_gemm_op_calls") != exp["total_gemm_push"] or profile.get("driver_timeout_calls", 0):
        raise ValueError("target call count or driver result invalid")
    return {"mac": exp["mac"], "wgt_loads": exp["wgt_loads"]}


def execution_sources():
    sources = base.execution_sources()
    for name in ("c3_banerjee_completion_board.py", "run_c3_banerjee_completion_board.py",
                 "audit_c3_banerjee_completion.py", "c3_banerjee_completion.py",
                 "build_c3_banerjee_runtime.py", "c3_banerjee_uop_queue.h",
                 "run_c3_banerjee_completion.py", "run_c3_banerjee_anchor_sim.py",
                 "prepare_c3_banerjee_completion.py", "c3_five_tradeoff_commands.py", "c3_thread_sync_audit.py"):
        sources[str(HERE / name)] = digest(HERE / name)
    return sources


def clone(method, replacements=()):
    source = textwrap.dedent(inspect.getsource(method))
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError("experiment adaptation anchor not unique: " + old[:80])
        source = source.replace(old, new)
    proxy = SimpleNamespace(**vars(p))
    proxy.PROGRAMS, proxy.ORDERS, proxy.SCHEMA = PROGRAMS, ORDERS, SCHEMA
    proxy.expectations, proxy.verify_call_profile, proxy.balance = expectations, verify_profile, balance
    _PROXIES.append(proxy)
    ns = dict(base.__dict__, p=proxy, execution_sources=execution_sources)
    # Runtime metadata path depends on the root, installed by __init__ below.
    ns["RUNTIME"] = None
    result = {}
    exec(compile(source, __file__ + ":" + method.__name__, "exec"), ns, result)
    return result[method.__name__], ns


_init, _init_ns = clone(base.Runner.__init__, ((
    'p.read(directory / "result.json")["request_totals"]', 'record["request_totals"]'),))
_call, _ = clone(base.Runner.call)
_qualify, _ = clone(base.Runner.qualify, tuple((old, new) for old, new in (
    ('kinds["correctness"] != 18', 'kinds["correctness"] != 3 * len(p.PROGRAMS)'),
    ('kinds["weight_ABA"] != 18', 'kinds["weight_ABA"] != 3 * len(p.PROGRAMS)'),
    ('kinds["warmed_diagnostic"] != 54', 'kinds["warmed_diagnostic"] != 9 * len(p.PROGRAMS)'),
    ('kinds["diagnostic_warmup"] != 54', 'kinds["diagnostic_warmup"] != 9 * len(p.PROGRAMS)'),
    ('kinds["first_call_diagnostic"] != 6', 'kinds["first_call_diagnostic"] != len(p.PROGRAMS)'),
    ('"SIX_PROGRAMS_FPGA_AND_COMMAND_QUALIFIED"', '"POOL_FPGA_AND_COMMAND_QUALIFIED"'))))
_formal, _ = clone(base.Runner.formal, (
    ('"SIX_PROGRAMS_FPGA_AND_COMMAND_QUALIFIED"', '"POOL_FPGA_AND_COMMAND_QUALIFIED"'),
    ('"model": {"calibration": p.CALIBRATION, "validation": p.VALIDATION,\n'
     '                  "nonnegative": True, "threshold_absolute_relative_pct": 10}',
     '"analysis_scope": "fixed complete-software candidates and common runtime; no model fit"'),
    ('if kinds != {"warmup": 108, "timing": 108}:',
     'if kinds != {"warmup": len(p.PROGRAMS)*18, "timing": len(p.PROGRAMS)*18}:'),
    ('"timed_calls": 108, "warmup_calls": 108,',
     '"timed_calls": len(p.PROGRAMS)*18, "warmup_calls": len(p.PROGRAMS)*18,')))


class Runner(base.Runner):
    def __init__(self, args):
        self.completion_root = args.output_dir.resolve()
        if args.phase == "formal":
            selected = p.read(self.completion_root / "selected_pool.json")
            for proxy in _PROXIES:
                proxy.PROGRAMS = tuple(selected["programs"])
                proxy.ORDERS = tuple(tuple(row) for row in selected["orders"])
                proxy.balance = lambda: {"six_program_Williams_order": True}
                proxy.seed_order = lambda _r: p.SEEDS
        _init_ns["RUNTIME"] = self.completion_root / "runtime_contract.json"
        _EXPECTATIONS.clear()
        _EXPECTATIONS.update(p.read(self.completion_root / "protocol.json")["expectations"])
        _init(self, args)

    call_raw = _call
    formal = _formal

    def record(self, row):
        row.setdefault("latency_ms", None)
        row.update(boot_id=self.boot, invocation_id=self.out.name + ":%06d" % len(self.rows))
        row["status"] = "PASS" if row.get("correct") and row.get("arguments_unchanged") else "FAIL"
        base.append(self.out / "calls.jsonl", row)
        if row.get("program") in PROGRAMS:
            base.append(self.out / "programs" / (row["program"] + ".jsonl"), row)
        self.rows.append(row)
        return row

    def qualify(self):
        self.start_phase("qualification")
        infra = self.root / "infrastructure"
        infra.mkdir(exist_ok=False)
        counter = base.compile_counter(infra)
        helpers = base.helper_certificate()
        write_new(self.out / "contract.json", {"schema": SCHEMA, "boot_id": self.boot,
            "data_contract": self.data_contract, "prepared_sha256": digest(self.root / "prepared.json"),
            "helpers": helpers, "sources": execution_sources(), "no_reboot_or_SD_writes": True})
        with self.session("helper_regression", counter) as (remote, lib, funcs, refs, sid):
            refs.append(base.load_ephemeral(remote, base.TIMER))
            reset = remote.get_function("c3.node34.reset_counts")
            counts = remote.get_function("c3.node34.counts")
            timer = remote.get_function("c3.node34.fused_timed_once")
            buffers = tuple(tvm.nd.array(np.array([v], dtype="int32"), remote.cpu(0)) for v in (1, 2, 3, 0))
            reset()
            for i in range(3):
                funcs["vta.runtime.profiler_clear"]()
                timer(lib, "c3_four_arg_counter", *buffers)
                profile = json.loads(str(funcs["vta.runtime.profiler_status"]()))
                if int(buffers[-1].numpy()[0]) != 123*(i+1) or int(counts()) != i+1 or profile["driver_run_calls"]:
                    raise ValueError("single-call timer/runtime regression failed")
                self.record({"program": "helper_regression", "call_kind": "helper_regression", "session_id": sid,
                    "correct": True, "arguments_unchanged": True, "helper_calls": int(counts()), "seed": None})
        signatures = {}
        for name in PROGRAMS:
            with self.session(name, self.libraries[name]) as (remote, lib, funcs, refs, sid):
                refs.append(base.load_ephemeral(remote, base.DIAGNOSTIC))
                diag = remote.get_function("c3.node24.fused_diagnostic_once")
                buffers = self.buffers(remote, p.SEEDS[0])
                first = None
                for i, seed in enumerate(p.SEEDS):
                    row = self.call(lib, funcs, name, buffers, seed, "correctness", sid,
                        diagnostic=diag if i == 0 else None)
                    if i == 0: first = row["command_signature"]
                handles = base.buffer_handles(buffers)
                for label, weight, expected in (("A", self.weight, self.y[0]), ("B", self.weight_b, self.y_b),
                                                 ("A_return", self.weight, self.y[0])):
                    self.call(lib, funcs, name, buffers, 0, "weight_ABA", sid, weight=weight, expected=expected, aba_label=label)
                    if base.buffer_handles(buffers) != handles:
                        raise ValueError("ABA changed weight buffer handle")
            with self.session(name, self.libraries[name]) as (remote, lib, funcs, refs, sid):
                refs.append(base.load_ephemeral(remote, base.DIAGNOSTIC))
                diag = remote.get_function("c3.node24.fused_diagnostic_once")
                buffers = self.buffers(remote, p.SEEDS[0])
                for pos, seed in enumerate(p.SEEDS):
                    self.call(lib, funcs, name, buffers, seed, "diagnostic_warmup", sid, call_position=pos+1)
                warmed = [self.call(lib, funcs, name, buffers, seed, "warmed_diagnostic", sid,
                    diagnostic=diag, call_position=pos+4)["command_signature"] for pos, seed in enumerate(p.SEEDS)]
            # All formal cells use exactly this seed order and initialization.
            signatures[name] = {"first_call": first, "rotations": {str(r): warmed for r in range(3)}}
            print("FPGA correctness, ABA and completion qualified: " + name, flush=True)
        kinds = Counter(r["call_kind"] for r in self.rows)
        expected = {"correctness": 3*len(PROGRAMS), "weight_ABA": 3*len(PROGRAMS),
                    "diagnostic_warmup": 3*len(PROGRAMS), "warmed_diagnostic": 3*len(PROGRAMS), "helper_regression": 3}
        if kinds != expected:
            raise ValueError("qualification call ledger mismatch")
        state = self.final_state()
        write_new(self.out / "diagnostic_signatures.json", signatures)
        write_new(self.out / "summary.json", {"schema": SCHEMA, "status": "POOL_FPGA_AND_COMMAND_QUALIFIED",
            "boot_id": self.boot, "all_correct": True, "call_counts": dict(kinds), "final_state": state,
            "target_fpga_calls": len(self.rows)-3, "prepared_sha256": digest(self.root / "prepared.json"),
            "contract_sha256": digest(self.out / "contract.json"),
            "diagnostic_signatures_sha256": digest(self.out / "diagnostic_signatures.json")})

    def select(self):
        qualified = p.read(self.root / "qualification/summary.json")
        if qualified["status"] != "POOL_FPGA_AND_COMMAND_QUALIFIED" or qualified["boot_id"] != self.boot:
            raise ValueError("selection requires a qualified pool on this boot")
        self.start_phase("selection")
        signatures = p.read(self.root / "qualification/diagnostic_signatures.json")
        scores = {}
        # Only paper-score tied TPS candidates are selected. Own programs stay fixed.
        for name in TPS:
            with self.session(name, self.libraries[name]) as (remote, lib, funcs, refs, sid):
                refs.append(base.load_ephemeral(remote, base.TIMER))
                timer = remote.get_function("c3.node34.fused_timed_once")
                reset = remote.get_function("c3.node34.reset_counts")
                count = remote.get_function("c3.node34.counts")
                reset()
                buffers = self.buffers(remote, p.SEEDS[0])
                for pos, seed in enumerate(p.SEEDS):
                    self.call(lib, funcs, name, buffers, seed, "selection_warmup", sid, call_position=pos+1)
                samples = []
                for pos, seed in enumerate(p.SEEDS):
                    row = self.call(lib, funcs, name, buffers, seed, "selection_timing", sid, timer=timer,
                        signature=signatures[name]["rotations"]["0"][pos], call_position=pos+4)
                    if int(count()) != pos+1: raise ValueError("extra selection invocation")
                    samples.append(row["latency_ms"])
                scores[name] = median(samples)
            print("TPS analytical tie measured: " + name, flush=True)
        winners = tuple(min((n for n in TPS if n.startswith(prefix)), key=lambda n: (scores[n], n))
                        for prefix in ("TPS-single", "TPS-oc-reuse", "TPS-h-reuse"))
        programs = winners + ("R13-G", "N24-R12-G2-PREFETCH", "Input-7x7")
        orders = [[programs[i] for i in row] for row in ((0,1,5,2,4,3),(1,2,0,3,5,4),(2,3,1,4,0,5),
                                                        (3,4,2,5,1,0),(4,5,3,0,2,1),(5,0,4,1,3,2))]
        self.final_state()
        if Counter(r["call_kind"] for r in self.rows) != {"selection_warmup": 30, "selection_timing": 30}:
            raise ValueError("selection call ledger mismatch")
        write_new(self.root / "selected_pool.json", {"status": "TPS_FAMILY_WINNERS_SEALED_BEFORE_CONFIRMATION",
            "boot_id": self.boot, "programs": programs, "orders": orders, "scores_ms": scores,
            "primary_TPS_baseline": min(TPS, key=lambda n: (scores[n], n)),
            "selection_data_sha256": digest(self.out / "calls.jsonl"), "own_programs_not_reselected": True})

    def call(self, *args, **kwargs):
        row = self.call_raw(*args, **kwargs)
        if row.get("dump_file"):
            text = (self.out / row["dump_file"]).read_text()
            if row["program"] in TPS:
                contract = p.read(self.root / "binaries" / row["program"] / "contract.json")
                proof = native_certificate(text, contract)
            elif row["program"] == "N24-R12-G2-PREFETCH":
                proof = certify_prefetch_dump(text)
            else:
                for block in decode_submissions(text):
                    completion_dag(block)
                proof = {"status": "NATIVE_UOP_AND_CLOSURE_PASSED", "sealed_data_schedule_unchanged": True}
            base.append(self.out / "native_certificates.jsonl", {
                "invocation_id": row["invocation_id"], "program": row["program"], "certificate": proof})
        return row


def freeze(root):
    records, exp = {}, {}
    old = p.read(GROUPED / "20261012_c3_banerjee_double_buffer_compare_run01/prepared.json")["programs"]
    local_index = p.read(root / "local_index.json")
    for name in PROGRAMS:
        if name in TPS:
            directory = root / "binaries" / name
            result = p.read(directory / "result.json")
            proof = p.read(Path(local_index[name]) / "native_certificate.json")
            first = proof["proof"][0]["capacity"]
            binary = directory / "graphlib.so"
            totals = result["request_totals"]
            if result["status"] != "ARM_MATCHES_QUALIFIED_SIM_FPGA_PENDING":
                raise ValueError("candidate is not cross-qualified")
        else:
            source = old[name]
            directory, binary = Path(source["directory"]), Path(source["binary"])
            if digest(binary) != source["binary_sha256"] or digest(directory / "node24_final.tir.json") != source["tir_sha256"]:
                raise ValueError("sealed anchor identity changed")
            anchor = root / ("anchors_final" if name == "Input-7x7" else "anchors_attempt02") / name
            result = p.read(anchor / "result.json")
            if result["status"] != "SEALED_PROGRAM_NEW_RUNTIME_FSIM_PASSED":
                raise ValueError("anchor not correct in common runtime")
            first = p.read(anchor / "native_certificate.json")["proof"][0]
            totals = result["request_totals"]
            if name == "Input-7x7":
                directory = root / "binaries" / name
                binary = directory / "graphlib.so"
        signature = base.dump_signature(first)
        exp[name] = {"mac": signature["mac"], "reset": signature["reset"],
                     "total_gemm_push": signature["mac"] + signature["reset"],
                     "wgt_loads": signature["wgt_loads"], "wgt_payload_bytes": signature["wgt_payload_bytes"]}
        records[name] = {"directory": str(directory), "binary": str(binary), "binary_sha256": digest(binary),
                         "tir_sha256": digest(directory / "node24_final.tir.json"), "request_totals": totals}
    runtime = p.read(base.RUNTIME)
    runtime["runtime_paths"]["libvta.so"] = str(root / "runtime_board_attempt02/libvta.so")
    runtime["runtime_sha256"]["libvta.so"] = digest(root / "runtime_board_attempt02/libvta.so")
    runtime["common_experimental_runtime"] = True
    write_new(root / "runtime_contract.json", runtime)
    protocol = {"schema": SCHEMA, "programs": PROGRAMS, "orders": ORDERS, "balance": balance(),
        "seeds": p.SEEDS, "expectations": exp, "source_sha256": execution_sources(),
        "document": str(root / "PROTOCOL.md"), "document_sha256": digest(root / "PROTOCOL.md"),
        "helpers": base.helper_certificate(), "runtime_sha256": runtime["runtime_sha256"],
        "expected_calls": {"FPGA_qualification_and_diagnostics": len(PROGRAMS)*12,
                           "selection_warmup": 30, "selection_timing": 30,
                           "warmup": 108, "timing": 108, "helper_regression": 3},
        "latency_selection": "all analytical ci ties measured; one per family sealed before independent formal data",
        "no_full_author_hardware_or_TPS_implementation_claim": True}
    write_new(root / "protocol.json", protocol)
    write_new(root / "prepared.json", {"schema": SCHEMA, "status": "ALL_LOCAL_GATES_PASSED_BOARD_NOT_QUALIFIED",
        "programs": records, "protocol_sha256": digest(root / "protocol.json")})


def analyze(root):
    summary = p.read(root / "board/summary.json")
    if summary["status"] != "FORMAL_ACQUISITION_COMPLETE" or (root / "board/invalid_attempt.json").exists():
        raise ValueError("incomplete or invalid timing attempt")
    rows = [json.loads(s) for s in (root / "board/calls.jsonl").read_text().splitlines()]
    selected = p.read(root / "selected_pool.json")
    programs = tuple(selected["programs"])
    medians = p.round_medians(rows, programs)
    values = {n: median(medians[r][n] for r in range(6)) for n in programs}
    # Descriptive reference fastest, not a frozen deployment winner.
    baselines = tuple(n for n in programs if n in TPS)
    best = min(baselines, key=lambda n: (values[n], n))
    result = {"status": "COMMON_RUNTIME_SINGLE_BOOT_COMPARISON_COMPLETE", "boot_id": summary["boot_id"],
        "median_ms": values, "round_medians": medians, "fastest_measured_TPS_reference": best,
        "primary_TPS_baseline": selected["primary_TPS_baseline"],
        "comparisons": [p.compare(medians, "N24-R12-G2-PREFETCH", n) for n in baselines + ("Input-7x7", "R13-G")],
        "all_correct": True, "not_graph_FPS": True, "raw_calls_sha256": digest(root / "board/calls.jsonl")}
    write_new(root / "results.json", result)
    print(json.dumps({"median_ms": values, "fastest_TPS_reference": best}, indent=2))
