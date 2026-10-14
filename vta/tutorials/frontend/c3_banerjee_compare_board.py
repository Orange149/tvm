"""Six fixed programs, isolated adapter over sealed single-call board machinery."""

from collections import Counter
import inspect
import json
from pathlib import Path
from statistics import median
import textwrap
from types import SimpleNamespace

import c3_residency_tradeoff as p
import c3_residency_tradeoff_board as base
from audit_c3_banerjee_local import native_certificate
from c3_five_tradeoff_commands import certify_prefetch_dump

HERE = Path(__file__).resolve().parent
GROUPED = base.OLD.parent
LOCAL = GROUPED / "20261012_c3_banerjee_double_buffer_local_run01"
FIVE = GROUPED / "20261012_c3_five_tradeoff_selection_run01"
SCHEMA = "c3_banerjee_six_program_comparison_v1"
NEW = "N24-R12-G2-PREFETCH"
PROGRAMS = ("Ordinary-DB", "Banerjee-DB-Adapted", "R13-G", "R12-G", NEW, "Input-7x7")
ORDERS = tuple(tuple(PROGRAMS[i] for i in row) for row in
               ((0, 1, 5, 2, 4, 3), (1, 2, 0, 3, 5, 4), (2, 3, 1, 4, 0, 5),
                (3, 4, 2, 5, 1, 0), (4, 5, 3, 0, 2, 1), (5, 0, 4, 1, 3, 2)))


def expectations(name):
    if name not in PROGRAMS:
        raise ValueError("outside frozen pool")
    if name in PROGRAMS[:2]:
        return dict(reset=112, mac=896, total_gemm_push=1008,
                    wgt_loads=896, wgt_payload_bytes=2016 * 1024)
    return p.expectations("R12-G" if name == NEW else name)


def verify_profile(name, profile, signature):
    exp = expectations(name)
    if any(signature[k] != exp[k] for k in ("reset", "mac", "wgt_loads", "wgt_payload_bytes")):
        raise ValueError("compiled/actual command mismatch")
    if (profile.get("push_gemm_op_calls") != exp["total_gemm_push"] or
            profile.get("driver_timeout_calls", 0) or profile.get("driver_run_calls", 0) <= 0):
        raise ValueError("incomplete or different device work")
    return {"mac": signature["mac"], "wgt_loads": signature["wgt_loads"]}


def balance():
    positions = Counter((n, i) for row in ORDERS for i, n in enumerate(row))
    adjacent = Counter((a, b) for row in ORDERS for a, b in zip(row, row[1:]))
    if len(positions) != 36 or set(positions.values()) != {1} or len(adjacent) != 30 or set(adjacent.values()) != {1}:
        raise ValueError("unbalanced frozen order")
    return {"positions_once": True, "directed_within_round_neighbors_once": True}


def execution_sources():
    paths = set(base.execution_sources())
    paths.update(str(HERE / n) for n in (
        "c3_banerjee_compare_board.py", "run_c3_banerjee_compare.py", "test_c3_banerjee_compare.py",
        "audit_c3_banerjee_local.py", "c3_five_tradeoff_commands.py", "c3_thread_sync_audit.py"))
    paths.update(p.read(LOCAL / "local_manifest.json")["sources"])
    return {path: p.digest(path) for path in sorted(paths)}


def adapt(method, replacements=()):
    """Clone only the experiment wrapper; original module and functions stay untouched."""
    source = textwrap.dedent(inspect.getsource(method))
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError("sealed adapter anchor is not unique")
        source = source.replace(old, new)
    proxy = SimpleNamespace(**vars(p))
    proxy.PROGRAMS, proxy.ORDERS, proxy.SCHEMA = PROGRAMS, ORDERS, SCHEMA
    proxy.expectations, proxy.verify_call_profile, proxy.balance = expectations, verify_profile, balance
    namespace = dict(base.__dict__, p=proxy, execution_sources=execution_sources)
    result = {}
    exec(compile(source, __file__ + ":" + method.__name__, "exec"), namespace, result)
    return result[method.__name__]


def freeze(root):
    root = Path(root).resolve()
    manifest = p.read(LOCAL / "local_manifest.json")
    for rel, sha in manifest["artifacts"].items():
        if p.digest(LOCAL / rel) != sha:
            raise ValueError("sealed local artifact changed: " + rel)
    for path, sha in manifest["sources"].items():
        if p.digest(path) != sha:
            raise ValueError("sealed local source changed: " + path)
    qualification = p.read(LOCAL / "local_qualification.json")
    if qualification["status"] != "LOCAL_FULL_FUSION_AND_NATIVE_LIFETIME_AUDIT_PASSED":
        raise ValueError("local qualification absent")
    records = p.read(FIVE / "rule_freeze.json")["programs"]
    for label, key in zip(PROGRAMS[:2], ("ordinary", "reuse")):
        rec = manifest["ARM_programs"][key]
        directory = Path(rec["directory"])
        records[label] = dict(directory=str(directory), binary=str(directory / "graphlib.so"),
                              binary_sha256=rec["binary_sha256"], tir_sha256=rec["TIR_sha256"],
                              request_totals=qualification["programs"][key]["request_totals"])
    for label in PROGRAMS[2:]:
        records[label]["request_totals"] = p.read(Path(records[label]["directory"]) / "result.json")["request_totals"]
    records = {n: records[n] for n in PROGRAMS}
    for label, rec in records.items():
        if p.digest(rec["binary"]) != rec["binary_sha256"] or p.digest(Path(rec["directory"]) / "node24_final.tir.json") != rec["tir_sha256"]:
            raise ValueError("program identity changed: " + label)
    protocol = dict(schema=SCHEMA, source_sha256=execution_sources(), document=str(root / "PROTOCOL.md"),
                    document_sha256=p.digest(root / "PROTOCOL.md"), orders=ORDERS, balance=balance(),
                    seeds=p.SEEDS, helpers=base.helper_certificate(),
                    local_manifest_sha256=p.digest(LOCAL / "local_manifest.json"),
                    historical_rule_sha256=p.digest(FIVE / "rule_freeze.json"),
                    expected_calls=dict(qualification_FPGA=150, software_helper=3, formal_warmup=108, formal_timing=108),
                    analysis_scope="fixed program comparison; no model fitting, reselect, graph or reboot",
                    primary_candidate=NEW, primary_baseline=PROGRAMS[1],
                    primary_engineering_gate=dict(median_paired_gain_pct=2, wins=5))
    p.write_new(root / "protocol.json", protocol)
    p.write_new(root / "prepared.json", dict(schema=SCHEMA,
        status="ALL_LOCAL_GATES_PASSED_BOARD_NOT_QUALIFIED", programs=records,
        protocol_sha256=p.digest(root / "protocol.json")))


class Runner(base.Runner):
    __init__ = adapt(base.Runner.__init__, ((
        'p.read(directory / "result.json")["request_totals"]', 'record["request_totals"]'),))
    _native_call = adapt(base.Runner.call)
    qualify = adapt(base.Runner.qualify)
    formal = adapt(base.Runner.formal, ((
        '"model": {"calibration": p.CALIBRATION, "validation": p.VALIDATION,\n'
        '                  "nonnegative": True, "threshold_absolute_relative_pct": 10}',
        '"analysis_scope": "comparison only; no model fit or winner selection"'),))

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
        if row.get("dump_file"):
            text = (self.out / row["dump_file"]).read_text()
            if row["program"] in PROGRAMS[:2]:
                proof = native_certificate(text, row["program"] == PROGRAMS[1])
            elif row["program"] == NEW:
                proof = certify_prefetch_dump(text)
            else:
                proof = None
            if proof is not None:
                base.append(self.out / "native_certificates.jsonl", dict(
                    invocation_id=row["invocation_id"], program=row["program"],
                    dump_sha256=row["dump_sha256"], actual_FPGA_encoding=True, certificate=proof))
        return row


def analyze(root):
    root = Path(root)
    summary = p.read(root / "board/summary.json")
    if summary["status"] != "FORMAL_ACQUISITION_COMPLETE" or (root / "board/invalid_attempt.json").exists():
        raise ValueError("incomplete or invalid session")
    rows = [json.loads(line) for line in (root / "board/calls.jsonl").read_text().splitlines()]
    if Counter(r["call_kind"] for r in rows) != {"warmup": 108, "timing": 108}:
        raise ValueError("call ledger mismatch")
    medians = p.round_medians(rows, PROGRAMS)
    result = dict(status="SINGLE_BOOT_SIX_PROGRAM_COMPARISON_COMPLETE", boot_id=summary["boot_id"],
                  round_medians=medians, median_ms={n: median(medians[r][n] for r in range(6)) for n in PROGRAMS},
                  comparisons=[p.compare(medians, NEW, n) for n in PROGRAMS if n != NEW],
                  source_sha256=execution_sources(), raw_calls_sha256=p.digest(root / "board/calls.jsonl"),
                  no_graph_FPS_or_general_buffer_optimum_claim=True)
    result["program_profiles"] = {n: {key: median(r["runtime_profile"][key] for r in rows
        if r["program"] == n and r["call_kind"] == "timing") for key in (
        "device_run_wait_us", "driver_run_calls", "driver_run_insns", "push_gemm_op_calls")}
        for n in PROGRAMS}
    p.write_new(root / "comparison.json", result)
    print(json.dumps(result, indent=2))

