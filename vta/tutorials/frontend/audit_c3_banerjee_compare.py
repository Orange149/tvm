#!/usr/bin/env python3
"""Read-only independent gate over actual FPGA journals before formal timing."""
import argparse
from collections import Counter
import json
from pathlib import Path

from c3_banerjee_compare_board import PROGRAMS, NEW, ORDERS, execution_sources, expectations
from c3_residency_tradeoff import digest, read, write_new, SEEDS, seed_order, round_medians
from c3_residency_tradeoff_board import dump_signature, validate_request_profile, helper_certificate
from c3_thread_sync_audit import instruction_dump
from audit_c3_banerjee_local import native_certificate
from c3_five_tradeoff_commands import certify_prefetch_dump


def audit(root):
    root = Path(root)
    out = root / "qualification"
    if (out / "invalid_attempt.json").exists() or (root / "invalid_attempt.json").exists():
        raise ValueError("invalid board attempt")
    summary, contract = read(out / "summary.json"), read(out / "contract.json")
    prepared, protocol = read(root / "prepared.json"), read(root / "protocol.json")
    if summary["status"] != "SIX_PROGRAMS_FPGA_AND_COMMAND_QUALIFIED" or \
            contract["sources"] != execution_sources() or contract["helpers"] != helper_certificate() or \
            contract["prepared_sha256"] != digest(root / "prepared.json") or \
            prepared["protocol_sha256"] != digest(root / "protocol.json") or \
            protocol["source_sha256"] != execution_sources():
        raise ValueError("unsealed or stale identity")
    rows = [json.loads(line) for line in (out / "calls.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (out / "dispatch.jsonl").read_text().splitlines()]
    target = [r for r in rows if r["program"] in PROGRAMS]
    if len(rows) != 153 or len(target) != 150 or len({r["invocation_id"] for r in rows}) != 153 or \
            {r["boot_id"] for r in rows} != {summary["boot_id"]}:
        raise ValueError("qualification ledger or boot inconsistent")
    for row in rows:
        if row["status"] != "PASS" or not row["correct"] or not row["arguments_unchanged"] or row["latency_ms"] is not None:
            raise ValueError("qualification is wrong or timed")
    if Counter((r["invocation_id"], r["event"]) for r in events) != Counter(
            (r["invocation_id"], event) for r in target for event in ("begin", "end")):
        raise ValueError("hidden, incomplete or duplicate dispatch")
    certificates = []
    for name in PROGRAMS:
        selected = [r for r in target if r["program"] == name]
        if Counter(r["call_kind"] for r in selected) != {
                "correctness": 3, "weight_ABA": 3, "first_call_diagnostic": 1,
                "diagnostic_warmup": 9, "warmed_diagnostic": 9}:
            raise ValueError("per-program ledger incomplete")
        if {r["seed"] for r in selected if r["call_kind"] == "correctness"} != set(SEEDS):
            raise ValueError("three seed coverage missing")
        aba = [r for r in selected if r["call_kind"] == "weight_ABA"]
        if ([r["aba_label"] for r in aba] != ["A", "B", "A_return"] or
                any(r["buffer_handles"] != aba[0]["buffer_handles"] or r["session_id"] != aba[0]["session_id"] for r in aba) or
                aba[0]["output_sha256"] == aba[1]["output_sha256"] or
                aba[0]["output_sha256"] != aba[2]["output_sha256"]):
            raise ValueError("same-handle ABA not established")
        for row in selected:
            if row["complete_operator_calls"] != 1 or row["hidden_warmup_calls"] != 0 or not row["profile_before_output_reads"]:
                raise ValueError("call boundary unproven")
            validate_request_profile(row["runtime_profile"], prepared["programs"][name]["request_totals"])
            if row.get("dump_file"):
                path = out / row["dump_file"]
                if digest(path) != row["dump_sha256"]:
                    raise ValueError("actual native dump changed")
                text = path.read_text()
                signature = dump_signature(instruction_dump(text))
                exp = expectations(name)
                if signature != row["command_signature"] or any(signature[k] != exp[k] for k in (
                        "reset", "mac", "wgt_loads", "wgt_payload_bytes")) or \
                        signature["instructions"] != row["runtime_profile"]["driver_run_insns"] or \
                        signature["submissions"] != row["runtime_profile"]["driver_run_calls"]:
                    raise ValueError("actual native/profile signature mismatch")
                if name in PROGRAMS[:2]:
                    proof = native_certificate(text, name == PROGRAMS[1])
                elif name == NEW:
                    proof = certify_prefetch_dump(text)
                else:
                    proof = {"status": "EXISTING_PROGRAM_CAPACITY_CHECK_PASSED"}
                certificates.append(dict(invocation_id=row["invocation_id"], program=name,
                                         dump_sha256=digest(path), proof=proof))
    sessions = [json.loads(line) for line in (out / "sessions.jsonl").read_text().splitlines()]
    if len(sessions) != 31 or any(s["status"] != "PASS" or s.get("cleanup_error") for s in sessions):
        raise ValueError("invalid session or cleanup")
    result = dict(status="INDEPENDENT_FPGA_COMPARISON_QUALIFICATION_PASSED", boot_id=summary["boot_id"],
                  target_calls=150, software_helper_calls=3, dump_certificates=certificates,
                  sources=execution_sources(), audit_source_sha256=digest(__file__),
                  artifacts={p.name: digest(p) for p in (out / "calls.jsonl", out / "dispatch.jsonl",
                    out / "sessions.jsonl", out / "summary.json", out / "contract.json",
                    out / "diagnostic_signatures.json")}, latency_ms=None)
    write_new(out / "independent_audit.json", result)
    return result


def audit_formal(root):
    root = Path(root)
    out = root / "board"
    if list(root.rglob("invalid_attempt.json")):
        raise ValueError("invalid attempt cannot contribute performance")
    qualification = read(root / "qualification/independent_audit.json")
    summary, contract = read(out / "summary.json"), read(out / "contract.json")
    prepared, protocol = read(root / "prepared.json"), read(root / "protocol.json")
    if (qualification["status"] != "INDEPENDENT_FPGA_COMPARISON_QUALIFICATION_PASSED" or
            summary["status"] != "FORMAL_ACQUISITION_COMPLETE" or
            summary["boot_id"] != qualification["boot_id"] or
            contract["sources"] != execution_sources() or protocol["source_sha256"] != execution_sources() or
            contract["helpers"] != helper_certificate() or contract["programs"] != prepared["programs"] or
            prepared["protocol_sha256"] != digest(root / "protocol.json")):
        raise ValueError("formal identity not bound to the qualified programs")
    for name, sha in qualification["artifacts"].items():
        if digest(root / "qualification" / name) != sha:
            raise ValueError("qualification artifact changed: " + name)
    for rec in prepared["programs"].values():
        if digest(rec["binary"]) != rec["binary_sha256"] or \
                digest(Path(rec["directory"]) / "node24_final.tir.json") != rec["tir_sha256"]:
            raise ValueError("measured binary changed")
    rows = [json.loads(x) for x in (out / "calls.jsonl").read_text().splitlines()]
    expected = [(r, n, kind, seed, pos + start) for r, order in enumerate(ORDERS)
                for n in order for kind, start in (("warmup", 1), ("timing", 4))
                for pos, seed in enumerate(seed_order(r))]
    observed = [(r["round"], r["program"], r["call_kind"], r["seed"], r["call_position"]) for r in rows]
    if observed != expected or len(rows) != 216 or len({r["session_id"] for r in rows}) != 36:
        raise ValueError("frozen order, seeds, positions or session count differ")
    signatures = read(root / "qualification/diagnostic_signatures.json")
    for i in range(0, len(rows), 6):
        cell = rows[i:i + 6]
        if any(r["session_id"] != cell[0]["session_id"] or
               r["buffer_handles"] != cell[0]["buffer_handles"] for r in cell):
            raise ValueError("cell buffer or RPC lifetime changed")
    for row in rows:
        if row["status"] != "PASS" or not row["correct"] or not row["arguments_unchanged"] or \
                row["complete_operator_calls"] != 1 or row["hidden_warmup_calls"] != 0:
            raise ValueError("invalid complete call")
        validate_request_profile(row["runtime_profile"], prepared["programs"][row["program"]]["request_totals"])
        if row["call_kind"] == "warmup":
            if row["latency_ms"] is not None:
                raise ValueError("warmup included in time data")
        else:
            sig = signatures[row["program"]]["rotations"][str(row["round"] % 3)][row["seed_position"]]
            if not row["diagnostic_signature_matched"] or \
                    row["runtime_profile"]["driver_run_insns"] != sig["instructions"] or \
                    row["runtime_profile"]["driver_run_calls"] != sig["submissions"]:
                raise ValueError("warmed diagnostic does not match formal call")
    events = [json.loads(x) for x in (out / "dispatch.jsonl").read_text().splitlines()]
    if Counter((e["invocation_id"], e["event"]) for e in events) != Counter(
            (r["invocation_id"], e) for r in rows for e in ("begin", "end")):
        raise ValueError("unexpected target dispatch")
    counters = [json.loads(x) for x in (out / "helper_counts.jsonl").read_text().splitlines()]
    timing = [r for r in rows if r["call_kind"] == "timing"]
    if len(counters) != 108 or any(c["invocation_id"] != r["invocation_id"] or
            c["count"] != r["seed_position"] + 1 or c["expected"] != c["count"]
            for c, r in zip(counters, timing)):
        raise ValueError("single-call timer counter mismatch")
    sessions = [json.loads(x) for x in (out / "sessions.jsonl").read_text().splitlines()]
    if len(sessions) != 36 or any(s["status"] != "PASS" or s.get("cleanup_error") for s in sessions):
        raise ValueError("session failure")
    medians = round_medians(rows, PROGRAMS)
    result = dict(status="INDEPENDENT_FORMAL_COMPARISON_AUDIT_PASSED", boot_id=summary["boot_id"],
                  qualification_FPGA_calls=150, explicit_warmups=108, timed_calls=108,
                  total_FPGA_calls=366, software_helper_calls=3, timed_sessions=36,
                  six_rounds_correct=True, round_medians=medians,
                  frozen_sources=execution_sources(), audit_source_sha256=digest(__file__),
                  artifacts={str(x.relative_to(root)): digest(x) for x in root.rglob("*")
                             if x.is_file() and x.name != "final_audit.json"})
    write_new(root / "final_audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    print((audit_formal if args.formal else audit)(args.root)["status"])
