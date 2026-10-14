#!/usr/bin/env python3
"""Read-only stage transaction audit; cannot retry or dispatch device work."""

import argparse
from collections import Counter
import json
from pathlib import Path

import c3_residency_tradeoff as p
from c3_five_tradeoff_board import PROGRAMS, SEEDS, NEW, ORDERS, ROTATIONS
from c3_five_tradeoff_policy import choose_visible
from c3_five_tradeoff_commands import certify_prefetch_dump


def journal(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def audit(root, phase):
    root = Path(root)
    expected_status = {"qualification": "FOUR_PROGRAMS_FPGA_AND_COMMAND_QUALIFIED",
                       "selection": "SELECTION_COMPLETE_WINNERS_SEALED",
                       "confirmation": "CONFIRMATION_COMPLETE_NO_RESELECTION"}
    expected_counts = {
        "qualification": {"helper_regression": 3, "correctness": 12, "weight_ABA": 12,
                          "first_call_diagnostic": 4, "diagnostic_warmup": 36, "warmed_diagnostic": 36},
        "selection": {"selection": 12, "selection_warmup": 36},
        "confirmation": {"timing": 72, "warmup": 72},
    }
    stage = root / phase
    summary, contract = p.read(stage / "summary.json"), p.read(stage / "contract.json")
    rule = p.read(root / "rule_freeze.json")
    if (stage / "invalid_attempt.json").exists() or summary["status"] != expected_status[phase]:
        raise ValueError("stage did not commit valid completion")
    for path, sha in contract["sources"].items():
        if p.digest(path) != sha:
            raise ValueError("execution source changed after stage")
    if contract["rule_sha256"] != p.digest(root / "rule_freeze.json"):
        raise ValueError("rule changed after observations")
    rows = journal(stage / "calls.jsonl")
    if dict(Counter(r["call_kind"] for r in rows)) != expected_counts[phase] or \
            any(r["status"] != "PASS" or not r["correct"] or not r["arguments_unchanged"] for r in rows):
        raise ValueError("calls incomplete/incorrect")
    if len({r["invocation_id"] for r in rows}) != len(rows) or \
            {r["boot_id"] for r in rows} != {summary["boot_id"]}:
        raise ValueError("call identity mixed or duplicated")
    if any(not r["profile_before_output_reads"] or r["complete_operator_calls"] != 1
           or r["hidden_warmup_calls"] != 0 for r in rows if r["call_kind"] != "helper_regression"):
        raise ValueError("call/profile boundary absent")
    dispatch = journal(stage / "dispatch.jsonl")
    real = [r for r in rows if r["call_kind"] != "helper_regression"]
    if len(dispatch) != len(real) * 2:
        raise ValueError("dispatch begin/end ledger incomplete")
    for row, start, end in zip(real, dispatch[::2], dispatch[1::2]):
        if start["invocation_id"] != row["invocation_id"] != end["invocation_id"]:
            raise ValueError("dispatch invocation identity differs")
        if start["invocation_id"] != end["invocation_id"] or start["event"] != "begin" or end["event"] != "end" \
                or start["program"] != row["program"]:
            raise ValueError("dispatch program/order differs")
    for session in journal(stage / "sessions.jsonl"):
        if session["status"] != "PASS" or "cleanup_error" in session or \
                session["runtime_attestation"] != rule["hardware_runtime"]["runtime_sha256"] or \
                set(session["environment"]["masks"]) != {"3"} or \
                session["environment"]["boot_id"] != summary["boot_id"] or \
                session["end_environment"]["boot_id"] != summary["boot_id"]:
            raise ValueError("session identity/CPU3/completion invalid")
    if phase == "qualification":
        for name in PROGRAMS:
            aba = [r for r in rows if r["program"] == name and r["call_kind"] == "weight_ABA"]
            if [r["aba_label"] for r in aba] != ["A", "B", "A_return"] or \
                    len({r["session_id"] for r in aba}) != 1 or \
                    aba[0]["buffer_handles"] != aba[1]["buffer_handles"] or \
                    aba[0]["buffer_handles"] != aba[2]["buffer_handles"] or \
                    aba[0]["output_sha256"] != aba[2]["output_sha256"] or \
                    aba[0]["output_sha256"] == aba[1]["output_sha256"] or \
                    [r["seed"] for r in aba] != [0, 0, 0]:
                raise ValueError("fixed-input same-handle ABA proof absent")
        certified = journal(stage / "prefetch_native_certificates.jsonl")
        actual = [r for r in rows if r["program"] == NEW and r.get("dump_file")]
        if len(certified) != 10 or len(actual) != 10:
            raise ValueError("new actual FPGA native states not all certified")
        for row, proof in zip(actual, certified):
            raw = stage / row["dump_file"]
            if p.digest(raw) != row["dump_sha256"] or proof["invocation_id"] != row["invocation_id"]:
                raise ValueError("native dump/invocation identity changed")
            if certify_prefetch_dump(raw.read_text())["status"] != proof["certificate"]["status"]:
                raise ValueError("actual native prefetch certificate failed")
    if phase in ("selection", "confirmation"):
        seal = p.read(root / "winners.json")
        selection = journal(root / "selection/calls.jsonl")
        if seal["boot_id"] != summary["boot_id"] or \
                seal["selection_journal_sha256"] != p.digest(root / "selection/calls.jsonl"):
            raise ValueError("winner transaction identity mismatch")
        for method, winner in seal["winners"].items():
            if choose_visible([r for r in selection if r["call_kind"] == "selection"],
                              rule["prefixes"][method]) != winner:
                raise ValueError("method used outside-prefix data or reselected winner")
        if summary["winners_sha256"] != p.digest(root / "winners.json"):
            raise ValueError("winner payload differs from committed stage summary")
        counters = journal(stage / "helper_counts.jsonl")
        required = 12 if phase == "selection" else 72
        if len(counters) != required or any(r["count"] != r["expected"] for r in counters):
            raise ValueError("exact one-call timer evidence absent")
    return {"status": "STAGE_TRANSACTION_AUDIT_PASSED", "phase": phase,
            "boot_id": summary["boot_id"], "call_counts": expected_counts[phase],
            "summary_sha256": p.digest(stage / "summary.json"),
            "call_journal_sha256": p.digest(stage / "calls.jsonl"),
            "winner_seal_if_present_sha256": p.digest(root / "winners.json")
                    if (root / "winners.json").exists() else None,
            "audit_source_sha256": p.digest(__file__), "FPGA_calls": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=("qualification", "selection", "confirmation"), required=True)
    args = parser.parse_args()
    result = audit(args.output_dir, args.phase)
    p.write_new(args.output_dir / (args.phase + "_transaction_audit.json"), result)
    print(result["status"] + ": " + args.phase)
