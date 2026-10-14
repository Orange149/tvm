#!/usr/bin/env python3
"""Finalize a local-only anchor certificate, never a full Banerjee reproduction claim."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import traceback

from c3_banerjee_baseline import digest, verify_protected, write_new
from c3_thread_sync_audit import instruction_dump
from c3_five_tradeoff_commands import decode_submissions, completion_dag


def certificate(text):
    capacity = instruction_dump(text)
    blocks = decode_submissions(text)
    if len(blocks) != 1:
        raise ValueError("single-thread TPS anchor submission contract changed")
    records = blocks[0]
    ancestors = completion_dag(records)
    before = lambda a, b: bool(ancestors[b] & (1 << a))
    inp = [r for r in records if r["description"] == "LOAD INP"]
    wgt = [r for r in records if r["description"] == "LOAD WGT"]
    mac = [r for r in records if r["description"] == "GEMM" and "reset_out: 0" in r["detail"]]
    reset = [r for r in records if r["description"] == "GEMM" and "reset_out: 1" in r["detail"]]
    stores = [r for r in records if r["description"].startswith("STORE")]
    if (len(inp), len(wgt), len(mac), len(reset), len(stores)) != (16, 16, 16, 2, 2):
        raise ValueError("not the sealed 14x14 ci1 co8 single-thread geometry")
    for i, compute in enumerate(mac):
        m = compute["index"]
        if not before(reset[i // 8]["index"], m) or not before(m, stores[i // 8]["index"]):
            raise ValueError("ACC initialization/completion not ordered")
        for loads in (inp, wgt):
            if not before(loads[i]["index"], m):
                raise ValueError("input/weight LOAD may not finish before MAC")
            if i and not before(mac[i - 1]["index"], loads[i]["index"]):
                raise ValueError("single-slot overwrite may race previous MAC")
            address = re.search(r"SRAM:0x([0-9a-fA-F]+)", loads[i]["detail"])
            if not address or int(address[1], 16) != 0:
                raise ValueError("single-slot physical address changed")
    if not before(stores[0]["index"], reset[1]["index"]):
        raise ValueError("ACC reuse may race previous STORE")
    return {"status": "FIXED_ANCHOR_NATIVE_LIFETIMES_PASSED", "capacity": capacity,
            "counts": dict(Counter(r["description"] for r in records)),
            "mac": len(mac), "reset": len(reset), "not_hardware_trace": True,
            "requires_FPGA_qualification": True}


def finalize(root, attempt):
    root, attempt = Path(root), Path(attempt)
    verify_protected(root)
    read = lambda name: json.loads((attempt / name).read_text())
    result = read("result.json")
    if result["status"] != "TPS_SINGLE_THREAD_ANCHOR_FSIM_PASSED_NOT_FULL_BASELINE":
        raise ValueError("not a successful local anchor")
    calls = [read("call_%02d.json" % i) for i in range(1, 7)]
    if calls != result["calls"] or any(not all(c[k] for k in ("correct", "same_handles", "arguments_unchanged"))
            or c["target_calls"] != 1 or c["latency_ms"] is not None or c["board_contacted"] for c in calls):
        raise ValueError("incomplete/incorrect local full-fusion calls")
    aba = calls[3:]
    if aba[0]["output_sha256"] != aba[2]["output_sha256"] or aba[0]["output_sha256"] == aba[1]["output_sha256"]:
        raise ValueError("weight ABA did not distinguish contents")
    source = Path(__file__).with_name("run_c3_banerjee_baseline.py")
    if digest(source) != result["worker_sha256"] or digest(attempt / "fsim_module.so") != result["binary_sha256"]:
        raise ValueError("compiled source/binary identity drift")
    log_path = root / (attempt.name + ".log")
    log = log_path.read_text()
    segments = re.findall(r"C3_NATIVE_CALL_BEGIN (\d+)\n(.*?)C3_NATIVE_CALL_END (\d+)\n", log, re.S)
    if len(segments) != 6 or any(int(a) != i + 1 or a != b for i, (a, _, b) in enumerate(segments)):
        raise ValueError("unbound or missing native invocation")
    proof = [certificate(part) for _, part, _ in segments]
    write_new(root / "local_anchor_certificate.json", {
        "status": "LOCAL_ANCHOR_QUALIFIED_FULL_BASELINE_INCOMPLETE", "native": proof,
        "board_contacted": False, "latency_ms": None, "FSim_calls": 6,
        "attempt": str(attempt), "log_sha256": digest(log_path),
        "result_sha256": digest(attempt / "result.json"),
        "historical_and_production_bindings_unchanged": True,
        "requests": result["request_totals"],
        "limitations": ["reconstructed TPS with deterministic tie-break, not author search entry",
                        "single-thread anchor, not improved double-buffer implementation",
                        "author UOP runtime extensions not yet ported", "FPGA qualification pending"]})


def record_incomplete_status(root, attempt):
    """Keep a passed simulation distinct from a failed safety qualification."""
    verify_protected(root)
    result = json.loads((attempt / "result.json").read_text())
    failure = json.loads((root / (attempt.name + "_audit_failure.json")).read_text())
    text = (root / (attempt.name + ".log")).read_text()
    segments = re.findall(r"C3_NATIVE_CALL_BEGIN (\d+)\n(.*?)C3_NATIVE_CALL_END (\d+)\n", text, re.S)
    diagnostics = []
    for iid, part, end in segments:
        if iid != end:
            raise ValueError("native invocation marker mismatch")
        record = instruction_dump(part)
        balances = []
        for block in decode_submissions(part):
            balance = Counter()
            for rec in block:
                stage = rec["stage"]
                pp, pn, sp, sn = rec["bits"]
                balance[(stage - 1, stage)] -= pp
                balance[(stage + 1, stage)] -= pn
                balance[(stage, stage - 1)] += sp
                balance[(stage, stage + 1)] += sn
            balances.append({"%d->%d" % k: v for k, v in balance.items() if v})
        diagnostics.append({"invocation_id": int(iid), "instruction_summary": record,
                            "token_push_minus_pop": balances})
    write_new(root / "completion_state.json", {
        "status": "BASELINE_INCOMPLETE_NATIVE_DEPENDENCY_GATE_FAILED",
        "board_ready": False, "board_contacted": False, "latency_ms": None,
        "FSim_calls": len(result["calls"]), "all_FSim_calls_correct": all(c["correct"] for c in result["calls"]),
        "static_address_status": result["address_status"], "native_failure": failure["error"],
        "diagnostics": diagnostics, "historical_protected_files_unchanged": True,
        "next_action": "audit author/local tail dependency generation and submission semantics; no token deletion or board retry",
        "performance_claim_allowed": False})
    write_new(root / "manifest.json", {
        "files": {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()},
        "implementation": {p.name: digest(p) for p in Path(__file__).parent.glob("*c3_banerjee_baseline.py")},
        "manifest_self_excluded": True, "board_contacted": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--record-incomplete-status", action="store_true")
    args = parser.parse_args()
    if args.record_incomplete_status:
        record_incomplete_status(args.root, args.attempt)
        raise SystemExit(0)
    try:
        finalize(args.root, args.attempt)
    except Exception as error:
        write_new(args.root / (args.attempt.name + "_audit_failure.json"), {
            "status": "LOCAL_ANCHOR_NATIVE_QUALIFICATION_FAILED_NO_BOARD",
            "error": str(error), "traceback": traceback.format_exc(),
            "board_contacted": False, "latency_ms": None,
            "FSim_correctness_does_not_override_dependency_gate": True})
        raise
