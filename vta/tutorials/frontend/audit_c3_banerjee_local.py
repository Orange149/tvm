#!/usr/bin/env python3
"""Independent local full-fusion and encoded lifetime audit; never uses FPGA."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import tvm

from c3_residency_tradeoff import read, write_new, digest
from c3_thread_sync_audit import instruction_dump
from c3_five_tradeoff_commands import decode_submissions, completion_dag


def address(record):
    value = re.search(r"SRAM:0x([0-9a-fA-F]+)", record["detail"])
    if value is None:
        raise ValueError("native memory address missing")
    return int(value[1], 16)


def native_certificate(text, reuse):
    capacity = instruction_dump(text)
    submissions = decode_submissions(text)
    if len(submissions) != 1:
        raise ValueError("this fixed local baseline must have exactly one complete submission")
    records = submissions[0]
    ancestors = completion_dag(records)
    before = lambda a, b: bool(ancestors[b] & (1 << a))
    inp = [r for r in records if r["description"] == "LOAD INP"]
    wgt = [r for r in records if r["description"] == "LOAD WGT"]
    reset = [r for r in records if r["description"] == "GEMM" and re.search(r"reset_out: 1", r["detail"])]
    mac = [r for r in records if r["description"] == "GEMM" and re.search(r"reset_out: 0", r["detail"])]
    store = [r for r in records if r["description"].startswith("STORE")]
    if (len(inp), len(wgt), len(reset), len(mac), len(store)) != (448 if reuse else 896, 896, 112, 896, 112):
        raise ValueError("not the fixed complete node24 two-context program")
    readers = {"INP": defaultdict(list), "WGT": defaultdict(list)}
    for i, compute in enumerate(mac):
        index = compute["index"]
        loads = {"INP": inp[i // 2 if reuse else i], "WGT": wgt[i]}
        for domain, load in loads.items():
            if not before(load["index"], index):
                raise ValueError(domain + " may be read before its LOAD completes")
            readers[domain][load["index"]].append(index)
        pair, lane = i // 16, i % 2
        if not before(reset[pair * 2 + lane]["index"], index) or \
                not before(index, store[pair * 2 + lane]["index"]):
            raise ValueError("ACC RESET/MAC/STORE lifetime not ordered")
    overwrites = Counter()
    for domain, loads, span in (("INP", inp, 64), ("WGT", wgt, 9)):
        last = {}
        for pos, load in enumerate(loads):
            expected_bank = (pos % 2) if reuse and domain == "INP" else pos % 2
            if address(load) != expected_bank * span:
                raise ValueError("unexpected physical " + domain + " slot")
            bank = address(load)
            if bank in last:
                old = last[bank]
                if not all(before(consumer, load["index"]) for consumer in readers[domain][old]):
                    raise ValueError(domain + " overwrite can begin before all previous readers complete")
                overwrites[domain] += 1
            last[bank] = load["index"]
    for pair in range(1, 56):
        for lane in range(2):
            if not before(store[(pair - 1) * 2 + lane]["index"], reset[pair * 2 + lane]["index"]):
                raise ValueError("ACC reuse can race previous STORE")
    return {"status": "NATIVE_FIFO_LIFETIMES_PASSED", "capacity": capacity,
            "native_counts": {"INP_LOAD": len(inp), "WGT_LOAD": len(wgt), "RESET": len(reset),
                              "MAC": len(mac), "STORE": len(store)},
            "proved_overwrite_edges": dict(overwrites), "same_input_shared_by_two_MACs": reuse,
            "UOP_writes_ordered_in_compute_FIFO": True,
            "no_wait_cycle_or_unmatched_tokens": True, "not_hardware_trace_or_performance": True}


def audit(root):
    root = Path(root)
    programs = {"ordinary": root / "local/ordinary_relay_fsim02", "reuse": root / "local/reuse_relay_fsim02"}
    results = {name: read(path / "result.json") for name, path in programs.items()}
    contracts = {name: read(path / "contract.json") for name, path in programs.items()}
    addresses = {name: read(path / "addresses.json") for name, path in programs.items()}
    certificates = {}
    for name, path in programs.items():
        result, contract = results[name], contracts[name]
        if result["status"] != "LOCAL_DB_FSIM_PASSED_NOT_FPGA_QUALIFIED" or \
                not result["actual_frozen_Relay_compiled"] or not contract["actual_frozen_Relay_compiled"] or \
                not result["shared_safe_uop_order"] or not result["shared_context_specialisation"]:
            raise ValueError("actual fused compiler result or common adaptation absent")
        for key, source in (("source_sha256", "run_c3_banerjee_local.py"), ("adapter_source_sha256", "c3_banerjee_local.py")):
            if result[key] != digest(Path(__file__).with_name(source)):
                raise ValueError("compiled local source identity changed")
        calls = [read(path / ("call_%02d.json" % i)) for i in range(1, 7)]
        if calls != result["fsim_calls"] or any(not c["correct"] or not c["arguments_unchanged"]
                or not c["same_handles"] or c["latency_ms"] is not None or c["target_calls"] != 1 for c in calls):
            raise ValueError("full fused FSim calls incomplete or incorrect")
        aba = calls[3:]
        if [c["call_kind"] for c in aba] != ["weight_ABA_A", "weight_ABA_B", "weight_ABA_A_return"] or \
                aba[0]["output_sha256"] != aba[2]["output_sha256"] or \
                aba[0]["output_sha256"] == aba[1]["output_sha256"] or \
                any(c["buffer_handles"] != aba[0]["buffer_handles"] for c in aba):
            raise ValueError("same-handle fixed-input ABA reference absent")
        log = (path / "native.log").read_text()
        segments = re.findall(r"C3_NATIVE_CALL_BEGIN (\d+)\n(.*?)C3_NATIVE_CALL_END (\d+)\n", log, re.S)
        if len(segments) != 6 or any(int(a) != i + 1 or a != b for i, (a, _, b) in enumerate(segments)):
            raise ValueError("native dump is not bound to six exact complete invocations")
        certificates[name] = [native_certificate(part, name == "reuse") for _, part, _ in segments]
    for key in ("frozen_relay_sha256", "abi_order", "buffers", "compute", "tile", "oc_nthread", "h_nthread"):
        if contracts["ordinary"][key] != contracts["reuse"][key]:
            raise ValueError("control fusion or Tile differs")
    old, new = addresses["ordinary"], addresses["reuse"]
    if old["status"] != "passed" or new["status"] != "passed" or old["debug_mode_transitions"] or new["debug_mode_transitions"] \
            or len(old["sync_event_positions"]) != len(new["sync_event_positions"]):
        raise ValueError("address, force-serial or sync difference")
    def without_input(kernels):
        kernels = json.loads(json.dumps(kernels))
        for kernel in kernels:
            for op in kernel:
                if op["args"][0:2] == [0, 0]:
                    op["args"][3] = "ALLOWED_INP_BANK_DIFFERENCE"
        return kernels
    if without_input(old["kernels"]) != without_input(new["kernels"]):
        raise ValueError("math, WGT, ACC or non-Input UOP arrangement changed")
    totals = {name: result["requests"] for name, result in results.items()}
    for key in totals["ordinary"]:
        if "_inp_" not in key and key not in ("load_buffer_2d_bytes", "load_buffer_2d_calls") \
                and totals["ordinary"][key] != totals["reuse"][key]:
            raise ValueError("non-Input request changed")
    return {"status": "LOCAL_FULL_FUSION_AND_NATIVE_LIFETIME_AUDIT_PASSED",
            "programs": {n: {"directory": str(d), "binary_sha256": digest(d / "fsim_module.so"),
                        "TIR_sha256": digest(d / "lowered.json"), "request_totals": totals[n],
                        "max_kernel_uops": addresses[n]["max_kernel_uops"]} for n, d in programs.items()},
            "native_certificates": certificates, "calls": {"FSim_complete_fused_calls": 12, "FPGA_calls": 0, "timing_calls": 0},
            "common_Tile_four_buffer_contract_and_nonInput_kernel_match": True,
            "author_source_mechanism_scope": "output-channel double buffering with reduction-index-alternating shared input; no spatial-thread or TPS reproduction",
            "not_exact_author_implementation_or_performance_reproduction": True,
            "board_qualification_still_required": True, "latency_ms": None,
            "audit_source_sha256": digest(__file__)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    write_new(args.root / "local_qualification.json", result)
    print(result["status"])
