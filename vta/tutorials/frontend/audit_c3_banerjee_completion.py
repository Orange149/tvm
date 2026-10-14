#!/usr/bin/env python3
"""Independent fixed-Tile final native lifetime checks for both thread axes."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
import re

from c3_banerjee_baseline import write_new, digest, HERE
from c3_five_tradeoff_commands import decode_submissions, completion_dag
from c3_thread_sync_audit import instruction_dump


def native_certificate(text, contract):
    capacity = instruction_dump(text)
    blocks = decode_submissions(text)
    if len(blocks) != 1:
        raise ValueError("unexpected multiple submits for the fixed TPS candidate")
    records = blocks[0]
    ancestors = completion_dag(records)
    before = lambda a, b: bool(ancestors[b] & (1 << a))
    by_desc = {desc: [r for r in records if r["description"] == desc] for desc in ("LOAD INP", "LOAD WGT")}
    mac = [r for r in records if r["description"] == "GEMM" and "reset_out: 0" in r["detail"]]
    reset = [r for r in records if r["description"] == "GEMM" and "reset_out: 1" in r["detail"]]
    stores = [r for r in records if r["description"].startswith("STORE")]
    tile = contract["selection"]["tile"]
    threads = tile["oc_nthread"] * tile["h_nthread"]
    groups = (14 // tile["h"]) * (14 // tile["w"]) * (16 // tile["co"])
    ci_groups = 8 // tile["ci"]
    if len(mac) != groups * ci_groups or len(reset) != groups or len(stores) != groups:
        raise ValueError("unexpected compute geometry")
    shared = contract["shared_domain"]
    if shared not in (None, 1, 2):
        raise ValueError("unexpected shared domain")
    readers = {1: defaultdict(list), 2: defaultdict(list)}
    for i, compute in enumerate(mac):
        index = compute["index"]
        pair = i // (ci_groups * threads)
        lane = i % threads
        acc_index = pair * threads + lane
        if not before(reset[acc_index]["index"], index) or not before(index, stores[acc_index]["index"]):
            raise ValueError("ACC initialization/reduction/output lifetime not ordered")
        for domain, desc in ((2, "LOAD INP"), (1, "LOAD WGT")):
            loads = by_desc[desc]
            slices = tile["ci"] if domain == 2 else 1
            expected_count = (len(mac) // threads if domain == shared else len(mac)) * slices
            if len(loads) != expected_count:
                raise ValueError("shared/nonshared LOAD count mismatch")
            start = (i // threads if domain == shared else i) * slices
            for load in loads[start:start + slices]:
                if not before(load["index"], index):
                    raise ValueError("operand read before LOAD completes")
                readers[domain][load["index"]].append(index)
    overwrites = Counter()
    for domain, desc in ((2, "LOAD INP"), (1, "LOAD WGT")):
        last = {}
        span = ((tile["h"] + 2) * (tile["w"] + 2) * tile["ci"] if domain == 2 else
                tile["co"] * tile["ci"] * 9)
        for i, load in enumerate(by_desc[desc]):
            slices = tile["ci"] if domain == 2 else 1
            bank = (i // slices) % threads
            base = bank * span + (i % slices) * span // slices
            address = re.search(r"SRAM:0x([0-9a-fA-F]+)", load["detail"])
            if not address or int(address[1], 16) != base:
                raise ValueError("actual operand bank differs from qualified mapping")
            if base in last:
                if not all(before(m, load["index"]) for m in readers[domain][last[base]]):
                    raise ValueError("operand overwritten before every old reader completes")
                overwrites[desc] += 1
            last[base] = load["index"]
    for i in range(threads, groups):
        if not before(stores[i - threads]["index"], reset[i]["index"]):
            raise ValueError("ACC reset races previous output store")
    return {"status": "NATIVE_FIFO_LIFETIMES_PASSED", "capacity": capacity,
        "counts": dict(Counter(r["description"] for r in records)), "reset": len(reset), "mac": len(mac),
        "proved_overwrites": dict(overwrites), "shared_domain": shared,
        "all_tokens_paired": True, "no_queue_cycle": True,
        "UOP_writes_and_reads_ordered_by_Compute_FIFO": True, "not_hardware_trace": True}


def audit(root, program, directory):
    result = json.loads((directory / "result.json").read_text())
    contract = json.loads((directory / "contract.json").read_text())
    if result["status"] != "FSIM_PASSED_NATIVE_AUDIT_PENDING" or result["program"] != program:
        raise ValueError("local correctness gate missing")
    calls = [json.loads((directory / ("call_%02d.json" % i)).read_text()) for i in range(1, 7)]
    if calls != result["calls"] or any(not all(c[k] for k in ("correct", "arguments_unchanged", "same_handles"))
            or c["latency_ms"] is not None or c["target_calls"] != 1 for c in calls):
        raise ValueError("invalid full-fusion calls")
    if calls[3]["output_sha256"] != calls[5]["output_sha256"] or calls[3]["output_sha256"] == calls[4]["output_sha256"]:
        raise ValueError("weight ABA lacks distinguishing references")
    label = directory.relative_to(root).as_posix().replace("/", "__")
    log = root / (label + ".log")
    segments = re.findall(r"C3_NATIVE_CALL_BEGIN (\d+)\n(.*?)C3_NATIVE_CALL_END (\d+)\n", log.read_text(), re.S)
    if len(segments) != 6 or any(a != b or int(a) != i + 1 for i, (a, _, b) in enumerate(segments)):
        raise ValueError("native diagnostics not bound to six calls")
    proof = [native_certificate(part, contract) for _, part, _ in segments]
    write_new(directory / "native_certificate.json", {"status": "LOCAL_FULL_FUSION_AND_NATIVE_LIFETIMES_PASSED",
        "proof": proof, "log_sha256": digest(log), "result_sha256": digest(directory / "result.json"),
        "module_sha256": result["module_sha256"], "runtime_sha256": result["runtime_sha256"],
        "board_qualification_required": True, "board_contacted": False, "latency_ms": None})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root.resolve(), args.program, args.directory.resolve())
