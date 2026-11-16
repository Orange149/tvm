#!/usr/bin/env python3
"""Freeze local window certificates; no device access or performance samples."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import tvm
import vta

from c3_valid_window import certify_trace, digest, trace_function
from c3_valid_window_native import native_binding_details
from c3_valid_window_uop_values import certify_uop_values
from c3_thread_sync_audit import instruction_dump


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def fixed_convolution(certificate, x, w, reference):
    """Check all logical packed products, then evaluate raw int32 reduction."""
    output = certificate["output_terms"]
    if len(output) != 3136:
        raise ValueError("fixed complete output coverage differs")
    xflat, wflat = x.reshape(-1, 16).astype(np.int32), w.reshape(-1, 16, 16).astype(np.int32)
    accum = np.zeros((3136, 16), dtype=np.int32)
    expected_phases = lambda group: [[2, 0, ("tensor", (2, group))],
        [3, 1, ("imm", 7)], [0, 1, ("imm", 127)], [1, 1, ("imm", -127)]]
    visited = set()
    for logical, terms, phases in output:
        tensor, index = logical
        if tensor != 3 or not 0 <= index < 3136 or index in visited:
            raise ValueError("fixed logical output identity differs")
        visited.add(index)
        group, position = divmod(index, 196)
        h, col = divmod(position, 14)
        expected = Counter()
        for ci in range(8):
            for ky in range(3):
                for kx in range(3):
                    ih, iw = h + ky - 1, col + kx - 1
                    inp = (0, ci * 196 + ih * 14 + iw) if 0 <= ih < 14 and 0 <= iw < 14 else ("zero",)
                    weight = (1, group * 72 + ci * 9 + ky * 3 + kx)
                    expected[inp, weight] += 1
        actual = Counter({(tuple(inp), tuple(weight)): count for inp, weight, count in terms})
        if actual != expected:
            raise ValueError("wrong/missing/duplicated convolution product at output %d" % index)
        if json.loads(json.dumps(phases)) != json.loads(json.dumps(expected_phases(group))):
            raise ValueError("broadcast ADD/shift/clip semantics differ")
        for inp, weight, count in terms:
            if inp != ["zero"]:
                accum[index] += count * (wflat[weight[1]] @ xflat[inp[1]])
    packed_reference = reference.reshape(14, 14, 16, 16).transpose(2, 0, 1, 3).reshape(3136, 16)
    if not np.array_equal(accum, packed_reference):
        raise ValueError("raw int32 convolution differs before ADD/shift/clip")
    return {"status": "ALL_LOGICAL_PRODUCTS_AND_RAW_INT32_CONVOLUTION_PASSED",
            "packed_outputs": len(visited), "packed_terms_per_output": 72,
            "scalar_products_per_output": 1152, "total_scalar_products": 57802752,
            "raw_accum_sha256": hashlib.sha256(accum.tobytes()).hexdigest(),
            "reference_int32_sha256": hashlib.sha256(packed_reference.tobytes()).hexdigest(),
            "raw_accum_min": int(accum.min()), "raw_accum_max": int(accum.max()),
            "int8_partial_sum_bound": certificate["int8_partial_sum_bound"],
            "ADD_overflow_bound_claimed": False}


def audit_program(directory):
    function = tvm.ir.load_json((directory / "node24_final.tir.json").read_text())
    events = trace_function(function)
    kernels = [event for event in events if event["kind"] == "kernel"]
    contract = json.loads((directory / "contract.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    if result["status"] != "LOCAL_FSIM_PASSED_NATIVE_PROOF_PENDING" or len(result["calls"]) != 7:
        raise ValueError("complete local correctness gate missing")
    if file_sha(directory / "fsim_module.so") != result["module_sha256"] or file_sha(directory / "node24_final.tir.json") != result["final_tir_sha256"]:
        raise ValueError("local binary/TIR changed")
    if contract["aggregation_source_sha256"] != file_sha(directory / "aggregation_source.py"):
        raise ValueError("actual loaded aggregation source differs")
    source = Path(__file__).with_name("c3_valid_window.py")
    if contract["aggregation_source_sha256"] != file_sha(source):
        raise ValueError("candidate was compiled before final aggregation source freeze")
    calls, definitions, proofs, logical, native_seen = [], None, {}, None, {}
    for iid in range(1, 8):
        row = json.loads((directory / ("call_%02d.json" % iid)).read_text())
        if row != result["calls"][iid - 1] or not all(row[key] for key in ("correct", "same_handles", "arguments_unchanged")):
            raise ValueError("FSim call/buffer identity gate differs")
        if row["buffer_handles"] != result["calls"][0]["buffer_handles"]:
            raise ValueError("weight/module handle changed during local qualification")
        text = (directory / ("native_call_%02d.txt" % iid)).read_text()
        native = native_binding_details(events, text)
        key = digest(native["ancestors"])
        if key not in native_seen:
            proof = certify_trace(events, native["ancestors"])
            proof["status"] = "ACTUAL_FIFO_VERSIONS_ACC_PRODUCTS_PASSED"
            native_seen[key] = proof
        proof = native_seen[key]
        if logical is None:
            logical = proof
        elif any(logical[field] != proof[field] for field in (
                "tensor_requests_sha256", "control_trace_sha256", "output_product_postprocess_sha256")):
            raise ValueError("actual invocation semantics changed")
        uop = certify_uop_values(text, kernels, definitions)
        definitions = uop["cache_definitions"]
        del uop["cache_definitions"]
        native.pop("ancestors")
        write(directory / ("actual_native_proof_%02d.json" % iid), {"binding": native, "uop_values": uop})
        calls.append({"invocation_id": iid, "call_kind": row["call_kind"],
            "seed": row["seed"], "status": "PASS", "same_session_same_module": True,
            "version_proof_identity": key, "uop_status": uop["status"],
            "uop_sram_wraps": uop["uop_sram_wraps"],
            "instructions": instruction_dump(text)["submissions"][0]["instructions"],
            "native_dump_sha256": file_sha(directory / ("native_call_%02d.txt" % iid))})
    with gzip.open(directory / "logical_versions_products.json.gz", "wt") as stream:
        json.dump(logical, stream, sort_keys=True)
    conv = fixed_convolution(logical, np.load(directory / "directed_x.npy"),
                             np.load(directory / "directed_w.npy"),
                             np.load(directory / "directed_reference_conv_int32.npy"))
    write(directory / "raw_convolution_certificate.json", conv)
    signature = instruction_dump((directory / "native_call_04.txt").read_text())
    abi = [{"shape": [int(x) for x in function.buffer_map[param].shape],
            "dtype": str(function.buffer_map[param].dtype),
            "strides": [str(x) for x in function.buffer_map[param].strides],
            "element_offset": str(function.buffer_map[param].elem_offset)} for param in function.params]
    write(directory / "expected_trace.json", events)
    execution = [{k: value for k, value in event.items() if k not in ("scope", "signature", "cache_identity", "mode")}
                 for event in events]
    summary = {"program": contract["program"], "status": "ALL_LOCAL_WINDOW_GATES_PASSED",
        "parent_sha256": contract["parent_sha256"], "aggregation_source_sha256": contract["aggregation_source_sha256"],
        "runtime_sha256": contract["runtime_sha256"], "module_sha256": result["module_sha256"],
        "final_tir_sha256": result["final_tir_sha256"], "normalized_execution_sha256": digest({"ABI": abi, "events": execution}),
        "ABI": abi, "expected_trace_sha256": file_sha(directory / "expected_trace.json"),
        "tensor_requests_sha256": logical["tensor_requests_sha256"],
        "necessary_control_sha256": logical["control_trace_sha256"],
        "output_product_postprocess_sha256": logical["output_product_postprocess_sha256"],
        "MAC_GEMM": sum(event["mode_reset"] == [0, 0] for event in kernels),
        "RESET_GEMM": sum(event["mode_reset"] == [0, 1] for event in kernels),
        "max_kernel_uops": max(len(event["uops"]) for event in kernels),
        "native_warm_signature": signature, "calls": calls,
        "raw_convolution": conv, "board_contacted": False, "latency_ms": None}
    write(directory / "full_local_certificate.json", summary)
    print(json.dumps({key: summary[key] for key in ("program", "status", "MAC_GEMM", "RESET_GEMM", "max_kernel_uops")}), flush=True)
    return summary


def freeze(root):
    rows = [json.loads((root / "local_final02" / ("Q%d" % q) / "full_local_certificate.json").read_text()) for q in (1, 2, 4, 8)]
    for field in ("parent_sha256", "aggregation_source_sha256", "runtime_sha256", "tensor_requests_sha256",
                  "necessary_control_sha256", "output_product_postprocess_sha256"):
        if len({row[field] for row in rows}) != 1:
            raise ValueError("controlled Q pool contract differs: " + field)
    canonical, aliases = {}, {}
    for row in rows:
        key = row["normalized_execution_sha256"]
        canonical.setdefault(key, row["program"])
        aliases[row["program"]] = canonical[key]
    order = [["Q1", "Q2", "Q8", "Q4", "TPS"], ["Q2", "Q8", "Q4", "TPS", "Q1"],
             ["Q8", "Q4", "TPS", "Q1", "Q2"], ["Q4", "TPS", "Q1", "Q2", "Q8"],
             ["TPS", "Q1", "Q2", "Q8", "Q4"], ["TPS", "Q4", "Q8", "Q2", "Q1"]]
    actual = []
    for round_order in order:
        seen, retained = set(), []
        for label in round_order:
            value = aliases.get(label, label)
            if value not in seen:
                retained.append(value)
                seen.add(value)
        actual.append(retained)
    write(root / "local_pool.json", {"status": "Q_LOCAL_POOL_FROZEN_BOARD_NOT_QUALIFIED",
        "programs": rows, "aliases": aliases, "round_order": actual,
        "seed_orders": [[0, 20250901, 20260910], [20250901, 20260910, 0], [20260910, 0, 20250901]] * 2,
        "formal_warmups": sum(map(len, actual)) * 3, "formal_timings": sum(map(len, actual)) * 3,
        "primary_comparison": "Q8 versus Q1", "TPS_modified": False,
        "board_contacted": False, "no_performance_result": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("program", "freeze"))
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args()
    (audit_program if args.phase == "program" else freeze)(args.path.resolve())
