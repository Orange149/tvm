#!/usr/bin/env python3
"""Local-only qualification of Q1/Q2/Q4/Q8 from one frozen post-Fold parent."""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def buffer_handles(buffers):
    return [int(ctypes.cast(value.handle, ctypes.c_void_p).value) for value in buffers]


def worker(parent, runtime_dir, output, q):
    os.environ["TVM_LIBRARY_PATH"] = str(runtime_dir)
    os.environ["VTA_REPLAY_POLICY"] = "disabled"
    os.environ.pop("VTA_COMMAND_MANIFEST_ID", None)
    import numpy as np
    import tvm
    from tvm import tir
    import vta
    from vta.testing import simulator
    from run_vta_c3_node24_partial_local import _arrays, _reference, SYMBOL, SEEDS, TILE
    aggregation_source = Path(__file__).with_name("c3_valid_window.py")
    aggregation_bytes = aggregation_source.read_bytes()
    from c3_valid_window import AggregateWindows
    from c3_banerjee_completion import CloseCallBoundary
    from c3_reuse_semantics import extract_requests
    from c3_thread_sync_audit import final_tir_audit

    output.mkdir(parents=True, exist_ok=False)
    try:
        if aggregation_bytes != aggregation_source.read_bytes():
            raise ValueError("aggregation source changed during import")
        (output / "aggregation_source.py").write_bytes(aggregation_bytes)
        runtime = Path(simulator.LIBS[0]._name).resolve()
        runtime_record = json.loads((runtime_dir / "build.json").read_text())
        if runtime != (runtime_dir / "libvta_fsim.so").resolve() or digest(runtime) != runtime_record["binary_sha256"]:
            raise ValueError("frozen simulation runtime identity mismatch")
        required_defines = ("-DC3_BANERJEE_VALUE_UOPS=1", "-DC3_BANERJEE_GREEDY_UOPS=1")
        if not all(flag in runtime_record["command"] for flag in required_defines):
            raise ValueError("shared runtime optimization contract missing")
        original = tvm.ir.load_json(parent.read_text())
        if str(original.attrs["global_symbol"]) != SYMBOL or len(original.params) != 4:
            raise ValueError("parent full fusion ABI changed")
        module = tvm.IRModule({"main": original})
        with vta.Environment(dict(vta.get_env().cfg_dict, TARGET="sim")):
            repaired = CloseCallBoundary()(module)
            decisions = []
            prepared = AggregateWindows(q, decisions=decisions)(repaired)
            write(output / "aggregation_decisions.json", decisions)
            before_cpu = prepared["main"]
            (output / "post_aggregate_pre_cpu.tir.json").write_text(tvm.ir.save_json(before_cpu))
            (output / "post_aggregate_pre_cpu.tir").write_text(before_cpu.script())
            final = vta.transform.CPUAccessRewrite()(prepared)["main"]
            (output / "node24_final.tir.json").write_text(tvm.ir.save_json(final))
            (output / "node24_final.tir").write_text(final.script())
            final_module = tvm.IRModule({"main": final})
            requests = extract_requests(final_module, vta.get_env())
            address = final_tir_audit(final_module, requests)
            write(output / "request_semantics.json", requests)
            write(output / "address_audit.json", address)
            target = tvm.target.Target(vta.get_env().target, host=vta.get_env().target_host)
            lib = tvm.build(tvm.IRModule({SYMBOL: final}), target=target)
            lib.export_library(str(output / "fsim_module.so"))
            llvm = lib.get_source("ll")
            (output / "module.ll").write_text(llvm)
            static_globals = [line.strip() for line in llvm.splitlines()
                              if line.startswith("@") and "static_handle" in line]
            gemm_calls = [line.strip() for line in llvm.splitlines()
                          if "call" in line and "@VTAPushGEMMOp" in line]
            cache_sites = []
            control_locations, current_block = [], None
            for line in llvm.splitlines():
                label = re.match(r"^([A-Za-z0-9_.$]+):", line)
                if label:
                    current_block = label[1]
                if "call" in line and "@VTAPushGEMMOp" in line:
                    control_locations.append({"basic_block": current_block, "call": line.strip()})
            for call in gemm_calls:
                match = re.search(r"@VTAPushGEMMOp\(([^)]*)\)", call)
                fields = match[1].split(", ") if match else []
                size = re.fullmatch(r"i32 (\d+)", fields[-1]) if len(fields) == 4 else None
                if size is None or int(size[1]) not in (0, 4) or not fields[2].startswith("ptr "):
                    raise ValueError("unsupported LLVM UOP cache closure ABI")
                handle_name, initializer_name = fields[0].split()[-1], fields[1].split()[-1]
                if not handle_name.startswith("@__tvm_static_handle") or not initializer_name.startswith("@__tvm_static_init_lambda"):
                    raise ValueError("unknown LLVM static initialization identity")
                cache_sites.append({"cache_handle": handle_name, "init_lambda": initializer_name,
                    "signature_pointer": fields[2].split()[-1], "signature_bytes": int(size[1])})
            tir_gemm_scopes = []
            tir.stmt_functor.post_order_visit(final.body, lambda node: tir_gemm_scopes.append(node)
                if isinstance(node, tir.AttrStmt) and node.attr_key == "coproc_uop_scope"
                and isinstance(node.value, tir.StringImm) and node.value.value == "VTAPushGEMMOp" else None)
            # LLVM may clone a single scope's call into mutually exclusive
            # control-flow blocks. Count distinct initializer identities, while
            # requiring every occurrence of a handle to name the same closure.
            cache_groups = {}
            for site in cache_sites:
                handle_key = site["cache_handle"]
                contract = (site["init_lambda"], site["signature_pointer"], site["signature_bytes"])
                if handle_key in cache_groups and cache_groups[handle_key] != contract:
                    raise ValueError("LLVM cache handle names incompatible initializer closures")
                cache_groups[handle_key] = contract
            if len(cache_groups) != len(tir_gemm_scopes):
                raise ValueError("GEMM scope count differs from distinct LLVM cache handles")
            expected_alu = []
            def collect_alu(node):
                if isinstance(node, tir.Call) and node.op.name == "tir.vta.uop_push" and int(node.args[0]) == 1:
                    if any(not isinstance(arg, tir.IntImm) for arg in node.args):
                        raise ValueError("LLVM ALU field gate requires constant folded ALU UOP")
                    expected_alu.append([int(arg) for arg in node.args])
            tir.stmt_functor.post_order_visit(final.body, collect_alu)
            actual_alu = []
            for line in llvm.splitlines():
                if "call" not in line or "@VTAUopPush(i32 1," not in line:
                    continue
                match = re.search(r"@VTAUopPush\(([^)]*)\)", line)
                fields = match[1].split(", ") if match else []
                numbers = [re.fullmatch(r"i32 (-?\d+)", field) for field in fields]
                if len(numbers) != 8 or any(value is None for value in numbers):
                    raise ValueError("unsupported LLVM ALU field expression")
                actual_alu.append([int(value[1]) for value in numbers])
            if sorted(actual_alu) != sorted(expected_alu):
                raise ValueError("LLVM ALU mode/reset/address/opcode/immediate differs from final TIR")
            write(output / "alu_fields_llvm.json", {"status": "LLVM_ALU_FIELDS_MATCH_FINAL_TIR",
                "expected": expected_alu, "llvm_calls": actual_alu,
                "actual_instruction_dump_immediate_visible": False,
                "scope": "compiled initializer fields; not direct hardware immediate observation"})
            write(output / "cache_identity_llvm.json", {"static_globals": static_globals,
                "gemm_calls": gemm_calls, "cache_sites": cache_sites,
                "control_flow_locations": control_locations,
                "control_flow_source": "module.ll preserves complete branches and predecessor labels",
                "distinct_gemm_cache_handles": len(cache_groups),
                "gemm_call_occurrences": len(cache_sites),
                "distinct_cache_handles_passed": True,
                "llvm_sha256": digest(output / "module.ll")})
            write(output / "contract.json", {"program": "Q%d" % q, "q_max": q,
                "parent": str(parent), "parent_sha256": digest(parent),
                "ABI": ["X", "W", "ADD", "Y"], "tile": TILE, "threads": [1, 1],
                "resident_groups": 13, "stream_slot_count": 1, "group_count": 16,
                "runtime_path": str(runtime), "runtime_sha256": digest(runtime),
                "runtime_build_sha256": digest(runtime_dir / "build.json"),
                "boundary_repair_count": int(repaired["main"].attrs["c3_boundary_pop_before_sync"]),
                "builder_sha256": digest(__file__),
                "aggregation_source_sha256": digest(output / "aggregation_source.py"),
                "board_contacted": False, "latency_ms": None})
            handle = simulator.LIBS[0].VTATLSCommandHandle
            handle.restype = ctypes.c_void_p
            debug = simulator.LIBS[0].VTASetDebugMode
            debug.argtypes = (ctypes.c_void_p, ctypes.c_int)
            debug(handle(), 6)
            dev = tvm.device("ext_dev", 0)
            x, wa, add = _arrays(0)
            wb = _arrays(20250901)[1]
            if np.array_equal(_reference(x, wa, add), _reference(x, wb, add)):
                raise ValueError("same-weight-handle ABA references do not differ")
            initial_expected = _reference(x, wa, add)
            buffers = tuple(tvm.nd.array(value, dev) for value in
                (x, wa, add, np.bitwise_xor(initial_expected, np.int8(1))))
            handles = buffer_handles(buffers)
            cases = [("three_seed", seed, *_arrays(seed)) for seed in SEEDS]
            cases += [("weight_ABA_" + label, 0, x, w, add)
                      for label, w in (("A", wa), ("B", wb), ("A_return", wa))]
            # Directed low-amplitude inputs preserve signed nonzero results and
            # avoid clip hiding differences. Keep the original full fusion ABI.
            rng = np.random.default_rng(20260910)
            directed_x = rng.integers(-4, 5, x.shape, dtype=np.int8)
            directed_w = rng.integers(-3, 4, wa.shape, dtype=np.int8)
            directed_add = rng.integers(-64, 65, add.shape, dtype=np.int32)
            inp = directed_x[0, :, :, :, 0, :].transpose(1, 2, 0, 3).reshape(14, 14, 128).astype(np.int32)
            weights = directed_w.transpose(2, 3, 1, 5, 0, 4).reshape(3, 3, 128, 256).astype(np.int32)
            padded = np.pad(inp, ((1, 1), (1, 1), (0, 0)))
            conv = np.zeros((196, 256), dtype=np.int32)
            for ky in range(3):
                for kx in range(3):
                    conv += padded[ky:ky + 14, kx:kx + 14, :].reshape(196, 128) @ weights[ky, kx]
            np.save(output / "directed_reference_conv_int32.npy", conv.reshape(14, 14, 256))
            cases.append(("directed_low_saturation", 20260910, directed_x, directed_w, directed_add))
            rows = []
            for iid, (kind, seed, x, w, add) in enumerate(cases, 1):
                expected = _reference(x, w, add)
                values = (x, w, add, np.bitwise_xor(expected, np.int8(1)))
                for obj, value in zip(buffers, values):
                    obj.copyfrom(value)
                ctypes.CDLL(None).fflush(None)
                print("C3_WINDOW_CALL_BEGIN %d" % iid, flush=True)
                lib[SYMBOL](*buffers)
                ctypes.CDLL(None).fflush(None)
                print("C3_WINDOW_CALL_END %d" % iid, flush=True)
                actual = buffers[-1].numpy()
                mismatch = np.argwhere(actual != expected)
                row = {"invocation_id": iid, "call_kind": kind, "seed": seed,
                    "target_calls": 1, "correct": len(mismatch) == 0,
                    "same_module": True, "buffer_handles": buffer_handles(buffers),
                    "same_handles": handles == buffer_handles(buffers),
                    "arguments_unchanged": all(np.array_equal(obj.numpy(), val)
                        for obj, val in zip(buffers[:3], (x, w, add))),
                    "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                    "expected_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                    "first_mismatch": mismatch[0].tolist() if len(mismatch) else None,
                    "negative_count": int((expected < 0).sum()),
                    "positive_count": int((expected > 0).sum()),
                    "clip_boundary_count": int((np.abs(expected.astype(np.int16)) == 127).sum()),
                    "latency_ms": None, "board_contacted": False}
                write(output / ("call_%02d.json" % iid), row)
                rows.append(row)
                if not all(row[key] for key in ("correct", "same_handles", "arguments_unchanged")):
                    raise ValueError("full-fusion FSim/handle/argument gate failed")
                if kind == "directed_low_saturation":
                    if not row["negative_count"] or not row["positive_count"] or row["clip_boundary_count"]:
                        raise ValueError("directed input lacks nonsaturated signed output")
                    for label, array in zip(("x", "w", "add", "output"), (x, w, add, expected)):
                        np.save(output / ("directed_%s.npy" % label), array)
            if rows[3]["actual_sha256"] != rows[5]["actual_sha256"] or rows[3]["actual_sha256"] == rows[4]["actual_sha256"]:
                raise ValueError("same module/weight handle ABA intervention failed")
            write(output / "result.json", {"program": "Q%d" % q,
                "status": "LOCAL_FSIM_PASSED_NATIVE_PROOF_PENDING", "calls": rows,
                "request_totals": requests["totals"], "max_kernel_uops": address["max_kernel_uops"],
                "module_sha256": digest(output / "fsim_module.so"),
                "final_tir_sha256": digest(output / "node24_final.tir.json"),
                "board_contacted": False, "latency_ms": None})
    except Exception as error:
        write(output / "failure.json", {"status": "LOCAL_GATE_FAILED", "error": str(error),
            "traceback": traceback.format_exc(), "board_contacted": False})
        raise


def run(parent, runtime_dir, output, q):
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(".log")
    try:
        with log.open("x") as stream:
            process = subprocess.run([sys.executable, __file__, "worker", "--parent", str(parent),
                "--runtime-dir", str(runtime_dir), "--output-dir", str(output), "--q", str(q)],
                stdout=stream, stderr=subprocess.STDOUT, timeout=300)
    except subprocess.TimeoutExpired:
        write(output.with_name(output.name + "_exit.json"), {"status": "LOCAL_WORKER_TIMEOUT",
            "timeout_seconds": 300, "board_contacted": False, "log": str(log)})
        raise
    write(output.with_name(output.name + "_exit.json"), {"returncode": process.returncode,
        "board_contacted": False, "log": str(log)})
    if process.returncode:
        raise RuntimeError("local worker failed; retained attempt " + str(output))
    from c3_thread_sync_audit import instruction_dump
    from c3_five_tradeoff_commands import decode_submissions, completion_dag
    text = log.read_text()
    parts = re.findall(r"C3_WINDOW_CALL_BEGIN (\d+)\n(.*?)C3_WINDOW_CALL_END (\d+)\n", text, re.S)
    if len(parts) != 7 or [int(a) for a, _, b in parts] != list(range(1, 8)) or any(a != b for a, _, b in parts):
        raise ValueError("incomplete or mismatched local native invocations")
    proof = []
    for iid, part, _ in parts:
        parsed = instruction_dump(part)
        for block in decode_submissions(part):
            completion_dag(block)
        proof.append({"invocation_id": int(iid), "instruction_audit": parsed})
        (output / ("native_call_%02d.txt" % int(iid))).write_text(part)
    write(output / "native_certificate.json", {"status": "FIFO_BALANCE_AND_RANGES_PASSED_FULL_VERSION_UOP_PROOF_PENDING",
        "proof": proof, "all_tokens_paired": True, "log_sha256": digest(log),
        "runtime_sim_only": True, "board_contacted": False, "latency_ms": None})
    print(json.dumps({"program": "Q%d" % q, "status": "LOCAL_FSIM_PASSED_FIFO_RANGES_PASSED_FULL_NATIVE_PROOF_PENDING",
        "output": str(output), "calls": 7, "board_contacted": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("worker", "run"))
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--q", type=int, choices=(1, 2, 4, 8), required=True)
    args = parser.parse_args()
    (worker if args.phase == "worker" else run)(args.parent.resolve(), args.runtime_dir.resolve(),
        args.output_dir.resolve(), args.q)
