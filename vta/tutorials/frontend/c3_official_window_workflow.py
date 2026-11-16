#!/usr/bin/env python3
"""Serial local gates for the approved official-window experiment.

No board execution is permitted by this local command.  A later board stage
requires a frozen pool with different representations and every local gate.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUN = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout/20261014_c3_official_valid_window_run01"
PYTHON = Path("/home/orange/miniconda3/envs/vta-resnet/bin/python")
BASELINES = ("resnet18_224_final01", "yolov3_tiny_416_final01")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def execution_identity(events, abi):
    copied = []
    for event in events:
        copied.append({k: v for k, v in event.items()
                       if k not in ("source_scope_sha256", "expanded_uop_steps")})
    return hashlib.sha256(json.dumps([abi, copied], sort_keys=True).encode()).hexdigest()


def native_certificate_for_storage(native):
    """Keep arbitrary-width exact dependency sets without decimal-size limits."""
    return {**native, "ancestor_encoding": "hexadecimal_integer_bitset",
            "ancestors": [hex(value) for value in native["ancestors"]]}


def llvm_alu_contract(function, llvm):
    from c3_official_window_llvm import llvm_alu_contract as check
    return check(function, llvm)


def llvm_cache_contract(llvm):
    sites = []
    for line in llvm.splitlines():
        if "call" not in line or not re.search(r"@VTAPush(?:GEMM|ALU)Op\(", line):
            continue
        match = re.search(r"@VTAPush(?:GEMM|ALU)Op\(([^)]*)\)", line)
        fields = match[1].split(", ") if match else []
        count = re.fullmatch(r"i32 (\d+)", fields[-1]) if len(fields) == 4 else None
        if count is None or int(count[1]) not in (0, 4):
            raise ValueError("COMPILED_KERNEL_CACHE_SIGNATURE_NOT_EMPTY_OR_SINGLE_INT32")
        sites.append({"handle": fields[0], "initializer": fields[1], "signature_bytes": int(count[1])})
    return {"status": "ALL_GEMM_AND_ALU_COMPILED_CACHE_SIGNATURES_SUPPORTED", "sites": sites}


def worker(descriptor, output, runtime, helper, mode):
    if (output.parents[1] / "STOP_REQUESTED").exists():
        raise RuntimeError("LOCAL_ATTEMPT_STOPPED_AFTER_SYSTEMIC_RUNNER_ERROR")
    os.environ["TVM_LIBRARY_PATH"] = str(runtime)
    os.environ["C3_OFFICIAL_WINDOW_COLLECTION"] = mode
    os.environ["VTA_REPLAY_POLICY"] = "disabled"
    os.environ.pop("VTA_COMMAND_MANIFEST_ID", None)
    import numpy as np
    import tvm
    import vta
    from vta.testing import simulator
    from c3_official_window_method import aggregate_function
    from c3_valid_window import certify_trace
    from c3_official_window_reference import (array_sha, from_workload,
        postprocess_from_output_terms, semantic_gates)
    from c3_official_window_runtime_native import certify_native, HEADER

    output.mkdir(parents=True, exist_ok=False)
    row = read(descriptor)
    if not row["official_baseline_bound"] or not row["roundtrip_passed"]:
        raise ValueError("BASELINE_NOT_QUALIFIED")
    if digest(row["parent_module"]) != row["parent_module_sha256"] or digest(row["final_module"]) != row["final_module_sha256"]:
        raise ValueError("BASELINE_MODULE_IDENTITY_CHANGED")
    loaded_runtime = Path(simulator.LIBS[0]._name).resolve()
    build = read(runtime / "build.json")
    if loaded_runtime != (runtime / "libvta_fsim.so").resolve() or digest(loaded_runtime) != build["binary_sha256"]:
        raise ValueError("ACTUAL_FSIM_RUNTIME_IDENTITY_CHANGED")
    if build["source_bindings"].get(str(HEADER.resolve())) != digest(HEADER):
        raise ValueError("NATIVE_QUEUE_MODEL_NOT_BOUND_TO_LOADED_RUNTIME_BUILD")
    parent = tvm.ir.load_json(Path(row["parent_module"]).read_text())
    function = parent[row["global_var"]]
    final_module = tvm.ir.load_json(Path(row["final_module"]).read_text())
    final = final_module[row["global_var"]]
    transformed, aggregation = aggregate_function(function, hw_contract={
        "status": "UNRESOLVED_BITSTREAM_BUILD_BINDING"})
    events = aggregation["B"]["events"]
    # The complete products below are consumed in an exact comparison before
    # their hashes are used as compact audit identity.
    full = certify_trace(events)
    tail = postprocess_from_output_terms(full["output_terms"], row["abi"])
    spec = from_workload(row["workload"], row["abi"], tail)
    independent = semantic_gates(spec, full["output_terms"])
    write(output / "reference_contract.json", spec.contract())
    write(output / "independent_semantics.json", independent)
    write(output / "window_certificate.json", aggregation)
    representation_equal = tvm.ir.structural_equal(function, transformed, map_free_vars=False)
    if not representation_equal:
        # Current unbound bitstream contract must never publish a new join.
        raise ValueError("CANDIDATE_JOIN_WITHOUT_VERIFIED_HARDWARE_CONTRACT")
    # Registry callbacks do not own the shared library which defines their
    # machine code. Keep this module alive through every target invocation.
    helper_module = tvm.runtime.load_module(str(helper))
    diagnostic = tvm.get_global_func("c3.official_window.fused_diagnostic_once")
    reset = tvm.get_global_func("c3.official_window.reset_counts")
    counts = tvm.get_global_func("c3.official_window.counts")
    helper_metadata = tvm.get_global_func("c3.official_window.metadata")
    ledger_clear = tvm.get_global_func("vta.runtime.official_submit_ledger_clear")
    ledger_status = tvm.get_global_func("vta.runtime.official_submit_ledger")
    collection = tvm.get_global_func("vta.runtime.official_collection_status")
    status = json.loads(collection())
    if status["mode"] != mode:
        raise ValueError("COLLECTION_MODE_NOT_EFFECTIVE")
    with vta.Environment(dict(vta.get_env().cfg_dict, TARGET="sim")):
        target = tvm.target.Target(vta.get_env().target, host=vta.get_env().target_host)
        # Preserve the original compilation context. In particular, a second
        # default CSE pass can introduce an extra free ALU closure variable and
        # turn the original 4-byte signature into an unsupported 8-byte one.
        with tvm.transform.PassContext(opt_level=3,
                disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}):
            lib = tvm.build(final_module, target=target)
        lib.export_library(str(output / "fsim_module.so"))
        llvm = lib.get_source("ll")
        (output / "module.ll").write_text(llvm)
        write(output / "alu_initializer_contract.json", llvm_alu_contract(final, llvm))
        write(output / "compiled_cache_contract.json", llvm_cache_contract(llvm))
        # Use the exported DSO, as the sealed runner and eventual board do.
        # LLVM JIT external-symbol resolution is not the experiment's ABI.
        lib = tvm.runtime.load_module(str(output / "fsim_module.so"))
        cases = spec.qualification_cases()
        dev = tvm.device("ext_dev", 0)
        buffers = tuple(tvm.nd.array(value, dev) for value in cases[0]["arrays"])
        handles = [int(ctypes.cast(buf.handle, ctypes.c_void_p).value) for buf in buffers]
        storage = [int(buf.handle.contents.data) for buf in buffers]
        rows = []
        definitions, last_ancestors, native_semantics = None, None, None
        reset()
        for index, case in enumerate(cases, 1):
            for buf, value in zip(buffers, case["arrays"]):
                buf.copyfrom(value)
            ledger_clear()
            write(output / ("dispatch_%02d_begin.json" % index), {
                "call_index": index, "kind": case["kind"], "seed": case["seed"],
                "target_calls_planned": 1, "status": "BEGIN", "board_contacted": False})
            dump = diagnostic(lib, row["global_symbol"], *buffers)
            ledger = json.loads(ledger_status())
            meta = json.loads(helper_metadata())
            actual = buffers[-1].numpy()
            if not np.array_equal(actual, case["expected"]):
                wrong = np.argwhere(actual != case["expected"])
                np.save(output / ("mismatch_actual_%02d.npy" % index), actual)
                np.save(output / ("mismatch_expected_%02d.npy" % index), case["expected"])
                raise ValueError("FSIM_FULL_FUSION_OUTPUT_MISMATCH: " + str(wrong[:3].tolist()))
            if any(not np.array_equal(buf.numpy(), value) for buf, value in zip(buffers[:-1], case["arrays"][:-1])):
                raise ValueError("READONLY_ARGUMENT_CHANGED")
            if handles != [int(ctypes.cast(buf.handle, ctypes.c_void_p).value) for buf in buffers] or storage != [int(buf.handle.contents.data) for buf in buffers]:
                raise ValueError("ACTUAL_PARAMETER_STORAGE_CHANGED")
            if int(counts()) != index or meta["started_calls"] != index or meta["completed_calls"] != index:
                raise ValueError("HIDDEN_OR_MISSING_TARGET_CALL")
            if ledger["mode"] != mode or ledger["overflow"] or any(not s["completion"] or s["timeout"] for s in ledger["submits"]):
                raise ValueError("SUBMISSION_NOT_COMPLETED")
            if not ledger["submits"] or ledger["recorded_submits"] != ledger["attempted_submits"]:
                raise ValueError("INCOMPLETE_SUBMISSION_LEDGER")
            (output / ("native_call_%02d.txt" % index)).write_text(dump)
            native = certify_native(events, dump, ledger, definitions)
            definitions = native["cache_definitions"]
            if native["ancestors"] != last_ancestors:
                native_semantics = certify_trace(events, ancestors=native["ancestors"])
                if native_semantics["output_terms"] != full["output_terms"]:
                    raise ValueError("ACTUAL_NATIVE_OUTPUT_TERMS_OR_TAIL_CHANGED")
                last_ancestors = native["ancestors"]
            write(output / ("native_certificate_%02d.json" % index), native_certificate_for_storage(native))
            collection_after = json.loads(collection())
            if collection_after["mode"] != mode or collection_after["debug_flag"] != 0:
                raise ValueError("COLLECTION_OR_DIAGNOSTIC_DEBUG_STATE_CHANGED")
            if collection_after["replay"]["mode"] != "disabled" or collection_after["queue"]["replay_policy"] != "disabled":
                raise ValueError("REPLAY_NOT_DISABLED_FOR_OFFICIAL_QUALIFICATION")
            if mode == "minimal" and collection_after["detailed_clock_reads"] != 0:
                raise ValueError("MINIMAL_COLLECTION_RETAINED_DETAILED_CLOCK_READS")
            call = {"invocation_id": row["object_id"] + ":" + mode + ":%02d" % index,
                "call_index": index, "call_kind": case["kind"], "seed": case["seed"],
                "target_calls": 1, "helper_count": index, "runtime_metadata": ledger,
                "helper_metadata": meta, "buffer_handles": handles, "actual_storage": storage,
                "collection_status_after_call": collection_after,
                "correct": True, "arguments_unchanged": True,
                "output_sha256": array_sha(actual), "expected_sha256": array_sha(case["expected"]),
                "input_sha256": [array_sha(v) for v in case["arrays"][:-1]],
                "native_dump_sha256": hashlib.sha256(dump.encode()).hexdigest(),
                "native_certificate_status": native["status"],
                "native_output_terms_exactly_match_independent_workload": True,
                "board_contacted": False, "performance_sample": False,
                "scope": "FSim complete fusion and collection equivalence; not FPGA latency"}
            if case["kind"] == "directed":
                np.save(output / "directed_conv_int32.npy", case["raw_conv_int32"])
                call["output_distribution"] = {"zero_outputs": int(np.count_nonzero(actual == 0)),
                    "nonzero_outputs": int(np.count_nonzero(actual)),
                    "clip_boundary_outputs": case["clip_boundary_outputs"],
                    "nonzero_unsaturated_outputs": int(np.count_nonzero((actual != 0) & (actual != 127) & (actual != -127))),
                    "total_outputs": int(actual.size), "directed_scale": case["directed_scale"]}
            rows.append(call)
            write(output / ("dispatch_%02d_end.json" % index), call)
            print(json.dumps({"object": row["object_id"], "mode": mode, "local_call": index, "status": "PASS"}), flush=True)
        write(output / "calls.json", rows)
        write(output / "native_semantics.json", {k: v for k, v in native_semantics.items()
              if k not in ("output_terms", "consumer_versions")})
    if helper_module is None:
        raise ValueError("HELPER_MODULE_LIFETIME_NOT_RETAINED")
    identity = execution_identity(events, row["abi"])
    write(output / "summary.json", {"status": "ALL_LOCAL_OFFICIAL_WINDOW_GATES_PASSED",
        "object_id": row["object_id"], "collection": mode, "local_target_calls": len(rows),
        "runtime_path": str(loaded_runtime), "runtime_sha256": digest(loaded_runtime),
        "helper_path": str(helper), "helper_sha256": digest(helper), "abi": row["abi"],
        "B_A_representation_equal": representation_equal, "execution_sha256": identity,
        "aggregation_status": aggregation["status"],
        "hardware_contract_valid_for_join": aggregation["hardware_contract_valid_for_join"],
        "baseline_native_passed": True,
        "queue_source_bound_to_loaded_runtime_build": True,
        "statistics": aggregation["B"]["statistics"], "counts": aggregation["B"]["counts"],
        "semantic": aggregation["B"]["semantic"], "independent_semantics": independent,
        "native_submissions": native["submissions"], "native_uop_values": native["uop_values"],
        "board_contacted": False, "latency_ms": None})


def _tree_rss(pid):
    # Linux /proc supplies process-tree RSS without an optional psutil package.
    pending, visited, total = [pid], set(), 0
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        try:
            status = Path("/proc/%d/status" % current).read_text()
            match = re.search(r"^VmRSS:\s+(\d+) kB", status, re.M)
            total += int(match[1]) * 1024 if match else 0
            children = Path("/proc/%d/task/%d/children" % (current, current)).read_text()
            pending.extend(map(int, children.split()))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return total


def run_local(output, only_object=None):
    output.mkdir(parents=True, exist_ok=False)
    grouped = {}
    for name in BASELINES:
        baseline = read(RUN / "baseline" / name / "baseline_identity.json")
        if baseline["status"] != "OFFICIAL_QUERY_AND_ROUNDTRIP_PASSED":
            raise ValueError("BASELINE_GATE_NOT_PASSED")
        for row in baseline["callsites"]:
            identity = (row["parent_module"], row["global_var"])
            if identity not in grouped:
                grouped[identity] = {**row, "graph_callsite_aliases": [row["object_id"]]}
            else:
                grouped[identity]["graph_callsite_aliases"].append(row["object_id"])
    if only_object:
        grouped = {k: row for k, row in grouped.items() if row["object_id"] == only_object}
        if len(grouped) != 1:
            raise ValueError("UNKNOWN_OR_NONUNIQUE_LOCAL_REQUALIFICATION_OBJECT")
    write(output / "semantic_instance_inventory.json", list(grouped.values()))
    runtime = RUN / "runtime_local_final"
    helper = RUN / "helper/local_final02/c3_official_window_runtime_helper.so"
    rows = []
    for row in sorted(grouped.values(), key=lambda r: (r["network"], r["earliest_graph_node"])):
        descriptor = output / (row["object_id"] + "_descriptor.json")
        write(descriptor, row)
        for mode in ("detailed", "minimal"):
            local = output / row["object_id"] / mode
            local.parent.mkdir(exist_ok=True)
            log = output / (row["object_id"] + "_" + mode + ".log")
            command = [str(PYTHON), str(Path(__file__).resolve()), "--worker", str(descriptor),
                "--output-dir", str(local), "--runtime-dir", str(runtime), "--helper", str(helper), "--mode", mode]
            environment = dict(os.environ, PYTHONPATH=":".join(str(REPO/p) for p in ("python", "vta/python", "vta/tutorials/frontend")),
                               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
            start, peak, limit = time.monotonic(), 0, None
            with log.open("x") as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
                while child.poll() is None:
                    peak = max(peak, _tree_rss(child.pid))
                    if peak > 8 * 1024**3:
                        limit = "AUDIT_MEMORY_LIMIT"
                    elif time.monotonic()-start > 600:
                        limit = "AUDIT_TIME_LIMIT"
                    if limit:
                        import signal
                        os.killpg(child.pid, signal.SIGKILL)
                        break
                    time.sleep(0.2)
                code = child.wait()
            result = {"object_id": row["object_id"], "collection": mode, "returncode": code,
                "wall_seconds": time.monotonic()-start, "process_tree_peak_rss": peak,
                "status": limit or ("PASS" if code == 0 else "LOCAL_GATE_FAILED"),
                "log": str(log), "output": str(local), "board_contacted": False}
            write(output / (row["object_id"] + "_" + mode + "_worker.json"), result)
            rows.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
            if code < 0:
                write(output / "systemic_runner_stop.json", result)
                write(output / "worker_results.json", rows)
                return
            # Object-local support/resource failure remains in the coverage
            # funnel.  It never produces a candidate or a performance sample.
    write(output / "worker_results.json", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--worker")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--helper")
    parser.add_argument("--mode", choices=("detailed", "minimal"))
    parser.add_argument("--only-object", help="Fresh full local qualification after a diagnosed tool fix; no board sample reuse")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    if args.worker:
        worker(Path(args.worker), output, Path(args.runtime_dir), Path(args.helper), args.mode)
    else:
        run_local(output, args.only_object)


if __name__ == "__main__":
    main()
