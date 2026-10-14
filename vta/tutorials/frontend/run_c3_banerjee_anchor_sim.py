#!/usr/bin/env python3
"""Recheck sealed own-program TIR against the same experimental FSim runtime."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from c3_banerjee_baseline import GROUPED, write_new, digest


def worker(root, output, program):
    os.environ["TVM_LIBRARY_PATH"] = str(root / "runtime_sim")
    import numpy as np
    import tvm
    import vta
    from vta.testing import simulator
    from run_vta_c3_node24_partial_local import _arrays, _reference, SEEDS, SYMBOL
    from run_vta_c3_node24_partial_board import sentinel
    from c3_residency_tradeoff_board import buffer_handles, array_sha
    from c3_reuse_semantics import extract_requests
    from c3_thread_sync_audit import final_tir_audit, instruction_dump
    from c3_five_tradeoff_commands import decode_submissions, completion_dag, certify_prefetch_dump
    from c3_banerjee_completion import CloseCallBoundary

    output.mkdir(parents=True, exist_ok=False)
    prior = json.loads((GROUPED / "20261012_c3_banerjee_double_buffer_compare_run01/prepared.json").read_text())["programs"][program]
    directory = Path(prior["directory"])
    if digest(directory / "node24_final.tir.json") != prior["tir_sha256"] or digest(prior["binary"]) != prior["binary_sha256"]:
        raise ValueError("historical anchor identity drift")
    function = tvm.ir.load_json((directory / "node24_final.tir.json").read_text())
    if program == "Input-7x7":
        function = CloseCallBoundary()(tvm.IRModule({"main": function}))["main"]
    runtime = Path(simulator.LIBS[0]._name).resolve()
    if runtime != (root / "runtime_sim/libvta_fsim.so").resolve():
        raise ValueError("wrong simulator runtime")
    cfg = dict(vta.get_env().cfg_dict, TARGET="sim")
    with vta.Environment(cfg):
        # Captured CPUAccessRewrite TIR already contains the complete scheduling.
        lib = tvm.build(tvm.IRModule({SYMBOL: function}),
                        target=tvm.target.Target(vta.get_env().target, host=vta.get_env().target_host))
        lib.export_library(str(output / "fsim_module.so"))
        requests = extract_requests(tvm.IRModule({"main": function}), vta.get_env())
        address = final_tir_audit(tvm.IRModule({"main": function}), requests)
        if address["status"] != "passed":
            raise ValueError("final address qualification failed")
        write_new(output / "address_audit.json", address)
        write_new(output / "node24_final.tir.json", json.loads(tvm.ir.save_json(function)))
        write_new(output / "historical_binding.json", prior)
        handle = simulator.LIBS[0].VTATLSCommandHandle
        handle.restype = ctypes.c_void_p
        debug = simulator.LIBS[0].VTASetDebugMode
        debug.argtypes = (ctypes.c_void_p, ctypes.c_int)
        debug(handle(), 6)
        dev = tvm.device("ext_dev", 0)
        x, wa, add = _arrays(0)
        wb = _arrays(20250901)[1]
        if np.array_equal(_reference(x, wa, add), _reference(x, wb, add)):
            raise ValueError("ABA has no distinguishing power")
        buffers = tuple(tvm.nd.array(v, dev) for v in (x, wa, add, sentinel(_reference(x, wa, add))))
        handles = buffer_handles(buffers)
        cases = [("correctness", s, *_arrays(s)) for s in SEEDS]
        cases += [("ABA_" + label, 0, x, w, add) for label, w in (("A", wa), ("B", wb), ("A_return", wa))]
        rows = []
        for iid, (kind, seed, x, w, add) in enumerate(cases, 1):
            expected = _reference(x, w, add)
            for obj, value in zip(buffers, (x, w, add, sentinel(expected))):
                obj.copyfrom(value)
            ctypes.CDLL(None).fflush(None)
            print("C3_NATIVE_CALL_BEGIN %d" % iid, flush=True)
            lib[SYMBOL](*buffers)
            ctypes.CDLL(None).fflush(None)
            print("C3_NATIVE_CALL_END %d" % iid, flush=True)
            row = {"invocation_id": iid, "call_kind": kind, "seed": seed, "target_calls": 1,
                "correct": np.array_equal(buffers[-1].numpy(), expected), "same_handles": handles == buffer_handles(buffers),
                "arguments_unchanged": all(np.array_equal(obj.numpy(), val) for obj, val in zip(buffers[:3], (x, w, add))),
                "output_sha256": array_sha(buffers[-1].numpy()), "latency_ms": None, "board_contacted": False}
            write_new(output / ("call_%02d.json" % iid), row)
            rows.append(row)
            if not all(row[k] for k in ("correct", "same_handles", "arguments_unchanged")):
                raise ValueError("same-runtime anchor full fusion failed")
    write_new(output / "result.json", {"status": "SEALED_PROGRAM_NEW_RUNTIME_FSIM_PASSED", "program": program,
        "calls": rows, "request_totals": requests["totals"], "historical_binding": prior,
        "runtime_sha256": digest(runtime), "board_contacted": False, "latency_ms": None,
        "boundary_repaired": program == "Input-7x7"})


def run(root, output, program):
    log = root / (output.relative_to(root).as_posix().replace("/", "__") + ".log")
    with log.open("x") as stream:
        process = subprocess.run([sys.executable, __file__, "worker", "--root", str(root),
            "--output-dir", str(output), "--program", program], stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        write_new(root / (output.name + "_failed.json"), {"returncode": process.returncode, "log": str(log), "board_contacted": False})
        raise RuntimeError("local anchor worker failed; see " + str(log))
    from c3_thread_sync_audit import instruction_dump
    from c3_five_tradeoff_commands import decode_submissions, completion_dag, certify_prefetch_dump
    parts = re.findall(r"C3_NATIVE_CALL_BEGIN (\d+)\n(.*?)C3_NATIVE_CALL_END (\d+)\n", log.read_text(), re.S)
    if len(parts) != 6:
        raise ValueError("missing native invocation")
    proof = []
    for _, part, _ in parts:
        p = instruction_dump(part)
        for block in decode_submissions(part):
            completion_dag(block)
        if program == "N24-R12-G2-PREFETCH":
            certify_prefetch_dump(part)
        proof.append(p)
    write_new(output / "native_certificate.json", {"status": "SAME_RUNTIME_NATIVE_GATE_PASSED",
        "proof": proof, "all_tokens_paired": True, "log_sha256": digest(log),
        "inheritance": "sealed data arrangement unchanged; new UOP intervals and queue closure checked",
        "board_contacted": False, "latency_ms": None})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("worker", "check", "cross"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--program", choices=("R13-G", "Input-7x7", "N24-R12-G2-PREFETCH"), required=True)
    args = parser.parse_args()
    if args.phase == "cross":
        import tvm
        import vta
        from build_vta_yolov3_tiny_relay_residency_dispatch_pair import cross_compiler
        from run_vta_c3_node24_partial_local import SYMBOL
        root, output = args.root.resolve(), args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
        source = root / "anchors_final" / args.program
        f = tvm.ir.load_json((source / "node24_final.tir.json").read_text())
        target = tvm.target.Target(vta.get_env().target, host=vta.get_env().target_host)
        lib = tvm.build(tvm.IRModule({SYMBOL: f}), target=target)
        lib.export_library(str(output / "graphlib.so"), fcompile=cross_compiler())
        write_new(output / "node24_final.tir.json", json.loads(tvm.ir.save_json(f)))
        write_new(output / "result.json", {"request_totals": json.loads((source / "result.json").read_text())["request_totals"],
            "status": "BOUNDARY_REPAIRED_INPUT_ARM_MATCHES_FSIM_PENDING_FPGA", "board_contacted": False,
            "binary_sha256": digest(output / "graphlib.so"), "sim_native_certificate_sha256": digest(source / "native_certificate.json")})
    else:
        (worker if args.phase == "worker" else run)(args.root.resolve(), args.output_dir.resolve(), args.program)
