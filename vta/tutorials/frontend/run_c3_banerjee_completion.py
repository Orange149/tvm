#!/usr/bin/env python3
"""Complete fixed node24 software baselines in isolated compiler/runtime workers."""

import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import traceback

from c3_banerjee_baseline import HERE, OLD, GROUPED, digest, write_new

BASE_PROGRAMS = ("TPS-single", "TPS-oc-ordinary", "TPS-oc-reuse", "TPS-h-ordinary", "TPS-h-reuse")
PROGRAMS = BASE_PROGRAMS + tuple("TPS-single-ci%d" % ci for ci in (2, 4, 8)) + tuple(
    "TPS-%s-reuse-ci%d" % (axis, ci) for axis in ("oc", "h") for ci in (2, 4))


def compile_worker(root, output, program, target):
    import os
    if target == "sim":
        # route/reference modules import the simulator, so bind it first.
        os.environ["TVM_LIBRARY_PATH"] = str((root / "runtime_sim").resolve())
    import numpy as np
    import tvm
    from tvm import tir
    import vta
    from c3_banerjee_completion import CloseCallBoundary, ImprovedDoubleBuffer, SplitPaddedInputChannels
    from c3_banerjee_local import author_namespace, private_author_schedule, Archive, SpecializeNode24Contexts
    from run_vta_c3_node24_partial_local import route, frozen_fusion, _arrays, _reference, SEEDS, SYMBOL
    from run_vta_c3_node24_partial_board import sentinel
    from c3_residency_tradeoff_board import array_sha, buffer_handles
    from c3_reuse_semantics import extract_requests
    from c3_thread_sync_audit import final_tir_audit

    output.mkdir(parents=True, exist_ok=False)
    try:
        earlier = GROUPED / "20261013_c3_banerjee_baseline_completion_run01"
        selections = json.loads((earlier / "audit.json").read_text())
        kind = "selected_no_thread" if program.startswith("TPS-single") else "selected_oc_thread" if "-oc-" in program else "selected_h_thread"
        selected = selections[kind]
        selected = json.loads(json.dumps(selected))
        tile = selected["tile"]
        if "-ci" in program:
            from c3_banerjee_baseline import Tile, paper_cost
            ci = int(program.rsplit("-ci", 1)[1])
            tile["ci"] = ci
            hardware = vta.get_env()
            reconstructed = paper_cost(Tile(**tile), {"INP": hardware.INP_BUFF_SIZE,
                "WGT": hardware.WGT_BUFF_SIZE, "ACC": hardware.ACC_BUFF_SIZE})
            if not reconstructed["necessary_conditions_passed"] or reconstructed["paper_score_bytes"] != selected["paper_score_bytes"]:
                raise ValueError("candidate is not a qualified analytical tie")
            selected = reconstructed
            tile = selected["tile"]
        values = tuple(tile[k] for k in ("b", "h", "w", "ci", "co"))
        reuse = "-reuse" in program
        candidate = route("input_stationary", values)
        candidate["candidate_id"] = "Banerjee-completion-" + program + "-" + selected["id"]
        candidate["identity"].update(public_mode="original", implementation_mode=0)
        for entity in candidate["identity"]["complete_config_entity"]["entity"]:
            if entity[0] in ("oc_nthread", "h_nthread"):
                entity[2] = tile[entity[0]]
        cfg = dict(vta.get_env().cfg_dict, TARGET="sim" if target == "sim" else "axu5evb")
        with vta.Environment(cfg):
            env = vta.get_env()
            namespace = author_namespace(OLD / "author_sources", safe_uop_order=True)
            archive = Archive(output)
            config = dict(vta.build_config().config)
            passes = list(config["tir.add_lower_pass"])
            before_dma = next(i for i, (_phase, p) in enumerate(passes) if p.info.name == "tir.InjectCopyIntrin")
            passes.insert(before_dma, (1, SplitPaddedInputChannels()))
            pos = next(i for i, (phase, _) in enumerate(passes) if phase >= 2)
            additions = [(2, CloseCallBoundary())]
            if not program.startswith("TPS-single"):
                additions.append((2, ImprovedDoubleBuffer() if reuse else SpecializeNode24Contexts()))
            passes[pos:pos] = additions
            config["tir.add_lower_pass"] = passes
            tvm.relay.backend.te_compiler.get().clear()
            actual_target = tvm.target.Target(env.target, host=env.target_host)
            from vta.top.residency_dispatch import ExplicitResidencyDispatch
            with private_author_schedule(namespace), ExplicitResidencyDispatch([candidate]) as dispatch:
                with tvm.transform.PassContext(opt_level=3, config=config, instruments=[archive],
                        disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}):
                    graph, lib, params = tvm.relay.build(frozen_fusion(), target=actual_target)
            if params or not dispatch.summary()["all_routes_scheduled"]:
                raise ValueError("complete frozen fusion did not use the exact candidate")
            last = next(r for r in reversed(archive.rows) if r["pass"].endswith("CPUAccessRewrite") and r["when"] == "after")
            module = tvm.ir.load_json((output / (last["file"] + ".json")).read_text())
            funcs = [f for f in module.functions.values() if isinstance(f, tir.PrimFunc)
                     and f.attrs is not None and "global_symbol" in f.attrs and str(f.attrs["global_symbol"]) == SYMBOL]
            if len(funcs) != 1:
                raise ValueError("actual complete fusion symbol not unique")
            func = funcs[0]
            write_new(output / "node24_final.tir.json", json.loads(tvm.ir.save_json(func)))
            requests = extract_requests(tvm.IRModule({"main": func}), env)
            audit = final_tir_audit(tvm.IRModule({"main": func}), requests)
            write_new(output / "request_semantics.json", requests)
            write_new(output / "address_audit.json", audit)
            write_new(output / "graph.json", json.loads(graph))
            write_new(output / "contract.json", {"program": program, "selection": selected, "route": candidate,
                "ABI": ["X", "W", "ADD", "Y"], "actual_Relay_compiled": True,
                "TPS_origin": "paper formula reconstruction with frozen deterministic tie-break",
                "shared_safe_uop_order": True, "boundary_repair": "move terminal loop-exit pop before existing sync",
                "shared_domain": int(func.attrs["c3_shared_domain"]) if reuse else None,
                "shared_bank_elements": int(func.attrs["c3_shared_bank_elements"]) if reuse else None})
            if audit["status"] != "passed":
                raise ValueError("static final-address gate failed")
            if target == "board":
                from build_vta_yolov3_tiny_relay_residency_dispatch_pair import cross_compiler
                lib.export_library(str(output / "graphlib.so"), fcompile=cross_compiler())
                local_index = json.loads((root / "local_index.json").read_text())
                sim_f = tvm.ir.load_json((Path(local_index[program]) / "node24_final.tir.json").read_text())
                if not tvm.ir.structural_equal(func, sim_f, map_free_vars=True):
                    raise ValueError("board and qualified simulation target TIR differ")
                write_new(output / "result.json", {"program": program, "status": "ARM_MATCHES_QUALIFIED_SIM_FPGA_PENDING",
                    "request_totals": requests["totals"], "binary_sha256": digest(output / "graphlib.so"),
                    "board_contacted": False, "latency_ms": None})
                return
            # Select the experimental simulator before its first import.
            from vta.testing import simulator
            runtime_path = Path(simulator.LIBS[0]._name).resolve()
            if runtime_path != (root / "runtime_sim/libvta_fsim.so").resolve():
                raise ValueError("wrong simulator runtime loaded")
            handle = simulator.LIBS[0].VTATLSCommandHandle
            handle.restype = ctypes.c_void_p
            debug = simulator.LIBS[0].VTASetDebugMode
            debug.argtypes = (ctypes.c_void_p, ctypes.c_int)
            debug(handle(), 6)
            lib.export_library(str(output / "fsim_module.so"))
            dev = tvm.device("ext_dev", 0)
            x, wa, add = _arrays(0)
            wb = _arrays(20250901)[1]
            if np.array_equal(_reference(x, wa, add), _reference(x, wb, add)):
                raise ValueError("weight intervention lacks distinguishing outputs")
            buffers = tuple(tvm.nd.array(v, dev) for v in (x, wa, add, sentinel(_reference(x, wa, add))))
            handles = buffer_handles(buffers)
            cases = [("correctness", s, *_arrays(s)) for s in SEEDS]
            cases += [("weight_ABA_" + label, 0, x, w, add) for label, w in (("A", wa), ("B", wb), ("A_return", wa))]
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
                    "correct": np.array_equal(buffers[-1].numpy(), expected), "latency_ms": None,
                    "arguments_unchanged": all(np.array_equal(obj.numpy(), val) for obj, val in zip(buffers[:3], (x, w, add))),
                    "same_handles": handles == buffer_handles(buffers), "output_sha256": array_sha(buffers[-1].numpy()),
                    "expected_sha256": array_sha(expected), "board_contacted": False}
                write_new(output / ("call_%02d.json" % iid), row)
                rows.append(row)
                if not all(row[k] for k in ("correct", "arguments_unchanged", "same_handles")):
                    raise ValueError("full-fusion FSim correctness/argument gate failed")
            write_new(output / "result.json", {"program": program, "status": "FSIM_PASSED_NATIVE_AUDIT_PENDING",
                "calls": rows, "request_totals": requests["totals"], "address_status": audit["status"],
                "runtime_path": str(runtime_path), "runtime_sha256": digest(runtime_path),
                "module_sha256": digest(output / "fsim_module.so"), "worker_sha256": digest(__file__),
                "board_contacted": False, "latency_ms": None})
    except Exception as error:
        write_new(output / "failure.json", {"status": "LOCAL_FAILURE", "error": str(error),
            "traceback": traceback.format_exc(), "board_contacted": False, "latency_ms": None})
        raise


def isolated(root, output, program, target):
    label = output.relative_to(root).as_posix().replace("/", "__")
    with (root / (label + ".log")).open("x") as stream:
        cmd = [sys.executable, __file__, "worker", "--root", str(root), "--output-dir", str(output),
               "--program", program, "--target", target]
        if target == "board":
            import shlex
            cmd = ["bash", "-c", "unset LD_LIBRARY_PATH; source /home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux; exec " + shlex.join(cmd)]
        process = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
    write_new(root / (label + "_exit.json"), {"returncode": process.returncode,
        "board_contacted": False, "target": target})
    if process.returncode:
        raise RuntimeError("compiler/FSim worker failed; retained attempt " + str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("worker", "compile"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--program", choices=PROGRAMS, required=True)
    parser.add_argument("--target", choices=("sim", "board"), default="sim")
    args = parser.parse_args()
    if args.phase == "worker":
        compile_worker(args.root.resolve(), args.output_dir.resolve(), args.program, args.target)
    else:
        isolated(args.root.resolve(), args.output_dir.resolve(), args.program, args.target)
