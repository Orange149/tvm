#!/usr/bin/env python3
"""Compile a fixed ordinary double-buffer control locally; no FPGA or timer."""

import argparse
import ctypes
import json
from pathlib import Path
import time
import traceback

import tvm
import numpy as np
import vta
from vta.testing import simulator

from c3_banerjee_local import (author_namespace, complete_compute, Archive, SYMBOL,
                               BanerjeeNode24Reuse, SpecializeNode24Contexts, private_author_schedule)
from c3_residency_tradeoff import write_new, digest
from c3_residency_tradeoff_evidence import check_runtime
from c3_reuse_semantics import extract_requests
from c3_thread_sync_audit import final_tir_audit
from run_vta_c3_node24_partial_local import _arrays, _reference, SEEDS, frozen_fusion, route, TILE
from vta.top.residency_dispatch import ExplicitResidencyDispatch
from c3_residency_tradeoff_board import array_sha, buffer_handles
from run_vta_c3_node24_partial_board import sentinel


def run(root, output, reuse=False, fsim=False, dump_commands=False, safe_uop_order=False, relay_contract=False):
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        identity = check_runtime()
        cfg = dict(vta.get_env().cfg_dict, TARGET="sim")
        with vta.Environment(cfg):
            env = vta.get_env()
            namespace = author_namespace(root / "author_sources", safe_uop_order)
            schedule, args, contract = complete_compute(namespace)
            archive = Archive(output)
            with vta.build_config() as context:
                config = dict(context.config)
            if reuse or safe_uop_order:
                # Phase 2 follows the stock phase-1 virtual-thread injection;
                # original dependence tokens stay tied to WGT's two banks.
                passes = list(config["tir.add_lower_pass"])
                pos = next(i for i, (phase, _) in enumerate(passes) if phase >= 2)
                additions = []
                if safe_uop_order:
                    additions.append((2, SpecializeNode24Contexts()))
                if reuse:
                    additions.append((2, BanerjeeNode24Reuse()))
                passes[pos:pos] = additions
                config["tir.add_lower_pass"] = passes
            lib = None
            if relay_contract:
                candidate = route("input_stationary", TILE)
                candidate["candidate_id"] = "Banerjee-functional-adaptation" if reuse else "ordinary-double-buffer-control"
                candidate["identity"].update(public_mode="original", implementation_mode=0,
                                             complete_config_entity=contract["config_entity"])
                target = tvm.target.Target(env.target, host=env.target_host)
                tvm.relay.backend.te_compiler.get().clear()
                with private_author_schedule(namespace), ExplicitResidencyDispatch([candidate]) as dispatch:
                    with tvm.transform.PassContext(opt_level=3, config=config, instruments=[archive],
                            disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}):
                        graph, lib, parameters = tvm.relay.build(frozen_fusion(), target=target)
                if parameters or not dispatch.summary()["all_routes_scheduled"]:
                    raise ValueError("actual fused Relay was not exactly routed")
                final_snapshot = next(r for r in reversed(archive.rows)
                                      if r["pass"].endswith("CPUAccessRewrite") and r["when"] == "after")
                mod = tvm.ir.load_json((output / (final_snapshot["file"] + ".json")).read_text())
                functions = [f for f in mod.functions.values() if isinstance(f, tvm.tir.PrimFunc)
                             and f.attrs is not None and "global_symbol" in f.attrs
                             and str(f.attrs["global_symbol"]) == SYMBOL]
                if len(functions) != 1:
                    raise ValueError("actual target fused TIR missing")
                lowered = tvm.IRModule({SYMBOL: functions[0]})
                write_new(output / "graph.json", json.loads(graph))
                contract.update(actual_frozen_Relay_compiled=True, actual_dispatch=dispatch.summary())
            else:
                with tvm.transform.PassContext(opt_level=3, config=config, instruments=[archive]):
                    lowered = tvm.lower(schedule, args, name=SYMBOL)
                contract["actual_frozen_Relay_compiled"] = False
            write_new(output / "lowered.json", __import__("json").loads(tvm.ir.save_json(lowered)))
            (output / "lowered.tir").write_text(lowered.script())
            final = tvm.IRModule({"main": lowered[SYMBOL]})
            semantics = extract_requests(final, env)
            address = final_tir_audit(final, semantics)
            write_new(output / "requests.json", semantics)
            write_new(output / "addresses.json", address)
            write_new(output / "contract.json", contract)
            rows = []
            if fsim:
                if lib is None:
                    with tvm.transform.PassContext(opt_level=3, config=config, instruments=[archive]):
                        lib = tvm.build(schedule, args, target=env.target, target_host=env.target_host, name=SYMBOL)
                lib.export_library(str(output / "fsim_module.so"))
                if dump_commands:
                    handle = simulator.LIBS[0].VTATLSCommandHandle
                    handle.restype = ctypes.c_void_p
                    debug = simulator.LIBS[0].VTASetDebugMode
                    debug.argtypes = (ctypes.c_void_p, ctypes.c_int)
                    debug(handle(), 6)  # Existing native instruction/UOP diagnostic; never force-serial.
                dev = tvm.device("ext_dev", 0)
                x_a, w_a, add_a = _arrays(0)
                w_b = _arrays(20250901)[1]
                ref_a, ref_b = _reference(x_a, w_a, add_a), _reference(x_a, w_b, add_a)
                if np.array_equal(ref_a, ref_b):
                    raise ValueError("fixed-input weight ABA has no distinguishing power")
                buffers = tuple(tvm.nd.array(v, dev) for v in (x_a, w_a, add_a, sentinel(ref_a)))
                handles = buffer_handles(buffers)
                cases = [("correctness", seed, *_arrays(seed)) for seed in SEEDS]
                cases += [("weight_ABA_" + label, 0, x_a, w, add_a)
                          for label, w in (("A", w_a), ("B", w_b), ("A_return", w_a))]
                for kind, seed, x, w, add in cases:
                    expected = _reference(x, w, add)
                    for obj, value in zip(buffers, (x, w, add, sentinel(expected))):
                        obj.copyfrom(value)
                    iid = len(rows) + 1
                    write_new(output / ("dispatch_%02d.json" % iid), {"invocation": iid, "target_calls": 1,
                        "call_kind": kind, "seed": seed, "latency_ms": None, "board_contacted": False})
                    if dump_commands:
                        ctypes.CDLL(None).fflush(None)
                        print("C3_NATIVE_CALL_BEGIN %d" % iid, flush=True)
                    lib[SYMBOL](*buffers)
                    if dump_commands:
                        ctypes.CDLL(None).fflush(None)
                        print("C3_NATIVE_CALL_END %d" % iid, flush=True)
                    actual = buffers[-1].numpy()
                    unchanged = all(np.array_equal(obj.numpy(), value) for obj, value in zip(buffers[:3], (x, w, add)))
                    row = {"call_kind": kind, "seed": seed, "latency_ms": None, "target_calls": 1,
                           "correct": np.array_equal(actual, expected), "arguments_unchanged": unchanged,
                           "buffer_handles": buffer_handles(buffers), "output_sha256": array_sha(actual),
                           "expected_sha256": array_sha(expected), "weight_sha256": array_sha(w),
                           "same_handles": handles == buffer_handles(buffers), "board_contacted": False}
                    rows.append(row)
                    write_new(output / ("call_%02d.json" % len(rows)), row)
                    if not row["correct"] or not unchanged or not row["same_handles"]:
                        raise ValueError("complete fused FSim or same-handle ABA failed")
        write_new(output / "result.json", {"status": "LOCAL_DB_FSIM_PASSED_NOT_FPGA_QUALIFIED" if fsim else
                    "LOCAL_DB_LOWERED_NOT_FSIM_OR_FPGA_QUALIFIED", "reuse": reuse,
                   "shared_safe_uop_order": safe_uop_order,
                   "shared_context_specialisation": safe_uop_order,
                   "actual_frozen_Relay_compiled": relay_contract,
                   "board_contacted": False, "latency_ms": None, "seconds": time.monotonic() - start,
                   "runtime": identity, "snapshots": archive.rows, "requests": semantics["totals"],
                   "address_status": address["status"], "source_sha256": digest(__file__),
                   "adapter_source_sha256": digest(Path(__file__).with_name("c3_banerjee_local.py")),
                   "fsim_calls": rows, "native_dump_enabled": dump_commands,
                   "functional_adaptation_not_author_stack_reproduction": True})
        print("LOCAL_DB_FSIM_PASSED_NOT_FPGA_QUALIFIED" if fsim else "LOCAL_DB_LOWERED_NOT_FSIM_OR_FPGA_QUALIFIED", flush=True)
    except BaseException as error:
        write_new(output / "failure.json", {"status": "LOCAL_FAILURE_NO_BOARD_WORK", "error": str(error),
                   "traceback": traceback.format_exc(), "seconds": time.monotonic() - start,
                   "board_contacted": False, "latency_ms": None})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--fsim", action="store_true")
    parser.add_argument("--dump-commands", action="store_true")
    parser.add_argument("--safe-uop-order", action="store_true",
                        help="common exact-geometry adaptation for the frozen runtime's ACC hazard guard")
    parser.add_argument("--relay-contract", action="store_true", help="compile the actual archived four-parameter Relay fusion")
    args = parser.parse_args()
    if args.dump_commands and not args.fsim:
        parser.error("native commands require --fsim")
    run(args.root.resolve(), args.output_dir.resolve(), args.reuse, args.fsim, args.dump_commands,
        args.safe_uop_order, args.relay_contract)
