#!/usr/bin/env python3
"""Prepare an auditable software baseline and qualify a TPS single-thread anchor.

No board operations or timing. Existing baseline artifacts are read-only.
The single-thread anchor is NOT a completed improved-double-buffer baseline.
"""

import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import traceback

from c3_banerjee_baseline import prepare, verify_protected, write_new, digest, HERE, OLD


def qualify(root, output):
    import numpy as np
    import tvm
    import vta
    from vta.testing import simulator
    from c3_banerjee_local import author_namespace, private_author_schedule, Archive, SYMBOL
    from c3_residency_tradeoff_evidence import check_runtime
    from c3_reuse_semantics import extract_requests
    from c3_thread_sync_audit import final_tir_audit
    from c3_residency_tradeoff_board import array_sha, buffer_handles
    from run_vta_c3_node24_partial_board import sentinel
    from run_vta_c3_node24_partial_local import frozen_fusion, route, _arrays, _reference, SEEDS
    from vta.top.residency_dispatch import ExplicitResidencyDispatch

    verify_protected(root)
    output.mkdir(parents=True, exist_ok=False)
    try:
        audit = json.loads((root / "audit.json").read_text())
        selection = audit["selected_no_thread"]
        tile = selection["tile"]
        values = tuple(tile[k] for k in ("b", "h", "w", "ci", "co"))
        candidate = route("input_stationary", values)
        candidate["identity"].update(public_mode="original", implementation_mode=0)
        candidate["candidate_id"] = "Banerjee-paper-TPS-reconstructed-" + selection["id"]
        identity = check_runtime()
        cfg = dict(vta.get_env().cfg_dict, TARGET="sim")
        with vta.Environment(cfg):
            env = vta.get_env()
            # Capture the same Environment/axis objects as the actual compiler.
            namespace = author_namespace(OLD / "author_sources", safe_uop_order=True)
            archive = Archive(output)
            tvm.relay.backend.te_compiler.get().clear()
            target = tvm.target.Target(env.target, host=env.target_host)
            with private_author_schedule(namespace), ExplicitResidencyDispatch([candidate]) as dispatch:
                with tvm.transform.PassContext(opt_level=3, config=dict(vta.build_config().config),
                        instruments=[archive], disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}):
                    graph, lib, params = tvm.relay.build(frozen_fusion(), target=target)
            if params or not dispatch.summary()["all_routes_scheduled"]:
                raise ValueError("frozen complete Relay contract not exactly routed")
            last = next(r for r in reversed(archive.rows) if r["pass"].endswith("CPUAccessRewrite") and r["when"] == "after")
            module = tvm.ir.load_json((output / (last["file"] + ".json")).read_text())
            funcs = [f for f in module.functions.values() if isinstance(f, tvm.tir.PrimFunc)
                     and f.attrs is not None and "global_symbol" in f.attrs and str(f.attrs["global_symbol"]) == SYMBOL]
            if len(funcs) != 1:
                raise ValueError("frozen fused symbol missing")
            lowered = tvm.IRModule({"main": funcs[0]})
            requests = extract_requests(lowered, env)
            address = final_tir_audit(lowered, requests)
            write_new(output / "requests.json", requests)
            write_new(output / "addresses.json", address)
            write_new(output / "graph.json", json.loads(graph))
            write_new(output / "contract.json", {"selection": selection, "route": candidate,
                "actual_Relay_compiled": True, "abi_order": ["X", "W", "ADD", "Y"],
                "shared_safe_uop_order": True, "threading": "none",
                "runtime_uop_extensions": False, "author_double_buffer_reuse": False,
                "paper_TPS_reconstruction_not_author_search_code": True})
            if address["status"] != "passed":
                raise ValueError("final static address gate did not pass")
            lib.export_library(str(output / "fsim_module.so"))
            handle = simulator.LIBS[0].VTATLSCommandHandle
            handle.restype = ctypes.c_void_p
            debug = simulator.LIBS[0].VTASetDebugMode
            debug.argtypes = (ctypes.c_void_p, ctypes.c_int)
            debug(handle(), 6)
            dev = tvm.device("ext_dev", 0)
            x, wa, add = _arrays(0)
            wb = _arrays(20250901)[1]
            if np.array_equal(_reference(x, wa, add), _reference(x, wb, add)):
                raise ValueError("weight intervention is not observable")
            buffers = tuple(tvm.nd.array(v, dev) for v in (x, wa, add, sentinel(_reference(x, wa, add))))
            handles = buffer_handles(buffers)
            cases = [("correctness", seed, *_arrays(seed)) for seed in SEEDS]
            cases += [("weight_ABA_" + label, 0, x, w, add) for label, w in (("A", wa), ("B", wb), ("A_return", wa))]
            rows = []
            for iid, (kind, seed, x, w, add) in enumerate(cases, 1):
                expected = _reference(x, w, add)
                for obj, value in zip(buffers, (x, w, add, sentinel(expected))):
                    obj.copyfrom(value)
                write_new(output / ("dispatch_%02d.json" % iid), {"invocation_id": iid, "call_kind": kind,
                    "seed": seed, "target_calls": 1, "board_contacted": False, "latency_ms": None})
                ctypes.CDLL(None).fflush(None)
                print("C3_NATIVE_CALL_BEGIN %d" % iid, flush=True)
                lib[SYMBOL](*buffers)
                ctypes.CDLL(None).fflush(None)
                print("C3_NATIVE_CALL_END %d" % iid, flush=True)
                actual = buffers[-1].numpy()
                row = {"invocation_id": iid, "call_kind": kind, "seed": seed, "latency_ms": None,
                    "target_calls": 1, "correct": np.array_equal(actual, expected),
                    "arguments_unchanged": all(np.array_equal(obj.numpy(), val) for obj, val in zip(buffers[:3], (x, w, add))),
                    "same_handles": handles == buffer_handles(buffers), "expected_sha256": array_sha(expected),
                    "output_sha256": array_sha(actual), "board_contacted": False}
                write_new(output / ("call_%02d.json" % iid), row)
                rows.append(row)
                if not all(row[k] for k in ("correct", "arguments_unchanged", "same_handles")):
                    raise ValueError("full fusion FSim or immutable argument gate failed")
        verify_protected(root)
        write_new(output / "result.json", {"status": "TPS_SINGLE_THREAD_ANCHOR_FSIM_PASSED_NOT_FULL_BASELINE",
            "board_contacted": False, "latency_ms": None, "calls": rows, "runtime": identity,
            "request_totals": requests["totals"], "address_status": address["status"],
            "binary_sha256": digest(output / "fsim_module.so"), "worker_sha256": digest(__file__),
            "author_adapter_sha256": digest(HERE / "c3_banerjee_local.py")})
    except Exception as error:
        write_new(output / "failure.json", {"status": "LOCAL_FAILURE_NO_BOARD_WORK", "error": str(error),
            "traceback": traceback.format_exc(), "board_contacted": False, "latency_ms": None})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "qualify", "check", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--author-software", type=Path)
    parser.add_argument("--author-hardware", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.phase == "prepare":
        if not args.author_software or not args.author_hardware:
            parser.error("prepare requires both pinned author repositories")
        prepare(args.author_software, args.author_hardware, args.root)
    elif args.phase in ("qualify", "check"):
        if not args.output_dir:
            parser.error("qualify requires a fresh output directory")
        if args.phase == "qualify":
            qualify(args.root, args.output_dir)
        else:
            log = args.root / (args.output_dir.name + ".log")
            # Child aborts cannot erase evidence; no automatic retry or fallback.
            with log.open("x") as stream:
                process = subprocess.run([sys.executable, __file__, "qualify", "--root", str(args.root),
                    "--output-dir", str(args.output_dir)], stdout=stream, stderr=subprocess.STDOUT)
            write_new(args.root / (args.output_dir.name + "_exit.json"),
                      {"returncode": process.returncode, "log_sha256": digest(log), "board_contacted": False})
            sys.exit(0 if process.returncode == 0 else 1)
    else:
        verify_protected(args.root)
