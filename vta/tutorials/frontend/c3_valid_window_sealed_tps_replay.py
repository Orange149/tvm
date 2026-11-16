#!/usr/bin/env python3
"""Replay the original sealed TPS FSim module with same-session UOP evidence.

No compiler/build call, board access, new TPS candidate or latency sampling.
Six explicit calls: three seed warmups followed by positions 4/5/6.
"""

import argparse
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
HISTORY = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout/20261013_c3_banerjee_baseline_completion_run02"
SEALED = HISTORY / "local_final06/TPS-oc-reuse-ci4"
MODULE_SHA256 = "fcadee95a688ac970893722861a20da8f528eed5e9e2a1ab25ad39f9079b9c31"
RUNTIME_SHA256 = "70229fe2e5fad39d5faa64df7599d7dfd7f4b809a3c6d0a375aabb855871b6e8"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def handle_ids(buffers):
    return [int(ctypes.cast(value.handle,ctypes.c_void_p).value) for value in buffers]


def worker(output):
    runtime_dir = HISTORY / "runtime_sim"
    module_path = SEALED / "fsim_module.so"
    runtime_path = runtime_dir / "libvta_fsim.so"
    if digest(module_path) != MODULE_SHA256 or digest(runtime_path) != RUNTIME_SHA256:
        raise ValueError("sealed module/runtime identity mismatch before loading")
    os.environ["TVM_LIBRARY_PATH"] = str(runtime_dir)
    os.environ["VTA_REPLAY_POLICY"] = "disabled"
    os.environ.pop("VTA_COMMAND_MANIFEST_ID",None)
    sys.path[:0] = [str(REPO / "python"),str(REPO / "vta/python")]
    import numpy as np
    import tvm
    import vta
    with vta.Environment(dict(vta.get_env().cfg_dict,TARGET="sim")):
        from vta.testing import simulator
        from run_vta_c3_node24_partial_local import _arrays,_reference,SEEDS,SYMBOL
        loaded_runtime = Path(simulator.LIBS[0]._name).resolve()
        if loaded_runtime != runtime_path.resolve() or digest(loaded_runtime) != RUNTIME_SHA256:
            raise ValueError("different FSim DSO loaded")
        lib = tvm.runtime.load_module(str(module_path))
        function = lib[SYMBOL]
        handle = simulator.LIBS[0].VTATLSCommandHandle
        handle.restype = ctypes.c_void_p
        debug = simulator.LIBS[0].VTASetDebugMode
        debug.argtypes = (ctypes.c_void_p,ctypes.c_int)
        debug(handle(),6)
        dev = tvm.device("ext_dev",0)
        x,w,add = _arrays(0)
        expected = _reference(x,w,add)
        buffers = tuple(tvm.nd.array(value,dev) for value in (x,w,add,np.bitwise_xor(expected,np.int8(1))))
        handles = handle_ids(buffers)
        session_id = "sealed-tps-fsim-pid-%d" % os.getpid()
        write_new(output / "identity.json",{"session_id":session_id,"process_id":os.getpid(),
            "module_path":str(module_path),"module_sha256":digest(module_path),
            "runtime_path":str(loaded_runtime),"runtime_sha256":digest(loaded_runtime),
            "symbol":SYMBOL,"buffer_handles":handles,"debug_flag":6,
            "seed_order":list(SEEDS),"replay_policy":"disabled","compiler_called":False,
            "board_contacted":False,"latency_ms":None})
        rows = []
        for iid,seed in enumerate((*SEEDS,*SEEDS),1):
            x,w,add = _arrays(seed)
            expected = _reference(x,w,add)
            for obj,value in zip(buffers,(x,w,add,np.bitwise_xor(expected,np.int8(1)))):
                obj.copyfrom(value)
            ctypes.CDLL(None).fflush(None)
            print("C3_SEALED_TPS_CALL_BEGIN %d" % iid,flush=True)
            function(*buffers)
            ctypes.CDLL(None).fflush(None)
            print("C3_SEALED_TPS_CALL_END %d" % iid,flush=True)
            actual = buffers[-1].numpy()
            row = {"session_id":session_id,"invocation_id":iid,"call_position":iid,"seed":int(seed),
                "call_kind":"explicit_warmup" if iid <= 3 else "warmed_diagnostic",
                "correct":np.array_equal(actual,expected),"arguments_unchanged":all(np.array_equal(obj.numpy(),value) for obj,value in zip(buffers[:3],(x,w,add))),
                "same_module":True,"same_handles":handle_ids(buffers)==handles,
                "output_sha256":hashlib.sha256(actual.tobytes()).hexdigest(),
                "expected_sha256":hashlib.sha256(expected.tobytes()).hexdigest(),
                "target_calls":1,"buffer_handles":handle_ids(buffers),"board_contacted":False,"latency_ms":None}
            write_new(output / ("call_%02d.json" % iid),row)
            rows.append(row)
            if not all(row[k] for k in ("correct","arguments_unchanged","same_handles")):
                raise ValueError("sealed FSim output/parameter/handle verification failed")
        if digest(module_path) != MODULE_SHA256 or digest(runtime_path) != RUNTIME_SHA256:
            raise ValueError("sealed identity changed during execution")
        write_new(output / "calls.json",rows)


def run(output):
    output.mkdir(parents=True,exist_ok=False)
    try:
        with (output / "complete_worker.log").open("x") as stream:
            process = subprocess.run([sys.executable,__file__,"worker","--output-dir",str(output)],
                stdout=stream,stderr=subprocess.STDOUT,timeout=300)
        write_new(output / "exit.json",{"returncode":process.returncode,"board_contacted":False})
        if process.returncode:
            raise RuntimeError("sealed FSim worker failed; attempt retained")
        text = (output / "complete_worker.log").read_text()
        parts = re.findall(r"C3_SEALED_TPS_CALL_BEGIN (\d+)\n(.*?)C3_SEALED_TPS_CALL_END (\d+)\n",text,re.S)
        if len(parts) != 6 or [int(a) for a,_,b in parts] != list(range(1,7)) or any(a!=b for a,_,b in parts):
            raise ValueError("incomplete or unexpected sealed target invocations")
        from c3_valid_window_uop_values import certify_uop_values
        trace_path = SEALED / "address_audit.json"
        expected_kernels = json.loads(trace_path.read_text())["kernels"]
        definitions = None
        certificates = []
        for iid,part,_ in parts:
            dump_path = output / ("native_call_%02d.txt" % int(iid))
            dump_path.write_text(part)
            certificate = certify_uop_values(part,expected_kernels,definitions)
            if definitions is None:
                definitions = certificate["cache_definitions"]
            certificate["session_binding"] = "SAME_SESSION_SAME_MODULE"
            certificate["invocation_id"] = int(iid)
            write_new(output / ("uop_certificate_%02d.json" % int(iid)),certificate)
            certificates.append({k:v for k,v in certificate.items() if k not in ("cache_definitions","checks")})
        identity = json.loads((output / "identity.json").read_text())
        write_new(output / "result.json",{"status":"SEALED_TPS_SAME_SESSION_FSIM_AND_UOP_CONTENTS_PASSED",
            "identity":identity,"calls":json.loads((output / "calls.json").read_text()),
            "uop_certificates":certificates,"original_module_unmodified":True,"compiler_called":False,
            "expected_trace_path":str(trace_path),"expected_trace_sha256":digest(trace_path),
            "complete_log_sha256":digest(output / "complete_worker.log"),
            "source_sha256":digest(__file__),"certifier_sha256":digest(HERE / "c3_valid_window_uop_values.py"),
            "board_contacted":False,"latency_ms":None})
        print(json.dumps({"status":"SEALED_TPS_SAME_SESSION_FSIM_AND_UOP_CONTENTS_PASSED",
            "output":str(output),"target_calls":6,"board_contacted":False,"latency_ms":None}))
    except Exception as error:
        write_new(output / "failure.json",{"status":"SEALED_TPS_REPLAY_FAILED_STOPPED","error":str(error),
            "traceback":traceback.format_exc(),"board_contacted":False,"latency_ms":None})
        raise


def audit_existing(replay, output):
    """Strictly re-enroll existing same-session logs without another call."""
    from c3_valid_window_uop_values import certify_uop_values
    output.mkdir(parents=True,exist_ok=False)
    original = json.loads((replay / "result.json").read_text())
    if original["status"] != "SEALED_TPS_SAME_SESSION_FSIM_AND_UOP_CONTENTS_PASSED":
        raise ValueError("existing replay did not complete")
    identity = original["identity"]
    if digest(identity["module_path"]) != MODULE_SHA256 or digest(identity["runtime_path"]) != RUNTIME_SHA256:
        raise ValueError("sealed module/runtime identity changed")
    if digest(replay / "complete_worker.log") != original["complete_log_sha256"]:
        raise ValueError("original complete log changed")
    trace_path = Path(original["expected_trace_path"])
    if digest(trace_path) != original["expected_trace_sha256"]:
        raise ValueError("original final trace changed")
    expected = json.loads(trace_path.read_text())["kernels"]
    definitions,proof = None,[]
    for iid,row in enumerate(original["calls"],1):
        if row["session_id"] != identity["session_id"] or row["buffer_handles"] != identity["buffer_handles"] or not all(row[k] for k in ("correct","same_handles","arguments_unchanged")):
            raise ValueError("same-session module/parameter binding absent")
        path = replay / ("native_call_%02d.txt" % iid)
        if digest(path) != original["uop_certificates"][iid-1]["dump_sha256"]:
            raise ValueError("saved native call changed")
        certificate = certify_uop_values(path.read_text(),expected,definitions)
        if definitions is None:
            definitions = certificate["cache_definitions"]
        if not certificate["full_cache_contract_bound_across_calls"]:
            raise ValueError("strict v2 contract binding required")
        certificate.update(session_binding="SAME_SESSION_SAME_MODULE",session_id=identity["session_id"],invocation_id=iid)
        write_new(output / ("uop_certificate_%02d.json" % iid),certificate)
        proof.append({k:v for k,v in certificate.items() if k not in ("cache_definitions","checks")})
    if len(proof) != 6:
        raise ValueError("six original replay calls required")
    unique,identified = {},[]
    for rows in expected:
        key = json.dumps(rows,sort_keys=True)
        site = unique.setdefault(key,len(unique))
        identified.append({"cache_identity":site,"uops":copy.deepcopy(rows)})
    cold = certify_uop_values((replay / "native_call_01.txt").read_text(),identified)
    changed = copy.deepcopy(identified)
    for kernel in changed:
        for op in kernel["uops"]:
            if op["args"][0] == 1 and op["args"][5] == 3:
                op["args"][7] += 1
    try:
        certify_uop_values((replay / "native_call_04.txt").read_text(),changed,cold["cache_definitions"])
    except ValueError as error:
        rejection = str(error)
    else:
        raise ValueError("warm immediate mutation unexpectedly accepted")
    if rejection != "warm full kernel contract changes args/mode/loop/immediate":
        raise ValueError("warm immediate mutation rejected for an unrelated reason")
    legacy = copy.deepcopy(cold["cache_definitions"])
    legacy["schema"] = "c3_valid_window_uop_cache_definitions_v1"
    legacy.pop("contracts")
    weak = certify_uop_values((replay / "native_call_04.txt").read_text(),changed,legacy)
    if weak["full_cache_contract_bound_across_calls"] or not weak["legacy_definition_limit"]:
        raise ValueError("legacy binding did not expose its immediate limitation")
    negative = {"status":"STRICT_V2_WARM_IMMEDIATE_CHANGE_REJECTED","strict_rejection":rejection,
        "legacy_compatibility_only_base_values":{"full_cache_contract_bound_across_calls":False,
            "legacy_definition_limit":weak["legacy_definition_limit"]}}
    write_new(output / "warm_immediate_negative.json",negative)
    write_new(output / "result.json",{"status":"SEALED_TPS_EXISTING_LOGS_STRICT_V2_SAME_SESSION_UOP_CONTRACTS_PASSED",
        "additional_target_calls":0,"original_session_id":identity["session_id"],
        "original_replay_result_sha256":digest(replay / "result.json"),"proof":proof,"negative":negative,
        "certifier_sha256":digest(HERE / "c3_valid_window_uop_values.py"),"source_sha256":digest(__file__),
        "board_contacted":False,"latency_ms":None})
    print(json.dumps({"status":"SEALED_TPS_EXISTING_LOGS_STRICT_V2_SAME_SESSION_UOP_CONTRACTS_PASSED",
        "additional_target_calls":0,"output":str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("worker","run","audit-existing"))
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--replay-dir",type=Path)
    args = parser.parse_args()
    if args.phase == "worker":
        worker(args.output_dir.resolve())
    elif args.phase == "run":
        run(args.output_dir.resolve())
    else:
        if args.replay_dir is None:
            parser.error("audit-existing requires --replay-dir")
        audit_existing(args.replay_dir.resolve(),args.output_dir.resolve())
