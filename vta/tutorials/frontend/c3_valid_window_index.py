#!/usr/bin/env python3
"""Bind the fixed local certificates to newly compiled ARM and sealed TPS."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def index(root):
    import tvm
    import vta
    from c3_valid_window import digest, trace_function
    from c3_thread_sync_audit import instruction_dump

    raw = read(root / "binaries_final/arm_index_entries.json")
    records = {}
    for name in ("Q1", "Q2", "Q4", "Q8", "TPS"):
        arm = raw["Q_programs"][name] if name != "TPS" else raw["TPS_reference"]
        directory = Path(arm["directory"])
        if name != "TPS":
            local = root / "local_final02" / name
            certificate = read(local / "full_local_certificate.json")
            if certificate["status"] != "ALL_LOCAL_WINDOW_GATES_PASSED":
                raise ValueError("missing local gate: " + name)
            trace = local / "expected_trace.json"
            sim_tir = local / "node24_final.tir.json"
            first = local / "native_call_01.txt"
            warmed = [local / ("native_call_%02d.txt" % i) for i in (4, 5, 6)]
            if sha(sim_tir) != arm["source_sim_tir_sha256"] or certificate["aggregation_source_sha256"] != arm["aggregation_source_sha256"]:
                raise ValueError("ARM input does not bind final local source: " + name)
        else:
            old = root.parent / "20261013_c3_banerjee_baseline_completion_run02"
            local = old / "local_final06/TPS-oc-reuse-ci4"
            sim_tir = local / "node24_final.tir.json"
            trace = root / "sealed_tps_expected_trace.json"
            replay = root / "sealed_tps_replay_attempt01"
            if read(replay / "strict_reaudit_run02/result.json")["status"] != "SEALED_TPS_EXISTING_LOGS_STRICT_V2_SAME_SESSION_UOP_CONTRACTS_PASSED":
                raise ValueError("sealed TPS same-session strict gate missing")
            first = replay / "native_call_01.txt"
            warmed = [replay / ("native_call_%02d.txt" % i) for i in (4, 5, 6)]
        function = tvm.ir.load_json((directory / "node24_final.tir.json").read_text())
        sim_function = tvm.ir.load_json(sim_tir.read_text())
        if not tvm.ir.structural_equal(function, sim_function, map_free_vars=False):
            raise ValueError("ARM and qualified FSim final TIR differ: " + name)
        if digest(trace_function(function)) != digest(read(trace)):
            raise ValueError("independent expected trace differs from actual ARM TIR: " + name)
        if sha(arm["binary"]) != arm["binary_sha256"] or sha(directory / "node24_final.tir.json") != arm["tir_sha256"]:
            raise ValueError("ARM binary/TIR changed: " + name)
        sig = instruction_dump(warmed[0].read_text())
        semantic, payload = Counter(), Counter()
        for part in sig["submissions"]:
            semantic.update(part["semantic"])
            payload.update(part["load_bytes"])
        expectation = {"mac": semantic["MAC_GEMM"], "reset": semantic["RESET_GEMM"],
            "total_gemm_push": semantic["MAC_GEMM"] + semantic["RESET_GEMM"],
            "wgt_loads": semantic["LOAD_WGT"], "wgt_payload_bytes": payload["WGT"]}
        records[name] = {**arm, "source_sim_tir_sha256": sha(sim_tir),
            "source_sim_directory": str(local),
            "structural_ARM_FSim_equal": True,
            "request_totals": read(local / "request_semantics.json")["totals"],
            "expected_trace": {"path": str(trace), "sha256": sha(trace)},
            "expectation": expectation, "local_native_signature": sig,
            "local_dumps": {"first": str(first), "warmed": list(map(str, warmed))}}
    with (root / "arm_index.json").open("x") as stream:
        json.dump(records, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": "FIXED_FIVE_PROGRAM_ARM_INDEX_BOUND", "programs": list(records), "board_contacted": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    index(parser.parse_args().root.resolve())
