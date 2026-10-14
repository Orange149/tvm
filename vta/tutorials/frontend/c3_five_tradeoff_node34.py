"""Local fixed node34 Input nomination; old latency is exposed development data."""

from pathlib import Path
from collections import Counter

import numpy as np
import tvm

import c3_residency_tradeoff as p
from c3_residency_tradeoff_evidence import check_runtime
from c3_reuse_semantics import extract_requests
from c3_thread_sync_audit import final_tir_audit
from run_vta_c3_mode11_graph_context_diagnostic import local_params
from run_vta_c3_node34_board_qualify import GRAPH, PREPARED, SYMBOL, GROUPED
import vta

SEEDS = (0, 20250901, 20260910)
PROGRAMS = ("N34-G2", "N34-Input")
DIRECTORIES = {"N34-G2": GRAPH["G2"], "N34-Input": GRAPH["I1"]}
DATA = GROUPED / "20261008_c3_node34_board_qualify_run02"


def reference(x, w, add):
    if x.shape != (1, 32, 7, 7, 1, 16) or w.shape != (64, 32, 3, 3, 16, 16) or \
            add.shape != (64, 1, 1, 1, 16) or (x.dtype, w.dtype, add.dtype) != \
            (np.dtype("int8"), np.dtype("int8"), np.dtype("int32")):
        raise ValueError("not full node34 packed fusion contract")
    source = x[0, :, :, :, 0, :].transpose(1, 2, 0, 3).reshape(7, 7, 512).astype("int32")
    weights = w.transpose(2, 3, 1, 5, 0, 4).reshape(3, 3, 512, 1024).astype("int32")
    padded = np.pad(source, ((1, 1), (1, 1), (0, 0)))
    accum = np.zeros((7, 7, 1024), dtype="int32")
    for kh in range(3):
        for kw in range(3):
            accum += np.tensordot(padded[kh:kh + 7, kw:kw + 7], weights[kh, kw], axes=([2], [0]))
    accum += add.reshape(1024)
    result = np.clip(accum >> 6, -127, 127).astype("int8")
    return result.reshape(7, 7, 64, 16).transpose(2, 0, 1, 3)[None, :, :, :, None, :]


def function(directory, name):
    path = directory / (SYMBOL + ".tir.json") if name == "N34-G2" else directory / "node34.tir.json"
    return tvm.ir.load_json(path.read_text()), path


def freeze(root):
    root = Path(root).resolve()
    historical = p.read(DATA / "summary.json")
    if historical["status"] != "BOARD_FUSION_AND_GRAPH_QUALIFIED_NO_TIMING" or not historical["all_correct"]:
        raise ValueError("node34 full fusion and graph evidence absent")
    weight, weight_b, add = (np.load(DATA / f) for f in ("W_A.npy", "W_B.npy", "ADD.npy"))
    data_files = {"weight_A": DATA / "W_A.npy", "weight_B": DATA / "W_B.npy", "add": DATA / "ADD.npy"}
    for s in SEEDS:
        x, y = np.load(DATA / ("X_%d.npy" % s)), np.load(DATA / ("Y34_%d.npy" % s))
        if not np.array_equal(reference(x, weight, add), y):
            raise ValueError("node34 CPU/reference contract mismatch")
        data_files["X_%d" % s], data_files["Y_%d" % s] = DATA / ("X_%d.npy" % s), DATA / ("Y34_%d.npy" % s)
    y_b = np.load(DATA / "Y_B.npy")
    if not np.array_equal(reference(np.load(DATA / "X_0.npy"), weight_b, add), y_b) or \
            np.array_equal(y_b, np.load(DATA / "Y34_0.npy")):
        raise ValueError("weight B full reference is incorrect or indistinguishable")
    data_files["Y_B"] = DATA / "Y_B.npy"
    graph = p.read(DIRECTORIES["N34-G2"] / "graph.json")
    params = local_params(DIRECTORIES["N34-G2"])
    if not np.array_equal(params[graph["nodes"][32]["name"]], weight) or \
            not np.array_equal(params[graph["nodes"][33]["name"]], add):
        raise ValueError("snapshot parameters do not match actual graph package")
    baseline, entries = p.read(DIRECTORIES["N34-G2"] / "build.json"), {}
    for name in PROGRAMS:
        directory = DIRECTORIES[name]
        build = p.read(directory / "build.json")
        binary = directory / "graphlib.so"
        old_name = "G2" if name == "N34-G2" else "I1"
        if p.digest(binary) != build["graphlib_sha256"] or p.digest(binary) != historical["graphlib_sha256"][old_name] \
                or p.digest(directory / "graph.json") != baseline["graph_sha256"]:
            raise ValueError("node34 historical binary/graph identity drift")
        if name == "N34-Input" and not (build["non_target_tir_unchanged"] and build["params_semantics_unchanged"]):
            raise ValueError("target-only graph certificate incomplete")
        f, fpath = function(directory, name)
        semantics = extract_requests(tvm.IRModule({"main": f}), vta.get_env())
        addresses = final_tir_audit(tvm.IRModule({"main": f}), semantics)
        if addresses["status"] != "passed" or addresses["debug_mode_transitions"]:
            raise ValueError("node34 effective addresses/debug completion unresolved")
        count = Counter()
        for kernel in addresses["kernels"]:
            first = kernel[0]["args"]
            if first[0] == 0:
                count["reset" if first[1] else "mac"] += 1
        totals = semantics["totals"]
        entries[name] = {"directory": str(directory), "binary": str(binary),
                         "binary_sha256": p.digest(binary), "tir_path": str(fpath), "tir_sha256": p.digest(fpath),
                         "graph_sha256": baseline["graph_sha256"], "params_sha256": p.digest(directory / "params.bin"),
                         "request_totals": totals, "max_kernel_uops": addresses["max_kernel_uops"],
                         "expected": {"reset": count["reset"], "mac": count["mac"],
                                      "total_gemm_push": count["reset"] + count["mac"],
                                      "wgt_loads": totals["load_buffer_2d_wgt_calls"],
                                      "wgt_payload_bytes": totals["load_buffer_2d_wgt_bytes"]},
                         "historical_latency_exposed": True}
    policy = {
        "status": "NODE34_KNOWN_INPUT_NOMINATION_FROZEN",
        "programs": entries, "selected_program": "N34-Input", "current_program": "N34-G2",
        "basis": "historical full-fusion same-Tile Input evidence and input reload reduction; not latency-unseen algorithm test",
        "no_new_node34_schedule_or_program": True,
        "scope": "validate known Input on actual hotspot and transfer into its unchanged target-only graph",
        "data_files": {k: {"path": str(path), "sha256": p.digest(path)} for k, path in data_files.items()},
        "full_fusion_CPU_and_graph_params_verified": True,
        "hardware_runtime": check_runtime(), "reference_source_sha256": p.digest(__file__),
        "confirmation_orders": [("N34-G2", "N34-Input"), ("N34-Input", "N34-G2"), ("N34-G2", "N34-Input")],
        "seed_order_rotates": SEEDS, "confirmation_calls": 18, "confirmation_warmups": 18,
        "gate": {"paired_median_improvement_pct": 2, "required_round_wins": 3},
        "no_reselection_or_inference_from_old_latency": True,
    }
    p.write_new(root / "node34_nomination.json", policy)
    print("NODE34_KNOWN_INPUT_NOMINATION_FROZEN; CPU/reference/identity verified")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    freeze(parser.parse_args().output_dir)
