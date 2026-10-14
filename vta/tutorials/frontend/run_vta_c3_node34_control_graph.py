#!/usr/bin/env python3
"""Cross-build the fixed-Tile node34 controls without changing frozen G2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import traceback

import tvm
from tvm import autotvm, relay, tir
import vta
from vta.top.residency_dispatch import ExplicitResidencyDispatch

from build_vta_resnet50_generic_residency_tir_audit import FullTIRArchive
from build_vta_yolov3_tiny_relay_residency_dispatch_pair import cross_compiler, make_relay_program
from c3_reuse_semantics import extract_requests
from c3_thread_sync_audit import final_tir_audit
from run_vta_yolov3_tiny_relay_residency_dispatch_pair_board import semantic_params_hash

HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
G2 = GROUPED / "20261008_c3_yolo224_incremental_prepare_run03/G2"
ASSETS = GROUPED / "20260926_c3_yolo224_graph_build_run02/summary.json"
GATES = {
    "O1": GROUPED / "20261008_c3_node34_original_o1_local_gate_run01",
    "I1": GROUPED / "20261008_c3_node34_input_i1_local_gate_run01",
    "W1": GROUPED / "20261008_c3_node34_mode4_local_gate_run02",
}
NODE = 34
SYMBOL = "tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast_5"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build(mode, output, weights):
    gate = GATES[mode]
    fsim = json.loads((gate / "fsim_results.json").read_text())
    if fsim["status"] != "passed" or len(fsim["seeds"]) != 3 or not all(
        row["correct"] for row in fsim["seeds"]
    ):
        raise ValueError("full-fusion three-seed local FSim did not pass")
    frozen = json.loads((G2 / "build.json").read_text())
    assets = json.loads(ASSETS.read_text())["assets"]
    paths = {key: Path(assets[key]["path"]) for key in ("cfg", "darknet_lib")}
    paths["weights"] = weights
    for key, path in paths.items():
        if not path.is_file() or sha(path) != assets[key]["sha256"]:
            raise ValueError("model asset changed: " + key)
    if sha(G2 / "graph.json") != frozen["graph_sha256"]:
        raise ValueError("frozen G2 graph changed")
    route = json.loads((gate / "candidate.json").read_text())["routes"][-1]
    routes = list(frozen["routes"]) + [route]
    if len(routes) != 6:
        raise ValueError("expected five frozen routes and node34")
    env = vta.get_env()
    if env.TARGET != "axu5evb":
        raise ValueError("wrong VTA target")
    args = SimpleNamespace(**{key: str(path) for key, path in paths.items()})
    program, params, shape = make_relay_program(env, args)
    if tuple(shape) != (1, 3, 224, 224):
        raise ValueError("YOLO input shape changed")
    target = tvm.target.Target(env.target, host=env.target_host)
    archive = FullTIRArchive(env)
    relay.backend.te_compiler.get().clear()
    with autotvm.tophub.context(target), ExplicitResidencyDispatch(routes) as dispatch:
        with vta.build_config(opt_level=3,
                              disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
                              instruments=[archive]):
            graph, lib, lowered = relay.build(program, target=target, params=params)
    hits = dispatch.summary()
    if not hits["all_routes_scheduled"] or len(hits["schedule_hits"]) != 6:
        raise ValueError("control route did not schedule exactly once")
    graph_hash = hashlib.sha256(graph.encode()).hexdigest()
    if graph_hash != frozen["graph_sha256"]:
        raise ValueError("Graph JSON changed")
    (output / "graph.json").write_text(graph)
    (output / "params.bin").write_bytes(relay.save_param_dict(lowered))
    if semantic_params_hash(output / "params.bin")[0] != semantic_params_hash(G2 / "params.bin")[0]:
        raise ValueError("parameter semantics changed")
    if json.loads(graph)["nodes"][NODE]["attrs"]["func_name"] != SYMBOL:
        raise ValueError("node34 symbol changed")
    captured = {}
    for snapshot in archive.snapshots:
        module = tvm.ir.load_json(snapshot["json"])
        for function in module.functions.values():
            if isinstance(function, tir.PrimFunc) and function.attrs is not None \
                    and "global_symbol" in function.attrs:
                name = str(function.attrs["global_symbol"])
                if name in captured:
                    raise ValueError("duplicate PrimFunc: " + name)
                captured[name] = function
    old_inventory = json.loads((G2 / "functions.json").read_text())
    if set(captured) != set(old_inventory):
        raise ValueError("fused function inventory changed")
    for name, function in captured.items():
        if name == SYMBOL:
            expected = tvm.ir.load_json((gate / "node34.tir.json").read_text())
        else:
            expected = tvm.ir.load_json((G2 / (name + ".tir.json")).read_text())
        if not tvm.ir.structural_equal(function, expected, map_free_vars=True):
            raise ValueError("fused TIR changed unexpectedly: " + name)
    node = captured[SYMBOL]
    (output / "node34.tir").write_text(node.script())
    (output / "node34.tir.json").write_text(tvm.ir.save_json(node))
    single = tvm.IRModule({"main": node})
    requests = extract_requests(single, env)
    address = final_tir_audit(single, requests)
    if address["status"] != "passed":
        raise ValueError("final node34 address audit failed")
    lib.export_library(str(output / "graphlib.so"), cross_compiler())
    result = {
        "status": "CONTROL_GRAPH_BUILT_BOARD_NOT_QUALIFIED", "mode": mode,
        "frozen_g2_graphlib_sha256": frozen["graphlib_sha256"],
        "graphlib_sha256": sha(output / "graphlib.so"),
        "graph_sha256": graph_hash, "params_semantics_unchanged": True,
        "non_target_tir_unchanged": True, "target_tir_equals_local_gate": True,
        "target": {"node": NODE, "symbol": SYMBOL, "mode": route["identity"]["implementation_mode"],
                   "request_totals": requests["totals"], "address_status": address["status"],
                   "function_syncs": len(address["sync_event_positions"]),
                   "max_kernel_uops": address["max_kernel_uops"]},
        "board_contacted": False,
    }
    write(output / "build.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(GATES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path,
                        default=Path(".tvm_test_data/darknet/yolov3-tiny.weights"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = build(args.mode, args.output_dir, args.weights.resolve())
    except BaseException as error:
        result = {"status": "CONTROL_GRAPH_BUILD_FAILED", "mode": args.mode,
                  "error": str(error), "traceback": traceback.format_exc(),
                  "board_contacted": False}
        write(args.output_dir / "failure.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "target"}, indent=2))
    if result["status"] == "CONTROL_GRAPH_BUILD_FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
