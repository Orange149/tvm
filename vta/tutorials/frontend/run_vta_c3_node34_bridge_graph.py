#!/usr/bin/env python3
"""Cross-build only node34's experiment-only grouped token bridge in frozen G2."""

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
from build_vta_yolov3_tiny_relay_residency_dispatch_pair import (
    cross_compiler, make_relay_program,
)
from c3_node34_group_bridge import SYMBOL, build_config_with_bridge
from c3_reuse_semantics import extract_requests
from c3_thread_sync_audit import final_tir_audit
from run_vta_yolov3_tiny_relay_residency_dispatch_pair_board import semantic_params_hash


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
G2 = GROUPED / "20261008_c3_yolo224_incremental_prepare_run03/G2"
MODE4 = GROUPED / "20261008_c3_node34_mode4_local_gate_run02"
BRIDGE = GROUPED / "20261008_c3_node34_bridge_fsim_run01"
ASSETS = GROUPED / "20260926_c3_yolo224_graph_build_run02/summary.json"
NODE = 34


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(output, weights):
    local = json.loads((BRIDGE / "result.json").read_text())
    if local["status"] != "BRIDGE_LOCAL_FSIM_PASSED_BOARD_NOT_QUALIFIED":
        raise ValueError("experiment-only bridge local FSim is not qualified")
    frozen = json.loads((G2 / "build.json").read_text())
    assets = json.loads(ASSETS.read_text())["assets"]
    paths = {key: Path(assets[key]["path"]) for key in ("cfg", "darknet_lib")}
    paths["weights"] = weights
    for key, path in paths.items():
        if not path.is_file() or sha(path) != assets[key]["sha256"]:
            raise ValueError("frozen YOLO asset missing or changed: " + key)
    if sha(G2 / "graph.json") != frozen["graph_sha256"]:
        raise ValueError("frozen G2 graph changed")
    routes = list(frozen["routes"]) + [json.loads((MODE4 / "candidate.json").read_text())["routes"][-1]]
    if len(routes) != 6:
        raise ValueError("five frozen routes plus node34 required")
    env = vta.get_env()
    if env.TARGET != "axu5evb":
        raise ValueError("expected fixed axu5evb target")
    args = SimpleNamespace(**{key: str(value) for key, value in paths.items()})
    program, params, shape = make_relay_program(env, args)
    if tuple(shape) != (1, 3, 224, 224):
        raise ValueError("YOLO-224 shape changed")
    target = tvm.target.Target(env.target, host=env.target_host)
    archive = FullTIRArchive(env)
    context, changed = build_config_with_bridge(SYMBOL)
    relay.backend.te_compiler.get().clear()
    with autotvm.tophub.context(target), ExplicitResidencyDispatch(routes) as dispatch:
        with tvm.transform.PassContext(
            opt_level=3, config=context.config,
            disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
            instruments=[archive],
        ):
            graph, lib, lowered = relay.build(program, target=target, params=params)
    audit = dispatch.summary()
    if not audit["all_routes_scheduled"] or len(audit["schedule_hits"]) != 6 or len(changed) != 1:
        raise ValueError("exact node34 bridge routing or pass did not fire once")
    graph_hash = hashlib.sha256(graph.encode()).hexdigest()
    if graph_hash != frozen["graph_sha256"]:
        raise ValueError("bridge changed Graph JSON")
    (output / "graph.json").write_text(graph)
    (output / "params.bin").write_bytes(relay.save_param_dict(lowered))
    if semantic_params_hash(output / "params.bin")[0] != semantic_params_hash(G2 / "params.bin")[0]:
        raise ValueError("bridge graph parameters changed")
    graph_obj = json.loads(graph)
    if graph_obj["nodes"][NODE]["attrs"]["func_name"] != SYMBOL:
        raise ValueError("node34 symbol changed")
    captured = {}
    for snapshot in archive.snapshots:
        module = tvm.ir.load_json(snapshot["json"])
        for function in module.functions.values():
            if not isinstance(function, tir.PrimFunc) or function.attrs is None \
                    or "global_symbol" not in function.attrs:
                continue
            name = str(function.attrs["global_symbol"])
            if name in captured:
                raise ValueError("duplicate fused function in TIR archive: " + name)
            captured[name] = function
    frozen_functions = json.loads((G2 / "functions.json").read_text())
    if set(captured) != set(frozen_functions) or SYMBOL not in captured:
        raise ValueError("non-target function inventory drift")
    for name, function in captured.items():
        if name == SYMBOL:
            continue
        historical = tvm.ir.load_json((G2 / (name + ".tir.json")).read_text())
        if not tvm.ir.structural_equal(function, historical, map_free_vars=True):
            raise ValueError("non-target TIR changed: " + name)
    function = captured[SYMBOL]
    (output / "node34.tir.json").write_text(tvm.ir.save_json(function))
    (output / "node34.tir").write_text(function.script())
    single = tvm.IRModule({"main": function})
    requests = extract_requests(single, env)
    address = final_tir_audit(single, requests)
    old = json.loads((MODE4 / "local_evidence.json").read_text())
    if address["status"] != "passed" or len(address["sync_event_positions"]) != 1:
        raise ValueError("bridge address or function-end sync audit failed")
    if requests["totals"] != old["request_semantics"]["totals"] \
            or address["ordered_events"] != old["address_audit"]["ordered_events"]:
        raise ValueError("bridge changed node34 requests or compute order")
    lib.export_library(str(output / "graphlib.so"), cross_compiler())
    result = {
        "status": "EXPERIMENTAL_GRAPH_BUILT_BOARD_NOT_QUALIFIED",
        "frozen_g2_graphlib_sha256": frozen["graphlib_sha256"],
        "bridge_graphlib_sha256": sha(output / "graphlib.so"),
        "graph_sha256": graph_hash,
        "parameter_semantics_unchanged": True,
        "non_target_tir_structurally_equal": True,
        "old_routes_unchanged": True,
        "target": {"node": NODE, "symbol": SYMBOL, "changed": changed,
                   "request_totals": requests["totals"], "address_status": address["status"],
                   "function_syncs": len(address["sync_event_positions"]),
                   "max_kernel_uops": address["max_kernel_uops"]},
        "dispatch": audit, "board_contacted": False,
    }
    save(output / "build.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path,
                        default=Path(".tvm_test_data/darknet/yolov3-tiny.weights"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = run(args.output_dir, args.weights.resolve())
    except BaseException as error:
        result = {"status": "EXPERIMENTAL_GRAPH_BUILD_FAILED", "error": str(error),
                  "traceback": traceback.format_exc(), "board_contacted": False}
        save(args.output_dir / "failure.json", result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("dispatch", "target", "traceback")}, indent=2))
    if result["status"] == "EXPERIMENTAL_GRAPH_BUILD_FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
