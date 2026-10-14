#!/usr/bin/env python3
"""Build the frozen node24 first-boot winner into YOLO-224, locally only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import tvm
from tvm import autotvm, relay, tir
import vta

from build_vta_c3_evaluation_v5_graphs import save_build
from build_vta_resnet50_generic_residency_tir_audit import FullTIRArchive
from build_vta_yolov3_tiny_relay_residency_dispatch_pair import make_relay_program, cross_compiler
from run_vta_yolov3_tiny_relay_residency_dispatch_pair_board import semantic_params_hash
from run_vta_c3_node24_partial_board import G2, GROUPED, GRAPH_SYMBOL, sha, write
from run_vta_c3_node24_partial_local import route
from vta.top.residency_dispatch import ExplicitResidencyDispatch
from vta.top.vta_node24_partial import (
    Node24SpatialBoundaryBridge, Node24WeightSlots, SplitNode24InputDMA,
)


PREPARED = GROUPED / "20261008_c3_yolo224_incremental_prepare_run03"


def _json(path):
    return json.loads(Path(path).read_text())


def _symbol_tir(archive, symbol):
    result = {}
    for snapshot in archive.snapshots:
        module = tvm.ir.load_json(snapshot["json"])
        for function in module.functions.values():
            if not isinstance(function, tir.PrimFunc) or function.attrs is None or \
                    "global_symbol" not in function.attrs:
                continue
            name = str(function.attrs["global_symbol"])
            if name in result:
                raise ValueError("duplicate function TIR: " + name)
            result[name] = function
    if symbol not in result:
        raise ValueError("node24 full-fusion TIR missing: " + symbol)
    return result


def build_variant(name, routes, program, params, output, env, winner):
    archive = FullTIRArchive(env)
    target = tvm.target.Target(env.target, host=env.target_host)
    relay.backend.te_compiler.get().clear()
    build = vta.build_config()
    lower = list(build.config["tir.add_lower_pass"])
    if name == "F":
        resident = 12 if winner == "f12" else 13
        lower.insert(1, (1, SplitNode24InputDMA(GRAPH_SYMBOL)))
        lower.append((2, Node24WeightSlots(resident, True, GRAPH_SYMBOL)))
        lower.append((2, Node24SpatialBoundaryBridge(GRAPH_SYMBOL)))
    with autotvm.tophub.context(target), ExplicitResidencyDispatch(routes) as dispatch:
        with tvm.transform.PassContext(
                opt_level=3,
                disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
                instruments=[archive],
                config={"tir.add_lower_pass": lower}):
            graph, lib, lowered = relay.build(program, target=target, params=params)
    audit = dispatch.summary()
    if not audit["all_routes_scheduled"] or len(audit["schedule_hits"]) != len(routes):
        raise ValueError("incomplete exact graph route: " + name)
    location = output / name
    location.mkdir()
    info = save_build(location, graph, lib, lowered, cross_compiler(), {
        "deployment_variant": name, "routes": routes, "dispatch_audit": audit,
        "node24_winner": winner,
    })
    symbol_tir = _symbol_tir(archive, GRAPH_SYMBOL)
    function_ids = {}
    for symbol, function in symbol_tir.items():
        encoded = tvm.ir.save_json(function)
        (location / (symbol + ".tir.json")).write_text(encoded)
        function_ids[symbol] = sha(encoded.encode())
    write(location / "functions.json", function_ids)
    return info, symbol_tir


def prepare(args):
    result = _json(args.operator / "summary.json")
    sealed = _json(args.operator / "winner.json")
    winner = sealed["candidate"]
    if result["status"] != "OPERATOR_FIRST_BOOT_COMPLETE" or \
            not result["all_correct"] or winner not in ("f12", "f13") or \
            winner != result["graph_winner"]:
        raise ValueError("no eligible, sealed node24 graph winner")
    if sha(args.operator / "contract.json") != sealed["contract_sha256"]:
        raise ValueError("operator timing contract drift")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    base = _json(G2 / "build.json")
    source = _json(PREPARED / "summary.json")["assets"]
    for entry in ("cfg", "weights", "darknet_lib"):
        asset = source[entry]
        if sha(asset["path"]) != asset["sha256"]:
            raise ValueError("frozen YOLO asset drift: " + entry)
    args.output_dir.mkdir(parents=True)
    shutil.copytree(G2, args.output_dir / "G0")
    builder = SimpleNamespace(**{key: source[key]["path"] for key in
                                 ("cfg", "weights", "darknet_lib")})
    env = vta.get_env()
    program, params, shape = make_relay_program(env, builder)
    if list(shape) != [1, 3, 224, 224]:
        raise ValueError("not the frozen YOLO-224 input")
    old_routes = base["routes"]
    node24 = {
        "Input": route("input_stationary"),
        "F": route("node24_partial_input_experimental"),
    }
    node24["F"]["candidate_id"] += "-" + winner.upper()
    builds = {}
    functions = {}
    for name in ("Input", "F"):
        info, tir_map = build_variant(name, old_routes + [node24[name]],
                                      program, params, args.output_dir, env, winner)
        builds[name], functions[name] = info, tir_map
    if not all(info["graph_sha256"] == base["graph_sha256"] for info in builds.values()):
        raise ValueError("Graph JSON identity drift")
    old_params = semantic_params_hash(G2 / "params.bin")[0]
    for name in ("Input", "F"):
        if semantic_params_hash(args.output_dir / name / "params.bin")[0] != old_params:
            raise ValueError("frozen graph parameter semantics drift")
    reference = _json(G2 / "functions.json")
    for name in ("Input", "F"):
        if set(functions[name]) != set(reference):
            raise ValueError("graph function inventory drift")
        for symbol, function in functions[name].items():
            if symbol == GRAPH_SYMBOL:
                continue
            old = tvm.ir.load_json((G2 / (symbol + ".tir.json")).read_text())
            if not tvm.ir.structural_equal(old, function, map_free_vars=True):
                raise ValueError("non-node24 function changed: " + symbol)
    if not all(info["memory_contract"]["fits"] for info in builds.values()):
        raise ValueError("device memory contract exceeded")
    write(args.output_dir / "contract.json", {
        "status": "GRAPH_LOCAL_BUILT_BOARD_NOT_QUALIFIED",
        "frozen_winner": winner,
        "operator_winner_sha256": sha(args.operator / "winner.json"),
        "base_graphlib_sha256": sha(G2 / "graphlib.so"),
        "base_graph_sha256": base["graph_sha256"],
        "target_node": 24,
        "target_symbol": GRAPH_SYMBOL,
        "unchanged_non_target_tir": True,
        "graph_builds": builds,
    })
    print(json.dumps({"status": "GRAPH_LOCAL_BUILT_BOARD_NOT_QUALIFIED",
                      "winner": winner}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
