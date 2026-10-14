#!/usr/bin/env python3
"""Build G2, same-node Input control and frozen F13-G YOLO-224 graphs locally."""

from __future__ import annotations

import argparse
import hashlib
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
from run_vta_c3_node24_partial_graph_prepare import _symbol_tir
from run_vta_c3_node24_partial_local import route
from vta.top.residency_dispatch import ExplicitResidencyDispatch
from vta.top.vta_node24_partial import (
    Node24GroupGEMM, Node24GroupWeightDMA, Node24SpatialBoundaryBridge,
    Node24WeightSlots, SplitNode24AltOutputDMA, SplitNode24InputDMA,
)


PREPARED = GROUPED / "20261008_c3_yolo224_incremental_prepare_run03"
PILOT = GROUPED / "20261009_c3_node24_group_gemm_run01"
ALT_TILE = (1, 7, 7, 4, 2)


def read(path):
    return json.loads(Path(path).read_text())


def build_variant(name, routes, program, params, output, env):
    archive = FullTIRArchive(env)
    target = tvm.target.Target(env.target, host=env.target_host)
    relay.backend.te_compiler.get().clear()
    lower = list(vta.build_config().config["tir.add_lower_pass"])
    if name == "InputAlt":
        lower.insert(1, (1, SplitNode24InputDMA(GRAPH_SYMBOL)))
        lower.insert(2, (1, SplitNode24AltOutputDMA(GRAPH_SYMBOL)))
    elif name == "F13G":
        lower.insert(1, (1, SplitNode24InputDMA(GRAPH_SYMBOL)))
        lower.append((2, Node24WeightSlots(13, True, GRAPH_SYMBOL)))
        lower.append((2, Node24GroupWeightDMA(GRAPH_SYMBOL)))
        lower.append((2, Node24SpatialBoundaryBridge(GRAPH_SYMBOL)))
        before_cpu = next(i for i, (_, p) in enumerate(lower)
                          if "CPUAccessRewrite" in str(p))
        lower.insert(before_cpu, (3, Node24GroupGEMM(GRAPH_SYMBOL)))
    else:
        raise ValueError("unexpected graph variant")
    with autotvm.tophub.context(target), ExplicitResidencyDispatch(routes) as dispatch:
        with tvm.transform.PassContext(
            opt_level=3,
            disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
            instruments=[archive],
            config={"tir.add_lower_pass": lower},
        ):
            graph, lib, lowered = relay.build(program, target=target, params=params)
    audit = dispatch.summary()
    if not audit["all_routes_scheduled"] or len(audit["schedule_hits"]) != len(routes):
        raise ValueError("incomplete exact graph route: " + name)
    local = output / name
    local.mkdir()
    info = save_build(local, graph, lib, lowered, cross_compiler(), {
        "deployment_variant": name, "routes": routes, "dispatch_audit": audit,
        "target_node": 24, "target_program": name,
    })
    functions = _symbol_tir(archive, GRAPH_SYMBOL)
    identities = {}
    for symbol, function in functions.items():
        encoded = tvm.ir.save_json(function)
        (local / (symbol + ".tir.json")).write_text(encoded)
        identities[symbol] = hashlib.sha256(encoded.encode()).hexdigest()
    write(local / "functions.json", identities)
    return info, functions


def prepare(args):
    first = read(PILOT / "board_run02/summary.json")
    second = read(args.operator_boot2 / "cross_boot_analysis.json")
    if first["status"] != "SINGLE_BOOT_COMPLETE" or \
            second["status"] != "TWO_BOOT_OPERATOR_CONFIRMED" or \
            not second["graph_eligible"]:
        raise ValueError("frozen F13-G has not passed two operator boots")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    base = read(G2 / "build.json")
    base_source = Path(read(PREPARED / "contract.json")["sources"]["base"])
    assets = read(base_source / "summary.json")["assets"]
    if not Path(assets["weights"]["path"]).is_file():
        assets["weights"] = {**assets["weights"], "path": str(
            Path(__file__).resolve().parents[3] /
            ".tvm_test_data/darknet/yolov3-tiny.weights")}
    for item in ("cfg", "weights", "darknet_lib"):
        if sha(assets[item]["path"]) != assets[item]["sha256"]:
            raise ValueError("frozen YOLO asset changed: " + item)
    original = read(PILOT / "board_run02/contract.json")
    confirmed = read(args.operator_boot2 / "contract.json")
    if confirmed["binaries_sha256"] != original["binaries_sha256"]:
        raise ValueError("operator binary identity changed across boots")
    args.output_dir.mkdir(parents=True)
    shutil.copytree(G2, args.output_dir / "G0")
    shutil.copytree(PREPARED / "stock", args.output_dir / "stock")
    builder = SimpleNamespace(**{key: assets[key]["path"] for key in
                                 ("cfg", "weights", "darknet_lib")})
    env = vta.get_env()
    program, params, shape = make_relay_program(env, builder)
    if list(shape) != [1, 3, 224, 224]:
        raise ValueError("network input is not frozen 224x224")
    old_routes = base["routes"]
    input_route = route("input_stationary", tile=ALT_TILE)
    fused_route = route("node24_partial_input_experimental")
    fused_route["candidate_id"] += "-F13-B-G"
    builds, maps = {}, {}
    for name, extra in (("InputAlt", input_route), ("F13G", fused_route)):
        builds[name], maps[name] = build_variant(
            name, old_routes + [extra], program, params, args.output_dir, env)
    if any(info["graph_sha256"] != base["graph_sha256"] for info in builds.values()):
        raise ValueError("graph JSON changed with target-only implementation")
    reference_params = semantic_params_hash(G2 / "params.bin")[0]
    reference_functions = read(G2 / "functions.json")
    for name in ("InputAlt", "F13G"):
        local = args.output_dir / name
        if semantic_params_hash(local / "params.bin")[0] != reference_params:
            raise ValueError("graph parameter semantics changed: " + name)
        if set(maps[name]) != set(reference_functions):
            raise ValueError("graph function inventory changed: " + name)
        for symbol, function in maps[name].items():
            if symbol == GRAPH_SYMBOL:
                continue
            old = tvm.ir.load_json((G2 / (symbol + ".tir.json")).read_text())
            if not tvm.ir.structural_equal(old, function, map_free_vars=True):
                raise ValueError("non-target implementation changed: " + symbol)
        if not builds[name]["memory_contract"]["fits"]:
            raise ValueError("graph memory contract exceeded: " + name)
    for name, binary in (("InputAlt", PILOT.parent /
                          "20261009_c3_node24_partial_run01/binaries/input_alt/node24_final.tir.json"),
                         ("F13G", PILOT / "binaries/f13_g/node24_final.tir.json")):
        isolated = tvm.ir.load_json(binary.read_text())
        if not tvm.ir.structural_equal(isolated.body, maps[name][GRAPH_SYMBOL].body,
                                       map_free_vars=True):
            raise ValueError("graph target does not match isolated operator: " + name)
    write(args.output_dir / "contract.json", {
        "status": "GRAPH_LOCAL_BUILT_BOARD_NOT_QUALIFIED",
        "operator_boot1": first["boot_id"],
        "operator_boot2": second["boot_ids"][1],
        "operator_cross_boot_sha256": sha(args.operator_boot2 / "cross_boot_analysis.json"),
        "base_graphlib_sha256": sha(G2 / "graphlib.so"),
        "base_graph_sha256": base["graph_sha256"],
        "stock_graphlib_sha256": sha(PREPARED / "stock/graphlib.so"),
        "target_node": 24, "target_symbol": GRAPH_SYMBOL,
        "unchanged_non_target_tir": True,
        "target_body_matches_isolated_operator": True,
        "graph_builds": builds,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-boot2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
