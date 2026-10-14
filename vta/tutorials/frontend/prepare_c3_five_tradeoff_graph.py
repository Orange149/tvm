#!/usr/bin/env python3
"""Build only the sealed, independently confirmed node24 prefetch graph."""

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import tvm
from tvm import autotvm, relay, tir
import vta

from c3_residency_tradeoff import read, digest, write_new
from c3_five_tradeoff_policy import NEW
from c3_five_tradeoff_prefetch import make_probe
from run_vta_c3_node24_group_gemm_graph_prepare import PREPARED, ALT_TILE
from run_vta_c3_node24_partial_board import G2, GRAPH_SYMBOL
from run_vta_c3_node24_partial_local import route
from run_vta_c3_node24_partial_graph_prepare import _symbol_tir
from build_vta_c3_evaluation_v5_graphs import save_build
from build_vta_resnet50_generic_residency_tir_audit import FullTIRArchive
from build_vta_yolov3_tiny_relay_residency_dispatch_pair import make_relay_program, cross_compiler
from run_vta_yolov3_tiny_relay_residency_dispatch_pair_board import semantic_params_hash
from vta.top.residency_dispatch import ExplicitResidencyDispatch
from vta.top.vta_node24_partial import SplitNode24InputDMA, Node24SpatialBoundaryBridge


def target_only(factory, *args, **kwargs):
    underlying = factory(*args, **kwargs)
    @tir.transform.prim_func_pass(opt_level=0, name="tir.vta.TargetOnlyNode24Prefetch")
    def wrapped(function, _module, _context):
        if function.attrs is None or "global_symbol" not in function.attrs or \
                str(function.attrs["global_symbol"]) != GRAPH_SYMBOL:
            return function
        return underlying(tvm.IRModule({"main": function}))["main"]
    return wrapped


def prepare(root):
    root = Path(root).resolve()
    results, winner = read(root / "analysis/results.json"), read(root / "winners.json")
    if not results["node24_new_graph_eligible"] or winner["winners"]["rule"] != NEW or \
            results["winners_sha256"] != digest(root / "winners.json"):
        raise ValueError("only the originally sealed and confirmed NEW may enter graph")
    bindings = read(root / "rule_freeze.json")
    for path, sha in bindings["source_bindings"].items():
        if digest(path) != sha:
            raise ValueError("frozen candidate source changed before graph build")
    destination = root / "graph_prepare"
    destination.mkdir(exist_ok=False)
    baseline = read(G2 / "build.json")
    source = Path(read(PREPARED / "contract.json")["sources"]["base"])
    assets = read(source / "summary.json")["assets"]
    if not Path(assets["weights"]["path"]).is_file():
        assets["weights"] = {**assets["weights"], "path": str(Path(__file__).resolve().parents[3] /
                              ".tvm_test_data/darknet/yolov3-tiny.weights")}
    for name in ("cfg", "weights", "darknet_lib"):
        if digest(assets[name]["path"]) != assets[name]["sha256"]:
            raise ValueError("frozen model asset identity changed")
    env = vta.get_env()
    program, params, shape = make_relay_program(env, SimpleNamespace(**{
        name: assets[name]["path"] for name in ("cfg", "weights", "darknet_lib")}))
    if list(shape) != [1, 3, 224, 224]:
        raise ValueError("not frozen YOLO-224 model")
    extra = route("node24_partial_input_experimental")
    extra["candidate_id"] = NEW
    routes = baseline["routes"] + [extra]
    namespace = make_probe().__globals__
    slots, dma, merged = (namespace[name] for name in ("Node24WeightSlots", "Node24GroupWeightDMA", "Node24GroupGEMM"))
    lower = list(vta.build_config().config["tir.add_lower_pass"])
    lower.insert(1, (1, SplitNode24InputDMA(GRAPH_SYMBOL)))
    lower.append((2, target_only(slots, 12, True)))
    lower.append((2, target_only(dma)))
    lower.append((2, Node24SpatialBoundaryBridge(GRAPH_SYMBOL)))
    before_cpu = next(i for i, (_, transform) in enumerate(lower) if "CPUAccessRewrite" in str(transform))
    lower.insert(before_cpu, (3, target_only(merged)))
    archive = FullTIRArchive(env)
    target = tvm.target.Target(env.target, host=env.target_host)
    relay.backend.te_compiler.get().clear()
    with autotvm.tophub.context(target), ExplicitResidencyDispatch(routes) as dispatch:
        with tvm.transform.PassContext(opt_level=3,
                disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}, instruments=[archive],
                config={"tir.add_lower_pass": lower}):
            graph, lib, lowered = relay.build(program, target=target, params=params)
    hits = dispatch.summary()
    if not hits["all_routes_scheduled"] or len(hits["schedule_hits"]) != len(routes):
        raise ValueError("graph route missing or silently fell back")
    functions = _symbol_tir(archive, GRAPH_SYMBOL)
    original = read(G2 / "functions.json")
    if set(functions) != set(original):
        raise ValueError("graph function inventory changed")
    for name, function in functions.items():
        if name == GRAPH_SYMBOL:
            isolated = tvm.ir.load_json((Path(bindings["programs"][NEW]["directory"]) / "node24_final.tir.json").read_text())
            if not tvm.ir.structural_equal(isolated.body, function.body, map_free_vars=True):
                raise ValueError("graph target does not implement exact isolated winner")
        else:
            prior = tvm.ir.load_json((G2 / (name + ".tir.json")).read_text())
            if not tvm.ir.structural_equal(prior, function, map_free_vars=True):
                raise ValueError("non-target implementation drift: " + name)
    info = save_build(destination, graph, lib, lowered, cross_compiler(), {
        "deployment_variant": NEW, "target_node": 24, "routes": routes, "dispatch_audit": hits})
    if info["graph_sha256"] != baseline["graph_sha256"] or \
            semantic_params_hash(destination / "params.bin")[0] != semantic_params_hash(G2 / "params.bin")[0] \
            or not info["memory_contract"]["fits"]:
        raise ValueError("graph/params/memory contract differs")
    identities = {}
    for name, function in functions.items():
        encoded = tvm.ir.save_json(function)
        (destination / (name + ".tir.json")).write_text(encoded)
        identities[name] = hashlib.sha256(encoded.encode()).hexdigest()
    write_new(destination / "functions.json", identities)
    # Reuse already compiled identical Input control graph and stock, rather
    # than alter non-target routes or recompile a reference.
    old = PREPARED.parent / "20261009_c3_node24_group_gemm_run01/graph_prepare_run02"
    input_root = old / "InputAlt"
    control = read(input_root / "build.json")
    if digest(input_root / "graphlib.so") != control["graphlib_sha256"] or \
            digest(input_root / "graph.json") != baseline["graph_sha256"]:
        raise ValueError("frozen practical Input graph missing or changed")
    write_new(destination / "deployment_contract.json", {
        "status": "EXACT_SEALED_PREFETCH_GRAPH_BUILT_FPGA_PENDING",
        "winner_sha256": digest(root / "winners.json"), "operator_result_sha256": digest(root / "analysis/results.json"),
        "deployments": {"current": str(G2), "Input": str(input_root), "rule": str(destination),
                        "stock": str(PREPARED / "stock")},
        "binary_sha256": {"current": digest(G2 / "graphlib.so"), "Input": digest(input_root / "graphlib.so"),
                          "rule": digest(destination / "graphlib.so"), "stock": digest(PREPARED / "stock/graphlib.so")},
        "target_node": 24, "target_symbol": GRAPH_SYMBOL, "target_program": NEW,
        "all_non_target_implementations_equal": True, "graph_JSON_and_params_equal": True,
        "single_boot_development_only": True, "source_sha256": digest(__file__)})
    print("EXACT_SEALED_PREFETCH_GRAPH_BUILT_FPGA_PENDING")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    prepare(parser.parse_args().output_dir)
