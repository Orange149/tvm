#!/usr/bin/env python3
"""Capture the exact v6.1 four-argument fusion contract without contacting the board."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import time

import tvm
from tvm import autotvm, relay, tir
import vta
from vta.top.residency_dispatch import ExplicitResidencyDispatch

from build_vta_c3_evaluation_v5_graphs import (
    DEFAULT_PREPARE,
    make_relay_program,
    sha256,
)
from build_vta_resnet50_generic_residency_tir_audit import FullTIRArchive
from c3_reuse_semantics import extract_requests
from c3_mode11_graph_context import (
    ADD_NODE,
    TARGET_SYMBOL,
    frozen_graph_contract,
    sha256_file,
    write_json,
)
from run_vta_yolov3_tiny_relay_residency_dispatch_pair_board import semantic_params_hash


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
DEFAULT_MAIN = GROUPED / "20260926_c3_yolo512_evaluation_v5_graph_build_run01"
DEFAULT_DIAGNOSTIC = (
    GROUPED / "20260926_c3_yolo512_evaluation_v5_graph_correctness_diagnostic_build_run01"
)
DEFAULT_OUTPUT = GROUPED / "20260926_c3_mode11_graph_context_contract_v61_run03"
TASK_ID = "eec24a993bafd33913b45b041fb85141365507dc23ac18c7d0f1115f39c93f18"
PROGRAM_ID = "f1b9f216d9cf3dcbd8420e62517be02083c9110ba7411c7ee7f3a813bf3ef5c5"
CANDIDATE_ID = "ce645766d9859fcc4131a3bedc9156945d9101a4e2b61c6e25a720408d5afa82"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exact_routes(build_dir):
    rows = read(Path(build_dir) / "build.json")["routes"]
    if len(rows) != 2:
        raise ValueError("frozen graph build must contain exactly two routes")
    return rows


@tvm.instrument.pass_instrument
class RelayFusionArchive:
    """Retain the actual Relay modules immediately after relevant fusion passes."""

    def __init__(self):
        self.snapshots = []

    def run_after_pass(self, mod, info):
        if "FuseOps" not in str(info.name):
            return
        text = mod.astext(show_meta_data=False)
        self.snapshots.append({"pass": str(info.name), "relay": text})


def save_target_tir(output, archive, env):
    matches = []
    for index, snapshot in enumerate(archive.snapshots):
        functions = [row for row in snapshot["functions"] if row["global_symbol"] == TARGET_SYMBOL]
        if not functions:
            continue
        if len(functions) != 1:
            raise RuntimeError("ambiguous target PrimFunc in CPUAccessRewrite snapshot")
        stem = "target_cpu_access_rewrite_{:03d}".format(index)
        json_path, tir_path = output / (stem + ".json"), output / (stem + ".tir")
        json_path.write_text(snapshot["json"], encoding="utf-8")
        tir_path.write_text(snapshot["tir"], encoding="utf-8")
        module = tvm.ir.load_json(snapshot["json"])
        target_functions = [
            function for function in module.functions.values()
            if isinstance(function, tir.PrimFunc)
            and function.attrs is not None
            and "global_symbol" in function.attrs
            and str(function.attrs["global_symbol"]) == TARGET_SYMBOL
        ]
        if len(target_functions) != 1:
            raise RuntimeError("target function missing from captured TIR JSON")
        requests = extract_requests(tvm.IRModule({"main": target_functions[0]}), env)
        matches.append({
            "pass": snapshot["pass"],
            "function": functions[0],
            "json_file": json_path.name,
            "json_sha256": sha256_file(json_path),
            "tir_file": tir_path.name,
            "tir_sha256": sha256_file(tir_path),
            "coproc_sync_occurrences": int(snapshot["coproc_sync_occurrences"]),
            "request_semantics": requests,
        })
    if len(matches) != 1:
        raise RuntimeError("target TIR capture was not unique: {}".format(len(matches)))
    return matches[0]


def build_observed(name, routes, program, params, output, env, frozen):
    local = output / name
    local.mkdir()
    target = tvm.target.Target(env.target, host=env.target_host)
    tir_archive, relay_archive = FullTIRArchive(env), RelayFusionArchive()
    relay.backend.te_compiler.get().clear()
    started = time.perf_counter()
    with autotvm.tophub.context(target):
        with ExplicitResidencyDispatch(routes) as dispatch:
            with vta.build_config(
                opt_level=3,
                disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
                instruments=[relay_archive, tir_archive],
            ):
                graph, _lib, lowered_params = relay.build(program, target=target, params=params)
    audit = dispatch.summary()
    if not audit["all_routes_scheduled"] or len(audit["schedule_hits"]) != 2:
        raise RuntimeError(name + " exact route dispatch failed")
    graph_path = local / "graph.json"
    params_path = local / "params.bin"
    graph_path.write_text(graph, encoding="utf-8")
    params_path.write_bytes(relay.save_param_dict(lowered_params))
    if sha256_file(graph_path) != sha256_file(frozen / "graph.json"):
        raise RuntimeError(name + " observation rebuild changed graph JSON")
    if semantic_params_hash(params_path)[0] != semantic_params_hash(frozen / "params.bin")[0]:
        raise RuntimeError(name + " observation rebuild changed parameter semantics")
    relay_files = []
    for index, snapshot in enumerate(relay_archive.snapshots):
        path = local / "relay_after_fuse_ops_{:02d}.txt".format(index)
        path.write_text(snapshot["relay"], encoding="utf-8")
        relay_files.append({"pass": snapshot["pass"], "file": path.name, "sha256": sha256_file(path)})
    if not relay_files:
        raise RuntimeError("no Relay FuseOps snapshot was captured")
    target_tir = save_target_tir(local, tir_archive, env)
    return {
        "variant": name,
        "frozen_dir": str(frozen),
        "frozen_graphlib_sha256": sha256_file(frozen / "graphlib.so"),
        "frozen_graph_sha256": sha256_file(frozen / "graph.json"),
        "frozen_params_sha256": sha256_file(frozen / "params.bin"),
        "observation_graph_sha256": sha256_file(graph_path),
        "observation_params_sha256": sha256_file(params_path),
        "routes": routes,
        "dispatch_audit": audit,
        "relay_fusion_snapshots": relay_files,
        "target_tir": target_tir,
        "build_seconds": time.perf_counter() - started,
    }


def _exact_relay_fusion(text):
    blocks = re.findall(
        r"(?P<block>fn \([^}]+?Primitive=1\)[^{]+\{.*?\n  \} /\* ty=fn)",
        text,
        flags=re.S,
    )
    required = (
        "Tensor[(1, 32, 16, 16, 1, 16), int8]",
        "Tensor[(16, 32, 1, 1, 16, 16), int8]",
        "Tensor[(16, 1, 1, 1, 16), int32]",
        "kernel_size=[1, 1]",
        'out_dtype="int32"',
        "right_shift(",
        ", 6 /* ty=int32 */)",
        "clip(",
        "a_min=-127f, a_max=127f",
        "cast(",
        'dtype="int8"',
    )
    matching = [block for block in blocks if all(token in block for token in required)]
    if len(matching) != 1:
        raise ValueError("actual Relay target fusion was not uniquely resolved")
    body = matching[0]
    order = [body.find(token) for token in ("nn.conv2d", "add(", "right_shift(", "clip(", "cast(")]
    if any(index < 0 for index in order) or order != sorted(order):
        raise ValueError("actual Relay fusion operation order is unresolved")
    return body


def semantic_contract(graph_contract, observed):
    """Resolve semantics from the frozen graph ABI plus captured compiler IR."""
    p28 = graph_contract["tensors"]["p28"]
    y = graph_contract["tensors"]["Y"]
    if p28["shape"] != [16, 1, 1, 1, 16] or y["shape"] != [1, 16, 16, 16, 1, 16]:
        raise ValueError("p28/Y shape no longer proves the frozen channel broadcast")
    evidence = []
    for row in observed:
        variant_evidence = []
        for snapshot in row["relay_fusion_snapshots"]:
            path = Path(row["_output_dir"]) / row["variant"] / snapshot["file"]
            block = _exact_relay_fusion(path.read_text(encoding="utf-8"))
            variant_evidence.append({
                "file": snapshot["file"],
                "sha256": snapshot["sha256"],
                "exact_fusion_block_sha256": __import__("hashlib").sha256(
                    block.encode("utf-8")
                ).hexdigest(),
            })
        if not variant_evidence:
            raise ValueError(row["variant"] + " lacks actual fused Relay operation evidence")
        tir_path = Path(row["_output_dir"]) / row["variant"] / row["target_tir"]["tir_file"]
        tir_text = tir_path.read_text(encoding="utf-8")
        tir_required = (
            'p2: T.Buffer((16, 1, 1, 1, 16), "int32")',
            '"VTALoadBuffer2D"',
            "p2.data",
            "T.tir.vta.uop_push(1, 0, 0,",
            ", 2, 0, 0)",
            ", 3, 1, 6)",
            ", 0, 1, 127)",
            ", 1, 1, -127)",
            '"VTAStoreBuffer2D"',
        )
        if not all(token in tir_text for token in tir_required):
            raise ValueError(row["variant"] + " target TIR ALU contract is unresolved")
        requests = row["target_tir"]["request_semantics"]
        store_bytes = requests["totals"].get("store_buffer_2d_out_bytes")
        unique_store = requests["unique_source_region_bytes"].get("store_out")
        if store_bytes != y["nbytes"] or unique_store != y["nbytes"]:
            raise ValueError(row["variant"] + " target output STORE coverage is incomplete")
        evidence.append({"variant": row["variant"], "relay_snapshots": variant_evidence})
    return {
        "status": "FULL_FUSION_CONTRACT_RESOLVED",
        "source": "frozen graph ABI plus exact-route Relay FuseOps and CPUAccessRewrite observations",
        "operation_order": [
            "conv2d int8*int8 accumulated in int32",
            "add p28 in int32",
            "arithmetic right shift by 6 in signed int32",
            "clip to [-127, 127]",
            "cast to int8",
        ],
        "convolution_accumulator_dtype": "int32",
        "p28_add_dtype": "int32",
        "p28_add_position": "after conv2d accumulation and before right_shift",
        "p28_broadcast_index": "p28[co_outer, 0, 0, 0, co_inner]",
        "p28_broadcast_axes_in_Y": ["batch", "height", "width", "batch_inner"],
        "right_shift": 6,
        "negative_right_shift_semantics": "signed arithmetic right shift",
        "clip": [-127, 127],
        "cast_order": "after clip",
        "output_store_coverage": {
            "status": "FULL_STATIC_COVERAGE",
            "bytes": y["nbytes"],
            "proof": "total and unique VTA OUT STORE regions both equal output nbytes",
        },
        "evidence": evidence,
    }


def run(args):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    main, diagnostic = Path(args.main_build).resolve(), Path(args.diagnostic_build).resolve()
    incumbent = main / "incumbent"
    mode11 = diagnostic / "mode11_only_eec24a99"
    for path in (incumbent, mode11, Path(args.cfg), Path(args.weights), Path(args.darknet_lib)):
        if not path.exists():
            raise FileNotFoundError(path)
    incumbent_routes, mode11_routes = exact_routes(incumbent), exact_routes(mode11)
    selected = [row for row in mode11_routes if row["execution_task_id"] == TASK_ID]
    if len(selected) != 1 or selected[0]["candidate_id"] != PROGRAM_ID:
        raise ValueError("frozen mode11 program identity drift")
    if selected[0]["representative_candidate_id"] != CANDIDATE_ID:
        raise ValueError("frozen mode11 candidate binary identity drift")
    output.mkdir(parents=True)
    graph = json.loads((incumbent / "graph.json").read_text(encoding="utf-8"))
    graph_contract = frozen_graph_contract(graph)
    env = vta.get_env()
    program, params, shape = make_relay_program(env, args)
    if tuple(shape) != (1, 3, 512, 512):
        raise ValueError("diagnostic model is not frozen YOLO-512")
    observed = [
        build_observed("incumbent", incumbent_routes, program, params, output, env, incumbent),
        build_observed("mode11_singleton", mode11_routes, program, params, output, env, mode11),
    ]
    for row in observed:
        row["_output_dir"] = str(output)
    semantics = semantic_contract(graph_contract, observed)
    for row in observed:
        row.pop("_output_dir", None)
    contract = {
        "schema": "c3_mode11_graph_context_contract_v61",
        "status": semantics["status"],
        "formal_latency_forbidden": True,
        "seed": 0,
        "execution_task_id": TASK_ID,
        "program_id": PROGRAM_ID,
        "candidate_binary_id": CANDIDATE_ID,
        "target_symbol": TARGET_SYMBOL,
        "target_node": 61,
        "input_node": 58,
        "parameter_nodes": [59, ADD_NODE],
        "successor_node": 62,
        "graph_contract": graph_contract,
        "fusion_semantics": semantics,
        "observations": observed,
        "frozen_candidate_so": str(Path(args.candidate_so).resolve()),
        "frozen_candidate_so_sha256": sha256_file(args.candidate_so),
        "debug_api_policy": {
            "allowed": ["execute_node", "get_node_output_immediately_after_last_node"],
            "forbidden": ["run", "debug_get_output", "run_individual*", "profile*"],
            "nodes_strictly_increasing": True,
        },
        "latency_fields": None,
        "board_contacted": False,
    }
    write_json(output / "contract.json", contract)
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({
        "status": contract["status"],
        "target_symbol": TARGET_SYMBOL,
        "variants": [row["variant"] for row in observed],
    }, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-build", default=str(DEFAULT_MAIN))
    parser.add_argument("--diagnostic-build", default=str(DEFAULT_DIAGNOSTIC))
    parser.add_argument("--cfg", default=str(DEFAULT_PREPARE / "resolution_512/yolov3-tiny-512.cfg"))
    parser.add_argument("--weights", default="/tmp/tvm_test_data/darknet/yolov3-tiny.weights")
    parser.add_argument("--darknet-lib", default="/home/orange/.tvm_test_data/darknet/libdarknet2.0.so")
    parser.add_argument("--candidate-so", default=str(
        GROUPED / "20260924_c3_yolo512_selector_v4_production_cross_compile_run01/binaries/"
        "ce645766d9859fcc4131a3bedc9156945d9101a4e2b61c6e25a720408d5afa82.so"
    ))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
