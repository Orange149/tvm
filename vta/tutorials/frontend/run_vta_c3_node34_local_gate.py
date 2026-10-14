#!/usr/bin/env python3
"""Fail-closed, local-only first gate for YOLO-224 node34 grouped W.

This intentionally does not contact the FPGA or create a new schedule.  It
compiles the *full fused graph* with only node34 changed from frozen G2 to
mode4, then records the actual fused TIR and request/address evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import traceback


HERE = Path(__file__).resolve().parent
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
G2 = GROUPED / "20261008_c3_yolo224_incremental_prepare_run03/G2"
ASSETS = GROUPED / "20260926_c3_yolo224_graph_build_run02/summary.json"
TASKS = GROUPED / "20260926_c3_yolo224_static_prepare_run01/resolution_224/result.json"
TASK_ID = "8db9d39edb5574ceb8d6220fc3b3a8676ec13915c11a2c6f3fb6f6d8f33411ad"
NODE = 34
SYMBOL = "tvmgen_default_fused_nn_conv2d_add_right_shift_clip_cast_5"
TILE = (1, 7, 1, 1, 1)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def route(workload):
    knobs = ("tile_b", "tile_h", "tile_w", "tile_ci", "tile_co")
    entity = [[key, "sp", [-1, value]] for key, value in zip(knobs, TILE)]
    entity.extend((["oc_nthread", "ot", 1], ["h_nthread", "ot", 1]))
    return {
        "candidate_id": "node34-fixed-tile-mode4-local-gate",
        "execution_task_id": TASK_ID,
        "identity": {
            "workload": workload,
            "complete_config_entity": {"code_hash": None, "entity": entity},
            "public_mode": "weight_resident_barrier",
            "implementation_mode": 4,
        },
    }


def run(output, weights):
    import tvm
    from tvm import autotvm, relay, tir
    import vta
    from vta.top.residency_dispatch import ExplicitResidencyDispatch
    from build_vta_yolov3_tiny_relay_residency_dispatch_pair import make_relay_program
    from build_vta_resnet50_generic_residency_tir_audit import FullTIRArchive
    from build_vta_c3_mode11_graph_context_contract import RelayFusionArchive
    from c3_reuse_semantics import extract_requests
    from c3_thread_sync_audit import final_tir_audit
    import c3_yolo224_incremental as protocol

    frozen = json.loads((G2 / "build.json").read_text())
    task_rows = json.loads(TASKS.read_text())["tasks"]
    task = next(row for row in task_rows if row["task_id"] == TASK_ID)
    asset_rows = json.loads(ASSETS.read_text())["assets"]
    assets = {key: Path(asset_rows[key]["path"]) for key in ("cfg", "darknet_lib")}
    assets["weights"] = weights
    for key, path in assets.items():
        if not path.is_file() or digest(path) != asset_rows[key]["sha256"]:
            raise ValueError("frozen model asset missing or drifted: " + key)
    if digest(G2 / "graph.json") != frozen["graph_sha256"]:
        raise ValueError("frozen G2 graph identity drift")
    routes = list(frozen["routes"]) + [route(task["workload"])]
    write(output / "candidate.json", {"frozen_g2_graph_sha256": frozen["graph_sha256"],
                                      "frozen_g2_graphlib_sha256": frozen["graphlib_sha256"],
                                      "routes": routes, "node": NODE, "symbol": SYMBOL,
                                      "assets": {k: {"path": str(v), "sha256": digest(v)}
                                                 for k, v in assets.items()}})
    env = vta.get_env()
    args = SimpleNamespace(**{k: str(v) for k, v in assets.items()})
    program, params, shape = make_relay_program(env, args)
    if tuple(shape) != (1, 3, 224, 224):
        raise ValueError("YOLO shape drift")
    target = tvm.target.Target(env.target, host=env.target_host)
    archive, relay_archive = FullTIRArchive(env), RelayFusionArchive()
    relay.backend.te_compiler.get().clear()
    with autotvm.tophub.context(target), ExplicitResidencyDispatch(routes) as dispatch:
        with vta.build_config(opt_level=3,
                              disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
                              instruments=[archive, relay_archive]):
            graph, lib, lowered = relay.build(program, target=target, params=params)
    audit = dispatch.summary()
    write(output / "dispatch.json", audit)
    if not audit["all_routes_scheduled"] or len(audit["schedule_hits"]) != len(routes):
        raise ValueError("node34 mode4 exact route not scheduled once")
    if hashlib.sha256(graph.encode()).hexdigest() != frozen["graph_sha256"]:
        raise ValueError("graph JSON differs from frozen G2")
    parsed = json.loads(graph)
    if parsed["nodes"][NODE]["attrs"]["func_name"] != SYMBOL:
        raise ValueError("node34 symbol drift")
    (output / "graph.json").write_text(graph)
    (output / "params.bin").write_bytes(relay.save_param_dict(lowered))
    matches = []
    for snapshot in archive.snapshots:
        module = tvm.ir.load_json(snapshot["json"])
        for function in module.functions.values():
            if isinstance(function, tir.PrimFunc) and function.attrs is not None \
                    and str(function.attrs["global_symbol"]) == SYMBOL:
                matches.append(function)
    if len(matches) != 1:
        raise ValueError("missing or ambiguous full-fusion node34 TIR")
    function = matches[0]
    (output / "node34.tir").write_text(function.script())
    (output / "node34.tir.json").write_text(tvm.ir.save_json(function))
    single = tvm.IRModule({"main": function})
    requests = extract_requests(single, env)
    address = final_tir_audit(single, requests)
    inventory = next(row for row in protocol.graph_inventory(parsed) if row["node"] == NODE)
    fusion = protocol.fusion_semantics([x["relay"] for x in relay_archive.snapshots], inventory)
    write(output / "local_evidence.json", {
        "status": "FULL_FUSION_LOWERED_NOT_FSIM_QUALIFIED",
        "node": NODE, "symbol": SYMBOL, "abi": inventory, "fusion": fusion,
        "request_semantics": requests, "address_audit": address,
        "coproc_sync_occurrences": function.script().count("coproc_sync"),
        "runtime_uop_and_submission_evidence": "unknown",
        "fsim_three_seed_correctness": "not_run",
        "board_contacted": False,
    })
    return {"status": "FULL_FUSION_LOWERED_NOT_FSIM_QUALIFIED",
            "wgt_request_bytes": requests["totals"].get("load_buffer_2d_wgt_bytes"),
            "address_audit_status": address.get("status"),
            "coproc_sync_occurrences": function.script().count("coproc_sync")}


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
        result = {"status": "LOCAL_GATE_FAILED", "error": str(error),
                  "traceback": traceback.format_exc(), "board_contacted": False}
    write(args.output_dir / "status.json", result)
    write(args.output_dir / "hashes.json", {
        str(path.relative_to(args.output_dir)): digest(path)
        for path in sorted(args.output_dir.rglob("*"))
        if path.is_file() and path.name != "hashes.json"
    })
    print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, indent=2))
    if result["status"] == "LOCAL_GATE_FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
