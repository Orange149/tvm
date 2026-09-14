#!/usr/bin/env python3
"""Build stock-TopHub YOLO and pair it with an existing selected residency build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import tvm
from tvm import autotvm, relay
import vta

from build_vta_yolov3_tiny_relay_residency_dispatch_pair import (
    cross_compiler,
    make_relay_program,
    sha256,
    write_json,
)


def run(args):
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError("refusing to overwrite " + str(output))
    output.mkdir(parents=True)
    env = vta.get_env()
    relay_program, params, shape = make_relay_program(env, args)
    target = tvm.target.Target(env.target, host=env.target_host)
    relay.backend.te_compiler.get().clear()
    started = time.perf_counter()
    with autotvm.tophub.context(target):
        with vta.build_config(
            opt_level=3,
            disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
        ):
            graph, lib, lowered_params = relay.build(relay_program, target=target, params=params)
    stock = output / "stock_tophub"
    stock.mkdir()
    (stock / "graph.json").write_text(graph, encoding="utf-8")
    (stock / "params.bin").write_bytes(relay.save_param_dict(lowered_params))
    lib.export_library(str(stock / "graphlib.so"), cross_compiler())
    stock_build = {
        "candidate_id": None,
        "public_mode": "stock_tophub",
        "build_seconds": time.perf_counter() - started,
        "graph_sha256": sha256(stock / "graph.json"),
        "params_sha256": sha256(stock / "params.bin"),
        "graphlib_sha256": sha256(stock / "graphlib.so"),
    }

    selected_root = Path(args.selected_build_dir)
    selected_summary = json.loads((selected_root / "summary.json").read_text(encoding="utf-8"))
    selected_build = next(
        row for row in selected_summary["builds"] if row["public_mode"] == "input_stationary"
    )
    shutil.copytree(selected_root / "input_stationary", output / "input_stationary")
    if sha256(output / "input_stationary/graph.json") != stock_build["graph_sha256"]:
        raise RuntimeError("stock and selected graph JSON differ")
    selected = dict(selected_build)
    selected["source_build_dir"] = str(selected_root.resolve())
    summary = {
        "schema": "c3_vta_yolov3_tiny_stock_selected_build_v1",
        "status": "yolov3_tiny_stock_selected_cross_build_verified",
        "model": "yolov3-tiny",
        "input_shape": list(shape),
        "builds": [stock_build, selected],
        "assets": selected_summary["assets"],
        "board_contacted": False,
        "claim_boundary": (
            "Stock TopHub full graph versus the already selected Y00 input-stationary route; "
            "board correctness and latency require a clean-start run"
        ),
    }
    write_json(output / "summary.json", summary)
    (output / "command.txt").write_text(" ".join(__import__("sys").argv) + "\n", encoding="utf-8")
    write_json(output / "artifact_hashes.json", {"artifacts": {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }})
    print(json.dumps({
        "status": summary["status"], "input_shape": summary["input_shape"],
        "builds": [{key: row.get(key) for key in (
            "candidate_id", "public_mode", "build_seconds", "graph_sha256", "graphlib_sha256"
        )} for row in summary["builds"]],
    }, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-build-dir", required=True)
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--darknet-lib", required=True)
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
