#!/usr/bin/env python3
"""Frozen two-target C3 follow-up. Board stages are explicitly opt-in.

prepare is local-only; it freezes provenance, builds observed full fusions,
and compiles a diagnostic proxy. analyze never contacts the board.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import traceback
from types import SimpleNamespace

import c3_yolo224_incremental as p

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GROUPED = HERE / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
BASE = GROUPED / "20260926_c3_yolo224_graph_build_run02"
POOL = GROUPED / "20260927_c3_yolo224_remaining_full_fit_board_pool_freeze_run01"
CROSS = GROUPED / "20260927_c3_yolo224_remaining_full_fit_cross_compile_run01"
SDK = Path("/home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux")


def helper_build(output, sdk):
    binary = output / "c3_yolo224_graph_observer.so"
    command = ["bash", "-c", 'unset LD_LIBRARY_PATH; source "$1" >/dev/null && shift && exec $CXX "$@"',
               "c3-observer", str(sdk), "-std=c++17", "-shared", "-fPIC",
               str(HERE / "c3_yolo224_graph_observer.cc"), "-I" + str(ROOT / "include"),
               "-I" + str(ROOT / "3rdparty/dlpack/include"),
               "-I" + str(ROOT / "3rdparty/dmlc-core/include"),
               "-L" + str(ROOT / "build_axu_aarch64"), "-ltvm_runtime",
               "-Wl,-rpath,$ORIGIN", "-o", str(binary)]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    p.write_new(output / "observer_build.json", {"command": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr, "links_vta_runtime": False})
    if result.returncode:
        raise RuntimeError("observer compilation failed; see observer_build.json")


def observed_build(name, routes, program, params, output, env, compiler):
    import tvm
    from tvm import autotvm, relay, tir
    import vta
    from vta.top.residency_dispatch import ExplicitResidencyDispatch
    from build_vta_resnet50_generic_residency_tir_audit import FullTIRArchive
    from build_vta_c3_mode11_graph_context_contract import RelayFusionArchive
    from build_vta_c3_evaluation_v5_graphs import save_build
    from c3_reuse_semantics import extract_requests
    from c3_thread_sync_audit import final_tir_audit

    local = output / name
    local.mkdir()
    archive, relay_archive = FullTIRArchive(env), RelayFusionArchive()
    target = tvm.target.Target(env.target, host=env.target_host)
    relay.backend.te_compiler.get().clear()
    with autotvm.tophub.context(target), ExplicitResidencyDispatch(routes) as dispatch:
        with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"},
                              instruments=[archive, relay_archive]):
            graph_text, lib, lowered = relay.build(program, target=target, params=params)
    audit = dispatch.summary()
    if not audit["all_routes_scheduled"] or len(audit["schedule_hits"]) != len(routes):
        raise ValueError("incomplete exact routing")
    result = save_build(local, graph_text, lib, lowered, compiler,
                        {"deployment_variant": name, "routes": routes, "dispatch_audit": audit})
    graph = json.loads(graph_text)
    function_ids, contracts = {}, {}
    for i, snapshot in enumerate(relay_archive.snapshots):
        (local / ("relay_%03d.txt" % i)).write_text(snapshot["relay"])
    if not relay_archive.snapshots:
        raise ValueError("missing actual fused Relay")
    for snapshot in archive.snapshots:
        mod = tvm.ir.load_json(snapshot["json"])
        for func in mod.functions.values():
            if not isinstance(func, tir.PrimFunc) or func.attrs is None or "global_symbol" not in func.attrs:
                continue
            symbol = str(func.attrs["global_symbol"])
            encoded = tvm.ir.save_json(func)
            # Structural equality is checked below; text hashes are provenance, not a normalization proof.
            if symbol in function_ids:
                raise ValueError("ambiguous function capture " + symbol)
            function_ids[symbol] = p.digest(encoded)
            (local / (symbol + ".tir.json")).write_text(encoded)
            (local / (symbol + ".tir")).write_text(func.script())
            nodes = [n for n in p.TARGETS if graph["nodes"][n]["attrs"]["func_name"] == symbol]
            if nodes:
                single = tvm.IRModule({"main": func})
                semantics = extract_requests(single, env)
                address = final_tir_audit(single, semantics)
                node = nodes[0]
                abi = p.graph_inventory(graph)
                item = next(r for r in abi if r["node"] == node)
                if len(item["inputs"]) != 3 or len(item["outputs"]) != 1:
                    raise ValueError("expected four-argument fusion")
                fusion = p.fusion_semantics([x["relay"] for x in relay_archive.snapshots], item)
                contracts[str(node)] = {"abi": item, "function_tir_sha256": function_ids[symbol],
                    "semantics_source": symbol + ".tir.json", "request_semantics": semantics,
                    "address_audit": address, "fusion_semantics": fusion,
                    "fusion_semantics_status": fusion["status"],
                    "graphlib_sha256": result["graphlib_sha256"],
                    "fused_program_id": p.digest([result["graphlib_sha256"], symbol, function_ids[symbol]])}
    if set(contracts) != {str(n) for n in p.TARGETS}:
        raise ValueError("missing full fusion target")
    p.write_new(local / "functions.json", function_ids)
    p.write_new(local / "fusion_contracts.json", contracts)
    return result


def compile_graphs(args, output, contract, chosen):
    import tvm
    import vta
    from build_vta_c3_evaluation_v5_graphs import route_payload
    from build_vta_yolov3_tiny_relay_residency_dispatch_pair import make_relay_program, cross_compiler
    from run_vta_yolov3_tiny_relay_residency_dispatch_pair_board import semantic_params_hash

    assets = p.read(args.base / "summary.json")["assets"]
    if args.weights is not None:
        assets["weights"] = {**assets["weights"], "path": str(args.weights.resolve())}
    helper_build(output, args.sdk_environment)
    build_args = SimpleNamespace(**{key: assets[key]["path"] for key in ("cfg", "weights", "darknet_lib")})
    for key in ("cfg", "weights", "darknet_lib"):
        if not Path(assets[key]["path"]).is_file():
            raise FileNotFoundError("Frozen model asset missing: " + assets[key]["path"] +
                                    "; expected SHA-256 " + assets[key]["sha256"])
        if p.sha(assets[key]["path"]) != assets[key]["sha256"]:
            raise ValueError("model asset identity drift: " + key)
    env = vta.get_env()
    program, params, shape = make_relay_program(env, build_args)
    if list(shape) != [1, 3, 224, 224]:
        raise ValueError("not YOLO-224")
    old_routes = p.read(args.base / "selector_v4/build.json")["routes"]
    if len(old_routes) != 3 or {r["execution_task_id"] for r in old_routes} & {
            t["task_id"] for t in p.TARGETS.values()}:
        raise ValueError("historical baseline route drift")
    builds = []
    for name in ("G0", "G1", "G2"):
        routes = list(old_routes)
        if name != "G0":
            routes += [route_payload(chosen[str(node)][0 if name == "G1" else 1]) for node in p.TARGETS]
        builds.append(observed_build(name, routes, program, params, output, env, cross_compiler()))
    if len({r["graph_sha256"] for r in builds}) != 1 or builds[0]["graph_sha256"] != p.sha(args.base / "selector_v4/graph.json"):
        raise ValueError("graph JSON drift")
    params_hash = semantic_params_hash(args.base / "selector_v4/params.bin")[0]
    for name in ("G0", "G1", "G2"):
        if semantic_params_hash(output / name / "params.bin")[0] != params_hash:
            raise ValueError("parameter semantics drift")
    # A rebuild of the current baseline cannot silently become a new baseline.
    if builds[0]["graphlib_sha256"] != p.sha(args.base / "selector_v4/graphlib.so"):
        raise ValueError("G0 rebuild is not byte-identical to historical graphlib; audit before proceeding")
    graph = p.read(output / "G0/graph.json")
    targets = {graph["nodes"][n]["attrs"]["func_name"] for n in p.TARGETS}
    names = p.read(output / "G0/functions.json")
    for name in ("G1", "G2"):
        if set(p.read(output / name / "functions.json")) != set(names):
            raise ValueError("function inventory drift")
        for symbol in set(names) - targets:
            left = tvm.ir.load_json((output / "G0" / (symbol + ".tir.json")).read_text())
            right = tvm.ir.load_json((output / name / (symbol + ".tir.json")).read_text())
            if not tvm.ir.structural_equal(left, right, map_free_vars=True):
                raise ValueError("non-target implementation changed: " + symbol)
    if not all(r["memory_contract"]["fits"] for r in builds):
        raise ValueError("192 MiB device memory contract exceeded")
    shutil.copytree(args.base / "stock", output / "stock")
    p.write_new(output / "build_summary.json", {"status": "LOCAL_BUILDS_READY_BOARD_NOT_RUN",
        "builds": builds, "graph_identity": builds[0]["graph_sha256"],
        "non_target_tir_structurally_equal": True, "historical_g0_binary_identical": True,
        "board_contacted": False})


def prepare(args):
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    chosen = p.bind_candidates([json.loads(line) for line in
                               (args.pool / "candidates.jsonl").read_text().splitlines() if line.strip()])
    ledgers = {"base": p.verify(args.base), "pool": p.verify(args.pool), "cross": p.verify(args.cross)}
    binaries = {}
    for pair in chosen.values():
        for row in pair:
            binary = args.cross / "binaries" / (row["candidate_id"] + ".so")
            binaries[row["program_id"]] = {"path": str(binary.resolve()), "sha256": p.sha(binary)}
    output.mkdir(parents=True)
    contract = {**p.protocol(), "sources": {"base": str(args.base.resolve()), "pool": str(args.pool.resolve()),
                 "cross": str(args.cross.resolve())}, "source_ledgers": ledgers,
                "isolated_binaries": binaries, "sources_exposed_development_data": True,
                "source_hashes": {name: p.sha(HERE / name) for name in (
                    Path(__file__).name, "c3_yolo224_incremental.py", "c3_yolo224_graph_observer.cc",
                    "c3_yolo224_incremental_board.py")}}
    p.write_new(output / "contract.json", contract)
    p.write_new(output / "selected_candidates.json", chosen)
    p.write_new(output / "graph_inventory.json", p.graph_inventory(p.read(args.base / "selector_v4/graph.json")))
    p.write_new(output / "interpretation_correction.json", {
        "supersedes_claim_not_raw_data": str(GROUPED / "20260927_c3_yolo224_remaining_full_fit_analysis_run01/REPORT.md"),
        "isolated_savings_ms": 3.330978, "graph_projection_valid": False,
        "upper_bound_0_78_percent_valid": False, "full_residency_exhausted_established": False,
        "reason": "Isolated three-argument Input controls have not been equated with actual full-fusion graph implementations.",
        "historical_data_modified": False})
    try:
        if args.freeze_only:
            p.write_new(output / "status.json", {"status": "FROZEN_NOT_COMPILED", "board_contacted": False})
        else:
            compile_graphs(args, output, contract, chosen)
            p.write_new(output / "status.json", {"status": "PREPARED_BOARD_PAUSED", "board_contacted": False})
    except BaseException as error:
        p.write_new(output / "status.json", {"status": "LOCAL_PREPARATION_FAILED",
                    "board_contacted": False, "message": str(error)})
        p.write_new(output / "local_failure.json", {"status": "LOCAL_PREPARATION_FAILED",
            "message": str(error), "traceback": traceback.format_exc(), "board_contacted": False})
        raise
    finally:
        p.seal(output)
    print(json.dumps(p.read(output / "status.json"), ensure_ascii=False))


def analyze(args):
    first = p.read(args.boot1 / "summary.json")
    p.verify(args.boot1)
    if (args.boot1 / "invalid_session.json").exists() or first.get("status") != "GRAPH_COMPLETE":
        raise ValueError("invalid first boot")
    rows = [json.loads(line) for line in (args.boot1 / "calls.jsonl").read_text().splitlines()]
    if first["evaluation"] != p.evaluation(rows, 1):
        raise ValueError("summary disagrees with raw calls")
    result = {"boot1": first, "cross_boot": None, "paused_after_experiment": True}
    if args.boot2:
        p.verify(args.boot2)
        second = p.read(args.boot2 / "summary.json")
        if (args.boot2 / "invalid_session.json").exists():
            raise ValueError("invalid second boot")
        rows = [json.loads(line) for line in (args.boot2 / "calls.jsonl").read_text().splitlines()]
        if second["evaluation"] != p.evaluation(rows, 2):
            raise ValueError("second summary disagrees with raw calls")
        result.update(boot2=second, cross_boot=p.cross_boot(first, second))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    p.write_new(args.output_dir / "analysis.json", result)
    lines = ["# YOLO-224 新增驻留整图结果", "", "FPS 为单图等效 FPS，不是图片流水吞吐。", "",
             "| 启动 | 版本 | 中位时延 ms | 单图等效 FPS | 相对 G0 节省 ms |", "|---|---|---:|---:|---:|"]
    for key in ("boot1", "boot2"):
        if key not in result:
            continue
        stats = result[key]["evaluation"]["statistics"]
        for variant, row in stats.items():
            lines.append(f"| {key} | {variant} | {row['median_ms']:.3f} | {row['equivalent_fps']:.3f} | "
                         f"{stats['G0']['median_ms'] - row['median_ms']:.3f} |")
    lines.extend(["", "独立确认结论：" + json.dumps(result["cross_boot"], ensure_ascii=False),
                  "", "本轮完成后暂停，不自动换候选或开启部分驻留。"])
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    p.seal(args.output_dir)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--base", type=Path, default=BASE)
    prep.add_argument("--pool", type=Path, default=POOL)
    prep.add_argument("--cross", type=Path, default=CROSS)
    prep.add_argument("--sdk-environment", type=Path, default=SDK)
    prep.add_argument("--weights", type=Path, help="relocated original weights; frozen SHA-256 must match")
    prep.add_argument("--freeze-only", action="store_true")
    prep.add_argument("--output-dir", type=Path, required=True)
    for name in ("profile", "qualify", "graph"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--prepared", type=Path, required=True)
        cmd.add_argument("--enable-board", action="store_true", help="explicitly resume this board stage")
        cmd.add_argument("--runtime-contract", type=Path, required=True)
        cmd.add_argument("--helper-build-dir", type=Path, required=True)
        cmd.add_argument("--completion-dir", type=Path, required=True)
        cmd.add_argument("--qualification", type=Path)
        cmd.add_argument("--profile-result", type=Path)
        cmd.add_argument("--boot-index", type=int, choices=(1, 2), default=1)
        cmd.add_argument("--boot1", type=Path)
        cmd.add_argument("--host", default="192.168.1.247")
        cmd.add_argument("--port", type=int, default=9090)
        cmd.add_argument("--session-timeout", type=int, default=1800)
        cmd.add_argument("--known-hosts", default="/home/orange/.ssh/known_hosts")
        cmd.add_argument("--output-dir", type=Path, required=True)
    ana = sub.add_parser("analyze")
    ana.add_argument("--boot1", type=Path, required=True)
    ana.add_argument("--boot2", type=Path)
    ana.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "analyze":
        analyze(args)
    else:
        if not args.enable_board:
            raise SystemExit("BOARD_PAUSED: use --enable-board only after explicitly resuming board work")
        from c3_yolo224_incremental_board import run
        run(args)


if __name__ == "__main__":
    main()
