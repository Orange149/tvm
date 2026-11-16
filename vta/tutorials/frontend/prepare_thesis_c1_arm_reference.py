#!/usr/bin/env python3
"""Qualify independent AArch64 full-graph and full-frontier CPU references."""

import argparse
import copy
from pathlib import Path
import shlex
import sys

from prepare_thesis_c1_memory_inputs import reference
from prepare_thesis_c1_multisland_qualification import BASE, HISTORY
from run_thesis_c1_memory_ablation import Collector, session_sources
from thesis_c1_memory_ablation import digest, read_json, seal, verify_ref, write_new
from thesis_c1_memory_evidence import _output_path, inputs_manifest, jsonl
from thesis_supplement_run import RunStore


def compile_worker(directory):
    import tvm
    from tvm import autotvm, relay
    import vta
    from thesis_canonical_graph_split import UnitGraph
    from deploy_classification_stage_pipeline_native import export_stage_lib
    store = RunStore(directory)
    graph = UnitGraph(tvm.ir.load_json(Path(store.contract["canonical_qir"]["path"]).read_text()))
    edges = sorted({n for b in range(22) for n in graph.frontier(b)}, key=lambda n: graph.ids[n])
    env = vta.get_env()
    target = tvm.target.Target(env.target_vta_cpu, host=env.target_host)
    modules = []
    for name in ("complete_graph", "complete_frontier_probe"):
        directory = store.directory / "raw" / name
        directory.mkdir()
        body = graph.mod["main"].body if name == "complete_graph" else relay.Tuple(edges)
        mod = relay.transform.InferType()(tvm.IRModule.from_expr(relay.Function(list(graph.mod["main"].params), body)))
        (directory / "qir.json").write_text(tvm.ir.save_json(mod))
        with autotvm.apply_history_best([store.contract["history"]["path"]]), vta.build_config(
                opt_level=3, disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}):
            factory = relay.build(mod, target=target)
        (directory / "graph.json").write_text(factory.get_graph_json())
        (directory / "params.bin").write_bytes(relay.save_param_dict(factory.get_params()))
        export_stage_lib(factory.get_lib(), directory / "module.so")
        modules.append({"name": name, "begin": 0, "end": 21, "device": "cpu", "cpu_affinity": [0, 1, 2, 3],
                        "input_names": [p.name_hint for p in mod["main"].params],
                        "output_node_ids": [graph.ids[graph.mod["main"].body]] if name == "complete_graph" else [graph.ids[n] for n in edges],
                        **{key: reference(directory / filename) for key, filename in
                           (("graph", "graph.json"), ("lib", "module.so"), ("params", "params.bin"))}})
        print(name + ": independent same-qIR CPU reference compiled", flush=True)
    write_new(store.directory / "modules.json", modules)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inputs-manifest", type=Path, default=BASE / "20261017_common_inputs01/inputs_manifest.json")
    parser.add_argument("--runner-build", type=Path, default=BASE / "20261017_native_build02/build_manifest.json")
    parser.add_argument("--host", default="root@192.168.1.247")
    parser.add_argument("--sdk", type=Path, default=Path("/home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux"))
    parser.add_argument("--compiled-from", type=Path, help="reuse hash-verified CPU reference modules, never prior outputs")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return compile_worker(args.output_dir)
    build = read_json(args.runner_build)
    old = read_json(args.inputs_manifest)
    from prepare_thesis_c1_memory_inputs import CANONICAL
    artifacts = {key: verify_ref(build[key]) for key in ("runner", "runtime", "driver")}
    artifacts.update(canonical_qir=reference(CANONICAL), inputs=reference(args.inputs_manifest),
                     bitstream=reference(Path(__file__).resolve().parents[3] / "apps/vta_rpc/firmware/vta_hpc.bit"))
    records = inputs_manifest({"artifacts": artifacts})
    store = RunStore.create(args.output_dir, {"run_id": args.output_dir.name, "stage": "C1_ARM_common_reference",
                "campaign_id": "c1_memory_ablation", "canonical_qir": artifacts["canonical_qir"], "history": reference(HISTORY),
                "old_inputs_manifest": reference(args.inputs_manifest), "runner_build": reference(args.runner_build),
                "source": reference(__file__), "extra_quantize_calls": 0,
                "compiled_from": reference(args.compiled_from / "modules.json") if args.compiled_from else None,
                "scope": "independent complete CPU graph and complete boundary probe; no split-program outputs used as reference"})
    script = store.directory / "raw/build.sh"
    script.write_text("set -e\nunset LD_LIBRARY_PATH\nsource " + shlex.quote(str(args.sdk)) + "\nexec " +
                      shlex.join([sys.executable, str(Path(__file__).resolve()), "--worker", "--output-dir", str(store.directory)]) + "\n")
    if args.compiled_from:
        prior = RunStore(args.compiled_from)
        if any(prior.contract[key]["sha256"] != store.contract[key]["sha256"] for key in ("canonical_qir", "history")):
            raise ValueError("prior reference compiler inputs differ")
        if not any(e["event"] == "action_end" and e.get("action") == "compile_ARM_common_references" and e["status"] == "ok"
                   for e in prior.events()):
            raise ValueError("prior reference modules have no successful compilation record")
        modules = read_json(args.compiled_from / "modules.json")
        for stage in modules:
            for key in ("graph", "lib", "params"):
                verify_ref(stage[key])
        write_new(store.directory / "modules.json", modules)
    else:
        action = store.execute(["bash", str(script)], "compile_ARM_common_references", "host_compilation", 1800)
        if action["status"] != "ok":
            raise RuntimeError("ARM reference compilation failed: " + action["stderr"])
    support = []
    for stage in read_json(store.directory / "modules.json"):
        identity = {"stages": [stage], "thread_vector": [4], "policy": "ordinary_copy", "shared_plan": None}
        support.append({"configuration_id": digest(identity), "identity": identity, "policy": "ordinary_copy"})
    contract = seal({"schema": "c1_ARM_common_reference_v1", "artifacts": artifacts, "support": support,
                     "queue_capacity_bytes": {"instruction": 2 << 20, "uop": 2 << 20}, "performance_windows": 0})
    write_new(store.directory / "reference_contract.json", contract)
    collector = Collector(contract, store, args.host, 0, records)
    boot = collector.identity()
    # Prior probe deployment archives have identical local copies. Remove only
    # those owned transport copies to leave RAM for all-frontier raw dumps.
    owned = []
    for run in ("20261017_multisland_board_preflight01", "20261017_multisland_board_preflight02", "20261017_multisland_board_preflight03"):
        root = BASE / run
        for event in jsonl(root / "timeline.jsonl"):
            if event["event"] == "action_start" and event.get("action") == "upload_bundle":
                if reference(root / "raw/deployment.tar")["sha256"] != event["stdin_sha256"]:
                    raise ValueError("prior deployment local archive drift")
                remote_file = shlex.split(event["argv"][-1])[-1]
                if not remote_file.startswith("/var/volatile/c1_memory_") or not remote_file.endswith("/deployment.tar"):
                    raise ValueError("not an owned prior transport archive")
                owned.append(shlex.quote(remote_file))
    if owned:
        collector.execute("rm -f " + " ".join(owned), "release_owned_transport_archives", "preparation")
    collector.deploy()
    sessions = {}
    for config in support:
        name = config["identity"]["stages"][0]["name"]
        print("collecting independent ARM reference: " + name, flush=True)
        directory, _ = collector.session(config, "reference_run_" + name, True)
        sessions[name] = directory
    import numpy as np
    full = jsonl(sessions["complete_graph"] / "result.jsonl")
    probe = jsonl(sessions["complete_frontier_probe"] / "result.jsonl")
    if len(full) != 67 or len(probe) != 67:
        raise ValueError("incomplete independent reference frames")
    mapping = support[1]["identity"]["stages"][0]["output_node_ids"]
    new_records = []
    for index, (left, right, source) in enumerate(zip(full, probe, records)):
        if any(row.get("frame_id") != index or row.get("input_index") != index for row in (left, right)):
            raise ValueError("reference frame/input identity drift")
        def values(directory, item):
            return np.fromfile(_output_path(directory, item["path"]), dtype=item["dtype"]).reshape(item["shape"])
        output = values(sessions["complete_graph"], left["raw_output_files"][0])
        if len(right["raw_output_files"]) != len(mapping):
            raise ValueError("missing common probe frontiers")
        arrays = {node: values(sessions["complete_frontier_probe"], item) for node, item in zip(mapping, right["raw_output_files"])}
        final_node = support[0]["identity"]["stages"][0]["output_node_ids"][0]
        if output.tobytes() != arrays[final_node].tobytes() or any(not np.isfinite(x).all() for x in arrays.values()):
            raise ValueError("independent ARM complete graph/probe mismatch")
        row = copy.deepcopy(source)
        target = store.directory / "raw/references" / ("input_{:02}".format(index))
        target.mkdir(parents=True)
        row["frontiers"] = {}
        for node, value in arrays.items():
            path = target / ("node_" + str(node) + ".npy")
            np.save(path, value, allow_pickle=False)
            row["frontiers"][str(node)] = reference(path)
        new_records.append(row)
    manifest = seal({**{k: v for k, v in old.items() if k not in ("artifact_sha256", "real_inputs", "diagnostic_inputs", "reference_artifacts", "board_calls")},
                     "real_inputs": new_records[:64], "diagnostic_inputs": new_records[64:],
                     "reference_artifacts": [reference(store.directory / "contract.json"), reference(store.directory / "modules.json"),
                           *[stage[key] for config in support for stage in config["identity"]["stages"] for key in ("graph", "lib", "params")]],
                     "reference_execution_target": "AArch64 CPU LLVM; canonical full graph and independent complete frontier probe",
                     "prior_x86_reference_manifest": reference(args.inputs_manifest), "board_calls": 134, "reference_boot_id": boot,
                     "inputs_changed": False, "extra_quantize_calls": 0})
    write_new(store.directory / "inputs_manifest.json", manifest)
    write_new(store.directory / "reference_proof.json", seal({"status": "ARM_COMPLETE_GRAPH_PROBE_BYTE_EXACT_67",
              "boot_id": boot, "sessions": {name: session_sources(path) for name, path in sessions.items()},
              "inputs_manifest": reference(store.directory / "inputs_manifest.json"), "new_performance_windows": 0}))
    print(store.directory / "inputs_manifest.json")


if __name__ == "__main__":
    main()
