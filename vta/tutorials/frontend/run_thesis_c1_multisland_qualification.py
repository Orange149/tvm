#!/usr/bin/env python3
"""Run fixed 2/3-island completion probes; collect no ranking performance labels."""

import argparse
import copy
from pathlib import Path
import subprocess

from prepare_thesis_c1_memory_inputs import reference
from prepare_thesis_c1_multisland_qualification import BASE
from thesis_c1_memory_ablation import digest, read_json, seal, verify_ref, verify_seal, write_new
from thesis_c1_memory_evidence import inputs_manifest, validate_qualification
from thesis_supplement_run import RunStore
from run_thesis_c1_memory_ablation import Collector, session_sources


def assemble(components, build, inputs, directory):
    import numpy as np
    import tvm
    from audit_shared_buffer_storage import classify, finalize_bundles, graph_entries
    verify_seal(components)
    verify_seal(build)
    if b"--process-timeout-s" not in Path(verify_ref(build["runner"])["path"]).read_bytes():
        raise ValueError("runner lacks the native process deadline; rebuild it first")
    artifacts = {key: build[key] for key in ("runner", "runtime", "driver")}
    artifacts.update(canonical_qir=components["canonical_qir"], inputs=reference(inputs),
                     bitstream=reference(Path(__file__).resolve().parents[3] / "apps/vta_rpc/firmware/vta_hpc.bit"))
    records = inputs_manifest({"artifacts": artifacts})
    references = {node: np.load(ref["path"], allow_pickle=False) for node, ref in records[0]["frontiers"].items()}
    lookup = {(r["begin"], r["end"], r["device"]): r for r in components["components"]}
    support = []
    for name, ranges in components["probes"].items():
        stages, edges = [], []
        for begin, end, device in ranges:
            component = copy.deepcopy(lookup[(begin, end, device)])
            for field in ("graph", "lib", "params", "final_tir"):
                verify_ref(component[field])
            for ref in component["lowered_modules"]:
                verify_ref(ref)
            component["cpu_affinity"] = [0, 1, 2, 3] if device == "cpu" else []
            stages.append(component)
        for index, (producer, consumer) in enumerate(zip(stages, stages[1:])):
            if producer["output_node_ids"] != consumer["input_node_ids"]:
                raise ValueError("probe boundary does not preserve the complete live frontier")
            entries = graph_entries(read_json(consumer["graph"]["path"]), 12 if consumer["device"] == "vta" else 1)
            parameters = set(tvm.runtime.load_param_dict(Path(consumer["params"]["path"]).read_bytes()))
            symbols = subprocess.run(["readelf", "-Ws", consumer["lib"]["path"]],
                                     capture_output=True, text=True, check=True).stdout
            linked = "present" if "__tvm_lookup_linked_param" in symbols else "absent" if ".symtab" in symbols else "unknown"
            direction = producer["device"] + "_to_" + consumer["device"]
            tensors = []
            for i, (input_name, node) in enumerate(zip(consumer["input_names"], consumer["input_node_ids"])):
                value = references[str(node)]
                tensor = classify(entries, input_name, parameters, linked,
                                  {"shape": list(value.shape), "dtype": str(value.dtype)}, direction)
                tensor.update(tensor_index=i, contract_bytes=value.nbytes)
                tensors.append(tensor)
            edges.append({"producer_stage": index, "consumer_stage": index + 1,
                          "direction": direction, "tensors": tensors})
        finalize_bundles(edges)
        plan = {"schema": "shared-buffer-plan-v1", "stages": [{"device": s["device"], "input_names": s["input_names"],
                    **{field + "_sha256": s[field]["sha256"] for field in ("graph", "lib", "params")}} for s in stages],
                "edges": edges, "runtime_sha256": artifacts["runtime"]["sha256"], "driver_sha256": artifacts["driver"]["sha256"],
                "predicted_saved_bytes": sum(e["predicted_saved_bytes"] for e in edges)}
        target = directory / "raw" / (name + "_shared_plan.json")
        write_new(target, plan)
        materialized = sum(references[str(node)].nbytes for stage in stages[1:] for node in stage["input_node_ids"])
        materialized += sum(references[str(node)].nbytes for stage in stages[:-1]
                            for node, live in zip(stage["output_node_ids"], stage["output_live_input_indices"]) if live < 0)
        identity = {"topology_id": name, "stages": stages, "artifacts": artifacts,
                    "tile_history": components["history"], "thread_vector": [4 if s["device"] == "cpu" else 1 for s in stages]}
        pair_id = digest(identity)
        for policy in ("ordinary_copy", "pool_anchor_k2_shared" if any(e["source"] == "pool-anchor" for e in edges) else "external_k2_shared"):
            obj = {**identity, "policy": policy, "shared_plan": None if policy == "ordinary_copy" else reference(target)}
            support.append({"configuration_id": digest(obj), "pair_id": pair_id, "identity": obj, "policy": policy,
                            "topology_id": name, "vta_islands": sum(s["device"] == "vta" for s in stages),
                            "framework_materialization_bytes": materialized if policy == "ordinary_copy" else 0,
                            "additional_slot_bytes": 0 if policy == "ordinary_copy" else
                                sum(e["external_k2_bytes"] for e in edges) - plan["predicted_saved_bytes"],
                            "vta_dma_calls": sum(s["vta_dma_calls"] for s in stages)})
    # Qualify fewer worker queues first. Policy expansion remains paired.
    support.sort(key=lambda r: (r["vta_islands"], r["policy"] != "ordinary_copy"))
    return seal({"schema": "c1_multisland_completion_preflight_v1", "support": support, "artifacts": artifacts,
                 "queue_capacity_bytes": {"instruction": 2 << 20, "uop": 2 << 20},
                 "scope": "predeclared completion probes, not M1-selected ablation support", "performance_windows": 0}), records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--runner-build", type=Path, required=True)
    parser.add_argument("--inputs-manifest", type=Path, default=BASE / "20261017_common_inputs01/inputs_manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="root@192.168.1.247")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    store = RunStore.create(args.output_dir, {"run_id": args.output_dir.name, "stage": "C1_multisland_completion_preflight",
            "campaign_id": "c1_memory_ablation", "components": reference(args.components), "build": reference(args.runner_build),
            "inputs": reference(args.inputs_manifest), "controller": reference(__file__), "board_host": args.host,
            "performance_windows": 0, "scope": "single-boot prerequisite only; not the 16-configuration experiment"})
    state = {"status": "preparing", "performance_windows": 0, "ranking_pool": False, "sessions": [], "boot_id": None}
    try:
        contract, records = assemble(read_json(args.components), read_json(args.runner_build), args.inputs_manifest, store.directory)
        write_new(store.directory / "probe_contract.json", contract)
        if args.prepare_only:
            state["status"] = "local_probe_contract_ready"
            return
        collector = Collector(contract, store, args.host, 0, records)
        state["boot_id"] = collector.identity()
        collector.deploy()
        pairs = {}
        for config in contract["support"]:
            token = "prequal_" + config["configuration_id"]
            print("running {} {}".format(config["topology_id"], config["policy"]), flush=True)
            row = {"configuration_id": config["configuration_id"], "session": token, "status": "attempting"}
            state["sessions"].append(row)
            try:
                directory, _ = collector.session(config, token, True)
            except BaseException:
                directory = store.directory / "raw" / token
                row.update(status="native_failed", files=session_sources(directory) if directory.exists() else {})
                raise
            row.update(files=session_sources(directory), status="collected_pending_audit")
            proof = validate_qualification(contract, config, directory, records)
            if config["pair_id"] in pairs and pairs[config["pair_id"]] != proof["final_output_sha256"]:
                raise ValueError("same-program copy/shared outputs are not byte exact")
            pairs[config["pair_id"]] = proof["final_output_sha256"]
            row.update(status="qualified_this_boot_only", proof=proof)
            print("qualified {} {}".format(config["topology_id"], config["policy"]), flush=True)
        state["status"] = "four_probe_configurations_qualified_on_one_boot"
    except BaseException as error:
        state.update(status="prerequisite_failed", failure=str(error))
        raise
    finally:
        state["cost_actions"] = [e for e in store.events() if e["event"] == "action_end"]
        write_new(store.directory / "preflight.json", seal(state))


if __name__ == "__main__":
    main()
