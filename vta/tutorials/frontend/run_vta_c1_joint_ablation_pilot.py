#!/usr/bin/env python3
"""Fresh single-boot pilot for the frozen joint-resource ablation choices.

All programs are cuts of one quantized DAG. Qualify full frontiers before timing.
This pilot does not replace the separate multi-boot memory-policy campaign.
"""

import argparse
import copy
import json
from pathlib import Path
import shlex
import shutil
import statistics
import tarfile
import uuid

from thesis_supplement_run import RunStore, sha256, write_new_json


HERE = Path(__file__).resolve().parent
REPORT = HERE / "report_out/stage_tile_cotuning"
SCORES = REPORT / "c1_static_score_comparison"
PRIOR = REPORT / "thesis_supplement_experiments/results/06/20260916T124859Z_single_image_audit_bfec496d"
INPUTS = REPORT / "c1_memory_ablation/20261017_common_inputs01/inputs_manifest.json"
BUILD = SCORES / "20261017_joint_pilot_runner_run01/build_manifest.json"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=yes",
       "-o", "HostKeyAlgorithms=+ssh-rsa", "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa", "root@192.168.1.247"]


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256(path)}


def verify(item):
    if sha256(item["path"]) != item["sha256"]:
        raise ValueError("artifact drift: " + item["path"])
    return Path(item["path"])


def create(output, prepared_run=None):
    from solve_cpu_vta_pipeline_v1_p3 import candidate_id, validate_artifact_sha256
    inputs, old, build = read(INPUTS), read(PRIOR / "contract.json"), read(BUILD)
    if b"--process-timeout-s" not in verify(build["runner"]).read_bytes():
        raise ValueError("runner does not support the frozen process deadline")
    sources = [INPUTS, BUILD, Path(__file__), PRIOR / "contract.json"]
    if prepared_run:
        sources += [prepared_run / "contract.json", prepared_run / "analysis/program_manifest.json"]
    selected, groups = {}, {}
    for name, cell in (("paper_style_baseline", "B"), ("without_cpu_pool", "C"),
                       ("without_boundary", "D"), ("full", "F"), ("without_internal_features", "F")):
        path = SCORES / "20261017_cpu_transport_ablation_run01" / (cell + ".json")
        payload = read(path)
        validate_artifact_sha256(payload)
        row = payload["ranked_candidates"][0]
        label = "three_islands" if cell in ("B", "C") else "two_islands" if cell == "D" else "full_one_island"
        selected.setdefault(label, {"candidate_id": row["candidate_id"],
                                   "path": [(s["segment_id"], s["threads"]) for s in row["stages"]]})
        groups[name] = {"selected_program": label, "scoring_cell": cell,
                        "predicted_ii_ms": row["predicted_ii_ms"]}
        sources.append(path)
    for end in (15, 16):
        path = (("cpu:00:02", 4), ("vta:03:{:02}".format(end), 1), ("cpu:{:02}:20".format(end + 1), 2))
        selected["endpoint_" + str(end)] = {"candidate_id": candidate_id(path), "path": path}
    # Bind the full internal rule and its remove-internal comparison explicitly.
    internal_path = SCORES / "20261017_internal_dma_ablation_run01/orders_frozen.json"
    internal = read(internal_path)
    for key in ("deployed_full", "deployed_off"):
        assert internal["current_orders"][key]["top5"][0]["candidate_id"] == selected["full_one_island"]["candidate_id"]
    sources.append(internal_path)
    records = [inputs["real_inputs"][0], inputs["diagnostic_inputs"][0], inputs["diagnostic_inputs"][2]]
    programs = sorted(selected)
    contract = {
        "run_id": output.name, "stage": "C1_JOINT_RESOURCE_PILOT", "campaign_id": "c1_joint_resource_ablation_pilot",
        "scope": "single_boot_three_round_engineering_pilot", "groups": groups, "programs": selected,
        "performance_orders": [programs[i:] + programs[:i] for i in (0, 2, 4)],
        "performance": {"runs": 307, "warmup_rows": 5, "discard_window_prefix": 2, "scored_gaps": 299},
        "qualification": {"runs": 6, "modes": ["serial", "pipeline"], "inputs": "one real image, zeros, normal repeated twice",
                          "full_stage_frontiers": True, "timing_after_all_programs_qualified": True},
        "input_records": records, "canonical": ref(old["canonical_source"]),
        "history": ref(old["baseline_history"]), "cached_components": ref(PRIOR / "raw/c1_components_int8/component_manifest.json"),
        "runner": build["runner"], "runtime": build["runtime"], "driver": build["driver"],
        "bitstream": ref(HERE.parents[2] / "apps/vta_rpc/firmware/vta_hpc.bit"),
        "remote": "/var/volatile/c1_joint_" + uuid.uuid4().hex[:12],
        "numeric": {"integer": "exact", "float_atol": 1e-5, "float_rtol": 1e-4},
        "source_refs": [ref(p) for p in sources], "extra_quantize_calls": 0,
        "candidate_board_timings_consumed_for_selection": False,
        "prepared_run": str(prepared_run.resolve()) if prepared_run else None,
        "limitations": ["Five programs, one kernel boot; rounds are not independent boots.",
                        "Top-1 comparison only; not a full-domain Top-5 online search study.",
                        "Full and remove-internal rules select the same Top-5; no independent internal-feature performance contrast.",
                        "Paper-style maximum-load scoring adaptation, not full original algorithm reproduction.",
                        "Original component cost fits are reused across canonical builds; fresh timing tests the resulting choices."],
    }
    if inputs["canonical_qir_sha256"] != contract["canonical"]["sha256"]:
        raise ValueError("common references use different graph")
    return RunStore.create(output, contract)


def reuse_programs(store, prepared_run):
    source = RunStore(prepared_run)
    for field in ("canonical", "history"):
        if source.contract[field] != store.contract[field]:
            raise ValueError("prepared graph/history mismatch")
    if source.contract["programs"] != store.contract["programs"]:
        raise ValueError("prepared candidate selections differ")
    manifest = copy.deepcopy(read(prepared_run / "analysis/program_manifest.json"))
    for name, row in manifest["programs"].items():
        destination = store.directory / "raw/programs" / name
        destination.mkdir(parents=True)
        for filename, item in row["files"].items():
            shutil.copyfile(verify(item), destination / filename)
        row["files"] = {p.name: ref(p) for p in destination.iterdir()}
        row["prepared_source_manifest"] = ref(prepared_run / "analysis/program_manifest.json")
    write_new_json(store.directory / "analysis/program_manifest.json", manifest)
    print("Reused all exact, host-prepared canonical components", flush=True)
    return manifest


def build_programs(store):
    import tvm
    from tvm import autotvm, relay
    import vta
    from thesis_canonical_graph_split import UnitGraph
    from thesis_single_image_c1_prepare import packed_stage
    from deploy_classification_stage_pipeline_native import export_stage_lib

    contract, root = store.contract, store.directory
    graph = UnitGraph(tvm.ir.load_json(verify(contract["canonical"]).read_text()))
    history = verify(contract["history"])
    cache_path = verify(contract["cached_components"])
    cache = read(cache_path)
    if cache["canonical_qir_sha256"] != contract["canonical"]["sha256"] or cache["tune_log_sha256"] != contract["history"]["sha256"]:
        raise ValueError("cached programs graph or Tile history mismatch")
    cached = {r["name"]: r for r in cache["programs"]}
    specs = set()
    for program in contract["programs"].values():
        ranges = []
        for sid, _ in program["path"]:
            device, begin, end = sid.split(":")
            specs.add((int(begin), int(end) + 1, device))
            ranges.append((int(begin), int(end) + 1))
        graph.recompose(ranges)
    base = root / "raw/programs"
    base.mkdir()
    env = vta.get_env()
    cpu = tvm.target.Target(env.target_vta_cpu, host=env.target_host)
    ext = tvm.target.Target(env.target, host=env.target_host)
    rows = {}
    for begin, end, device in sorted(specs):
        name = "{}_{:02}_{:02}".format(device, begin, end - 1)
        directory = base / name
        directory.mkdir()
        original, incoming, outgoing = graph.stage(begin, end)
        row = {"name": name, "device": device, "begin": begin, "end_exclusive": end,
               "input_names": [p.name_hint for p in original["main"].params],
               "input_node_ids": [graph.ids[n] for n in incoming], "output_node_ids": [graph.ids[n] for n in outgoing]}
        if name in cached:
            source = cache_path.parent / name
            for filename, item in cached[name]["files"].items():
                if sha256(source / filename) != item["sha256"]:
                    raise ValueError("cached component changed")
                shutil.copyfile(source / filename, directory / filename)
            if row["input_node_ids"] != cached[name]["input_node_ids"] or row["output_node_ids"] != cached[name]["output_node_ids"]:
                raise ValueError("cached frontier differs")
            row["build_source"] = "exact_graph_history_cache"
        else:
            mod = packed_stage(original, env) if device == "vta" else original
            (directory / "lowering_qir.json").write_text(tvm.ir.save_json(mod))
            with autotvm.apply_history_best([str(history)]), vta.build_config(
                opt_level=3, disabled_pass={"AlterOpLayout", "tir.CommonSubexprElimTIR"}):
                factory = relay.build(mod, target=ext if device == "vta" else cpu)
            (directory / "graph.json").write_text(factory.get_graph_json())
            (directory / "params.bin").write_bytes(relay.save_param_dict(factory.get_params()))
            export_stage_lib(factory.get_lib(), directory / "module.so")
            row["build_source"] = "new_canonical_cut"
        row["files"] = {p.name: ref(p) for p in directory.iterdir() if p.is_file()}
        rows[name] = row
        print(name + ": " + row["build_source"], flush=True)
    manifest = {"canonical_sha256": contract["canonical"]["sha256"], "history_sha256": contract["history"]["sha256"],
                "recomposition_exact": True, "programs": rows, "final_node_id": graph.ids[graph.mod["main"].body]}
    write_new_json(root / "analysis/program_manifest.json", manifest)
    return manifest


def execute_script(store, script, name, timeout=60, check=True):
    path = store.directory / "raw" / (name + ".sh")
    with path.open("x") as stream:
        stream.write("set -eu\n" + script + "\n")
    action = store.execute(SSH + ["sh -s"], name, "board_qualification_or_measurement", timeout, path)
    if check and action["status"] != "ok":
        raise RuntimeError(name + ": " + action["status"] + "; see " + action["stderr"])
    return action


def deploy(store, manifest):
    root, c = store.directory, store.contract
    stage = root / "raw/deployment"
    stage.mkdir()
    for name, key in (("runner", "runner"), ("libvta.so", "driver"), ("libtvm_runtime.so", "runtime"), ("vta_hpc.bit", "bitstream")):
        shutil.copyfile(verify(c[key]), stage / name)
    # Lowering IR stays on the host; the board needs only executor artifacts.
    for name in manifest["programs"]:
        target = stage / "programs" / name
        target.mkdir(parents=True)
        for filename in ("graph.json", "module.so", "params.bin"):
            shutil.copyfile(root / "raw/programs" / name / filename, target / filename)
    (stage / "inputs").mkdir()
    for i, record in enumerate(c["input_records"]):
        shutil.copyfile(verify(record["input"]), stage / "inputs" / (str(i) + ".bin"))
        for item in record["frontiers"].values():
            verify(item)
    (stage / "inputs.txt").write_text("\n".join(c["remote"] + "/inputs/" + str(i) + ".bin" for i in range(3)) + "\n")
    bundle = root / "raw/deployment.tar"
    with tarfile.open(bundle, "w") as tar:
        for path in stage.iterdir():
            tar.add(path, arcname=path.name)
    identity = execute_script(store, "cat /proc/sys/kernel/random/boot_id", "boot_identity")
    boot = (root / identity["stdout"]).read_text().strip()
    uuid.UUID(boot)
    receipt = execute_script(store, "mkdir " + shlex.quote(c["remote"]), "create_remote_directory")
    action = store.execute(SSH + ["cat > " + shlex.quote(c["remote"] + "/deployment.tar")],
                           "upload_deployment", "deployment", 120, bundle)
    if action["status"] != "ok":
        raise RuntimeError("deployment upload failed")
    checks = []
    for path in stage.rglob("*"):
        if path.is_file():
            checks.append("test \"$(sha256sum {} | cut -d ' ' -f 1)\" = {}".format(
                shlex.quote(c["remote"] + "/" + str(path.relative_to(stage))), sha256(path)))
    execute_script(store, "cd " + shlex.quote(c["remote"]) + "\ntar -xf deployment.tar\nchmod 755 runner\n"
                   + "\n".join(checks) + "\nrm deployment.tar", "verify_remote_artifacts", 120)
    write_new_json(root / "analysis/board.json", {"boot_id": boot, "remote_directory": c["remote"],
                                                 "deployment_sha256": sha256(bundle)})
    return boot


def native_args(contract, manifest, key, mode, destination):
    remote = contract["remote"]
    args = [remote + "/runner"]
    for index, (sid, threads) in enumerate(contract["programs"][key]["path"]):
        device, begin, end = sid.split(":")
        name = "{}_{}_{}".format(device, begin, end)
        row = manifest["programs"][name]
        prefix = "--stage{}-".format(index)
        args += [prefix + "name", name, prefix + "device", device,
                 prefix + "graph", remote + "/programs/" + name + "/graph.json",
                 prefix + "lib", remote + "/programs/" + name + "/module.so",
                 prefix + "params", remote + "/programs/" + name + "/params.bin",
                 prefix + "input-names", ",".join(row["input_names"]),
                 prefix + "runtime-num-threads", str(threads)]
        if device == "cpu":
            args += [prefix + "cpu-affinity", ",".join(map(str, range(threads)))]
    args += ["--input-list", remote + "/inputs.txt", "--runs", "307" if mode == "performance" else "6",
             "--queue-depth", "2", "--warmup-runs", "0", "--runtime-num-threads", "4",
             "--process-timeout-s", "180", "--defer-result-write", "--output-jsonl", destination + "/result.jsonl",
             "--output-mode", "classification" if mode == "performance" else "raw_all_stages"]
    if mode == "serial":
        args.append("--serial")
    if mode != "performance":
        args += ["--output-dump-dir", destination + "/outputs", "--c1-memory-audit"]
    return args


def collect_session(store, manifest, boot, key, mode, tag):
    import numpy as np
    from thesis_single_image_board_check import numeric
    from run_thesis_c1_memory_ablation import safe_extract

    c, root = store.contract, store.directory
    destination = root / "raw/sessions" / tag
    destination.mkdir(parents=True)
    remote_output = c["remote"] + "/" + tag
    args = native_args(c, manifest, key, mode, remote_output)
    # Reload only while no unrelated native process or RPC child exists.
    reset = "\n".join([
        "test \"$(cat /proc/sys/kernel/random/boot_id)\" = " + shlex.quote(boot),
        "for p in /proc/[0-9]*/exe; do x=$(readlink \"$p\" 2>/dev/null || true); case \"$x\" in *vta_stage_pipeline_runner*|*/runner|*/tvm_rpc) echo 'VTA workload active' >&2; exit 1;; esac; done",
        "test \"$(sha256sum /lib/firmware/vta_hpc.bit | cut -d ' ' -f 1)\" = " + c["bitstream"]["sha256"],
        "printf 'vta_hpc.bit\\n' > /sys/class/fpga_manager/fpga0/firmware",
        "test \"$(cat /sys/class/fpga_manager/fpga0/state)\" = operating",
        "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
    ])
    execute_script(store, reset, "reset_" + tag)
    env = ["env", "-i", "PATH=/usr/sbin:/usr/bin:/sbin:/bin", "LD_LIBRARY_PATH=" + c["remote"],
           "LD_PRELOAD=" + c["remote"] + "/libvta.so", "TVM_NUM_THREADS=4", "TVM_THREAD_POOL_SPIN_COUNT=0",
           "VTA_REPLAY_POLICY=disabled", "VTA_QUEUE_DIAGNOSTICS=" + ("0" if mode == "performance" else "1"),
           "C3_COMPLETION_TRACE=" + ("0" if mode == "performance" else "1")]
    script = "\n".join([
        "mkdir " + shlex.quote(remote_output),
        "if {} > {} 2>&1; then code=0; else code=$?; fi".format(
            shlex.join(env + args), shlex.quote(remote_output + "/native.log")),
        "tar -cf - -C " + shlex.quote(remote_output) + " .", "exit \"$code\"",
    ])
    action = execute_script(store, script, "session_" + tag, 210, check=False)
    archive = root / action["stdout"]
    if tarfile.is_tarfile(archive):
        safe_extract(archive, destination)
    if action["status"] != "ok":
        raise RuntimeError("native session failed: " + tag + "; raw evidence preserved")
    rows = [json.loads(line) for line in (destination / "result.jsonl").read_text().splitlines()]
    expected_count = 307 if mode == "performance" else 6
    if len(rows) != expected_count or sorted(r["frame_id"] for r in rows) != list(range(expected_count)):
        raise ValueError("incomplete native frames")
    rows.sort(key=lambda r: r["completion_index"])
    if [r["completion_index"] for r in rows] != list(range(expected_count)):
        raise ValueError("incomplete completion indices")
    checks = 0
    for row in rows:
        record = c["input_records"][row["input_index"]]
        expected = np.load(verify(record["frontiers"][str(manifest["final_node_id"])]), allow_pickle=False)
        if row["top1"] != int(np.argsort(-expected.reshape(-1), kind="stable")[0]):
            raise ValueError("classification differs from common reference")
        if row["input_index"] != row["frame_id"] % 3:
            raise ValueError("input/frame binding changed")
        if mode != "performance":
            for stage_index, (sid, _) in enumerate(c["programs"][key]["path"]):
                name = sid.replace(":", "_")
                outputs = row["stage_raw_outputs"][stage_index]["output_files"]
                nodes = manifest["programs"][name]["output_node_ids"]
                if len(outputs) != len(nodes):
                    raise ValueError("missing stage frontier")
                for item, node in zip(outputs, nodes):
                    relative = Path(item["path"]).relative_to(remote_output)
                    actual = np.frombuffer((destination / relative).read_bytes(), dtype=item["dtype"]).reshape(item["shape"])
                    expected_frontier = np.load(verify(record["frontiers"][str(node)]), allow_pickle=False)
                    numeric(actual, expected_frontier, c["numeric"])
                    checks += 1
    result = {"program": key, "mode": mode, "tag": tag, "boot_id": boot, "rows": len(rows),
              "full_frontier_checks": checks, "raw_result": ref(destination / "result.jsonl")}
    if mode == "pipeline":
        from thesis_c1_memory_evidence import audit_stage_log
        end_markers = [json.loads(line.split(" ", 1)[1]) for line in (destination / "native.log").read_text().splitlines()
                       if line.startswith("[C1_MEMORY_STAGE] ") and '"event":"end"' in line]
        if not end_markers:
            raise ValueError("missing actual worker completion markers")
        status = end_markers[0]["queue_status"]
        stages = [{"device": sid.split(":")[0]} for sid, _ in c["programs"][key]["path"]]
        result["worker_completion_audit"] = audit_stage_log(
            destination / "native.log", {"identity": {"stages": stages}},
            {"instruction": status["insn_capacity_bytes"], "uop": status["uop_capacity_bytes"]}, expected_frames=6)
    if mode == "performance":
        scored = rows[7:]
        result["ii_ms"] = (scored[-1]["completion_ms"] - scored[0]["completion_ms"]) / 299
        result["stage_run_ms_median"] = [statistics.median(r["stage{}_run_ms".format(i)] for r in scored)
                                         for i in range(len(c["programs"][key]["path"]))]
    write_new_json(destination / "metrics.json", result)
    # Retain local raw evidence, reclaim only this run's disposable remote output.
    execute_script(store, "rm -r " + shlex.quote(remote_output), "reclaim_" + tag)
    print(tag + ": " + ("II {:.6f} ms".format(result["ii_ms"]) if mode == "performance" else "{} frontier checks PASS".format(checks)), flush=True)
    return result


def run(output, prepare_only=False, prepared_run=None):
    store = create(output, prepared_run)
    try:
        manifest = reuse_programs(store, prepared_run) if prepared_run else build_programs(store)
        if prepare_only:
            return
        boot = deploy(store, manifest)
        qualification = []
        # Check the full choice first, then all ablations before any timing.
        keys = ["full_one_island", "endpoint_15", "endpoint_16", "two_islands", "three_islands"]
        for key in keys:
            for mode in ("serial", "pipeline"):
                qualification.append(collect_session(store, manifest, boot, key, mode, mode + "_" + key))
        write_new_json(output / "analysis/qualification.json", {"boot_id": boot, "sessions": qualification})
        observations = []
        for round_index, order in enumerate(store.contract["performance_orders"]):
            for key in order:
                observations.append(collect_session(store, manifest, boot, key, "performance", "r{}_{}".format(round_index, key)))
        summary = {key: {"ii_ms_by_round": [r["ii_ms"] for r in observations if r["program"] == key],
                         "ii_ms_median": statistics.median(r["ii_ms"] for r in observations if r["program"] == key)} for key in keys}
        result = {"status": "completed_single_boot_pilot", "boot_id": boot, "programs": summary,
                  "groups": {name: {**group, **summary[group["selected_program"]]} for name, group in store.contract["groups"].items()},
                  "independent_boots": 1, "performance_windows": len(observations), "limitations": store.contract["limitations"]}
        write_new_json(output / "analysis/result.json", result)
        lines = ["# 联合资源模型同板消融试运行", "", "状态：完成单启动三轮实测；不是三独立启动结论。", "",
                 "所有程序来自同一份一次量化图，共用 Tile 历史、运行库、runner 和输入。全部程序先通过串行/流水完整阶段输出检查，再计时。", "",
                 "| 组 | 实际执行方案 | 完成间隔中位数/ms |", "|---|---|---:|"]
        for name, group in result["groups"].items():
            lines.append("| {} | {} | {:.6f} |".format(name, group["selected_program"], group["ii_ms_median"]))
        lines.extend(["", "邻近切点：", ""])
        for key in ("endpoint_15", "endpoint_16", "full_one_island"):
            lines.append("- {}：{} ms。".format(key, summary[key]["ii_ms_by_round"]))
        lines.extend(["", "完整规则和去内部项规则选择相同，二者不能提供内部特征独立收益证据。", "",
                      "本轮限定为 Top-1 方案及邻近切点试运行，不能证明全域最优或一般化性能收益。", ""])
        (output / "REPORT.md").write_text("\n".join(lines))
    except Exception as error:
        write_new_json(output / "analysis/failure.json", {"status": "stopped", "reason": str(error),
                                                          "performance_results_valid": False})
        print("Stopped: " + str(error), flush=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared-run", type=Path)
    args = parser.parse_args()
    run(args.output_dir.resolve(), args.prepare_only, args.prepared_run.resolve() if args.prepared_run else None)
