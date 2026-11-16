#!/usr/bin/env python3
"""Frozen 4/1/4 CPU overlap intervention, using validated native deployment helpers."""
import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import statistics
import uuid

import run_cpu_vta_residual_validation as native
from analyze_cpu_stage_overlap_validation import analyze_gate_rows, pair_effect, repeated_pair_inference
from cpu_vta_residual_protocol import ROOT, matrix, verify_assets
from thesis_supplement_run import RunStore, sha256, utc, write_new_json

VERSION = "cpu_stage_overlap_4_1_4_v1"
DEFAULT_SOURCE = ROOT / ("vta/tutorials/frontend/report_out/stage_tile_cotuning/"
                        "cpu_vta_residual_validation/20260916T135816Z_residual_pilot_v2")
SOURCE_NAMES = ("vta_stage_pipeline_runner.cc", "vta_cpu_stage_gate.h", "vta_shared_buffer_plan.h",
                "vta_managed_lifecycle_control.h", "vta_command_capacity_certificate.h")
CONTROLS = (Path(__file__).name, "analyze_cpu_stage_overlap_validation.py", "validate_cpu_stage_gate_host.py",
            "test_cpu_stage_overlap_validation.py", "test_cpu_stage_gate.cc") + native.CONTROL_FILES


def read(path):
    return json.loads(Path(path).read_text())


def freeze(directory, source):
    prior = RunStore(source)
    verify_assets(prior)
    if read(prior.directory / "analysis/FINAL_DELIVERY_STATUS.json")["independent_boots"] != 3:
        raise ValueError("Expected the completed three-boot residual source")
    keys = ("assets", "paths", "programs", "topologies", "udmabuf_bytes", "numeric", "numeric_input",
            "source_identity", "canonical_qir_sha256", "bitstream_sha256", "completion", "board_state")
    contract = {key: copy.deepcopy(prior.contract[key]) for key in keys}
    for relative, identity in contract["assets"].items():
        identity["source_path"] = str(prior.directory / relative)
    sources = {"raw/runner_sources/" + name: ROOT / "vta/apps/native_deploy" / name for name in SOURCE_NAMES}
    sources["raw/runner_sources/picojson.h"] = ROOT / "3rdparty/picojson/picojson.h"
    sources.update({"raw/controller_sources/" + name: Path(__file__).with_name(name) for name in CONTROLS})
    for relative, path in sources.items():
        contract["assets"][relative] = {"source_path": str(path.resolve()), "sha256": sha256(path),
                                       "bytes": path.stat().st_size}
    contract.update(protocol_version=VERSION, run_id=Path(directory).name, stage="cpu_overlap",
                    campaign_id="cpu_overlap_" + uuid.uuid4().hex[:12], created_at=utc(),
                    prior_completed_run=str(prior.directory), actual_policy="ordinary_copy", queue_depth=2,
                    main_topology="03:17", threads=[4, 1, 4], affinity=[[0, 1, 2, 3]] * 3, spin_count=0,
                    intervention="FIFO whole CPU set/run/get/output-snapshot admission; VTA bypasses gate",
                    performance_window={"runs": 307, "warmup": 5, "settling": 2, "completions": 300, "gaps": 299},
                    correctness={"serial": 10, "pipeline_off_observe_exclusive": 10, "complete_frontier": True},
                    calibration_orders=[["off", "observe"], ["observe", "off"]],
                    exploratory_orders=[["A", "B"], ["B", "A"], ["A", "B"]],
                    replication_orders=[["A", "B"], ["B", "A"]],
                    primary="front CPU mean run reduction; tail and II secondary",
                    trigger="front run reduction >=5% and control CPU overlap >0 and exclusive overlap=0",
                    primary_telemetry="defer JSON write on; queue timing off; observe only if both paired II changes <=2%",
                    replication="fresh two balanced pairs after trigger; then two user-performed independent boots",
                    timing_rule="gate wait precedes stage_start and is excluded from stage/run busy",
                    inference="three paired boot means, Student-t df2; no single-boot causal closure",
                    drift_rule="record before/after matched serial; >5% front drift prevents replication",
                    stop_rule="correctness, timeout, identity or gate-order failure stops affected run; preserve partial evidence",
                    actual_fpga_frequency="unknown", formal_static_score_changed=False)
    store = RunStore.create(directory, contract)
    for relative, identity in contract["assets"].items():
        target = store.directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(identity["source_path"], target)
        if sha256(target) != identity["sha256"]:
            raise ValueError("Freeze copy drift: " + relative)
    bindings = {name: {"host_path": str(store.directory / "raw/runtime" / name),
                       "sha256": contract["assets"]["raw/runtime/" + name]["sha256"],
                       "bytes": contract["assets"]["raw/runtime/" + name]["bytes"]}
                for name in ("libtvm_runtime.so", "libvta.so", "tvm_rpc", "vta_hpc.bit", "u-dma-buf.ko")}
    write_new_json(store.directory / "analysis/board_artifact_bindings.json", bindings)
    store.event("cpu_overlap_protocol_frozen", parent_run=str(prior.directory))
    return store


def load(directory):
    store = RunStore(directory)
    if store.contract["protocol_version"] != VERSION:
        raise ValueError("Wrong protocol")
    verify_assets(store)
    for relative, identity in store.contract["assets"].items():
        if relative.startswith(("raw/runner_sources/", "raw/controller_sources/")):
            if sha256(identity["source_path"]) != identity["sha256"]:
                raise ValueError("Live control/runner source drift: " + relative)
    return store


def build_runner(store):
    path = store.directory / "analysis/runner_binding.json"
    if path.exists():
        binding = read(path)
        if sha256(store.directory / binding["path"]) != binding["sha256"]:
            raise ValueError("Runner drift")
        return binding
    directory = store.directory / "raw/build"
    directory.mkdir(exist_ok=False)
    source = store.directory / "raw/runner_sources"
    output = directory / "vta_stage_pipeline_runner_cpu_overlap"
    includes = [ROOT / name for name in ("include", "3rdparty/dlpack/include", "3rdparty/dmlc-core/include",
                                         "3rdparty/vta-hw/include", "vta/include")]
    script = directory / "build.sh"
    with script.open("x") as stream:
        stream.write("unset LD_LIBRARY_PATH\nsource " + shlex.quote(native.SDK) + "\n" +
                     'aarch64-xilinx-linux-g++ -std=c++17 -O2 --sysroot="$SDKTARGETSYSROOT" ' +
                     " ".join("-I " + shlex.quote(str(p)) for p in includes) + " " +
                     shlex.quote(str(source / SOURCE_NAMES[0])) + " -o " + shlex.quote(str(output)) +
                     " -L " + shlex.quote(str(store.directory / "raw/runtime")) +
                     ' -Wl,-rpath-link,"$SDKTARGETSYSROOT/lib" -Wl,-rpath-link,"$SDKTARGETSYSROOT/usr/lib"' +
                     " -ltvm_runtime -lvta -ldl -pthread\n")
    native.checked(store, ["bash", str(script)], "compile_frozen_cpu_overlap_runner", 300)
    binding = {"path": str(output.relative_to(store.directory)), "sha256": sha256(output),
               "runtime_bindings": read(store.directory / "analysis/board_artifact_bindings.json"),
               "source_files": {name: sha256(source / name) for name in SOURCE_NAMES},
               "defaults": {"cpu_stage_gate": "off"}}
    write_new_json(path, binding)
    return binding


@contextmanager
def adapters():
    """Reuse session/reset/partial recovery without editing frozen legacy helpers."""
    old_build, old_argv = native.build_runner, native.runner_argv
    def argv(board, output, stages, condition, mode, runs, telemetry=False, profile=False):
        result = old_argv(board, output, stages, condition, mode, runs, telemetry, profile)
        if "--defer-result-write" not in result:
            result += ["--defer-result-write"]
        if mode in ("pipeline", "correctness_pipeline"):
            result += ["--cpu-stage-gate", condition["gate_mode"]]
        return result
    native.build_runner, native.runner_argv = build_runner, argv
    try:
        yield
    finally:
        native.build_runner, native.runner_argv = old_build, old_argv


def condition(name, mode):
    value = copy.deepcopy(matrix()[0])
    value.update(condition=name, gate_mode=mode)
    return value


def session(store, boot, name, gate, mode, block):
    load(store.directory)
    record = native.session(store, boot, condition(name, gate), mode, block, telemetry=False)
    if mode == "pipeline" or mode == "correctness_pipeline":
        rows = [json.loads(s) for s in (store.directory / record["raw_jsonl"]).read_text().splitlines()]
        record["gate_metrics"] = analyze_gate_rows(rows, gate, discard=7 if mode == "pipeline" else 0)
        if abs(record["gate_metrics"]["ii_ms"] - record["metrics"].get("ii_ms", record["gate_metrics"]["ii_ms"])) > 1e-7:
            raise ValueError("Independent II mismatch")
    record["cpu_stage_gate_mode"] = gate if mode in ("pipeline", "correctness_pipeline") else "off"
    write_new_json(store.directory / "raw/sessions" / record["session_id"] / "gate_summary.json", record)
    return record


def collect_pairs(store, boot, control, prefix, orders):
    pairs = []
    for index, order in enumerate(orders):
        arms = {name: session(store, boot, name, control if name == "A" else "exclusive", "pipeline", f"{prefix}{index}")
                for name in order}
        effect = pair_effect(arms["A"], arms["B"])
        pairs.append({"boot_id": boot["boot_id"], "block": index,
                      "control": arms["A"], "exclusive": arms["B"], "effect": effect})
    return pairs


def execute(store, replicate=False):
    if (store.directory / "invalid_run.json").exists():
        raise ValueError("Failed run is immutable; use a new protocol")
    existing = [read(p) for p in sorted((store.directory / "analysis").glob("boot_*.json"))]
    if not replicate and existing:
        raise ValueError("Pilot cannot be repeated")
    if replicate and (not (store.directory / "analysis/pilot.json").exists() or len(existing) >= 3):
        raise ValueError("Successful pilot and fewer than three boots required")
    if replicate and not read(store.directory / "analysis/pilot.json")["replication_triggered"]:
        raise ValueError("Pilot did not pass the preregistered replication trigger")
    if not (store.directory / "analysis/host_validation.json").exists() or not read(
            store.directory / "analysis/host_validation.json")["passed"]:
        raise ValueError("New native host validation must pass first")
    boot = native.preflight(store)
    if boot["boot_id"] in {b["boot_id"] for b in existing}:
        raise ValueError("Another actual kernel boot is required")
    boot["boot_index"] = len(existing) + 1
    with adapters():
        native.deploy(store, boot, build_runner(store))
        try:
            session(store, boot, "serial", "off", "correctness_serial", "qualification")
            gates = [("A", read(store.directory / "analysis/pilot.json")["control_gate_mode"]), ("B", "exclusive")] if replicate else [
                ("off", "off"), ("observe", "observe"), ("exclusive", "exclusive")]
            for name, gate in gates:
                session(store, boot, name, gate, "correctness_pipeline", "qualification")
            before = session(store, boot, "serial", "off", "serial", "matched_before")
            if replicate:
                control = read(store.directory / "analysis/pilot.json")["control_gate_mode"]
                pairs = collect_pairs(store, boot, control, "replicate", store.contract["replication_orders"])
                write_new_json(store.directory / f"analysis/paired_boot_{boot['boot_index']}.json", pairs)
            else:
                calibration = []
                for index, order in enumerate(store.contract["calibration_orders"]):
                    arms = {gate: session(store, boot, gate, gate, "pipeline", f"calibration{index}") for gate in order}
                    calibration.append(arms["observe"]["metrics"]["ii_ms"] / arms["off"]["metrics"]["ii_ms"] - 1)
                control = "observe" if max(map(abs, calibration)) <= 0.02 else "off"
                write_new_json(store.directory / "analysis/gate_timing_calibration.json", {
                    "paired_ii_changes": calibration, "threshold": 0.02, "control_gate_mode": control})
                pairs = collect_pairs(store, boot, control, "explore", store.contract["exploratory_orders"])
            after = session(store, boot, "serial", "off", "serial", "matched_after")
            drift = after["metrics"]["stages"][0]["run_ms"]["mean"] / before["metrics"]["stages"][0]["run_ms"]["mean"] - 1
            write_new_json(store.directory / f"analysis/serial_drift_boot_{boot['boot_index']}.json", {
                "before": before, "after": after, "front_run_drift_fraction": drift})
            if not replicate:
                front = statistics.median(r["effect"]["front_run_reduction_fraction"] for r in pairs)
                trace = all(r["control"]["gate_metrics"]["cpu_run_overlap_ms"] > 0 and
                            r["exclusive"]["gate_metrics"]["cpu_stage_overlap_ms"] == 0 for r in pairs)
                triggered = front >= 0.05 and trace and abs(drift) <= 0.05
                write_new_json(store.directory / "analysis/pilot.json", {
                    "control_gate_mode": control, "exploratory_pairs": pairs,
                    "front_run_reduction_fraction": front, "trace_gate_passed": trace,
                    "front_serial_drift_fraction": drift, "replication_triggered": triggered,
                    "independent_boots": 1, "causal_status": "single boot exploratory"})
                if triggered:
                    fresh = collect_pairs(store, boot, control, "replicate", store.contract["replication_orders"])
                    write_new_json(store.directory / "analysis/paired_boot_1.json", fresh)
            return boot
        except Exception as error:
            write_new_json(store.directory / "invalid_run.json", {"error": str(error), "boot_id": boot["boot_id"], "time": utc()})
            raise
        finally:
            native.cleanup_rpc(store, boot)


def analyze(store):
    output = store.directory / "analysis" / ("report_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6])
    output.mkdir(exist_ok=False)
    summaries = [read(p) for p in sorted((store.directory / "raw/sessions").glob("*/gate_summary.json"))]
    pilot = read(store.directory / "analysis/pilot.json") if (store.directory / "analysis/pilot.json").exists() else {}
    paired = [r for p in sorted((store.directory / "analysis").glob("paired_boot_*.json")) for r in read(p)]
    boot_count = len({r["boot_id"] for r in paired})
    repeated = repeated_pair_inference(paired) if boot_count == 3 else None
    lines = ["# 原4/1/4 CPU阶段重叠干预", "", "CPU gate等待在stage_start之前；run与阶段busy均不包含等待。CPU线程数、线程池、mask不变，VTA不经过CPU门。结果仍不能分开调度、cache与DDR贡献。", "",
             "| 会话 | 门模式 | 首CPU run ms | VTA run ms | 尾CPU run ms | CPU run重叠ms | 首CPU门等待ms | II ms |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for summary in summaries:
        if summary["mode"] != "pipeline":
            continue
        m = summary["gate_metrics"]
        runs = [s["run_mean_ms"] for s in m["stages"]]
        wait = m["stages"][0]["gate_wait_mean_ms"]
        lines.append(f"| {summary['session_id']} | {m['gate_mode']} | {runs[0]:.3f} | {runs[1]:.3f} | {runs[2]:.3f} | {m['cpu_run_overlap_ms']:.3f} | {wait if wait is not None else 'unknown'} | {m['ii_ms']:.3f} |")
        rows = [json.loads(s) for s in (store.directory / summary["raw_jsonl"]).read_text().splitlines()][7:27]
        start = min(r["stage0_start_ms"] for r in rows)
        finish = max(r["stage2_end_ms"] for r in rows)
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="240">', '<rect width="100%" height="100%" fill="white"/>']
        for i in range(3):
            svg.append(f'<text x="5" y="{45+i*65}">stage{i}</text>')
            for row in rows:
                def x(value): return 90 + (value-start)/(finish-start)*1080
                request, acquire = row.get(f"stage{i}_cpu_gate_request_ms"), row.get(f"stage{i}_cpu_gate_acquire_ms")
                if request is not None:
                    svg.append(f'<rect x="{x(request):.3f}" y="{25+i*65}" width="{max(0,x(acquire)-x(request)):.3f}" height="25" fill="#bbb"><title>gate wait</title></rect>')
                svg.append(f'<rect x="{x(row[f"stage{i}_start_ms"]):.3f}" y="{25+i*65}" width="{x(row[f"stage{i}_end_ms"])-x(row[f"stage{i}_start_ms"]):.3f}" height="25" fill="{("#2674a5","#dc8c20","#2b9563")[i]}"><title>frame {row["frame_id"]}</title></rect>')
        svg.append('</svg>')
        with (output / (summary["session_id"] + ".svg")).open("x") as stream:
            stream.write("\n".join(svg))
    lines += ["", f"首CPU exploratory run中位变化（减少为正）：{pilot.get('front_run_reduction_fraction', 'unknown')}。",
              f"是否触发固定对照复现：{pilot.get('replication_triggered', False)}。独立boot数：{boot_count}；不足3时不宣称因果闭合。阶段增量不相加为II，旧proxy与正式排名未改变。",
              "匹配串行漂移、开关扰动、正确性和原始身份分别保留。unknown硬件compute/DMA/DDR不拟合固定34ms。"]
    if repeated:
        ci = repeated["front_run_ci95"]
        lines.append(f"三boot首CPU run减少均值{ci['mean_ms']:.6f} ms，df2 Student-t 95%CI [{ci['lower_ms']:.6f}, {ci['upper_ms']:.6f}]；机制门通过：{repeated['primary_gate_passed']}。")
    with (output / "CONCLUSION_ZH.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    write_new_json(output / "analysis.json", {"pilot": pilot, "sessions": summaries,
                                               "independent_boot_count": boot_count, "replication": repeated})
    print(output, flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("phase", choices=("freeze", "pilot", "replicate", "analyze"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-run", default=str(DEFAULT_SOURCE))
    parser.add_argument("--execute-board", action="store_true")
    args = parser.parse_args(argv)
    if args.phase in ("pilot", "replicate") and not args.execute_board:
        parser.error("Board execution requires --execute-board")
    store = freeze(args.run_dir, args.source_run) if args.phase == "freeze" else load(args.run_dir)
    if args.phase in ("pilot", "replicate"):
        execute(store, replicate=args.phase == "replicate")
        analyze(store)
    elif args.phase == "analyze":
        analyze(store)
    return store


if __name__ == "__main__":
    main()
