#!/usr/bin/env python3
"""Compile and exercise frozen native CPU gate before any board performance calls."""
import argparse
import json
import os
from pathlib import Path
import sys

from run_cpu_vta_all_stage_gate_validation import load, ROOT
DEFAULT_SOURCE = ROOT / "vta/tutorials/frontend/report_out/stage_tile_cotuning/cpu_vta_residual_validation/20260916T135816Z_residual_pilot_v2"
from thesis_supplement_run import sha256, write_new_json


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    store = load(args.run_dir)
    directory = store.directory / "raw/host_validation"
    directory.mkdir(exist_ok=False)
    source = store.directory / "raw/runner_sources"
    runner = directory / "cpu_stage_gate_host_runner"
    command = ["g++", "-std=c++17", "-O2", str(source / "vta_stage_pipeline_runner.cc"), "-o", str(runner)]
    for name in ("include", "3rdparty/dlpack/include", "3rdparty/dmlc-core/include", "3rdparty/vta-hw/include", "vta/include"):
        command += ["-I", str(ROOT / name)]
    command += ["-L", str(ROOT / "build"), "-Wl,-rpath," + str(ROOT / "build"),
                "-ltvm_runtime", "-lvta_fsim", "-ldl", "-pthread"]
    action = store.execute(command, "compile_cpu_stage_gate_host", "host_validation", 180)
    if action["status"] != "ok":
        raise RuntimeError("Host compile failed: " + action["stderr"])
    fixture = DEFAULT_SOURCE / "raw/host_validation/historical_fixture"
    cpus = sorted(os.sched_getaffinity(0))[:4]
    if len(cpus) < 4:
        raise ValueError("Four host cores required")
    checks = []
    def execute(name, flags=(), runs=200, failing=False):
        output = directory / (name + ".jsonl")
        args = [str(runner), "--input", str(fixture / "input.bin"), "--runs", str(runs),
                "--runtime-num-threads", "2", "--queue-depth", "2", "--output-jsonl", str(output)]
        for i in range(2):
            prefix = f"--stage{i}-"
            args += [prefix+"graph", str(fixture / "graph.json"), prefix+"params", str(fixture / "params.bin"),
                     prefix+"lib", str(fixture / ("failing_module.so" if failing and i == 1 else "module.so")),
                     prefix+"device", "cpu", prefix+"name", "cpu"+str(i), prefix+"input-names", "data0",
                     prefix+"runtime-num-threads", "2", prefix+"cpu-affinity", ",".join(map(str, cpus[:2]))]
        args += ["--stage1-input-sources", "input:0"] + list(flags)
        args = ["env", "TVM_NUM_THREADS=2", "TVM_THREAD_POOL_SPIN_COUNT=0"] + args
        action = store.execute(args, "host_gate_"+name, "host_validation", 30)
        if failing:
            if action["status"] != "failed" or action["returncode"] == 0:
                raise ValueError("Expected clean injected failure, not timeout: " + name)
            if "injected failure" not in (store.directory / action["stderr"]).read_text():
                raise ValueError("Unexpected failure: " + name)
            checks.append({"check": name, "passed": True, "expected_failure": True})
            return []
        if action["status"] != "ok":
            raise RuntimeError("Native host failure: " + action["stderr"])
        rows = [json.loads(s) for s in output.read_text().splitlines()]
        assert [r["frame_id"] for r in rows] == list(range(runs))
        assert all(r["top1"] == 4095 for r in rows)
        gate = flags[flags.index("--cpu-stage-gate")+1] if "--cpu-stage-gate" in flags else "off"
        for row in rows:
            assert row.get("cpu_stage_gate_mode", "off") == gate
            if gate != "off":
                scope = flags[flags.index("--stage-gate-scope")+1] if "--stage-gate-scope" in flags else "cpu"
                assert row["stage_gate_scope"] == scope
            assert all(row[f"stage{i}_process_cpu_ms"] is None for i in range(2))
            for i in range(2):
                components = sum(row[f"stage{i}_{name}_ms"] for name in ("set", "run", "get"))
                assert abs(components-row[f"stage{i}_ms"]) < 0.01
                if gate == "off":
                    assert not any("cpu_gate" in name for name in row)
                else:
                    req, acq, rel = [row[f"stage{i}_cpu_gate_{point}_ms"] for point in ("request", "acquire", "release")]
                    assert req <= acq <= row[f"stage{i}_start_ms"] <= row[f"stage{i}_end_ms"] <= rel
        if gate == "exclusive":
            intervals = sorted((r[f"stage{i}_cpu_gate_acquire_ms"], r[f"stage{i}_cpu_gate_release_ms"])
                               for r in rows for i in range(2))
            assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
        checks.append({"check": name, "passed": True, "frames": runs})
        return rows
    default = execute("default")
    new_runner = runner
    runner = store.directory / "raw/host_baseline_runner"
    legacy = execute("legacy_default")
    runner = new_runner
    assert set(default[0]) == set(legacy[0])
    assert all(a["top1"] == b["top1"] and a["stage_count"] == b["stage_count"]
               for a, b in zip(default, legacy))
    off = execute("explicit_off", ["--cpu-stage-gate", "off"])
    assert set(default[0]) == set(off[0])
    execute("observe", ["--cpu-stage-gate", "observe", "--defer-result-write"])
    execute("exclusive_stress", ["--cpu-stage-gate", "exclusive", "--defer-result-write"], runs=2000)
    execute("exclusive_queue", ["--cpu-stage-gate", "exclusive", "--queue-timing", "--defer-result-write"], runs=1000)
    execute("all_exclusive_stress", ["--cpu-stage-gate", "exclusive", "--stage-gate-scope", "all", "--defer-result-write"], runs=2000)
    execute("all_exclusive_queue", ["--cpu-stage-gate", "exclusive", "--stage-gate-scope", "all", "--queue-timing", "--defer-result-write"], runs=1000)
    execute("all_exclusive_failure", ["--cpu-stage-gate", "exclusive", "--stage-gate-scope", "all", "--defer-result-write"], failing=True)
    execute("exclusive_injected_failure", ["--cpu-stage-gate", "exclusive", "--defer-result-write"], failing=True)
    gate_test = directory / "cpu_stage_gate_concurrency_check"
    compile_test = store.execute(["g++", "-std=c++17", "-O2", "-pthread", "-I", str(source),
                                  str(store.directory / "raw/controller_sources/test_cpu_stage_gate.cc"),
                                  "-o", str(gate_test)], "compile_cpu_gate_concurrency_check", "host_validation", 60)
    assert compile_test["status"] == "ok"
    check = store.execute([str(gate_test)], "cpu_gate_concurrency_check", "host_validation", 30)
    assert check["status"] == "ok"
    checks.append({"check": "8000 admissions, exception release, cancellation wake and post-cancel rejection", "passed": True})
    tests = store.execute([sys.executable, "-m", "pytest", "-q", str(Path(__file__).with_name("test_cpu_vta_all_stage_gate_validation.py"))],
                          "pytest_cpu_stage_overlap", "host_validation", 60)
    assert tests["status"] == "ok"
    checks.append({"check": "pytest", "passed": True, "evidence": tests})
    write_new_json(store.directory / "analysis/host_validation.json", {
        "passed": True, "checks": checks, "host_runner_sha256": sha256(runner),
        "native_source_sha256": sha256(source / "vta_stage_pipeline_runner.cc"), "board_calls": 0})
    print(json.dumps({"passed": True, "checks": checks}, indent=2), flush=True)


if __name__ == "__main__":
    main()
