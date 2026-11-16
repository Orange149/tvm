"""Native task identity and byte-bound short-window evidence, without SSH."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from cpu_policies import pilot_runtime as runtime
from cpu_policies.pilot_common import DEFAULT_BUNDLE, load_legacy, protocol


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    load_legacy(DEFAULT_BUNDLE)
    root = tmp_path / "bundle"
    (root / "plans").mkdir(parents=True)
    for name in ("programs.json", "inputs.json", "executors.json"):
        (root / "plans" / name).write_bytes((DEFAULT_BUNDLE / "plans" / name).read_bytes())
    (root / "manifest.json").write_text(json.dumps({"bundle_id": "pilot-test"}))
    monkeypatch.setattr(runtime, "_legacy", lambda unused: load_legacy(DEFAULT_BUNDLE))
    return root


def configuration(reserved=False):
    from cpu_strategy.policy.domain import candidate, stage
    host = (3,) if reserved else (0, 1, 2, 3)
    return candidate([stage("cpu", 0, 3, 2, (0, 1)),
                      stage("vta", 3, 18, 1, host),
                      stage("cpu", 18, 21, 1, (2,))],
                     "pinned", "resource_group", host, host)


def component(device="cpu", reserved=False):
    from cpu_strategy.policy.domain import component as make_component, stage
    if device == "cpu":
        return make_component(stage("cpu", 0, 3, 2, (0, 1)), "pinned")
    host = (3,) if reserved else (0, 1, 2, 3)
    return make_component(stage("vta", 3, 18, 1, host), "pinned", host)


@pytest.mark.parametrize("reserved", [False, True])
def test_vta_uses_policy_runner_without_disguising_binding(bundle, reserved):
    config = component("vta", reserved)
    assert config["affinity_mode"] == "group"
    doc = runtime.make_task("calibration", config, bundle, warmup=2, scored=8)
    assert doc["executor"] == "policy"
    assert doc["experiment"] == protocol()["experiment"]
    argv, inputs = runtime.native_command(doc, bundle, "/board/bundle", "/board/job")
    assert "/board/bundle/runner/vta_stage_pipeline_runner_pinned" in argv
    assert "--cpu-worker-pinning" not in argv
    assert "--serial" in argv
    assert argv[argv.index("--runs") + 1] == "10"
    assert argv[argv.index("--control-cpu-affinity") + 1] == ("3" if reserved else "0,1,2,3")
    assert len(inputs.strip().splitlines()) == 1


@pytest.mark.parametrize("reserved", [False, True])
def test_pipeline_short_window_and_independent_host_control_affinity(bundle, reserved):
    doc = runtime.make_task("performance", configuration(reserved), bundle, warmup=8, scored=16, tail=4)
    argv, inputs = runtime.native_command(doc, bundle, "/board/bundle", "/board/job")
    assert "--cpu-worker-pinning" in argv
    assert "TVM_EXCLUDE_WORKER0=1" in argv
    assert "--serial" not in argv
    assert argv[argv.index("--runs") + 1] == "28"
    assert argv[argv.index("--cpu-stage-gate") + 1] == "resource_group"
    assert argv[argv.index("--stage1-cpu-affinity") + 1] == ("3" if reserved else "0,1,2,3")
    assert len(inputs.strip().splitlines()) == 1


def test_execution_and_host_settings_participate_in_identity(bundle):
    shared = runtime.make_task("performance", configuration(), bundle, warmup=8, scored=16, tail=4)
    reserved = runtime.make_task("performance", configuration(True), bundle, warmup=8, scored=16, tail=4)
    assert shared["task_id"] != reserved["task_id"]
    for key, value in (("executor", "legacy"), ("executor_sha256", "0" * 64),
                       ("executor_path", "runner/other"), ("bundle_id", "other")):
        bad = deepcopy(shared)
        bad[key] = value
        with pytest.raises(ValueError, match="identity|executor|configuration"):
            runtime.validate_document(bad, bundle)


@pytest.mark.parametrize("kwargs", [{"indices": [16]}, {"indices": [False]}, {"indices": [0, 16]},
                                    {"warmup": -1}, {"scored": True}, {"scored": 1}])
def test_invalid_windows_and_additional_images_rejected(bundle, kwargs):
    with pytest.raises(ValueError):
        runtime.make_task("performance", configuration(), bundle, **kwargs)


def test_frozen_labels_and_frame_configuration_cannot_be_overridden(bundle):
    with pytest.raises(ValueError, match="override"):
        runtime.make_task("calibration", component(), bundle, environment={})
    doc = runtime.make_task("performance", configuration(), bundle, warmup=8, scored=16, tail=4)
    doc["stages"][0]["threads"] = 1
    with pytest.raises(ValueError, match="identity|configuration"):
        runtime.native_command(doc, bundle, "/b", "/j")


def _affinities(config):
    stages = []
    for i, stage in enumerate(config["stages"]):
        tids = [100 + 10 * i + n for n in range(stage["threads"])]
        cpus = [[c] for c in stage["mask"]] if stage["device"] == "cpu" else [stage["mask"]]
        stages.append({"requested_cpus": stage["mask"], "active_runtime_threads": stage["threads"],
                       "thread_ids": tids, "caller_tid": tids[0],
                       "effective_affinities": [{"tid": t, "cpus": c, "cpu_time_delta_ms": 1.0}
                                                for t, c in zip(tids, cpus)]})
    return {"stages": stages, "control_threads": [
        {"role": role, "cpus": config["control_mask"]} for role in ("producer", "collector")]}


def _performance_raw(path, doc):
    path.mkdir(parents=True)
    completion = 0
    rows = []
    for index in range(28):
        # Deliberately large variation must remain accepted in the pilot.
        completion += 10 if index % 2 else 30
        rows.append({"frame_id": index, "completion_index": index, "input_index": 0,
                     "completion_ms": completion, "total_latency_ms": 80,
                     "completion_process_cpu_ms": completion / 2,
                     "cpu_stage_gate_mode": "resource_group"})
    (path / "result.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (path / "affinity.json").write_text(json.dumps(_affinities(doc["configuration"])))


def test_short_window_reports_fifteen_gaps_and_accepts_variation(bundle, tmp_path):
    doc = runtime.make_task("performance", configuration(), bundle, warmup=8, scored=16, tail=4)
    raw = tmp_path / "raw"
    _performance_raw(raw, doc)
    summary = runtime.validate_summary(raw, doc, bundle)
    assert summary["kept_timestamps"] == 16
    assert summary["gaps"] == 15
    assert summary["process_core_ms_per_completion"] == pytest.approx(summary["process_cpu_ms"] / 15)
    assert summary["p95_gap_ms"] == 30


def test_affinity_escape_and_duplicate_frame_rejected(bundle, tmp_path):
    doc = runtime.make_task("performance", configuration(), bundle, warmup=8, scored=16, tail=4)
    raw = tmp_path / "raw"
    _performance_raw(raw, doc)
    affinity = json.loads((raw / "affinity.json").read_text())
    affinity["stages"][0]["effective_affinities"][0]["cpus"] = [3]
    (raw / "affinity.json").write_text(json.dumps(affinity))
    with pytest.raises(ValueError, match="escaped"):
        runtime.validate_summary(raw, doc, bundle)
    (raw / "affinity.json").write_text(json.dumps(_affinities(doc["configuration"])))
    rows = [json.loads(line) for line in (raw / "result.jsonl").read_text().splitlines()]
    rows[1]["frame_id"] = 0
    (raw / "result.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="duplicate/missing"):
        runtime.validate_summary(raw, doc, bundle)


def test_receipt_replay_rejects_wrong_boot_and_modified_raw(bundle, tmp_path):
    from cpu_vta_partitioning.common import inventory, sha256, write
    doc = runtime.make_task("performance", configuration(), bundle, warmup=8, scored=16, tail=4)
    root = tmp_path / "run"
    boot = "12345678-1234-1234-1234-123456789abc"
    raw = root / "raw" / boot / doc["task_id"]
    _performance_raw(raw, doc)
    summary = runtime.validate_summary(raw, doc, bundle)
    receipt = root / "accepted" / boot / (doc["task_id"] + ".json")
    files = {str(raw.relative_to(root) / k): v for k, v in inventory(raw).items()}
    write(receipt, {"task": doc, "boot_id": boot, "summary": summary, "files": files})
    value = {**summary, "_receipt": {"path": str(receipt.relative_to(root)), "sha256": sha256(receipt)}}
    evidence = runtime.Evidence(root, bundle)
    assert evidence.summary(value, doc, boot)["summary"] == summary
    with pytest.raises(ValueError, match="receipt"):
        evidence.summary(value, doc, "different-boot")
    (raw / "result.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError):
        evidence.summary(value, doc, boot)


def test_component_uses_eight_scored_frames_and_exclusive_cpu_clock(bundle, tmp_path):
    doc = runtime.make_task("calibration", component(), bundle, warmup=2, scored=8)
    raw = tmp_path / "component"
    raw.mkdir()
    rows = []
    for i in range(10):
        rows.append({"frame_id": i, "completion_index": i, "input_index": 0,
                     "completion_ms": (i + 1) * 10, "cpu_stage_gate_mode": "off",
                     "stage_cpu_time_scope": "exclusive_process_cpu_time",
                     "stage0_set_ms": .1, "stage0_run_ms": 1000 if i < 2 else 2.,
                     "stage0_get_ms": .1, "stage0_set_process_cpu_ms": .05,
                     "stage0_run_process_cpu_ms": 4., "stage0_get_process_cpu_ms": .05})
    (raw / "result.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (raw / "affinity.json").write_text(json.dumps(_affinities(doc["configuration"])))
    result = runtime.validate_summary(raw, doc, bundle)
    assert result["service_ms"] == pytest.approx(2.2)
    assert result["run_core_ms"] == 4.
    assert result["boundary_core_ms"] == .1
    assert result["cpu_time_scope"] == "exclusive_process_cpu_time"


def test_single_input_qualification_checks_entire_output(bundle, tmp_path):
    import numpy as np
    doc = runtime.make_task("qualify_component", component(), bundle)
    programs = json.loads((bundle / "plans/programs.json").read_text())
    nodes = programs[doc["stages"][0]["program"]]["output_node_ids"]
    inputs = json.loads((bundle / "plans/inputs.json").read_text())
    raw = tmp_path / "qualification"
    (raw / "outputs/run_0").mkdir(parents=True)
    (bundle / "references").mkdir()
    outputs = []
    for n in nodes:
        path = f"references/node_{n}.npy"
        reference = np.array([1, 2], dtype="int8")
        np.save(bundle / path, reference)
        inputs["records"][0]["frontiers"][str(n)]["npy"] = path
        name = f"node_{n}.bin"
        reference.tofile(raw / "outputs/run_0" / name)
        outputs.append({"path": "/board/outputs/" + name, "shape": [2], "dtype": "int8"})
    (bundle / "plans/inputs.json").write_text(json.dumps(inputs))
    row = {"frame_id": 0, "completion_index": 0, "input_index": 0, "completion_ms": 1.,
           "cpu_stage_gate_mode": "off", "stage_raw_outputs": [{"output_files": outputs}]}
    (raw / "result.jsonl").write_text(json.dumps(row) + "\n")
    report = _affinities(doc["configuration"])
    stage = report["stages"][0]
    stage["worker_slots"] = [{"slot": i, "tid": row["tid"], "cpus": row["cpus"]}
                             for i, row in enumerate(stage["effective_affinities"])]
    (raw / "affinity.json").write_text(json.dumps(report))
    assert runtime.validate_summary(raw, doc, bundle)["correct"] is True
    np.array([1, 3], dtype="int8").tofile(raw / "outputs/run_0" / Path(outputs[0]["path"]).name)
    with pytest.raises(ValueError, match="incorrect full output"):
        runtime.validate_summary(raw, doc, bundle)


def test_ssh_reuses_connections_and_keeps_host_identity_checks(bundle):
    transport = runtime.SSH({"host": "192.168.1.247", "user": "root",
                             "remote_base": "/media/cpu_vta_data/cpu_vta_partitioning"},
                            execute_board=True, bundle=bundle)
    assert "ControlMaster=auto" in transport.options
    assert "ControlPersist=60" in transport.options
    assert "ControlPath=/tmp/cpu-vta-pilot-%C" in transport.options
    assert "StrictHostKeyChecking=yes" in transport.options
    assert "BatchMode=yes" in transport.options


def test_existing_detached_job_polls_half_second_without_relaunch(bundle, tmp_path, monkeypatch):
    from cpu_vta_partitioning.common import digest
    doc = runtime.make_task("performance", configuration(), bundle, warmup=8, scored=16, tail=4)
    boot = "12345678-1234-1234-1234-123456789abc"
    output = tmp_path / "collected"
    output.mkdir()
    _, inputs = runtime.native_command(doc, bundle, "/board/bundle", "/board/job")
    for name, value in (("boot_id", boot), ("task_hash", digest(doc)), ("inputs.txt", inputs)):
        (output / name).write_text(value)
    polls = iter(["", "0"])
    commands = []

    class ExistingJob:
        config = {"remote_base": "/board"}

        def boot_id(self):
            return boot

        def job_path(self, *unused):
            return "/board/job"

        def command(self, script):
            commands.append(script)
            if "task_hash" in script:
                return digest(doc)
            if "exit_code" in script:
                return next(polls)
            assert script == "mkdir -p /board/job"
            return ""

        def collect(self, *unused):
            return output

        def upload(self, *unused):
            pytest.fail("an existing detached job must not be relaunched")

    sleeps = []
    monkeypatch.setattr(runtime.time, "sleep", sleeps.append)
    assert runtime._measure(ExistingJob(), doc, bundle, "/board/bundle", tmp_path / "run", boot) == output
    assert sleeps == [.5]
    assert not any("nohup" in command for command in commands)
