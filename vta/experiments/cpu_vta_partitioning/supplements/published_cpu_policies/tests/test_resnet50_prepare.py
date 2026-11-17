"""Local preparation invariants; no TVM import, board access or inference."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from cpu_vta_partitioning.common import digest, inventory, write
from resnet50_validation.assets import checked_file
from resnet50_validation.compiler import _history_context
from resnet50_validation.prepare import _frozen, graph_storage_estimate, inspect_module, next_build_log, pilot_programs, validate_build
from resnet50_validation.references import fnv1a64, qemu_affinity


def test_asset_same_name_does_not_replace_the_required_hash(tmp_path):
    path = tmp_path / "libvta.so"
    path.write_bytes(b"different build")
    with pytest.raises(ValueError, match="identity differs"):
        checked_file(path, "0" * 64)


def test_non_AArch64_ELF_is_rejected_before_dependency_inspection(tmp_path):
    path = tmp_path / "module.so"
    path.write_bytes(b"\x7fELF\x02\x01" + b"\x00" * 12 + b"\x3e\x00")
    with pytest.raises(ValueError, match="AArch64"):
        inspect_module(path, tmp_path)


def test_graph_storage_reuses_slots_and_keeps_parameters_separate():
    graph = {"nodes": [{"name": "data"}, {"name": "weight"}, {"name": "output"}],
             "arg_nodes": [0, 1], "node_row_ptr": [0, 1, 2, 3], "heads": [[2, 0, 0]],
             "attrs": {"shape": ["list_shape", [[1, 4], [8], [1, 4]]],
                       "dltype": ["list_str", ["float32", "int8", "float32"]],
                       "storage_id": ["list_int", [0, 1, 0]]}}
    result = graph_storage_estimate(graph, 8)
    assert result["graph_tensor_storage_bytes"] == 24
    assert result["two_boundary_sets_bytes"] == 64
    assert result["visible_storage_plus_parameters_and_boundary_bytes"] == 96
    assert not result["is_measured_peak"] and not result["strict_upper_bound"]
    graph["attrs"]["device_index"] = ["list_int", [1, 12, 12]]
    split = graph_storage_estimate(graph, 8)
    assert split["physical_device_storage_bytes"] == {"1": 16, "12": 24}
    assert split["visible_ext_dev_storage_bytes"] == 24
    assert split["physical_device_mapping_known"]
    graph["attrs"]["shape"][1][0] = [-1, 4]
    with pytest.raises(ValueError, match="dynamic"):
        graph_storage_estimate(graph)


def test_frozen_preparation_never_overwrites_a_different_identity(tmp_path):
    path = tmp_path / "plan.json"
    _frozen(path, {"network": "resnet50_v2"})
    _frozen(path, {"network": "resnet50_v2"})
    with pytest.raises(ValueError, match="identity changed"):
        _frozen(path, {"network": "resnet18"})
    assert json.loads(path.read_text()) == {"network": "resnet50_v2"}


def test_failed_compiler_logs_are_not_overwritten_on_resume(tmp_path):
    logs = tmp_path / "build_logs"; logs.mkdir()
    (logs / "vta_01_01.log").write_text("original failure")
    first = next_build_log(tmp_path, "vta_01_01")
    assert first.name == "vta_01_01.attempt_001.log"
    assert (logs / "vta_01_01.attempt_000.log").read_text() == "original failure"
    first.write_text("retry failure")
    assert next_build_log(tmp_path, "vta_01_01").name == "vta_01_01.attempt_002.log"
    assert first.read_text() == "retry failure"


def test_build_resume_requires_source_and_module_hashes(tmp_path):
    spec = {"name": "cpu_00_00", "kind": "interval", "device": "cpu", "begin": 0, "end": 1}
    source = tmp_path / "sources/qir.json"
    source.parent.mkdir(); source.write_text("graph")
    identity = {"canonical_sha256": "graphsha", "programs": [spec],
                "files": {"sources/qir.json": inventory(source.parent)["qir.json"]}}
    write(tmp_path / "plans/compiler_identity.json", identity)
    assert validate_build(tmp_path, require_all=False)["missing"] == ["cpu_00_00"]
    destination = tmp_path / "programs/cpu_00_00"; destination.mkdir(parents=True)
    (destination / "module.so").write_bytes(b"module")
    write(destination / "record.json", {"spec": spec, "build_identity": digest(identity),
          "canonical_sha256": "graphsha", "files": inventory(destination)})
    assert list(validate_build(tmp_path)["complete"]) == ["cpu_00_00"]
    (destination / "module.so").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="changed artifact"):
        validate_build(tmp_path)


def test_history_audit_preserves_native_key_matching_and_full_workload(monkeypatch):
    class Target:
        def __init__(self, model): self.model, self.keys = model, ["vta", "cpu"]
        def __str__(self): return "ext_dev -model=" + self.model
    class Config:
        index = 42
        def to_json_dict(self): return {"index": 42, "entity": [["tile", [1, 16]]]}
    cfg = Config()
    inp = SimpleNamespace(target=Target("pynq"), task=SimpleNamespace(workload=("conv", (1, 16))), config=cfg)
    result = SimpleNamespace(error_no=0, costs=[2.0, 1.0])
    class NativeHistory:
        def __init__(self, history):
            self.best_by_model = {("pynq", inp.task.workload): (inp, result)}
            self.best_by_targetkey = {("vta", inp.task.workload): (inp, result)}
            self._best_user_defined = {}
            self.calls = []
        def _query_inside(self, target, workload):
            self.calls.append((str(target), workload))
            hit = self.best_by_model.get((target.model, workload))
            for key in target.keys:
                hit = hit or self.best_by_targetkey.get((key, workload))
            return hit[0].config if hit else None
    autotvm = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "tvm", SimpleNamespace(autotvm=autotvm))
    monkeypatch.setitem(sys.modules, "tvm.autotvm.task.dispatcher", SimpleNamespace(ApplyHistoryBest=NativeHistory))
    context = _history_context("unused")
    assert context._query_inside(Target("pynq"), ("conv", (1, 16))) is cfg
    assert context._query_inside(Target("axu5evb"), ("conv", (1, 16))) is cfg
    assert context._query_inside(Target("axu5evb"), ("conv", (1, 32))) is None
    assert len(context.queries) == 3
    hit = context.queries[(str(Target("axu5evb")), ("conv", (1, 16)))]
    assert hit["policy"] == "native_target_key_and_full_workload"
    assert hit["source_record"]["target"] == "ext_dev -model=pynq"
    assert hit["source_record"]["config_index"] == 42
    assert len(context.calls) == 3  # the wrapper delegated every selection


def test_memory_guard_rejects_compilation_before_launch(monkeypatch, tmp_path):
    from resnet50_validation import prepare
    monkeypatch.setattr(prepare, "available_memory_bytes", lambda: 1024**3)
    with pytest.raises(MemoryError, match="before compiler admission"):
        prepare.guarded_sdk_command({}, ["never", "run"], tmp_path / "log", cwd=tmp_path, samples=[])
    assert not (tmp_path / "log").exists()


def test_memory_guard_cancels_its_owned_process_group(monkeypatch, tmp_path):
    from resnet50_validation import prepare
    setup = tmp_path / "sdk.sh"; setup.write_text("true\n")
    local = {"sdk_setup": str(setup), "host_tvm_library": str(tmp_path / "libtvm.so")}
    values = iter([5 * 1024**3, 5 * 1024**3, 1024**3])
    monkeypatch.setattr(prepare, "available_memory_bytes", lambda: next(values))
    original = prepare.subprocess.Popen; children = []
    def tracked(*args, **kwargs):
        child = original(*args, **kwargs); children.append(child); return child
    monkeypatch.setattr(prepare.subprocess, "Popen", tracked)
    samples = []
    with pytest.raises(MemoryError, match="2 GiB"):
        prepare.guarded_sdk_command(local, [sys.executable, "-c", "import time; time.sleep(60)", "test"],
                                    tmp_path / "failure.log", cwd=tmp_path, samples=samples)
    assert len(children) == 1 and children[0].poll() is not None
    assert samples[0]["minimum_host_available_bytes"] == 1024**3
    assert (tmp_path / "failure.log").exists()


def test_native_recovery_cannot_copy_assets_into_the_old_output(tmp_path):
    from resnet50_validation.prepare import reuse_graph_assets
    with pytest.raises(ValueError, match="new output"):
        reuse_graph_assets(tmp_path, tmp_path)


def test_stopped_regression_output_and_children_are_read_only():
    from resnet50_validation.assets import LEGACY_OUTPUT
    from resnet50_validation.prepare import writable_output
    for path in (LEGACY_OUTPUT, LEGACY_OUTPUT / "build"):
        with pytest.raises(ValueError, match="read-only"):
            writable_output(path)


def test_reference_hash_uses_native_fnv1a64_convention():
    assert fnv1a64(b"") == "14650fb0739d0383"
    assert fnv1a64(b"hello") == "005a0d15131ec7a1"


def test_qemu_auxiliary_is_allowed_only_when_nonlogical_and_zero_CPU():
    report = {"cpu_worker_pinning": True, "stages": [{"caller_tid": 10, "active_runtime_threads": 1,
        "requested_cpus": [0], "worker_slots": [{"slot": 0, "tid": 10, "cpus": [0]}],
        "effective_affinities": [{"tid": 10, "cpus": [0], "cpu_time_delta_ms": 4},
                                 {"tid": 11, "cpus": [0, 1, 2, 3], "cpu_time_delta_ms": 0}]}]}
    stages = [{"threads": 1, "mask": [0]}]
    assert len(qemu_affinity(report, stages)) == 1
    report["stages"][0]["effective_affinities"][1]["cpu_time_delta_ms"] = 0.1
    with pytest.raises(ValueError, match="helper consumed"):
        qemu_affinity(report, stages)
    report["stages"][0]["effective_affinities"][1]["cpu_time_delta_ms"] = 0
    report["stages"][0]["worker_slots"][0]["cpus"] = [1]
    with pytest.raises(ValueError, match="active ARM worker"):
        qemu_affinity(report, stages)


def test_structural_pilot_covers_six_CPU_and_three_VTA_programs():
    from resnet50_validation.assets import DEFAULT_OUTPUT
    metadata = DEFAULT_OUTPUT / "units/units_and_boundaries.json"
    if not metadata.exists():
        pytest.skip("requires prepared typed ResNet50 units")
    units = json.loads(metadata.read_text())["units"]
    programs = json.loads((metadata.parent / "programs.json").read_text())
    result = pilot_programs(programs, units)
    assert len(set(result["cpu"])) == 6 and len(set(result["vta"])) == 3
    assert result["component_signatures"] == 39
    assert result == pilot_programs(programs, units)
