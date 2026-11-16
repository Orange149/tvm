"""The opt-in overlay, payload isolation, and actual aarch64 loader contract."""
import hashlib
import os
from pathlib import Path

import pytest

from cpu_vta_partitioning.common import REPO, inventory, read, write
from cpu_strategy.prepare import AVAILABLE_INDICES, PROGRAM_NAMES, _subset_payload, native_acceptance
from cpu_strategy.runner_patch import BASE_RUNNER_SHA256, patch_runner


def test_runner_overlay_preserves_historical_preimage_and_clock_event():
    original = REPO / "vta/apps/native_deploy/vta_stage_pipeline_runner.cc"
    before = original.read_bytes()
    patched, record = patch_runner(before.decode())
    assert hashlib.sha256(before).hexdigest() == BASE_RUNNER_SHA256
    assert original.read_bytes() == before
    assert "--allow-cpu-affinity-superset" in patched
    assert "args.allow_cpu_affinity_superset" in patched
    assert patched.count("completion_process_cpu_ms = ProcessCpuMillis();") == 4
    assert "frame->done_ms = NowMillis();\n                frame->completion_process_cpu_ms" in patched
    assert "frame.done_ms = NowMillis();\n      frame.completion_process_cpu_ms" in patched
    assert "disabled_concurrent_pipeline" in patched
    assert "cpu_time_delta_ms" in patched and "thread_role" in patched
    assert "FinishThreadActivity(&stage_affinities[stage_index])" in patched
    assert record["patched_sha256"] == hashlib.sha256(patched.encode()).hexdigest()
    assert record["patch_sha256"] == hashlib.sha256(record["unified_diff"].encode()).hexdigest()


def test_overlay_rejects_changed_preimage():
    source = (REPO / "vta/apps/native_deploy/vta_stage_pipeline_runner.cc").read_text()
    with pytest.raises(ValueError, match="preimage"):
        patch_runner(source + "\n")


def small_parent(tmp_path):
    parent = tmp_path / "parent"
    programs = {}
    for name in PROGRAM_NAMES:
        folder = parent / "programs" / name
        folder.mkdir(parents=True)
        for filename in ("graph.json", "module.so", "params.bin"):
            (folder / filename).write_bytes((name + filename).encode())
        device, begin, end = name.split("_")
        programs[name] = {"name": name, "device": device, "begin": int(begin), "end": int(end) + 1,
                          "input_node_ids": [13],
                          "files": inventory(folder)}
    write(parent / "plans/programs.json", programs)
    rows = []
    for index in range(19):
        record = {"input": f"inputs/{index}.bin", "frontiers": {
            "13": {"raw": f"inputs/frontiers/{index}.bin", "npy": f"references/{index}.npy",
                   "dtype": "int32", "shape": [1]}}}
        rows.append(record)
        for filename in (record["input"], record["frontiers"]["13"]["raw"], record["frontiers"]["13"]["npy"]):
            path = parent / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(str(index).encode())
        import numpy as np
        np.save(parent / record["frontiers"]["13"]["npy"], np.array([index], dtype="int32"))
    write(parent / "plans/inputs.json", {"records": rows, "files": {}, "reference_identity": "frozen-parent"})
    for folder in ("runtime", "firmware"):
        (parent / folder).mkdir()
        (parent / folder / "file").write_bytes(folder.encode())
    for filename in ("compiler_identity.json", "runtime_build.json", "sdk_runtime.json"):
        write(parent / "plans" / filename, {"fixture": True})
    return parent, {"files": inventory(parent)}


def test_subset_retains_indices_and_copies_only_available_data(tmp_path):
    parent, manifest = small_parent(tmp_path)
    workspace = tmp_path / "workspace"
    selected = _subset_payload(parent, workspace, manifest)
    assert set(selected) == set(PROGRAM_NAMES)
    inputs = read(workspace / "plans/inputs.json")
    assert len(inputs["records"]) == 19
    assert inputs["available_indices"] == list(AVAILABLE_INDICES)
    assert len(inputs["files"]) == 12
    assert (workspace / "inputs/16.bin").read_bytes() == b"16"
    assert not (workspace / "inputs/1.bin").exists()
    assert (parent / "inputs/1.bin").is_file()
    assert inputs["reference_identity"] == "frozen-parent"


def test_final_output_reference_need_not_have_a_raw_input_copy(tmp_path):
    parent, _ = small_parent(tmp_path)
    inputs = read(parent / "plans/inputs.json")
    inputs["records"][16]["frontiers"]["13"].pop("raw")
    write(parent / "plans/inputs.json", inputs, replace=True)
    _subset_payload(parent, tmp_path / "workspace", {"files": inventory(parent)})
    assert (tmp_path / "workspace/references/16.npy").is_file()
    selected = read(tmp_path / "workspace/plans/inputs.json")
    derived = selected["records"][16]["frontiers"]["13"]["raw"]
    import numpy as np
    assert (tmp_path / "workspace" / derived).read_bytes() == np.array([16], dtype="int32").tobytes()
    assert selected["derived_frontier_inputs"][0]["source_npy"] == "references/16.npy"


def test_subset_rejects_hash_changed_reference(tmp_path):
    parent, manifest = small_parent(tmp_path)
    (parent / "references/16.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        _subset_payload(parent, tmp_path / "workspace", manifest)


def test_aarch64_overlay_acceptance_when_prepared():
    workspace = os.environ.get("CPUVTA_CPU_STRATEGY_WORKSPACE")
    if not workspace:
        pytest.skip("requires locally prepared CPU strategy runner")
    result = native_acceptance(Path(workspace))
    assert result["passed"]
    assert "affinity and completion-clock self-test passed" in result["native_selftest"]
