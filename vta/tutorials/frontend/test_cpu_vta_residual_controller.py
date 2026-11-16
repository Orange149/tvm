"""Host-only checks for the opt-in residual controller; no board is contacted."""
import io
import json
from pathlib import Path
import tarfile
from unittest import mock

import pytest

import run_cpu_vta_residual_validation as controller
from cpu_vta_residual_protocol import matrix


def stages():
    result = []
    for index, device in enumerate(("cpu", "vta", "cpu")):
        name = f"stage{index}"
        inputs = ["edge0"] if index == 0 else ["edge0", "edge1"]
        result.append({"name": name, "device": device, "graph": name + "/graph.json",
                       "lib": name + "/module.so", "params": name + "/params.bin",
                       "input_names": inputs, "input_files": [name + "/input.bin"]})
    return result


def argv(mode, telemetry=False, component=False):
    specs = stages()
    condition = matrix()[0]
    if component:
        specs = [{**specs[1], "input_names": ["edge0"]}]
        condition = {"threads": [1], "affinity": [[0, 1, 2, 3]], "queue_depth": 2}
    return controller.runner_argv("/owned", "/owned/out", specs, condition, mode,
                                  10, telemetry=telemetry)


def test_default_keeps_queue_and_deferred_write_disabled():
    args = argv("pipeline")
    assert "--queue-timing" not in args
    assert "--defer-result-write" not in args
    assert args[args.index("--thread-affinity-report") + 1] == "/owned/out/affinity.json"
    assert args[args.index("--stage1-input-sources") + 1] == "stage0:0,stage0:1"
    assert args[args.index("--stage2-input-sources") + 1] == "stage1:0,stage1:1"


@pytest.mark.parametrize("mode", ["pipeline", "correctness_pipeline"])
def test_pipeline_telemetry_and_explicit_vta_host_affinity(mode):
    args = argv(mode, telemetry=True)
    assert "--queue-timing" in args and "--defer-result-write" in args
    assert "--serial" not in args
    assert args[args.index("--stage1-runtime-num-threads") + 1] == "1"
    assert args[args.index("--stage1-cpu-affinity") + 1] == "0,1,2,3"


@pytest.mark.parametrize("mode", ["serial", "correctness_serial"])
def test_serial_telemetry_never_passes_pipeline_queue_flag(mode):
    args = argv(mode, telemetry=True)
    assert "--serial" in args and "--defer-result-write" in args
    assert "--queue-timing" not in args


def test_integer_vta_component_uses_raw_output():
    args = argv("component", telemetry=True, component=True)
    assert args[args.index("--output-mode") + 1] == "raw"
    assert "--serial" in args and "--queue-timing" not in args


def archive(path, members):
    with tarfile.open(path, "w") as stream:
        for item, contents in members:
            stream.addfile(item, io.BytesIO(contents) if contents is not None else None)


def file_member(name, contents=b"evidence"):
    item = tarfile.TarInfo(name)
    item.size = len(contents)
    return item, contents


def test_standard_tar_root_and_nested_regular_file(tmp_path):
    root = tarfile.TarInfo(".")
    root.type = tarfile.DIRTYPE
    root.mode = 0o755
    source = tmp_path / "results.tar"
    archive(source, [(root, None), file_member("./outputs/complete.bin")])
    destination = tmp_path / "local"
    destination.mkdir()
    controller.safe_extract(source, destination)
    assert (destination / "outputs/complete.bin").read_bytes() == b"evidence"


@pytest.mark.parametrize("name", ["../escape", "nested/../../escape", "/absolute"])
def test_archive_rejects_escape_before_extracting_any_file(tmp_path, name):
    source = tmp_path / "results.tar"
    archive(source, [file_member("valid.bin"), file_member(name)])
    destination = tmp_path / "local"
    destination.mkdir()
    with pytest.raises(ValueError):
        controller.safe_extract(source, destination)
    assert not (destination / "valid.bin").exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_archive_rejects_links_and_special_files(tmp_path, kind):
    item = tarfile.TarInfo("link")
    item.type = kind
    item.linkname = "../external"
    source = tmp_path / "results.tar"
    archive(source, [(item, None)])
    destination = tmp_path / "local"
    destination.mkdir()
    with pytest.raises(ValueError):
        controller.safe_extract(source, destination)


def test_archive_cannot_replace_existing_local_evidence(tmp_path):
    source = tmp_path / "results.tar"
    archive(source, [file_member("session_contract.json")])
    destination = tmp_path / "local"
    destination.mkdir()
    existing = destination / "session_contract.json"
    existing.write_bytes(b"original")
    with pytest.raises(ValueError):
        controller.safe_extract(source, destination)
    assert existing.read_bytes() == b"original"


def test_archive_rejects_duplicate_normalized_file_paths(tmp_path):
    source = tmp_path / "results.tar"
    archive(source, [file_member("result.jsonl", b"first"),
                     file_member("./result.jsonl", b"replacement")])
    destination = tmp_path / "local"
    destination.mkdir()
    with pytest.raises(ValueError):
        controller.safe_extract(source, destination)
    assert not (destination / "result.jsonl").exists()


def affinity_report(condition, mode):
    rows = []
    for index, cpus in enumerate(condition["affinity"]):
        caller = 100 * (index + 1)
        tids = [caller, caller + 1]
        rows.append({"stage_index": index, "caller_tid": caller, "requested_cpus": cpus,
                     "thread_ids": tids,
                     "effective_affinities": [{"tid": tid, "cpus": cpus} for tid in tids]})
    return {"mode": mode, "pool_ownership": "shared_main_thread_pool" if mode == "serial"
            else "distinct_stage_thread_pools", "stages": rows}


@pytest.mark.parametrize("mode", ["pipeline", "correctness_pipeline", "serial", "correctness_serial"])
@pytest.mark.parametrize("stage_index", [0, 1, 2])
def test_effective_worker_mask_drift_is_rejected(tmp_path, mode, stage_index):
    condition = matrix()[0]
    report = affinity_report(condition, "serial" if "serial" in mode else "pipeline")
    report["stages"][stage_index]["effective_affinities"][1]["cpus"] = [0]
    path = tmp_path / "affinity.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        controller.verify_affinity(path, condition, stages(), mode)


def test_affinity_report_requires_the_caller_to_be_an_owned_thread(tmp_path):
    condition = matrix()[0]
    report = affinity_report(condition, "pipeline")
    report["stages"][0]["caller_tid"] = 999
    path = tmp_path / "affinity.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        controller.verify_affinity(path, condition, stages(), "pipeline")


@pytest.mark.parametrize("phase", ["pilot", "replicate"])
def test_board_phase_requires_explicit_opt_in_before_any_ssh(phase, tmp_path):
    with mock.patch.object(controller, "script_action") as remote, \
         mock.patch.object(controller, "load_protocol") as load:
        with pytest.raises(SystemExit) as stopped:
            controller.main([phase, "--run-dir", str(tmp_path / "unused")])
        assert stopped.value.code == 2
        remote.assert_not_called()
        load.assert_not_called()
