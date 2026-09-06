#!/usr/bin/env python3
"""Local-only tests for the CPU-VTA Pipeline V1 P5A gate."""

from argparse import Namespace

import numpy as np
import pytest

import build_cpu_vta_pipeline_v1_p5a as p5a
import compare_cpu_vta_pipeline_v1_reference as comparator
import deploy_classification_stage_pipeline_native as deploy


def _args():
    return Namespace(
        board="root@192.168.1.247",
        remote_dir="/mnt/sd/test",
        deploy_mode="sync",
        image_dir="",
        runtime_num_threads=4,
        stage_runtime_num_threads="2,1,3",
        stage0_runtime_num_threads=0,
        stage1_runtime_num_threads=0,
        stage2_runtime_num_threads=0,
        vta_runtime_profile_dir="",
        vta_runtime_profile_events_limit=20,
        vta_runtime_profile_checkpoint_every=0,
        output_dump_dir="correctness_outputs",
        runs=20,
        queue_depth=2,
        runner_output_mode="raw",
    )


def test_run_scripts_dump_serial_and_pipeline_tensors_separately(tmp_path):
    stages = []
    for index, device in enumerate(("cpu", "vta", "cpu")):
        name = "stage{}".format(index)
        stages.append(
            {
                "name": name,
                "device": device,
                "graph": "stages/{}/graph.json".format(name),
                "lib": "stages/{}/graphlib.so".format(name),
                "params": "stages/{}/params.params".format(name),
                "input_names": ["data0"],
            }
        )
    args = _args()
    deploy.write_run_script(args, tmp_path, stages, True, "serial.jsonl", "serial.sh")
    deploy.write_run_script(args, tmp_path, stages, False, "pipeline.jsonl", "pipeline.sh")
    serial = (tmp_path / "serial.sh").read_text(encoding="utf-8")
    pipeline = (tmp_path / "pipeline.sh").read_text(encoding="utf-8")
    assert "--output-mode raw" in serial
    assert "--output-dump-dir correctness_outputs/serial" in serial
    assert "--output-dump-dir correctness_outputs/pipeline" in pipeline
    assert "--stage0-cpu-affinity 0,1" in pipeline
    assert "--stage2-cpu-affinity 0,1,2" in pipeline


def test_tensor_comparator_accepts_identity_and_rejects_wrong_logits():
    reference = np.linspace(-2.0, 2.0, 1000, dtype="float32").reshape(1, 1000)
    gate = {
        "min_cosine_similarity": 0.95,
        "max_normalized_rmse": 0.50,
        "top_k": 10,
        "min_top_k_overlap": 5,
    }
    exact = comparator.compare_tensors(reference, reference.copy(), gate)
    assert exact["passed"] is True
    wrong = comparator.compare_tensors(reference, -reference, gate)
    assert wrong["passed"] is False
    assert "cosine_similarity" in wrong["failed_checks"]


def test_p5a_build_command_is_local_only(tmp_path):
    row = {
        "candidate_id": "cpu-00-02_t2__vta-03-17_t1__cpu-18-20_t2",
        "stage_runtime_threads": [2, 1, 2],
    }
    command = p5a.build_command(row, tmp_path, tmp_path / "cache")
    assert "--package-only" in command
    assert "--board" not in command
    assert command[command.index("--stage-runtime-num-threads") + 1] == "2,1,2"
    assert command[command.index("--output-dump-dir") + 1] == "correctness_outputs"
    assert command[command.index("--runs") + 1] == "22"


def test_stage_accounting_ids_are_unique():
    stages = [
        {"name": "stage0", "device": "cpu"},
        {"name": "stage1", "device": "vta"},
        {"name": "stage2", "device": "cpu"},
    ]
    ids = p5a._stage_accounting_ids("candidate", stages)
    assert len(ids) == len(set(ids))
    assert len(ids) == 5


def test_run_script_thread_audit_reads_actual_runner_arguments(tmp_path):
    script = tmp_path / "run_stage_pipeline.sh"
    script.write_text(
        "./runner --stage0-runtime-num-threads 4 "
        "--stage1-runtime-num-threads 1 --stage2-runtime-num-threads 2\n",
        encoding="utf-8",
    )
    assert p5a._run_script_thread_map(script, 3) == {
        "stage0": 4,
        "stage1": 1,
        "stage2": 2,
    }


def test_reuse_staging_does_not_alias_source_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("original\n", encoding="utf-8")
    work_dir, staged = deploy.stage_reused_package(source)
    try:
        (staged / "manifest.json").write_text("changed\n", encoding="utf-8")
        assert (source / "manifest.json").read_text(encoding="utf-8") == "original\n"
    finally:
        __import__("shutil").rmtree(work_dir)


def test_ext4_health_parser_rejects_only_target_device_corruption():
    log = """
[ 1.0] EXT4-fs error (device mmcblk1p2): bad block bitmap checksum
[ 2.0] EXT4-fs error (device mmcblk0p2): bad extra_isize 65535
[ 3.0] unrelated warning
"""
    assert deploy.ext4_error_lines(log, "/dev/mmcblk1p2") == [
        "[ 1.0] EXT4-fs error (device mmcblk1p2): bad block bitmap checksum"
    ]
    assert deploy.ext4_error_lines(log, "/dev/mmcblk9p9") == []


def test_df_device_parser_rejects_header_only_output():
    with pytest.raises(RuntimeError, match="filesystem data row"):
        deploy.parse_df_device("Filesystem 1024-blocks Used Available Capacity Mounted on\n")
    assert (
        deploy.parse_df_device(
            "Filesystem 1024-blocks Used Available Capacity Mounted on\n"
            "/dev/mmcblk1p2 1048576 100 1048476 1% /mnt/sd\n"
        )
        == "/dev/mmcblk1p2"
    )


def test_reuse_package_inherits_stage_threads_when_cli_omits_them(tmp_path, monkeypatch):
    manifest = {
        "stages": [
            {"name": "stage0", "device": "cpu"},
            {"name": "stage1", "device": "vta"},
            {"name": "stage2", "device": "cpu"},
        ],
        "pipeline_stage_runtime_threads": {"stage0": 4, "stage1": 1, "stage2": 2},
    }
    (tmp_path / "manifest.json").write_text(__import__("json").dumps(manifest), encoding="utf-8")
    args = _args()
    args.stage_runtime_num_threads = ""
    args.stage0_runtime_num_threads = 0
    args.stage1_runtime_num_threads = 0
    args.stage2_runtime_num_threads = 0
    args.candidate_id = ""
    args.serial = False
    args.run_serial_before_pipeline = True
    args.serial_output_jsonl = "serial.jsonl"
    args.pipeline_output_jsonl = "pipeline.jsonl"
    args.rpc_baseline_result = ""
    args.correctness_policy = "exact"
    monkeypatch.setattr(deploy, "compile_runner", lambda _path: None)
    monkeypatch.setattr(deploy, "copy_runtime_libs", lambda _path: None)
    monkeypatch.setattr(deploy, "write_run_script", lambda *unused, **kwargs: None)
    monkeypatch.setattr(deploy, "check_package", lambda *unused, **kwargs: None)

    deploy.refresh_reused_package(args, tmp_path)

    assert args.stage_runtime_num_threads == "4,1,2"
    updated = __import__("json").loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert updated["pipeline_stage_runtime_threads"] == {
        "stage0": 4,
        "stage1": 1,
        "stage2": 2,
    }
    assert updated["deployment"] == {
        "ssh_target": "root@192.168.1.247",
        "remote_dir": "/mnt/sd/test",
        "deploy_mode": "sync",
    }


def test_serial_only_profile_may_use_all_cpu_stages():
    deploy.validate_stage_device_sequence(["cpu", "cpu", "cpu"], serial_only_profile=True)


def test_pipeline_candidate_still_requires_vta():
    with pytest.raises(RuntimeError, match="pipeline execution requires"):
        deploy.validate_stage_device_sequence(
            ["cpu", "cpu", "cpu"], serial_only_profile=False
        )


def test_serial_only_profile_disables_pipeline_execution():
    args = _args()
    args.serial = True
    args.run_serial_before_pipeline = False
    assert deploy.pipeline_execution_configured(args) is False
    args.run_serial_before_pipeline = True
    assert deploy.pipeline_execution_configured(args) is True


def test_boundary_abi_uses_order_shape_and_dtype_not_descriptive_role_names():
    producer = {
        "arity": 1,
        "slots": [{"role": "out", "shape": [1, 64, 56, 56], "dtype": "float32"}],
    }
    consumer = {
        "arity": 1,
        "slots": [{"role": "data", "shape": [1, 64, 56, 56], "dtype": "float32"}],
    }
    assert p5a._boundary_contract(producer) == p5a._boundary_contract(consumer)
    tuple_output = {
        "arity": 2,
        "slots": [
            {"slot_index": 0, "role": "out0", "shape": [1, 64], "dtype": "float32"},
            {"slot_index": 1, "role": "out1", "shape": [1, 32], "dtype": "float32"},
        ],
    }
    tuple_input = {
        "arity": 2,
        "slots": [
            {"slot_index": 0, "role": "main", "shape": [1, 64], "dtype": "float32"},
            {"slot_index": 1, "role": "residual", "shape": [1, 32], "dtype": "float32"},
        ],
    }
    assert p5a._boundary_contract(tuple_output) == p5a._boundary_contract(tuple_input)
