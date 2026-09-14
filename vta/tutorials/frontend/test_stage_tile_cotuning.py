"""Local checks of tuning coverage, cache identity, and iteration accounting."""

import numpy as np
import pytest
from types import SimpleNamespace
import tvm
from tvm import autotvm, te
import vta

from tune_resnet18_vta import register_vta_conv2d_template
from vta_tuning_history import audited_history, history_identity
from vta_autotvm_measure import reference_data, VTASequentialBuilder
from run_stage_tile_iteration import (candidate_manifest, local_candidates,
                                      projection_repair_candidates, rerank,
                                      write_safe_overlay, write_validated_baseline)
from analyze_stage_memory_additivity import occurrence_count


@pytest.fixture
def task():
    register_vta_conv2d_template()
    env = vta.get_env()
    a = te.placeholder((1, 1, 4, 4, env.BATCH, env.BLOCK_IN), dtype="int8")
    w = te.placeholder((1, 1, 1, 1, env.BLOCK_OUT, env.BLOCK_IN), dtype="int8")
    return autotvm.task.create("conv2d_packed.vta", args=(a, w, (1, 1), (0, 0, 0, 0),
        (1, 1), "NCHW%dn%dc" % (env.BATCH, env.BLOCK_IN), "int32"),
        target=env.target, target_host=env.target_host)


def record(task, index, cost=0.01):
    inp = autotvm.measure.MeasureInput(task.target, task, task.config_space.get(index))
    result = autotvm.measure.MeasureResult((cost,), 0, 0.02, 1)
    return autotvm.record.encode(inp, result) + "\n"


def test_history_rejects_empty_and_nonfinite_measurements(tmp_path, task):
    path = tmp_path / "empty.log"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        history_identity(path)
    path.write_text(record(task, 0, float("nan")))
    with pytest.raises(ValueError, match="no successful"):
        history_identity(path)


def test_history_content_changes_identity_and_dispatch(tmp_path, task):
    path = tmp_path / "tuning.log"
    path.write_text(record(task, 0))
    first = history_identity(path)
    audit = {}
    with audited_history(str(path), audit, require_tuned=True):
        cfg = autotvm.task.DispatchContext.current.query(task.target, task.workload)
        assert cfg.index == 0 and not cfg.is_fallback
    assert audit["tuned_workloads"] == 1 and audit["fallback_workloads"] == 0
    path.write_text(record(task, 1))
    assert history_identity(path)["sha256"] != first["sha256"]
    with audited_history(str(path), require_tuned=True):
        assert autotvm.task.DispatchContext.current.query(task.target, task.workload).index == 1


def test_explicit_tophub_survives_audit_and_records_file(tmp_path, task, monkeypatch):
    path = tmp_path / "vta_v0.10.log"
    path.write_text(record(task, 1))
    monkeypatch.setattr(autotvm.tophub, "AUTOTVM_TOPHUB_ROOT_PATH", tmp_path)
    audit = {}
    with audited_history("", audit, require_tuned=True, tophub_targets=[task.target]):
        cfg = autotvm.task.DispatchContext.current.query(task.target, task.workload)
        assert cfg.index == 1 and not cfg.is_fallback
    assert audit["mode"] == "tophub"
    assert audit["files"][0]["path"] == str(path.resolve())
    assert len(audit["files"][0]["sha256"]) == 64
    assert audit["tuned_workloads"] == 1 and audit["fallback_workloads"] == 0


def test_partial_history_overlays_tophub(tmp_path, task, monkeypatch):
    env = vta.get_env()
    a = te.placeholder((1, 1, 8, 8, env.BATCH, env.BLOCK_IN), dtype="int8")
    w = te.placeholder((1, 1, 1, 1, env.BLOCK_OUT, env.BLOCK_IN), dtype="int8")
    other = autotvm.task.create("conv2d_packed.vta", args=(a, w, (1, 1), (0, 0, 0, 0),
        (1, 1), "NCHW%dn%dc" % (env.BATCH, env.BLOCK_IN), "int32"),
        target=env.target, target_host=env.target_host)
    tophub = tmp_path / "vta_v0.10.log"
    tophub.write_text(record(task, 1) + record(other, 1))
    overlay = tmp_path / "overlay.log"
    overlay.write_text(record(task, 2))
    monkeypatch.setattr(autotvm.tophub, "AUTOTVM_TOPHUB_ROOT_PATH", tmp_path)
    with audited_history(str(overlay), require_tuned=True, tophub_targets=[task.target]):
        assert autotvm.task.DispatchContext.current.query(task.target, task.workload).index == 2
        assert autotvm.task.DispatchContext.current.query(other.target, other.workload).index == 1


def test_strict_coverage_rejects_unseen_workload(tmp_path, task):
    path = tmp_path / "tuning.log"
    path.write_text(record(task, 0))
    missing = task.workload[:-2] + ("different_layout", task.workload[-1])
    with pytest.raises(RuntimeError, match="Incomplete"):
        with audited_history(str(path), require_tuned=True):
            autotvm.task.DispatchContext.current.query(task.target, missing)


def test_candidate_neighbourhood_keeps_baseline(task):
    baseline = task.config_space.get(0).to_json_dict()
    indices = local_candidates(task, baseline, limit=8)
    manifest = candidate_manifest(task, indices, 0)
    assert indices[0] == 0 and len(indices) <= 8
    assert manifest[0]["is_incumbent"] is True
    assert all(len(row["changed_axes"]) <= 1 for row in manifest)
    assert len({axis for row in manifest for axis in row["changed_axes"]}) >= 3
    assert task.config_space.is_index_valid(0)


def test_projection_repair_has_bounded_aspect_ratio_candidates():
    register_vta_conv2d_template()
    env = vta.get_env()
    a = te.placeholder((1, 4, 56, 56, env.BATCH, env.BLOCK_IN), dtype="int8")
    w = te.placeholder((8, 4, 1, 1, env.BLOCK_OUT, env.BLOCK_IN), dtype="int8")
    task = autotvm.task.create("conv2d_packed.vta", args=(a, w, (2, 2), (0, 0, 0, 0),
        (1, 1), "NCHW%dn%dc" % (env.BATCH, env.BLOCK_IN), "int32"),
        target=env.target, target_host=env.target_host)
    indices = projection_repair_candidates(task)
    assert len(indices) == 2 and len(task.config_space) == 2
    configs = [task.config_space.get(i) for i in indices]
    assert {c["tile_h"].size[-1] for c in configs} == {1, 7}
    assert all(c["tile_w"].size[-1] == 28 for c in configs)


def test_build_failure_uses_two_field_autotvm_error_format(tmp_path, task, monkeypatch):
    def fail(_):
        raise RuntimeError("invalid tile")
    monkeypatch.setattr("vta_autotvm_measure.build_config_module", fail)
    inp = autotvm.measure.MeasureInput(task.target, task, task.config_space.get(0))
    result = VTASequentialBuilder(tmp_path).build([inp])[0]
    assert result.error_no == autotvm.measure.MeasureErrorNo.COMPILE_HOST
    traceback_text, message = result.costs
    assert "invalid tile" in traceback_text and message


def test_native_cache_key_tracks_history_contents(tmp_path, task):
    from deploy_classification_stage_pipeline_native import stage_build_cache_key
    path = tmp_path / "tuning.log"
    args = SimpleNamespace(tune_log=str(path), require_tuned_vta=True, image_size=224)
    stage = {"device": "vta", "unit_names": ["conv"]}
    path.write_text(record(task, 0))
    first, _ = stage_build_cache_key(args, vta.get_env(), stage, [], [], False)
    path.write_text(record(task, 1))
    second, _ = stage_build_cache_key(args, vta.get_env(), stage, [], [], False)
    assert first != second


def test_validated_baseline_prefers_passing_original_config(tmp_path, task):
    raw = tmp_path / "raw.log"
    raw.write_text(record(task, 0, 0.5) + record(task, 1, 0.01))
    output = tmp_path / "baseline.log"
    selected = [{"workload": task.workload, "selected_indices": [0, 1]}]
    manifest = write_validated_baseline(output, selected, [], [raw])
    records = list(autotvm.record.load_from_file(str(output)))
    assert manifest[0]["origin"] == "dispatched_incumbent"
    assert len(records) == 1 and records[0][0].config.index == 0


def test_validated_baseline_uses_declared_repair_if_original_failed(tmp_path, task):
    raw = tmp_path / "raw.log"
    raw.write_text(record(task, 1, 0.02))
    output = tmp_path / "baseline.log"
    selected = [{"workload": task.workload, "selected_indices": [0]}]
    repairs = [{"workload": task.workload, "selected_indices": [1]}]
    manifest = write_validated_baseline(output, selected, repairs, [raw])
    records = list(autotvm.record.load_from_file(str(output)))
    assert manifest[0]["origin"] == "correctness_repair"
    assert records[0][0].config.index == 1


def test_safe_overlay_only_replaces_passing_incumbent_with_clear_win(tmp_path, task):
    raw = tmp_path / "raw.log"
    raw.write_text(record(task, 0, 0.10) + record(task, 1, 0.05))
    output = tmp_path / "overlay.log"
    selected = [{"workload": task.workload, "incumbent_index": 0,
                 "selected_indices": [0, 1]}]
    manifest = write_safe_overlay(output, selected, [raw], min_improvement=0.02)
    assert manifest[0]["overridden"] is True
    assert list(autotvm.record.load_from_file(str(output)))[0][0].config.index == 1


def test_safe_overlay_keeps_tophub_if_direct_incumbent_failed(tmp_path, task):
    raw = tmp_path / "raw.log"
    raw.write_text(record(task, 1, 0.01))
    output = tmp_path / "overlay.log"
    selected = [{"workload": task.workload, "incumbent_index": 0,
                 "selected_indices": [0, 1]}]
    manifest = write_safe_overlay(output, selected, [raw])
    assert manifest[0]["overridden"] is False
    assert not output.exists()


def test_reference_1x1_matches_independent_matrix_product(task):
    data, weights, result = reference_data(task)
    # For this one-block 1x1 convolution each spatial vector is a matrix product.
    expected = np.matmul(data[:, 0].astype("int32"), weights[0, 0, 0, 0].astype("int32").T)
    expected = np.clip(expected >> 8, 0, 127).astype("int8")
    np.testing.assert_array_equal(result[:, 0], expected)


@pytest.mark.parametrize("kernel_size,stride,pad", [(1, 2, 0), (3, 1, 1), (3, 2, 1)])
def test_reference_matches_template_on_llvm(kernel_size, stride, pad):
    register_vta_conv2d_template()
    env = vta.get_env()
    a = te.placeholder((1, 4, 14, 14, env.BATCH, env.BLOCK_IN), dtype="int8")
    w = te.placeholder((8, 4, kernel_size, kernel_size, env.BLOCK_OUT, env.BLOCK_IN), dtype="int8")
    task = autotvm.task.create("conv2d_packed.vta", args=(a, w, (stride, stride),
        (pad, pad, pad, pad), (1, 1), "NCHW%dn%dc" % (env.BATCH, env.BLOCK_IN), "int32"),
        target="llvm")
    with task.target:
        schedule, tensors = task.instantiate(task.config_space.get(0))
    module = tvm.build(schedule, tensors, target="llvm")
    data, weights, expected = reference_data(task)
    output = tvm.nd.empty(expected.shape, "int8")
    module(tvm.nd.array(data), tvm.nd.array(weights), output)
    np.testing.assert_array_equal(output.numpy(), expected)


def test_rerank_preserves_cpu_bottleneck_and_does_not_add_dma_twice():
    stage = {"device": "vta", "unit_names": ["conv"]}
    row = {"candidate_id": "a", "topology_id": "a", "rank": 1,
           "scheme_cfg": [stage], "stages": [],
           "cost_provenance": {"vta_dma_qualification": {"load_bandwidth_GBps": 1, "store_bandwidth_GBps": 1}},
           "resource_load_ms": {"max_cpu_stage": 20, "cpu_physical_pool_lower_bound": 5,
                                "cpu_prefix_mask_lower_bound": 6, "vta_mutex_boundary_work_total": 1}}
    measurements = {"conv": {"tuned": {"median_ms": 10, "runtime_profile": {
        "load_buffer_2d_bytes": 8_000_000, "store_buffer_2d_bytes": 1_000_000}}}}
    result = rerank([row], measurements, "tuned")[0]
    assert result["score_ms"] == 20
    assert result["components_ms"]["vta_and_boundary"] == 11
    assert result["components_ms"]["ddr"] == 9


def test_resnet_workload_occurrences_follow_semantic_units():
    projection = ["conv2d_packed.vta", ["TENSOR", [1, 4, 56, 56, 1, 16], "int8"],
                  ["TENSOR", [8, 4, 1, 1, 16, 16], "int8"], [2, 2]]
    repeated = ["conv2d_packed.vta", ["TENSOR", [1, 8, 28, 28, 1, 16], "int8"],
                ["TENSOR", [8, 8, 3, 3, 16, 16], "int8"], [1, 1]]
    units = ["layer2_block0_main_preadd", "layer2_block0_skip_proj",
             "layer2_block1_main_preadd"]
    assert occurrence_count(projection, units) == 1
    assert occurrence_count(repeated, units) == 3
