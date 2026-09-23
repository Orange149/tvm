"""Regression checks for the O2-compatible two-vthread weight-residency prototype."""

from __future__ import annotations

import copy
import ctypes
import json
from pathlib import Path

import numpy as np
import pytest
import tvm
from tvm import autotvm
from tvm.autotvm.task.space import ConfigEntity
import vta
from vta.top.vta_staged_weight import staged_weight_build_config

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from c3_reuse_semantics import capture_lower, extract_requests
from c3_thread_sync_audit import final_tir_audit
from qualify_vta_residency_fsim import reference_data_from_workload
from run_vta_c3_resnet18_local_qualification import dependency_audit, sync_multiplicity
from tune_resnet18_vta import register_vta_conv2d_template


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREPARED = (
    HERE
    / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
    / "20260922_c3_thread_sync_prepare_run07/candidates.json"
)
Y00_POOL = (
    HERE
    / "report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout"
    / "20260912_p7r127_y00_v2_local_pool_run01/candidates_v2.jsonl"
)
Y00_CANDIDATE_ID = "423345af4098fd1b9f901f8b46e6f5a141d3cd187b6fce2b6951439c586a12a5"
Y00_F06_INPUT_ID = "7b958e8562ed5e04fd14a52e940aa7aa5518ba600d8284c41d5ecd0d9bf18e3c"
SEEDS = (0, 20250901, 20260910)


def _baseline():
    rows = json.loads(PREPARED.read_text())
    return next(row for row in rows if row["candidate_id"] == "T8_A")


def _y00_baseline():
    rows = [json.loads(line) for line in Y00_POOL.read_text().splitlines() if line.strip()]
    return next(row for row in rows if row["candidate_id"] == Y00_CANDIDATE_ID)


def _y00_f06_input():
    rows = [json.loads(line) for line in Y00_POOL.read_text().splitlines() if line.strip()]
    return next(row for row in rows if row["candidate_id"] == Y00_F06_INPUT_ID)


def _two_thread_config(row):
    config = copy.deepcopy(row["identity"]["complete_config_entity"])
    config.setdefault("index", -1)
    for item in config["entity"]:
        if item[0] == "oc_nthread":
            item[2] = 2
    return config


def _instantiate_row(row, mode, config_dict=None):
    env = vta.get_env()
    workload = row["identity"]["workload"]
    if mode == 0:
        register_vta_conv2d_template()
        name = workload[0]
        args = tuple(workload[1:])
    else:
        name = "conv2d_packed_residency.vta"
        args = tuple(workload[1:]) + (mode,)
    task = autotvm.task.create(name, args=args, target=env.target, target_host=env.target_host)
    entity = ConfigEntity.from_json_dict(config_dict or row["complete_config_entity"])
    with task.target:
        return task.instantiate(entity)


def _instantiate(mode, config_dict=None):
    row = _baseline()
    return _instantiate_row(row, mode, config_dict or row["complete_config_entity"])


def _instantiate_y00(mode):
    row = _y00_baseline()
    return _instantiate_row(row, mode, _two_thread_config(row))


def _instantiate_y00_f06(mode):
    row = _y00_f06_input()
    config = {"index": -1, "code_hash": None, **row["identity"]["complete_config_entity"]}
    return _instantiate_row(row, mode, config)


def _lower(mode):
    schedule, tensors = _instantiate(mode)
    build_config_factory = staged_weight_build_config if mode == 8 else None
    module, storage, arenas = capture_lower(
        schedule, tensors, build_config_factory=build_config_factory
    )
    semantics = extract_requests(module, vta.get_env())
    return schedule, tensors, module, storage, arenas, semantics


def _lower_y00(mode):
    schedule, tensors = _instantiate_y00(mode)
    build_config_factory = staged_weight_build_config if mode == 8 else None
    module, storage, arenas = capture_lower(
        schedule, tensors, build_config_factory=build_config_factory
    )
    semantics = extract_requests(module, vta.get_env())
    return schedule, tensors, module, storage, arenas, semantics


def _lower_y00_f06(mode):
    schedule, tensors = _instantiate_y00_f06(mode)
    build_config_factory = staged_weight_build_config if mode == 9 else None
    module, storage, arenas = capture_lower(
        schedule, tensors, build_config_factory=build_config_factory
    )
    semantics = extract_requests(module, vta.get_env())
    return schedule, tensors, module, storage, arenas, semantics


def test_two_vthread_reload_and_resident_static_contract():
    """The matched pair keeps compute fixed while changing weight lifetime."""
    results = {}
    for mode in (0, 6, 7):
        schedule, tensors, module, storage, arenas, semantics = _lower(mode)
        del schedule, tensors, storage
        dependencies = dependency_audit(module)
        assert dependencies["balanced"]
        assert dependencies["forbidden_direct_1_3_count"] == 0
        results[mode] = {
            "module": module,
            "arenas": arenas["scope_arena_span_sum_upper_bound_bytes"],
            "semantics": semantics,
            "sync": sync_multiplicity(module),
            "address": final_tir_audit(module, semantics),
        }

    original, reload, resident = (results[mode] for mode in (0, 6, 7))
    assert original["arenas"]["local.wgt_buffer"] == 9216
    assert reload["arenas"] == resident["arenas"] == {
        "local.acc_buffer": 114688,
        "local.inp_buffer": 18560,
        "local.wgt_buffer": 36864,
    }

    def total(row, key):
        return row["semantics"]["totals"][key]

    for key in (
        "load_buffer_2d_inp_bytes",
        "load_buffer_2d_inp_calls",
        "store_buffer_2d_out_bytes",
        "store_buffer_2d_out_calls",
    ):
        assert total(original, key) == total(reload, key) == total(resident, key)

    assert total(original, "load_buffer_2d_wgt_bytes") == 258048
    assert total(reload, "load_buffer_2d_wgt_bytes") == 258048
    assert total(resident, "load_buffer_2d_wgt_bytes") == 36864
    assert total(original, "load_buffer_2d_wgt_calls") == 56
    assert total(reload, "load_buffer_2d_wgt_calls") == 14
    assert total(resident, "load_buffer_2d_wgt_calls") == 2
    assert reload["sync"] == [7, 7, 1]
    assert resident["sync"] == [1]

    # The reload/resident pair has identical GEMM/ALU computation and legal
    # final addresses.  Weight LOAD placement and required drain structure are
    # intentionally part of the program-level mechanism difference.
    assert reload["address"]["compute_sha256"] == resident["address"]["compute_sha256"]
    assert reload["address"]["max_kernel_uops"] == resident["address"]["max_kernel_uops"] == 48
    assert reload["address"]["uop_effective_occupancy_peak"] is None
    assert resident["address"]["uop_effective_occupancy_peak"] is None


@pytest.mark.parametrize("mode", (6, 7, 8))
def test_two_vthread_modes_fail_closed_for_one_thread_entity(mode):
    config = copy.deepcopy(_baseline()["complete_config_entity"])
    for item in config["entity"]:
        if item[0] == "oc_nthread":
            item[2] = 1
    with pytest.raises(ValueError, match="require oc_nthread = 2"):
        _instantiate(mode, config)


@pytest.mark.parametrize("mode", (6, 7, 8))
def test_two_vthread_modes_match_reference_for_three_seeds(mode):
    """Also guards the runtime's one-int cached-uop-signature constraint."""
    backend = ROOT / "build/libvta_fsim.so"
    if not backend.is_file():
        pytest.skip("local FSim backend is unavailable")
    ctypes.CDLL(str(backend), mode=ctypes.RTLD_GLOBAL)
    schedule, tensors = _instantiate(mode)
    build_config_factory = staged_weight_build_config if mode == 8 else vta.build_config
    with build_config_factory(disabled_pass={"tir.CommonSubexprElimTIR"}):
        module = vta.build(
            schedule,
            tensors,
            target=tvm.target.Target(vta.get_env().target, host="llvm"),
            name="main",
        )
    function = module["main"]
    device = tvm.device("ext_dev", 0)
    workload = _baseline()["identity"]["workload"]
    for seed in SEEDS:
        data, weight, expected = reference_data_from_workload(workload, seed)
        output = tvm.nd.empty(expected.shape, "int8", device=device)
        function(tvm.nd.array(data, device), tvm.nd.array(weight, device), output)
        np.testing.assert_array_equal(output.numpy(), expected)


def test_staged_resident_static_contract_and_first_use_order():
    """P4-S loads one reduction group at first use and never adds a drain."""
    _, _, module, _, arenas, semantics = _lower(8)
    dependencies = dependency_audit(module)
    address = final_tir_audit(module, semantics)
    totals = semantics["totals"]

    assert dependencies["balanced"]
    assert dependencies["forbidden_direct_1_3_count"] == 0
    assert arenas["scope_arena_span_sum_upper_bound_bytes"] == {
        "local.acc_buffer": 114688,
        "local.inp_buffer": 18560,
        "local.wgt_buffer": 36864,
    }
    assert totals["load_buffer_2d_wgt_bytes"] == 36864
    assert totals["load_buffer_2d_wgt_calls"] == 8
    assert semantics["unique_source_region_bytes"]["load_wgt"] == 36864
    assert semantics["reload_ratios"]["load_wgt"] == 1
    assert sync_multiplicity(module) == [1]
    assert address["status"] == "passed"
    assert address["max_kernel_uops"] == 48
    assert address["uop_effective_occupancy_peak"] is None

    weight_groups = [
        row
        for row in semantics["request_groups"]
        if row["direction"] == "load" and row["memory_name"] == "wgt"
    ]
    assert len(weight_groups) == 8
    assert all(row["bytes_per_request"] == 4608 for row in weight_groups)
    assert sorted(row["sram_offset_elements"] for row in weight_groups) == [
        0,
        18,
        36,
        54,
        72,
        90,
        108,
        126,
    ]
    weight_sites = [
        site
        for site in semantics["load_store_sites"]
        if any("i2_outer" in condition["expr"] for condition in site["conditions"])
    ]
    assert len(weight_sites) == 8
    assert all(site["executed_calls"] == 1 for site in weight_sites)
    assert all(
        any(
            "i2_outer" in condition["expr"] and condition["expected"]
            for condition in site["conditions"]
        )
        for site in weight_sites
    )

    # Reset GEMMs do not consume weights.  Before the first effective MAC,
    # exactly the two cthread copies of reduction group k0 have been loaded.
    weight_loads_before_first_mac = 0
    first_mac_seen = False
    for event in address["ordered_events"]:
        if event[0] == "VTALoadBuffer2D" and event[1][-1] == 1:
            weight_loads_before_first_mac += 1
        if event[0] == "kernel" and any(
            uop["args"][0] == 0 and uop["args"][1] == 0 for uop in event[1]
        ):
            first_mac_seen = True
            break
    assert first_mac_seen
    assert weight_loads_before_first_mac == 2


def test_y00_staged_resident_is_a_matched_memory_pressure_pair():
    """Y00 changes only WGT requests while preserving compute and one submission."""
    results = {}
    for mode in (0, 8):
        _, _, module, _, arenas, semantics = _lower_y00(mode)
        dependencies = dependency_audit(module)
        address = final_tir_audit(module, semantics)
        assert dependencies["balanced"]
        assert dependencies["forbidden_direct_1_3_count"] == 0
        assert address["status"] == "passed"
        results[mode] = {
            "arenas": arenas["scope_arena_span_sum_upper_bound_bytes"],
            "semantics": semantics,
            "sync": sync_multiplicity(module),
            "address": address,
        }

    original, resident = (results[mode] for mode in (0, 8))
    expected_arenas = {
        "local.acc_buffer": 13312,
        "local.inp_buffer": 5376,
        "local.wgt_buffer": 4608,
    }
    assert original["arenas"] == resident["arenas"] == expected_arenas
    for key in (
        "load_buffer_2d_inp_bytes",
        "load_buffer_2d_inp_calls",
        "store_buffer_2d_out_bytes",
        "store_buffer_2d_out_calls",
    ):
        assert original["semantics"]["totals"][key] == resident["semantics"]["totals"][key]
    assert original["semantics"]["totals"]["load_buffer_2d_wgt_bytes"] == 1916928
    assert resident["semantics"]["totals"]["load_buffer_2d_wgt_bytes"] == 4608
    assert original["semantics"]["totals"]["load_buffer_2d_wgt_calls"] == 832
    assert resident["semantics"]["totals"]["load_buffer_2d_wgt_calls"] == 2
    assert original["sync"] == resident["sync"] == [1]
    assert original["address"]["compute_sha256"] == resident["address"]["compute_sha256"]
    assert original["address"]["max_kernel_uops"] == resident["address"]["max_kernel_uops"] == 12
    assert resident["semantics"]["unique_source_region_bytes"]["load_wgt"] == 4608
    assert resident["semantics"]["reload_ratios"]["load_wgt"] == 1

    weight_groups = [
        row
        for row in resident["semantics"]["request_groups"]
        if row["direction"] == "load" and row["memory_name"] == "wgt"
    ]
    assert len(weight_groups) == 2
    assert all(row["bytes_per_request"] == 2304 for row in weight_groups)
    assert sorted(row["sram_offset_elements"] for row in weight_groups) == [0, 9]


def test_y00_staged_resident_matches_reference_for_three_seeds():
    backend = ROOT / "build/libvta_fsim.so"
    if not backend.is_file():
        pytest.skip("local FSim backend is unavailable")
    ctypes.CDLL(str(backend), mode=ctypes.RTLD_GLOBAL)
    schedule, tensors = _instantiate_y00(8)
    with staged_weight_build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        module = vta.build(
            schedule,
            tensors,
            target=tvm.target.Target(vta.get_env().target, host="llvm"),
            name="main",
        )
    function = module["main"]
    device = tvm.device("ext_dev", 0)
    workload = _y00_baseline()["identity"]["workload"]
    for seed in SEEDS:
        data, weight, expected = reference_data_from_workload(workload, seed)
        output = tvm.nd.empty(expected.shape, "int8", device=device)
        function(tvm.nd.array(data, device), tvm.nd.array(weight, device), output)
        np.testing.assert_array_equal(output.numpy(), expected)


def test_y00_f06_input_weight_staged_is_a_matched_incremental_pair():
    results = {}
    for mode in (1, 9):
        _, _, module, _, arenas, semantics = _lower_y00_f06(mode)
        dependencies = dependency_audit(module)
        address = final_tir_audit(module, semantics)
        assert dependencies["balanced"]
        assert dependencies["forbidden_direct_1_3_count"] == 0
        assert address["status"] == "passed"
        results[mode] = {
            "arenas": arenas["scope_arena_span_sum_upper_bound_bytes"],
            "semantics": semantics,
            "sync": sync_multiplicity(module),
            "address": address,
        }
    input_only, combined = (results[mode] for mode in (1, 9))
    expected_arenas = {
        "local.acc_buffer": 106496,
        "local.inp_buffer": 15552,
        "local.wgt_buffer": 4608,
    }
    assert input_only["arenas"] == combined["arenas"] == expected_arenas
    for key in (
        "load_buffer_2d_inp_bytes",
        "load_buffer_2d_inp_calls",
        "store_buffer_2d_out_bytes",
        "store_buffer_2d_out_calls",
    ):
        assert input_only["semantics"]["totals"][key] == combined["semantics"]["totals"][key]
    assert input_only["semantics"]["totals"]["load_buffer_2d_wgt_bytes"] == 239616
    assert combined["semantics"]["totals"]["load_buffer_2d_wgt_bytes"] == 4608
    assert input_only["semantics"]["totals"]["load_buffer_2d_wgt_calls"] == 52
    assert combined["semantics"]["totals"]["load_buffer_2d_wgt_calls"] == 1
    assert input_only["sync"] == combined["sync"] == [1]
    assert input_only["address"]["compute_sha256"] == combined["address"]["compute_sha256"]
    assert input_only["address"]["max_kernel_uops"] == combined["address"]["max_kernel_uops"] == 96
    assert combined["semantics"]["reload_ratios"]["load_wgt"] == 1


def test_y00_f06_input_weight_staged_matches_reference_for_three_seeds():
    backend = ROOT / "build/libvta_fsim.so"
    if not backend.is_file():
        pytest.skip("local FSim backend is unavailable")
    ctypes.CDLL(str(backend), mode=ctypes.RTLD_GLOBAL)
    schedule, tensors = _instantiate_y00_f06(9)
    with staged_weight_build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        module = vta.build(
            schedule,
            tensors,
            target=tvm.target.Target(vta.get_env().target, host="llvm"),
            name="main",
        )
    function = module["main"]
    device = tvm.device("ext_dev", 0)
    workload = _y00_f06_input()["identity"]["workload"]
    for seed in SEEDS:
        data, weight, expected = reference_data_from_workload(workload, seed)
        output = tvm.nd.empty(expected.shape, "int8", device=device)
        function(tvm.nd.array(data, device), tvm.nd.array(weight, device), output)
        np.testing.assert_array_equal(output.numpy(), expected)


def test_y00_f06_combined_mode_rejects_two_output_vthreads():
    row = _y00_f06_input()
    config = _two_thread_config(row)
    with pytest.raises(ValueError, match="requires oc_nthread = 1"):
        _instantiate_row(row, 9, config)


def test_staged_resident_reuses_handles_across_distinct_weights():
    """A second call must not accidentally reuse the first call's resident SRAM."""
    backend = ROOT / "build/libvta_fsim.so"
    if not backend.is_file():
        pytest.skip("local FSim backend is unavailable")
    ctypes.CDLL(str(backend), mode=ctypes.RTLD_GLOBAL)
    schedule, tensors = _instantiate(8)
    with staged_weight_build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        module = vta.build(
            schedule,
            tensors,
            target=tvm.target.Target(vta.get_env().target, host="llvm"),
            name="main",
        )
    function = module["main"]
    device = tvm.device("ext_dev", 0)
    workload = _baseline()["identity"]["workload"]
    first_data, first_weight, first_expected = reference_data_from_workload(workload, SEEDS[0])
    data_handle = tvm.nd.array(first_data, device)
    weight_handle = tvm.nd.array(first_weight, device)
    output_handle = tvm.nd.array(np.full(first_expected.shape, -113, dtype="int8"), device)
    observed_hashes = []
    for seed in SEEDS:
        data, weight, expected = reference_data_from_workload(workload, seed)
        data_handle.copyfrom(data)
        weight_handle.copyfrom(weight)
        output_handle.copyfrom(np.full(expected.shape, -113, dtype="int8"))
        function(data_handle, weight_handle, output_handle)
        observed = output_handle.numpy()
        np.testing.assert_array_equal(observed, expected)
        observed_hashes.append(hash(observed.tobytes()))
    assert len(set(observed_hashes)) == len(SEEDS)


if __name__ == "__main__":
    pytest.main([__file__])
