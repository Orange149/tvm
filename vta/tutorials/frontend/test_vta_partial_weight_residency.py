"""Local qualification for R18-H1 T8 P1/P2 partial weight residency."""

import ctypes

import numpy as np
import pytest
import tvm
import vta

from c3_reuse_semantics import capture_lower, extract_requests
from c3_thread_sync_audit import final_tir_audit
from qualify_vta_residency_fsim import reference_data_from_workload
from run_vta_c3_resnet18_local_qualification import dependency_audit, sync_multiplicity
from test_vta_vthread_weight_residency import ROOT, SEEDS, _baseline, _instantiate
from vta.top.vta_partial_weight import partial_weight_build_config


def _build_config(resident_groups):
    return lambda **kwargs: partial_weight_build_config(resident_groups, **kwargs)


def _lower(resident_groups):
    schedule, tensors = _instantiate(8)
    module, storage, arenas = capture_lower(
        schedule,
        tensors,
        build_config_factory=_build_config(resident_groups),
    )
    del schedule, tensors, storage
    semantics = extract_requests(module, vta.get_env())
    return module, arenas, semantics


@pytest.mark.parametrize(
    "resident_groups,wgt_arena,wgt_bytes,wgt_calls",
    (
        (1, 18432, 202752, 44),
        (2, 27648, 147456, 32),
    ),
)
def test_partial_residency_static_contract(
    resident_groups, wgt_arena, wgt_bytes, wgt_calls
):
    module, arenas, semantics = _lower(resident_groups)
    dependencies = dependency_audit(module)
    address = final_tir_audit(module, semantics)
    totals = semantics["totals"]

    assert dependencies["balanced"]
    assert dependencies["forbidden_direct_1_3_count"] == 0
    assert arenas["scope_arena_span_sum_upper_bound_bytes"] == {
        "local.acc_buffer": 114688,
        "local.inp_buffer": 18560,
        "local.wgt_buffer": wgt_arena,
    }
    assert totals["load_buffer_2d_wgt_bytes"] == wgt_bytes
    assert totals["load_buffer_2d_wgt_calls"] == wgt_calls
    assert totals["load_buffer_2d_inp_bytes"] == 487424
    assert totals["load_buffer_2d_inp_calls"] == 56
    assert totals["store_buffer_2d_out_bytes"] == 200704
    assert totals["store_buffer_2d_out_calls"] == 14
    assert semantics["unique_source_region_bytes"]["load_wgt"] == 36864
    assert sync_multiplicity(module) == [1]
    assert address["status"] == "passed"
    assert address["uop_effective_occupancy_peak"] is None

    weight_groups = [
        row
        for row in semantics["request_groups"]
        if row["direction"] == "load" and row["memory_name"] == "wgt"
    ]
    assert sum(row["multiplicity"] for row in weight_groups) == wgt_calls
    assert all(row["bytes_per_request"] == 4608 for row in weight_groups)


@pytest.mark.parametrize("resident_groups", (1, 2))
def test_partial_residency_matches_reference_and_reloads_each_invocation(resident_groups):
    backend = ROOT / "build/libvta_fsim.so"
    if not backend.is_file():
        pytest.skip("local FSim backend is unavailable")
    ctypes.CDLL(str(backend), mode=ctypes.RTLD_GLOBAL)
    schedule, tensors = _instantiate(8)
    with partial_weight_build_config(
        resident_groups, disabled_pass={"tir.CommonSubexprElimTIR"}
    ):
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


def test_partial_residency_rejects_unsupported_group_counts():
    for groups in (0, 3, 4):
        with pytest.raises(ValueError, match="exactly P1 or P2"):
            partial_weight_build_config(groups)
