"""Local-only checks for the isolated VTA residency schedule prototype."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tvm
from tvm import autotvm, rpc
from tvm.contrib import utils
import vta

import vta.top.vta_conv2d_residency  # pylint: disable=unused-import
from extract_static_vta_dma import extract_module_dma
from tune_resnet18_vta import register_vta_conv2d_template
from vta_autotvm_measure import reference_data


BASE = Path(__file__).resolve().parent / "report_out" / "stage_tile_cotuning"
SELECTED = json.loads(
    (BASE / "iteration6_safe_overlay_final" / "selected_tasks.json").read_text()
)

REPRESENTATIVES = {
    # The experimental configs copy the frozen tile geometry where it is legal, force
    # both virtual-thread knobs to one, and are never substituted for the incumbents.
    0: {"name": "W00", "config_index": 252},
    2: {"name": "W02", "config_index": 177},
    9: {"name": "W09", "config_index": 83},
}

MODES = {
    0: "original",
    1: "input_stationary",
    2: "weight_stationary",
    3: "paper_inspired_hybrid",
}

FROZEN_ORIGINAL_TIR_SHA256 = {
    0: "597666fdc6b1f366edd557e357ada30872bfddf1ea67eec47dec4e9c07be4bc1",
    2: "1067810187559ffb7943426d76991d9cbd5791a2058643579198397c6924b047",
    9: "483ed2916b9aef8d68fbf510ae740ce952d93a92a21b2b8676dd9ad68ccbdd08",
}

EXPECTED_DMA = {
    0: {
        0: (112, 487424, 258048, 14),
        1: (56, 243712, 258048, 14),
        2: (112, 487424, 258048, 14),
        3: (56, 243712, 258048, 14),
    },
    2: {
        0: (256, 487424, 589824, 16),
        1: (64, 121856, 589824, 16),
        2: (256, 487424, 589824, 16),
        3: (128, 243712, 589824, 16),
    },
    9: {
        0: (64, 86528, 131072, 2),
        1: (32, 43264, 131072, 2),
        2: (64, 86528, 131072, 2),
        3: (32, 43264, 131072, 2),
    },
}


def _original_task(workload_index, env):
    register_vta_conv2d_template()
    item = SELECTED[workload_index]
    return autotvm.task.create(
        item["workload"][0],
        args=item["workload"][1:],
        target=env.target,
        target_host=env.target_host,
    )


def _residency_task(workload_index, mode, env):
    item = SELECTED[workload_index]
    return autotvm.task.create(
        "conv2d_packed_residency.vta",
        args=tuple(item["workload"][1:]) + (mode,),
        target=env.target,
        target_host=env.target_host,
    )


def _lower(task, config):
    with task.target:
        schedule, tensors = task.instantiate(config)
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        module = tvm.lower(schedule, tensors, name="main")
    return module, tensors


def _dma_tuple(module, env):
    totals = extract_module_dma(module, env)["totals"]
    return (
        totals.get("load_buffer_2d_calls", 0),
        totals.get("load_buffer_2d_inp_bytes", 0),
        totals.get("load_buffer_2d_wgt_bytes", 0),
        totals.get("store_buffer_2d_calls", 0),
    )


@pytest.mark.parametrize("workload_index", REPRESENTATIVES)
def test_original_schedule_keeps_frozen_tir(workload_index):
    """The refactor must preserve the exact frozen original schedule."""
    env = vta.get_env()
    item = SELECTED[workload_index]
    task = _original_task(workload_index, env)
    module, _ = _lower(task, task.config_space.get(item["incumbent_index"]))
    assert hashlib.sha256(module.script().encode()).hexdigest() == FROZEN_ORIGINAL_TIR_SHA256[
        workload_index
    ]


@pytest.mark.parametrize("workload_index", REPRESENTATIVES)
@pytest.mark.parametrize("mode", MODES)
def test_residency_modes_lower_with_expected_dma(workload_index, mode):
    env = vta.get_env()
    task = _residency_task(workload_index, mode, env)
    config = task.config_space.get(REPRESENTATIVES[workload_index]["config_index"])
    assert config["oc_nthread"].val == 1
    assert config["h_nthread"].val == 1
    module, _ = _lower(task, config)
    assert _dma_tuple(module, env) == EXPECTED_DMA[workload_index][mode]


def test_safe_residency_modes_preserve_the_verified_dma_boundary():
    env = vta.get_env()
    for workload_index, representative in REPRESENTATIVES.items():
        hashes = {}
        for mode in MODES:
            task = _residency_task(workload_index, mode, env)
            module, _ = _lower(task, task.config_space.get(representative["config_index"]))
            hashes[mode] = hashlib.sha256(module.script().encode()).hexdigest()
        assert hashes[0] != hashes[1]

        original = EXPECTED_DMA[workload_index][0]
        input_stationary = EXPECTED_DMA[workload_index][1]
        assert input_stationary[1] <= original[1] * 0.8

        # Keeping the weight cache inside the VTA task is the current verified safety
        # boundary.  It avoids an unsupported STORE->LOAD dependency, but therefore
        # does not yet reduce weight traffic on these conservative representatives.
        assert EXPECTED_DMA[workload_index][2] == original
        hybrid = EXPECTED_DMA[workload_index][3]
        assert hybrid[1] < original[1]
        assert hybrid[2] == original[2]


def test_p7r_w04_vthread_joint_capacity_and_dma_certificates():
    """P7R keeps only joint candidates that lower and reduce real logical DMA."""
    env = vta.get_env()
    workload_index = 4

    # The protected TopHub tile is legal on the original schedule, but extending
    # all-output-channel accumulator lifetime for input residency exceeds the
    # configured 128-KiB accumulator SRAM after virtual-thread lowering.
    original_task = _residency_task(workload_index, 0, env)
    original_config = original_task.config_space.get(463)
    assert original_config["oc_nthread"].val == 2
    _lower(original_task, original_config)

    residency_task = _residency_task(workload_index, 1, env)
    with pytest.raises(tvm.error.InternalError, match="Allocation exceed bound"):
        _lower(residency_task, residency_task.config_space.get(463))

    # A smaller spatial tile is a legal joint candidate.  Its input traffic is
    # halved relative to the exact same tile under the original schedule.
    legal_index = 455
    legal_config = residency_task.config_space.get(legal_index)
    assert legal_config["tile_h"].size[-1] == 14
    assert legal_config["tile_w"].size[-1] == 2
    assert legal_config["tile_co"].size[-1] == 4
    assert legal_config["oc_nthread"].val == 2
    original_module, _ = _lower(original_task, original_task.config_space.get(legal_index))
    residency_module, _ = _lower(residency_task, legal_config)
    original_dma = _dma_tuple(original_module, env)
    residency_dma = _dma_tuple(residency_module, env)
    assert residency_dma[1] * 2 == original_dma[1]
    assert residency_dma[0] < original_dma[0]


def test_p7r_only_opens_bounded_hybrid_oc_vthread_combination():
    env = vta.get_env()
    weight_task = _residency_task(4, 2, env)
    with pytest.raises(ValueError, match="Only input-stationary and bounded-hybrid"):
        with weight_task.target:
            weight_task.instantiate(weight_task.config_space.get(455))

    hybrid_task = _residency_task(4, 3, env)
    hybrid_config = hybrid_task.config_space.get(455)
    assert hybrid_config["oc_nthread"].val == 2
    _lower(hybrid_task, hybrid_config)


@pytest.mark.parametrize("workload_index", REPRESENTATIVES)
@pytest.mark.parametrize("mode", MODES)
def test_residency_modes_match_reference_in_fsim(workload_index, mode):
    """Run only under an explicitly selected local simulator environment."""
    env = vta.get_env()
    if env.TARGET != "sim":
        pytest.skip("Set VTA_HW_PATH to a local fsim config; this test never uses board RPC")
    from vta.testing import simulator

    assert simulator.enabled()
    original_task = _original_task(workload_index, env)
    data, weight, expected = reference_data(original_task, seed=0)

    task = _residency_task(workload_index, mode, env)
    config = task.config_space.get(REPRESENTATIVES[workload_index]["config_index"])
    with task.target:
        schedule, tensors = task.instantiate(config)
    with vta.build_config(disabled_pass={"tir.CommonSubexprElimTIR"}):
        module = vta.build(
            schedule,
            tensors,
            target=tvm.target.Target(env.target, host=env.target_host),
            name="main",
        )

    module_name = "{}_mode{}.o".format(REPRESENTATIVES[workload_index]["name"], mode)
    temporary = utils.tempdir()
    module.save(temporary.relpath(module_name))
    remote = rpc.LocalSession()
    remote.upload(temporary.relpath(module_name))
    function = remote.load_module(module_name)
    device = remote.ext_dev(0)
    output = tvm.nd.empty(expected.shape, "int8", device=device)
    simulator.clear_stats()
    function(tvm.nd.array(data, device), tvm.nd.array(weight, device), output)
    np.testing.assert_array_equal(output.numpy(), expected)


if __name__ == "__main__":
    pytest.main([__file__])
