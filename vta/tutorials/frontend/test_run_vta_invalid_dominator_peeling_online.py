import run_vta_invalid_dominator_peeling_online as target
import run_vta_resnet50_fused_program_pareto_board as board_target

import numpy as np


def program(name, b, n, s=0):
    return {"candidate_id": name, "dma_bytes": b, "dma_calls": n, "extra_submissions": s}


def test_invalid_front_point_exposes_its_dominated_successor():
    programs = [program("bad", 1, 1), program("successor", 2, 2), program("tradeoff", 3, 0)]
    assert target.next_wave(programs, set(), set()) == ["bad", "tradeoff"]
    assert target.next_wave(programs, {"bad"}, {"bad", "tradeoff"}) == ["successor"]


def test_correct_front_point_keeps_dominating_after_other_failure():
    programs = [program("good", 1, 1), program("bad_tradeoff", 0, 3), program("dominated", 2, 2)]
    assert target.next_wave(programs, set(), set()) == ["bad_tradeoff", "good"]
    assert target.next_wave(programs, {"bad_tradeoff"}, {"bad_tradeoff", "good"}) == []


def test_initial_wave_uses_frozen_control_prefix_without_reordering():
    front = {
        "wave0_candidate_ids": ["p1", "p2"],
        "control_orders": {"calls_lazy": ["c1", "c2", "c3"]},
    }
    assert target.initial_wave(front, "peeling") == ["p1", "p2"]
    assert target.initial_wave(front, "calls_lazy", 2) == ["c1", "c2"]


def test_initial_wave_rejects_nonpositive_candidate_budget():
    front = {"wave0_candidate_ids": ["p1"], "control_orders": {}}
    try:
        target.initial_wave(front, "peeling", 0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("nonpositive candidate budget was accepted")


def test_model_input_spec_is_deterministic_and_honors_yolo_shape():
    spec = {"shape": [1, 3, 10, 12], "uniform_range": [0.0, 1.0], "all_outputs": True}
    first = board_target.model_input_for_seed(17, spec)
    second = board_target.model_input_for_seed(17, spec)
    assert first.shape == (1, 3, 10, 12)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert float(first.min()) >= 0.0 and float(first.max()) <= 1.0


def test_multi_output_mismatch_counts_every_output():
    left = [np.array([1, 2]), np.array([3, 4, 5])]
    right = [np.array([1, 0]), np.array([0, 4, 0])]
    assert board_target.model_mismatch_count(left, right, True) == 3
