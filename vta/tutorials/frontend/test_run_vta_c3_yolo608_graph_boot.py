import numpy as np

from run_vta_c3_yolo608_graph_boot import (
    TIMING_REPEATS,
    VARIANTS,
    mismatch_counts,
    random_input,
    variant_order,
)


def test_inputs_are_deterministic_and_608():
    lhs = random_input(7)
    rhs = random_input(7)
    assert lhs.shape == (1, 3, 608, 608)
    assert lhs.dtype == np.float32
    assert np.array_equal(lhs, rhs)


def test_variant_order_reverses_across_boots():
    assert variant_order(1) == list(VARIANTS)
    assert variant_order(2) == list(reversed(VARIANTS))
    assert TIMING_REPEATS == 10
    assert [9090 + 2 * index for index, _ in enumerate(VARIANTS)] == [9090, 9092, 9094]


def test_mismatch_counts_are_per_output():
    lhs = [np.array([1, 2]), np.array([3])]
    rhs = [np.array([1, 0]), np.array([3])]
    assert mismatch_counts(lhs, rhs) == [1, 0]
