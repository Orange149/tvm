import numpy as np
import pytest

from c3_five_tradeoff_node34 import reference


def test_full_contract_negative_shift_and_channel_broadcast():
    x = np.zeros((1, 32, 7, 7, 1, 16), dtype="int8")
    w = np.zeros((64, 32, 3, 3, 16, 16), dtype="int8")
    add = np.full((64, 1, 1, 1, 16), -65, dtype="int32")
    add[3, 0, 0, 0, 5] = 128
    y = reference(x, w, add)
    assert y.shape == (1, 64, 7, 7, 1, 16)
    assert y[0, 3, 0, 0, 0, 5] == 2
    assert y[0, 4, 0, 0, 0, 5] == -2
    assert np.all(y[0, 3, :, :, 0, 5] == 2)


def test_three_parameter_or_other_shape_cannot_replace_full_fusion():
    with pytest.raises(ValueError, match="contract"):
        reference(np.zeros((1,), dtype="int8"), np.zeros((1,), dtype="int8"), np.zeros((1,), dtype="int32"))
