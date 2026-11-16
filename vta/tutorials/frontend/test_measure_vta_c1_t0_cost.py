import math
import pytest

from measure_vta_c1_t0_cost import qualifying, seal, verify_rows, verify_performance


def test_fixed_threshold_and_invalid_values():
    assert qualifying(85.6, 85.6)
    assert not qualifying(85.61, 85.6)
    for value in (math.nan, math.inf, 0, -1):
        with pytest.raises(ValueError):
            qualifying(value, 85.6)


def test_full_output_input_gate():
    verify_rows([{'input_index': i} for i in range(3)], 3)
    with pytest.raises(RuntimeError):
        verify_rows([{'input_index': 0}] * 3, 3)


def test_performance_drift_rejected():
    references = [{'raw_outputs': [{'fnv1a64': str(i)}]} for i in range(3)]
    rows = [{'input_index': i % 3, **references[i % 3]} for i in range(302)]
    verify_performance(rows, references, 302)
    with pytest.raises(RuntimeError):
        verify_performance(rows[:-1], references, 302)
    rows[-1] = {'input_index': 1, 'raw_outputs': [{'fnv1a64': 'corrupt'}]}
    with pytest.raises(RuntimeError):
        verify_performance(rows, references, 302)


def test_seal_order_independent_not_content_independent():
    assert seal({'a': 1, 'b': 2}) == seal({'b': 2, 'a': 1})
    assert seal({'a': 1}) != seal({'a': 2})
