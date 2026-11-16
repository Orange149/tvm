import pytest

from account_vta_c1_topk_observed_budget import prefix_cost


def fixture_rows():
    return {'a': {'correct': True, 'correctness_wall_clock_s': 2.0,
                  'performance_wall_clock_s': 3.0},
            'b': {'correct': True, 'correctness_wall_clock_s': 5.0,
                  'performance_wall_clock_s': 7.0}}


def test_prefix_sums_only_observed_subprocess_costs():
    result = prefix_cost(['a', 'b'], 1, fixture_rows(),
                         {'a': {'driver_run_calls': 10}, 'b': {'driver_run_calls': 20}})
    assert result['observed_correctness_plus_performance_subprocess_cost_s'] == 5
    assert result['performance_driver_run_calls'] == 10
    assert 'full_end_to_end_t0_wallclock_s' not in result


def test_missing_cost_is_not_silently_replaced_with_zero():
    rows = fixture_rows()
    del rows['a']['performance_wall_clock_s']
    with pytest.raises(KeyError):
        prefix_cost(['a'], 1, rows, {'a': {'driver_run_calls': 1}})


def test_invalid_rank_and_dispatch_are_rejected():
    with pytest.raises(ValueError):
        prefix_cost(['a'], 0, fixture_rows(), {})
    with pytest.raises(ValueError):
        prefix_cost(['a'], 1, fixture_rows(), {'a': {'driver_run_calls': -1}})
