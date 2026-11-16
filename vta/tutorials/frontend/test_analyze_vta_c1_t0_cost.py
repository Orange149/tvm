import copy
import pytest

from analyze_vta_c1_t0_cost import verify_events


def sample():
    names = ['audit_and_order_recomputation_complete',
             'deployment_and_hash_verification_complete',
             'candidate_1_qualification_decision']
    return {'t0_monotonic_ns': 100, 'rows': [{}], 'events': [
        {'name': name, 'monotonic_ns': 100 + (i + 1) * 1000000000, 'elapsed_s': i + 1.0}
        for i, name in enumerate(names)]}


def test_actual_monotonic_elapsed_not_prefix_sum():
    arm = sample()
    assert verify_events(arm)[-1]['elapsed_s'] == 3.0
    arm['events'][-1]['elapsed_s'] = 2.5
    with pytest.raises(ValueError, match='observed monotonic clock'):
        verify_events(arm)


def test_non_monotonic_or_missing_events_rejected():
    arm = sample()
    broken = copy.deepcopy(arm)
    broken['events'][-1]['monotonic_ns'] = 99
    with pytest.raises(ValueError, match='non-monotonic'):
        verify_events(broken)
    arm['events'].pop()
    with pytest.raises(ValueError, match='missing or reordered'):
        verify_events(arm)


def test_wrong_event_role_rejected():
    arm = sample()
    arm['events'][-1]['name'] = 'candidate_5_qualification_decision'
    with pytest.raises(ValueError, match='missing or reordered'):
        verify_events(arm)
