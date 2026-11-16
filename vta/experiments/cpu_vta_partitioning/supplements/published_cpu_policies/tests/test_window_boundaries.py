from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from audit_window_boundaries import sufficient_tail,partition_gaps


def test_boundary_split_preserves_every_gap_without_clipping_or_double_counting():
    r=partition_gaps([0,10,20,30],15)
    assert [x['part'] for x in r['intervals']]==['before','crossing','after']
    assert sum(g['duration_ns'] for g in r['groups'].values())==30
    assert sum(g['intervals'] for g in r['groups'].values())==3


def test_boundary_at_timestamp_creates_no_artificial_crossing():
    r=partition_gaps([0,10,20,30],10)
    assert [x['part'] for x in r['intervals']]==['before','after','after']
    assert r['groups']['crossing']['observed_subwindow_FPS'] is None


def test_absent_post_boundary_data_stays_unknown_instead_of_zero_FPS():
    r=partition_gaps([0,10,25],100)
    assert r['groups']['after']['observed_subwindow_FPS'] is None
    assert r['groups']['before']['mean_interval_ns']==dict(numerator=25,denominator=2)


def test_tail_bound_covers_each_stage_in_the_full_seven_stage_domain():
    assert sufficient_tail(7,2)['max_ahead_conservative']==19
    assert sufficient_tail(7,2)['sufficient_tail']==20
    assert all(sufficient_tail(stages,2)['sufficient_tail']<=24 for stages in range(1,8))
    assert sufficient_tail(7,2)['does_not_prove_stationarity']


def test_invalid_timestamp_or_queue_contract_is_rejected():
    with pytest.raises(ValueError,match='increase'):partition_gaps([0,10,10],12)
    with pytest.raises(ValueError,match='queue'):sufficient_tail(0,2)
