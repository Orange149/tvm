from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_formal_window_resources import complement, dependency_unready_prefix, ns


def test_window_clips_holds_without_counting_an_entire_frame_outside_window():
    held, gaps = complement(10, 50, [(0, 15), (20, 35), (48, 80)])
    assert held == 22
    assert gaps == [(15, 20), (35, 48)]
    assert held + sum(b-a for a,b in gaps) == 40


def test_one_ready_predecessor_prevents_claim_that_all_inputs_are_unready():
    assert dependency_unready_prefix(10, 30, [dict(predecessor_end_ns=20), dict(predecessor_end_ns=25)]) == 10
    assert dependency_unready_prefix(10, 30, [dict(predecessor_end_ns=9), dict(predecessor_end_ns=25)]) == 0
    assert dependency_unready_prefix(10, 30, []) == 0


def test_resource_overlap_and_fractional_ns_are_not_silently_rounded():
    with pytest.raises(ValueError, match='overlap'):
        complement(0, 30, [(0, 20), (15, 25)])
    assert ns('12.345678') == 12345678
    with pytest.raises(ValueError, match='sub-nanosecond'):
        ns('0.0000001')
