from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_current_gap_geometry import partition


def test_nonoverlapping_critical_paths_preserve_all_caller_and_return_gaps():
    rows=[dict(launch_start_ns=10,last_start_ns=14,last_end_ns=30),
        dict(launch_start_ns=50,last_start_ns=60,last_end_ns=90)]
    r=partition(0,100,rows)
    assert r==dict(run_ns=100,critical_entry_ns=14,critical_body_ns=46,
        caller_and_other_outside_kernel_envelope_ns=40)
    assert sum(v for k,v in r.items() if k!='run_ns')==r['run_ns']


def test_overlapping_launches_cannot_double_count_the_wall_path():
    rows=[dict(launch_start_ns=10,last_start_ns=14,last_end_ns=30),
        dict(launch_start_ns=25,last_start_ns=26,last_end_ns=50)]
    with pytest.raises(ValueError,match='overlap'):partition(0,100,rows)
