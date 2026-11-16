from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analyze_run_scope_board import budget_choices,require_complete


def test_identical_lists_share_measurement_and_budget_is_a_frozen_prefix():
    rows=[dict(id=f'cfg_{i:02}',static_rank=i+1) for i in range(20)]
    measured={r['id']:dict(II_ms=100-i,FPS=1000/(100-i)) for i,r in enumerate(rows)}
    values=budget_choices(dict(original_complete=rows,run_scope_complete=rows),measured)
    a=[r for r in values if r['method']=='original_complete'];b=[r for r in values if r['method']=='run_scope_complete']
    assert [r['configuration'] for r in a]==['cfg_00','cfg_04','cfg_19']
    assert [r['FPS'] for r in a]==[r['FPS'] for r in b]


def test_inconsistent_FPS_and_incomplete_batch_cannot_produce_a_report():
    rows=[dict(id=f'cfg_{i:02}',static_rank=i+1) for i in range(20)]
    measured={r['id']:dict(II_ms=100,FPS=10) for r in rows};measured['cfg_00']['FPS']=20
    with pytest.raises(ValueError,match='interval or FPS'):
        budget_choices(dict(original_complete=rows),measured)
    plan=dict(configurations=88,native_sessions=176,DNN_inferences=2552)
    state=dict(status='RUNNING',completed_configurations=87,native_sessions=174,DNN_inferences=2523)
    with pytest.raises(ValueError,match='incomplete'):
        require_complete(state,plan)
