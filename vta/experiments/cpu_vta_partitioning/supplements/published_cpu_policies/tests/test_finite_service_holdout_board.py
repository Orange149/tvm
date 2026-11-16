from pathlib import Path
import sys,pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
import finite_service_holdout_board as w


def test_exact_frozen_six_configurations_and_174_inferences():
    specs=w.specs();seen=set();count=0
    for s in specs:
        key=s['network'],s['case']
        if s['phase']=='qualify':seen.add(key);count+=1
        else:assert key in seen;count+=28
        assert not s['probe'] and s['runner']=='phase'
    assert len(seen)==6 and len(specs)==12 and count==174


def test_board_operations_require_explicit_gate():
    with pytest.raises(ValueError,match='execute-board'):w.board()
