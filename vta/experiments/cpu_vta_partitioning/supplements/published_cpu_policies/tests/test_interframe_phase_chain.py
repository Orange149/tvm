from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_interframe_phase_chain import phase_index, phase_chain, relative_pair


def rows(origin=0):
    result=[]
    for frame in range(2):
        for name, delay in [('a',0),('b',8)]:
            for op, start, end in [('set',0,1),('run',1,4),('get',4,5)]:
                result.append(dict(stage=name,call=frame,operation=op,
                    wall_start_ns=origin+frame*20+delay+start,
                    wall_end_ns=origin+frame*20+delay+end))
    return result


def test_chain_includes_dependency_gap_once_and_conserves_elapsed():
    index=phase_index(rows(),['a','b'],[0,1])
    chain=phase_chain(index,['a','b'],0)
    assert [r['service_ns'] for r in chain]==[5,5]
    assert [r['predecessor_to_start_ns'] for r in chain]==[0,3]
    assert sum(r['service_ns']+r['predecessor_to_start_ns'] for r in chain)==13


def test_relative_alignment_is_invariant_to_each_sources_clock_origin():
    a=phase_index(rows(),['a','b'],[0,1])
    b=phase_index(rows(10**15),['a','b'],[0,1])
    assert relative_pair(a,['a','b'],1)==relative_pair(b,['a','b'],1)
    assert relative_pair(a,['a','b'],1)[1]['relative_run_start_ns']==-12


def test_missing_duplicate_and_out_of_order_records_are_rejected():
    with pytest.raises(ValueError,match='missing'):
        phase_index(rows()[:-1],['a','b'],[0,1])
    with pytest.raises(ValueError,match='duplicate'):
        phase_index(rows()+rows()[:1],['a','b'],[0,1])
    data=rows(); data[1]['wall_end_ns']=6
    with pytest.raises(ValueError,match='overlaps'):
        phase_index(data,['a','b'],[0,1])


def test_dependence_violation_cannot_be_interpreted_as_negative_wait():
    index=phase_index(rows(),['a','b'],[0,1])
    index['b',0,'set']['wall_start_ns']=3
    with pytest.raises(ValueError,match='predecessor'):
        phase_chain(index,['a','b'],0)
