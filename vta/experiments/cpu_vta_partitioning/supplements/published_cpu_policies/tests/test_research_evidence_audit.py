from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from research_evidence_audit import budget_prefix


def rows():
    # The global best sits beyond budget five; budget reporting cannot use it.
    return [dict(configuration_id=str(i),measured_II_ms=100-i,FPS=1000/(100-i))
            for i in range(20)]


def test_only_the_actual_prefix_is_available():
    data=rows()
    assert budget_prefix(data,1)['configuration_id']=='0'
    assert budget_prefix(data,5)['configuration_id']=='4'
    assert budget_prefix(data,20)['configuration_id']=='19'


def test_ties_preserve_rank_and_incomplete_results_are_rejected():
    data=rows()
    data[0].update(measured_II_ms=1,FPS=1000)
    data[1].update(measured_II_ms=1,FPS=1000)
    assert budget_prefix(data,5)['rank']==1
    with pytest.raises(ValueError,match='twenty'):budget_prefix(data[:-1],5)
    data[1]['configuration_id']=data[0]['configuration_id']
    with pytest.raises(ValueError,match='twenty'):budget_prefix(data,5)


def test_frame_count_and_FPS_cannot_be_substituted():
    data=rows();data[0]['FPS']=16/data[0]['measured_II_ms']*1000
    with pytest.raises(ValueError,match='FPS and completion'):budget_prefix(data,1)
