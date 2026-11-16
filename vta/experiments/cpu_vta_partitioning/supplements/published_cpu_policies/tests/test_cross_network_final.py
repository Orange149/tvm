import pytest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from cross_network_final import budgets


def test_all_twenty_required_and_ties_use_frozen_order():
    rows=[dict(id=str(i)) for i in range(20)]
    values={str(i):dict(II_ms=100 if i<5 else 90) for i in range(20)}
    result=budgets(rows,values)
    assert result['1']['configuration_id']=='0'
    assert result['5']['configuration_id']=='0'
    assert result['20']['configuration_id']=='5'
    assert result['20']['FPS']==1000/90
    with pytest.raises(ValueError,match='incomplete'):budgets(rows,{k:v for k,v in values.items() if k!='19'})
    with pytest.raises(ValueError,match='unique'):budgets(rows[:-1]+[rows[0]],values)


def test_reporting_has_no_implicit_board_permission():
    from cross_network_final import long
    with pytest.raises(ValueError,match='execute-board'):long()
