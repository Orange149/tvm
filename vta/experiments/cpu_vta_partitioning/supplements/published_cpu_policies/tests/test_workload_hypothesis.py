"""Conditional guard applicability is execution-based, never network-name-based."""
from pathlib import Path
import sys
from copy import deepcopy
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from workload_model.experiments.host_exposure import exposed


@pytest.mark.parametrize('masks,gate,result',[
    ([[0,1,2,3],[1,2,3],[0]],'off',True),
    ([[0,1],[2,3]],'off',False),
    ([[0],[0]],'off',False),
    ([[0,1],[0,1]],'off',True),
    ([[0,1],[0,1]],'resource_group',False)])
def test_exposure_uses_actual_overlap_and_execution_mode(masks,gate,result):
    cfg=dict(stages=[dict(device='cpu',mask=m,threads=len(m)) for m in masks],execution_mode=gate)
    assert exposed(cfg)==result
    renamed=deepcopy(cfg);renamed['network']='arbitrary_network_name'
    assert exposed(renamed)==result
