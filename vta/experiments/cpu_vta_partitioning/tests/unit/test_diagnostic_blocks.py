from pathlib import Path
import pytest
from cpu_vta_partitioning.campaign import diagnostic_order, run_diagnostics
from cpu_vta_partitioning.common import read
from cpu_vta_partitioning.domain import Stage


def test_diagnostic_orders_are_balanced_and_repeatable():
    for islands in (1,2,3):
        sums=dict.fromkeys((('shared',False),('shared',True),('dedicated',False),('dedicated',True)),0)
        for boot in range(6):
            order=diagnostic_order(boot,islands,20261023)
            assert order==diagnostic_order(boot,islands,20261023)
            assert set(order)==set(sums)
            for position,item in enumerate(order): sums[item]+=position
            if boot%2: assert order==list(reversed(diagnostic_order(boot-1,islands,20261023)))
        assert set(sums.values())=={9}


def test_diagnostic_drift_replaces_whole_block_then_resumes_without_recount(tmp_path):
    stages=[Stage('cpu',0,21,1,(0,))]
    pairs=[{'islands':1,'shared':stages,'dedicated':stages}]
    calls=[]
    def execute(t):
        calls.append(t)
        bad=t.get('position')=='diagnostic_after' and t['diagnostic_attempt']==0
        return {'ii_ms':11. if bad else 10.}
    args=(execute,pairs,tmp_path,stages,{'cores':4},20,{'boot_index':0,'campaign_phase':'confirmation'},'boot',13)
    result=run_diagnostics(*args)
    assert len([t for t in calls if t['kind']=='qualify_pipeline'])==2
    assert len([t for t in calls if t['kind']=='diagnostic'])==8
    assert (tmp_path/'invalid-diagnostic-islands1-attempt0.json').exists()
    assert result['blocks'][0]['attempt']==1 and len(result['measurements'])==4
    count=len(calls)
    assert run_diagnostics(*args)==result and len(calls)==count


def test_diagnostic_drift_exhaustion_keeps_failed_blocks(tmp_path):
    stages=[Stage('cpu',0,21,1,(0,))]
    def execute(t): return {'ii_ms':11. if t.get('position')=='diagnostic_after' else 10.}
    with pytest.raises(RuntimeError,match='diagnostic drift exhausted'):
        run_diagnostics(execute,[{'islands':1,'shared':stages,'dedicated':stages}],tmp_path,stages,{},20,
                        {'boot_index':0},'boot',13)
    assert len(list(tmp_path.glob('invalid-diagnostic-*.json')))==3
    assert not (tmp_path/'diagnostic-islands1.json').exists()
