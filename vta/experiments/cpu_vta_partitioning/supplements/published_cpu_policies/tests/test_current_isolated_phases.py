from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_current_isolated_phases as m
from cpu_vta_partitioning.common import read


def prepared():
    if not (m.OUT/'PLAN.json').exists():pytest.skip('prepared capsule is an integration fixture')
    m.validate()


def test_only_five_components_no_pipeline_or_adaptive_tasks():
    specs=m.specifications()
    assert len(specs)==15 and {s['scope'] for s in specs}=={'component'}
    assert len({s['case'] for s in specs})==5
    with pytest.raises(ValueError,match='execute-board'):m.board(execute_board=False)
    with pytest.raises(ValueError,match='outside'):m.task(dict(scope='pipeline',case='cpu_02_04',mode='clean'))


def test_prepared_identity_counts_and_reused_assets():
    prepared();tasks=[m.task(s) for s in m.specifications()]
    assert len({t['task_id'] for t in tasks})==15
    assert sum(t['warmup']+t['scored']+t['tail'] for t in tasks)==105
    assert {t['executor_sha256'] for t in tasks}=={'12365f158a3780293e1bd62dcf0317d73da045cb2fdef9e94a164ccf324bd280'}
    assert {t['runtime_sha256'] for t in tasks}=={'724d15640bc5f75548725048f9b6adb1750bce768aa6ea39fe3fbaeb6726af98'}
    assert read(m.OUT/'PLAN.json')['protocol']['DNN_inferences']==0


def test_probe_changes_only_optional_phase_and_original_wait_observation():
    prepared()
    for case in m.CASES:
        clean=m.task(dict(scope='component',case=case,mode='clean'))
        observed=m.task(dict(scope='component',case=case,mode='observed'))
        a,_=m.command(clean,m.BUNDLE,'/capsule','/job');b,inputs=m.command(observed,m.BUNDLE,'/capsule','/job')
        assert '--serial' in a and '--serial' in b and '--queue-timing' not in b
        assert '--phase-event-json' not in a and '--phase-event-json' in b
        if case.startswith('cpu'):
            assert '--cpu-worker-pinning' in b and 'VTA_WAIT_POLICY=original' in b
            assert 'VTA_WAIT_PROBE=1' in b and 'libwait_adapter.so' in next(v for v in b if v.startswith('LD_PRELOAD='))
        else:
            assert '--cpu-worker-pinning' not in b and clean['configuration']['host_mask']==[3]
            assert 'VTA_DEVICE_PHASE_TRACE=/job/device_trace.jsonl' in b
        wrong=deepcopy(observed);wrong['tail']=1
        with pytest.raises(ValueError,match='changed'):m.command(wrong,m.BUNDLE,'/capsule','/job')
    assert len(inputs.strip().split(','))==3


def test_frozen_controller_snapshots_are_copied_not_mutable_hardlinks():
    prepared()
    for i,(path,h) in enumerate(sorted(m.sources().items())):
        a=Path(path);b=m.BUNDLE/'sources'/f'{i:02}_{a.name}'
        assert a.stat().st_ino!=b.stat().st_ino
