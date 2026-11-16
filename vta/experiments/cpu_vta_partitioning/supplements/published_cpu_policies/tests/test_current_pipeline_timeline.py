from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_current_pipeline_timeline as m
from cpu_vta_partitioning.common import read


def prepared():
    if not (m.OUT/'PLAN.json').exists():pytest.skip('requires prepared integration capsule')
    m.validate()


def test_explicit_board_optin_and_actual_middle_CPU_target():
    with pytest.raises(ValueError,match='execute-board'):m.board(execute_board=False)
    function,proof=m.worker_parser()
    assert proof['original_sha256']!=proof['adapted_sha256']
    assert 'CPU2-4' in str(function.__code__.co_consts)
    with pytest.raises(ValueError,match='three-session'):m.task('additional')


def test_fixed_single_layout_exact_three_sessions_97_frames():
    prepared();tasks=[m.task(v) for v in ('qualify','clean','observed')]
    assert sum(sum(t[k] for k in ('warmup','scored','tail')) for t in tasks)==97
    assert {t['configuration']['id'] for t in tasks}=={'cfg_2787a77188dd5f9363d00b15'}
    assert {t['executor_sha256'] for t in tasks}=={'12365f158a3780293e1bd62dcf0317d73da045cb2fdef9e94a164ccf324bd280'}
    assert all(t['kind']!='calibration' for t in tasks)


def test_clean_and_observed_commands_share_program_waiting_and_window():
    prepared();clean=m.task('clean');observed=m.task('observed')
    a,inputs=m.command(clean,m.BUNDLE,'/capsule','/job')
    b,other=m.command(observed,m.BUNDLE,'/capsule','/job')
    assert inputs==other
    assert '--serial' not in b and '--queue-timing' not in b and '--process-cpu-affinity' not in b
    assert 'VTA_WAIT_POLICY=original' in b and '--phase-event-json' in b
    assert '--phase-event-json' not in a and 'VTA_WAIT_PROBE=1' not in a
    assert 'libdevice_phase.so:/capsule/runner/libwait_adapter.so' in next(v for v in b if v.startswith('LD_PRELOAD='))
    changed=deepcopy(observed);changed['tail']=4
    with pytest.raises(ValueError,match='differs'):m.command(changed,m.BUNDLE,'/capsule','/job')


def test_source_snapshots_and_observer_payload_independent():
    prepared()
    for i,(path,h) in enumerate(sorted(m.sources().items())):
        a=Path(path);b=m.BUNDLE/'sources'/f'{i:02}_{a.name}'
        assert a.stat().st_ino!=b.stat().st_ino
    payload=read(m.BUNDLE/'plans/board_payload.json')
    assert {'runner/libwait_adapter.so','runner/libdevice_phase.so'}.issubset(payload['files'])
