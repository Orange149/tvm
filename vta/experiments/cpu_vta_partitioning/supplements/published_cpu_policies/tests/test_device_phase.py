"""Device observer coverage must be demonstrated, not inferred from zero events."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from device_phase.collect import validate_events


def sample():
    span=dict(operation='run',tid=3,call=0,wall_start_ns=0,wall_end_ns=100,cpu_start_ns=0,cpu_end_ns=70)
    event=dict(kind='device',tid=3,wall_start_ns=20,wall_end_ns=80,cpu_start_ns=10,cpu_end_ns=50,
               result=0,instructions=30,core_start=0,core_end=1)
    status=dict(kind='status',events=1,overflow=False,provider='/capsule/runtime/libvta.so')
    return [status,event],[span],dict(warmup=0,scored=1,tail=0)


def test_real_driver_call_and_clock_containment():
    rows,spans,doc=sample();result=validate_events(rows,spans,doc)
    assert result['device_calls']==1 and result['mapped'][0]['call']==0


@pytest.mark.parametrize('change',('missing','overflow','provider','wrong_tid','error','clock','duplicate'))
def test_invalid_or_missing_driver_coverage_rejected(change):
    rows,spans,doc=sample()
    if change=='missing':rows=[{**rows[0],'events':0}]
    if change=='overflow':rows[0]['overflow']=True
    if change=='provider':rows[0]['provider']='/tmp/libfake.so'
    if change=='wrong_tid':rows[1]['tid']=4
    if change=='error':rows[1]['result']=1
    if change=='clock':rows[1]['cpu_end_ns']=90
    if change=='duplicate':rows.append(deepcopy(rows[1]));rows[0]['events']=2
    with pytest.raises(ValueError):validate_events(rows,spans,doc)


def test_additional_stage_frame_cannot_be_unobserved():
    rows,spans,doc=sample();doc['scored']=2
    with pytest.raises(ValueError,match='no real device call'):validate_events(rows,spans,doc)


def test_group_only_VTA_command_does_not_request_CPU_worker_pinning():
    from device_phase.collect import task,command
    bundle=ROOT/'artifacts/cross_network_model_01/device_phase/resnet18/capsule'
    if not bundle.exists():pytest.skip('local ARM fixture capsule absent')
    doc=task(bundle,dict(network='resnet18',case='vta_03_04',mode='qualify'))
    argv,_=command(doc,bundle,'/capsule','/job')
    assert doc['configuration']['affinity_mode']=='group'
    assert '--cpu-worker-pinning' not in argv and '--phase-event-json' in argv


def test_device_CPU_and_between_call_host_work_conserve_caller_work():
    from device_phase.extract import split_run
    run=dict(call=2,cpu_start_ns=10,cpu_end_ns=110,wall_start_ns=0,wall_end_ns=230)
    events=[dict(call=2,wall_start_ns=10,wall_end_ns=100,cpu_start_ns=30,cpu_end_ns=40,instructions=1),
            dict(call=2,wall_start_ns=120,wall_end_ns=200,cpu_start_ns=70,cpu_end_ns=90,instructions=2)]
    value=split_run(run,events)
    assert value['host_serial_CPU_ns']==[20,30,20]
    assert value['run_CPU_ns']==100
    assert sum(v['CPU_ns'] for v in value['devices'])==30
    assert sum(v['wall_ns'] for v in value['devices'])==170


def test_backwards_device_CPU_cannot_be_projected():
    from device_phase.extract import split_run
    run=dict(call=0,cpu_start_ns=20,cpu_end_ns=50)
    event=dict(call=0,wall_start_ns=0,wall_end_ns=90,cpu_start_ns=10,cpu_end_ns=30,instructions=1)
    with pytest.raises(ValueError):split_run(run,[event])
