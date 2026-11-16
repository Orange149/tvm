from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_vta_fallback_threads as w
from cpu_vta_partitioning.common import read,write
from cpu_policies.open_resources.runtime import _affinity
from cpu_policies.open_resources.domain import validate_candidate


def test_bounded_component_domain_does_not_change_the_old_search_domain():
    configs=w.configurations();assert len(configs)==4 and len(w.specs())==8
    assert len({c['id'] for c in configs.values()})==4
    calls=sum(sum(w.protocol()['qualification' if s['phase']=='qualify' else 'calibration'].values()) for s in w.specs())
    assert calls==44
    for c in configs.values():
        w.validate_configuration(c)
        if c['stages'][0]['threads']==4:
            with pytest.raises(ValueError,match='VTA host affinity'):validate_candidate(c,component=True)
    bad=deepcopy(configs['vta_03_06_t4']);bad['stages'][0]['threads']=2
    with pytest.raises(ValueError):w.validate_configuration(bad)


def test_actual_multithread_VTA_workers_are_validated_as_group_workers():
    c=w.configurations()['vta_03_06_t4']
    row=dict(requested_cpus=[0,1,2,3],active_runtime_threads=4,caller_tid=100,
        thread_ids=[100,101,102,103],effective_affinities=[dict(tid=t,cpus=[0,1,2,3],cpu_time_delta_ms=20) for t in range(100,104)])
    report=dict(cpu_worker_pinning=False,stages=[row])
    ledger,idle=_affinity(report,c,True)
    assert len(ledger)==4 and not idle
    for change in ('wrong_count','escaped_core','duplicate_tid'):
        bad=deepcopy(report)
        if change=='wrong_count':bad['stages'][0]['active_runtime_threads']=1
        elif change=='escaped_core':bad['stages'][0]['effective_affinities'][1]['cpus']=[4]
        else:bad['stages'][0]['thread_ids'][1]=100
        with pytest.raises(ValueError):_affinity(bad,c,True)


def test_native_command_preserves_full_frontier_and_real_four_thread_pool(tmp_path):
    bundle=tmp_path/'capsule';m=read(w.PARENT/'manifest.json')
    write(bundle/'manifest.json',{k:v for k,v in m.items() if k!='files'})
    for name in ('programs.json','inputs.json'):write(bundle/'plans'/name,read(w.PARENT/'plans'/name))
    write(bundle/'plans/configurations.json',dict(components=w.configurations(),pipelines={}))
    spec=next(s for s in w.specs() if s['case']=='vta_09_12_t4' and s['phase']=='clean')
    doc=w.make_task(bundle,spec);argv,inputs=w.native_command(doc,bundle,'/bundle','/job')
    assert argv[argv.index('--stage0-runtime-num-threads')+1]=='4'
    assert argv[argv.index('--stage0-cpu-affinity')+1]=='0,1,2,3'
    program=read(w.PARENT/'plans/programs.json')['vta_09_12']
    assert len(program['input_node_ids'])==3
    records=read(w.PARENT/'plans/inputs.json')['records']
    assert inputs.strip().split(',')==[
        '/bundle/'+records[0]['frontiers'][str(node)]['raw'] for node in program['input_node_ids']]
    assert '--cpu-worker-pinning' not in argv and '--phase-event-json' not in argv
    assert not any('libwait_adapter' in v for v in argv)


def test_board_access_is_explicit():
    with pytest.raises(ValueError,match='execute-board'):w.board()
