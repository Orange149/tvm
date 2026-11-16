"""Offline parsing, independent DAG arithmetic, source isolation and no I/O."""
from copy import deepcopy
from decimal import Decimal
import builtins
import json
from pathlib import Path
import socket
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from cpu_vta_partitioning.common import read,sha256,verify_inventory,write
from dependency_audit.model import replay,graph,schedule
from dependency_audit.trace import parse
from dependency_audit.common import offline_guard


def model_input(kernels,cores=None,gaps=None):
    cores=list(range(len(kernels[0]))) if cores is None else cores
    return dict(schema=1,source_role='isolated_original_wait_callbacks',cases={'case':dict(core_map=cores,
        frames=[dict(frame_id=0,phase='scored',worker_cpu_ns=kernels,
                     interlaunch_caller_cpu_ns=gaps if gaps is not None else [0]*(len(kernels)-1))])})


@pytest.mark.parametrize('kernels',[[[4],[6]],[[9,1],[1,9]],[[9,9],[1,1]],
    [[1,9,2],[7,2,1],[2,1,8]],[[1,2,3,4],[4,3,2,1]]])
def test_graph_matches_independent_scalar_and_completion_dependencies(kernels):
    data=model_input(kernels,gaps=list(range(len(kernels)-1)))
    result=replay(data)['cases']['case']['frames'][0]
    assert result['join_ns']==sum(max(row) for row in kernels)
    assert result['aggregate_ns']==max(sum(row[i] for row in kernels) for i in range(len(kernels[0])))
    assert result['callback_plus_observed_gap_ns']==result['join_ns']+sum(data['cases']['case']['frames'][0]['interlaunch_caller_cpu_ns'])
    timeline={n['id']:n for n in result['timeline']}
    for k in range(1,len(kernels)):
        previous=max(timeline[f'kernel{k-1}_worker{w}']['end_ns'] for w in range(len(kernels[0])))
        assert all(timeline[f'kernel{k}_worker{w}']['start_ns']>=previous for w in range(len(kernels[0])))
    assert result==replay(data)['cases']['case']['frames'][0]


def test_same_per_thread_totals_different_join_cost():
    a=replay(model_input([[9,1],[1,9]]))['cases']['case']['frames'][0]
    b=replay(model_input([[9,9],[1,1]]))['cases']['case']['frames'][0]
    assert a['aggregate_ns']==b['aggregate_ns']==10
    assert (a['join_ns'],b['join_ns'])==(18,10)


def test_replay_uses_no_files_and_preserves_fractional_means(monkeypatch):
    data=model_input([[1,2]])
    second=deepcopy(data['cases']['case']['frames'][0]);second['frame_id']=1;second['worker_cpu_ns']=[[1,3]]
    data['cases']['case']['frames'].append(second)
    def deny(*a,**k):raise AssertionError('pure model attempted file I/O')
    monkeypatch.setattr(builtins,'open',deny)
    result=replay(data)['cases']['case']['phases']['scored']['means']['join_ns']
    assert result==dict(numerator=5,denominator=2,unit='ns')


@pytest.mark.parametrize('field',['fps','pipeline_duration_ns','migration_trajectory','raw_path'])
def test_model_rejects_nonisolated_fields(field):
    data=model_input([[1,2]])
    data['cases']['case']['frames'][0][field]=123
    with pytest.raises(ValueError):replay(data)


def test_model_rejects_pipeline_role_and_noninteger_cost():
    data=model_input([[1,2]]);data['source_role']='pipeline_observation'
    with pytest.raises(ValueError):replay(data)
    data=model_input([[1,2.0]])
    with pytest.raises(ValueError):replay(data)
    data=model_input([[1,2]],cores=[0,0])
    with pytest.raises(ValueError):replay(data)


def test_dag_cycles_and_unknown_dependency_rejected():
    nodes=graph(model_input([[1,2]])['cases']['case']['frames'][0],[0,1])
    nodes[0]['parents']=['kernel0_join']
    with pytest.raises(ValueError,match='cyclic'):schedule(nodes)
    nodes[0]['parents']=['missing']
    with pytest.raises(ValueError,match='unknown'):schedule(nodes)


def trace_fixture(kernels):
    threads=len(kernels[0]);events=[];caller=100;clock={i:1000 for i in range(threads)}
    for k,work in enumerate(kernels):
        ws=100000+k*10000;cs=clock[0]
        group=[]
        for j,duration in enumerate(work):
            task=dict(kind=1,launch=k+1,caller=caller,tid=caller+j,task=j,tasks=threads,
                wall_start_ns=ws+100,wall_end_ns=ws+100+duration,
                cpu_start_ns=clock[j]+10,cpu_end_ns=clock[j]+10+duration,
                function_offset=16+k*16,result=0,module='/bundle/programs/cpu_00_02/module.so')
            group.append(task);clock[j]=task['cpu_end_ns']
        end=max(v['wall_end_ns'] for v in group)+20
        launch=dict(group[0],kind=0,task=-1,wall_start_ns=ws,wall_end_ns=end,
                    cpu_start_ns=cs,cpu_end_ns=clock[0]+10,last_task_end_ns=end-20)
        events+=group+[launch]  # Real records are stored at completion, not start.
        clock[0]=launch['cpu_end_ns']+30
    def ms(n):return str(Decimal(n)/1000000)
    start=events[0]['wall_start_ns']-1100
    end=events[-1]['wall_end_ns']+1000
    native=[dict(frame_id=0,completion_index=0,stage0_run_start_ms=ms(start),stage0_run_end_ms=ms(end),
        stage0_run_ms=ms(end-start),stage0_set_ms=ms(100),stage0_get_ms=ms(100),
        stage0_run_process_cpu_ms=ms(100000),stage_cpu_time_scope='exclusive_process_cpu_time')]
    affinity=dict(caller_tid=caller,requested_cpus=list(range(threads)),active_runtime_threads=threads,
        effective_affinities=[dict(tid=caller+j,cpus=[j]) for j in range(threads)])
    return events,affinity,native


def parsed(events,affinity,native):
    return parse(events,affinity,native,'a'*64,expected_kernels=2)


def test_completion_order_records_and_frame_clock_interval():
    events,affinity,native=trace_fixture([[9,1],[1,9]])
    a=parsed(events,affinity,native)
    b=parsed(list(reversed(events)),affinity,native)
    assert a==b and a['frame_mapping']
    assert a['clock_offset_consistency_interval']['lower_ns']<=a['clock_offset_consistency_interval']['upper_ns']
    assert not a['exact_cross_clock_anchor']


@pytest.mark.parametrize('problem',['missing','duplicate','wrong_tid','wrong_kernel','wrong_module','clock_overlap','bad_join'])
def test_corrupt_callback_or_order_rejected(problem):
    events,affinity,native=trace_fixture([[9,1],[1,9]])
    if problem=='missing':events.pop(0)
    elif problem=='duplicate':events.append(deepcopy(events[0]))
    elif problem=='wrong_tid':events[0]['tid']=999
    elif problem=='wrong_kernel':events[0]['function_offset']=999
    elif problem=='wrong_module':events[0]['module']='/bundle/programs/cpu_07_08/module.so'
    elif problem=='clock_overlap':events[-1]['wall_start_ns']=events[2]['wall_start_ns']+1
    elif problem=='bad_join':events[2]['last_task_end_ns']+=1
    with pytest.raises(ValueError):parsed(events,affinity,native)


def test_unreliable_frame_mapping_retained_as_whole_session_only():
    events,affinity,native=trace_fixture([[9,1],[1,9]])
    native.append(dict(native[0],frame_id=1,completion_index=1))
    value=parsed(events,affinity,native)
    assert not value['frame_mapping'] and value['frames']==[] and len(value['launches'])==2


def test_concurrent_process_CPU_time_not_used_for_stage_accounting():
    events,affinity,native=trace_fixture([[9,1],[1,9]])
    native[0]['stage_cpu_time_scope']='disabled_concurrent_pipeline'
    native[0]['stage0_run_process_cpu_ms']=None
    value=parsed(events,affinity,native)
    assert value['frames'][0]['native_run_cpu_ns'] is None
    assert value['frames'][0]['unlocated_CPU_residual_ns'] is None


def test_offline_guard_denies_network_and_shell():
    with offline_guard() as events:
        with pytest.raises(RuntimeError,match='network'):socket.create_connection(('192.168.1.247',22))
        with pytest.raises(RuntimeError,match='process'):subprocess.run(['ssh','board','true'])
        with pytest.raises(RuntimeError,match='process'):subprocess.run(['sh','-c','scp old new'])
    assert events['network_attempts']==1 and events['forbidden_process_attempts']==2


def test_changed_source_file_hash_rejected(tmp_path):
    f=tmp_path/'module.so';f.write_bytes(b'original')
    files={'module.so':dict(sha256=sha256(f),bytes=f.stat().st_size)}
    verify_inventory(tmp_path,files)
    f.write_bytes(b'changed!')
    with pytest.raises(ValueError):verify_inventory(tmp_path,files)


def test_cli_has_no_board_action():
    from run_dependency_audit import main
    with pytest.raises(SystemExit):main(['board','--execute-board'])


def test_real_isolated_source_replay_is_offline_and_exact():
    from dependency_audit.evidence import wait_receipt
    with offline_guard() as observed:
        value,proof=wait_receipt('overlap4','component',True,False)
    assert value['summary']['correct'] and proof['scope']=='component'
    assert observed['network_attempts']==observed['forbidden_process_attempts']==0


def test_diagnostics_cannot_be_opened_before_replay_freeze(tmp_path,monkeypatch):
    from dependency_audit import analysis
    attempted=[]
    def forbidden():
        attempted.append(True)
        raise AssertionError('diagnostic source read before input/replay freeze')
    monkeypatch.setattr(analysis,'diagnostic_sources',forbidden)
    with pytest.raises(FileNotFoundError):analysis.analyze(tmp_path)
    assert attempted==[]


def test_internal_barrier_is_unsupported_instead_of_counted_as_compute():
    events,affinity,native=trace_fixture([[9,1],[1,9]])
    events.append(dict(events[0],kind=2))
    with pytest.raises(ValueError,match='barrier'):parsed(events,affinity,native)


def test_unverified_module_fingerprint_rejected():
    events,affinity,native=trace_fixture([[9,1],[1,9]])
    with pytest.raises(ValueError,match='SHA256'):parse(events,affinity,native,'unverified',expected_kernels=2)
