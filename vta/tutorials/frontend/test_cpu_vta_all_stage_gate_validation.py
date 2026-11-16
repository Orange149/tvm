import copy
import pytest

from analyze_cpu_vta_all_stage_gate_validation import analyze, effect, IDENTITY
from run_cpu_vta_all_stage_gate_validation import derive_runner, SOURCE, adapters
import run_cpu_vta_residual_validation as native


def rows(scope='all', mode='exclusive'):
    result=[]
    for frame in range(10):
        r={'frame_id':frame,'mode':'pipeline','stage_count':3,'completion_ms':frame*20+16,
           'cpu_stage_gate_mode':mode,'stage_gate_scope':scope}
        for i,offset in enumerate((1,6,11)):
            t=frame*20+offset
            r.update({f'stage{i}_start_ms':t,f'stage{i}_end_ms':t+4,
                      f'stage{i}_run_start_ms':t+1,f'stage{i}_run_end_ms':t+3,
                      f'stage{i}_run_ms':2,f'stage{i}_ms':4,f'stage{i}_process_cpu_ms':None})
            for field,value in (('request',t-1),('acquire',t-.5),('release',t+4.5),('wait',.5)):
                r[f'stage{i}_cpu_gate_{field}_ms']=value if mode!='off' and (scope=='all' or i!=1) else None
        result.append(r)
    return result


def test_all_wait_excluded_and_vta_gated():
    m=analyze(rows(),'exclusive','all',0)
    assert m['ii_ms']==20 and m['stages'][1]['gate_wait_mean_ms']==.5
    assert m['stages'][1]['run_mean_ms']==2 and all(x==0 for x in m['pairwise_stage_overlap_ms'].values())


def test_cpu_only_bypasses_vta():
    assert analyze(rows('cpu'),'exclusive','cpu',0)['stages'][1]['gate_wait_mean_ms'] is None


@pytest.mark.parametrize('fault',['vta_missing','scope','cross_frame_overlap','wait_inside','frame_missing','process_cpu','wait_mismatch'])
def test_rejects_contract_fault(fault):
    r=rows()
    if fault=='vta_missing':r[0]['stage1_cpu_gate_acquire_ms']=None
    elif fault=='scope':r[0]['stage_gate_scope']='cpu'
    elif fault=='cross_frame_overlap':r[0]['stage2_cpu_gate_release_ms']=21
    elif fault=='wait_inside':r[0]['stage1_cpu_gate_acquire_ms']=7
    elif fault=='frame_missing':r.pop(1)
    elif fault=='process_cpu':r[0]['stage1_process_cpu_ms']=1
    else:r[0]['stage1_cpu_gate_wait_ms']=99
    with pytest.raises(ValueError):analyze(r,'exclusive','all',0)


def test_pair_identity_and_scope():
    b={k:'fixture' for k in IDENTITY};b['gate_metrics']=analyze(rows('cpu'),'exclusive','cpu',0)
    c=copy.deepcopy(b);c['gate_metrics']=analyze(rows(),'exclusive','all',0)
    assert effect(b,c)['front_run_reduction_fraction']==0
    c['threads']=[4,1,1]
    with pytest.raises(ValueError):effect(b,c)


def test_patch_anchors_and_release_before_push():
    s=(SOURCE/'raw/runner_sources/vta_stage_pipeline_runner.cc').read_text()
    derived=derive_runner(s)
    assert 'args.stage_gate_scope == "all" || stages[stage_index].device != "vta"' in derived
    assert derived.index('admission_lease.reset();') < derived.index('queues[stage_index + 1]->Push(frame',derived.index('admission_lease.reset();'))
    with pytest.raises(ValueError):derive_runner(s.replace('  std::string cpu_stage_gate = "off";',''))


def test_adapters_restore_on_exception():
    build,argv=native.build_runner,native.runner_argv
    with pytest.raises(RuntimeError):
        with adapters():raise RuntimeError('injected')
    assert native.build_runner is build and native.runner_argv is argv
