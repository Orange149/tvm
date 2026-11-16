import copy,json
import pytest
from analyze_cpu_vta_core_partition_validation import dedicated_inflation,serial_drift,summarize
from run_cpu_vta_residual_validation import verify_affinity
from run_cpu_vta_core_partition_validation import configuration


def serial(name='D',values=(50,65,35)):
    r={k:'frozen' for k in ('runner_sha256','runtime_bindings','program_identity','numeric_input_sha256','spin_count','queue_depth','cpu_frequency_khz')}
    r.update(mode='serial',boot_id='kernel1',boot_index=1,condition=name,threads=[3,1,3],affinity=configuration(name)['affinity'],
             metrics={'stages':[{'run_ms':{'mean':v}} for v in values]})
    return r


def perf(values=(60,66,42)):
    r=serial();r.update(mode='pipeline',gate_scope='cpu',driver_snapshots_enabled=False,
                       gate_metrics={'stages':[{'run_mean_ms':v} for v in values]});return r


def test_trigger_uses_matching_off_service_and_strict_threshold():
    a,b=serial(),serial();assert dedicated_inflation(a,b,[perf(),perf()])['trigger']
    assert not dedicated_inflation(a,b,[perf((52,70,36)),perf((52,70,36))])['trigger']
    on=perf();on['driver_snapshots_enabled']=True
    with pytest.raises(ValueError):dedicated_inflation(a,b,[on,perf()])


def test_identity_and_baseline_drift_stop():
    a,b=serial(),serial();b['numeric_input_sha256']='changed'
    with pytest.raises(ValueError):dedicated_inflation(a,b,[perf(),perf()])
    before={n:serial(n) for n in ('S','D')};after=copy.deepcopy(before)
    assert serial_drift(before,after)['passed']
    after['D']['metrics']['stages'][0]['run_ms']['mean']=53
    assert not serial_drift(before,after)['passed']


def test_actual_pool_mask_mismatch_rejected(tmp_path):
    stages=[]
    for i,mask in enumerate(configuration('D')['affinity']):
        stages.append({'stage_index':i,'requested_cpus':mask,'thread_ids':[i+1],'caller_tid':i+1,'effective_affinities':[{'tid':i+1,'cpus':mask}]})
    p=tmp_path/'aff.json';p.write_text(json.dumps({'stages':stages}));verify_affinity(p,configuration('D'),range(3),'pipeline')
    stages[1]['effective_affinities'][0]['cpus']=[0,1,2,3];p.write_text(json.dumps({'stages':stages}))
    with pytest.raises(ValueError):verify_affinity(p,configuration('D'),range(3),'pipeline')


def test_missing_and_duplicate_primary_data_rejected():
    r=perf();r.update(session_id='x',block='pair0_off')
    with pytest.raises(ValueError):summarize([r])
    with pytest.raises(ValueError):summarize([r,r])
