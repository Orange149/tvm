import json
import pytest
from cpu_vta_partitioning.measurement import frames, window, calibration, affinity
from cpu_vta_partitioning.analysis import paired, holm, report


def test_completion_window_and_duplicates(tmp_path):
    data=[{'frame_id':i,'completion_index':i,'completion_ms':10.*i,'total_latency_ms':20} for i in range(340)]
    path=tmp_path/'frames.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in data))
    result=window(frames(path,340))
    assert result['ii_ms']==10 and result['kept_timestamps']==300 and result['gaps']==299
    data[0]['frame_id']=1
    path.write_text('\n'.join(json.dumps(r) for r in data))
    with pytest.raises(ValueError,match='duplicate'):
        frames(path,340)
    with pytest.raises(ValueError,match='expected'):
        frames(path,341)


def test_exclusive_core_time():
    row={'stage_cpu_time_scope':'exclusive_process_cpu_time','stage0_set_ms':1,'stage0_run_ms':3,'stage0_get_ms':2,
         'stage0_set_process_cpu_ms':.4,'stage0_run_process_cpu_ms':5,'stage0_get_process_cpu_ms':.6}
    assert calibration([row]*37)=={'service_ms':6,'run_core_ms':5,'boundary_core_ms':1}
    with pytest.raises(ValueError,match='isolated'):
        calibration([{**row,'stage_cpu_time_scope':'disabled_concurrent_pipeline'}]*37)


def test_active_threads_and_affinity():
    stage={'mask':[0,1],'threads':2}
    record={'stages':[{'requested_cpus':[0,1],'thread_ids':[10,11,12], 'active_runtime_threads':2,
                       'effective_affinities':[{'tid':i,'cpus':[0,1]} for i in (10,11,12)]}]}
    assert affinity(record,[stage])  # dormant pool capacity is separate from active workers
    record['stages'][0]['effective_affinities'][0]['cpus']=[2]
    with pytest.raises(ValueError,match='escapes'):
        affinity(record,[stage])


def test_boot_pairing_holm_equivalence():
    a={f'boot{i}':10+i/10 for i in range(6)}
    b={k:v*1.10 for k,v in a.items()}
    result=paired(a,b)
    assert result['gain']==pytest.approx(1-1/1.1) and result['p_two_sided']<.001
    assert paired(a,a)['equivalent_5pct'] and paired(a,a)['p_two_sided']==1
    assert holm({'a':.01,'b':.03,'c':.04})=={'a':.03,'b':.06,'c':.06}
    with pytest.raises(ValueError,match='paired'):
        paired(a,{'another':1})
    selection={g:{'1':'same','5':'same'} for g in 'RCHAOGD'}
    result=report(selection,{'same':a})
    assert all(c['same_configuration'] and c['gain']==0 for c in result['comparisons'].values())
    assert not result['continue_sensitivity']


def test_non_significance_is_not_equivalence():
    a={str(i):10 for i in range(6)}
    b={str(i):v for i,v in enumerate([5,20,5,20,5,20])}
    result=paired(a,b)
    assert result['p_two_sided']>.05 and not result['equivalent_5pct']
