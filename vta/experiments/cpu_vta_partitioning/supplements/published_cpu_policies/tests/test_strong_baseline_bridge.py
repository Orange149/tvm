from pathlib import Path
import sys,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from strong_baseline import bridge as w


def test_frozen_baseline_is_the_complete_model_and_original_13fps_layout():
    b=w.baseline()
    assert all(b['model_switches'].values())
    assert [s['threads'] for s in b['configuration']['stages'] if s['device']=='cpu']==[3,1,1,1]
    assert b['configuration']['host_mask']==[2]
    assert b['configuration']['execution_mode']=='off'
    assert [round(c['performance']['FPS'],3) for c in b['cohorts']]==[13.001,13.043]


def test_bridge_changes_only_executor_or_output_not_resource_configuration():
    bundle=w.CURRENT;docs=[];commands=[]
    for spec in w.specs():
        if spec['phase']!='clean':continue
        doc=w.make_task(bundle,spec);argv,_=w.native_command(doc,bundle,'/bundle','/job')
        docs.append(doc);commands.append(argv)
        assert argv[argv.index('--output-mode')+1]==spec['performance_output']
        assert '--resource-affinity-probe' not in argv
        assert '--cpu-worker-pinning' in argv and '--process-cpu-affinity' not in argv
        assert not any('libwait_adapter.so' in x for x in argv)
    assert len({d['task_id'] for d in docs})==3
    assert all(d['configuration']==docs[0]['configuration'] for d in docs)
    assert {d['executor_sha256'] for d in docs}=={w.read(bundle/'manifest.json')[k] for k in ('old_runner_sha256','phase_runner_sha256')}
    assert len(w.specs())==6 and sum(sum(w.make_task(bundle,s)[k] for k in ('warmup','scored','tail')) for s in w.specs())==87


def test_qualification_keeps_full_output_probe_and_board_requires_explicit_optin():
    spec=next(s for s in w.specs() if s['phase']=='qualify');doc=w.make_task(w.CURRENT,spec)
    argv,_=w.native_command(doc,w.CURRENT,'/bundle','/job')
    assert '--resource-affinity-probe' in argv and doc['output_mode']=='raw'
    with pytest.raises(ValueError,match='execute-board'):w.board(execute_board=False)
