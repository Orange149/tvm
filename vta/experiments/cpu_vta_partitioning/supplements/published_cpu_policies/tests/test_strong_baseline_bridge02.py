from pathlib import Path
import sys,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
import strong_baseline_bridge_repair as w


def test_qualification_exports_all_stages_not_just_final_tensor():
    spec=next(s for s in w.specs() if s['phase']=='qualify');doc=w.make_task(w.base.CURRENT,spec)
    argv,_=w.native_command(doc,w.base.CURRENT,'/bundle','/job')
    assert argv[argv.index('--output-mode')+1]=='raw_all_stages'
    assert '--output-dump-dir' in argv and '--resource-affinity-probe' in argv
    assert doc['task_id']!=w.base.make_task(w.base.CURRENT,spec)['task_id']


def test_timed_modes_and_executor_hashes_do_not_change():
    for s in w.specs():
        if s['phase']!='clean':continue
        new=w.make_task(w.base.CURRENT,s);old=w.base.make_task(w.base.CURRENT,s)
        assert new['configuration']==old['configuration'] and new['executor_sha256']==old['executor_sha256']
        assert new['output_mode']==old['output_mode']
        newargv,_=w.native_command(new,w.base.CURRENT,'/bundle','/job')
        oldargv,_=w.base.native_command(old,w.base.CURRENT,'/bundle','/job')
        assert newargv==oldargv
    with pytest.raises(ValueError,match='execute-board'):w.board(execute_board=False)
