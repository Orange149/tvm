from pathlib import Path
from copy import deepcopy
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_vta_fallback_pipeline as w
from cpu_vta_partitioning.common import read,write


def test_matched_candidates_change_only_the_VTA_CPU_pool_setting():
    s=w.selection();cs=s['selected_configurations']['yolov3_tiny416'];base=cs[0]
    assert len(cs)==3 and len({c['id'] for c in cs})==3
    for c in cs:
        for a,b in zip(base['stages'],c['stages']):
            if a['device']=='cpu':assert a==b
            else:
                assert {k:v for k,v in a.items() if k not in ('threads','signature')}=={
                    k:v for k,v in b.items() if k not in ('threads','signature')}
        assert c['host_mask']==base['host_mask'] and c['control_mask']==base['control_mask']
        assert c['execution_mode']=='off'
    assert len(w.specs())==6
    assert sum(sum(w.protocol()['qualification' if s['phase']=='qualify' else 'pipeline'].values()) for s in w.specs())==87


def test_native_command_keeps_CPU_pinning_and_VTA_thread_configuration(tmp_path):
    b=tmp_path/'capsule';source=w.ROOT/'artifacts/scheduler_gap_audit_01/yolov3_tiny416/capsule'
    write(b/'manifest.json',{k:v for k,v in read(source/'manifest.json').items() if k!='files'})
    for name in ('programs.json','inputs.json'):write(b/'plans'/name,read(source/'plans'/name))
    configs=w.selection()['selected_configurations']['yolov3_tiny416']
    write(b/'plans/configurations.json',dict(pipelines={c['id']:c for c in configs},components={}))
    spec=next(s for s in w.specs() if w.selection()['labels'][s['case']]==[4,1] and s['phase']=='clean')
    doc=w.make_task(b,spec);argv,_=w.native_command(doc,b,'/bundle','/job')
    assert '--cpu-worker-pinning' in argv
    assert argv[argv.index('--stage1-runtime-num-threads')+1]=='4'
    assert argv[argv.index('--stage3-runtime-num-threads')+1]=='1'
    assert not any('libwait_adapter' in v for v in argv)
    assert '--phase-event-json' not in argv and '--session-profile-json' not in argv


def test_no_implicit_board_execution():
    with pytest.raises(ValueError,match='execute-board'):w.board()
