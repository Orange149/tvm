from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_scope_board as board
import common_top20.campaign as campaign
from cpu_vta_partitioning.common import read,write,digest
from cpu_policies.open_resources.domain import stage,candidate


def test_qualification_exports_all_stages_and_performance_keeps_raw_final_outputs(tmp_path):
    cfg=candidate([stage('cpu',0,1,1,[0]),stage('vta',1,13,1,[0,1,2,3]),
                   stage('cpu',13,14,1,[1])],gate='resource_group')
    programs={r['program']:dict(name=r['program'],device=r['device'],begin=r['begin'],end=r['end'],
        input_names=['edge'],input_node_ids=[i],output_node_ids=[i+1]) for i,r in enumerate(cfg['stages'])}
    write(tmp_path/'manifest.json',dict(network='yolov3_tiny416',bundle_id='synthetic',
        runner_sha256=board.RUNNER_SHA,runtime_sha256='synthetic-runtime'))
    write(tmp_path/'plans/protocol.json',dict(experiment='synthetic',short_window=dict(warmup=8,scored=16,tail=4)))
    write(tmp_path/'plans/programs.json',programs)
    write(tmp_path/'plans/inputs.json',dict(records=[dict(input='inputs/000.bin')]))
    for phase,mode in [('qualify','raw_all_stages'),('short','raw')]:
        doc=board.task(tmp_path,cfg,phase)
        argv,_=board.command(doc,tmp_path,'/remote/bundle','/remote/job')
        assert argv[argv.index('--output-mode')+1]==mode
        assert argv[argv.index('--cpu-stage-gate')+1]=='resource_group'
        assert '--queue-timing' not in argv and '--process-cpu-affinity' not in argv


def test_resume_rejects_changed_boot_without_overwriting_identity(tmp_path,monkeypatch):
    monkeypatch.setattr(board,'OUT',tmp_path/'output')
    write(board.OUT/'FREEZE.json',{'fixture':'selection'})
    write(board.OUT/'PREPARATION.json',{'fixture':'capsules'})
    run=tmp_path/'run';before=board.bind_run(run,'boot-A',{'resnet18':'capsule-A'})
    assert board.bind_run(run,'boot-A',{'resnet18':'capsule-A'})==before
    with pytest.raises(ValueError,match='frozen input'):
        board.bind_run(run,'boot-B',{'resnet18':'capsule-A'})
    assert read(run/'run.json')==before


def test_accepted_session_is_replayed_without_new_execution_and_hash_error_rejected(tmp_path,monkeypatch):
    def summarize(raw,document,bundle):
        return dict(correct=True,output_fingerprints=[['output-fixture']])
    monkeypatch.setattr(campaign,'summary',summarize)
    class FakeSSH:
        calls=0
        def measure(self,document,bundle,remote,run,boot):
            self.calls+=1;raw=Path(run)/'raw'/boot/document['task_id'];raw.mkdir(parents=True)
            (raw/'boot_id').write_text(boot);(raw/'task_hash').write_text(digest(document))
            (raw/'result.jsonl').write_text('synthetic result\n')
            return raw
        def acknowledge(self,*args):pass
    ssh=FakeSSH();doc=dict(task_id='task-fixture',kind='qualify_pipeline');run=tmp_path/'run'
    first=board.accept(ssh,doc,tmp_path,'/remote',run,'boot-A')
    assert board.accept(ssh,doc,tmp_path,'/remote',run,'boot-A')==first
    assert ssh.calls==1
    (Path(first['raw'])/'result.jsonl').write_text('tampered result\n')
    with pytest.raises(ValueError,match='changed artifact'):
        board.accept(ssh,doc,tmp_path,'/remote',run,'boot-A')
    assert ssh.calls==1


def test_board_requires_explicit_execution_switch_before_accessing_capsules():
    with pytest.raises(ValueError,match='execute-board'):
        board.board(execute_board=False)
