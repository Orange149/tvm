import io
import json
import subprocess
import tarfile
from pathlib import Path
import pytest
from cpu_vta_partitioning.common import read,write,inventory,verify_inventory
from cpu_vta_partitioning.transport import SSH,TransportError,extract_checked
from cpu_vta_partitioning.campaign import measured
from cpu_vta_partitioning.measurement.tasks import task
from cpu_vta_partitioning.domain import Stage


class FakeBoard:
    def __init__(self):
        self.calls=0
        self.disconnect=True
    def measure(self,document,bundle,remote,root,boot):
        self.calls+=1
        result=root/'raw'/boot/document['task_id']/'collected'
        result.mkdir(parents=True,exist_ok=True)
        rows=[{'frame_id':i,'input_index':i%len(document['input_indices']),'completion_index':i,'completion_ms':float(i+1),'total_latency_ms':2} for i in range(340)]
        (result/'result.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        write(result/'affinity.json',{'stages':[{'requested_cpus':[0],'thread_ids':[10],'active_runtime_threads':1,
                                                'effective_affinities':[{'tid':10,'cpus':[0]}]}]},replace=True)
        if self.disconnect:
            self.disconnect=False
            raise TransportError('simulated SSH disconnect after remote completion')
        return result


def test_disconnect_resume_accepts_once_and_checks_hash(tmp_path):
    bundle=tmp_path/'bundle'; bundle.mkdir(); write(bundle/'manifest.json',{'bundle_id':'test-bundle'})
    root=tmp_path/'run';root.mkdir()
    doc=task('screening',[Stage('cpu',0,21,1,(0,))],{'cores':4,'poll_sleep_ns':1000,'post_start_sleep_ns':1000})
    board=FakeBoard()
    with pytest.raises(TransportError):
        measured(doc,bundle,'/remote',root,'boot1',board)
    a=measured(doc,bundle,'/remote',root,'boot1',board)
    b=measured(doc,bundle,'/remote',root,'boot1',board)
    assert a==b and board.calls==2
    assert len(list((root/'accepted').rglob('*.json')))==1
    receipt=read(next((root/'accepted').rglob('*.json')))
    (root/next(iter(receipt['files']))).unlink()
    with pytest.raises(ValueError,match='missing or changed'):
        measured(doc,bundle,'/remote',root,'boot1',board)


def test_transport_requires_explicit_board_flag():
    with pytest.raises(ValueError,match='execute-board'):
        SSH({})


def test_ssh_timeout_and_wrong_boot(monkeypatch):
    connection={'host':'test','user':'root','remote_base':'/tmp/experiment'}
    ssh=SSH(connection,execute_board=True)
    def timeout(*a,**kw):
        raise subprocess.TimeoutExpired(a[0],1)
    monkeypatch.setattr(subprocess,'run',timeout)
    with pytest.raises(TransportError,match='timeout'):
        ssh.command('true')
    monkeypatch.setattr(ssh,'command',lambda *a,**kw:'not-a-boot-id')
    with pytest.raises(TransportError,match='boot ID'):
        ssh.boot_id()


def test_archive_and_hash_guards(tmp_path):
    write(tmp_path/'data.json',{'ok':True})
    expected=inventory(tmp_path)
    (tmp_path/'data.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):
        verify_inventory(tmp_path,expected)
    archive=tmp_path/'bad.tar'
    with tarfile.open(archive,'w') as tar:
        member=tarfile.TarInfo('../outside');member.size=1;tar.addfile(member,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='unsafe'):
        extract_checked(archive,tmp_path/'out')


def test_offline_sdk_rejects_transport(tmp_path):
    from cpu_vta_partitioning.prepare.build import sdk_command
    with pytest.raises(ValueError,match='forbidden'):
        sdk_command({},['ssh','test'],tmp_path/'log')


def test_process_crash_before_acceptance_keeps_raw(tmp_path):
    import multiprocessing
    import os
    import cpu_vta_partitioning.campaign as campaign
    bundle=tmp_path/'bundle';bundle.mkdir();write(bundle/'manifest.json',{'bundle_id':'test'})
    root=tmp_path/'run';root.mkdir()
    document=task('screening',[Stage('cpu',0,21,1,(0,))],{'cores':4,'poll_sleep_ns':1000,'post_start_sleep_ns':1000})
    def child():
        real_write=campaign.write
        def crash_before_receipt(path,*args,**kwargs):
            if 'accepted' in Path(path).parts:
                os._exit(23)
            return real_write(path,*args,**kwargs)
        campaign.write=crash_before_receipt
        board=FakeBoard();board.disconnect=False
        campaign.measured(document,bundle,'/remote',root,'boot1',board)
    process=multiprocessing.get_context('fork').Process(target=child)
    process.start();process.join(10)
    assert process.exitcode==23
    assert list((root/'raw').rglob('result.jsonl'))
    assert not list((root/'accepted').rglob('*.json'))
    board=FakeBoard();board.disconnect=False
    measured(document,bundle,'/remote',root,'boot1',board)
    measured(document,bundle,'/remote',root,'boot1',board)
    assert len(list((root/'accepted').rglob('*.json')))==1 and board.calls==1


def test_remote_boot_and_task_identity_guards(tmp_path,monkeypatch):
    import cpu_vta_partitioning.transport as module
    ssh=SSH({'host':'test','user':'root','remote_base':'/tmp/ex'},execute_board=True)
    document=task('screening',[Stage('cpu',0,21,1,(0,))],{'cores':4,'poll_sleep_ns':1000,'post_start_sleep_ns':1000},bundle_id='test')
    monkeypatch.setattr(ssh,'boot_id',lambda:'other')
    with pytest.raises(TransportError,match='boot changed'):
        ssh.measure(document,tmp_path,'/remote',tmp_path/'run','expected')
    monkeypatch.setattr(ssh,'boot_id',lambda:'expected')
    monkeypatch.setattr(module,'native_command',lambda *a:(['true'],'input\n'))
    monkeypatch.setattr(ssh,'command',lambda command,**kw:'wrong-task-hash' if '/task_hash' in command else '')
    with pytest.raises(TransportError,match='task hash mismatch'):
        ssh.measure(document,tmp_path,'/remote',tmp_path/'run','expected')
