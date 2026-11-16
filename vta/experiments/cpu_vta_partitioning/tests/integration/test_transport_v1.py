"""Exercise generated shell, tar transfer and cleanup locally, without SSH/FPGA."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
from unittest.mock import patch

import pytest
from cpu_vta_partitioning.common import digest, inventory, read, write
from cpu_vta_partitioning.domain import Stage
from cpu_vta_partitioning.measurement.tasks import task
from cpu_vta_partitioning.prepare.payload import create_payload_plan, EXECUTION_POLICY
from cpu_vta_partitioning.transport import SSH, TransportError
from cpu_vta_partitioning.transport.scripts import session_lock, session_script


@pytest.fixture
def tiny_bundle(tmp_path):
    bundle=tmp_path/'bundle'
    for name in ('runtime/libdummy.so','runner/vta_stage_pipeline_runner','firmware/u-dma-buf.ko',
                 'firmware/vta_hpc.bit','inputs/000.bin','references/host.npy','sources/host.py',
                 'programs/cpu_00_20/module.so','programs/cpu_00_20/graph.json','programs/cpu_00_20/params.bin'):
        path=bundle/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(name.encode())
    write(bundle/'plans/programs.json',{'cpu_00_20':{
        'files':inventory(bundle/'programs/cpu_00_20'), 'output_types':[{'dtype':'float32','shape':[1]}]}})
    create_payload_plan(bundle)
    write(bundle/'manifest.json',{'bundle_id':'test-bundle','files':inventory(bundle)})
    return bundle


class LocalSSH(SSH):
    """Only the command transport is substituted; generated shell executes locally."""
    def __init__(self, tmp_path):
        super().__init__({'host':'unused','user':'root','remote_base':str(tmp_path/'board')},execute_board=True)
        self.commands=[];self.transferred=[];self.launches=0
        self.detached=False
    def boot_id(self): return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    def command(self,script,*,timeout=60,output=None,input_path=None):
        self.commands.append(script)
        if input_path:
            with tarfile.open(input_path) as tar: self.transferred.append(tar.getnames())
        if 'nohup sh execute.sh' in script:
            self.launches+=1
            # Wait for the local stub only. Live transport retains detached nohup.
            if not self.detached:
                script=script.rstrip().removesuffix('&')
        with open(input_path,'rb') if input_path else open(os.devnull,'rb') as source:
            result=subprocess.run(['sh','-c',script],stdin=source,capture_output=True,timeout=timeout)
        if output: Path(output).write_bytes(result.stdout)
        if result.returncode: raise TransportError(result.stderr.decode())
        return None if output else result.stdout.decode()
    def upload(self,source,target):
        shutil.copyfile(source,target)
        if Path(target).name=='execute.sh':
            # These three hardware operations alone are substituted. Locking,
            # ownership, exit publication, nohup and all result I/O remain real.
            path=Path(target)
            path.write_text('\n'.join('true' if '/sys/class/fpga_manager/' in line else line
                                       for line in path.read_text().splitlines())+'\n')


def test_restricted_ssh_login_finds_admin_tools_and_preserves_binary_streams(tmp_path,monkeypatch):
    real_run=subprocess.run
    environment=dict(os.environ,PATH='/usr/bin:/bin')
    # Reproduce the actual board login environment before exercising SSH.command.
    assert real_run(['/bin/sh','-c','command -v insmod'],env=environment,capture_output=True).returncode != 0
    def local_remote(argv,**kwargs):
        assert argv[0]=='ssh'
        return real_run(shlex.split(argv[-1]),env=environment,**kwargs)
    monkeypatch.setattr(subprocess,'run',local_remote)
    board=SSH({'host':'unused','user':'root','remote_base':'/tmp/experiment'},execute_board=True)
    executable=board.command('command -v insmod').strip()
    assert Path(executable).is_file() and Path(executable).name=='insmod'
    source=tmp_path/'payload.bin';source.write_bytes(bytes(range(256))*5)
    output=tmp_path/'received.bin'
    assert board.command('cat',input_path=source,output=output) is None
    assert output.read_bytes()==source.read_bytes()


def test_streamed_payload_excludes_host_files_and_resumes_missing_files(tmp_path,tiny_bundle):
    board=LocalSSH(tmp_path);run=tmp_path/'run';run.mkdir()
    remote=Path(board.deploy(tiny_bundle,run))
    assert not (remote/'references').exists() and not (remote/'sources').exists()
    assert not (remote/'payload.tar').exists() and not (remote/'deployment.tar.gz').exists()
    assert (remote/'programs/cpu_00_20/module.so').read_bytes()==b'programs/cpu_00_20/module.so'
    assert (remote/'board.sha256').is_file()
    assert board.transferred[0][0]=='board.sha256'
    (remote/'.deployed').unlink();(remote/'inputs/000.bin').unlink()
    board.deploy(tiny_bundle,run)
    assert set(board.transferred[-1])=={'inputs/000.bin','board.sha256'}
    assert (remote/'inputs/000.bin').is_file()
    (remote/'programs/cpu_00_20/module.so').write_bytes(b'changed')
    with pytest.raises(TransportError): board.deploy(tiny_bundle,run)


def test_deploy_fails_before_transfer_when_space_or_mount_unavailable(tmp_path,tiny_bundle,monkeypatch):
    board=LocalSSH(tmp_path);run=tmp_path/'run';run.mkdir()
    original=board.command
    def low_space(script,**kw):
        if script.startswith('df -Pk'): return 'Filesystem 1024-blocks Used Available Capacity Mounted on\nx 100 99 1 99% /\n'
        return original(script,**kw)
    monkeypatch.setattr(board,'command',low_space)
    with pytest.raises(TransportError,match='storage'): board.deploy(tiny_bundle,run)
    assert not board.transferred
    absent=tmp_path/'not-mounted'
    board.config.update(storage_mount=str(absent),remote_base=str(absent/'experiment'))
    with pytest.raises(TransportError): board.deploy(tiny_bundle,run)
    assert not absent.exists()


def test_first_interrupted_stream_has_inventory_for_incremental_retry(tmp_path,tiny_bundle,monkeypatch):
    board=LocalSSH(tmp_path);run=tmp_path/'run';run.mkdir()
    original=board.command
    copied=[]
    def disconnect(script,**kw):
        if kw.get('input_path') and not copied:
            remote=Path(board.config['remote_base'])/'bundles/test-bundle'
            with tarfile.open(kw['input_path']) as tar:
                members=tar.getmembers()[:3]
                assert members[0].name=='board.sha256'
                tar.extractall(remote,members=members)
                copied.extend(m.name for m in members[1:])
            raise TransportError('first upload disconnected')
        return original(script,**kw)
    monkeypatch.setattr(board,'command',disconnect)
    with pytest.raises(TransportError,match='disconnected'): board.deploy(tiny_bundle,run)
    remote=Path(board.deploy(tiny_bundle,run))
    assert (remote/'.deployed').is_file()
    assert not set(copied) & set(board.transferred[-1])


def test_boot_scoped_lock_preserves_old_boot_and_refuses_live_owner(tmp_path):
    base=tmp_path/'persistent';boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    old=base/'locks/old-boot/session.lock';old.mkdir(parents=True);(old/'pid').write_text('1234')
    body='\n'.join(session_lock(str(base),boot,'local-test'))
    # Hold in a shell builtin, with no FPGA commands and no orphan subprocesses.
    process=subprocess.Popen(['sh','-c',body+'\nprintf "READY\\n"\nread value\n'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        assert process.stdout.readline().strip()=='READY'
        lock=base/'locks'/boot/'session.lock'
        assert (lock/'boot_id').read_text().strip()==boot
        assert (lock/'start_ticks').read_text().strip().isdigit()
        conflict=subprocess.run(['sh','-c',body],capture_output=True)
        assert conflict.returncode==95 and lock.exists()
        process.communicate('finish\n',timeout=5)
        assert not lock.exists() and old.exists()
        assert subprocess.run(['sh','-c',body],capture_output=True).returncode==0
    finally:
        if process.poll() is None: process.kill();process.wait()


def test_dma_allocation_and_validation_are_in_frozen_qualification(tmp_path,tiny_bundle,monkeypatch):
    board=LocalSSH(tmp_path);calls=[]
    monkeypatch.setattr(board,'command',lambda script,**kw:calls.append(script) or 'qualified')
    root=tmp_path/'run';root.mkdir();boot=board.boot_id()
    board.qualify_environment('/board/bundle',boot,root,tiny_bundle)
    assert 'udmabuf0=201326592' in calls[0]
    assert 'test -c /dev/udmabuf0' in calls[0]
    assert '$(cat /sys/class/u-dma-buf/udmabuf0/size)' in calls[0]
    assert '/locks/'+boot+'/session.lock' in calls[0]
    assert read(root/('environment-'+boot+'.json'))['identity']['boot_id']==boot
    board.qualify_environment('/board/bundle',boot,root,tiny_bundle)
    assert 'test -c /dev/udmabuf0' in calls[1] and 'insmod' not in calls[1]


def test_timeout_collects_without_exit_code_and_preserves_each_attempt(tmp_path,tiny_bundle,monkeypatch):
    import cpu_vta_partitioning.transport as module
    board=LocalSSH(tmp_path);boot=board.boot_id();root=tmp_path/'run'
    document=task('screening',[Stage('cpu',0,21,1,(0,))],{'cores':4},bundle_id='test-bundle')
    remote=Path(board.job_path(document,root,boot));remote.mkdir(parents=True)
    (remote/'task_hash').write_text(digest(document));(remote/'boot_id').write_text(boot)
    (remote/'native.log').write_text('partial failure evidence')
    monkeypatch.setattr(module,'native_command',lambda *a:(['true'],'input\n'))
    for _ in range(2):
        with patch.object(module.time,'monotonic',side_effect=[0,1000]):
            with pytest.raises(TransportError,match='incomplete evidence'):
                board.measure(document,tiny_bundle,'/unused',root,boot)
    captures=list((root/'raw').rglob('capture.json'))
    assert len(captures)==2 and all(read(p)['incomplete'] for p in captures)
    assert all((p.parent/'collected/native.log').read_text()=='partial failure evidence' for p in captures)
    assert (remote/'native.log').exists() and board.launches==0
    # Recovery after an interrupted boot also fetches evidence without execution.
    board.recover_abandoned(root,boot)
    assert len(list((root/'raw').rglob('capture.json')))==3 and board.launches==0


def test_accepted_job_is_cleaned_only_after_durable_verified_receipt(tmp_path,tiny_bundle,monkeypatch):
    import cpu_vta_partitioning.transport as module
    from cpu_vta_partitioning.campaign import measured
    board=LocalSSH(tmp_path);boot=board.boot_id();root=tmp_path/'run';root.mkdir()
    document=task('screening',[Stage('cpu',0,21,1,(0,))],{'cores':4},warmup=0,scored=3,tail=0)
    def native(doc,bundle,remote_bundle,remote):
        rows=[{'frame_id':i,'input_index':i%len(doc['input_indices']),'completion_index':i,'completion_ms':float(i+1),'total_latency_ms':1.} for i in range(3)]
        affinity={'stages':[{'requested_cpus':[0],'thread_ids':[1],'active_runtime_threads':1,'effective_affinities':[{'tid':1,'cpus':[0]}]}]}
        code='from pathlib import Path; p=Path('+repr(remote)+'); '
        code+='(p/"result.jsonl").write_text('+repr('\n'.join(json.dumps(r) for r in rows))+'); '
        code+='(p/"affinity.json").write_text('+repr(json.dumps(affinity))+')'
        return [sys.executable,'-c',code],'input\n'
    monkeypatch.setattr(module,'native_command',native)
    original=board.acknowledge;observations=[]
    def ack(doc,run,boot):
        receipt=read(run/'accepted'/boot/(doc['task_id']+'.json'))
        assert all((run/path).is_file() for path in receipt['files'])
        observations.append(doc)
        if len(observations)==1: raise TransportError('lost cleanup response')
        original(doc,run,boot)
    monkeypatch.setattr(board,'acknowledge',ack)
    a=measured(document,tiny_bundle,'/unused',root,boot,board)
    remote=Path(board.job_path(observations[0],root,boot))
    assert (remote/'result.jsonl').exists()  # simulated cleanup failure remains retryable
    assert a==measured(document,tiny_bundle,'/unused',root,boot,board)
    assert board.launches==1 and len(observations)==2
    assert not (remote/'result.jsonl').exists() and (remote/'collected_by_host').is_file()
    assert a==measured(document,tiny_bundle,'/unused',root,boot,board)
    receipt=read(next((root/'accepted').rglob('*.json')))
    (root/next(iter(receipt['files']))).unlink()
    count=len(observations)
    with pytest.raises(ValueError,match='missing or changed'):
        measured(document,tiny_bundle,'/unused',root,boot,board)
    assert len(observations)==count


def test_failed_native_job_retains_evidence_and_cannot_be_acknowledged(tmp_path):
    board=LocalSSH(tmp_path);boot=board.boot_id();root=tmp_path/'run'
    document=task('screening',[Stage('cpu',0,21,1,(0,))],{'cores':4},bundle_id='test-bundle')
    remote=Path(board.job_path(document,root,boot));remote.mkdir(parents=True)
    for name,value in {'task_hash':digest(document),'boot_id':boot,'exit_code':'94','native.log':'failed'}.items():
        (remote/name).write_text(value)
    with pytest.raises(TransportError): board.acknowledge(document,root,boot)
    assert (remote/'native.log').read_text()=='failed'
