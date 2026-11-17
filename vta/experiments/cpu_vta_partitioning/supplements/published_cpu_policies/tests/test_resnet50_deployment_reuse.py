"""No network: real deployment shell/locking on a temporary local filesystem."""
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

import pytest

from cpu_vta_partitioning.common import digest, read, sha256, write
from cpu_vta_partitioning.transport import SSH, TransportError
from resnet50_validation import deployment_reuse as reuse


class LocalSSH(SSH):
    def __init__(self, base):
        self.config = dict(remote_base=str(base))
        self.calls = []
        self.fail_next_tar = False

    def command(self, script, *, timeout=60, output=None, input_path=None):
        self.calls.append(dict(script=script, input_path=str(input_path) if input_path else None))
        if self.fail_next_tar and 'tar -xzf -' in script and input_path is not None:
            self.fail_next_tar = False
            broken = Path(input_path).with_name('simulated-interrupted-stream.gz')
            raw = Path(input_path).read_bytes(); broken.write_bytes(raw[:max(1, len(raw)//2)])
            input_path = broken
        with (Path(input_path).open('rb') if input_path else open(os.devnull, 'rb')) as source:
            result = subprocess.run(['sh', '-c', script], stdin=source, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=timeout)
        if result.returncode:
            raise TransportError(result.stderr.decode(errors='replace'))
        if output is not None:
            Path(output).write_bytes(result.stdout); return None
        return result.stdout.decode()


def bundle(path, assets, *, label):
    path.mkdir()
    for name, contents in assets.items():
        p = path/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(contents)
    write(path/'plans/board_execution.json', dict(storage_margin_bytes=0, identity=label))
    payload = {str(p.relative_to(path)): dict(bytes=p.stat().st_size, sha256=sha256(p))
        for p in sorted(path.rglob('*')) if p.is_file()}
    write(path/'plans/board_payload.json', dict(files=payload, unpacked_bytes=sum(x['bytes'] for x in payload.values())))
    manifest = dict(network='test', files={str(p.relative_to(path)): dict(bytes=p.stat().st_size, sha256=sha256(p))
        for p in sorted(path.rglob('*')) if p.is_file()})
    manifest['bundle_id'] = digest(manifest)[:24]; write(path/'manifest.json', manifest)
    return path


def setup_pair(tmp_path, *, different_names=False):
    shared=b'large unchanged VTA parameter asset\n'*8192
    old=bundle(tmp_path/'old', {'programs/old/params.bin':shared, 'programs/cpu/module.so':b'old module',
        'inputs/input.bin':b'input0'}, label='old')
    param='programs/new/params.bin' if different_names else 'programs/old/params.bin'
    new=bundle(tmp_path/'new', {param:shared, 'programs/cpu/module.so':os.urandom(65536),
        'inputs/input.bin':b'input0'}, label='new')
    ssh=LocalSSH(tmp_path/'remote'); run=tmp_path/'initial'; run.mkdir()
    source=Path(SSH.deploy(ssh,old,run))
    return old,new,ssh,source,param


def snapshot(root):
    return {str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file()}


def receipt(run):
    return read(sorted((run/'deployments/reuse').glob('*.json'))[-1])


@pytest.mark.parametrize('different_names',[False,True])
def test_precise_content_reuse_original_full_verification(tmp_path,different_names):
    old,new,ssh,source,param=setup_pair(tmp_path,different_names=different_names)
    before=snapshot(source); run=tmp_path/'run'
    target=Path(reuse.deploy(ssh,new,run,old)); doc=receipt(run)
    assert doc['status']=='COMPLETE' and doc['reused_bytes'] >= (old/'programs/old/params.bin').stat().st_size
    assert source.joinpath('programs/old/params.bin').stat().st_ino == target.joinpath(param).stat().st_ino
    assert source.joinpath('programs/cpu/module.so').stat().st_ino != target.joinpath('programs/cpu/module.so').stat().st_ino
    assert snapshot(source)==before
    assert (target/'.deployed').read_text().strip()==sha256(new/'manifest.json')
    assert doc['transferred_payload_bytes'] < sum(x['bytes'] for x in read(new/'plans/board_payload.json')['files'].values())
    assert all(param not in x['members'] for x in doc['transfers'])
    for name,item in read(new/'plans/board_payload.json')['files'].items():assert sha256(target/name)==item['sha256']
    assert any('session.lock' in x['script'] for x in ssh.calls)


def test_mismatched_link_unlinked_before_original_tar_rewrite(tmp_path):
    # Explicitly force a tar rewrite of a shared inode. Only the new path may
    # change, including after a preflight/transfer race or interrupted copy.
    ssh=LocalSSH(tmp_path/'remote'); remote=tmp_path/'remote/bundles/new';remote.mkdir(parents=True)
    old=tmp_path/'source.bin';old.write_bytes(b'old immutable bytes')
    target=remote/'asset.bin';os.link(old,target)
    payload=b'new bytes';archive=tmp_path/'payload.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        info=tarfile.TarInfo('asset.bin');info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
    doc={};proxy=reuse._SafeTransfer(ssh,str(remote),{'asset.bin':dict(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())},doc)
    proxy.idle_command('cd '+shlex.quote(str(remote))+' && tar -xzf -',input_path=archive,boot=ssh.boot_id())
    assert old.read_bytes()==b'old immutable bytes' and target.read_bytes()==payload
    assert old.stat().st_ino!=target.stat().st_ino
    assert doc['transfers'][0]['destinations_unlinked_before_rewrite']


def test_interrupted_stream_resumes_without_corrupting_source(tmp_path):
    old,new,ssh,source,param=setup_pair(tmp_path)
    before=snapshot(source);run=tmp_path/'run';ssh.fail_next_tar=True
    with pytest.raises(TransportError):reuse.deploy(ssh,new,run,old)
    assert receipt(run)['status']=='FAILED' and snapshot(source)==before
    target=Path(reuse.deploy(ssh,new,run,old));doc=receipt(run)
    assert doc['status']=='COMPLETE' and snapshot(source)==before
    assert sha256(target/'programs/cpu/module.so')==sha256(new/'programs/cpu/module.so')
    assert source.joinpath('programs/old/params.bin').stat().st_ino==target.joinpath(param).stat().st_ino
    assert doc['authenticated_inventory_resume'] and doc['preseed_skipped']


def test_repeat_deployment_reuses_already_verified_bytes(tmp_path):
    old,new,ssh,source,param=setup_pair(tmp_path);run=tmp_path/'run'
    target=Path(reuse.deploy(ssh,new,run,old));inode=(target/param).stat().st_ino;before=snapshot(source)
    count=len(ssh.calls)
    assert reuse.deploy(ssh,new,run,old)==str(target)
    doc=receipt(run);calls=ssh.calls[count:]
    assert doc['target_already_deployed'] and doc['preseed_skipped']
    assert doc['transferred_payload_bytes']==0 and not doc['transfers']
    assert (target/param).stat().st_ino==inode
    # One original full inventory check, no individual target/source hashes,
    # no preseed script and no discarded deployment marker.
    assert sum('sha256sum -c board.sha256' in x['script'] for x in calls)==1
    assert not any(x['input_path'] and x['input_path'].endswith('reuse-preseed.sh') for x in calls)
    assert not any(str(source) in x['script'] for x in calls)
    assert not any('rm -f '+shlex.quote(str(target/'.deployed')) in x['script'] for x in calls)
    assert snapshot(source)==before


@pytest.mark.parametrize('source_state',['local_missing','remote_missing','marker_wrong','source_symlink'])
def test_unusable_source_falls_back_to_original_transfer(tmp_path,source_state):
    old,new,ssh,source,param=setup_pair(tmp_path);run=tmp_path/'run'
    if source_state=='local_missing':source_arg=tmp_path/'missing'
    else:source_arg=old
    if source_state=='remote_missing':
        import shutil
        shutil.rmtree(source)
    elif source_state=='marker_wrong':(source/'.deployed').write_text('wrong marker\n')
    elif source_state=='source_symlink':
        asset=source/'programs/old/params.bin';contents=asset.read_bytes();asset.unlink()
        outside=tmp_path/'outside.bin';outside.write_bytes(contents);asset.symlink_to(outside)
    before=snapshot(source) if source.exists() else {}
    target=Path(reuse.deploy(ssh,new,run,source_arg));doc=receipt(run)
    assert doc['status']=='COMPLETE' and sha256(target/param)==sha256(new/param)
    if source.exists():assert snapshot(source)==before
    if source_state!='source_symlink':assert doc['reused_bytes']==0
    else:assert param not in {x['target'] for x in doc['reused_files']}


@pytest.mark.parametrize('name',['../escape','a/../escape','/absolute','a//b','a/./b','a\\b','a\nb'])
def test_bad_paths_rejected_before_any_remote_access(tmp_path,name):
    good=bundle(tmp_path/'good',{'asset.bin':b'hello'},label='test')
    m=read(good/'manifest.json');m['files'][name]=dict(bytes=1,sha256='0'*64)
    m.pop('bundle_id');m['bundle_id']=digest(m)[:24];write(good/'manifest.json',m,replace=True)
    ssh=LocalSSH(tmp_path/'remote')
    with pytest.raises(ValueError):reuse.deploy(ssh,good,tmp_path/'run',None)
    assert not ssh.calls


def test_bad_identity_and_changed_inventory_rejected(tmp_path):
    good=bundle(tmp_path/'good',{'asset.bin':b'hello'},label='test');ssh=LocalSSH(tmp_path/'remote')
    m=read(good/'manifest.json');m['bundle_id']='../../source';write(good/'manifest.json',m,replace=True)
    with pytest.raises(ValueError):reuse.deploy(ssh,good,tmp_path/'run')
    assert not ssh.calls


def test_target_symlink_parent_refused_and_source_untouched(tmp_path):
    old,new,ssh,source,param=setup_pair(tmp_path);before=snapshot(source)
    target=Path(ssh.config['remote_base'])/'bundles'/read(new/'manifest.json')['bundle_id'];target.mkdir()
    (target/'programs').symlink_to(source/'programs',target_is_directory=True)
    with pytest.raises(TransportError):reuse.deploy(ssh,new,tmp_path/'run',old)
    assert snapshot(source)==before


def test_wrong_local_inventory_hash_rejected_without_remote_calls(tmp_path):
    old,new,ssh,source,param=setup_pair(tmp_path)
    plan=new/'plans/board_payload.json';plan.write_text(plan.read_text()+' ')
    count=len(ssh.calls)
    with pytest.raises(ValueError,match='inventory changed'):reuse.deploy(ssh,new,tmp_path/'run',old)
    assert len(ssh.calls)==count


def test_unsafe_archive_cannot_reach_transport(tmp_path):
    ssh=LocalSSH(tmp_path/'remote');remote=str(tmp_path/'remote/bundles/new');archive=tmp_path/'unsafe.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        info=tarfile.TarInfo('../source');info.size=1;tar.addfile(info,io.BytesIO(b'x'))
    proxy=reuse._SafeTransfer(ssh,remote,{},{});calls=len(ssh.calls)
    with pytest.raises(ValueError):proxy.idle_command('cd '+shlex.quote(remote)+' && tar -xzf -',input_path=archive)
    assert len(ssh.calls)==calls


def test_no_matching_content_uses_original_deploy(tmp_path):
    old=bundle(tmp_path/'old',{'asset.bin':b'old'},label='old')
    new=bundle(tmp_path/'new',{'asset.bin':b'new'},label='new')
    ssh=LocalSSH(tmp_path/'remote');run0=tmp_path/'initial';run0.mkdir();source=Path(SSH.deploy(ssh,old,run0))
    before=snapshot(source);run=tmp_path/'run';target=Path(reuse.deploy(ssh,new,run,old));doc=receipt(run)
    assert doc['planned_reuse_files']==0 and doc['fallback_reason']=='no_exact_content_matches'
    assert target.joinpath('asset.bin').read_bytes()==b'new' and snapshot(source)==before


def test_corrupt_source_bytes_not_reused_and_not_repaired_in_place(tmp_path):
    old,new,ssh,source,param=setup_pair(tmp_path)
    original=source/'programs/old/params.bin';original.write_bytes(b'X'*original.stat().st_size)
    before=snapshot(source);run=tmp_path/'run';target=Path(reuse.deploy(ssh,new,run,old));doc=receipt(run)
    assert param not in {x['target'] for x in doc['reused_files']}
    assert target.joinpath(param).stat().st_ino!=original.stat().st_ino
    assert snapshot(source)==before and sha256(target/param)==sha256(new/param)


def test_boot_change_during_preseed_and_deploy_refused(tmp_path):
    old,new,ssh,source,param=setup_pair(tmp_path);before=snapshot(source)
    original=ssh.boot_id;calls=[]
    def boot():
        calls.append(1)
        return original() if len(calls)==1 else '00000000-0000-0000-0000-000000000000'
    ssh.boot_id=boot
    with pytest.raises(TransportError,match='boot changed'):
        reuse.deploy(ssh,new,tmp_path/'run',old)
    assert snapshot(source)==before and receipt(tmp_path/'run')['status']=='FAILED'



def test_current_marker_does_not_skip_original_full_hash_check(tmp_path):
    old,new,ssh,source,param=setup_pair(tmp_path);run=tmp_path/'run'
    target=Path(reuse.deploy(ssh,new,run,old));before=snapshot(source)
    # CPU module was transferred into its own inode, so this corruption cannot
    # mutate the readonly source fixture.
    (target/'programs/cpu/module.so').write_bytes(b'bad')
    count=len(ssh.calls)
    with pytest.raises(TransportError):reuse.deploy(ssh,new,run,old)
    calls=ssh.calls[count:];doc=receipt(run)
    assert doc['target_already_deployed'] and doc['preseed_skipped'] and doc['status']=='FAILED'
    assert sum('sha256sum -c board.sha256' in x['script'] for x in calls)==1
    assert not any(str(source) in x['script'] for x in calls)
    assert snapshot(source)==before


def test_current_target_does_not_read_source_inventory_or_preseed(tmp_path,monkeypatch):
    old,new,ssh,source,param=setup_pair(tmp_path);run=tmp_path/'run'
    target=reuse.deploy(ssh,new,run,old);original=reuse._bundle
    def checked(path):
        if Path(path)==old:raise AssertionError('source inventory re-read for current target')
        return original(path)
    def forbidden(*args,**kwargs):raise AssertionError('preseed rerun for current target')
    monkeypatch.setattr(reuse,'_bundle',checked);monkeypatch.setattr(reuse,'_preseed',forbidden)
    assert reuse.deploy(ssh,new,run,old)==target



def test_hash_timeout_extended_only_for_original_payload_checks(tmp_path):
    old,new,ssh,source,param=setup_pair(tmp_path);run=tmp_path/'run'
    target=Path(reuse.deploy(ssh,new,run,old));original=ssh.command;observed=[]
    def slow_hash(script,**kwargs):
        if 'sha256sum -c board.sha256' in script:
            observed.append(kwargs.get('timeout'))
            if kwargs.get('timeout',60)<1800:
                raise TransportError('simulated complete hash exceeds old 600-second limit')
        return original(script,**kwargs)
    ssh.command=slow_hash
    assert reuse.deploy(ssh,new,run,old)==str(target)
    doc=receipt(run)
    assert observed==[1800]
    assert doc['hash_timeout_overrides']==[dict(original_timeout_seconds=600,effective_timeout_seconds=1800,
        operation='full_verification')]


def test_matching_inventory_resumes_bad_data_and_preserves_shared_old_inode(tmp_path,monkeypatch):
    old,new,ssh,source,param=setup_pair(tmp_path);run=tmp_path/'run'
    target=Path(reuse.deploy(ssh,new,run,old));before=snapshot(source)
    (target/'.deployed').unlink()
    # A partial/corrupt target may still share an immutable old inode. The
    # original missing/transfer path must detach it before writing new bytes.
    old_bad=tmp_path/'readonly-old-inode';old_bad.write_bytes(b'old bytes remain unchanged')
    (target/param).unlink();os.link(old_bad,target/param)
    (target/'programs/cpu/module.so').write_bytes(b'partial')
    inventory_before=sha256(target/'board.sha256')
    def forbidden(*args,**kwargs):raise AssertionError('authenticated inventory reran source preseed')
    monkeypatch.setattr(reuse,'_preseed',forbidden)
    count=len(ssh.calls);assert reuse.deploy(ssh,new,run,old)==str(target)
    doc=receipt(run);calls=ssh.calls[count:]
    assert not doc['target_already_deployed'] and doc['authenticated_inventory_resume'] and doc['preseed_skipped']
    assert old_bad.read_bytes()==b'old bytes remain unchanged' and old_bad.stat().st_ino!=(target/param).stat().st_ino
    assert sha256(target/param)==sha256(new/param) and snapshot(source)==before
    assert sha256(target/'board.sha256')==inventory_before
    assert sum('sha256sum -c board.sha256' in x['script'] for x in calls)==2
    assert {x['operation'] for x in doc['hash_timeout_overrides']}=={'missing_scan','full_verification'}
    assert all(x['effective_timeout_seconds']==1800 for x in doc['hash_timeout_overrides'])
    assert not any(str(source) in x['script'] for x in calls)
    assert (target/'.deployed').read_text().strip()==sha256(new/'manifest.json')


def test_inventory_marker_never_substitutes_for_final_payload_verification(tmp_path,monkeypatch):
    old,new,ssh,source,param=setup_pair(tmp_path);run=tmp_path/'run'
    target=Path(reuse.deploy(ssh,new,run,old));before=snapshot(source);(target/'.deployed').unlink()
    module=target/'programs/cpu/module.so';module.write_bytes(b'bad remote bytes')
    local=new/'programs/cpu/module.so';local.write_bytes(b'X'*local.stat().st_size)
    def forbidden(*args,**kwargs):raise AssertionError('authenticated inventory reran source preseed')
    monkeypatch.setattr(reuse,'_preseed',forbidden)
    with pytest.raises(TransportError):reuse.deploy(ssh,new,run,old)
    doc=receipt(run)
    assert doc['authenticated_inventory_resume'] and doc['preseed_skipped'] and doc['status']=='FAILED'
    assert not (target/'.deployed').exists() and snapshot(source)==before
    assert any(x['operation']=='full_verification' for x in doc['hash_timeout_overrides'])
