"""Real detached shell jobs exercise recovery without any network or FPGA access."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import sys
import threading
import time

import pytest

from test_transport_v1 import LocalSSH, tiny_bundle
from cpu_vta_partitioning.campaign import measured
from cpu_vta_partitioning.common import read, write
from cpu_vta_partitioning.domain import Stage
from cpu_vta_partitioning.measurement.tasks import task
from cpu_vta_partitioning.transport import TransportError


def test_disconnect_defers_bulk_io_until_detached_native_job_finishes(tmp_path, tiny_bundle, monkeypatch):
    import cpu_vta_partitioning.transport as module
    board = LocalSSH(tmp_path)
    board.detached = True
    root = tmp_path / 'run'; root.mkdir()
    remote_bundle = board.deploy(tiny_bundle, root)
    boot = board.boot_id()
    document = task('screening', [Stage('cpu',0,21,1,(0,))], {'cores':4}, warmup=0,scored=3,tail=0)
    job = None
    observed_busy = threading.Event()
    injected = False
    marker = tmp_path / 'bulk-io-executed'
    original = board.command

    def native(doc, bundle, remote, remote_job):
        nonlocal job
        job = Path(remote_job)
        rows = [{'frame_id':i,'input_index':i,'completion_index':i,'completion_ms':float(i+1),'total_latency_ms':1.} for i in range(3)]
        af = {'stages':[{'requested_cpus':[0],'thread_ids':[1],'active_runtime_threads':1,
                        'effective_affinities':[{'tid':1,'cpus':[0]}]}]}
        code = ('import time\nfrom pathlib import Path\np=Path(' + repr(remote_job) + ')\n'
                '(p/"started").touch()\nlimit=time.monotonic()+20\n'
                'while not (p/"release").exists() and time.monotonic()<limit: time.sleep(.01)\n'
                '(p/"result.jsonl").write_text(' + repr('\n'.join(json.dumps(r) for r in rows)) + ')\n'
                '(p/"affinity.json").write_text(' + repr(json.dumps(af)) + ')\n'
                '(p/"finished").touch()\n')
        return [sys.executable,'-c',code], 'input\n'

    def command(script, **kwargs):
        nonlocal injected
        if job and not injected and script.startswith('test -f ') and '/exit_code' in script:
            deadline = time.monotonic()+4
            while not (job/'started').exists() and time.monotonic()<deadline:
                time.sleep(.01)
            assert (job/'started').exists()
            injected = True
            raise TransportError('injected poll disconnect')
        # Sentinels execute inside the actual guarded shell, after lock acquisition.
        for operation in ('tar -cf -', 'sha256sum -c board.sha256 > verification.log'):
            script = script.replace(operation, 'touch ' + shlex.quote(str(marker)) + '; ' + operation)
        result = original(script, **kwargs)
        if result and result.strip() == 'busy':
            observed_busy.set()
        return result

    monkeypatch.setattr(module,'native_command',native)
    monkeypatch.setattr(board,'command',command)
    try:
        with pytest.raises(TransportError, match='injected poll disconnect'):
            measured(document,tiny_bundle,remote_bundle,root,boot,board)
        captures = list((root/'raw').rglob('capture.json'))
        assert len(captures)==1 and 'bulk I/O deferred' in read(captures[0])['transport_error']
        assert not marker.exists() and not (job/'finished').exists()
        # Even a racing maintenance request cannot run while the native lock is held.
        with pytest.raises(TransportError, match='bulk I/O deferred'):
            board.idle_command('touch ' + shlex.quote(str(marker)))
        assert not marker.exists()
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(board.deploy,tiny_bundle,root)
            try:
                assert observed_busy.wait(4)
                assert not pending.done() and not marker.exists()
            finally:
                (job/'release').touch()
            assert pending.result(timeout=15)==remote_bundle
        assert marker.stat().st_mtime_ns >= (job/'finished').stat().st_mtime_ns
        a = measured(document,tiny_bundle,remote_bundle,root,boot,board)
        assert measured(document,tiny_bundle,remote_bundle,root,boot,board)==a
        assert board.launches==1 and len(list((root/'accepted').glob('*/*.json')))==1
    finally:
        if job:
            (job/'release').touch()


def test_controller_waits_before_cleanup_or_any_other_remote_work(tmp_path, monkeypatch):
    import cpu_vta_partitioning.campaign as campaign
    import cpu_vta_partitioning.prepare.package as package
    root = tmp_path/'run';root.mkdir()
    identity = {'bundle_id':'recovery-test', 'controller_identity':campaign.controller_identity()}
    write(tmp_path/'manifest.json', identity)
    write(root/'state.json', {**identity, 'bundle':str(tmp_path), 'status':'COMPLETE'})
    calls=[]
    class Board:
        def boot_id(self): calls.append('boot');return 'boot'
        def wait_idle(self,boot): calls.append('wait');raise TransportError('still active')
    monkeypatch.setattr(package,'validate',lambda _:None)
    monkeypatch.setattr(campaign,'retry_cleanup',lambda *args:calls.append('cleanup'))
    with pytest.raises(TransportError,match='still active'):
        campaign.run_current(root,Board())
    assert calls==['boot','wait']


def test_idle_wait_timeout_never_reclaims_lock_or_runs_bulk_io(tmp_path, monkeypatch):
    import cpu_vta_partitioning.transport as module
    board=LocalSSH(tmp_path);calls=[]
    monkeypatch.setattr(board,'command',lambda script,**kw:calls.append(script) or 'busy')
    times=iter([0,1000])
    monkeypatch.setattr(module.time,'monotonic',lambda:next(times))
    with pytest.raises(TransportError,match='no bulk I/O'):
        board.wait_idle('boot')
    assert len(calls)==1 and all(x not in calls[0] for x in ('tar ', 'sha256sum', 'rm ', 'mkdir '))
