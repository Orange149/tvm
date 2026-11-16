"""Kill only the shell wrapper: surviving work must keep maintenance excluded."""
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

import pytest
from cpu_vta_partitioning.transport import SSH, TransportError
from cpu_vta_partitioning.transport.scripts import session_lock


@pytest.fixture(params=['host-sh', 'bash-as-sh'])
def shell(request, tmp_path):
    if request.param == 'host-sh':
        return shutil.which('sh')
    binary = tmp_path/'bin'/'sh'; binary.parent.mkdir()
    binary.symlink_to(shutil.which('bash'))
    return str(binary)


class ShellTransport(SSH):
    def __init__(self, base, shell):
        super().__init__({'host':'unused', 'user':'root', 'remote_base':str(base)}, execute_board=True)
        self.shell = shell

    def boot_id(self):
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip()

    def command(self, script, **kwargs):
        result = subprocess.run([self.shell, '-c', script], capture_output=True, text=True, timeout=5)
        if result.returncode:
            raise TransportError(result.stderr)
        return result.stdout


def wait_for(path):
    deadline = time.monotonic() + 5
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert path.exists()


@pytest.mark.parametrize('sig', [signal.SIGHUP, signal.SIGINT, signal.SIGQUIT, signal.SIGTERM, signal.SIGKILL])
def test_wrapper_signal_preserves_lock_while_child_survives(tmp_path, shell, sig):
    base = tmp_path/'board'
    transport = ShellTransport(base, shell)
    boot = transport.boot_id()
    lock = base/'locks'/boot/'session.lock'
    child = tmp_path/'child.pid'; release = tmp_path/'release'; finished = tmp_path/'finished'
    marker = tmp_path/'maintenance-ran'
    child_code = ("from pathlib import Path\nimport os,time\n"
                  f"Path({str(child)!r}).write_text(str(os.getpid()))\n"
                  f"while not Path({str(release)!r}).exists(): time.sleep(.01)\n"
                  f"Path({str(finished)!r}).touch()\n")
    script = '\n'.join(session_lock(str(base), boot, 'signal-test')) + '\n' + shlex.join([sys.executable, '-c', child_code])
    process = subprocess.Popen([shell, '-c', script], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        wait_for(child)
        os.kill(process.pid, sig)
        time.sleep(.1)
        # Some shells defer traps until the foreground child returns; others
        # exit immediately. Neither may release the lock on the signal path.
        assert not finished.exists() and lock.is_dir()
        with pytest.raises(TransportError, match='bulk I/O deferred'):
            transport.idle_command('touch ' + shlex.quote(str(marker)), boot=boot)
        assert not marker.exists()
        release.touch(); wait_for(finished)
        assert process.wait(timeout=5) != 0
        assert lock.is_dir()  # no PID-only reclamation even after child exit
        with pytest.raises(TransportError, match='bulk I/O deferred'):
            transport.idle_command('true', boot=boot)
    finally:
        release.touch()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        if child.exists():
            try:
                os.kill(int(child.read_text()), signal.SIGTERM)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize('code', [0, 7])
def test_normal_shell_completion_releases_lock_including_command_failure(tmp_path, shell, code):
    transport = ShellTransport(tmp_path/'board', shell)
    if code:
        with pytest.raises(TransportError):
            transport.idle_command('exit ' + str(code))
    else:
        transport.idle_command('true')
    # A normal exit (including set -e failures) has no surviving foreground work.
    assert not (tmp_path/'board'/'locks'/transport.boot_id()/'session.lock').exists()
    assert transport.idle_command('printf available') == 'available'
