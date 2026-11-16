from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from board_transport_without_reprogram import session_script,transport


def test_script_keeps_boot_lock_and_execution_without_reprogramming():
    script=session_script('/tmp/experiment','/tmp/experiment/job','boot-fixture','task-fixture',
                          ['env','RUN_FIXTURE=1','/tmp/runner','--cpu-stage-gate','resource_group'])
    assert "cat /proc/sys/kernel/random/boot_id" in script
    assert 'session.lock' in script and '--cpu-stage-gate resource_group' in script
    assert 'cat /sys/class/fpga_manager/fpga0/state' in script
    assert ' > /sys/class/fpga_manager/fpga0/firmware' not in script
    assert ' > /sys/class/fpga_manager/fpga0/flags' not in script


def test_transport_cannot_fall_back_to_original_startup_script():
    config=dict(host='127.0.0.1',user='root',remote_base='/tmp/experiment')
    command=lambda *args:None
    with pytest.raises(ValueError,match='execute-board'):
        transport(config,command)
    ssh=transport(config,command,execute_board=True)
    bindings=ssh.measure.__func__.__globals__
    assert bindings['session_script'] is session_script
    assert bindings['native_command'] is command
    assert ssh.qualify_environment.__func__.__module__=='board_transport_without_reprogram'
