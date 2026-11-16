"""One zero-DNN instruction-event capability check; no retry after start."""
from pathlib import Path
import argparse
import json
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cpu_vta_partitioning.common import read, sha256, write
from cpu_vta_partitioning.transport import SSH

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/completion_breakpoint_capability_01'
BOOT='6018b3b7-bb29-4f80-92c1-0bfd8c81bb99'


def validate():
    b=read(OUT/'BUILD.json')
    if sha256(b['source'])!=b['source_sha256'] or sha256(OUT/'completion_breakpoint_probe')!=b['binary_sha256']:
        raise ValueError('probe identity differs')
    tc=b['toolchain']
    if sha256(tc['sdk_setup'])!=tc['sdk_setup_sha256']:
        raise ValueError('SDK setup changed')
    return b


def qemu():
    b=validate();r=subprocess.run(['qemu-aarch64-static','-L',b['toolchain']['sysroot'],
        str(OUT/'completion_breakpoint_probe')],capture_output=True,text=True,timeout=5)
    result=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,
        scope='CPU program loading only; QEMU cannot prove real kernel hardware-breakpoint support')
    write(OUT/'QEMU.json',result,replace=True)
    if r.returncode or json.loads(r.stdout)['status']!='UNAVAILABLE':
        raise ValueError('unexpected QEMU self-test result')
    return result


def board(*, execute_board=False):
    b=validate();receipt=OUT/'BOARD_RECEIPT.json'
    if receipt.exists():return dict(already_completed=True,**read(receipt))
    ssh=SSH(read(ROOT/'artifacts/cross_network_model_01/board.local.json')['board'],execute_board=execute_board)
    if ssh.boot_id()!=BOOT:raise ValueError('boot changed; no native test executed')
    capsule=ROOT/'artifacts/cpu_exclusion_gap_01/yolov3_tiny416/capsule'
    manifest=read(capsule/'manifest.json')
    remote_bundle=ssh.config['remote_base'].rstrip('/')+'/bundles/'+manifest['bundle_id']
    remote=ssh.config['remote_base'].rstrip('/')+'/capability/completion_breakpoint_01'
    q=shlex.quote
    ssh.idle_command('mkdir -p '+q(remote)+'; test ! -e '+q(remote+'/started'),boot=BOOT)
    ssh.command('cat > '+q(remote+'/probe.upload'),input_path=OUT/'completion_breakpoint_probe')
    script='\n'.join(['set -eu',
        'test "$(sha256sum '+q(remote+'/probe.upload')+' | cut -d \' \' -f 1)" = '+q(b['binary_sha256']),
        'test ! -e '+q(remote+'/started'),
        'mv '+q(remote+'/probe.upload')+' '+q(remote+'/probe'),
        'chmod 700 '+q(remote+'/probe'),
        'printf started > '+q(remote+'/started'),
        'set +e',
        q(remote_bundle+'/runtime/ld-linux-aarch64.so.1')+' --library-path '+q(remote_bundle+'/runtime')+
            ' '+q(remote+'/probe')+' > '+q(remote+'/stdout.json')+' 2> '+q(remote+'/stderr.txt'),
        'code=$?', 'printf "%s\\n" "$code" > '+q(remote+'/exit_code'),
        'cat '+q(remote+'/stdout.json')])
    local=OUT/'NATIVE_STARTED.json'
    if local.exists():raise ValueError('prior native start requires read-only recovery; never rerun automatically')
    write(local,dict(expected_boot=BOOT,remote=remote,binary_sha256=b['binary_sha256'],
        source_sha256=sha256(__file__),DNN_inferences=0,atomic_operations=8))
    stdout=ssh.idle_command(script,boot=BOOT,timeout=15)
    after=ssh.boot_id()
    if after!=BOOT:raise ValueError('boot changed during capability test')
    exit_code=int(ssh.command('cat '+q(remote+'/exit_code')).strip())
    stderr=ssh.command('cat '+q(remote+'/stderr.txt'))
    write(OUT/'RAW_STDOUT.json',dict(stdout=stdout,stderr=stderr,exit_code=exit_code))
    result=json.loads(stdout)
    if exit_code!=0 or result['status'] not in ('VALIDATED','UNAVAILABLE'):
        raise ValueError('native instruction-event test invalid; raw result preserved, no retry')
    receipt_data=dict(boot_id=BOOT,binary_sha256=b['binary_sha256'],result=result,
        raw_stdout_sha256=sha256(OUT/'RAW_STDOUT.json'),new_DNN_inferences=0,
        new_native_selftest_processes=1,FPGA_access=False,system_parameters_changed=False,
        old_runtime_executor_modules_untouched=True,no_automatic_repeat=True)
    write(receipt,receipt_data)
    return receipt_data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('validate','qemu','board'))
    p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    r=board(execute_board=a.execute_board) if a.command=='board' else globals()[a.command]()
    print(json.dumps(r if a.command!='validate' else dict(valid=True,binary_sha256=r['binary_sha256']),ensure_ascii=False,indent=2))
