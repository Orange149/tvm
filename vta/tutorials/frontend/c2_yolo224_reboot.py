#!/usr/bin/env python3
"""Start a genuine fresh Linux boot and redeploy byte-identical C2 assets.

The RAM-root image may regenerate its Dropbear public host key. A fresh boot
uses its own pinned known_hosts file, after checking the same live board MAC
and SD-card identity. Global SSH configuration and persistent boot assets are
never edited.
"""
import argparse,json,re,shutil,subprocess,time
from pathlib import Path
from c2_yolo224_prepare import write,sha
from c2_yolo224_board import Board,SSH,read


def neighbor(ip):
    text=subprocess.check_output(['ip','neigh','show','to',ip],text=True)
    match=re.search(r'\blladdr\s+([0-9a-f:]{17})',text)
    if not match:raise RuntimeError('live target MAC is unavailable')
    return match[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--master',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--host',default='root@192.168.1.247')
    p.add_argument('--resume-reconnect',action='store_true',help='resume a failed reconnect without issuing another reboot')
    args=p.parse_args();master=args.master.resolve();out=args.out.resolve()
    if not args.resume_reconnect:out.mkdir(parents=True,exist_ok=False)
    elif not out.is_dir() or (out/'ssh_connection.json').exists() or (out/'board_state.json').exists():
        raise RuntimeError('reconnect resume requires the original failed pre-deployment attempt')
    previous=Board(args.previous,args.host);ip=args.host.rsplit('@',1)[-1]
    module='/media/sd-mmcblk1p2/tvm_deploy/hp/u-dma-buf.ko'
    if args.resume_reconnect:
        contract=read(out/'reboot_contract.json')
        if contract['host']!=args.host or not (out/'reboot_request.json').exists():raise ValueError('missing original reboot identity')
        before={'boot_id':contract['old_boot_id']};identity=[contract['sd_cid'],contract['mac']];mac=contract['mac']
        module_hash=contract['module_sha256'];kernel=contract['kernel']
        expected_frequency=str(contract['expected_CPU_kHz']);expected_governor=contract['expected_governor'];expected_pl=str(contract['expected_PL0_Hz'])
    else:
        before=previous.preflight('independent_reboot_before')
        for line in before['raw'].splitlines():
            if 'sh -c ' not in line and any(x in line for x in ('/c2_yolo224_runner --','/vta_stage_pipeline_runner --','./tvm_rpc server')):
                raise RuntimeError('do not reboot while an accelerator workload is active')
        previous.ssh('cat /proc/cmdline; ifconfig eth0; cat /etc/network/interfaces 2>/dev/null; '
            'ls -ld /etc/dropbear /root/.ssh /root/.ssh/authorized_keys 2>/dev/null; '
            'readlink -f /root/.ssh/authorized_keys; mount | head -n 12','reboot_startup_environment')
        identity=previous.ssh('cat /sys/block/mmcblk1/device/cid; cat /sys/class/net/eth0/address','reboot_device_identity').decode().splitlines()
        driver=previous.ssh('sha256sum '+module+'; uname -r; modinfo '+module+' 2>/dev/null | head -n 8 || true','reboot_module_identity').decode().splitlines()
        module_hash=driver[0].split()[0];kernel=driver[1]
        initial=json.loads((master/'transport/formal_main_r1_M0_before.json').read_text())['stdout']
        frequency=re.search(r'\n([0-9]+)\n([a-z]+)\n',initial)
        pl_clock=re.search(r'^\s*pl0_ref\s+\d+\s+\d+\s+\d+\s+(\d+)',initial,re.M)
        if not frequency or not pl_clock:raise ValueError('first formal block lacks the frequency reference')
        expected_frequency,expected_governor=frequency.groups();expected_pl=pl_clock[1]
        mac=neighbor(ip)
        if len(identity)!=2 or identity[1].lower()!=mac:raise ValueError('SSH board and live layer-2 identity differ')
        write(out/'reboot_contract.json',{'old_boot_id':before['boot_id'],'host':args.host,'mac':mac,'sd_cid':identity[0],
            'module_sha256':module_hash,'kernel':kernel,
            'expected_CPU_kHz':int(expected_frequency),'expected_governor':expected_governor,'expected_PL0_Hz':int(expected_pl),
            'purpose':'independent Linux boot for the user-approved C2 replication',
            'no_unowned_workload_active':True,'persistent_firmware_and_SSH_configuration_modified':False,
            'controller_sha256':sha(Path(__file__))})
        command=['ssh',*previous.options,args.host,'sync; /sbin/reboot']
        try:
            result=subprocess.run(command,capture_output=True,text=True,timeout=15)
            action={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
            if result.returncode not in (0,255):raise RuntimeError('software reboot was rejected: '+result.stderr)
        except subprocess.TimeoutExpired:action={'timeout':True,'disconnect_expected':True}
        write(out/'reboot_request.json',{'argv':command,**action})
    previous_attempts=[int(x.stem.split('_')[-1]) for x in out.glob('reconnect_*.json')]
    deadline=time.monotonic()+240;attempt=max(previous_attempts,default=0);accepted=None
    first_attempt=attempt;serial_attempted=False
    if args.resume_reconnect:write(out/f'reconnect_resume_{attempt+1:02d}.json',{
        'no_additional_reboot_issued':True,'original_contract_sha256':sha(out/'reboot_contract.json')})
    while time.monotonic()<deadline:
        time.sleep(5);attempt+=1
        scan=subprocess.run(['ssh-keyscan','-T','5','-t','rsa',ip],capture_output=True,text=True)
        record={'scan_returncode':scan.returncode,'scan_stderr':scan.stderr}
        lines=[x for x in scan.stdout.splitlines() if x.startswith(ip+' ') and ' ssh-rsa ' in x]
        if lines:
            if neighbor(ip)!=mac:raise RuntimeError('board MAC changed after reboot; refuse the new endpoint')
            candidate=out/f'known_hosts_probe_{attempt:02d}';candidate.write_text('\n'.join(lines)+'\n')
            try:
                probe=subprocess.run(['ssh',*SSH,'-o','UserKnownHostsFile='+str(candidate),args.host,
                    'cat /proc/sys/kernel/random/boot_id; cat /sys/block/mmcblk1/device/cid; cat /sys/class/net/eth0/address'],
                    capture_output=True,text=True,timeout=15)
                record.update(probe_returncode=probe.returncode,stdout=probe.stdout,stderr=probe.stderr)
                value=probe.stdout.splitlines()
                if probe.returncode==0 and len(value)==3 and value[0]!=before['boot_id']:
                    if value[1]!=identity[0] or value[2].lower()!=mac:raise RuntimeError('fresh-boot hardware identity differs')
                    accepted=(candidate,value[0])
            except subprocess.TimeoutExpired:
                record['probe_timed_out']=True
        write(out/f'reconnect_{attempt:02d}.json',record)
        if accepted:break
        serial=master/'serial_connection.json';candidate_port=master/'serial_port_candidate.json'
        serial_record=serial if serial.exists() else candidate_port
        if attempt-first_attempt>=6 and not serial_attempted and serial_record.exists():
            serial_attempted=True;connection=read(serial_record)
            if connection['MAC']!=mac or connection['SD_CID']!=identity[0]:raise ValueError('serial connection belongs to a different board')
            from c2_yolo224_serial import recover
            try:
                actual=recover(out,connection['port'],ip,mac,identity[0])
                if not serial.exists():write(serial,{'port':connection['port'],'baud':115200,
                    'MAC':actual['MAC'],'SD_CID':actual['SD_CID'],
                    'actual_verified_recovery':str(out/'serial_network_recovery.json')})
            except Exception as error:
                write(out/f'serial_auto_recovery_failed_{attempt:02d}.json',{'error':str(error),
                    'no_retry':True,'manual_console_access_may_be_required':True})
    if not accepted:raise RuntimeError('board did not return with a verified new boot; manual recovery is required')
    candidate,boot=accepted;known=out/'known_hosts';shutil.copy2(candidate,known)
    fingerprint=subprocess.check_output(['ssh-keygen','-lf',str(known)],text=True).strip()
    write(out/'ssh_connection.json',{'host':args.host,'known_hosts':str(known),'public_fingerprint':fingerprint,
        'verification':'same live MAC and SD CID, actual boot_id changed after the controlled reboot',
        'global_known_hosts_modified':False})
    for name in ('package','runtime'):shutil.copytree(master/name,out/name)
    for name in ('shared_buffer_plan.json','storage_audit.json','last_reader_certificate.json',
                 'formal_protocol.json','formal_analysis_policy.json','address_observer_config.json','c2_yolo224_address_observer.so'):
        shutil.copy2(master/name,out/name)
    write(out/'frozen_asset_lineage.json',{'master':str(master),'previous':str(args.previous.resolve()),
        'old_boot_id':before['boot_id'],'new_boot_id':boot,'same_program':True,
        'runner_sha256':sha(out/'package/c2_yolo224_runner'),'manifest_sha256':sha(out/'package/manifest.json')})
    board=Board(out,args.host)
    board.ssh('set -eu\ntest "$(sha256sum '+module+" | cut -d' ' -f1)\" = "+module_hash+
        '\ntest "$(uname -r)" = '+kernel,'fresh_boot_module_and_kernel')
    board.restore();board.deploy();board.configure()
    command='cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq '+\
        '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; '+\
        "awk '$1 == \"pl0_ref\" {print $5}' /sys/kernel/debug/clk/clk_summary"
    actual=board.ssh(command,'fresh_boot_frequency_before').decode().splitlines()
    if actual[:2]!=[expected_frequency,expected_governor]:
        board.ssh('set -eu\nfor policy in /sys/devices/system/cpu/cpufreq/policy*; do\n'+
            "printf '%s' "+expected_governor+' > "$policy/scaling_governor"\n'+
            "printf '%s' "+expected_frequency+' > "$policy/scaling_setspeed"\ndone','fresh_boot_match_CPU_frequency')
    matched=board.ssh(command,'fresh_boot_frequency_after').decode().splitlines()
    if matched!=[expected_frequency,expected_governor,expected_pl]:
        raise ValueError('fresh boot frequencies differ from the frozen first boot')
    write(out/'frequency_contract.json',{'first_formal_reference':str(master/'transport/formal_main_r1_M0_before.json'),
        'before':actual,'matched':matched,'CPU_frequency_changed':actual[:2]!=matched[:2],
        'PL_clock_changed':False})
    write(out/'reboot_complete.json',{'old_boot_id':before['boot_id'],'new_boot_id':boot,
        'MAC':mac,'sd_cid':identity[0],'program_unchanged':True,'public_fingerprint':fingerprint})
    if args.resume_reconnect:write(out/'reboot_resume_complete.json',{
        'original_attempt_retained':True,'additional_reboot_issued':False,
        'reboot_complete_sha256':sha(out/'reboot_complete.json'),'new_boot_id':boot,
        'runner_sha256':sha(out/'package/c2_yolo224_runner')})
    print(json.dumps({'new_boot_id':boot,'out':str(out),'redeployed':True}),flush=True)


if __name__=='__main__':main()
