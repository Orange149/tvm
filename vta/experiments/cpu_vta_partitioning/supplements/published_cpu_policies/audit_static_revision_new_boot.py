#!/usr/bin/env python3
"""Independent read-only capability audit after an externally changed boot."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,now
from cpu_vta_partitioning.transport import SSH
from static_revision.common import ROOT,OUTPUT,freeze
from audit_static_revision_platform import SCRIPT


def run():
    changed=read(OUTPUT/'PLATFORM_BOOT_CHANGE.json');expected=changed['observed_boot']
    dest=OUTPUT/'platform'/expected
    script=SCRIPT+'''\nprintf 'machine_model='\nif test -r /proc/device-tree/model; then tr '\\000' ' ' < /proc/device-tree/model; fi\nprintf '\\n'\nprintf 'CPU0_frequency_khz='\ncat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq\n'''
    plan=dict(expected_new_boot=expected,historical_boot=changed['expected_boot'],
        old_batch_not_resumed=True,historical_data_not_relabelled=True,read_only=True,
        script=script,source_sha256=sha256(__file__),script_source_sha256=sha256(Path(__file__).with_name('audit_static_revision_platform.py')),
        new_DNN_inferences=0,CPU_load_cases=0,no_system_parameter_changes=True,no_reboot=True)
    freeze(dest/'PLAN.json',plan)
    ssh=SSH(read(ROOT/'artifacts/cross_network_model_01/board.local.json')['board'],execute_board=True)
    if ssh.boot_id()!=expected:raise ValueError('boot changed again; capability read stopped')
    raw=ssh.idle_command(script,boot=expected,timeout=30)
    values={}
    for line in raw.splitlines():
        if '=' in line:
            k,v=line.split('=',1);values[k]=v
    record=dict(boot_id=expected,raw=raw,values=values,read_only=True,
        old_measurement_batch_unchanged=True,no_new_DNN_inferences=True,no_reboot=True,
        plan_sha256=sha256(dest/'PLAN.json'),recorded_at=now())
    write(dest/'AUDIT.json',record)
    return dict(new_boot_capability_only=True,boot_id=expected,scheduler_parameters=values,
                no_DNN_inferences=True,no_system_changes=True)


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False))
