#!/usr/bin/env python3
"""Read scheduler metadata only; no workload, parameter changes, or reboot."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,now
from cpu_vta_partitioning.transport import SSH
from static_revision.common import ROOT,OUTPUT,freeze

SCRIPT=r'''
uname -r
for name in sched_latency_ns sched_min_granularity_ns sched_wakeup_granularity_ns sched_migration_cost_ns sched_tunable_scaling sched_child_runs_first; do
  path=/proc/sys/kernel/$name
  printf '%s=' "$name"
  if test -r "$path"; then cat "$path"; else printf 'UNAVAILABLE\n'; fi
done
printf 'online_cores='
cat /sys/devices/system/cpu/online
printf 'sched_debug_readable='
if test -r /proc/sched_debug; then printf 'yes\n'; else printf 'no\n'; fi
if test -r /proc/config.gz; then
  zcat /proc/config.gz | grep -E '^(CONFIG_(HZ|HZ_[0-9]+|PREEMPT|PREEMPT_NONE|PREEMPT_VOLUNTARY|SCHED_DEBUG|SCHED_HRTICK|FAIR_GROUP_SCHED|CGROUP_SCHED)=|# CONFIG_SCHED_DEBUG is not set)'
fi
'''


def run(execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    baseline=read(ROOT/'runs/yolo_scheduler_trace_repair_01/state.json')['boot_id']
    plan=dict(operation='read-only scheduler metadata',expected_boot_id=baseline,script=SCRIPT,
              source_sha256=sha256(__file__),DNN_inferences=0,CPU_workload_cases=0,
              sysctls_changed=False,kernel_changed=False,reboot_forbidden=True)
    freeze(OUTPUT/'PLATFORM_AUDIT_PLAN.json',plan)
    ssh=SSH(read(ROOT/'artifacts/cross_network_model_01/board.local.json')['board'],execute_board=True)
    boot=ssh.boot_id()
    if boot!=baseline:
        write(OUTPUT/'PLATFORM_BOOT_CHANGE.json',dict(expected_boot=baseline,observed_boot=boot,
            no_measurement_started=True,old_batch_not_resumed=True,recorded_at=now()))
        raise ValueError('boot changed; old experimental batch will not be resumed')
    raw=ssh.idle_command(SCRIPT,boot=boot,timeout=30)
    values={}
    for line in raw.splitlines():
        if '=' in line:
            key,value=line.split('=',1);values[key]=value
    result=dict(boot_id=boot,raw=raw,values=values,source_sha256=sha256(__file__),
        read_only=True,new_DNN_inferences=0,CPU_workload_cases=0,sysctls_changed=False,recorded_at=now())
    write(OUTPUT/'PLATFORM_AUDIT.json',result)
    return dict(boot_unchanged=True,scheduler_parameters={k:v for k,v in values.items() if k.startswith('sched_')},
                new_DNN_inferences=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.execute_board),ensure_ascii=False))
