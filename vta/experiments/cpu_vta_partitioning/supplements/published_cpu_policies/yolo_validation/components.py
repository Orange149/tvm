"""Fill only missing isolated signatures, then keep a multi-cohort source index."""
from pathlib import Path
import shlex
import time
from cpu_vta_partitioning.common import (inventory, locked, now, read, sha256, sync_tree, write)
from yolo_transfer import profile, traffic
from .common import assert_sources, frozen
from .evidence import replay_imports, receipt_record

def hardware_check(ssh,bundle,boot):
    ssh.wait_idle(boot)
    raw=ssh.idle_command('sha256sum /lib/firmware/vta_hpc.bit\n'
        'cat /sys/class/fpga_manager/fpga0/state\ncat /sys/class/u-dma-buf/udmabuf0/size',boot=boot)
    lines=raw.splitlines()
    if (len(lines)!=3 or lines[0].split()[0]!=sha256(Path(bundle)/'firmware/vta_hpc.bit')
            or lines[1]!='operating' or lines[2]!='201326592'):
        raise ValueError('hardware identity differs; no implicit hardware retuning')
    frequencies=ssh.command('for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq; do cat "$f"; done').split()
    if frequencies!=['1066666']*4:
        raise ValueError('CPU frequency differs; no automatic retuning')
    return dict(boot_id=boot,raw=raw,frequency_khz=frequencies)

def complete(output,run_dir,local_config,*,execute_board=False):
    output,run_dir=Path(output).resolve(),Path(run_dir).resolve()
    prep=assert_sources(output);bundle=Path(prep['profile_bundle'])
    records=replay_imports(output)
    ssh=profile.matched_ssh(read(local_config)['board'],execute_board=execute_board)
    with locked(run_dir):
        boot=ssh.boot_id()
        identity=dict(boot_id=boot,bundle_id=read(bundle/'manifest.json')['bundle_id'])
        frozen(run_dir/'run.json',identity)
        remote=ssh.deploy(bundle,run_dir)
        write(run_dir/'hardware.json',hardware_check(ssh,bundle,boot),replace=True)
        order=read(bundle/'plans/profile_order.json')
        started=time.monotonic()
        for key in order:
            if key in records:continue
            doc=profile.make_task(bundle,key)
            receipt=run_dir/'accepted'/boot/(doc['task_id']+'.json')
            if receipt.exists():
                result=receipt_record(bundle,key,receipt)
            else:
                raw=ssh.measure(doc,bundle,remote,run_dir,boot)
                result=profile.summary(raw,doc,bundle)
                write(receipt,dict(task=doc,boot_id=boot,raw=str(raw),files=inventory(raw),summary=result,
                    accepted_at=now()))
                ssh.acknowledge(doc,run_dir,boot)
                result=receipt_record(bundle,key,receipt)
            records[key]=result
            table=dict(status='COMPLETE' if len(records)==len(order) else 'PARTIAL',accepted=len(records),
                required=len(order),costs=records,cross_boot_sources_allowed=True,
                source_boot_ids=sorted({r['source']['boot_id'] for r in records.values()}),pipeline_FPS_used=False)
            write(output/'isolated_costs.json',table,replace=True)
            write(run_dir/'state.json',dict(status=table['status'],accepted=len(records),required=len(order),
                newly_measured=len(records)-prep_import_count(output),updated_at=now()),replace=True)
            print('YOLO components',len(records),'/',len(order),doc['stages'][0]['signature'],flush=True)
        if set(records)!=set(order):raise ValueError('incomplete component table')
        table=dict(status='COMPLETE',accepted=len(records),required=len(order),costs=records,
            cross_boot_sources_allowed=True,source_boot_ids=sorted({r['source']['boot_id'] for r in records.values()}),pipeline_FPS_used=False)
        write(output/'isolated_costs.json',table,replace=True)
        write(run_dir/'state.json',dict(status='COMPLETE',accepted=len(records),required=len(order),
            newly_measured=len(records)-prep_import_count(output),boot_id=boot,elapsed_seconds=time.monotonic()-started),replace=True)
        ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > post_run_verification.log && sync',boot=boot,timeout=600)
        sync_tree(run_dir)
    return dict(accepted=len(records),newly_measured=len(records)-prep_import_count(output),boot_id=boot)

def prep_import_count(output):
    return read(Path(output)/'imported_costs.json')['accepted']

def replay_complete(output):
    output=Path(output);prep=assert_sources(output);bundle=Path(prep['profile_bundle'])
    table=read(output/'isolated_costs.json');keys=read(bundle/'plans/component_configurations.json')
    profile.validate(bundle)
    if table['status']!='COMPLETE' or set(table['costs'])!=set(keys):
        raise ValueError('complete 714-component source index required')
    for key,row in table['costs'].items():
        if receipt_record(bundle,key,row['source']['receipt'],expected_record=row)!=row:
            raise ValueError('component source index no longer replays')
    return table['costs']
