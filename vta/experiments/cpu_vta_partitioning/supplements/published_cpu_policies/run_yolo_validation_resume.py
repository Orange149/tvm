#!/usr/bin/env python3
"""Cross-boot preparation recovery; frozen scoring/execution sources stay intact.

Only isolated component/DMA evidence may cross preparation boots. All pipeline
screening and fixed confirmation sessions belong to one new execution cohort.
"""
import argparse
from pathlib import Path
import shlex
import sys
import time
from types import FunctionType

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import (digest,inventory,locked,now,read,sha256,sync_tree,verify_inventory,write)
from cpu_vta_partitioning.prepare.build import copy_checked
from cpu_vta_partitioning.transport import SSH
from yolo_transfer import traffic
from yolo_validation.common import DEFAULT_OUTPUT,DEFAULT_RUN,LOCAL_CONFIG,assert_sources,frozen

def replay_record(bundle,record,origin):
    source=record['source'];path=Path(source.get('receipt',Path(origin)/'accepted'/source['boot_id']/(source['task_id']+'.json')))
    if sha256(path)!=source['receipt_sha256']:raise ValueError('DMA receipt hash changed')
    item=read(path);doc=item['task'];raw=Path(item['raw'])
    if doc!=traffic.task(bundle,doc['component_key']) or item['boot_id']!=source['boot_id']:
        raise ValueError('DMA original task/boot mismatch')
    if (raw/'boot_id').read_text().strip()!=item['boot_id'] or (raw/'task_hash').read_text().strip()!=digest(doc):
        raise ValueError('DMA raw identity mismatch')
    verify_inventory(raw,item['files'])
    result=traffic.summary(raw,doc,bundle)
    if result!=item['summary'] or any(record.get(k)!=v for k,v in result.items()):
        raise ValueError('DMA counters or correctness no longer replay')
    return {**result,'source':{**source,'receipt':str(path.resolve())}}

def replay_index(bundle,run_dir):
    run_dir=Path(run_dir);data=read(run_dir/'traffic.json')
    programs=read(Path(bundle)/'plans/programs.json')
    expected={n for n,p in programs.items() if p['device']=='vta'}
    if data['status']!='COMPLETE' or set(data['records'])!=expected:
        raise ValueError('complete measured DMA index required')
    return {n:replay_record(bundle,r,run_dir) for n,r in data['records'].items()}

def complete_traffic(bundle,original,cohort,config,boot,*,execute_board=False):
    if not execute_board:raise ValueError('explicit --execute-board required')
    bundle,original,cohort=map(Path,(bundle,original,cohort));run=cohort/'traffic';run.mkdir(parents=True,exist_ok=True)
    original_data=read(original/'traffic/traffic.json')
    records={n:replay_record(bundle,r,original/'traffic') for n,r in original_data['records'].items()}
    measure=FunctionType(SSH.measure.__code__,{**SSH.measure.__globals__,'native_command':traffic.native_command},
        SSH.measure.__name__,SSH.measure.__defaults__,SSH.measure.__closure__)
    ssh=type('ResumedYOLOTrafficSSH',(SSH,),{'measure':measure})(read(config)['board'],execute_board=True)
    if ssh.boot_id()!=boot:raise ValueError('preparation boot changed')
    frozen(run/'run.json',dict(boot_id=boot,bundle_id=read(bundle/'manifest.json')['bundle_id'],
        controller_sha256=sha256(Path(__file__))))
    remote=ssh.deploy(bundle,run)
    from yolo_validation.components import hardware_check
    write(run/'hardware.json',hardware_check(ssh,bundle,boot),replace=True)
    configurations=read(bundle/'plans/component_configurations.json')
    keys=sorted([k for k,c in configurations.items() if c['stages'][0]['device']=='vta' and c['context_id']=='shared4'],
        key=lambda k:configurations[k]['stages'][0]['program'])
    for key in keys:
        doc=traffic.task(bundle,key);name=doc['stages'][0]['program']
        if name in records:continue
        receipt=run/'accepted'/boot/(doc['task_id']+'.json')
        if receipt.exists():
            item=read(receipt)
            if item['task']!=doc or item['boot_id']!=boot:raise ValueError('new DMA receipt boot/task mismatch')
            result=item['summary']
        else:
            raw=ssh.measure(doc,bundle,remote,run,boot)
            result=traffic.summary(raw,doc,bundle)
            write(receipt,dict(task=doc,boot_id=boot,raw=str(raw),summary=result,files=inventory(raw),accepted_at=now()))
            ssh.acknowledge(doc,run,boot)
        row={**result,'source':dict(boot_id=boot,task_id=doc['task_id'],receipt=str(receipt.resolve()),receipt_sha256=sha256(receipt))}
        records[name]=replay_record(bundle,row,run)
        data=dict(status='COMPLETE' if len(records)==78 else 'PARTIAL',accepted=len(records),required=78,
            records=records,source_boot_ids=sorted({r['source']['boot_id'] for r in records.values()}),
            validation_source_sha256=sha256(Path(__file__)),pipeline_FPS_used=False,
            cross_boot_scope='isolated DMA bytes only; original receipt boot/task identities preserved')
        write(run/'traffic.json',data,replace=True)
        print('resumed DMA accepted',len(records),'/ 78',name,flush=True)
    if len(records)!=78:raise ValueError('incomplete DMA table')
    if not (run/'traffic.json').exists():raise ValueError('missing aggregate source index')
    replay_index(bundle,run)
    return run

def run(output,original,config,*,execute_board=False):
    if not execute_board:raise ValueError('explicit --execute-board required')
    from yolo_validation import components,ranking,bundle,pipeline,analysis
    output,original=Path(output).resolve(),Path(original).resolve();prep=assert_sources(output)
    ssh=SSH(read(config)['board'],execute_board=True);boot=ssh.boot_id()
    cohort=original/'executions'/boot
    # No old pipeline result may be transferred into this execution cohort.
    if (original/'pipeline/run.json').exists():raise ValueError('old pipeline cohort exists; do not combine boots')
    started=time.monotonic()
    with locked(cohort):
        frozen(cohort/'run.json',dict(boot_id=boot,output=str(output),original_run=str(original),
            preparation_manifest_sha256=sha256(output/'PREPARATION.json'),resume_controller_sha256=sha256(Path(__file__))))
        frozen(original/'ACTIVE_EXECUTION.json',dict(run_dir=str(cohort),boot_id=boot))
        def state(stage,**fields):
            value=dict(status=stage,boot_id=boot,output=str(output),active_execution=str(cohort),updated_at=now(),**fields)
            write(cohort/'state.json',value,replace=True);write(original/'state.json',value,replace=True)
        try:
            state('VERIFYING_COMPLETED_COMPONENTS')
            records=components.replay_complete(output)
            if len(records)!=714:raise ValueError('all existing components must replay')
            print('714 component signatures reused with original boot identities; no new calibration',flush=True)
            state('DMA_TRAFFIC')
            traffic_run=complete_traffic(prep['profile_bundle'],original,cohort,config,boot,execute_board=True)
            state('RANKING')
            # Narrow adapter: only the source validator understands a multi-boot
            # DMA index. DDR arithmetic, scalar score, DP and oracle are unchanged.
            prepare=FunctionType(ranking.prepare_inputs.__code__,
                {**ranking.prepare_inputs.__globals__,'replay_traffic':replay_index},
                ranking.prepare_inputs.__name__,ranking.prepare_inputs.__defaults__,ranking.prepare_inputs.__closure__)
            prepare(output,traffic_run)
            copy_checked(Path(__file__),output/'ranking/inputs/resumption_controller.py')
            frozen(output/'ranking/inputs/resumption_protocol.json',dict(
                controller_sha256=sha256(Path(__file__)),execution_boot_id=boot,
                component_costs_reused=714,DMA_old_records_reused=64,DMA_new_records=14,
                frozen_model_sources_changed=False,old_receipts_rewritten=False,
                pipeline_scope='one new boot for all short screens and fixed long confirmations'))
            ranking.rank(output)
            state('PREPARING_PIPELINE')
            bundle.prepare(output)
            if ssh.boot_id()!=boot:raise ValueError('execution boot changed before pipeline')
            state('PIPELINE')
            result=pipeline.run(output/'pipeline_bundle',cohort/'pipeline',config,execute_board=True)
            state('ANALYZING',pipeline=result)
            report=analysis.analyze(output,cohort)
            state('COMPLETE',pipeline=result,report=report,resume_elapsed_seconds=time.monotonic()-started)
            sync_tree(cohort)
            return read(cohort/'state.json')
        except Exception as error:
            state('STOPPED_ON_ERROR',error_type=type(error).__name__,error=str(error));raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--run-dir',type=Path,default=DEFAULT_RUN);p.add_argument('--local-config',type=Path,default=LOCAL_CONFIG)
    p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if not a.execute_board:p.error('--execute-board is required')
    print(__import__('json').dumps(run(a.output,a.run_dir,a.local_config,execute_board=True),ensure_ascii=False))

if __name__=='__main__':main()
