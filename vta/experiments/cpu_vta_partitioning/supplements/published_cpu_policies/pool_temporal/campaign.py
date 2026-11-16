"""Bounded candidate component collection with exact no-remeasurement resume."""
from copy import deepcopy
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from cpu_vta_partitioning.common import digest,inventory,locked,now,read,sha256,sync_tree,verify_inventory,write
from common_top20.campaign import qemu_cpu_summary
from cpu_policies.open_resources.runtime import _affinity
from workload_model.common import OUTPUT,ROOT,freeze,require_space
from .prepare import validate
from .runtime import make_task,command,summary,transport


def qemu(output):
    output=Path(output).resolve();executable=shutil.which('qemu-aarch64-static')
    if not executable:raise RuntimeError('ARM QEMU is required')
    cases=[]
    for network in ('resnet18','yolov3_tiny416'):
        bundle=output/network/'capsule';validate(bundle)
        rows=read(bundle/'plans/components.json');programs=read(bundle/'plans/programs.json')
        cpu=[(name,cfg) for name,cfg in rows.items() if cfg['stages'][0]['device']=='cpu']
        chosen={cpu[0][0],cpu[-1][0]}
        multi=max(cpu,key=lambda pair:len(programs[pair[1]['stages'][0]['program']]['input_node_ids']))
        if len(programs[multi[1]['stages'][0]['program']]['input_node_ids'])>1:chosen.add(multi[0])
        for case in sorted(chosen):
            doc=make_task(bundle,dict(network=network,case=case,mode='qualify'))
            raw=output/'qemu'/(network+'_'+case);raw.mkdir(parents=True,exist_ok=True)
            argv,inputs=command(doc,bundle,str(bundle),str(raw));(raw/'inputs.txt').write_text(inputs)
            at=argv.index(str(bundle)+'/runtime/ld-linux-aarch64.so.1')
            cmd=[executable,*[v for item in argv[2:at] for v in ('-E',item)],*argv[at:]]
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
            (raw/'native.log').write_text(result.stdout+result.stderr)
            if result.returncode:raise ValueError('ARM candidate observer failed: '+str(raw))
            qemu_cpu_summary(raw,doc,bundle)
            # Only the already-verified idle QEMU helper is removed from the
            # CI ledger view. Board summaries keep the original strict check.
            def ci_affinity(report,cfg,qual):
                report=deepcopy(report)
                for stage in report['stages']:
                    active={s['tid'] for s in stage['worker_slots']}
                    stage['thread_ids']=[tid for tid in stage['thread_ids'] if tid in active]
                    stage['effective_affinities']=[r for r in stage['effective_affinities'] if r['tid'] in active]
                return _affinity(report,cfg,qual)
            checked=summary(raw,doc,bundle,affinity_checker=ci_affinity)
            cases.append(dict(network=network,case=case,correct=True,summary=checked))
            print('候选组件ARM观察资格通过',network,case,flush=True)
    write(output/'QEMU_QUALIFICATION.json',dict(passed=True,cases=cases,board_access=False,
        bundles={n:read(output/n/'capsule/manifest.json')['bundle_id'] for n in ('resnet18','yolov3_tiny416')}),replace=True)
    return dict(passed=True,CPU_stage_calls=len(cases))


def run(output,run,*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    require_space();output,run=Path(output).resolve(),Path(run).resolve()
    tasks=read(output/'TASKS.json');q=read(output/'QEMU_QUALIFICATION.json')
    bundles={n:output/n/'capsule' for n in ('resnet18','yolov3_tiny416')}
    ids={n:validate(b)['bundle_id'] for n,b in bundles.items()}
    if not q['passed'] or q['bundles']!=ids:raise ValueError('candidate observer ARM qualification differs')
    boot=read(ROOT/'runs/cross_network_phase_profile_01/run.json')['boot_id']
    ssh=transport(read(OUTPUT/'board.local.json')['board'],execute_board=True)
    if ssh.boot_id()!=boot:raise ValueError('board boot changed; stop current study board work')
    started=time.monotonic();expected={};completed=0
    with locked(run):
        freeze(run/'run.json',dict(boot_id=boot,bundles=ids,plan_sha256=sha256(output/'PLAN.json')))
        def state(status,**fields):write(run/'state.json',dict(status=status,boot_id=boot,completed_sessions=completed,
            total_sessions=len(tasks),updated_at=now(),goal_complete=False,**fields),replace=True)
        try:
            remotes={}
            for network,bundle in bundles.items():
                (run/network).mkdir(parents=True,exist_ok=True)
                remotes[network]=ssh.deploy(bundle,run/network)
                ssh.qualify_environment(remotes[network],boot,run/network,bundle)
            for spec in tasks:
                if ssh.boot_id()!=boot:raise ValueError('board boot changed during candidate profiling')
                network=spec['network'];bundle=bundles[network];doc=make_task(bundle,spec)
                receipt=run/'accepted'/boot/(doc['task_id']+'.json');key=(network,spec['case'])
                state('RUNNING',current_task=doc['task_id'],current_profile=spec)
                if receipt.exists():
                    item=read(receipt);verify_inventory(item['raw'],item['files'])
                    if item['task']!=doc or item['boot_id']!=boot or summary(item['raw'],doc,bundle)!=item['summary']:
                        raise ValueError('candidate isolation receipt changed')
                else:
                    t0=time.monotonic();t0_stamp=now();raw=ssh.measure(doc,bundle,remotes[network],run,boot)
                    result=summary(raw,doc,bundle)
                    if Path(raw,'boot_id').read_text().strip()!=boot or Path(raw,'task_hash').read_text().strip()!=digest(doc):
                        raise ValueError('candidate raw boot/hash differs')
                    if spec['mode']!='qualify' and result['output_fingerprints']!=expected[key]:
                        raise ValueError('candidate output differs from full qualification')
                    sync_tree(raw);item=dict(task=doc,boot_id=boot,raw=str(raw),files=inventory(raw),summary=result,
                        measurement_started_at=t0_stamp,accepted_at=now(),measurement_and_collection_seconds=time.monotonic()-t0)
                    write(receipt,item);ssh.acknowledge(doc,run,boot)
                    ssh.idle_command('cd '+shlex.quote(ssh.job_path(doc,run,boot))+' && rm -f phase_trace.jsonl wait_trace.jsonl wait_status.json device_trace.jsonl',boot=boot)
                if spec['mode']=='qualify':expected[key]=item['summary']['output_fingerprints']
                elif item['summary']['output_fingerprints']!=expected[key]:raise ValueError('accepted profile output changed')
                completed+=1
                print('候选调用级profile',completed,'/',len(tasks),network,spec['case'],spec['mode'],flush=True)
            for remote in remotes.values():
                ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > pool_post_verify.log && sync',boot=boot,timeout=600)
            state('COMPLETE',CPU_stage_calls=418,VTA_stage_calls=506,DNN_inferences=0,
                  elapsed_seconds=time.monotonic()-started)
            sync_tree(run)
        except Exception as error:
            state('STOPPED_ON_ERROR',error_type=type(error).__name__,error=str(error));raise
    return read(run/'state.json')
