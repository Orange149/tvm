"""Single-boot formal comparison, durable acceptance, deterministic resumption."""
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from types import FunctionType

from cpu_vta_partitioning.common import digest,inventory,locked,now,read,sha256,sync_tree,verify_inventory,write
from cpu_policies.open_resources.domain import component
from workload_model.common import OUTPUT,ROOT,freeze,require_space
from .prepare import validate
from .runtime import task,command,summary,transport


def qemu_cpu_summary(raw,document,bundle):
    """Output/active-slot CI checks, without absent phase or waiting observers."""
    from cpu_vta_partitioning.measurement import frames,correctness
    report=read(Path(raw)/'affinity.json');ignored=[]
    if report['cpu_worker_pinning'] is not True or len(report['stages'])!=len(document['stages']):
        raise ValueError('ARM qualification mode or stage count differs')
    for cfg,stage in zip(document['stages'],report['stages']):
        slots=sorted(stage['worker_slots'],key=lambda s:s['slot'])
        if (stage['requested_cpus']!=cfg['mask'] or stage['active_runtime_threads']!=cfg['threads']
                or [s['slot'] for s in slots]!=list(range(cfg['threads']))
                or len({s['tid'] for s in slots})!=cfg['threads']
                or slots[0]['tid']!=stage['caller_tid']):
            raise ValueError('ARM active logical slot or caller differs')
        active={s['tid'] for s in slots}
        for slot,core in zip(slots,cfg['worker_core_map']):
            effective=[r for r in stage['effective_affinities'] if r['tid']==slot['tid']]
            if slot['cpus']!=[core] or len(effective)!=1 or effective[0]['cpus']!=[core]:
                raise ValueError('ARM actual logical thread affinity differs')
        for helper in stage['effective_affinities']:
            if helper['tid'] in active:continue
            if helper['cpu_time_delta_ms']!=0:raise ValueError('nonlogical ARM emulation helper consumed CPU')
            ignored.append(helper)
    rows=frames(Path(raw)/'result.jsonl',1);correctness(rows,document,bundle,raw)
    return dict(correct=True,complete_outputs_checked=True,logical_slots_checked=True,
                CI_only_idle_emulation_helpers=ignored,phase_or_wait_observer_loaded=False)


def qemu(output):
    output=Path(output).resolve();qemu=shutil.which('qemu-aarch64-static')
    if not qemu:raise RuntimeError('QEMU required')
    results=[]
    for network in ('resnet18','yolov3_tiny416'):
        bundle=output/network/'capsule';validate(bundle)
        pool=read(bundle/'plans/rankings.json')['pool'];programs=read(bundle/'plans/programs.json')
        names=sorted({s['program'] for c in pool.values() for s in c['stages'] if s['device']=='cpu'},
                     key=lambda n:(programs[n]['begin'],programs[n]['end'],n))
        for name in [names[0],names[-1]]:
            choices=[s for c in pool.values() for s in c['stages'] if s['program']==name]
            row=max(choices,key=lambda s:(s['threads'],s['mask']));cfg=component(row)
            # Component qualification uses the same frozen original executable.
            doc=task(bundle,next(iter(pool.values())),'qualify')
            doc.update(configuration=cfg,stages=cfg['stages'],kind='qualify_component')
            doc.pop('task_id');doc['task_id']=digest(doc)[:24]
            from cpu_vta_partitioning.measurement.tasks import native_command as base
            raw=output/'qemu'/(network+'_'+name);raw.mkdir(parents=True,exist_ok=True)
            argv,inputs=base(doc,bundle,str(bundle),str(raw));(raw/'inputs.txt').write_text(inputs)
            at=argv.index(str(bundle)+'/runtime/ld-linux-aarch64.so.1')
            argv[at:at]=['TVM_BIND_THREADS=1','TVM_EXCLUDE_WORKER0=1']
            argv+=['--cpu-worker-pinning','--allow-cpu-affinity-superset','--resource-affinity-probe']
            at=argv.index(str(bundle)+'/runtime/ld-linux-aarch64.so.1')
            call=[qemu,*[x for item in argv[2:at] for x in ('-E',item)],*argv[at:]]
            result=subprocess.run(call,capture_output=True,text=True,timeout=180)
            (raw/'native.log').write_text(result.stdout+result.stderr)
            if result.returncode:raise ValueError('formal ARM output qualification failed: '+str(raw))
            value=qemu_cpu_summary(raw,doc,bundle)
            results.append(dict(network=network,program=name,summary=value))
            print('正式执行器ARM资格通过',network,name,flush=True)
    write(output/'QEMU_QUALIFICATION.json',dict(passed=True,cases=results,board_access=False,
        bundles={n:read(output/n/'capsule/manifest.json')['bundle_id'] for n in ('resnet18','yolov3_tiny416')}),replace=True)
    return dict(passed=True,CPU_calls=len(results))


def accept(ssh,document,bundle,remote,run,boot,expected=None):
    path=Path(run)/'accepted'/boot/(document['task_id']+'.json')
    if path.exists():
        item=read(path)
        if item['task']!=document or item['boot_id']!=boot:raise ValueError('accepted comparison identity changed')
        verify_inventory(item['raw'],item['files'])
        value=summary(item['raw'],document,bundle)
        if value!=item['summary']:raise ValueError('accepted comparison no longer replays')
    else:
        started=time.monotonic();started_at=now()
        raw=ssh.measure(document,bundle,remote,run,boot);value=summary(raw,document,bundle)
        if Path(raw,'boot_id').read_text().strip()!=boot or Path(raw,'task_hash').read_text().strip()!=digest(document):
            raise ValueError('raw comparison boot or task hash differs')
        if expected is not None and value['output_fingerprints']!=expected:
            raise ValueError('formal output differs from complete qualification')
        sync_tree(raw)
        item=dict(task=document,boot_id=boot,raw=str(raw),files=inventory(raw),summary=value,
                  measurement_started_at=started_at,accepted_at=now(),
                  measurement_and_collection_seconds=time.monotonic()-started)
        write(path,item);ssh.acknowledge(document,run,boot)
    if expected is not None and value['output_fingerprints']!=expected:
        raise ValueError('accepted formal output differs from qualification')
    return item


def run(output,run_dir,*,phase='short',execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    require_space();output,run_dir=Path(output).resolve(),Path(run_dir).resolve()
    proof=read(output/'QEMU_QUALIFICATION.json')
    bundles={n:output/n/'capsule' for n in ('resnet18','yolov3_tiny416')}
    ids={n:validate(b)['bundle_id'] for n,b in bundles.items()}
    if not proof['passed'] or proof['bundles']!=ids:raise ValueError('ARM qualification identity differs')
    boot=read(ROOT/'runs/cross_network_phase_profile_01/run.json')['boot_id']
    ssh=transport(read(OUTPUT/'board.local.json')['board'],execute_board=True)
    if ssh.boot_id()!=boot:raise ValueError('board boot changed; never merge with the current study')
    selected=read(output/'LONG_SELECTION_FREEZE.json') if phase=='long' else None
    started=time.monotonic()
    todo=[]
    for network,bundle in bundles.items():
        rows=read(bundle/'plans/rankings.json');order=selected['networks'][network]['configurations'] if selected else rows['execution_order']
        for identity in order:todo.append((network,rows['pool'][identity]))
    with locked(run_dir):
        freeze(run_dir/'run.json',dict(boot_id=boot,bundles=ids,rank_seal_sha256=sha256(output/'RANK_FREEZE.json')))
        def status(name,**extra):write(run_dir/'state.json',dict(status=name,boot_id=boot,phase=phase,goal_complete=False,updated_at=now(),**extra),replace=True)
        try:
            remotes={}
            for network,bundle in bundles.items():
                (run_dir/network).mkdir(parents=True,exist_ok=True)
                remotes[network]=ssh.deploy(bundle,run_dir/network)
                ssh.qualify_environment(remotes[network],boot,run_dir/network,bundle)
            completed=0
            for network,cfg in todo:
                if ssh.boot_id()!=boot:raise ValueError('boot changed during formal comparison')
                bundle=bundles[network];remote=remotes[network]
                status('RUNNING',completed_configurations=completed,total_configurations=len(todo),current_network=network,current_configuration=cfg['id'])
                q=task(bundle,cfg,'qualify')
                if phase=='long' and not (run_dir/'accepted'/boot/(q['task_id']+'.json')).exists():
                    raise ValueError('long review requires matching short qualification')
                qualified=accept(ssh,q,bundle,remote,run_dir,boot)
                performance=task(bundle,cfg,phase)
                accept(ssh,performance,bundle,remote,run_dir,boot,qualified['summary']['output_fingerprints'])
                completed+=1
                print('共同Top20实测',completed,'/',len(todo),network,cfg['id'],phase,flush=True)
            for remote in remotes.values():
                ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > common_post_verify.log && sync',boot=boot,timeout=600)
            status('SHORT_COMPLETE' if phase=='short' else 'LONG_COMPLETE',completed_configurations=completed,total_configurations=len(todo),
                   elapsed_seconds=time.monotonic()-started)
            sync_tree(run_dir)
        except Exception as error:
            status('STOPPED_ON_ERROR',error_type=type(error).__name__,error=str(error));raise
    return read(run_dir/'state.json')
