"""Short screening, immutable winner selection, then fixed longer-window checks."""
from copy import deepcopy
from pathlib import Path
import math
import shlex
from types import FunctionType
from cpu_vta_partitioning.common import digest, inventory, now, read, sha256, verify_inventory, write
from cpu_vta_partitioning.transport import SSH
from yolo_transfer.pipeline import summary as original_summary
from .common import METHODS, frozen, protocol

def make_task(bundle,name,phase='screen'):
    bundle=Path(bundle)
    if phase not in ('qualify','screen','confirm'):raise ValueError('unknown pipeline phase')
    cfg=read(bundle/'plans/pool.json')[name];p=read(bundle/'plans/protocol.json')
    window=p[dict(qualify='qualification',screen='performance',confirm='confirmation')[phase]]
    manifest=read(bundle/'manifest.json')
    doc=dict(kind='qualify_pipeline' if phase=='qualify' else 'performance',phase=phase,
        candidate_id=name,configuration=deepcopy(cfg),stages=deepcopy(cfg['stages']),
        environment=dict(cores=4,frequency_khz=1066666,poll_sleep_ns=1000,post_start_sleep_ns=1000,
                         spin_count=0,exclude_worker0=1,bind_threads=1),
        **window,input_indices=[0],executor_sha256=manifest['runner_sha256'],bundle_id=manifest['bundle_id'],
        experiment=p['experiment'],output_mode='all_stage_dumps' if phase=='qualify' else 'all_final_hashes')
    doc['task_id']=digest(doc)[:24]
    return doc

def native_command(doc,bundle,remote_bundle,remote_job):
    from cpu_vta_partitioning.measurement.tasks import native_command as base
    if doc!=make_task(bundle,doc['candidate_id'],doc['phase']):raise ValueError('pipeline task identity differs')
    argv,inputs=base(doc,bundle,remote_bundle,remote_job)
    loader=argv.index(remote_bundle+'/runtime/ld-linux-aarch64.so.1')
    argv[loader:loader]=['TVM_BIND_THREADS=1','TVM_EXCLUDE_WORKER0=1']
    cfg=doc['configuration']
    argv[argv.index('--cpu-stage-gate')+1]=cfg['execution_mode']
    argv+=['--allow-cpu-affinity-superset','--cpu-worker-pinning','--control-cpu-affinity',
           ','.join(map(str,cfg['control_mask']))]
    if doc['phase']=='qualify':argv.append('--resource-affinity-probe')
    else:argv[argv.index('--output-mode')+1]='raw'
    if '--process-cpu-affinity' in argv or '--queue-timing' in argv:raise ValueError('unexpected binding/probe')
    return argv,inputs

def matched_ssh(config,*,execute_board=False):
    measure=FunctionType(SSH.measure.__code__,{**SSH.measure.__globals__,'native_command':native_command},
        SSH.measure.__name__,SSH.measure.__defaults__,SSH.measure.__closure__)
    return type('YOLOValidationSSH',(SSH,),{'measure':measure})(config,execute_board=execute_board)

def work_boundaries(rows,document):
    """Reconstruct stage work inventories from unchanged executor timestamps."""
    first=document['warmup'];last=first+document['scored']-1
    begin,end=rows[first]['completion_ms'],rows[last]['completion_ms']
    stages=[]
    for i in range(len(document['stages'])):
        times=[r['stage'+str(i)+'_end_ms'] for r in rows]
        if any(type(t) not in (float,int) or not math.isfinite(t) or t<0 for t in times):
            raise ValueError('invalid stage completion timestamp')
        if any(b<a for a,b in zip(times,times[1:])):raise ValueError('stage completion order differs')
        a=sum(t<=begin for t in times);b=sum(t<=end for t in times)
        stages.append(dict(stage=i,completed_at_first=a,completed_at_last=b,
            completed_between=b-a,inventory_at_first=a-(first+1),inventory_at_last=b-(last+1)))
    return dict(first_frame=first,last_frame=last,interval_count=last-first,
        first_completion_ms=begin,last_completion_ms=end,stages=stages,
        output_intervals_ms=[rows[j+1]['completion_ms']-rows[j]['completion_ms'] for j in range(first,last)])

def summary(raw,document,bundle):
    result=original_summary(raw,document,bundle)
    if document['phase']!='qualify':
        rows=[__import__('json').loads(line) for line in (Path(raw)/'result.jsonl').read_text().splitlines()]
        result['window_work']=work_boundaries(rows,document)
    return result

def accept(ssh,doc,bundle,remote,run_dir,boot):
    run_dir=Path(run_dir);receipt=run_dir/'accepted'/boot/(doc['task_id']+'.json')
    if receipt.exists():
        item=read(receipt)
        if item['task']!=doc or item['boot_id']!=boot:raise ValueError('receipt boot/task mismatch')
        verify_inventory(item['raw'],item['files'])
        if summary(item['raw'],doc,bundle)!=item['summary']:raise ValueError('receipt no longer replays')
    else:
        import time
        started=time.monotonic();started_at=now()
        raw=ssh.measure(doc,bundle,remote,run_dir,boot)
        result=summary(raw,doc,bundle)
        item=dict(task=doc,boot_id=boot,raw=str(raw),files=inventory(raw),summary=result,
            measurement_started_at=started_at,accepted_at=now(),measurement_and_collection_seconds=time.monotonic()-started)
        write(receipt,item)
        ssh.acknowledge(doc,run_dir,boot)
    # Validate the immutable raw identity even when accepting a local receipt.
    raw=Path(item['raw'])
    if ((raw/'boot_id').read_text().strip()!=boot
            or (raw/'task_hash').read_text().strip()!=digest(doc)):
        raise ValueError('raw task or boot differs')
    return item

def select_winners(lists,screen):
    if set(lists)!=set(METHODS):raise ValueError('three frozen method lists required')
    selection={};union=set()
    for key in METHODS:
        rows=lists[key]
        if len(rows)!=20 or any(r['id'] not in screen for r in rows):raise ValueError('screening pool incomplete')
        winners={}
        for budget in (1,5,20):
            # Screening ties follow each method's frozen canonical order.
            index=min(range(budget),key=lambda i:(screen[rows[i]['id']]['ii_ms'],i))
            name=rows[index]['id'];winners[str(budget)]=dict(candidate_id=name,static_rank=index+1,
                screening_ii_ms=screen[name]['ii_ms'],screening_fps=screen[name]['fps'])
            union.add(name)
        selection[key]=winners
    if len(union)>9:raise ValueError('confirmation pool exceeds fixed budget')
    return dict(methods=selection,candidates=sorted(union),selection_metric='short-screen arithmetic mean completion interval')

def freeze_winners(bundle,run_dir,boot,screen):
    bundle,run_dir=Path(bundle),Path(run_dir)
    lists=read(bundle/'plans/rankings.json')['lists']
    seal=select_winners(lists,screen)
    seal.update(boot_id=boot,bundle_id=read(bundle/'manifest.json')['bundle_id'],
        screening_receipts={name:sha256(run_dir/'accepted'/boot/(make_task(bundle,name,'screen')['task_id']+'.json')) for name in sorted(screen)},
        rankings_sha256=sha256(bundle/'plans/rankings.json'))
    frozen(run_dir/'WINNERS.json',seal)
    return seal

def run(bundle,run_dir,local_config,*,execute_board=False):
    from .validation import validate_bundle
    from .components import hardware_check
    bundle,run_dir=Path(bundle).resolve(),Path(run_dir).resolve()
    manifest=validate_bundle(bundle)
    ssh=matched_ssh(read(local_config)['board'],execute_board=execute_board)
    boot=ssh.boot_id()
    frozen(run_dir/'run.json',dict(boot_id=boot,bundle_id=manifest['bundle_id'],bundle=str(bundle)))
    frozen(run_dir/'selection_freeze.json',read(bundle/'plans/ranking_freeze.json'))
    remote=ssh.deploy(bundle,run_dir)
    write(run_dir/'hardware.json',hardware_check(ssh,bundle,boot),replace=True)
    order=read(bundle/'plans/execution_order.json');screen={};accepted=0
    for name in order:
        for phase in ('qualify','screen'):
            item=accept(ssh,make_task(bundle,name,phase),bundle,remote,run_dir,boot)
            accepted+=1
            if phase=='screen':screen[name]=item['summary']
            write(run_dir/'state.json',dict(status='SCREENING',boot_id=boot,bundle=str(bundle),
                accepted_sessions=accepted,candidate_count=len(order),updated_at=now()),replace=True)
            print('YOLO pipeline',accepted,'/',2*len(order),phase,name,
                item['summary'].get('fps','qualified'),flush=True)
    winners=freeze_winners(bundle,run_dir,boot,screen)
    # Winner list is immutable before any long-window session is scheduled.
    for i,name in enumerate(winners['candidates'],1):
        item=accept(ssh,make_task(bundle,name,'confirm'),bundle,remote,run_dir,boot)
        accepted+=1
        print('YOLO confirmation',i,'/',len(winners['candidates']),name,item['summary']['fps'],flush=True)
        write(run_dir/'state.json',dict(status='CONFIRMING',boot_id=boot,bundle=str(bundle),
            accepted_sessions=accepted,confirmation_done=i,confirmation_required=len(winners['candidates'])),replace=True)
    ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > post_run_verification.log && sync',boot=boot,timeout=600)
    result=dict(status='COMPLETE',boot_id=boot,bundle=str(bundle),accepted_sessions=accepted,
        candidate_count=len(order),confirmation_candidates=len(winners['candidates']),
        complete_DNN_inferences=29*len(order)+104*len(winners['candidates']),
        post_deployment_hashes_passed=True,single_boot_only=True,finished_at=now())
    write(run_dir/'state.json',result,replace=True)
    return result
