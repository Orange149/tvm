"""Twelve bounded VTA component sessions; no pipeline or ranking changes."""
from copy import deepcopy
import json
from pathlib import Path
import shlex
import shutil
import time
from types import FunctionType

from cpu_vta_partitioning.common import digest,inventory,locked,now,read,sha256,sync_tree,verify_inventory,write
from cpu_vta_partitioning.measurement.tasks import native_command as base_command
from cpu_vta_partitioning.transport import SSH
from cpu_policies.open_resources.domain import component
from phase_profile.runtime import make_task as phase_task,phase_summary,summary as phase_summary_all
from workload_model.common import OUTPUT,ROOT,freeze,require_space


def task(bundle,spec):
    bundle=Path(bundle)
    m=read(bundle/'manifest.json')
    base=dict(network=spec['network'],scope='component',case=spec['case'],phase='clean',probe=False,runner='phase')
    doc=phase_task(bundle,base);doc.pop('task_id')
    qualify=spec['mode']=='qualify'
    doc.update(kind='qualify_component' if qualify else 'calibration',warmup=0 if qualify else 2,
               scored=1 if qualify else 8,tail=0,experiment='cross_network_device_phase_01',
               device_spec=spec,device_observer=spec['mode']!='clean',
               device_observer_sha256=m['device_observer_sha256'] if spec['mode']!='clean' else None)
    doc['task_id']=digest(doc)[:24]
    return doc


def command(doc,bundle,remote,job):
    if doc!=task(bundle,doc['device_spec']):raise ValueError('device task changed')
    argv,inputs=base_command(doc,bundle,remote,job)
    argv[argv.index(remote+'/runner/vta_stage_pipeline_runner')]=remote+'/runner/vta_stage_pipeline_runner_phase'
    at=argv.index(remote+'/runtime/ld-linux-aarch64.so.1')
    argv[at:at]=['TVM_BIND_THREADS=1','TVM_EXCLUDE_WORKER0=1']
    argv+=['--allow-cpu-affinity-superset','--resource-affinity-probe',
           '--control-cpu-affinity','0,1,2,3']
    if doc['configuration']['affinity_mode']=='pinned':
        argv.append('--cpu-worker-pinning')
    if doc['device_observer']:
        i=next(i for i,s in enumerate(argv) if s.startswith('LD_PRELOAD='))
        argv[i]='LD_PRELOAD='+remote+'/runner/libdevice_phase.so:'+remote+'/runtime/libvta.so'
        argv[at:at]=['VTA_DEVICE_PHASE_TRACE='+job+'/device_trace.jsonl']
        argv+=['--phase-event-json',job+'/phase_trace.jsonl']
    return argv,inputs


def validate_events(rows,phases,doc):
    if not rows or rows[0]['kind']!='status' or rows[0]['overflow'] or rows[0]['events']!=len(rows)-1:
        raise ValueError('missing or overflowed device observer')
    if not rows[0]['provider'].endswith('/runtime/libvta.so'):
        raise ValueError('device call resolved to the wrong provider')
    runs=[r for r in phases if r['operation']=='run'];mapped=[]
    for row in rows[1:]:
        if row['kind']!='device' or row['result']!=0 or row['instructions']<1:
            raise ValueError('device run failed or empty')
        if not (0<=row['cpu_start_ns']<=row['cpu_end_ns'] and 0<=row['wall_start_ns']<=row['wall_end_ns']):
            raise ValueError('invalid device clocks')
        if row['cpu_end_ns']-row['cpu_start_ns']>row['wall_end_ns']-row['wall_start_ns']+10000:
            raise ValueError('device CPU exceeds caller wall interval')
        group=[r for r in runs if r['tid']==row['tid'] and r['wall_start_ns']<=row['wall_start_ns']
               and row['wall_end_ns']<=r['wall_end_ns']]
        if len(group)!=1:
            raise ValueError('unmapped or ambiguous device run; initialization is not measured work')
        run=group[0]
        if not run['cpu_start_ns']<=row['cpu_start_ns']<=row['cpu_end_ns']<=run['cpu_end_ns']:
            raise ValueError('device CPU clock escapes caller run')
        mapped.append({**row,'call':run['call']})
    expected=set(range(doc['warmup']+doc['scored']+doc['tail']))
    if {r['call'] for r in mapped}!=expected:
        raise ValueError('a stage invocation has no real device call')
    for call in expected:
        events=sorted([r for r in mapped if r['call']==call],key=lambda r:r['wall_start_ns'])
        if any(a['wall_end_ns']>b['wall_start_ns'] for a,b in zip(events,events[1:])):
            raise ValueError('overlapping device runs in an isolated component')
    return dict(device_calls=len(mapped),provider=rows[0]['provider'],mapped=mapped)


def summarize(raw,doc,bundle):
    raw=Path(raw)
    def phases(path,document,affinity):
        adapted=deepcopy(document);adapted['spec']['probe']=document['device_observer']
        return phase_summary(path,adapted,affinity)
    check=FunctionType(phase_summary_all.__code__,{**phase_summary_all.__globals__,'phase_summary':phases})
    result=check(raw,doc,bundle)
    if doc['device_observer']:
        rows=[json.loads(s) for s in (raw/'device_trace.jsonl').read_text().splitlines()]
        spans=[json.loads(s) for s in (raw/'phase_trace.jsonl').read_text().splitlines()]
        result['device']=validate_events(rows,spans,doc)
    elif (raw/'device_trace.jsonl').exists():
        raise ValueError('unrequested device observer')
    return result


def prepare(output):
    output=Path(output).resolve();require_space();build=read(output/'build/BUILD.json');plans=[]
    for network in ('resnet18','yolov3_tiny416'):
        parent=OUTPUT/'phase_profile'/network/'capsule';old=read(parent/'manifest.json');bundle=output/network/'capsule'
        if not (bundle/'manifest.json').exists():
            for leaf,value in old['files'].items():
                source=parent/leaf;dest=bundle/leaf;dest.parent.mkdir(parents=True,exist_ok=True)
                if sha256(source)!=value['sha256']:raise ValueError('parent immutable asset changed')
                if leaf.startswith(('programs/','runtime/','firmware/','runner/','inputs/')):
                    import os
                    os.link(source,dest)
                else:shutil.copy2(source,dest)
            programs=read(bundle/'plans/programs.json')
            choices={}
            for row in read(parent/'plans/configurations.json')['pipelines']['overlap']['stages']:
                if row['device']=='vta':choices[row['program']]=component(row)
            cfg=read(bundle/'plans/configurations.json');cfg['components']=choices
            write(bundle/'plans/configurations.json',cfg,replace=True)
            shutil.copy2(output/'build/libdevice_phase.so',bundle/'runner/libdevice_phase.so')
            controllers={str(p):sha256(p) for p in Path(__file__).parent.iterdir() if p.suffix in ('.py','.cc')}
            controllers[str(ROOT/'run_device_phase.py')]=sha256(ROOT/'run_device_phase.py')
            write(bundle/'plans/device_sources.json',dict(parent_manifest_sha256=sha256(parent/'manifest.json'),
                build=build,controllers=controllers))
            payload=read(bundle/'plans/board_payload.json')
            payload['files']['runner/libdevice_phase.so']=inventory(bundle/'runner')['libdevice_phase.so']
            payload['unpacked_bytes']=sum(v['bytes'] for v in payload['files'].values())
            write(bundle/'plans/board_payload.json',payload,replace=True)
            m={k:v for k,v in old.items() if k not in ('files','bundle_id')}
            m.update(experiment='cross_network_device_phase_01',parent_bundle_id=old['bundle_id'],
                device_observer_sha256=build['observer_sha256'],files=inventory(bundle))
            m['bundle_id']=digest(m)[:24];write(bundle/'manifest.json',m)
        validate(bundle)
        for case in read(bundle/'plans/configurations.json')['components']:
            for mode in ('qualify','clean','observed'):plans.append(dict(network=network,case=case,mode=mode))
    freeze(output/'PLAN.json',dict(specs=plans,native_sessions=12,VTA_stage_calls=84,new_DNN_inferences=0,
        hypothesis='locate real host CPU work before/inside/after VTADeviceRun using isolated components',
        independent_of_pipeline_FPS=True,new_pipeline_measurement=False,no_reboot=True,
        automatic_remeasurement=False))
    return dict(sessions=len(plans),VTA_stage_calls=84)


def validate(bundle):
    bundle=Path(bundle);m=read(bundle/'manifest.json')
    if m['bundle_id']!=digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]:
        raise ValueError('device capsule seal differs')
    verify_inventory(bundle,m['files'])
    p=read(bundle/'plans/device_sources.json')
    for path,expected in p['controllers'].items():
        if sha256(path)!=expected:raise ValueError('device controller source changed')
    for path,expected in p['build']['source_files'].items():
        if sha256(path)!=expected:raise ValueError('device observer source changed')
    return m


def run(output,run_dir,*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    require_space();output,run_dir=Path(output).resolve(),Path(run_dir).resolve()
    plan=read(output/'PLAN.json');board=read(OUTPUT/'board.local.json')['board']
    measure=FunctionType(SSH.measure.__code__,{**SSH.measure.__globals__,'native_command':command})
    ssh=type('DevicePhaseSSH',(SSH,),{'measure':measure})(board,execute_board=True)
    boot=ssh.boot_id()
    expected_boot=read(ROOT/'runs/cross_network_phase_profile_01/run.json')['boot_id']
    if boot!=expected_boot:raise ValueError('board boot changed; do not mix this bounded study')
    started=time.monotonic();expected={}
    with locked(run_dir):
        bundles={network:output/network/'capsule' for network in ('resnet18','yolov3_tiny416')}
        identities={n:validate(b)['bundle_id'] for n,b in bundles.items()}
        freeze(run_dir/'run.json',dict(boot_id=boot,bundle_ids=identities,plan_sha256=sha256(output/'PLAN.json')))
        remotes={}
        for n,b in bundles.items():
            (run_dir/n).mkdir(parents=True,exist_ok=True)
            remotes[n]=ssh.deploy(b,run_dir/n);ssh.qualify_environment(remotes[n],boot,run_dir/n,b)
        for number,spec in enumerate(plan['specs'],1):
            b=bundles[spec['network']];doc=task(b,spec);remote=remotes[spec['network']]
            receipt=run_dir/'accepted'/boot/(doc['task_id']+'.json')
            if receipt.exists():
                item=read(receipt);verify_inventory(item['raw'],item['files'])
                if item['task']!=doc or summarize(item['raw'],doc,b)!=item['summary']:
                    raise ValueError('device receipt no longer replays')
            else:
                raw=ssh.measure(doc,b,remote,run_dir,boot);result=summarize(raw,doc,b);sync_tree(raw)
                if (Path(raw,'boot_id').read_text().strip()!=boot or
                    Path(raw,'task_hash').read_text().strip()!=digest(doc)):
                    raise ValueError('raw device task hash or boot differs')
                if spec['mode']!='qualify' and result['output_fingerprints']!=expected[(spec['network'],spec['case'])]:
                    raise ValueError('device output differs from qualification')
                item=dict(task=doc,boot_id=boot,raw=str(raw),files=inventory(raw),summary=result,accepted_at=now())
                write(receipt,item);ssh.acknowledge(doc,run_dir,boot)
            key=(spec['network'],spec['case'])
            if spec['mode']=='qualify':expected[key]=item['summary']['output_fingerprints']
            elif item['summary']['output_fingerprints']!=expected[key]:raise ValueError('device output differs from qualification')
            write(run_dir/'state.json',dict(status='RUNNING',accepted_sessions=number,total_sessions=12,
                boot_id=boot,current_task=doc['task_id']),replace=True)
            print('VTA主机时序',number,'/12',spec['network'],spec['case'],spec['mode'],flush=True)
        for remote in remotes.values():
            ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > device_post_verify.log && sync',boot=boot,timeout=600)
        state=dict(status='COMPLETE',boot_id=boot,native_sessions=12,VTA_stage_calls=84,new_DNN_inferences=0,
                   seconds=time.monotonic()-started,goal_complete=False)
        write(run_dir/'state.json',state,replace=True);sync_tree(run_dir)
    return state
