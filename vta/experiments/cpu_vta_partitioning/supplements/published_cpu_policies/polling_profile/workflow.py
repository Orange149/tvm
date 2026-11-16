from copy import deepcopy
from pathlib import Path
from types import FunctionType
import json,random,shlex,shutil,time

from cpu_vta_partitioning.common import digest,inventory,now,read,sha256,sync_tree,verify_inventory,write,locked
from cpu_vta_partitioning.transport import SSH
from cpu_policies.open_resources.domain import component
from device_phase import collect as legacy
from finite_service.common import freeze
from phase_profile.prepare import link_checked
from host_placement_check.prepare import validate_input_paths
from . import build as builder

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/polling_profile_01'
RUN=ROOT/'runs/polling_profile_01'
CONFIG=ROOT/'artifacts/cross_network_model_01/board.local.json'


def sources():
    files=[ROOT/'run_polling_profile.py',ROOT/'tests/test_polling_profile.py',
        *sorted(Path(__file__).parent.glob('*.py')),Path(__file__).with_name('observer.cc'),
        ROOT/'device_phase/collect.py',ROOT/'phase_profile/runtime.py',
        ROOT.parents[1]/'src/cpu_vta_partitioning/measurement/tasks.py']
    return {str(p):sha256(p) for p in files}


def protocol():
    return dict(experiment='polling_profile_01',input_index=0,seed=20261111,
        components={'resnet18':['vta_03_04','vta_08_08','vta_10_19'],
                    'yolov3_tiny416':['vta_03_06','vta_09_12']},
        modes=['qualify','clean','observed'],qualification=dict(warmup=0,scored=1,tail=0),
        calibration=dict(warmup=2,scored=8,tail=0),max_native_sessions=15,
        VTA_component_calls=105,DNN_inferences=0,
        hypothesis='aggregating all driver CPU work before one off-CPU delay loses repeated polling/wakeup chronology',
        scope='read existing driver status at each real VTADeviceRun boundary; isolated frontier only',
        quality='report paired clean/observed CPU and wall perturbation; if >20 percent do not promote its shape to a predictive input',
        no_per_poll_probe=True,no_runtime_or_runner_or_module_changes=True,
        no_reboot=True,no_anchors=True,no_drift_rejection=True,no_automatic_repeat=True,
        no_score_DP_or_Top20_changes=True,no_automatic_pipeline_test=True,
        no_counter_poll_count_as_pure_accelerator_time=True)


def prepare(output=OUTPUT):
    output=Path(output)
    if (output/'PLAN.json').exists():
        p=read(output/'PLAN.json')
        if p['sources']!=sources():raise ValueError('polling preparation identity changed')
        return {n:validate(output/n/'capsule') for n in protocol()['components']}
    if not (output/'build/BUILD.json').exists():builder.build(output/'build')
    build=read(output/'build/BUILD.json');identities={}
    for network,names in protocol()['components'].items():
        source=ROOT/'artifacts/finite_service_01/prospective'/network/'capsule'
        parent=read(source/'manifest.json');target=output/network/'capsule'
        configs=read(source/'plans/configurations.json')['pipelines']
        original=next(iter(configs.values()));components={}
        for name in names:
            row=next(s for s in original['stages'] if s['program']==name)
            components[name]=component(row)
        for leaf,info in parent['files'].items():
            if leaf.startswith('programs/') and leaf.split('/')[1] not in names:continue
            if leaf.startswith(('runtime/','runner/','firmware/','inputs/','references/','programs/')):
                link_checked(source/leaf,target/leaf,info['sha256'])
        shutil.copy2(output/'build/libdevice_phase.so',target/'runner/libdevice_phase.so')
        for leaf in ('programs.json','inputs.json'):
            item=read(source/'plans'/leaf)
            if leaf=='programs.json':item={n:r for n,r in item.items() if n in names}
            freeze(target/'plans'/leaf,item)
        write(target/'plans/configurations.json',dict(components=components,pipelines={}))
        write(target/'plans/board_execution.json',read(source/'plans/board_execution.json'))
        write(target/'plans/protocol.json',protocol())
        write(target/'plans/poll_sources.json',dict(sources=sources(),build=build,
            parent_manifest=str(source/'manifest.json'),parent_manifest_sha256=sha256(source/'manifest.json')))
        assets=inventory(target)
        payload={k:v for k,v in assets.items() if k.startswith(('runtime/','runner/','firmware/','inputs/','references/','programs/'))}
        payload['plans/board_execution.json']=assets['plans/board_execution.json']
        write(target/'plans/board_payload.json',dict(files=payload,unpacked_bytes=sum(v['bytes'] for v in payload.values())))
        m=dict(experiment='polling_profile_01',network=network,device_observer_sha256=build['observer_sha256'],
            **{k:parent[k] for k in ('phase_runner_sha256','old_runner_sha256','adapter_sha256','runtime_sha256')},files=inventory(target))
        m['bundle_id']=digest(m)[:24];write(target/'manifest.json',m);identities[network]=m['bundle_id']
    pairs=[(n,name) for n,names in protocol()['components'].items() for name in names]
    random.Random(protocol()['seed']).shuffle(pairs)
    specs=[dict(network=n,case=name,mode=mode) for n,name in pairs for mode in protocol()['modes']]
    boot=read(ROOT/'artifacts/finite_service_01/PLAN.json')['expected_boot_id']
    freeze(output/'PLAN.json',dict(protocol=protocol(),sources=sources(),bundle_ids=identities,
        specs=specs,expected_boot_id=boot,build_sha256=sha256(output/'build/BUILD.json'),
        task_limit_frozen_before_native=True))
    return {n:validate(output/n/'capsule') for n in identities}


def validate(bundle):
    bundle=Path(bundle);m=read(bundle/'manifest.json')
    if digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:
        raise ValueError('poll capsule manifest changed')
    verify_inventory(bundle,m['files']);validate_input_paths(bundle,m)
    proof=read(bundle/'plans/poll_sources.json')
    if proof['sources']!=sources() or read(bundle/'plans/protocol.json')!=protocol():
        raise ValueError('poll source/protocol identity changed')
    if sha256(proof['parent_manifest'])!=proof['parent_manifest_sha256']:
        raise ValueError('poll source capsule changed')
    return dict(validated=True,bundle_id=m['bundle_id'],network=m['network'])


def task(bundle,spec):
    doc=legacy.task(bundle,spec);doc.pop('task_id')
    doc['experiment']='polling_profile_01';doc['polling_protocol_sha256']=digest(protocol())
    doc['task_id']=digest(doc)[:24];return doc


def command(doc,bundle,remote,job):
    fn=FunctionType(legacy.command.__code__,{**legacy.command.__globals__,'task':task})
    return fn(doc,bundle,remote,job)


def validate_counters(mapped):
    if not mapped:raise ValueError('no observed driver calls')
    for event in mapped:
        if (not event.get('counters_monotonic') or event.get('driver_run_calls')!=1
            or event.get('driver_run_insns')!=event['instructions']
            or event.get('driver_timeout_calls')!=0 or event.get('driver_poll_iters',0)<1):
            raise ValueError('driver stats scope/ABI/call count invalid')
        fields=['driver_run_total_us','driver_submit_mmio_us','driver_post_start_sleep_us','driver_poll_wait_us']
        values=[event.get(k) for k in fields]
        import math
        if any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in values):
            raise ValueError('invalid driver profiler time')
        total,submit,post,poll=values
        wall=(event['wall_end_ns']-event['wall_start_ns'])/1000
        if not 0<total<=wall+100 or submit+post+poll>total+20:
            raise ValueError('driver clocks/counter offsets inconsistent')
    return dict(passed=True,device_calls=len(mapped),poll_checks=sum(r['driver_poll_iters'] for r in mapped),
        no_driver_timeout=True,driver_CPU_not_equal_to_driver_wall=True)


def summarize(raw,doc,bundle):
    value=legacy.summarize(raw,doc,bundle)
    if doc['device_observer']:
        value['driver_counters']=validate_counters(value['device']['mapped'])
        status=json.loads((Path(raw)/'device_trace.jsonl').read_text().splitlines()[0])
        if status.get('stats_bytes')!=64 or status.get('per_poll_probe') is not False:
            raise ValueError('unexpected driver status observer')
    return value


def board(output=OUTPUT,run_dir=RUN,*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    output=Path(output);run_dir=Path(run_dir);plan=read(output/'PLAN.json')
    if plan['sources']!=sources():raise ValueError('poll controller changed')
    measure=FunctionType(SSH.measure.__code__,{**SSH.measure.__globals__,'native_command':command})
    ssh=type('PollingProfileSSH',(SSH,),{'measure':measure})(read(CONFIG)['board'],execute_board=True)
    boot=ssh.boot_id()
    if boot!=plan['expected_boot_id']:raise ValueError('boot changed; stop without mixing cohorts')
    start=time.monotonic();expected={}
    with locked(run_dir):
        freeze(run_dir/'run.json',dict(boot_id=boot,bundles=plan['bundle_ids'],plan_sha256=sha256(output/'PLAN.json')))
        remotes={}
        for n in plan['bundle_ids']:
            b=output/n/'capsule';validate(b);(run_dir/n).mkdir(parents=True,exist_ok=True)
            remotes[n]=ssh.deploy(b,run_dir/n);ssh.qualify_environment(remotes[n],boot,run_dir/n,b)
        for number,spec in enumerate(plan['specs'],1):
            bundle=output/spec['network']/'capsule';doc=task(bundle,spec)
            receipt=run_dir/'accepted'/boot/(doc['task_id']+'.json')
            if receipt.exists():
                item=read(receipt);verify_inventory(item['raw'],item['files'])
                if item['task']!=doc or item['boot_id']!=boot or summarize(item['raw'],doc,bundle)!=item['summary']:
                    raise ValueError('accepted polling evidence differs')
            else:
                raw=ssh.measure(doc,bundle,remotes[spec['network']],run_dir,boot)
                value=summarize(raw,doc,bundle)
                if Path(raw,'boot_id').read_text().strip()!=boot or Path(raw,'task_hash').read_text().strip()!=digest(doc):
                    raise ValueError('poll task hash or boot changed')
                if spec['mode']!='qualify' and value['output_fingerprints']!=expected[spec['network'],spec['case']]:
                    raise ValueError('poll output differs from complete qualification')
                sync_tree(raw);item=dict(task=doc,boot_id=boot,raw=str(raw),files=inventory(raw),summary=value,
                    accepted_at=now())
                write(receipt,item);ssh.acknowledge(doc,run_dir,boot)
            if spec['mode']=='qualify':expected[spec['network'],spec['case']]=item['summary']['output_fingerprints']
            elif item['summary']['output_fingerprints']!=expected[spec['network'],spec['case']]:
                raise ValueError('accepted polling output differs from qualification')
            write(run_dir/'state.json',dict(status='RUNNING',boot_id=boot,accepted_sessions=number,
                total_sessions=15,DNN_inferences=0,current_task=doc['task_id']),replace=True)
            print('VTA轮询组件',number,'/15',spec['network'],spec['case'],spec['mode'],flush=True)
        for remote in remotes.values():
            ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > poll_post_verify.log && sync',
                boot=boot,timeout=120)
        state=dict(status='COMPLETE',boot_id=boot,native_sessions=15,VTA_component_calls=105,DNN_inferences=0,
            seconds=time.monotonic()-start,no_reboot=True,no_automatic_repeat=True)
        write(run_dir/'state.json',state,replace=True);return state
