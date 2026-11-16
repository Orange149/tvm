from pathlib import Path
from types import FunctionType
import json,shutil
from cpu_vta_partitioning.common import digest,inventory,read,sha256,write
from finite_service.common import freeze
from phase_profile.prepare import link_checked
import polling_profile_bridge as parent
from . import build as builder

ROOT=parent.ROOT
OUTPUT=ROOT/'artifacts/completion_profile_01'
RUN=ROOT/'runs/completion_profile_01'


def sources():
    paths=[ROOT/'run_completion_profile.py',ROOT/'tests/test_completion_profile.py',
        *Path(__file__).parent.glob('*.py'),Path(__file__).with_name('observer.cc')]
    return {str(p):sha256(p) for p in sorted(paths)}


def protocol():
    return dict(experiment='completion_profile_01',input_index=0,
        components=parent.base.protocol()['components'],modes=['qualify','clean','observed'],
        qualification=dict(warmup=0,scored=1,tail=0),calibration=dict(warmup=2,scored=8,tail=0),
        native_sessions=15,VTA_component_calls=105,DNN_inferences=0,
        hypothesis='device execution progresses while its polling host is not scheduled; a fixed isolation poll count is not independent device work',
        measurement='bracket completion with original last-false and first-true MMIO reads, relative to launch-write envelope',
        quality='abs paired clean/observed run CPU and wall differences <=20 percent before projection; otherwise diagnosis only',
        no_extra_MMIO_reads=True,no_extra_sleep=True,no_waiting_change=True,
        runtime_and_executor_unchanged=True,no_reboot=True,no_anchors=True,
        no_automatic_repeat=True,no_score_DP_Top20_change=True,
        completion_bounds_not_exact_hardware_duration=True,
        independent_device_duration_still_assumes_isolated_device_service=True,
        no_pipeline_performance_input=True,
        literature='GCAPS ECRTS2024 section4, host CPU operations and pure device execution; adapted resource distinction, not its priority scheduler')


def prepare(output=OUTPUT):
    output=Path(output)
    if (output/'PLAN.json').exists():
        p=read(output/'PLAN.json')
        if p['sources']!=sources():raise ValueError('completion sources changed')
        return {n:validate(output/n/'capsule') for n in p['bundle_ids']}
    if not (output/'build/BUILD.json').exists():builder.build(output/'build')
    ids={};old=read(parent.OUTPUT/'PLAN.json')
    for n in old['bundle_ids']:
        source=parent.OUTPUT/n/'capsule';parent.validate(source);target=output/n/'capsule';m=read(source/'manifest.json')
        for leaf,info in m['files'].items():
            if leaf in ('runner/libdevice_phase.so','plans/board_payload.json'):continue
            link_checked(source/leaf,target/leaf,info['sha256'])
        shutil.copy2(output/'build/libdevice_phase.so',target/'runner/libdevice_phase.so')
        write(target/'plans/completion.json',dict(sources=sources(),protocol=protocol(),
            build=read(output/'build/BUILD.json'),parent_manifest=str(source/'manifest.json'),
            parent_manifest_sha256=sha256(source/'manifest.json')))
        assets=inventory(target)
        payload={k:v for k,v in assets.items() if k.startswith(('runtime/','runner/','firmware/','inputs/','references/','programs/'))}
        payload['plans/board_execution.json']=assets['plans/board_execution.json']
        write(target/'plans/board_payload.json',dict(files=payload,unpacked_bytes=sum(v['bytes'] for v in payload.values())))
        new={k:v for k,v in m.items() if k not in ('bundle_id','files')}
        new.update(experiment='completion_profile_01',device_observer_sha256=sha256(target/'runner/libdevice_phase.so'),files=inventory(target))
        new['bundle_id']=digest(new)[:24];write(target/'manifest.json',new);ids[n]=new['bundle_id']
    freeze(output/'PLAN.json',dict(experiment='completion_profile_01',sources=sources(),protocol=protocol(),
        specs=old['specs'],bundle_ids=ids,expected_boot_id=old['expected_boot_id'],
        parent_plan_sha256=sha256(parent.OUTPUT/'PLAN.json'),build_sha256=sha256(output/'build/BUILD.json'),
        task_limit_frozen_before_native=True))
    return {n:validate(output/n/'capsule') for n in ids}


def validate(bundle):
    result=parent.validate(bundle);bundle=Path(bundle);proof=read(bundle/'plans/completion.json')
    if proof['sources']!=sources() or proof['protocol']!=protocol():raise ValueError('completion source/protocol changed')
    if sha256(proof['parent_manifest'])!=proof['parent_manifest_sha256']:raise ValueError('completion parent changed')
    if sha256(bundle/'runner/libdevice_phase.so')!=proof['build']['observer_sha256']:raise ValueError('completion observer changed')
    return result


def task(bundle,spec):
    doc=parent.task(bundle,spec);doc.pop('task_id');doc.update(experiment='completion_profile_01',
        completion_protocol_sha256=digest(protocol()),completion_observer=spec['mode']!='clean')
    doc['task_id']=digest(doc)[:24];return doc


def command(doc,bundle,remote,job):
    fn=FunctionType(parent.command.__code__,{**parent.command.__globals__,'task':task})
    return fn(doc,bundle,remote,job)


def completion_bounds(e):
    if e.get('MMIO_error') or e.get('start_writes')!=1 or e.get('auto_writes')!=3:
        raise ValueError('MMIO launch sequence mismatch')
    if e.get('true_reads')!=1 or e.get('observed_poll_reads')!=e.get('driver_poll_iters') or e.get('false_reads')!=e.get('observed_poll_reads',0)-1:
        raise ValueError('MMIO completion/read count mismatch')
    start,end=e['launch_before_ns'],e['launch_after_ns'];tb,te=e['first_true_before_ns'],e['first_true_after_ns']
    if not e['wall_start_ns']<=start<=end<=tb<=te<=e['wall_end_ns']:
        raise ValueError('MMIO completion chronology invalid')
    lb,le=e['last_false_before_ns'],e['last_false_after_ns']
    if e['false_reads']:
        if not end<=lb<=le<=tb:raise ValueError('MMIO last-false chronology invalid')
    elif lb or le:raise ValueError('false-read envelope without false read')
    lower=max(0,lb-end) if e['false_reads'] else 0
    upper=te-start
    return dict(lower_ns=lower,upper_ns=upper,width_ns=upper-lower,
        first_poll_completed=not bool(e['false_reads']),
        true_completion_not_exactly_observed=True)


def summarize(raw,doc,bundle):
    value=parent.base.legacy.summarize(raw,doc,bundle)
    if doc['device_observer']:
        value['driver_counters']=parent.base.validate_counters(value['device']['mapped'])
        lines=[json.loads(s) for s in (Path(raw)/'device_trace.jsonl').read_text().splitlines()]
        status=lines[0]
        if not status.get('completion_brackets') or status.get('extra_MMIO_reads')!=0 or not status.get('per_poll_probe'):
            raise ValueError('completion interposition status invalid')
        value['completion_brackets']=[completion_bounds(e) for e in lines[1:]]
        if read(Path(raw)/'queue_support_status.json')!=dict(enabled=False,errors=0,reads=0):
            raise ValueError('queue support unexpectedly active')
        w=read(Path(raw)/'wait_support_status.json')
        if w['policy']!='original' or w['probe'] or w['errors'] or w['overflow']:
            raise ValueError('original waiting support changed')
        value['queue_fields_are_not_observations']=True
    return value


def board(output=OUTPUT,run_dir=RUN,*,execute_board=False):
    fn=FunctionType(parent.base.board.__code__,{**parent.base.board.__globals__,'sources':sources,
        'validate':validate,'task':task,'command':command,'summarize':summarize})
    return fn(output,run_dir,execute_board=execute_board)
