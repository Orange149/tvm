"""Exactly 198 predefined local replays, sealed before performance analysis."""
from itertools import product
from pathlib import Path
import time
from cpu_vta_partitioning.common import digest,read,sha256,write
from .common import freeze,protocol,replay_input_guard,sources
from .simulator import AuditReplay


def validate_isolated(data):
    if (data.get('schema')!='isolated-semantics-input-v1' or data.get('origin')!='isolated_projection'
        or data.get('contains_pipeline_data') is not False or data.get('diagnostic_only')):
        raise ValueError('static input must contain only isolated projections')
    allowed={'configuration','stages','assumptions','expected_baseline'}
    def reject(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k.lower() in ('fps','measured_fps','pipeline_fps','cpu_delay_total','diagnostic_controls','pipeline_service_ms'):
                    raise ValueError('pipeline values forbidden in isolated input')
                reject(v)
        elif isinstance(value,list):
            for v in value:reject(v)
    reject(data)
    for case in data['cases'].values():
        if set(case)!=allowed:raise ValueError('isolated case has unexpected fields')
        if any(k in stage for stage in case['stages'] for k in ('pipeline_duration','control_CPU_ns','observed_service_ms')):
            raise ValueError('diagnostic fields cannot become stage work')
    return data


def scenario_specs(iso,diag):
    validate_isolated(iso)
    if diag.get('origin')!='observed_pipeline_control_CPU' or not diag.get('diagnostic_only') or not diag.get('forbidden_for_static_prediction'):
        raise ValueError('control input must be explicitly diagnostic')
    if set(iso['cases'])!=set(diag['cases']):raise ValueError('control/isolated case mismatch')
    result=[]
    for case,value in sorted(iso['cases'].items()):
        hosts=[(i,s['mask']) for i,s in enumerate(value['stages']) if s['device']=='vta']
        placements=[None]+[{str(i):core for (i,_),core in zip(hosts,cores)} for cores in product(*(m for _,m in hosts))]
        for control in ('zero','diagnostic_observed'):
            for assignment in placements:
                spec=dict(case=case,control_mode=control,host_assignment=assignment,
                          host_mode='greedy' if assignment is None else 'fixed_trajectory')
                spec['id']='replay_'+digest(spec)[:24];result.append(spec)
    return result


def run_isolated(case,window,*,fixed_hosts=None,trace=False):
    """Admissible entry: no control observations or measured durations accepted."""
    document=dict(schema='isolated-semantics-input-v1',origin='isolated_projection',contains_pipeline_data=False,
                  cases={'case':case})
    validate_isolated(document)
    return AuditReplay(case['stages'],fixed_hosts=fixed_hosts,trace=trace,
        gate=case['configuration']['execution_mode'],host_order='before',
        frames=window['frames'],warmup=window['warmup'],scored=window['scored'],queue_depth=window['queue_depth']).run()


def replay(output):
    output=Path(output).resolve();started=time.monotonic();resumed=new=0
    with replay_input_guard(output):
        seal=read(output/'INPUT_FREEZE.json')
        if seal['sources']!=sources():raise ValueError('offline code changed; use a new study identity')
        for f,h in seal['files'].items():
            if sha256(output/f)!=h:raise ValueError('input freeze changed')
        iso=validate_isolated(read(output/'ISOLATED_INPUTS.json'));diag=read(output/'DIAGNOSTIC_CONTROL_INPUTS.json')
        specs=scenario_specs(iso,diag)
        if len(specs)!=198 or len({s['id'] for s in specs})!=198:raise ValueError('expected 198 unique scenarios')
        plan=dict(protocol=protocol(),specs=specs,input_freeze_sha256=sha256(output/'INPUT_FREEZE.json'),sources=sources())
        freeze(output/'PLAN.json',plan);files={}
        for count,spec in enumerate(specs,1):
            path=output/'replays'/(spec['id']+'.json');case=iso['cases'][spec['case']]
            identity=dict(spec=spec,input_freeze_sha256=sha256(output/'INPUT_FREEZE.json'),sources=seal['sources'])
            if path.exists():
                item=read(path)
                if item['input_identity']!=digest(identity) or item['spec']!=spec or item['content_sha256']!=digest({k:v for k,v in item.items() if k!='content_sha256'}):
                    raise ValueError('replay checkpoint changed')
                resumed+=1
            else:
                tick=time.monotonic();window=iso['window']
                if spec['control_mode']=='zero':
                    result=run_isolated(case,window,fixed_hosts=spec['host_assignment'],trace=spec['host_assignment'] is None)
                else:
                    result=AuditReplay(case['stages'],fixed_hosts=spec['host_assignment'],
                        controls=diag['cases'][spec['case']],trace=spec['host_assignment'] is None,
                        gate=case['configuration']['execution_mode'],host_order='before',
                        frames=window['frames'],warmup=window['warmup'],scored=window['scored'],queue_depth=window['queue_depth']).run()
                if spec['control_mode']=='zero' and spec['host_assignment'] is None and result['replay']!=case['expected_baseline']:
                    raise ValueError('original frozen baseline not exactly restored: '+spec['case'])
                item=dict(spec=spec,input_identity=digest(identity),result=result,seconds=time.monotonic()-tick,
                          pipeline_performance_read=False,diagnostic_only=spec['control_mode']!='zero')
                item['content_sha256']=digest(item);freeze(path,item);new+=1
            files[str(path.relative_to(output))]=sha256(path)
            if count%10==0 or count==198:print('离线语义回放',count,'/198',spec['case'],spec['control_mode'],flush=True)
        freeze(output/'REPLAY_FREEZE.json',dict(files=files,plan_sha256=sha256(output/'PLAN.json'),
            input_freeze_sha256=sha256(output/'INPUT_FREEZE.json'),sources=sources(),scenario_count=len(specs),
            performance_decoded_before_freeze=False,new_native_sessions=0))
    write(output/'REPLAY_COST.json',dict(last_command_seconds=time.monotonic()-started,resumed=resumed,new=new,
        original_replay_seconds=sum(read(output/p)['seconds'] for p in files)),replace=True)
    return dict(replays=198,resumed=resumed,new=new,board_sessions=0)


def validate(output):
    output=Path(output);seal=read(output/'REPLAY_FREEZE.json')
    if seal['sources']!=sources():raise ValueError('code identity changed')
    if sha256(output/'PLAN.json')!=seal['plan_sha256'] or sha256(output/'INPUT_FREEZE.json')!=seal['input_freeze_sha256']:
        raise ValueError('replay provenance changed')
    inputs=read(output/'INPUT_FREEZE.json')
    for path,h in inputs['files'].items():
        if sha256(output/path)!=h:raise ValueError('projected input changed')
    for path,h in seal['files'].items():
        if sha256(output/path)!=h:raise ValueError('replay evidence changed')
    from common_top20.prepare import validate as old_common
    from pool_temporal.prepare import validate as old_intrinsic
    from .common import STUDY
    old=[]
    for network in ('resnet18','yolov3_tiny416'):
        old.append(old_common(STUDY/'common_top20'/network/'capsule')['bundle_id'])
        old.append(old_intrinsic(STUDY/'pool_temporal'/network/'capsule')['bundle_id'])
    return dict(validated=True,replays=len(seal['files']),old_capsules=old,no_board_access=True)
