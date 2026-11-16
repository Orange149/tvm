"""Two frozen YOLO VTA modules, runtime CPU pools of 1 or 4 threads.

Eight native sessions, 44 isolated VTA subgraph calls, zero complete DNNs.
The old search domain and its one-thread VTA profiles remain unchanged.
"""
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import argparse
import json
import random
import shlex
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,digest,inventory,verify_inventory,locked,now,sync_tree
from cpu_vta_partitioning.transport import SSH
from cpu_policies.open_resources.domain import stage,component
from phase_profile.prepare import link_checked
from phase_profile import runtime as phase
from phase_profile.campaign import accept as original_accept
from finite_service.common import freeze
from host_placement_check.prepare import validate_input_paths
from yolo_validation.components import hardware_check
from scheduler_gap_audit.curves import csv_write

ROOT=Path(__file__).resolve().parent
POOL=ROOT/'artifacts/vta_CPU_fallback_threads_01'
RUN=ROOT/'runs/vta_CPU_fallback_threads_01'
PARENT=ROOT/'artifacts/scheduler_gap_audit_01/yolov3_tiny416/capsule'
PLAN=ROOT/'artifacts/vta_host_work_origin_01/NEXT_COMPONENT_CHECK.json'
BUNDLE=POOL/'capsule'
CONFIG=ROOT/'artifacts/cross_network_model_01/board.local.json'


def protocol():
    p=read(PLAN)
    if p['source_evidence_sha256']!=sha256(PLAN.parent/'RESULTS.json'):
        raise ValueError('origin evidence changed')
    return p


def sources():
    paths=[Path(__file__),ROOT/'tests/test_vta_fallback_threads.py',ROOT/'phase_profile/runtime.py',
        ROOT/'phase_profile/campaign.py',ROOT/'cpu_policies/open_resources/runtime.py',
        ROOT.parents[1]/'src/cpu_vta_partitioning/measurement/tasks.py',
        ROOT.parents[1]/'src/cpu_vta_partitioning/transport/__init__.py']
    return {str(p):sha256(p) for p in paths}


def configurations():
    rows={}
    for program in protocol()['programs']:
        first,last=map(int,program.split('_')[1:])
        for n in protocol()['runtime_threads']:
            rows[program+'_t'+str(n)]=component(stage('vta',first,last+1,n,[0,1,2,3]))
    return rows


def validate_configuration(cfg):
    if cfg not in configurations().values():raise ValueError('unplanned extended component configuration')
    s=cfg['stages'][0]
    if (len(cfg['stages'])!=1 or s['device']!='vta' or s['threads'] not in (1,4)
        or s['mask']!=[0,1,2,3] or s['worker_core_map'] or s['caller_core'] is not None
        or cfg['affinity_mode']!='group' or cfg['execution_mode']!='off'):
        raise ValueError('VTA fallback configuration differs')


def specs():
    pairs=list(configurations());random.Random(20261111).shuffle(pairs)
    return [dict(network='yolov3_tiny416',case=key,scope='component',phase=p,probe=False,runner='phase')
        for key in pairs for p in ('qualify','clean')]


def prepare():
    if (BUNDLE/'manifest.json').exists():return validate()
    m=read(PARENT/'manifest.json')
    if digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:
        raise ValueError('frozen parent manifest differs')
    verify_inventory(PARENT,m['files'])
    freeze(POOL/'PLAN.json',dict(protocol=protocol(),sources=sources(),specs=specs(),
        origin_plan_sha256=sha256(PLAN),parent_manifest_sha256=sha256(PARENT/'manifest.json'),
        no_full_DNN=True,no_runtime_module_executor_rebuild=True,
        extended_VTA_CPU_pool_not_yet_in_old_search_domain=True))
    wanted=set(protocol()['programs'])
    for rel,info in m['files'].items():
        use=rel.startswith(('runtime/','firmware/','inputs/','references/'))
        use=use or rel in ('runner/vta_stage_pipeline_runner','runner/vta_stage_pipeline_runner_phase')
        use=use or (rel.startswith('programs/') and rel.split('/')[1] in wanted)
        if use:link_checked(PARENT/rel,BUNDLE/rel,info['sha256'])
    write(BUNDLE/'plans/programs.json',{n:p for n,p in read(PARENT/'plans/programs.json').items() if n in wanted})
    write(BUNDLE/'plans/inputs.json',read(PARENT/'plans/inputs.json'))
    write(BUNDLE/'plans/configurations.json',dict(components=configurations(),pipelines={}))
    write(BUNDLE/'plans/protocol.json',protocol())
    write(BUNDLE/'plans/preparation.json',dict(plan_sha256=sha256(POOL/'PLAN.json'),
        parent_manifest=str(PARENT/'manifest.json'),parent_manifest_sha256=sha256(PARENT/'manifest.json')))
    write(BUNDLE/'plans/board_execution.json',dict(udmabuf_bytes=201326592,storage_margin_bytes=268435456,
        boot_policy='existing_boot_no_reprogram_no_reboot'))
    assets=inventory(BUNDLE)
    payload={n:v for n,v in assets.items() if n.startswith(('runtime/','runner/','firmware/','inputs/','references/','programs/'))}
    payload['plans/board_execution.json']=assets['plans/board_execution.json']
    write(BUNDLE/'plans/board_payload.json',dict(files=payload,unpacked_bytes=sum(v['bytes'] for v in payload.values())))
    new=dict(experiment=protocol()['experiment'],network='yolov3_tiny416',adapter_sha256=None,
        **{k:m[k] for k in ('phase_runner_sha256','old_runner_sha256','runtime_sha256')},files=inventory(BUNDLE))
    new['bundle_id']=digest(new)[:24];write(BUNDLE/'manifest.json',new)
    return validate()


def validate():
    p=read(POOL/'PLAN.json');m=read(BUNDLE/'manifest.json')
    if p['sources']!=sources() or p['protocol']!=protocol() or p['specs']!=specs():
        raise ValueError('frozen controller/protocol changed')
    if sha256(PARENT/'manifest.json')!=p['parent_manifest_sha256'] or sha256(PLAN)!=p['origin_plan_sha256']:
        raise ValueError('frozen parent or intervention changed')
    if digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:
        raise ValueError('capsule identity changed')
    verify_inventory(BUNDLE,m['files']);validate_input_paths(BUNDLE,m)
    for cfg in read(BUNDLE/'plans/configurations.json')['components'].values():validate_configuration(cfg)
    return dict(validated=True,bundle_id=m['bundle_id'],native_sessions=8,VTA_component_calls=44,DNN_inferences=0)


def make_task(bundle,spec):
    if spec not in specs():raise ValueError('unplanned VTA pool task')
    bundle=Path(bundle);m=read(bundle/'manifest.json')
    cfg=read(bundle/'plans/configurations.json')['components'][spec['case']];validate_configuration(cfg)
    qualify=spec['phase']=='qualify';p=protocol()
    doc=dict(kind='qualify_component' if qualify else 'calibration',configuration=deepcopy(cfg),
        stages=deepcopy(cfg['stages']),spec=deepcopy(spec),**p['qualification' if qualify else 'calibration'],
        environment=dict(cores=4,frequency_khz=1066666,poll_sleep_ns=1000,post_start_sleep_ns=1000,
            spin_count=0,exclude_worker0=1,bind_threads=1),input_indices=[0],
        executor_sha256=m['phase_runner_sha256'],runtime_sha256=m['runtime_sha256'],adapter_sha256=None,
        bundle_id=m['bundle_id'],experiment=p['experiment'],startup_affinity_probe=True,
        CPU_pool_scope='CPU fallback inside VTA graph, group-migratable workers; one physical VTA device')
    doc['task_id']=digest(doc)[:24];return doc


def native_command(doc,bundle,remote_bundle,remote_job):
    f=phase.native_command
    argv,inputs=FunctionType(f.__code__,{**f.__globals__,'make_task':make_task})(doc,bundle,remote_bundle,remote_job)
    argv.remove('--cpu-worker-pinning')
    if any('libwait_adapter' in a for a in argv) or '--phase-event-json' in argv:
        raise ValueError('unexpected detailed observer')
    return argv,inputs


summary=phase.summary
accept=FunctionType(original_accept.__code__,{**original_accept.__globals__,'summary':summary},argdefs=original_accept.__defaults__)


def qemu():
    validate();loader=BUNDLE/'runtime/ld-linux-aarch64.so.1'
    p=subprocess.run(['qemu-aarch64-static',str(loader),'--library-path',str(BUNDLE/'runtime'),
        str(BUNDLE/'runner/vta_stage_pipeline_runner_phase'),'--help'],capture_output=True,text=True,timeout=30)
    if p.returncode:raise ValueError('AArch64 loading failed')
    value=dict(passed=True,scope='loader and CLI only; VTA numerical output is verified on real board',
        runner_sha256=sha256(BUNDLE/'runner/vta_stage_pipeline_runner_phase'))
    freeze(POOL/'QEMU.json',value);return value


def board(*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    valid=validate();boot=protocol()['expected_boot_id']
    if not read(POOL/'QEMU.json')['passed']:raise ValueError('loader check required')
    measure=FunctionType(SSH.measure.__code__,{**SSH.measure.__globals__,'native_command':native_command})
    ssh=type('VTAFallbackSSH',(SSH,),{'measure':measure})(read(CONFIG)['board'],execute_board=True)
    if ssh.boot_id()!=boot:raise ValueError('boot differs; do not mix the component batch')
    start=time.monotonic()
    with locked(RUN):
        freeze(RUN/'run.json',dict(boot_id=boot,bundle_id=valid['bundle_id'],plan_sha256=sha256(POOL/'PLAN.json')))
        def state(status,**kw):write(RUN/'state.json',dict(status=status,boot_id=boot,updated_at=now(),**kw),replace=True)
        try:
            remote=ssh.deploy(BUNDLE,RUN)
            # Read-only existing-hardware qualification; no bitstream reload.
            write(RUN/'hardware.json',hardware_check(ssh,BUNDLE,boot),replace=True)
            expected={};calls=0
            for index,spec in enumerate(specs(),1):
                doc=make_task(BUNDLE,spec);key=spec['case']
                if spec['phase']=='clean' and key not in expected:raise ValueError('calibration before qualification')
                state('RUNNING',accepted_sessions=index-1,total_sessions=8,current_task=doc['task_id'])
                receipt=accept(ssh,doc,BUNDLE,remote,RUN,boot,expected.get(key))
                if spec['phase']=='qualify':expected[key]=receipt['summary']['output_fingerprints']
                calls+=sum(doc[k] for k in ('warmup','scored','tail'))
                print('VTA子图CPU线程对照',index,'/8',key,spec['phase'],flush=True)
            if calls!=44:raise ValueError('component-call budget differs')
            ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > component_post_verify.log && sync',boot=boot,timeout=300)
            if ssh.boot_id()!=boot:raise ValueError('boot changed during component collection')
            state('COMPLETE',accepted_sessions=8,total_sessions=8,VTA_component_calls=calls,DNN_inferences=0,
                seconds=time.monotonic()-start,no_reboot=True,no_bitstream_reload=True,no_automatic_extra_measurement=True)
            sync_tree(RUN)
        except Exception as e:
            state('STOPPED_ON_ERROR',error_type=type(e).__name__,error=str(e));raise
    return read(RUN/'state.json')


def analyze():
    validate();state=read(RUN/'state.json');boot=protocol()['expected_boot_id']
    if state['status']!='COMPLETE' or state['accepted_sessions']!=8 or state['boot_id']!=boot:
        raise ValueError('component cohort incomplete')
    rows=[];hashes={};expected={}
    for spec in specs():
        t=make_task(BUNDLE,spec);p=RUN/'accepted'/boot/(t['task_id']+'.json');r=read(p)
        if r['task']!=t or r['boot_id']!=boot:raise ValueError('component receipt identity changed')
        verify_inventory(r['raw'],r['files']);value=summary(r['raw'],t,BUNDLE)
        if value!=r['summary']:raise ValueError('component raw evidence no longer replays')
        hashes[str(p)]=sha256(p)
        if spec['phase']=='qualify':expected[spec['case']]=value['output_fingerprints'];continue
        if value['output_fingerprints']!=expected[spec['case']]:raise ValueError('outputs differ from qualification')
        stage=t['stages'][0]
        rows.append(dict(program=stage['program'],runtime_threads=stage['threads'],mask=stage['mask'],
            **value['cost'],
            whole_session_helper_CPU_ms=sum(r['cpu_time_delta_ms'] for r in value['thread_ledger'] if r['role']=='worker'),
            helper_scope='whole session, coarse task CPU counters; not summed into phase process CPU time',
            task_id=t['task_id']))
    comparisons=[]
    for program in protocol()['programs']:
        a=next(r for r in rows if r['program']==program and r['runtime_threads']==1)
        b=next(r for r in rows if r['program']==program and r['runtime_threads']==4)
        comparisons.append(dict(program=program,one_thread_service_ms=a['service_ms'],four_thread_service_ms=b['service_ms'],
            service_speedup=a['service_ms']/b['service_ms'],run_core_time_ratio=b['run_core_ms']/a['run_core_ms'],
            no_complete_pipeline_improvement_proven=True))
    result=dict(rows=rows,comparisons=comparisons,source_receipts=hashes,new_native_sessions=8,
        VTA_component_calls=44,DNN_inferences=0,modules_runtime_executor_unchanged=True,
        no_scoring_DP_or_global_profile_change=True)
    write(POOL/'RESULTS.json',result,replace=True);csv_write(POOL/'components.csv',rows)
    print(json.dumps(result['comparisons'],ensure_ascii=False,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','validate','qemu','board','status','analyze'))
    p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    r=board(execute_board=a.execute_board) if a.command=='board' else read(RUN/'state.json') if a.command=='status' else globals()[a.command]()
    if a.command!='analyze':print(json.dumps(r,ensure_ascii=False,indent=2))
