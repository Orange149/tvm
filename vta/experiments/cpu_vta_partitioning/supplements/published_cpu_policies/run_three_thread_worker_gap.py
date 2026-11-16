"""Observe the exact three-thread host-hotspot cases; reuse clean references.

Six bounded sessions, 58 complete networks and 11 isolated CPU-stage calls.
The existing preload only observes original waits; no runtime/model changes.
"""
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import argparse
import json
import shlex
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,digest,locked,now,sync_tree,verify_inventory
from finite_service.common import freeze
from model_semantics_audit.common import metadata_projection
from phase_profile import runtime as phase
from yolo_validation.components import hardware_check
import finite_service_holdout_board as packager
import run_three_thread_host_gap as clean

ROOT=clean.ROOT
POOL=ROOT/'artifacts/three_thread_worker_gap_01'
RUN=ROOT/'runs/three_thread_worker_gap_01'
CONFIG=clean.CONFIG
OUTPUT=packager.OUTPUT
ROLES=('host_on_heavier_worker','host_outside_front_group')


def protocol():
    p=packager.protocol();p.update(experiment='three_thread_worker_gap_01',
        component=dict(warmup=2,scored=8,tail=0),new_DNN=58,CPU_stage_calls=11,
        max_native_sessions=6,observer='existing original-wait adapter plus phase timestamps',
        clean_references='previous accepted exact configurations, same boot/executor/input/window',
        known_layout_diagnosis=True,no_new_predictor=True,no_observer=False,
        no_new_runtime_or_executor=True,no_new_isolated_search_cost=True,
        heavy_case_quantification_limit_ms=22.897039875*0.2,
        quantification_limit_source='previously observed heavy-case residual; not a performance rejection gate')
    return p


def sources():
    return {**clean.sources(),str(Path(__file__)):sha256(__file__),
        str(ROOT/'phase_profile/worker.py'):sha256(ROOT/'phase_profile/worker.py'),
        str(ROOT/'wait_diagnosis/runtime.py'):sha256(ROOT/'wait_diagnosis/runtime.py')}


def selection():
    x=clean.selection();st=read(clean.RUN/'state.json')
    if st['status']!='COMPLETE':raise ValueError('clean matched references are incomplete')
    wanted=[c for c in x['selected_configurations']['yolov3_tiny416'] if x['roles'][c['id']] in ROLES]
    if len(wanted)!=2:raise ValueError('diagnostic pair changed')
    refs={};bundle=clean.POOL/'yolov3_tiny416/capsule'
    for s in clean.specs():
        if s['phase']!='clean':continue
        if s['scope']=='component' and s['case']!='3':continue
        if s['scope']=='pipeline' and s['case'] not in {c['id'] for c in wanted}:continue
        t=clean.make_task(bundle,s);p=clean.RUN/'accepted'/st['boot_id']/(t['task_id']+'.json')
        m=metadata_projection(p,{'task','boot_id'})
        if m['task']!=t or m['boot_id']!=st['boot_id']:raise ValueError('clean reference identity differs')
        refs[s['scope']+'/'+s['case']]=dict(receipt=str(p),sha256=sha256(p),task_id=t['task_id'])
    if len(refs)!=3:raise ValueError('missing isolated or pipeline clean reference')
    return dict(selected_configurations={'yolov3_tiny416':wanted},components={'3':x['components']['3']},
        roles={c['id']:x['roles'][c['id']] for c in wanted},references=refs,
        expected_boot_id=st['boot_id'],DNN_budget=58,
        known_result_development_diagnosis_not_ranking_validation=True)


def specs():
    s=selection();rows=[dict(network='yolov3_tiny416',case='3',scope='component',phase=p,runner='phase',probe=True)
                      for p in ('qualify','observed')]
    for c in s['selected_configurations']['yolov3_tiny416']:
        rows += [dict(network='yolov3_tiny416',case=c['id'],scope='pipeline',phase=p,runner='phase',probe=True)
                 for p in ('qualify','observed')]
    return rows


def adapt(name):
    f=getattr(packager,name);g=FunctionType(f.__code__,{**f.__globals__,**globals()},f.__name__,f.__defaults__)
    g.__kwdefaults__=f.__kwdefaults__;return g


def prepare():
    freeze(POOL/'SELECTION_FREEZE.json',dict(**selection(),sources=sources(),new_DNN=58,
        CPU_stage_calls=11,max_native_sessions=6,selection_uses_known_errors_for_diagnosis=True))
    return adapt('prepare')()


def validate(bundle):return adapt('validate')(bundle)
def qemu():return adapt('qemu')()


def make_task(bundle,spec):
    b=Path(bundle);m=read(b/'manifest.json')
    if spec['network']!='yolov3_tiny416' or spec['runner']!='phase' or spec['probe'] is not True:
        raise ValueError('observation identity differs')
    if spec['scope']=='component':cfg=selection()['components'][spec['case']]
    elif spec['scope']=='pipeline':cfg=read(b/'plans/configurations.json')['pipelines'][spec['case']]
    else:raise ValueError('unknown scope')
    if spec['phase'] not in ('qualify','observed'):raise ValueError('unknown phase')
    q=spec['phase']=='qualify';serial=spec['scope']=='component'
    doc=dict(kind=('qualify_component' if q else 'calibration') if serial else ('qualify_pipeline' if q else 'performance'),
        configuration=deepcopy(cfg),stages=deepcopy(cfg['stages']),spec=deepcopy(spec),
        environment=protocol()['environment'],**protocol()['qualification' if q else 'component' if serial else 'pipeline'],
        input_indices=[0],executor_sha256=m['phase_runner_sha256'],adapter_sha256=m['adapter_sha256'],
        runtime_sha256=m['runtime_sha256'],bundle_id=m['bundle_id'],experiment=protocol()['experiment'],
        startup_affinity_probe=True)
    doc['task_id']=digest(doc)[:24];return doc


def native_command(doc,bundle,remote_bundle,remote_job):return adapt('native_command')(doc,bundle,remote_bundle,remote_job)
def matched_ssh(config,*,execute_board=False):return adapt('matched_ssh')(config,execute_board=execute_board)
summary=packager.summary
accept=packager.accept


def board(*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    plan=read(POOL/'BOARD_PLAN.json')
    if plan['sources']!=sources() or plan['selection_sha256']!=sha256(POOL/'SELECTION_FREEZE.json'):
        raise ValueError('frozen controller changed')
    if not all(v['passed'] for v in read(POOL/'QEMU.json').values()):raise ValueError('QEMU required')
    ssh=matched_ssh(read(CONFIG)['board'],execute_board=True);boot=ssh.boot_id()
    if boot!=selection()['expected_boot_id'] or boot!=plan['expected_boot_id']:raise ValueError('boot changed; do not mix')
    started=time.monotonic();bundle=POOL/'yolov3_tiny416/capsule';validate(bundle)
    with locked(RUN):
        freeze(RUN/'run.json',dict(boot_id=boot,board_plan_sha256=sha256(POOL/'BOARD_PLAN.json'),protocol=protocol()))
        def state(status,**kw):write(RUN/'state.json',dict(status=status,boot_id=boot,updated_at=now(),**kw),replace=True)
        try:
            (RUN/'yolov3_tiny416').mkdir(parents=True,exist_ok=True)
            remote=ssh.deploy(bundle,RUN/'yolov3_tiny416');ssh.qualify_environment(remote,boot,RUN/'yolov3_tiny416',bundle)
            write(RUN/'hardware.json',hardware_check(ssh,bundle,boot),replace=True)
            expected={};DNN=CPU=0;todo=specs()
            for index,spec in enumerate(todo,1):
                t=make_task(bundle,spec);key=spec['scope'],spec['case']
                state('RUNNING',accepted_sessions=index-1,total_sessions=len(todo),current_spec=spec)
                if spec['phase']=='observed' and key not in expected:raise ValueError('missing observer qualification')
                r=accept(ssh,t,bundle,remote,RUN,boot,expected.get(key))
                if spec['phase']=='qualify':expected[key]=r['summary']['output_fingerprints']
                calls=sum(t[k] for k in ('warmup','scored','tail'))
                if spec['scope']=='pipeline':DNN+=calls
                else:CPU+=calls
                print('三线程残差观察',index,'/',len(todo),spec['scope'],spec['phase'],flush=True)
            if (DNN,CPU)!=(58,11):raise ValueError('bounded measurement budget differs')
            ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > worker_gap_post_verify.log && sync',boot=boot,timeout=300)
            if ssh.boot_id()!=boot:raise ValueError('boot changed at completion')
            state('COMPLETE',accepted_sessions=len(todo),DNN_inferences=DNN,CPU_stage_calls=CPU,
                seconds=time.monotonic()-started,no_reboot=True,no_automatic_extra_measurement=True)
            sync_tree(RUN)
        except Exception as e:
            state('STOPPED_ON_ERROR',error_type=type(e).__name__,error=str(e));raise
    return read(RUN/'state.json')


def verified_receipts():
    state=read(RUN/'state.json')
    if state['status']!='COMPLETE' or state['accepted_sessions']!=6:raise ValueError('observation incomplete')
    bundle=POOL/'yolov3_tiny416/capsule';validate(bundle);rows=[]
    for spec in specs():
        t=make_task(bundle,spec);p=RUN/'accepted'/state['boot_id']/(t['task_id']+'.json');r=read(p)
        if r['task']!=t or r['boot_id']!=state['boot_id']:raise ValueError('observation receipt identity differs')
        verify_inventory(r['raw'],r['files'])
        if summary(r['raw'],t,bundle)!=r['summary']:raise ValueError('observation no longer replays')
        rows.append((p,r))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','qemu','board','validate'))
    p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if a.command=='board' and not a.execute_board:p.error('--execute-board is required')
    if a.command=='board':r=board(execute_board=True)
    elif a.command=='validate':r=dict(passed=True,receipts=len(verified_receipts()))
    else:r=globals()[a.command]()
    print(json.dumps(r,ensure_ascii=False,indent=2))
