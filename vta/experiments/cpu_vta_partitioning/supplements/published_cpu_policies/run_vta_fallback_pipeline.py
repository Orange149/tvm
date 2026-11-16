"""Three matched known YOLO layouts, changing only CPU pools inside VTA stages.

Six sessions /87 complete DNNs; separate batch from the accepted components.
No new compiled code, DP ranking, cost fitting, or bitstream reload.
"""
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import argparse
import inspect
import json
import random
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,sha256,digest,write,verify_inventory
from cpu_vta_partitioning.transport import SSH
from finite_service.common import freeze
from phase_profile import runtime as phase
from cpu_policies.open_resources.domain import stage
from yolo_validation.components import hardware_check
import finite_service_holdout_board as parent
import run_vta_fallback_threads as components

ROOT=parent.ROOT;OUTPUT=parent.OUTPUT;CONFIG=parent.CONFIG
POOL=ROOT/'artifacts/vta_CPU_fallback_pipeline_01'
RUN=ROOT/'runs/vta_CPU_fallback_pipeline_01'


def protocol():
    p=parent.protocol();p.update(experiment='vta_CPU_fallback_pipeline_01',
        only_local_pool_validation=True,purpose='matched CPU interference from runtime pools inside the same VTA graphs',
        VTA_CPU_thread_pairs=[[1,1],[4,1],[4,4]],native_sessions=6,DNN_inferences=87,
        no_new_profile_or_prediction_fitting=True,no_static_score_DP_or_Top20_change=True)
    return p


def sources():
    return {**parent.sources(),**{str(ROOT/p):sha256(ROOT/p) for p in
        ('run_vta_fallback_pipeline.py','tests/test_vta_fallback_pipeline.py','run_vta_fallback_threads.py')}}


def selection():
    if read(components.RUN/'state.json')['status']!='COMPLETE':raise ValueError('component cohort incomplete')
    old=read(ROOT/'artifacts/cpu_exclusion_gap_01/PLAN.json')['selection']['selected_configurations']['yolov3_tiny416'][0]
    if old['execution_mode']!='off' or [s['threads'] for s in old['stages'] if s['device']=='cpu']!=[4,3,1]:
        raise ValueError('known CPU allocation differs')
    configs=[];labels={}
    for pair in protocol()['VTA_CPU_thread_pairs']:
        c=deepcopy(old);at=0
        for i,s in enumerate(c['stages']):
            if s['device']=='vta':
                c['stages'][i]=stage('vta',s['begin'],s['end'],pair[at],s['mask']);at+=1
        if pair!=[1,1]:c.pop('id');c['id']='cfg_'+digest(c)[:24]
        configs.append(c);labels[c['id']]=pair
    return dict(selected_configurations={'yolov3_tiny416':configs},labels=labels,
        DNN_budget=87,expected_boot_id=components.protocol()['expected_boot_id'])


def specs():
    pairs=[('yolov3_tiny416',c['id']) for c in selection()['selected_configurations']['yolov3_tiny416']]
    random.Random(protocol()['seed']).shuffle(pairs)
    return [dict(network=n,case=cid,scope='pipeline',phase=p,runner='phase',probe=False)
        for n,cid in pairs for p in ('qualify','clean')]


def adapt(name):
    f=getattr(parent,name);g=FunctionType(f.__code__,{**f.__globals__,**globals()},f.__name__,f.__defaults__)
    g.__kwdefaults__=f.__kwdefaults__;return g


def prepare():
    evidence=read(components.POOL/'RESULTS.json')
    for path,expected in evidence['source_receipts'].items():
        if sha256(path)!=expected:raise ValueError('isolated component source changed')
        r=read(path);verify_inventory(r['raw'],r['files'])
    # Frozen runner serializes actor initialization before /proc ownership deltas.
    source=ROOT/'artifacts/scheduler_gap_audit_01/build/sources/vta_stage_pipeline_runner_phase.cc'
    text=source.read_text()
    if ('initialization_ready.wait(lock, [&] { return initialization_finished[stage_index]; });' not in text
        or 'Do not create the next actor until this actor has registered its' not in text):
        raise ValueError('multi-pool thread ownership cannot be established')
    estimates=[]
    for cfg in selection()['selected_configurations']['yolov3_tiny416']:
        matches=[next(r for r in evidence['rows'] if r['program']==s['program'] and r['runtime_threads']==s['threads'])
            for s in cfg['stages'] if s['device']=='vta']
        estimates.append(dict(configuration=cfg['id'],VTA_CPU_threads=selection()['labels'][cfg['id']],
            isolated_VTA_service_sum_ms=sum(r['service_ms'] for r in matches),
            isolated_VTA_run_and_boundary_core_ms=sum(r['run_core_ms']+r['boundary_core_ms'] for r in matches),
            scope='isolated module sums only; not a new full-pipeline prediction'))
    freeze(POOL/'PLAN.json',dict(sources=sources(),protocol=protocol(),selection=selection(),
        component_evidence_sha256=sha256(components.POOL/'RESULTS.json'),estimates=estimates,
        no_pipeline_FPS_used_to_select_cases=True,old_binaries_reused=True,
        no_automatic_more_windows_or_cases=True,bitstream_reload=False,
        interpret='stage service, CPU work and full throughput jointly; no uniform choice inferred from isolated speedup',
        serialized_pool_initialization_source_sha256=sha256(source)))
    freeze(POOL/'SELECTION_FREEZE.json',dict(**selection(),plan_sha256=sha256(POOL/'PLAN.json')))
    result=adapt('prepare')()
    if read(POOL/'BOARD_PLAN.json')['expected_boot_id']!=selection()['expected_boot_id']:
        raise ValueError('packager expected a different boot')
    return result


def validate(bundle):
    if read(POOL/'PLAN.json')['sources']!=sources():raise ValueError('frozen controller differs')
    return adapt('validate')(bundle)


def make_task(bundle,spec):return adapt('make_task')(bundle,spec)


def native_command(doc,bundle,remote_bundle,remote_job):
    f=phase.native_command
    return FunctionType(f.__code__,{**f.__globals__,'make_task':make_task})(doc,bundle,remote_bundle,remote_job)


summary=phase.summary
accept=parent.accept


def matched_ssh(config,*,execute_board=False):
    measure=FunctionType(SSH.measure.__code__,{**SSH.measure.__globals__,'native_command':native_command})
    def readonly_environment(self,remote,boot,run_dir,bundle):
        result=hardware_check(self,bundle,boot)
        write(Path(run_dir)/'environment-read-only.json',result,replace=True)
    return type('VTAFallbackPipelineSSH',(SSH,),{'measure':measure,
        'qualify_environment':readonly_environment})(config,execute_board=execute_board)


def qemu():return adapt('qemu')()
def board(*,execute_board=False):return adapt('board')(execute_board=execute_board)


def analyze():
    state=read(RUN/'state.json');boot=selection()['expected_boot_id']
    if state['status']!='COMPLETE' or state['accepted_sessions']!=6 or state['boot_id']!=boot:
        raise ValueError('pipeline cohort incomplete')
    rows=[];hashes={};qualified={}
    b=POOL/'yolov3_tiny416/capsule';validate(b)
    for spec in specs():
        t=make_task(b,spec);p=RUN/'accepted'/boot/(t['task_id']+'.json');r=read(p)
        if r['task']!=t or r['boot_id']!=boot:raise ValueError('pipeline receipt differs')
        verify_inventory(r['raw'],r['files']);value=summary(r['raw'],t,b)
        if value!=r['summary']:raise ValueError('pipeline evidence does not replay')
        hashes[str(p)]=sha256(p)
        if spec['phase']=='qualify':qualified[spec['case']]=value['output_fingerprints'];continue
        if value['output_fingerprints']!=qualified[spec['case']]:raise ValueError('qualification output differs')
        rows.append(dict(configuration=spec['case'],VTA_CPU_threads=selection()['labels'][spec['case']],
            FPS=value['fps'],II_ms=value['ii_ms'],stage_service_ms=value['stage_service_ms'],
            whole_process_core_ms_per_completion=value['process_core_ms_per_completion'],
            helper_CPU_ms_by_VTA_stage={s['program']:sum(x['cpu_time_delta_ms'] for x in value['thread_ledger']
                if x.get('stage')==s['program'] and x['role']=='worker') for s in t['stages'] if s['device']=='vta'},
            thread_scope='whole session counters, distinct from retained-window process CPU time'))
    if len({json.dumps(v) for v in qualified.values()})!=1:raise ValueError('different runtime pools changed network outputs')
    result=dict(rows=rows,source_receipts=hashes,new_native_sessions=6,DNN_inferences=87,
        stage_profiles_used_for_preregistered_cases_only=True,no_new_formula_DP_or_Top20=True,
        single_boot_short_window=True)
    write(POOL/'RESULTS.json',result,replace=True);csv_write(POOL/'pipelines.csv',rows)
    print(json.dumps(rows,ensure_ascii=False,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','validate','qemu','board','status','analyze'))
    p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    r=board(execute_board=a.execute_board) if a.command=='board' else read(RUN/'state.json') if a.command=='status' else \
        {n:validate(POOL/n/'capsule') for n in selection()['selected_configurations']} if a.command=='validate' else globals()[a.command]()
    if a.command!='analyze':print(json.dumps(r,ensure_ascii=False,indent=2))
