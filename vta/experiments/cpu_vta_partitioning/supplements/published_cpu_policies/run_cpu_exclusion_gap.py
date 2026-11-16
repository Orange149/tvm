"""Matched diagnosis using the already-frozen runner's global CPU admission.

Four layouts: off/exclusive for one known layout in each network. No adapter,
new runtime, new modules, model fitting or Top20 claim. Eight sessions /116 DNN.
"""
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import argparse
import inspect
import json
import math
import random
import statistics as st
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,digest,verify_inventory
from finite_service.common import freeze
from phase_profile import runtime as phase
from phase_profile.campaign import accept as original_accept
from cpu_vta_partitioning.measurement import frames
from scheduler_gap_audit.curves import csv_write
import finite_service_holdout_board as parent

ROOT=parent.ROOT;OUTPUT=parent.OUTPUT;CONFIG=parent.CONFIG
POOL=ROOT/'artifacts/cpu_exclusion_gap_01'
RUN=ROOT/'runs/cpu_exclusion_gap_01'
BOOT='6018b3b7-bb29-4f80-92c1-0bfd8c81bb99'
CID='cfg_2ac037fa511b37a62f7c43a4'


def protocol():
    p=parent.protocol();p.update(experiment='cpu_exclusion_gap_01',seed=20261111,
        purpose='same layouts, only global CPU admission changes; stage-profile gap versus throughput tradeoff',
        known_case_only=True,no_observer=True,execution_mechanism_intervention=True,
        global_exclusion_scope='CPU set/run/get/output snapshot; VTA and controls still share original cores',
        no_CFS_model_branch=True,no_new_profiles=True,no_automatic_repeat=True)
    return p


def sources():
    return {**parent.sources(),**{str(ROOT/p):sha256(ROOT/p) for p in
        ('run_cpu_exclusion_gap.py','tests/test_cpu_exclusion_gap.py')}}


def intervention(config,mode):
    if mode not in ('off','exclusive'):raise ValueError('unplanned global CPU admission')
    if mode=='off':
        if config['execution_mode']!='off':raise ValueError('reference layout is not open execution')
        return deepcopy(config)
    c=deepcopy(config);c['execution_mode']=mode
    c.pop('id',None);c['id']='cfg_'+digest(c)[:24]
    return c


def selection():
    res=read(ROOT/'artifacts/compiled_gap_validation_01/INPUTS.json')['networks']['resnet18'][CID]['configuration']
    yolo=read(ROOT/'artifacts/scheduler_gap_audit_01/yolov3_tiny416/capsule/plans/configurations.json')['pipelines']['overlap']
    return dict(selected_configurations={n:[intervention(c,m) for m in ('off','exclusive')]
        for n,c in [('resnet18',res),('yolov3_tiny416',yolo)]},DNN_budget=116,expected_boot_id=BOOT)


def specs():
    pairs=[(n,c['id']) for n,rows in selection()['selected_configurations'].items() for c in rows]
    random.Random(protocol()['seed']).shuffle(pairs)
    return [dict(network=n,case=cid,scope='pipeline',phase=p,runner='phase',probe=False)
        for n,cid in pairs for p in ('qualify','clean')]


def adapt(name):
    f=getattr(parent,name);g=FunctionType(f.__code__,{**f.__globals__,**globals()},f.__name__,f.__defaults__)
    g.__kwdefaults__=f.__kwdefaults__;return g


def component_references():
    intr=read(ROOT/'artifacts/cross_network_model_01/pool_temporal/INTRINSICS.json');result={}
    for network,configs in selection()['selected_configurations'].items():
        cfg=configs[0];rows=[]
        for s in cfg['stages']:
            if s['device']!='cpu':continue
            x=next(v for v in intr['CPU'].values() if v['identity']['network']==network
                and v['identity']['program']==s['program'] and v['identity']['threads']==s['threads'])
            pv=x['provenance'];clean=pv.get('clean_receipt');path=clean or pv['receipt']
            expected=pv.get('clean_sha256') if clean else pv['sha256']
            if sha256(path)!=expected:raise ValueError('isolated reference changed')
            receipt=read(path);verify_inventory(receipt['raw'],receipt['files'])
            cost=receipt['summary']['cost']
            rows.append(dict(program=s['program'],threads=s['threads'],requested_mask=s['mask'],
                source_mask=x['identity']['mask'],homogeneous_core_permutation=s['mask']!=x['identity']['mask'],
                service_ms=cost['service_ms'],scope='clean isolated' if clean else 'observed isolated; no matching clean receipt in source index',
                receipt=path,receipt_sha256=expected,boot_id=receipt['boot_id'],
                module_sha256=x['identity']['module_sha256'],executor_sha256=receipt['task']['executor_sha256']))
        result[network]=rows
    return result


def prepare():
    references=component_references()
    freeze(POOL/'PLAN.json',dict(sources=sources(),protocol=protocol(),selection=selection(),
        isolated_reference=references,native_sessions=8,DNN_inferences=116,
        hypothesis='prevent CPU subgraphs from overlapping, then see whether service recovers and whether serialization sacrifices throughput',
        comparison='same current batch off/exclusive; no old FPS substituted',
        unchanged='topology, threads, physical maps, VTA host/control masks, binaries, inputs, runtime, queues, wait policy',
        gate_wait_is_outside_stage_service_but_inside_end_to_end=True,
        no_instrumented_performance=True,no_new_predictor=True,
        acceptance='complete outputs/affinity/boot/hash/frames, gate leases non-overlap and cover CPU service; report negative performance too',
        no_formula_change_or_followup_measurement_trigger=True))
    freeze(POOL/'SELECTION_FREEZE.json',dict(**selection(),plan_sha256=sha256(POOL/'PLAN.json')))
    return adapt('prepare')()


def validate(bundle):
    if read(POOL/'PLAN.json')['sources']!=sources():raise ValueError('frozen control changed')
    return adapt('validate')(bundle)


def make_task(bundle,spec):
    b=Path(bundle);m=read(b/'manifest.json')
    if spec not in specs():raise ValueError('unplanned CPU exclusion task')
    cfg=read(b/'plans/configurations.json')['pipelines'][spec['case']];q=spec['phase']=='qualify'
    doc=dict(kind='qualify_pipeline' if q else 'performance',configuration=deepcopy(cfg),stages=deepcopy(cfg['stages']),
        spec=deepcopy(spec),environment=protocol()['environment'],**protocol()['qualification' if q else 'pipeline'],
        input_indices=[0],executor_sha256=m['phase_runner_sha256'],adapter_sha256=None,
        runtime_sha256=m['runtime_sha256'],bundle_id=m['bundle_id'],experiment=protocol()['experiment'],
        startup_affinity_probe=True,cpu_admission=cfg['execution_mode'])
    doc['task_id']=digest(doc)[:24];return doc


def native_command(doc,bundle,remote_bundle,remote_job):
    argv,blobs=FunctionType(phase.native_command.__code__,{**phase.native_command.__globals__,'make_task':make_task})(doc,bundle,remote_bundle,remote_job)
    argv[argv.index('--cpu-stage-gate')+1]=doc['cpu_admission']
    if any('libwait_adapter' in a for a in argv) or '--phase-event-json' in argv or '--session-profile-json' in argv:
        raise ValueError('CPU admission comparison must remain uninstrumented')
    return argv,blobs


def gate_evidence(rows,doc):
    mode=doc['cpu_admission'];leases=[];waits={}
    if any(row.get('cpu_stage_gate_mode','off')!=mode for row in rows):raise ValueError('wrong native gate')
    if mode=='off':return dict(mode='off',exclusive_not_claimed=True)
    for row in rows:
        for index,stage in enumerate(doc['stages']):
            p='stage'+str(index)
            if stage['device']!='cpu':
                if any(row.get(p+'_cpu_gate_'+key+'_ms') is not None for key in ('request','acquire','release','wait')):
                    raise ValueError('VTA unexpectedly owns CPU admission')
                continue
            values=[row[p+'_cpu_gate_'+key+'_ms'] for key in ('request','acquire','release','wait')]
            if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in values):
                raise ValueError('invalid CPU admission timestamps')
            request,acquire,release,wait=values
            if (not request<=acquire<=row[p+'_start_ms']<=row[p+'_end_ms']<=release
                or abs(wait-(acquire-request))>0.000002):raise ValueError('CPU admission does not cover service')
            leases.append((acquire,release,index,row['frame_id']))
            if doc['warmup']<=row['frame_id']<doc['warmup']+doc['scored']:
                waits.setdefault(stage['program'],[]).append(wait)
    leases.sort()
    if any(a[1]>b[0]+0.000002 for a,b in zip(leases,leases[1:])):
        raise ValueError('globally excluded CPU services overlap')
    return dict(mode='exclusive',leases=len(leases),all_CPU_services_covered=True,
        observed_max_CPU_service_concurrency=1,gate_wait_ms={k:st.mean(v) for k,v in waits.items()},
        waits_not_added_to_stage_service_or_II=True,FIFO_tickets_not_directly_observed=True)


def summary(raw,doc,bundle):
    source=inspect.getsource(phase.summary);anchor="r.get('cpu_stage_gate_mode','off')!='off'"
    if source.count(anchor)!=1:raise ValueError('frozen phase validator gate hook changed')
    ns=dict(phase.summary.__globals__)
    exec(compile(source.replace(anchor,"r.get('cpu_stage_gate_mode','off')!=doc['cpu_admission']"),
        '<exact requested CPU admission validation>','exec'),ns)
    result=ns['summary'](raw,doc,bundle)
    result['CPU_admission']=gate_evidence(frames(Path(raw)/'result.jsonl',sum(doc[k] for k in ('warmup','scored','tail'))),doc)
    return result


accept=FunctionType(original_accept.__code__,{**original_accept.__globals__,'summary':summary},argdefs=original_accept.__defaults__)
def matched_ssh(config,*,execute_board=False):return adapt('matched_ssh')(config,execute_board=execute_board)
def qemu():return adapt('qemu')()
def board(*,execute_board=False):return adapt('board')(execute_board=execute_board)


def analyze():
    state=read(RUN/'state.json')
    if state['status']!='COMPLETE' or state['accepted_sessions']!=8 or state['boot_id']!=BOOT:
        raise ValueError('CPU admission cohort incomplete')
    rows=[];qualified={};hashes={};refs=read(POOL/'PLAN.json')['isolated_reference']
    for spec in specs():
        b=POOL/spec['network']/'capsule';validate(b);t=make_task(b,spec);path=RUN/'accepted'/BOOT/(t['task_id']+'.json');r=read(path)
        if r['task']!=t or r['boot_id']!=BOOT:raise ValueError('CPU admission receipt identity differs')
        verify_inventory(r['raw'],r['files'])
        if summary(r['raw'],t,b)!=r['summary']:raise ValueError('CPU admission evidence no longer replays')
        key=(spec['network'],spec['case']);hashes[str(path)]=sha256(path)
        if spec['phase']=='qualify':qualified[key]=r['summary']['output_fingerprints'];continue
        if r['summary']['output_fingerprints']!=qualified[key]:raise ValueError('outputs differ from qualification')
        value=r['summary']
        for i,s in enumerate(t['stages']):
            if s['device']!='cpu':continue
            reference=next(x for x in refs[spec['network']] if x['program']==s['program'])
            rows.append(dict(network=spec['network'],mode=t['cpu_admission'],program=s['program'],threads=s['threads'],mask=s['mask'],
                isolated_service_ms=reference['service_ms'],pipeline_service_ms=value['stage_service_ms'][i],
                service_gap_ms=value['stage_service_ms'][i]-reference['service_ms'],
                gate_wait_ms=value['CPU_admission'].get('gate_wait_ms',{}).get(s['program'],0),
                FPS=value['fps'],II_ms=value['ii_ms'],process_core_ms_per_completion=value['process_core_ms_per_completion']))
    result=dict(rows=rows,receipts=hashes,new_native_sessions=8,new_DNN=116,no_new_component_profiles=True,
        same_batch_matched_execution_intervention=True,no_model_or_Top20_claim=True,
        no_detailed_wait_or_scheduler_probe=True,source_sha256=sha256(__file__))
    write(POOL/'RESULTS.json',result,replace=True);csv_write(POOL/'stages.csv',rows)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','validate','qemu','board','status','analyze'));p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    if a.command=='board':r=board(execute_board=a.execute_board)
    elif a.command=='validate':r={n:validate(POOL/n/'capsule') for n in selection()['selected_configurations']}
    elif a.command=='status':r=read(RUN/'state.json') if (RUN/'state.json').exists() else dict(status='NOT_STARTED')
    else:r=globals()[a.command]()
    print(json.dumps(r,ensure_ascii=False,indent=2))
