"""One matched current-layout timeline; three sessions,97 DNN inferences.

The optional phase executable and two observers already exist. No scheduling,
wait policy, runtime, stage program, static formula or DP change is performed.
"""
from functools import lru_cache
from pathlib import Path
from types import FunctionType
import argparse
import inspect
import json
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,digest,sha256,inventory,verify_inventory,locked,now,sync_tree
from cpu_vta_partitioning.measurement.tasks import native_command as base_command
from cpu_policies.open_resources.domain import validate_candidate
from phase_profile.prepare import link_checked
from phase_profile.runtime import phase_summary
from phase_profile.qemu import qemu_summary
from common_top20.runtime import summary as original_summary
from common_top20.campaign import accept as original_accept
from device_phase.collect import validate_events as absolute_device_events
from matched_device_profile.records import validate_events
from board_transport_without_reprogram import transport
from run_scope_rank import freeze
import run_current_isolated_phases as isolated

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/current_pipeline_timeline_20261112_01'
RUN=ROOT/'runs/current_pipeline_timeline_20261112_01'
BUNDLE=OUT/'capsule'
CONFIG=isolated.CONFIG


@lru_cache(maxsize=1)
def worker_parser():
    from wait_diagnosis.runtime import trace_summary
    source=inspect.getsource(trace_summary)
    before="    if ('0' not in stage_stats or stage_stats['0']['program']!='cpu_00_02' or\n            not stage_stats['0']['kernel_launches']):raise ValueError('actual target CPU module not intercepted')"
    after="    if ('2' not in stage_stats or stage_stats['2']['program']!='cpu_02_04' or\n            not stage_stats['2']['kernel_launches']):raise ValueError('actual current CPU2-4 module not intercepted')"
    if source.count(before)!=1:raise ValueError('verified worker-parser target anchor changed')
    patched=source.replace(before,after);namespace=dict(trace_summary.__globals__)
    exec(compile(patched,'<current seven-stage CPU2-4 target only>','exec'),namespace)
    return namespace['trace_summary'],dict(original_sha256=__import__('hashlib').sha256(source.encode()).hexdigest(),
        adapted_sha256=__import__('hashlib').sha256(patched.encode()).hexdigest(),
        adaptation='required CPU target stage2 cpu_02_04; clock/module/TID/coverage checks unchanged')


def sources():
    names=['run_current_pipeline_timeline.py','tests/test_current_pipeline_timeline.py',
        'run_current_isolated_phases.py','board_transport_without_reprogram.py','common_top20/runtime.py',
        'common_top20/campaign.py','phase_profile/runtime.py','phase_profile/qemu.py','phase_profile/worker.py',
        'wait_diagnosis/runtime.py','wait_diagnosis/adapter.cc','phase_profile/phase_probe.h',
        'device_phase/collect.py','matched_device_profile/records.py','cpu_policies/open_resources/runtime.py']
    return {str(ROOT/n):sha256(ROOT/n) for n in names}


def prepare():
    if (OUT/'PLAN.json').exists():validate();return read(OUT/'PLAN.json')
    isolated.validate();native=read(isolated.RUN/'state.json')
    if native['status']!='COMPLETE':raise ValueError('current isolated inputs must be complete first')
    old=read(isolated.BUNDLE/'manifest.json')
    for leaf,info in old['files'].items():
        if leaf.startswith(('programs/','runtime/','firmware/','inputs/','references/','runner/')):
            link_checked(isolated.BUNDLE/leaf,BUNDLE/leaf,info['sha256'])
    cfg=read(isolated.PARENT/'plans/configuration.json')
    validate_candidate(cfg,read(isolated.BUNDLE/'plans/programs.json'),unit_count=14)
    if cfg['id']!='cfg_2787a77188dd5f9363d00b15' or cfg['execution_mode']!='off':raise ValueError('fixed current layout differs')
    protocol=dict(experiment='current_pipeline_timeline_20261112_01',
        qualification=dict(warmup=0,scored=1,tail=0),performance=dict(warmup=8,scored=16,tail=24),
        environment=read(isolated.BUNDLE/'plans/protocol.json')['environment'],expected_boot=native['boot_id'],
        native_sessions=3,DNN_inferences=97,new_component_calls=0,new_compilation=False,
        no_reboot=True,no_reprogram=True,no_formula_or_DP_change=True,no_automatic_repeat=True,
        original_wait=True,probe_modes=['qualify','clean','observed'],
        goal_authorization='Necessary small matched experiment already authorized by the active goal; no adaptive expansion.',
        purpose='Current CPU2-4 t3 completion tails versus VTA host execution; not old CPU0-2 t4 layout.',
        probe_quality='For CPU2-4, abs(observed-clean) service <=20% of abs(clean-isolated) service gap; otherwise diagnosis is qualitative only. Formal FPS always uses clean.',
        clock='CLOCK_MONOTONIC and same TID CLOCK_THREAD_CPUTIME_ID',
        stage_phase_event_capacity=65536,device_event_capacity=4096,
        no_scheduling_wait_claim_from_all_nonrunning_time=True)
    for leaf,value in [('configuration.json',cfg),('protocol.json',protocol),
        ('programs.json',read(isolated.BUNDLE/'plans/programs.json')),('inputs.json',read(isolated.BUNDLE/'plans/inputs.json')),
        ('board_execution.json',read(isolated.BUNDLE/'plans/board_execution.json')),
        ('preparation.json',dict(sources=sources(),worker_parser=worker_parser()[1],
            isolated_manifest_sha256=sha256(isolated.BUNDLE/'manifest.json'),
            isolated_phase_projection_sha256=sha256(isolated.OUT/'projection/ISOLATED_PHASES.json'),
            bridge='New same-executable clean session; previous original-executable clean remains a separate identity.'))]:
        freeze(BUNDLE/'plans'/leaf,value)
    for i,(path,h) in enumerate(sorted(sources().items())):
        dst=BUNDLE/'sources'/f'{i:02}_{Path(path).name}';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dst)
        if sha256(dst)!=h:raise ValueError('controller changed while copying')
    assets=inventory(BUNDLE);payload={k:v for k,v in assets.items() if k.startswith(('programs/','runtime/','firmware/','inputs/','references/','runner/'))}
    payload['plans/board_execution.json']=assets['plans/board_execution.json']
    freeze(BUNDLE/'plans/board_payload.json',dict(files=payload,unpacked_bytes=sum(v['bytes'] for v in payload.values())))
    m=dict(network='yolov3_tiny416',experiment=protocol['experiment'],runner_sha256=old['runner_sha256'],
        runtime_sha256=old['runtime_sha256'],adapter_sha256=old['adapter_sha256'],observer_sha256=old['observer_sha256'],files=inventory(BUNDLE))
    m['bundle_id']=digest(m)[:24];freeze(BUNDLE/'manifest.json',m)
    freeze(OUT/'PLAN.json',dict(sources=sources(),protocol=protocol,bundle_id=m['bundle_id']))
    freeze(OUT/'TASKS.json',dict(plan_sha256=sha256(OUT/'PLAN.json'),tasks=[task(s) for s in ('qualify','clean','observed')]))
    validate();return read(OUT/'PLAN.json')


def validate():
    m=read(BUNDLE/'manifest.json')
    if digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('capsule changed')
    verify_inventory(BUNDLE,m['files']);proof=read(BUNDLE/'plans/preparation.json')
    if proof['sources']!=sources() or proof['worker_parser']!=worker_parser()[1]:raise ValueError('frozen timeline source changed')
    if sha256(isolated.BUNDLE/'manifest.json')!=proof['isolated_manifest_sha256']:raise ValueError('isolated executor identity changed')
    if sha256(isolated.OUT/'projection/ISOLATED_PHASES.json')!=proof['isolated_phase_projection_sha256']:raise ValueError('isolated inputs changed')
    return m


def task(mode):
    if mode not in ('qualify','clean','observed'):raise ValueError('outside the three-session timeline scope')
    m=read(BUNDLE/'manifest.json');p=read(BUNDLE/'plans/protocol.json');cfg=read(BUNDLE/'plans/configuration.json')
    doc=dict(kind='qualify_pipeline' if mode=='qualify' else 'performance',phase=mode,
        configuration=cfg,stages=cfg['stages'],**p['qualification' if mode=='qualify' else 'performance'],
        environment=p['environment'],input_indices=[0],executor_sha256=m['runner_sha256'],runtime_sha256=m['runtime_sha256'],
        adapter_sha256=m['adapter_sha256'] if mode!='clean' else None,observer_sha256=m['observer_sha256'] if mode!='clean' else None,
        spec=dict(scope='pipeline',policy='original',probe=mode!='clean'),detailed=False,startup_affinity_probe=True,
        experiment=p['experiment'],bundle_id=m['bundle_id'])
    doc['task_id']=digest(doc)[:24];return doc


def command(doc,bundle,remote,job):
    if doc!=task(doc['phase']):raise ValueError('timeline task differs')
    argv,inputs=base_command(doc,bundle,remote,job)
    argv[argv.index(remote+'/runner/vta_stage_pipeline_runner')]=remote+'/runner/vta_stage_pipeline_runner_phase'
    at=argv.index(remote+'/runtime/ld-linux-aarch64.so.1');argv[at:at]=['TVM_BIND_THREADS=1','TVM_EXCLUDE_WORKER0=1']
    argv+=['--cpu-worker-pinning','--allow-cpu-affinity-superset','--resource-affinity-probe',
        '--control-cpu-affinity',','.join(map(str,doc['configuration']['control_mask']))]
    if doc['phase']!='qualify':argv[argv.index('--output-mode')+1]='raw'
    if doc['spec']['probe']:
        index=next(i for i,s in enumerate(argv) if s.startswith('LD_PRELOAD='))
        argv[index]='LD_PRELOAD='+remote+'/runner/libdevice_phase.so:'+remote+'/runner/libwait_adapter.so:'+remote+'/runtime/libvta.so'
        at=argv.index(remote+'/runtime/ld-linux-aarch64.so.1')
        argv[at:at]=['VTA_WAIT_POLICY=original','VTA_WAIT_PROBE=1','VTA_WAIT_TRACE='+job+'/wait_trace.jsonl',
            'VTA_WAIT_STATUS='+job+'/wait_status.json','VTA_DEVICE_PHASE_TRACE='+job+'/device_trace.jsonl']
        argv+=['--phase-event-json',job+'/phase_trace.jsonl']
    if '--process-cpu-affinity' in argv or '--queue-timing' in argv:raise ValueError('unrequested execution change')
    return argv,inputs


def summarize(raw,doc,bundle):
    if doc!=task(doc['phase']):raise ValueError('timeline receipt identity differs')
    def bound_task(bundle,cfg,phase):
        expected=task(phase)
        if cfg!=expected['configuration']:raise ValueError('configuration differs')
        return expected
    result=FunctionType(original_summary.__code__,{**original_summary.__globals__,'task':bound_task})(raw,doc,bundle)
    result['phase']=phase_summary(raw,doc,read(Path(raw)/'affinity.json'))
    if doc['spec']['probe']:
        parser,proof=worker_parser();affinity=read(Path(raw)/'affinity.json')
        result['worker']=parser(raw,doc,bundle,affinity);result['worker_parser']=proof
        phase=[json.loads(s) for s in (Path(raw)/'phase_trace.jsonl').read_text().splitlines()]
        events=[json.loads(s) for s in (Path(raw)/'device_trace.jsonl').read_text().splitlines()]
        rows=[json.loads(s) for s in (Path(raw)/'result.jsonl').read_text().splitlines()]
        result['absolute_device']=absolute_device_events(events,phase,doc)
        result['device']=validate_events(events,doc,affinity,rows)
    elif any((Path(raw)/name).exists() for name in ('phase_trace.jsonl','wait_trace.jsonl','device_trace.jsonl')):
        raise ValueError('unrequested observers in clean performance')
    return result


def qemu():
    validate();results=[]
    # CPU-only combined-preload qualification exercises the real runtime and
    # callbacks. No VTA graph or hardware emulation is attempted.
    for name in isolated.CASES[:2]:
        doc=isolated.task(dict(case=name,mode='qualify',scope='component'))
        raw=OUT/'qemu'/name;raw.mkdir(parents=True,exist_ok=True)
        argv,inputs=isolated.command(doc,BUNDLE,str(BUNDLE),str(raw))
        index=next(i for i,s in enumerate(argv) if s.startswith('LD_PRELOAD='))
        argv[index]='LD_PRELOAD='+str(BUNDLE)+'/runner/libdevice_phase.so:'+str(BUNDLE)+'/runner/libwait_adapter.so:'+str(BUNDLE)+'/runtime/libvta.so'
        (raw/'inputs.txt').write_text(inputs);at=argv.index(str(BUNDLE)+'/runtime/ld-linux-aarch64.so.1')
        call=[shutil.which('qemu-aarch64-static'),*[x for item in argv[2:at] for x in ('-E',item)],*argv[at:]]
        result=subprocess.run(call,capture_output=True,text=True,timeout=180)
        (raw/'qemu.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise ValueError('combined observers ARM CPU qualification failed')
        checked=qemu_summary(raw,doc,BUNDLE);results.append(dict(program=name,passed=True,summary=checked))
    value=dict(passed=True,bundle_id=read(BUNDLE/'manifest.json')['bundle_id'],CPU_cases=results,
        new_DNN_inferences=0,board_access=False,combined_observers_and_original_wait=True)
    freeze(OUT/'QEMU.json',value);return value


def accept(ssh,doc,remote,boot,expected=None):
    path=RUN/'accepted'/boot/(doc['task_id']+'.json')
    if path.exists():
        value=read(path)
        if value['task']!=doc or value['boot_id']!=boot:raise ValueError('accepted timeline identity changed')
        verify_inventory(value['raw'],value['files'])
        if digest(summarize(value['raw'],doc,BUNDLE))!=digest(value['summary']):raise ValueError('accepted timeline cannot replay')
        if expected is not None and value['summary']['output_fingerprints']!=expected:raise ValueError('accepted outputs differ')
        return value
    f=FunctionType(original_accept.__code__,{**original_accept.__globals__,'summary':summarize})
    return f(ssh,doc,BUNDLE,remote,RUN,boot,expected)


def board(*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    validate();plan=read(OUT/'PLAN.json');proof=read(OUT/'QEMU.json')
    if not proof['passed'] or proof['bundle_id']!=plan['bundle_id']:raise ValueError('matching combined ARM proof required')
    tasks=read(OUT/'TASKS.json')
    if tasks['plan_sha256']!=sha256(OUT/'PLAN.json') or tasks['tasks']!=[task(s) for s in ('qualify','clean','observed')]:
        raise ValueError('timeline frozen scope differs')
    ssh=transport(read(CONFIG)['board'],command,execute_board=True);boot=ssh.boot_id()
    if boot!=plan['protocol']['expected_boot']:raise ValueError('boot changed; do not mix timeline batches')
    completed=0;items={};start=time.monotonic()
    with locked(RUN):
        freeze(RUN/'run.json',dict(boot_id=boot,bundle_id=plan['bundle_id'],plan_sha256=sha256(OUT/'PLAN.json')))
        def state(status,**extra):write(RUN/'state.json',dict(status=status,accepted_sessions=completed,
            boot_id=boot,updated_at=now(),**extra),replace=True)
        try:
            remote=ssh.deploy(BUNDLE,RUN);ssh.qualify_environment(remote,boot,RUN,BUNDLE)
            for doc in tasks['tasks']:
                mode=doc['phase'];state('RUNNING',current_task=doc['task_id'],mode=mode)
                expected=None if mode=='qualify' else items['qualify']['summary']['output_fingerprints']
                items[mode]=accept(ssh,doc,remote,boot,expected);completed+=1
                print('Current seven-stage timeline',completed,'/3',mode,flush=True)
            ssh.idle_command('cd '+__import__('shlex').quote(remote)+' && sha256sum -c board.sha256 > timeline_post_verify.log && sync',boot=boot,timeout=600)
            if ssh.boot_id()!=boot:raise ValueError('boot changed after timeline')
            state('COMPLETE',native_sessions=3,new_DNN=97,new_component_calls=0,no_reboot=True,
                no_reprogram=True,seconds=time.monotonic()-start)
            sync_tree(RUN)
        except Exception as error:state('STOPPED_ON_ERROR',error=str(error),error_type=type(error).__name__);raise
    return read(RUN/'state.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','validate','qemu','board','status'))
    p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    v=prepare() if a.command=='prepare' else validate() if a.command=='validate' else qemu() if a.command=='qemu' else board(execute_board=a.execute_board) if a.command=='board' else read(RUN/'state.json') if (RUN/'state.json').exists() else dict(status='NOT_STARTED')
    print(json.dumps(v,ensure_ascii=False,indent=2))
