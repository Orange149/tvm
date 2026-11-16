"""Five missing current-YOLO isolation inputs; no pipeline or search commands.

Reuse the authenticated phase executor and original-wait observer. All native
tasks are frozen before deployment: 15 sessions,105 component calls,0 DNNs.
"""
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import argparse
import json
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,digest,inventory,verify_inventory,locked,now,sync_tree
from cpu_vta_partitioning.measurement.tasks import native_command as base_command
from cpu_policies.open_resources.domain import component,validate_candidate
from phase_profile.prepare import link_checked
from phase_profile.runtime import summary as base_summary,phase_summary
from phase_profile.qemu import qemu_summary
from matched_device_profile.records import validate_events
from device_phase.collect import validate_events as absolute_device_events
from common_top20.campaign import accept as original_accept
from board_transport_without_reprogram import transport
from run_scope_rank import freeze

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/current_isolated_phases_20261112_01'
RUN=ROOT/'runs/current_isolated_phases_20261112_01'
BUNDLE=OUT/'capsule'
PARENT=ROOT/'artifacts/matched_device_profile_20261112_01/capsule'
PHASE=ROOT/'artifacts/cross_network_model_01/phase_profile/yolov3_tiny416/capsule'
CONFIG=ROOT/'artifacts/cross_network_model_01/board.local.json'
CASES=('cpu_02_04','cpu_12_13','vta_01_01','vta_05_06','vta_09_11')


def sources():
    names=['run_current_isolated_phases.py','tests/test_current_isolated_phases.py',
        'board_transport_without_reprogram.py','common_top20/campaign.py',
        'phase_profile/runtime.py','phase_profile/worker.py','phase_profile/qemu.py',
        'phase_model/extract.py','phase_profile/patch.py','phase_profile/phase_probe.h',
        'matched_device_profile/records.py','device_phase/collect.py','completion_profile/workflow.py',
        'wait_diagnosis/runtime.py','cpu_policies/open_resources/runtime.py']
    return {str(ROOT/n):sha256(ROOT/n) for n in names}


def specifications():
    return [dict(case=case,mode=mode,scope='component') for case in CASES
            for mode in ('qualify','clean','observed')]


def prepare():
    if (OUT/'PLAN.json').exists():validate();return read(OUT/'PLAN.json')
    import run_matched_device_profile as matched
    matched.validate()
    native=read(matched.RUN/'state.json')
    if native['status']!='COMPLETE':raise ValueError('previous matched diagnosis must be complete')
    old=read(PARENT/'manifest.json');phase=read(PHASE/'manifest.json')
    if phase['runtime_sha256']!=old['runtime_sha256']:raise ValueError('phase runtime mismatch')
    verify_inventory(PHASE,phase['files'])
    for leaf,info in old['files'].items():
        if leaf.startswith(('programs/','runtime/','firmware/','inputs/','references/','runner/')):
            link_checked(PARENT/leaf,BUNDLE/leaf,info['sha256'])
    for leaf in ('runner/vta_stage_pipeline_runner_phase','runner/libwait_adapter.so'):
        link_checked(PHASE/leaf,BUNDLE/leaf,phase['files'][leaf]['sha256'])
    pipeline=read(PARENT/'plans/configuration.json')
    configs={name:component(next(s for s in pipeline['stages'] if s['program']==name),pipeline) for name in CASES}
    protocol=dict(experiment='current_isolated_phases_20261112_01',network='yolov3_tiny416',
        qualification=dict(warmup=0,scored=1,tail=0),component=dict(warmup=2,scored=8,tail=0),
        native_sessions=15,component_calls=105,CPU_component_calls=42,VTA_component_calls=63,
        DNN_inferences=0,no_reboot=True,no_reprogram=True,new_compilation=False,no_pipeline=True,
        original_wait=True,automatic_retry=False,extra_MMIO_reads=0,
        environment=read(PARENT/'plans/protocol.json')['environment'],
        expected_boot=native['boot_id'],probe_quality_limit_percent=20,
        decision='Use only if complete-output/thread/trace checks pass and clean/observed CPU/wall changes <=20%; otherwise retain raw and stop this input route without automatic repeats.',
        reason='Two current multithread modules lack fork/join profiles. Three current VTA modules lack a measured pre-first/post-last host CPU split. Whole-session thread shares are not exact per-thread phase inputs.',
        authorization='Active user goal permits necessary small-scale isolated measurements; no new DNN inference or full-domain calibration.')
    for leaf,value in [('programs.json',read(PARENT/'plans/programs.json')),
        ('inputs.json',read(PARENT/'plans/inputs.json')),('board_execution.json',read(PARENT/'plans/board_execution.json')),
        ('configurations.json',configs),('protocol.json',protocol),
        ('preparation.json',dict(sources=sources(),parent_manifest_sha256=sha256(PARENT/'manifest.json'),
            phase_manifest_sha256=sha256(PHASE/'manifest.json'),
            phase_patch_scope='same RunStage operations with opt-in boundary CLOCK_MONOTONIC/THREAD_CPUTIME; no execution strategy change',
            reused_phase_runner_sha256=phase['phase_runner_sha256'],reused_wait_adapter_sha256=phase['adapter_sha256']))]:
        freeze(BUNDLE/'plans'/leaf,value)
    for i,(path,h) in enumerate(sorted(sources().items())):
        dest=BUNDLE/'sources'/f'{i:02}_{Path(path).name}';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        if sha256(dest)!=h:raise ValueError('source changed while preparing')
    assets=inventory(BUNDLE)
    payload={k:v for k,v in assets.items() if k.startswith(('programs/','runtime/','firmware/','inputs/','references/','runner/'))}
    payload['plans/board_execution.json']=assets['plans/board_execution.json']
    freeze(BUNDLE/'plans/board_payload.json',dict(files=payload,unpacked_bytes=sum(v['bytes'] for v in payload.values())))
    manifest=dict(network='yolov3_tiny416',experiment=protocol['experiment'],
        runner_sha256=phase['phase_runner_sha256'],old_runner_sha256=old['runner_sha256'],
        runtime_sha256=old['runtime_sha256'],adapter_sha256=phase['adapter_sha256'],
        observer_sha256=old['observer_sha256'],files=inventory(BUNDLE))
    manifest['bundle_id']=digest(manifest)[:24];freeze(BUNDLE/'manifest.json',manifest)
    freeze(OUT/'PLAN.json',dict(bundle_id=manifest['bundle_id'],sources=sources(),protocol=protocol))
    freeze(OUT/'TASKS.json',dict(plan_sha256=sha256(OUT/'PLAN.json'),tasks=[task(s) for s in specifications()]))
    validate();return read(OUT/'PLAN.json')


def validate():
    m=read(BUNDLE/'manifest.json')
    if digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('capsule changed')
    verify_inventory(BUNDLE,m['files'])
    if read(OUT/'PLAN.json')['sources']!=sources():raise ValueError('frozen isolation controller changed')
    p=read(BUNDLE/'plans/preparation.json')
    if sha256(PARENT/'manifest.json')!=p['parent_manifest_sha256'] or sha256(PHASE/'manifest.json')!=p['phase_manifest_sha256']:
        raise ValueError('authenticated asset source changed')
    for path,key in [('runner/vta_stage_pipeline_runner_phase','runner_sha256'),('runner/libwait_adapter.so','adapter_sha256'),
                     ('runner/libdevice_phase.so','observer_sha256')]:
        if sha256(BUNDLE/path)!=m[key]:raise ValueError('actual native binary differs')
    return m


def task(spec):
    if spec not in specifications():raise ValueError('task outside five isolated programs')
    m=read(BUNDLE/'manifest.json');p=read(BUNDLE/'plans/protocol.json')
    cfg=read(BUNDLE/'plans/configurations.json')[spec['case']]
    validate_candidate(cfg,read(BUNDLE/'plans/programs.json'),component=True,unit_count=14)
    enabled=spec['mode']!='clean';cpu=cfg['stages'][0]['device']=='cpu'
    window=p['qualification' if spec['mode']=='qualify' else 'component']
    doc=dict(kind='qualify_component' if spec['mode']=='qualify' else 'calibration',**window,
        configuration=cfg,stages=cfg['stages'],environment=p['environment'],input_indices=[0],
        experiment=p['experiment'],bundle_id=m['bundle_id'],runtime_sha256=m['runtime_sha256'],
        executor_sha256=m['runner_sha256'],adapter_sha256=m['adapter_sha256'] if enabled and cpu else None,
        observer_sha256=m['observer_sha256'] if enabled and not cpu else None,
        spec={**spec,'probe':enabled and cpu,'policy':'original'},source_spec=spec,
        phase_observer=enabled,device_observer=enabled and not cpu,startup_affinity_probe=True)
    doc['task_id']=digest(doc)[:24];return doc


def command(doc,bundle,remote,job):
    if doc!=task(doc['source_spec']):raise ValueError('isolated task changed')
    argv,inputs=base_command(doc,bundle,remote,job)
    argv[argv.index(remote+'/runner/vta_stage_pipeline_runner')]=remote+'/runner/vta_stage_pipeline_runner_phase'
    at=argv.index(remote+'/runtime/ld-linux-aarch64.so.1')
    argv[at:at]=['TVM_BIND_THREADS=1','TVM_EXCLUDE_WORKER0=1']
    argv+=['--allow-cpu-affinity-superset','--resource-affinity-probe',
        '--control-cpu-affinity',','.join(map(str,doc['configuration']['control_mask']))]
    if doc['configuration']['affinity_mode']=='pinned':argv.append('--cpu-worker-pinning')
    if doc['phase_observer']:
        argv+=['--phase-event-json',job+'/phase_trace.jsonl']
        index=next(i for i,s in enumerate(argv) if s.startswith('LD_PRELOAD='))
        library='libdevice_phase.so' if doc['device_observer'] else 'libwait_adapter.so'
        argv[index]='LD_PRELOAD='+remote+'/runner/'+library+':'+remote+'/runtime/libvta.so'
        at=argv.index(remote+'/runtime/ld-linux-aarch64.so.1')
        env=(['VTA_DEVICE_PHASE_TRACE='+job+'/device_trace.jsonl'] if doc['device_observer'] else [
            'VTA_WAIT_POLICY=original','VTA_WAIT_PROBE=1','VTA_WAIT_TRACE='+job+'/wait_trace.jsonl',
            'VTA_WAIT_STATUS='+job+'/wait_status.json'])
        argv[at:at]=env
    if '--process-cpu-affinity' in argv or '--queue-timing' in argv:raise ValueError('unexpected execution change')
    return argv,inputs


def summarize(raw,doc,bundle):
    if doc!=task(doc['source_spec']):raise ValueError('receipt task changed')
    def phases(path,document,affinity):
        adapted=deepcopy(document);adapted['spec']['probe']=document['phase_observer']
        return phase_summary(path,adapted,affinity)
    result=FunctionType(base_summary.__code__,{**base_summary.__globals__,'phase_summary':phases})(raw,doc,bundle)
    raw=Path(raw)
    if doc['device_observer']:
        lines=[json.loads(s) for s in (raw/'device_trace.jsonl').read_text().splitlines()]
        phase=[json.loads(s) for s in (raw/'phase_trace.jsonl').read_text().splitlines()]
        rows=[json.loads(s) for s in (raw/'result.jsonl').read_text().splitlines()]
        result['absolute_device']=absolute_device_events(lines,phase,doc)
        result['device']=validate_events(lines,doc,read(raw/'affinity.json'),rows)
    return result


def qemu():
    validate();qemu=shutil.which('qemu-aarch64-static');results=[]
    for name in CASES[:2]:
        doc=task(dict(case=name,mode='qualify',scope='component'));raw=OUT/'qemu'/name;raw.mkdir(parents=True,exist_ok=True)
        argv,inputs=command(doc,BUNDLE,str(BUNDLE),str(raw));(raw/'inputs.txt').write_text(inputs)
        at=argv.index(str(BUNDLE)+'/runtime/ld-linux-aarch64.so.1')
        args=[qemu,*[x for item in argv[2:at] for x in ('-E',item)],*argv[at:]]
        result=subprocess.run(args,capture_output=True,text=True,timeout=180)
        (raw/'qemu.log').write_text(result.stdout+result.stderr)
        if result.returncode:raise ValueError('ARM output/observer qualification failed: '+str(raw))
        checked=qemu_summary(raw,doc,BUNDLE)
        results.append(dict(program=name,summary=checked,passed=True))
    loader=BUNDLE/'runtime/ld-linux-aarch64.so.1'
    check=subprocess.run([qemu,str(loader),'--library-path',str(BUNDLE/'runtime'),
        str(BUNDLE/'runner/vta_stage_pipeline_runner_phase'),'--help'],capture_output=True,text=True,timeout=30)
    if check.returncode or '--phase-event-json' not in check.stdout+check.stderr:raise ValueError('phase executable ABI changed')
    result=dict(passed=True,bundle_id=read(BUNDLE/'manifest.json')['bundle_id'],cases=results,
        board_access=False,real_CPU_component_calls=2,full_DNN_inferences=0)
    freeze(OUT/'QEMU.json',result);return result


def accept(ssh,doc,remote,boot,expected=None):
    path=RUN/'accepted'/boot/(doc['task_id']+'.json')
    if path.exists():
        value=read(path)
        if value['task']!=doc or value['boot_id']!=boot:raise ValueError('accepted identity changed')
        verify_inventory(value['raw'],value['files'])
        if digest(summarize(value['raw'],doc,BUNDLE))!=digest(value['summary']):raise ValueError('accepted raw cannot reconstruct')
        if expected is not None and value['summary']['output_fingerprints']!=expected:raise ValueError('accepted complete output differs')
        return value
    function=FunctionType(original_accept.__code__,{**original_accept.__globals__,'summary':summarize})
    return function(ssh,doc,BUNDLE,remote,RUN,boot,expected)


def board(*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    validate();plan=read(OUT/'PLAN.json');p=plan['protocol'];q=read(OUT/'QEMU.json')
    if not q['passed'] or q['bundle_id']!=plan['bundle_id']:raise ValueError('matching ARM proof required')
    tasks=read(OUT/'TASKS.json')
    if tasks['plan_sha256']!=sha256(OUT/'PLAN.json') or tasks['tasks']!=[task(s) for s in specifications()]:raise ValueError('task scope changed')
    ssh=transport(read(CONFIG)['board'],command,execute_board=True);boot=ssh.boot_id()
    if boot!=p['expected_boot']:raise ValueError('boot changed; do not mix this refinement batch')
    completed=0;started=time.monotonic();items={}
    with locked(RUN):
        freeze(RUN/'run.json',dict(boot_id=boot,bundle_id=plan['bundle_id'],plan_sha256=sha256(OUT/'PLAN.json')))
        def state(status,**extra):write(RUN/'state.json',dict(status=status,accepted_sessions=completed,
            boot_id=boot,updated_at=now(),**extra),replace=True)
        try:
            remote=ssh.deploy(BUNDLE,RUN);ssh.qualify_environment(remote,boot,RUN,BUNDLE)
            for doc in tasks['tasks']:
                name,mode=doc['source_spec']['case'],doc['source_spec']['mode']
                state('RUNNING',current_task=doc['task_id'],program=name,mode=mode)
                expected=None if mode=='qualify' else items[name,'qualify']['summary']['output_fingerprints']
                items[name,mode]=accept(ssh,doc,remote,boot,expected);completed+=1
                print('Current isolated phase',completed,'/15',name,mode,flush=True)
            ssh.idle_command('cd '+__import__('shlex').quote(remote)+' && sha256sum -c board.sha256 > phase_post_verify.log && sync',boot=boot,timeout=600)
            if ssh.boot_id()!=boot:raise ValueError('boot changed after component measurements')
            state('COMPLETE',native_sessions=15,component_calls=105,CPU_component_calls=42,
                VTA_component_calls=63,new_DNN=0,no_reboot=True,no_reprogram=True,
                elapsed_seconds=time.monotonic()-started)
            sync_tree(RUN)
        except Exception as error:state('STOPPED_ON_ERROR',error=str(error),error_type=type(error).__name__);raise
    return read(RUN/'state.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','validate','qemu','board','status'))
    parser.add_argument('--execute-board',action='store_true');args=parser.parse_args()
    if args.command=='prepare':value=prepare()
    elif args.command=='validate':value=validate()
    elif args.command=='qemu':value=qemu()
    elif args.command=='board':value=board(execute_board=args.execute_board)
    else:value=read(RUN/'state.json') if (RUN/'state.json').exists() else dict(status='NOT_STARTED')
    print(json.dumps(value,ensure_ascii=False,indent=2))
