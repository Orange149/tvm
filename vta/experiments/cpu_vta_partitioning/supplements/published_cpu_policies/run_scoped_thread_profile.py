"""Three isolated CPU signatures: run-scoped clocks, six sessions, 33 calls."""
from copy import deepcopy
from pathlib import Path
from types import FunctionType
import argparse
import json
import os
import random
import shlex
import shutil
import statistics as st
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,digest,inventory,verify_inventory,locked,now,sync_tree
from cpu_vta_partitioning.transport import SSH
from cpu_vta_partitioning.measurement import frames,correctness
from cpu_policies.open_resources.domain import stage,component
from phase_profile.prepare import link_checked
from phase_profile import runtime as phase
from phase_profile.campaign import accept as original_accept
from finite_service.common import freeze
from host_placement_check.prepare import validate_input_paths
from scheduler_gap_audit.curves import csv_write

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/scoped_thread_profile_01'
RUN=ROOT/'runs/scoped_thread_profile_01'
BOOT='6018b3b7-bb29-4f80-92c1-0bfd8c81bb99'
SOURCE=ROOT/'artifacts/scheduler_gap_audit_01/build/sources'
PARENTS={n:ROOT/'artifacts/scheduler_gap_audit_01'/n/'capsule' for n in ('resnet18','yolov3_tiny416')}
CONFIG=ROOT/'artifacts/cross_network_model_01/board.local.json'
CASES=(('yolov3_tiny416','cpu_00_02',4),('yolov3_tiny416','cpu_07_08',3),('resnet18','cpu_00_02',3))


def protocol():
    return dict(experiment='scoped_thread_profile_01',input_index=0,expected_boot_id=BOOT,
        qualification=dict(warmup=0,scored=1,tail=0),calibration=dict(warmup=2,scored=8,tail=0),
        cases=[list(r) for r in CASES],native_sessions=6,CPU_component_calls=33,DNN_inferences=0,
        new_registration_launches=6,new_registration_callbacks=20,
        clocks='pthread_getcpuclockid plus caller-read clock_gettime before/after run; read envelopes retained',
        clock_API_reference='https://man7.org/linux/man-pages/man3/pthread_getcpuclockid.3.html',
        no_wait_adapter=True,no_global_profile=True,no_cost_or_formula_or_Top20_change=True,
        no_reboot_or_FPGA_reprogram=True,no_automatic_remeasurement=True,
        adoption='measurement-scope validity and accounting first; no FPS gain claimed')


def patch(source):
    def one(old,new):
        nonlocal source
        if source.count(old)!=1:raise ValueError('frozen source hook missing or ambiguous')
        source=source.replace(old,new)
    one('#include "phase_probe.h"','#include "phase_probe.h"\n#include "scoped_thread_budget.h"')
    one('  const auto phase_call = vta_workload_phase::Next(stage->name);',
        '  if (stage->device == "cpu") scoped_thread_budget::Ensure(stage->name);\n'
        '  const auto phase_call = vta_workload_phase::Next(stage->name);')
    one('  vta_workload_phase::Span phase_run(stage->name, phase_call, "run");\n'
        '  stage->run();\n  phase_run.End();',
        '  vta_workload_phase::Span phase_run(stage->name, phase_call, "run");\n'
        '  scoped_thread_budget::Scope thread_budget(stage->name, phase_call);\n'
        '  stage->run();\n  thread_budget.End();\n  phase_run.End();')
    one('    vta_workload_phase::Configure(args.phase_event_json);',
        '    vta_workload_phase::Configure(args.phase_event_json);\n'
        '    if (scoped_thread_budget::Enabled() && (!args.serial || args.stages.size() != 1 ||\n'
        '        args.stages.front().device != "cpu" || args.phase_event_json.empty()))\n'
        '      throw std::runtime_error("scoped thread budget requires one serial CPU stage and phase clock");')
    return source


def build():
    target=OUT/'build';native=target/'sources';native.mkdir(parents=True,exist_ok=True)
    if (target/'BUILD.json').exists():
        p=read(target/'BUILD.json')
        if sha256(target/'runner')!=p['runner_sha256'] or sha256(ROOT/'scoped_thread_budget.h')!=p['header_sha256']:
            raise ValueError('existing optional build changed')
        return dict(built=True,reused=True,runner_sha256=p['runner_sha256'])
    old=SOURCE/'vta_stage_pipeline_runner_phase.cc'
    # The parent build record authenticates this exact source, not today's
    # mutable native-deploy checkout.
    parent=read(SOURCE.parent/'BUILD.json')
    if sha256(old)!=parent['source_files'][str(old)]:
        raise ValueError('frozen phase-runner source identity changed')
    for path,expected in parent['source_files'].items():
        if sha256(path)!=expected:raise ValueError('frozen parent build source changed')
    for p in SOURCE.glob('*.h'):shutil.copy2(p,native/p.name)
    clock_header=ROOT/'artifacts/cross_network_model_01/phase_profile/build/sources/phase_probe.h'
    if sha256(clock_header)!='08e325a452552684a0ed78f6de933d87ecae6f0599e0fd3a81c6bac3ffdc3b9e':
        raise ValueError('authenticated clock-only phase header changed')
    shutil.copy2(clock_header,native/'phase_probe.h')
    shutil.copy2(ROOT/'scoped_thread_budget.h',native/'scoped_thread_budget.h')
    patched=native/'vta_stage_pipeline_runner_phase.cc';patched.write_text(patch(old.read_text()))
    recipe_record=read(ROOT/'artifacts/bundles/27b583640470adcc2bbd2194/plans/runtime_build.json')
    tc=recipe_record['toolchain'];recipe=recipe_record['runner_compiler_recipe']
    for p,h in [(tc['sdk_setup'],tc['sdk_setup_sha256'])]+[(v['path'],v['sha256']) for v in tc['tools'].values()]:
        if sha256(p)!=h:raise ValueError('SDK identity changed')
    env=dict(os.environ);env.pop('LD_LIBRARY_PATH',None);binary=target/'runner'
    command=['bash','-c','source "$1" || exit $?; shift; exec "$@"','scoped-sdk',tc['sdk_setup'],
        'bash','-c',recipe,'scoped-compile',str(patched),str(binary),str(ROOT.parents[4]),str(PARENTS['resnet18']/'runtime')]
    started=time.monotonic();r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=180)
    (target/'compile.log').write_text(r.stdout+r.stderr)
    if r.returncode:raise RuntimeError('scoped runner compilation failed; compile log preserved')
    proof=dict(source_sha256=sha256(patched),parent_source_sha256=sha256(old),
        header_sha256=sha256(ROOT/'scoped_thread_budget.h'),runner_sha256=sha256(binary),
        clock_only_phase_header_sha256=sha256(clock_header),
        toolchain=tc,recipe=recipe,no_runtime_or_stage_rebuild=True,seconds=time.monotonic()-started)
    freeze(target/'BUILD.json',proof);return dict(built=True,runner_sha256=proof['runner_sha256'])


def configs(network):
    rows={}
    for net,name,n in CASES:
        if net!=network:continue
        begin,last=map(int,name.split('_')[1:])
        rows[name+'_t'+str(n)]=component(stage('cpu',begin,last+1,n,list(range(n))))
    return rows


def specs():
    cases=list(CASES);random.Random(20261111).shuffle(cases)
    return [dict(network=net,case=name+'_t'+str(n),scope='component',phase=p,probe=False,runner='phase')
        for net,name,n in cases for p in ('qualify','clean')]


def sources():
    paths=[Path(__file__),ROOT/'scoped_thread_budget.h',ROOT/'tests/test_scoped_thread_profile.py',
        ROOT/'phase_profile/runtime.py',ROOT/'phase_profile/campaign.py',
        ROOT/'cpu_policies/open_resources/runtime.py',
        ROOT.parents[1]/'src/cpu_vta_partitioning/transport/__init__.py',
        ROOT.parents[1]/'src/cpu_vta_partitioning/measurement/tasks.py']
    return {str(p):sha256(p) for p in paths}


def prepare():
    if (OUT/'PLAN.json').exists():return validate()
    proof=read(OUT/'build/BUILD.json')
    if proof['runner_sha256']!=sha256(OUT/'build/runner'):raise ValueError('new runner changed')
    freeze(OUT/'PLAN.json',dict(protocol=protocol(),sources=sources(),specs=specs(),
        build_sha256=sha256(OUT/'build/BUILD.json'),parent_manifests={n:sha256(p/'manifest.json') for n,p in PARENTS.items()}))
    for network,parent in PARENTS.items():
        target=OUT/network/'capsule';m=read(parent/'manifest.json');verify_inventory(parent,m['files'])
        if digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('parent manifest changed')
        wanted={cfg['stages'][0]['program'] for cfg in configs(network).values()}
        for rel,info in m['files'].items():
            use=rel.startswith(('runtime/','inputs/','references/')) or rel=='runner/vta_stage_pipeline_runner'
            use=use or rel.startswith('programs/') and rel.split('/')[1] in wanted
            if use:link_checked(parent/rel,target/rel,info['sha256'])
        link_checked(OUT/'build/runner',target/'runner/vta_stage_pipeline_runner_phase',proof['runner_sha256'])
        write(target/'plans/programs.json',{name:v for name,v in read(parent/'plans/programs.json').items() if name in wanted})
        write(target/'plans/inputs.json',read(parent/'plans/inputs.json'))
        write(target/'plans/configurations.json',dict(components=configs(network),pipelines={}))
        write(target/'plans/protocol.json',protocol())
        write(target/'plans/board_execution.json',dict(udmabuf_bytes=201326592,storage_margin_bytes=268435456,boot_policy='existing_boot_CPU_only'))
        assets=inventory(target);payload={k:v for k,v in assets.items() if k.startswith(('runtime/','inputs/','references/','programs/','runner/'))}
        payload['plans/board_execution.json']=assets['plans/board_execution.json']
        write(target/'plans/board_payload.json',dict(files=payload,unpacked_bytes=sum(v['bytes'] for v in payload.values())))
        manifest=dict(network=network,experiment=protocol()['experiment'],phase_runner_sha256=proof['runner_sha256'],
            old_runner_sha256=m['old_runner_sha256'],runtime_sha256=m['runtime_sha256'],adapter_sha256=None,files=inventory(target))
        manifest['bundle_id']=digest(manifest)[:24];write(target/'manifest.json',manifest)
    return validate()


def validate():
    plan=read(OUT/'PLAN.json')
    if plan['sources']!=sources() or plan['protocol']!=protocol() or plan['specs']!=specs():raise ValueError('frozen scoped protocol/controller changed')
    if sha256(OUT/'build/BUILD.json')!=plan['build_sha256']:raise ValueError('build proof changed')
    proof=read(OUT/'build/BUILD.json')
    if sha256(OUT/'build/runner')!=proof['runner_sha256']:raise ValueError('optional program changed')
    native=OUT/'build/sources'
    for path,expected in [(native/'vta_stage_pipeline_runner_phase.cc',proof['source_sha256']),
                          (native/'scoped_thread_budget.h',proof['header_sha256']),
                          (native/'phase_probe.h',proof['clock_only_phase_header_sha256'])]:
        if sha256(path)!=expected:raise ValueError('frozen optional build input changed')
    old=read(SOURCE.parent/'BUILD.json')['source_files']
    for path in SOURCE.glob('*.h'):
        if path.name!='phase_probe.h' and sha256(native/path.name)!=old[str(path)]:
            raise ValueError('copied native support header changed')
    result={}
    for network,parent in PARENTS.items():
        if sha256(parent/'manifest.json')!=plan['parent_manifests'][network]:raise ValueError('parent manifest changed')
        b=OUT/network/'capsule';m=read(b/'manifest.json')
        if digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('new bundle identity differs')
        verify_inventory(b,m['files']);validate_input_paths(b,m)
        result[network]=m['bundle_id']
    return dict(valid=True,bundles=result,native_sessions=6,CPU_component_calls=33,DNN_inferences=0)


def make_task(bundle,spec):
    if spec not in specs():raise ValueError('unplanned scoped CPU task')
    bundle=Path(bundle);m=read(bundle/'manifest.json')
    if spec['network']!=m['network']:raise ValueError('wrong network bundle')
    cfg=read(bundle/'plans/configurations.json')['components'][spec['case']]
    window=protocol()['qualification' if spec['phase']=='qualify' else 'calibration']
    doc=dict(kind='qualify_component' if spec['phase']=='qualify' else 'calibration',configuration=deepcopy(cfg),
        stages=deepcopy(cfg['stages']),spec=deepcopy(spec),**window,
        environment=dict(cores=4,frequency_khz=1066666,poll_sleep_ns=1000,post_start_sleep_ns=1000,
            spin_count=0,exclude_worker0=1,bind_threads=1),input_indices=[0],
        executor_sha256=m['phase_runner_sha256'],runtime_sha256=m['runtime_sha256'],adapter_sha256=None,
        bundle_id=m['bundle_id'],experiment=protocol()['experiment'],startup_affinity_probe=True,
        run_scoped_thread_clocks=True,clock_registration_outside_phase=True,CPU_only=True)
    doc['task_id']=digest(doc)[:24];return doc


def native_command(doc,bundle,remote,job):
    f=phase.native_command
    argv,inputs=FunctionType(f.__code__,{**f.__globals__,'make_task':make_task})(doc,bundle,remote,job)
    at=argv.index(remote+'/runtime/ld-linux-aarch64.so.1')
    argv[at:at]=['CPU_RUN_BUDGET_JSON='+job+'/thread_budget.json']
    argv+=['--phase-event-json',job+'/phase_trace.jsonl']
    if any('libwait_adapter' in a for a in argv):raise ValueError('detailed callback observer forbidden')
    return argv,inputs


def phase_only(raw,doc,affinity):
    # Reuse the strict phase parser independently of its old wait-adapter flag.
    return phase.phase_summary(raw,{**doc,'spec':{**doc['spec'],'probe':True}},affinity)


def budget_summary(data,doc,affinity):
    cfg=doc['stages'][0];n=cfg['threads'];total=sum(doc[k] for k in ('warmup','scored','tail'))
    if (data.get('schema')!='scoped_thread_budget_v1' or data['stage']!=cfg['program'] or
        data['registration_launches']!=1 or data['registration_callbacks']!=n or
        data['registration_outside_stage_phases'] is not True or
        not 0<data['registration_begin_ns']<=data['registration_end_ns']):raise ValueError('scoped registration differs')
    slots=data['slots'];expected=affinity['stages'][0]['worker_slots']
    if len(slots)!=n or len({s['clock_id'] for s in slots})!=n or len({s['tid'] for s in slots})!=n:
        raise ValueError('thread clock IDs or TIDs are not unique')
    for i,s in enumerate(slots):
        e=next(w for w in expected if w['slot']==i)
        if type(s['clock_id']) is not int or s['slot']!=i or s['tid']!=e['tid'] or s['cpus']!=[cfg['worker_core_map'][i]]:
            raise ValueError('clock slot/TID/actual affinity differs')
    if slots[0]['tid']!=affinity['stages'][0]['caller_tid']:raise ValueError('clock slot zero is not caller')
    records=data['records']
    if [r['call'] for r in records]!=list(range(total)):raise ValueError('missing/duplicate/out-of-order budget frame')
    previous=[0]*n;rows=[]
    for r in records:
        if r['stage']!=cfg['program'] or not 0<=r['graph_begin_ns']<=r['graph_end_ns'] or len(r['threads'])!=n:
            raise ValueError('invalid graph clock window')
        process=r['process_CPU_end_ns']-r['process_CPU_begin_ns']
        if process<0:raise ValueError('process CPU clock reversed')
        values=[];lower=[];widths=[]
        for i,s in enumerate(r['threads']):
            if s['slot']!=i or s['tid']!=slots[i]['tid']:raise ValueError('budget thread identity changed')
            order=[s['before_read_begin_ns'],s['before_read_end_ns'],r['graph_begin_ns'],r['graph_end_ns'],s['after_read_begin_ns'],s['after_read_end_ns']]
            if any(type(x) is not int or x<0 for x in order) or order!=sorted(order):raise ValueError('thread clock envelope reversed')
            if not previous[i]<=s['CPU_begin_ns']<=s['CPU_end_ns']:raise ValueError('thread CPU counter reversed')
            value=s['CPU_end_ns']-s['CPU_begin_ns'];previous[i]=s['CPU_end_ns']
            widening=r['graph_begin_ns']-s['before_read_begin_ns']+s['after_read_end_ns']-r['graph_end_ns']
            if value>s['after_read_end_ns']-s['before_read_begin_ns']:raise ValueError('single thread exceeded wall capacity')
            values.append(value);lower.append(max(0,value-widening));widths.append(widening)
        rows.append(dict(call=r['call'],thread_CPU_ns=values,thread_CPU_lower_ns=lower,
            clock_window_widening_ns=widths,process_run_CPU_ns=process,
            process_minus_thread_sum_ns=process-sum(values),
            process_inside_thread_read_envelope=sum(lower)<=process<=sum(values),
            graph_wall_ns=r['graph_end_ns']-r['graph_begin_ns']))
    kept=rows[doc['warmup']:doc['warmup']+doc['scored']]
    means=[st.mean(r['thread_CPU_ns'][i] for r in kept) for i in range(n)]
    if sum(means)<=0:raise ValueError('no measured CPU work')
    return dict(registration_launches=1,registration_callbacks=n,records=total,
        mean_thread_run_CPU_ms=[x/1e6 for x in means],run_thread_weights=[x/sum(means) for x in means],
        mean_process_run_CPU_ms=st.mean(r['process_run_CPU_ns'] for r in kept)/1e6,
        mean_process_minus_thread_sum_ms=st.mean(r['process_minus_thread_sum_ns'] for r in kept)/1e6,
        max_thread_read_window_widening_ms=max(v for r in kept for v in r['clock_window_widening_ns'])/1e6,
        all_process_budgets_inside_thread_envelope=all(r['process_inside_thread_read_envelope'] for r in kept),
        rows=rows,scope='run only; per-thread CPU clocks widened by declared before/after read envelopes')


base_summary=FunctionType(phase.summary.__code__,{**phase.summary.__globals__,'phase_summary':phase_only})
def match_phase_windows(raw,data):
    phases=[json.loads(s) for s in (Path(raw)/'phase_trace.jsonl').read_text().splitlines()]
    first_set=next(p for p in phases if p['operation']=='set' and p['call']==0)
    if data['registration_end_ns']>first_set['wall_start_ns']:
        raise ValueError('clock registration overlaps a stage phase')
    for r in data['records']:
        phase=next(p for p in phases if p['operation']=='run' and p['call']==r['call'])
        start=min(s['before_read_begin_ns'] for s in r['threads'])
        end=max(s['after_read_end_ns'] for s in r['threads'])
        if not phase['wall_start_ns']<=start<=r['graph_begin_ns']<=r['graph_end_ns']<=end<=phase['wall_end_ns']:
            raise ValueError('thread CPU snapshot escapes its run phase')
    return True


def summary(raw,doc,bundle):
    value=base_summary(raw,doc,bundle)
    data=read(Path(raw)/'thread_budget.json');match_phase_windows(raw,data)
    value['run_thread_budget']=budget_summary(data,doc,read(Path(raw)/'affinity.json'))
    return value


accept=FunctionType(original_accept.__code__,{**original_accept.__globals__,'summary':summary},argdefs=original_accept.__defaults__)


def qemu():
    validate();results=[]
    for spec in [s for s in specs() if s['phase']=='qualify']:
        bundle=OUT/spec['network']/'capsule';raw=OUT/'qemu'/(spec['network']+'_'+spec['case']);raw.mkdir(parents=True,exist_ok=True)
        doc=make_task(bundle,spec);argv,inputs=native_command(doc,bundle,str(bundle),str(raw));(raw/'inputs.txt').write_text(inputs)
        at=argv.index(str(bundle)+'/runtime/ld-linux-aarch64.so.1')
        command=['qemu-aarch64-static',*[a for v in argv[2:at] for a in ('-E',v)],*argv[at:]]
        r=subprocess.run(command,capture_output=True,text=True,timeout=120)
        (raw/'native.log').write_text(r.stdout+r.stderr)
        if r.returncode:raise RuntimeError('QEMU CPU/clock execution failed: '+str(raw/'native.log'))
        rows=frames(raw/'result.jsonl',1);correctness(rows,doc,bundle,raw)
        b=budget_summary(read(raw/'thread_budget.json'),doc,read(raw/'affinity.json'))
        match_phase_windows(raw,read(raw/'thread_budget.json'))
        phase_only(raw,doc,read(raw/'affinity.json'))
        results.append(dict(network=spec['network'],case=spec['case'],correct=True,clock_records=b['records']))
    freeze(OUT/'QEMU.json',dict(passed=True,results=results,scope='real ARM CPU modules, outputs and clock API; not board performance'))
    return dict(passed=True,cases=len(results))


def board(*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board required')
    valid=validate()
    if not read(OUT/'QEMU.json')['passed']:raise ValueError('QEMU required')
    measure=FunctionType(SSH.measure.__code__,{**SSH.measure.__globals__,'native_command':native_command})
    ssh=type('ScopedThreadClockSSH',(SSH,),{'measure':measure})(read(CONFIG)['board'],execute_board=True)
    if ssh.boot_id()!=BOOT:raise ValueError('boot changed; no mixing or automatic reboot')
    started=time.monotonic()
    with locked(RUN):
        freeze(RUN/'run.json',dict(boot_id=BOOT,plan_sha256=sha256(OUT/'PLAN.json'),bundle_ids=valid['bundles']))
        def state(status,**fields):write(RUN/'state.json',dict(status=status,boot_id=BOOT,updated_at=now(),**fields),replace=True)
        try:
            remotes={net:ssh.deploy(OUT/net/'capsule',RUN/net) for net in PARENTS}
            expected={};calls=0
            for index,spec in enumerate(specs(),1):
                net=spec['network'];bundle=OUT/net/'capsule';doc=make_task(bundle,spec);key=(net,spec['case'])
                if spec['phase']=='clean' and key not in expected:raise ValueError('component must be qualified first')
                state('RUNNING',accepted_sessions=index-1,total_sessions=6,current_task=doc['task_id'])
                item=accept(ssh,doc,bundle,remotes[net],RUN,BOOT,expected.get(key))
                if spec['phase']=='qualify':expected[key]=item['summary']['output_fingerprints']
                calls+=sum(doc[k] for k in ('warmup','scored','tail'))
                print('run区间逐线程核时',index,'/6',net,spec['case'],spec['phase'],flush=True)
            if calls!=33:raise ValueError('bounded component calls differ')
            for remote in remotes.values():ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > scoped_post_verify.log && sync',boot=BOOT,timeout=300)
            if ssh.boot_id()!=BOOT:raise ValueError('boot changed after collection')
            state('COMPLETE',accepted_sessions=6,total_sessions=6,CPU_component_calls=33,DNN_inferences=0,
                no_FPGA_access=True,no_reboot=True,seconds=time.monotonic()-started)
            sync_tree(RUN)
        except Exception as error:
            state('STOPPED_ON_ERROR',error_type=type(error).__name__,error=str(error));raise
    return read(RUN/'state.json')


def analyze():
    validate();state=read(RUN/'state.json')
    if state['status']!='COMPLETE' or state['boot_id']!=BOOT:raise ValueError('bounded scoped cohort incomplete')
    rows=[];source_receipts={}
    for spec in specs():
        bundle=OUT/spec['network']/'capsule';doc=make_task(bundle,spec)
        p=RUN/'accepted'/BOOT/(doc['task_id']+'.json');r=read(p)
        verify_inventory(r['raw'],r['files'])
        if r['task']!=doc or summary(r['raw'],doc,bundle)!=r['summary']:raise ValueError('accepted budget no longer replays')
        source_receipts[str(p)]=sha256(p)
        if spec['phase']!='clean':continue
        value=r['summary'];budget=value['run_thread_budget'];n=doc['stages'][0]['threads']
        ledger={a['logical_worker']:a['cpu_time_delta_ms'] for a in value['thread_ledger'] if a['role'] in ('caller','worker')}
        whole=[ledger[i] for i in range(n)];whole=[v/sum(whole) for v in whole]
        rows.append(dict(network=spec['network'],program=doc['stages'][0]['program'],threads=n,
            run_thread_CPU_ms=budget['mean_thread_run_CPU_ms'],run_thread_weights=budget['run_thread_weights'],
            whole_session_weights=whole,caller_whole_minus_run_percentage_points=100*(whole[0]-budget['run_thread_weights'][0]),
            process_minus_thread_sum_ms=budget['mean_process_minus_thread_sum_ms'],
            max_read_widening_ms=budget['max_thread_read_window_widening_ms'],
            all_process_budgets_inside_thread_envelope=budget['all_process_budgets_inside_thread_envelope'],
            isolated_cost=value['cost'],task_id=doc['task_id']))
    result=dict(rows=rows,source_receipts=source_receipts,native_sessions=6,CPU_component_calls=33,DNN_inferences=0,
        independent_initialization_CPU_not_claimed_from_difference=True,
        no_new_static_cost_or_formula_or_Top20=True,complete_goal_not_proven=True)
    write(OUT/'RESULTS.json',result,replace=True);csv_write(OUT/'budgets.csv',rows)
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('build','prepare','validate','qemu','board','status','analyze'))
    p.add_argument('--execute-board',action='store_true');a=p.parse_args()
    value=board(execute_board=a.execute_board) if a.command=='board' else read(RUN/'state.json') if a.command=='status' else globals()[a.command]()
    if a.command!='analyze':print(json.dumps(value,ensure_ascii=False,indent=2))
