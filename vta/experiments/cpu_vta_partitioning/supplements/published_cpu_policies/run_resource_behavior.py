#!/usr/bin/env python3
"""Independent resource-mechanism campaign; unchanged frozen native runtime.

Matched placement/gate interventions are diagnostic references. They are not
automatically inserted into a production Top-20. Every plan and prediction is
sealed before its qualification/performance sessions. Old sources stay intact.
"""
from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import argparse
import csv
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

from cpu_policies.core_model.common import api as legacy_api,PROJECT

ROOT=Path(__file__).resolve().parent
_CAPSULE=ROOT.parent.parent
PARENT=ROOT/'artifacts/bundles/8a942b2047c031339168f9d2'
SELECTION=ROOT/'artifacts/bundles/27b583640470adcc2bbd2194'
SOURCE=ROOT/'artifacts/frame_stream_top20_static_validation'
RUN_ID='resnet18_resource_behavior_mechanism_01'
PROTOCOL={'experiment':'resnet18_resource_behavior_mechanism','revision':1,
    'input_index':0,'queue_depth':2,'seed':20261109,'performance':{'warmup':8,'scored':16,'tail':4},
    'rounds':2,'ordering':'seeded first round and exact reverse second round',
    'anchors':False,'timing_rejection_threshold':None,'automatic_remeasurement':False,
    'boot_policy':'one_boot_no_automatic_reboot','component_measurements':0,
    'board_safety':{'reboot':False,'partition_changes':False,'boot_file_changes':False,
        'frequency_changes':False,'exclusive_session_lock':True,'unchanged_qualified_firmware_driver':True},
    'environment':{'cores':4,'frequency_khz':1066666,'poll_sleep_ns':1000,
        'post_start_sleep_ns':1000,'spin_count':0,'exclude_worker0':1,'bind_threads':1}}

def api(bundle=None):
    return legacy_api(bundle or (_CAPSULE if (_CAPSULE/'manifest.json').is_file() else PARENT))

def identity():
    paths=[Path(__file__),ROOT/'tests/test_resource_behavior.py']
    paths+=sorted(p for p in (ROOT/'cpu_policies').rglob('*') if p.suffix in ('.py','.cc','.h'))
    paths+=sorted((ROOT/'configs').rglob('*.json'))
    return sha256(json.dumps({str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest()
        for p in paths},sort_keys=True,separators=(',',':')).encode()).hexdigest()

def configurations(parent=PARENT,selection=SELECTION):
    from cpu_policies.open_resources.domain import candidate,validate_candidate
    c=api(parent);programs=c.read(Path(parent)/'plans/programs.json')
    # Only frozen, premeasurement selection documents are read here. The
    # current coverage winner is explicitly a known diagnostic reference.
    rankings=c.read(Path(selection)/'plans/selection_rankings.json')
    pool=c.read(Path(selection)/'plans/static_pool.json')
    best_id='cfg_29180a74ebff881ccda90ddd'
    old=rankings['pure_static20'][0];best=pool[best_id]
    historical=next(r for r in c.read(Path(parent)/'plans/historical_references.json')
                    if r['id']=='cfg_85c3cc9bfdcdcbae714e259d')
    result={'static_first':candidate(old['stages'],old['context_id'],old['execution_mode']),
            'coverage_reference':candidate(best['stages'],best['context_id'],best['execution_mode'])}
    for host in ('shared4','host_shared_2'):
        for gate in ('off','resource_group'):
            rows=deepcopy(historical['stages'])
            for r in rows:
                if r['device']=='vta':r['mask']=[0,1,2,3] if host=='shared4' else [2]
            name=('shared' if host=='shared4' else 'host2')+('_serial' if gate=='resource_group' else '_open')
            result[name]=candidate(rows,host,gate)
    for cfg in result.values():validate_candidate(cfg,programs)
    if len(result)!=6 or len({r['id'] for r in result.values()})!=6:raise ValueError('six distinct diagnostic cases required')
    return result

def _make_task(bundle,case,kind,round_index):
    from cpu_policies.open_resources.runtime import make_task
    c=api(bundle);cfg=c.read(Path(bundle)/'plans/configurations.json')[case]
    qualify=kind=='qualify_pipeline'
    if kind not in ('qualify_pipeline','performance') or round_index not in ((-1,) if qualify else (0,1)):
        raise ValueError('unsupported session kind or round')
    doc=make_task(kind,cfg,bundle,warmup=0 if qualify else 8,scored=1 if qualify else 16,
        tail=0 if qualify else 4,indices=[0],campaign=PROTOCOL['experiment'],
        candidate_id=cfg['id'],behavior_case=case,behavior_round=round_index)
    if doc['environment']!=PROTOCOL['environment']:raise ValueError('execution environment differs')
    return doc

def tasks(bundle):
    cases=sorted(api(bundle).read(Path(bundle)/'plans/configurations.json'))
    if len(cases)!=6:raise ValueError('matched experiment requires six cases')
    random.Random(PROTOCOL['seed']).shuffle(cases)
    result=[_make_task(bundle,name,'qualify_pipeline',-1) for name in cases]
    for block,order in enumerate((cases,list(reversed(cases)))):
        result.extend(_make_task(bundle,name,'performance',block) for name in order)
    return result

def validate_document(doc,bundle):
    if _make_task(bundle,doc['behavior_case'],doc['kind'],doc['behavior_round'])!=doc:
        raise ValueError('task identity, case or round differs')

def summary(raw,doc,bundle):
    from cpu_policies.open_resources.runtime import validate_summary
    validate_document(doc,bundle);result=validate_summary(raw,doc,bundle)
    if doc['kind']=='performance':
        rows=[json.loads(line) for line in (Path(raw)/'result.jsonl').read_text().splitlines() if line]
        if any(r.get('queue_timing_enabled') is not False or r.get('defer_result_write_enabled') is not True for r in rows):
            raise ValueError('performance logging policy differs')
        result['fps']=1000/result['ii_ms']
    return result

@lru_cache(maxsize=4)
def _ssh_class(bundle):
    api(bundle)
    from cpu_strategy.policy.transport import SSH as Base
    from cpu_policies.open_resources.runtime import _measure
    class BehaviorSSH(Base):
        def measure(self,doc,bundle,remote,root,boot):
            validate_document(doc,bundle)
            return _measure(self,doc,bundle,remote,root,boot)
    return BehaviorSSH

def ssh(config,bundle):
    obj=_ssh_class(str(bundle))(config,execute_board=True)
    obj.options+=['-o','ControlMaster=auto','-o','ControlPersist=60','-o','ControlPath=/tmp/cpu-vta-behavior-%C']
    return obj

def validate_bundle(bundle):
    bundle=Path(bundle).resolve();c=api(bundle);m=c.read(bundle/'manifest.json')
    if c.digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('manifest differs')
    if m['controller_identity']!=identity():raise ValueError('controller changed; use frozen behavior_controller')
    c.verify_inventory(bundle,m['files'])
    if c.read(bundle/'plans/protocol.json')!=PROTOCOL:raise ValueError('protocol differs')
    cfg=c.read(bundle/'plans/configurations.json');programs=c.read(bundle/'plans/programs.json')
    from cpu_policies.open_resources.domain import validate_candidate
    for value in cfg.values():validate_candidate(value,programs)
    if set(programs)!={s['program'] for r in cfg.values() for s in r['stages']}:raise ValueError('program closure differs')
    from cpu_vta_partitioning.prepare.payload import execution_files
    if execution_files(bundle)!=c.read(bundle/'plans/board_payload.json')['files']:raise ValueError('payload differs')
    plan=tasks(bundle)
    if len(plan)!=18 or sum(sum(d[k] for k in ('warmup','scored','tail')) for d in plan)!=342:
        raise ValueError('measurement budget differs')
    return m

def prepare():
    from cpu_policies.diagnosis.prepare import _copy
    from cpu_policies.open_resources.costs import Costs,score_configuration
    c=api();code=identity();started=time.monotonic()
    for parent in (PARENT,SELECTION):
        m=c.read(parent/'manifest.json')
        if c.digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:raise ValueError('source manifest differs')
        c.verify_inventory(parent,m['files'])
    cfg=configurations();all_programs=c.read(PARENT/'plans/programs.json')
    records=c.read(SOURCE/'isolated_costs.json')['costs'];ddr=c.read(SOURCE/'ddr.json')
    source_manifest=c.read(SELECTION/'manifest.json')
    for name in ('isolated_costs.json','ddr.json'):
        if c.sha256(SOURCE/name)!=source_manifest['files']['plans/'+name]['sha256']:raise ValueError('cost source differs')
    costs=Costs(records,programs=all_programs,ddr=ddr)
    predictions={name:score_configuration(value,costs) for name,value in cfg.items()}
    folder=ROOT/'artifacts/resource_behavior_builds';folder.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='prepare-',dir=folder))
    for name in ('runtime','firmware','inputs','references'):
        for path,info in c.inventory(PARENT/name).items():_copy(PARENT/name/path,work/name/path,info['sha256'])
    for name in ('base_controller','cpu_strategy','native_source'):
        shutil.copytree(PARENT/'sources'/name,work/'sources'/name,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    for path in (SELECTION/'sources/reference').rglob('*'):
        if path.is_file():
            rel=str(path.relative_to(SELECTION));_copy(path,work/rel,source_manifest['files'][rel]['sha256'])
    programs={name:all_programs[name] for name in sorted({s['program'] for r in cfg.values() for s in r['stages']})}
    for name,p in programs.items():
        for leaf,info in p['files'].items():_copy(PARENT/'programs'/name/leaf,work/'programs'/name/leaf,info['sha256'])
    for name in ('inputs.json','runtime_build.json','compiler_identity.json','sdk_runtime.json'):
        _copy(PARENT/'plans'/name,work/'plans'/name)
    for name in ('isolated_costs.json','ddr.json'):_copy(SOURCE/name,work/'plans'/name)
    policy=c.read(PARENT/'plans/executors.json')['policy'];_copy(PARENT/policy['path'],work/policy['path'],policy['sha256'])
    for name,value in [('configurations.json',cfg),('programs.json',programs),('protocol.json',PROTOCOL),
        ('predictions.json',predictions),('executors.json',{'policy':policy}),
        ('board_execution.json',{'revision':'resource-behavior-1','udmabuf_bytes':201326592,
          'storage_margin_bytes':256*1024**2,'boot_policy':PROTOCOL['boot_policy']})]:c.write(work/'plans'/name,value)
    c.write(work/'plans/source_provenance.json',{'source_manifest_sha256':c.sha256(PARENT/'manifest.json'),
        'selection_manifest_sha256':c.sha256(SELECTION/'manifest.json'),'costs_sha256':c.sha256(SOURCE/'isolated_costs.json'),
        'ddr_sha256':c.sha256(SOURCE/'ddr.json'),'known_references_for_diagnosis_only':True,
        'pipeline_FPS_used_for_predictions':False,'stage_modules_recompiled':0,'runtime_rebuilt':False})
    tests=subprocess.run([sys.executable,'-m','pytest','-c','pyproject.toml','-q','tests/test_resource_behavior.py'],cwd=ROOT,
        env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'),capture_output=True,text=True,timeout=180)
    c.write(work/'plans/offline_tests.json',{'passed':tests.returncode==0,'stdout':tests.stdout,'stderr':tests.stderr})
    if tests.returncode:raise RuntimeError(tests.stdout+tests.stderr)
    emulator=shutil.which('qemu-aarch64-static')
    if not emulator:raise RuntimeError('QEMU missing')
    result=subprocess.run([emulator,str(work/'runtime/ld-linux-aarch64.so.1'),'--library-path',str(work/'runtime'),
        str(work/policy['path']),'--help'],capture_output=True,text=True,timeout=45)
    if result.returncode or '--cpu-worker-pinning' not in result.stdout+result.stderr:raise ValueError('QEMU loader failed')
    if identity()!=code:raise ValueError('controller changed during preparation')
    snapshot=work/'sources/behavior_controller'
    for name in (Path(__file__).name,'pyproject.toml'):_copy(ROOT/name,snapshot/name)
    _copy(ROOT/'tests/test_resource_behavior.py',snapshot/'tests/test_resource_behavior.py')
    for name in ('cpu_policies','configs'):
        shutil.copytree(ROOT/name,snapshot/name,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    from cpu_vta_partitioning.prepare.payload import execution_files
    payload=execution_files(work);c.write(work/'plans/board_payload.json',{'files':payload,
        'unpacked_bytes':sum(v['bytes'] for v in payload.values())})
    c.write(work/'plans/preparation_cost.json',{'seconds':time.monotonic()-started,
        'execution_bytes':sum(v['bytes'] for v in payload.values()),'new_component_profiles':0,'compiled_runners':0})
    c.write(work/'OFFLINE_READINESS.json',{'status':'PREPARED_OFFLINE','cases':6,'tasks':18,'inferences':342,
        'tests_passed':True,'runtime_and_pipeline_executor_unchanged':True})
    m={'experiment':PROTOCOL['experiment'],'controller_identity':code,'files':c.inventory(work)}
    m['bundle_id']=c.digest(m)[:24];c.write(work/'manifest.json',m)
    target=ROOT/'artifacts/bundles'/m['bundle_id'];work.rename(target);validate_bundle(target)
    return target

def _save(root,state):api(state['bundle']).write(Path(root)/'state.json',state,replace=True)

def _pin(root,state,*names):
    c=api(state['bundle'])
    for name in names:
        p=c.relative_file(root,name);info={'sha256':c.sha256(p),'bytes':p.stat().st_size}
        if name in state['frozen_files'] and state['frozen_files'][name]!=info:raise ValueError('frozen artifact changed')
        state['frozen_files'][name]=info
    _save(root,state)

def create(bundle,root,connection):
    root,bundle=Path(root).resolve(),Path(bundle).resolve();m=validate_bundle(bundle);c=api(bundle)
    if root.exists():raise FileExistsError('run exists; use resume')
    root.mkdir(parents=True);state={'bundle':str(bundle),'bundle_id':m['bundle_id'],
        'manifest_sha256':c.sha256(bundle/'manifest.json'),'controller_identity':identity(),'connection':deepcopy(connection),
        'status':'CREATED','boot_id':None,'active_boot':None,'completed_tasks':0,'expected_tasks':18,'frozen_files':{},'created':c.now()}
    c.write(root/'tasks.json',tasks(bundle));c.write(root/'prediction_freeze.json',{
        'predictions_sha256':c.sha256(bundle/'plans/predictions.json'),'configurations_sha256':c.sha256(bundle/'plans/configurations.json'),
        'frozen_at':c.now(),'pipeline_FPS_used':False})
    _pin(root,state,'tasks.json','prediction_freeze.json');return state

def validate_state(root):
    root=Path(root);c=api();state=c.read(root/'state.json');bundle=Path(state['bundle'])
    if state['controller_identity']!=identity():raise ValueError('controller changed; use frozen behavior_controller')
    if c.sha256(bundle/'manifest.json')!=state['manifest_sha256']:raise ValueError('bundle manifest changed')
    c.verify_inventory(root,state['frozen_files'])
    if c.read(root/'tasks.json')!=tasks(bundle):raise ValueError('task plan changed')
    freeze=c.read(root/'prediction_freeze.json')
    if (freeze['predictions_sha256']!=c.sha256(bundle/'plans/predictions.json')
            or freeze['configurations_sha256']!=c.sha256(bundle/'plans/configurations.json')):
        raise ValueError('prediction freeze differs')
    if state.get('active_boot') not in (None,state['boot_id']):raise ValueError('active boot differs')
    if state['status']=='COMPLETE' and (state['completed_tasks']!=18 or state['active_boot'] is not None):
        raise ValueError('incomplete run marked complete')
    return state

def accepted(root,state,doc,transport=None,remote=None):
    c=api(state['bundle']);root=Path(root).resolve();boot=state['boot_id'];validate_document(doc,state['bundle'])
    path=root/'accepted'/boot/(doc['task_id']+'.json')
    if path.exists():
        rec=c.read(path)
        if rec['task']!=doc or rec['boot_id']!=boot:raise ValueError('receipt identity differs')
        c.verify_inventory(root,rec['files']);raw=c.relative_file(root,rec['raw'])
        if not raw.is_relative_to(root/'raw'/boot/doc['task_id']):raise ValueError('raw receipt escaped task')
        value=summary(raw,doc,state['bundle'])
        if value!=rec['summary']:raise ValueError('raw-backed summary differs')
    else:
        if transport is None:raise ValueError('missing accepted session')
        print('[资源机制] '+doc['behavior_case']+' '+doc['kind']+' round='+str(doc['behavior_round']),flush=True)
        started=time.monotonic();raw=Path(transport.measure(doc,state['bundle'],remote,root,boot)).resolve()
        if not raw.is_relative_to(root/'raw'/boot/doc['task_id']):raise ValueError('transport raw escaped task')
        value=summary(raw,doc,state['bundle']);c.sync_tree(raw)
        rec={'task':doc,'boot_id':boot,'raw':str(raw.relative_to(root)),'summary':value,
            'measurement_collection_seconds':time.monotonic()-started,
            'files':{str(raw.relative_to(root)/n):v for n,v in c.inventory(raw).items()}}
        c.write(path,rec)
    _pin(root,state,str(path.relative_to(root)))
    if transport is not None:transport.acknowledge(doc,root,boot)
    return value

def analyze(root):
    from cpu_policies.open_resources.costs import Costs
    root=Path(root);state=validate_state(root);bundle=Path(state['bundle']);c=api(bundle)
    cfg=c.read(bundle/'plans/configurations.json');pred=c.read(bundle/'plans/predictions.json')
    costs=Costs(c.read(bundle/'plans/isolated_costs.json')['costs'],programs=c.read(bundle/'plans/programs.json'),ddr=c.read(bundle/'plans/ddr.json'))
    windows={};stage_rows=[];raw_files=0;raw_bytes=0;elapsed=0
    for doc in tasks(bundle):
        value=accepted(root,state,doc);receipt=c.read(root/'accepted'/state['boot_id']/(doc['task_id']+'.json'))
        raw_files+=len(receipt['files']);raw_bytes+=sum(v['bytes'] for v in receipt['files'].values())
        elapsed+=receipt['measurement_collection_seconds']
        if doc['kind']!='performance':continue
        case=doc['behavior_case'];block=doc['behavior_round'];windows[case,block]=value
        rows=[json.loads(line) for line in (root/receipt['raw']/'result.jsonl').read_text().splitlines() if line][8:24]
        for i,s in enumerate(cfg[case]['stages']):
            iso=costs.get(s,cfg[case]);service=statistics.mean(r[f'stage{i}_ms'] for r in rows)
            stage_rows.append({'case':case,'round':block,'program':s['program'],'device':s['device'],
                'threads':s['threads'],'mask':str(s['mask']),'isolated_service_ms':iso['service_ms'],
                'pipeline_service_ms':service,'service_ratio':service/iso['service_ms'],
                'pipeline_run_ms':statistics.mean(r[f'stage{i}_run_ms'] for r in rows)})
    cases=[]
    for name in sorted(cfg):
        ii=statistics.mean(windows[name,b]['ii_ms'] for b in (0,1))
        cases.append({'case':name,'id':cfg[name]['id'],'predicted_ii_ms':pred[name]['score_ms'],
            'round0_fps':windows[name,0]['fps'],'round1_fps':windows[name,1]['fps'],
            'mean_ii_ms':ii,'mean_fps':1000/ii,'prediction_error_percent':(pred[name]['score_ms']/ii-1)*100})
    effects=[]
    for a,b,label in [('host2_serial','shared_serial','固定主机效果（串行）'),
        ('host2_open','shared_open','固定主机效果（开放）'),('shared_serial','shared_open','串行效果（共享主机）'),
        ('host2_serial','host2_open','串行效果（固定主机）')]:
        per_round=[(windows[b,i]['ii_ms']/windows[a,i]['ii_ms']-1)*100 for i in (0,1)]
        effects.append({'comparison':label,'treatment':a,'reference':b,'round0_FPS_gain_percent':per_round[0],
            'round1_FPS_gain_percent':per_round[1],'direction_consistent':all(x>0 for x in per_round) or all(x<0 for x in per_round)})
    output=root/'analysis';output.mkdir(exist_ok=True)
    def csv_write(name,rows):
        with (output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    csv_write('cases.csv',cases);csv_write('matched_effects.csv',effects);csv_write('stage_services.csv',stage_rows)
    result={'cases':cases,'effects':effects,'boot_id':state['boot_id'],'sessions':18,'full_inferences':342,
        'raw_files':raw_files,'raw_bytes':raw_bytes,'measurement_collection_seconds':elapsed,
        'program_runtime_executor_match_profiles':True,'same_size_core_mask_alias_used':True,
        'new_components':0,'scope':'single-boot two reverse-order short rounds; diagnostic references, not Top-20 selection'}
    c.write(output/'analysis.json',result,replace=True)
    lines=['# 资源机制六项匹配对照','',
        '原模块、原运行库和原执行器；输入0；六项资格检查及两轮正反顺序短窗口，共18会话、342次完整推理。','',
        '|配置|预测ms|第一轮FPS|第二轮FPS|平均FPS|','|---|---:|---:|---:|---:|']
    labels={'static_first':'原静态第一名','coverage_reference':'上轮覆盖名单最佳参照','shared_open':'同一历史计算分配／共享主机／开放',
        'shared_serial':'同一历史计算分配／共享主机／同组串行','host2_open':'同一历史计算分配／主机固定2／开放',
        'host2_serial':'同一历史计算分配／主机固定2／同组串行'}
    for r in cases:lines.append('|'+labels[r['case']]+'|'+'|'.join(f"{r[k]:.3f}" for k in ('predicted_ii_ms','round0_fps','round1_fps','mean_fps'))+'|')
    lines+=['','平均FPS由同启动两轮平均完成间隔换算；不是把帧当独立样本。历史方案仅作机制参照，不进入生产Top-20优待。',
        '没有锚点、漂移拒收、自动延长或追加；所有原始输出、绑核和哈希校验保留。',
        '[阶段服务](stage_services.csv) · [匹配干预](matched_effects.csv) · [逐配置](cases.csv)']
    (output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n');return result

def run_current(root,transport):
    root=Path(root).resolve();c=api()
    from cpu_vta_partitioning.transport import TransportError
    with c.locked(root):
        state=validate_state(root);validate_bundle(state['bundle'])
        if state['status']=='COMPLETE':
            for doc in tasks(state['bundle']):accepted(root,state,doc)
            return state
        if state['status'].startswith('STOPPED'):raise RuntimeError('invalid evidence retained; establish a new batch')
        try:
            boot=transport.boot_id()
            if state['boot_id'] not in (None,boot):raise ValueError('one-boot experiment cannot mix boot IDs')
            state.update(boot_id=boot,active_boot=boot,status='PREFLIGHT');state.pop('last_error',None);_save(root,state)
            transport.wait_idle(boot);transport.mount_storage();remote=transport.deploy(state['bundle'],root)
            transport.verify_payload(remote,boot);transport.qualify_environment(remote,boot,root,state['bundle'])
            state['status']='MEASURING';_save(root,state)
            for done,doc in enumerate(tasks(state['bundle']),1):
                accepted(root,state,doc,transport,remote);state['completed_tasks']=done;_save(root,state)
            state['status']='ANALYZING';_save(root,state);result=analyze(root)
            transport.verify_payload(remote,boot);transport.idle_command('sync',boot=boot,timeout=60)
            state.update(status='COMPLETE',active_boot=None,report=str(root/'analysis/REPORT_zh.md'))
            state.pop('last_error',None);_save(root,state);c.sync_tree(root);return state
        except TransportError as error:
            state.update(status='INTERRUPTED',last_error=str(error));_save(root,state);raise
        except Exception as error:
            state.update(status='STOPPED_INVALID_RESULT',last_error=str(error));_save(root,state);raise

def main(argv=None):
    p=argparse.ArgumentParser(description='CPU资源行为：冻结匹配干预与原生证据回收');sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    a=sub.add_parser('validate');a.add_argument('--bundle',type=Path,required=True)
    for name in ('status','analyze'):
        a=sub.add_parser(name);a.add_argument('--run-dir',type=Path,required=True)
    b=sub.add_parser('board').add_subparsers(dest='action',required=True)
    a=b.add_parser('run');a.add_argument('--bundle',type=Path,required=True);a.add_argument('--run-id',default=RUN_ID)
    a.add_argument('--local-config',type=Path,default=PROJECT/'configs/local.json');a.add_argument('--execute-board',action='store_true')
    a=b.add_parser('resume');a.add_argument('--run-dir',type=Path,required=True);a.add_argument('--execute-board',action='store_true')
    args=p.parse_args(argv)
    if args.command=='prepare':result={'status':'PREPARED_OFFLINE','bundle':str(prepare())}
    elif args.command=='validate':result={'bundle_id':validate_bundle(args.bundle)['bundle_id']}
    elif args.command in ('status','analyze'):
        result=validate_state(args.run_dir) if args.command=='status' else analyze(args.run_dir)
    else:
        if not args.execute_board:p.error('board operations require --execute-board')
        if args.action=='run':
            if not re.fullmatch('[A-Za-z0-9_-]+',args.run_id):p.error('invalid run ID')
            root=ROOT/'runs'/args.run_id;connection=json.loads(args.local_config.read_text())['board'];state=create(args.bundle,root,connection)
        else:root=args.run_dir.resolve();state=validate_state(root);connection=state['connection']
        result=run_current(root,ssh(connection,state['bundle']))
    print(json.dumps(result,ensure_ascii=False,allow_nan=False));return result

if __name__=='__main__':main()
