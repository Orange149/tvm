"""Four offline diagnostic counterfactuals; never a deployable predictor.

Only callback CPU work is substituted. No pipeline wall time, CPU queue count,
FPS, serial work, launch prefix, host placement or scheduler parameter is fitted.
The source observation has a declared probe-quality limitation.
"""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import json
import statistics
import time

import analyze_resnet_middle_switch as observed
from cpu_vta_partitioning.common import read,write,sha256
from finite_service.common import freeze
from phase_model.extract import project_call,average_calls
from phase_model.simulator import rational
from isolated_output_control_model import IsolatedOutputFinite
from scheduler_gap_audit.curves import csv_write

ROOT=observed.ROOT
OUT=ROOT/'artifacts/competitor_work_audit_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_competitor_work_audit'
VARIANTS={'isolated_control':{},'front_worker1':{0:[1]},
    'front_all_workers':{0:[0,1,2]},'middle_all_workers':{2:[0,1,2]}}


def stage_program(stage):
    program=stage.get('program') or Path(stage['profile']['kernels'][0]['module']).parent.name
    if any(not k.get('module','/programs/'+program+'/module.so').endswith('/programs/'+program+'/module.so')
        for k in stage['profile']['kernels']):raise ValueError('CPU profile mixes programs')
    return program


def diagnostic_projection(receipt,stages,programs):
    raw=Path(receipt['raw']);doc=receipt['task']
    phase=[json.loads(s) for s in (raw/'phase_trace.jsonl').read_text().splitlines()]
    rows=[json.loads(s) for s in (raw/'wait_trace.jsonl').read_text().splitlines()]
    launches=[r for r in rows if r['kind']==0];callbacks=defaultdict(list);barriers=defaultdict(list)
    for row in rows:
        if row['kind'] in (1,2):(callbacks if row['kind']==1 else barriers)[row['launch']].append(row)
    projected={}
    for index in (0,2):
        stage=stages[index];native=doc['stages'][index]
        if (stage_program(stage)!=native['program'] or stage['device']!='cpu' or native['threads']!=3
            or stage['worker_core_map']!=native['worker_core_map']):
            raise ValueError('fixed diagnostic stage identity differs')
        selected=sorted((p for p in phase if p['stage']==native['program'] and p['operation']=='run'
            and doc['warmup']<=p['call']<doc['warmup']+doc['scored']),key=lambda p:p['call'])
        if [p['call'] for p in selected]!=list(range(doc['warmup'],doc['warmup']+doc['scored'])):
            raise ValueError('diagnostic callback frames incomplete')
        avg=average_calls([project_call(p,launches,callbacks,barriers) for p in selected])
        old=stage['profile']['kernels']
        if [(k['function_offset'],k['tasks'],k['barrier_generations']) for k in old]!=[
            (k['function_offset'],k['tasks'],k['barrier_generations']) for k in avg['kernels']]:
            raise ValueError('isolated and observed kernel sequence differs')
        for kernel in avg['kernels']:
            if not kernel['module'].endswith('/programs/'+native['program']+'/module.so'):
                raise ValueError('observed module path differs')
        identity=next(v for v in read(ROOT/'artifacts/cross_network_model_01/pool_temporal/INTRINSICS.json')['CPU'].values()
            if v['identity']['network']=='resnet18' and v['identity']['program']==native['program']
            and v['identity']['threads']==native['threads'])
        if identity['identity']['module_sha256']!=programs[native['program']]['files']['module.so']['sha256']:
            raise ValueError('isolated/observed module hash differs')
        provenance=identity['provenance'];source=provenance.get('receipt',provenance.get('observed_receipt'))
        expected=provenance.get('sha256',provenance.get('observed_sha256'))
        if sha256(source)!=expected:raise ValueError('isolated provenance changed')
        # Deliberately project no wall, queue, FPS, prefix, serial or tail values.
        projected[str(index)]=dict(program=native['program'],module_sha256=identity['identity']['module_sha256'],
            kernels=[dict(function_offset=k['function_offset'],tasks=k['tasks'],barrier_generations=k['barrier_generations'],
                workers=[dict(logical_worker=w['logical_worker'],compute_segments=w['compute_segments']) for w in k['workers']])
                for k in avg['kernels']],isolated_source_sha256=expected)
    return dict(schema='diagnostic-observed-callback-work-v1',diagnostic_only=True,
        contains_pipeline_measurements=True,not_a_static_cost_source=True,CPU_callbacks=projected)


def substitute(stages,projection,selected):
    if (set(projection)!={'schema','diagnostic_only','contains_pipeline_measurements','not_a_static_cost_source','CPU_callbacks'}
        or projection['schema']!='diagnostic-observed-callback-work-v1'
        or projection['diagnostic_only'] is not True or projection['contains_pipeline_measurements'] is not True
        or projection['not_a_static_cost_source'] is not True):
        raise ValueError('explicit diagnostic projection required')
    result=deepcopy(stages)
    for index,slots in selected.items():
        p=projection['CPU_callbacks'][str(index)]
        if stage_program(result[index])!=p['program']:raise ValueError('diagnostic program changed')
        kernels=result[index]['profile']['kernels']
        if len(kernels)!=len(p['kernels']):raise ValueError('diagnostic kernel count differs')
        for dst,src in zip(kernels,p['kernels']):
            if (dst['function_offset'],dst['tasks'],dst['barrier_generations'])!=(src['function_offset'],src['tasks'],src['barrier_generations']):
                raise ValueError('diagnostic callback sequence changed')
            for slot in slots:
                if (dst['workers'][slot]['logical_worker']!=slot or src['workers'][slot]['logical_worker']!=slot
                    or len(src['workers'][slot]['compute_segments'])!=dst['barrier_generations']+1
                    or any(rational(x)<0 for x in src['workers'][slot]['compute_segments'])):
                    raise ValueError('diagnostic worker identity/work invalid')
                dst['workers'][slot]['compute_segments']=deepcopy(src['workers'][slot]['compute_segments'])
    return result


def run():
    started=time.monotonic();receipts,hashes=observed.verified()
    source=ROOT/'artifacts/compiled_gap_validation_01'
    inputs=read(source/'INPUTS.json');case=inputs['networks']['resnet18'][observed.campaign.CID]
    stages=case['stages'];bundle=observed.campaign.POOL/'resnet18/capsule'
    projection=diagnostic_projection(receipts['observed'],stages,read(bundle/'plans/programs.json'))
    freeze(OUT/'DIAGNOSTIC_CALLBACKS.json',projection)
    ref=read(source/'SELECTION_FREEZE.json')['prediction_files']['resnet18/'+observed.campaign.CID+'/compiled_output_ring1']
    if sha256(ref['path'])!=ref['sha256']:raise ValueError('frozen original replay differs')
    source_files=[Path(__file__),ROOT/'tests/test_competitor_work_audit.py',ROOT/'isolated_output_control_model.py',
        ROOT/'compiled_sleep_model.py',ROOT/'finite_service_replay.py',ROOT/'finite_service/scheduler.py',
        ROOT/'model_semantics_audit/simulator.py',ROOT/'phase_model/extract.py',ROOT/'thread_history_audit/simulator.py']
    freeze(OUT/'PLAN.json',dict(source_hashes={str(p):sha256(p) for p in source_files},
        isolated_inputs_sha256=sha256(source/'INPUTS.json'),diagnostic_projection_sha256=sha256(OUT/'DIAGNOSTIC_CALLBACKS.json'),
        source_receipts=hashes,reference=ref,variants=VARIANTS,new_native_sessions=0,
        no_scheduler_or_device_model_change=True,not_a_deployable_predictor=True,
        observed_CPU_work_not_static_profile=True,no_pipeline_wall_or_FPS_in_model_inputs=True,
        source_observation_probe_quality_passed=False,
        hypothesis='small competing callback work differences may change schedule phase; compare all predeclared substitutions without fitting',
        stop='Only assess diagnostic sensitivity. Even improved error cannot justify promotion because pipeline work and perturbed timing were used.',
        literature=dict(paper='Daydream ATC 2020',section='4.1 phase 3 and Algorithm 1',
            url='https://www.cs.toronto.edu/~pekhimenko/Papers/Daydream-ATC_20.pdf',
            reuse='task-work substitution and fixed dependency replay for bounded what-if diagnosis',
            not_reproduced='CUPTI collection, full Daydream system, or actual Linux scheduling')))
    files={};p=inputs['parameters']
    for name,selected in VARIANTS.items():
        path=OUT/'replays'/(name+'.json')
        if not path.exists():
            changed=substitute(stages,projection,selected)
            result=IsolatedOutputFinite(changed,quanta=inputs['quanta_ns'],wake=True,sleeper=True,
                wakeup_granularity_ns=p['wakeup_granularity_ns'],sleeper_credit_ns=p['sleeper_credit_ns'],
                placement='wake_boundary',completion_model='independent',completion_point='midpoint',
                runnable_completion_wait=True,controls=None,gate='off',host_order='before',
                isolated_output_cost=inputs['output_costs']['resnet18_ring1'],
                collector_mask=case['configuration']['control_mask'],**inputs['window']).run()
            freeze(path,dict(diagnostic_only=True,contains_pipeline_callback_work=bool(selected),
                plan_sha256=sha256(OUT/'PLAN.json'),result=result))
        if read(path)['plan_sha256']!=sha256(OUT/'PLAN.json'):raise ValueError('diagnostic replay changed')
        files[name]=dict(path=str(path),sha256=sha256(path))
        print('离线工作量敏感性',len(files),'/',len(VARIANTS),name,flush=True)
    reference=read(ref['path'])['result'];control=read(files['isolated_control']['path'])['result']
    if control!=reference:raise ValueError('unmodified control does not exactly replay original prediction')
    freeze(OUT/'REPLAY_FREEZE.json',dict(files=files,plan_sha256=sha256(OUT/'PLAN.json'),
        unmodified_control_exactly_reproduced=True,no_native_work=True))
    # Native phase values are compared only after all four replay files freeze.
    rows=[];worker_rows=[]
    for name,selected in VARIANTS.items():
        m=read(files[name]['path'])['result'];services=[x/1e6 for x in m['replay']['stage_service_ns']]
        rows.append(dict(variant=name,front_stage_ms=services[0],middle_stage_ms=services[2],
            target_change_from_original_ms=services[2]-reference['replay']['stage_service_ns'][2]/1e6,
            observed_target_ms=receipts['observed']['summary']['stage_service_ms'][2],
            clean_target_ms=receipts['clean']['summary']['stage_service_ms'][2],
            all_stage_service_ms=services,interval_ms=m['replay']['interval_ns']/1e6))
    for index in (0,2):
        for slot in range(3):
            iso=sum((sum((rational(x) for x in k['workers'][slot]['compute_segments'])) for k in stages[index]['profile']['kernels']))
            current=sum((sum((rational(x) for x in k['workers'][slot]['compute_segments'])) for k in projection['CPU_callbacks'][str(index)]['kernels']))
            worker_rows.append(dict(stage=stage_program(stages[index]),slot=slot,isolated_CPU_ms=float(iso)/1e6,
                pipeline_observed_CPU_ms=float(current)/1e6,growth_percent=float((current/iso-1)*100)))
    output=dict(rows=rows,workers=worker_rows,diagnostic_only=True,probe_quality_supported=False,
        no_model_promotion=True,new_native_sessions=0,new_DNN=0,
        no_FPS_or_pipeline_wall_fitted=True,plan_sha256=sha256(OUT/'PLAN.json'),
        freeze_sha256=sha256(OUT/'REPLAY_FREEZE.json'),offline_seconds=time.monotonic()-started)
    write(OUT/'RESULTS.json',output,replace=True);csv_write(OUT/'counterfactuals.csv',rows);csv_write(OUT/'worker_work.csv',worker_rows)
    render(output);return output


def render(output):
    rows=output['rows'];base=rows[0]['middle_stage_ms']
    lines=['已有轨迹的离线诊断：竞争子图工作量误差是否足以改变完成拖尾？','',
        '四项对照只替换计算回调CPU工作量，线程位置、串行工作、主机、设备、队列与调度规则均保持不变。原对照精确回放旧结果。',
        '该替换来自流水内探针，且探针扰动条件未通过；仅判断模型敏感性，不是可用于未知候选的修正。不新增板测、不修改原公式或DP。','',
        '子图/逻辑线程 | 隔离计算核时ms | 流水探针核时ms | 变化%']
    for w in output['workers']:lines.append(f"{w['stage']}/{w['slot']} | {w['isolated_CPU_ms']:.3f} | {w['pipeline_observed_CPU_ms']:.3f} | {w['growth_percent']:+.2f}")
    lines+=['','对照 | CPU0–2服务ms | CPU5–7服务ms | 后者相对原模型变化ms']
    for r in rows:lines.append(f"{r['variant']} | {r['front_stage_ms']:.3f} | {r['middle_stage_ms']:.3f} | {r['target_change_from_original_ms']:+.3f}")
    lines+=['',f"同一新执行器关闭探针的目标阶段{rows[0]['clean_target_ms']:.3f} ms，开启后{rows[0]['observed_target_ms']:.3f} ms；不能按最接近其中一个值挑选模型或参数。",
        'Daydream第4节的任务工作量替换和固定依赖回放是此诊断的依据；这里没有复现其CUPTI采集或完整系统，也没有证明当前近似调度等同Linux。',
        '论文：https://www.cs.toronto.edu/~pekhimenko/Papers/Daydream-ATC_20.pdf','',
        '剩余问题：若仅小幅改变工作量即可大幅移动拖尾，单个平均阶段/线程成本不足以确定共享执行时序；若替换影响很小，则该工作量差异不足以解释本例剩余偏差。两种结果均不能推出一个通用慢化系数。',
        f"本轮板端新增0会话、0推理；四次固定离线回放约{output['offline_seconds']:.1f}秒。全部四项报告，不按误差选择有利变体。",'']
    (OUT/'FINDINGS_zh.txt').write_text('\n'.join(lines),encoding='utf-8');REPORT.mkdir(parents=True,exist_ok=True)
    import shutil
    for name in ('RESULTS.json','FINDINGS_zh.txt','counterfactuals.csv','worker_work.csv','PLAN.json','REPLAY_FREEZE.json'):
        shutil.copy2(OUT/name,REPORT/name)


if __name__=='__main__':
    result=run();print(json.dumps({k:result[k] for k in ('rows','workers','offline_seconds')},ensure_ascii=False,indent=2))
