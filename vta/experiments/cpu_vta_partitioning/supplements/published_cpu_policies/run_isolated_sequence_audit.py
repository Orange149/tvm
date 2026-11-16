"""Offline input-order audit: preserve isolated samples, never fit native times.

All eight circular origins are reported, not selected. Each scored window has
two complete cycles, so every variant has exactly the same consumed CPU work
per worker and kernel. This is a sensitivity audit, not a new Top20 predictor.
"""
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import json
import statistics as st
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,verify_inventory,digest
from finite_service.common import freeze
from phase_model.extract import project_call,average_calls
from phase_model.simulator import rational
from isolated_output_control_model import IsolatedOutputFinite
from policy_branch_eval.common import isolated_inputs_only,offline_guard
from scheduler_gap_audit.curves import csv_write

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/isolated_sequence_audit_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_isolated_sequence_audit'
CID='cfg_2ac037fa511b37a62f7c43a4'
PROFILE_FIELDS={'serial_cpu_segments','kernels'}
KERNEL_FIELDS={'module','function_offset','tasks','barrier_generations','launch_prefix_cpu','workers'}


def consumed_profile(profile):
    """Only the CPU work and dependency fields consumed by the existing engine."""
    return dict(serial_cpu_segments=deepcopy(profile['serial_cpu_segments']),
        kernels=[{k:deepcopy(kernel[k]) for k in KERNEL_FIELDS} for kernel in profile['kernels']])


def validate_profile(profile):
    if set(profile)!=PROFILE_FIELDS:raise ValueError('only isolated CPU work and dependencies accepted')
    if len(profile['serial_cpu_segments'])!=len(profile['kernels'])+1:
        raise ValueError('isolated serial segments incomplete')
    for k in profile['kernels']:
        if set(k)!=KERNEL_FIELDS or len(k['workers'])!=k['tasks']:
            raise ValueError('unexpected isolated kernel fields/participants')
        if [w['logical_worker'] for w in k['workers']]!=list(range(k['tasks'])):
            raise ValueError('isolated worker order differs')
        for w in k['workers']:
            if set(w)!={'logical_worker','compute_segments'} or len(w['compute_segments'])!=k['barrier_generations']+1:
                raise ValueError('isolated compute segments differ')
            if any(rational(x)<0 for x in w['compute_segments']):raise ValueError('negative isolated CPU work')
    return profile


def cpu_vector(profile):
    validate_profile(profile)
    return [rational(x) for x in profile['serial_cpu_segments']]+[
        x for k in profile['kernels'] for x in ([rational(k['launch_prefix_cpu'])]+
            [rational(q) for w in k['workers'] for q in w['compute_segments']])]


def validate_samples(stages,source):
    if (set(source)!={'schema','contains_pipeline_measurements','CPU_samples'}
        or source['schema']!='isolated-sequence-work-v1' or source['contains_pipeline_measurements'] is not False):
        raise ValueError('isolated sample contract required')
    expected={str(i) for i,s in enumerate(stages) if s['device']=='cpu'}
    if set(source['CPU_samples'])!=expected:raise ValueError('all CPU stages required')
    for key,samples in source['CPU_samples'].items():
        stage=stages[int(key)];base=consumed_profile(stage['profile'])
        if len(samples)!=8:raise ValueError('exactly eight isolated frames required')
        identity=lambda p:[(k['module'],k['function_offset'],k['tasks'],k['barrier_generations']) for k in p['kernels']]
        if any(identity(p)!=identity(base) for p in samples):raise ValueError('sample kernel identity differs')
        vectors=[cpu_vector(p) for p in samples]
        if [sum(row[j] for row in vectors)/8 for j in range(len(vectors[0]))]!=cpu_vector(base):
            raise ValueError('sample means do not exactly reproduce frozen isolated work')
    return source


class SequencedInputs(IsolatedOutputFinite):
    def __init__(self,*args,isolated_samples,origin,**kwargs):
        if type(origin) is not int or not 0<=origin<8:raise ValueError('predeclared sample origin 0..7 required')
        self.samples=deepcopy(validate_samples(args[0],isolated_samples)['CPU_samples']);self.origin=origin
        super().__init__(*args,**kwargs)

    def actions(self,stage):
        if stage['device']!='cpu':yield from super().actions(stage);return
        index=next(i for i,s in enumerate(self.stages) if s is stage)
        frame=self.state[index]['frame'];original=stage['profile']
        stage['profile']=self.samples[str(index)][(frame+self.origin)%8]
        try:yield from super().actions(stage)
        finally:stage['profile']=original


def project():
    src=ROOT/'artifacts/compiled_gap_validation_01';x=read(src/'INPUTS.json')
    case=x['networks']['resnet18'][CID];intr=read(ROOT/'artifacts/cross_network_model_01/pool_temporal/INTRINSICS.json')
    sources={};samples={}
    for index,stage in enumerate(case['stages']):
        if stage['device']!='cpu':continue
        program=case['configuration']['stages'][index]['program'];threads=len(stage['worker_core_map'])
        old=next(v for v in intr['CPU'].values() if v['identity']['network']=='resnet18'
            and v['identity']['program']==program and v['identity']['threads']==threads)
        pv=old['provenance'];path=pv.get('receipt',pv.get('observed_receipt'));expected=pv.get('sha256',pv.get('observed_sha256'))
        if sha256(path)!=expected:raise ValueError('isolated sample source changed')
        receipt=read(path);doc=receipt['task'];raw=Path(receipt['raw'])
        if (doc['kind']!='calibration' or doc['spec']['scope']!='component' or len(doc['stages'])!=1
            or doc['stages'][0]['program']!=program or doc['stages'][0]['threads']!=threads
            or doc['scored']!=8 or (raw/'task_hash').read_text().strip()!=digest(doc)
            or (raw/'boot_id').read_text().strip()!=receipt['boot_id']):
            raise ValueError('non-isolated or mismatched sample source')
        verify_inventory(raw,receipt['files']);sources[path]=expected
        if doc['runtime_sha256']!=pv['runtime_sha256']:raise ValueError('isolated runtime identity differs')
        phase=[json.loads(s) for s in (raw/'phase_trace.jsonl').read_text().splitlines()]
        events=[json.loads(s) for s in (raw/'wait_trace.jsonl').read_text().splitlines()]
        launches=[r for r in events if r['kind']==0];callbacks=defaultdict(list);barriers=defaultdict(list)
        for r in events:
            if r['kind'] in (1,2):(callbacks if r['kind']==1 else barriers)[r['launch']].append(r)
        selected=sorted((r for r in phase if r['stage']==program and r['operation']=='run'
            and doc['warmup']<=r['call']<doc['warmup']+8),key=lambda r:r['call'])
        if [r['call'] for r in selected]!=list(range(doc['warmup'],doc['warmup']+8)):
            raise ValueError('isolated frame sequence incomplete')
        calls=[project_call(r,launches,callbacks,barriers) for r in selected]
        samples[str(index)]=[consumed_profile(average_calls([call])) for call in calls]
    payload=dict(schema='isolated-sequence-work-v1',contains_pipeline_measurements=False,CPU_samples=samples)
    validate_samples(case['stages'],payload)
    freeze(OUT/'ISOLATED_SAMPLES.json',payload)
    freeze(OUT/'SOURCE.json',dict(case=case,window=x['window'],quanta_ns=x['quanta_ns'],
        parameters=x['parameters'],output_cost=x['output_costs']['resnet18_ring1'],isolated_receipts=sources,
        original_input_sha256=sha256(src/'INPUTS.json'),sample_sha256=sha256(OUT/'ISOLATED_SAMPLES.json')))
    return dict(isolated_stages=len(samples),isolated_frames_per_stage=8,all_means_exact=True)


def run():
    started=time.monotonic();project()
    with offline_guard(OUT) as guard,isolated_inputs_only():
        x=read(OUT/'SOURCE.json');samples=read(OUT/'ISOLATED_SAMPLES.json');case=x['case'];p=x['parameters']
        freeze(OUT/'PLAN.json',dict(source_sha256=sha256(OUT/'SOURCE.json'),sample_sha256=sha256(OUT/'ISOLATED_SAMPLES.json'),
            source_hashes={str(f):sha256(f) for f in (Path(__file__),ROOT/'tests/test_isolated_sequence_audit.py',
                ROOT/'isolated_output_control_model.py',ROOT/'compiled_sleep_model.py',ROOT/'finite_service_replay.py',
                ROOT/'finite_service/scheduler.py',ROOT/'model_semantics_audit/simulator.py')},
            origins=list(range(8)),fixed_same_origin_for_all_CPU_stages=True,all_variants_required=True,
            independent_cross_stage_sample_correlation_unknown=True,
            scored_window_per_variant_has_two_complete_isolated_cycles=True,
            hypothesis='averaging before nonlinear contention replay can hide schedule sensitivity',
            stop='Report all eight outcomes and no fitted point; narrow variation excludes this simple explanation, wide variation shows input-order uncertainty only.',
            scheduler_and_device_rules_unchanged=True,no_pipeline_cost_or_FPS_inputs=True,new_native_sessions=0,
            no_static_score_DP_or_Top20_change=True))
        files={}
        for origin in range(8):
            path=OUT/'replays'/(str(origin)+'.json')
            if not path.exists():
                result=SequencedInputs(deepcopy(case['stages']),isolated_samples=samples,origin=origin,
                    quanta=x['quanta_ns'],wake=True,sleeper=True,wakeup_granularity_ns=p['wakeup_granularity_ns'],
                    sleeper_credit_ns=p['sleeper_credit_ns'],placement='wake_boundary',completion_model='independent',
                    completion_point='midpoint',runnable_completion_wait=True,controls=None,gate='off',host_order='before',
                    isolated_output_cost=x['output_cost'],collector_mask=case['configuration']['control_mask'],**x['window']).run()
                freeze(path,dict(origin=origin,result=result,plan_sha256=sha256(OUT/'PLAN.json'),isolated_samples_only=True))
            if read(path)['plan_sha256']!=sha256(OUT/'PLAN.json'):raise ValueError('input-order audit changed')
            files[str(origin)]=dict(path=str(path),sha256=sha256(path));print('隔离样本顺序核查',origin+1,'/8',flush=True)
        freeze(OUT/'REPLAY_FREEZE.json',dict(files=files,plan_sha256=sha256(OUT/'PLAN.json'),guard=guard,
            no_native_results_read=True,all_origins_reported=True))
    rows=[]
    for origin,item in files.items():
        r=read(item['path'])['result']['replay']
        rows.append(dict(origin=int(origin),front_ms=r['stage_service_ns'][0]/1e6,
            middle_ms=r['stage_service_ns'][2]/1e6,interval_ms=r['interval_ns']/1e6))
    frozen=read(ROOT/'artifacts/compiled_gap_validation_01/SELECTION_FREEZE.json')['prediction_files']['resnet18/'+CID+'/compiled_output_ring1']
    if sha256(frozen['path'])!=frozen['sha256']:raise ValueError('old mean-cost prediction changed')
    reference=read(frozen['path'])['result']['replay']
    result=dict(rows=rows,mean_input_middle_ms=reference['stage_service_ns'][2]/1e6,
        middle_range_ms=[min(r['middle_ms'] for r in rows),max(r['middle_ms'] for r in rows)],
        middle_mean_ms=st.mean(r['middle_ms'] for r in rows),same_window_CPU_work=True,
        all_origins_kept_no_favorable_selection=True,not_an_independent_validation=True,
        no_claim_of_Linux_scheduling_reproduction=True,no_new_model_adopted=True,
        new_native_sessions=0,new_DNN=0,offline_seconds=time.monotonic()-started)
    write(OUT/'RESULTS.json',result,replace=True);csv_write(OUT/'origins.csv',rows)
    render(result);return result


def render(r):
    lines=['同一隔离CPU工作量，改变原始样本起始位置：离线输入顺序核查','',
        '使用四个CPU阶段各自原有8次隔离记录，只保留引擎消费的CPU工作及依赖字段；流水时间、排队及FPS不进入回放。',
        '每种起始位置的16帧窗口恰好包含两轮原始样本，所以每个worker、每个内核的总计算、串行与启动CPU工作都相同。八种起始偏移全部报告，不择优。',
        '不同阶段使用相同样本编号只是事前固定口径，不代表真实跨阶段相关性；结果范围不是严格上下界，也不是一个已验证的新预测公式。','',
        '起始偏移 | CPU0–2服务ms | CPU5–7服务ms | 全流水预测间隔ms']
    for x in r['rows']:lines.append(f"{x['origin']} | {x['front_ms']:.3f} | {x['middle_ms']:.3f} | {x['interval_ms']:.3f}")
    lines+=['',f"原平均输入预测CPU5–7为{r['mean_input_middle_ms']:.3f} ms；保留隔离样本后范围{r['middle_range_ms'][0]:.3f}—{r['middle_range_ms'][1]:.3f} ms，八项平均{r['middle_mean_ms']:.3f} ms。",
        '该对照不使用流水内CPU工作，不修改调度规则。它仅检查平均化与顺序敏感性，不以接近某一实测值作为采用依据。',
        f"新增板端会话及推理均为0，八次离线回放约{r['offline_seconds']:.1f}秒。",'']
    (OUT/'FINDINGS_zh.txt').write_text('\n'.join(lines),encoding='utf-8');REPORT.mkdir(parents=True,exist_ok=True)
    import shutil
    for name in ('RESULTS.json','FINDINGS_zh.txt','origins.csv','PLAN.json','SOURCE.json','REPLAY_FREEZE.json'):
        shutil.copy2(OUT/name,REPORT/name)


if __name__=='__main__':
    r=run();print(json.dumps(r,ensure_ascii=False,indent=2))
