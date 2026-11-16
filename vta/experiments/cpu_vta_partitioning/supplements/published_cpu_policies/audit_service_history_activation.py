"""Observe the existing model's service-history grants without changing choices.

Two already-frozen known-layout replays, no new scheduler or native execution.
The empirical budget is not identified with the real kernel's exact time slice.
"""
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import json
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256
from finite_service.common import freeze
from phase_model.simulator import ceiling
from isolated_output_control_model import IsolatedOutputFinite
from device_completion_replay import CompletionFinite
from policy_branch_eval.common import isolated_inputs_only

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/service_history_activation_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_service_history_activation'


class ObserveHistory:
    """Delegate all scheduling state changes; keep independent diagnostic state."""
    def __init__(self,original,metadata=lambda key:{},clock=lambda:0):
        self.original=original;self.metadata=metadata;self.clock=clock
        self.service=defaultdict(Fraction);self.pending=defaultdict(list);self.rows=[]

    def __getattr__(self,key):return getattr(self.original,key)

    def select(self,tasks):
        before=dict(self.original.current);left=dict(self.original.left);counts=dict(self.original.last_count)
        chosen,expiry=self.original.select(tasks)
        for core in range(4):
            old=before.get(core);current=self.original.current.get(core)
            dispatch=old!=current or (left.get(core) is not None and left[core]<=0)
            if dispatch or current is None:
                for row in self.pending.pop(core,[]):row['end_observed_dispatch_ns']=ceiling(self.clock())
                self.service[core]=Fraction(0)
            keys=[key for key,t in tasks.items() if t['core']==core]
            if (current is not None and current==old and not dispatch and len(keys)>=2
                and (counts.get(core,0)<2 or left.get(core) is None) and self.service[core]>0):
                key=chosen[core];q=self.original.quantum(len(keys))
                row=dict(model_time_ns=ceiling(self.clock()),core=core,
                    actor=list(current),selected_task=self.metadata(key),runnable=len(keys),
                    service_since_observed_dispatch_ns=ceiling(self.service[core]),
                    new_empirical_budget_ns=ceiling(q),
                    prior_service_exceeds_new_budget=self.service[core]>=q,
                    contending_tasks=[self.metadata(k) for k in keys if k!=key],
                    post_grant_contended_service_ns=Fraction(0),
                    end_observed_dispatch_ns=None)
                self.rows.append(row);self.pending[core].append(row)
        return chosen,expiry

    def consume(self,tasks,duration):
        self.original.consume(tasks,duration)
        for core,key in self.original.choices.items():
            self.service[core]+=duration
            if sum(t['core']==core for t in tasks.values())>=2:
                for row in self.pending[core]:row['post_grant_contended_service_ns']+=duration

    def result(self):
        return [{**r,'post_grant_contended_service_ns':ceiling(r['post_grant_contended_service_ns'])} for r in self.rows]


def inputs():
    res_path=ROOT/'artifacts/compiled_gap_validation_01/INPUTS.json';res=read(res_path)
    res_id='cfg_2ac037fa511b37a62f7c43a4';res_case=res['networks']['resnet18'][res_id]
    ref=read(ROOT/'artifacts/compiled_gap_validation_01/SELECTION_FREEZE.json')['prediction_files']['resnet18/'+res_id+'/compiled_output_ring1']
    yolo_path=ROOT/'artifacts/current_model_gap_01/INPUTS.json';yolo=read(yolo_path)
    yolo_plan=read(ROOT/'artifacts/current_model_gap_01/PLAN.json')
    if sha256(yolo_path)!=yolo_plan['input_sha256']:raise ValueError('YOLO isolated input changed')
    yolo_ref=ROOT/'artifacts/current_model_gap_01/independent.json'
    seal=read(ROOT/'artifacts/current_model_gap_01/REPLAY_FREEZE.json')
    if sha256(yolo_ref)!=seal['files']['independent.json']:raise ValueError('YOLO frozen reference changed')
    if res['contains_pipeline_performance'] or yolo['native_trace_or_performance_included']:
        raise ValueError('input contains pipeline performance')
    cases=dict(resnet18=dict(kind='compiled_output',stages=res_case['stages'],quanta=res['quanta_ns'],
        window=res['window'],params=res['parameters'],output_cost=res['output_costs']['resnet18_ring1'],
        collector_mask=res_case['configuration']['control_mask'],reference=ref),
        yolov3_tiny416=dict(kind='original_finite_independent',stages=yolo['case']['stages'],
            quanta=yolo['quanta_ns'],window=yolo['window'],
            reference=dict(path=str(yolo_ref),sha256=sha256(yolo_ref))))
    return cases,{str(res_path):sha256(res_path),str(yolo_path):sha256(yolo_path)}


def run():
    start=time.monotonic();cases,input_hashes=inputs()
    source_files=[Path(__file__),ROOT/'tests/test_service_history_activation.py',ROOT/'finite_service/scheduler.py',
        ROOT/'finite_service_replay.py',ROOT/'compiled_sleep_model.py',ROOT/'isolated_output_control_model.py']
    freeze(OUT/'PLAN.json',dict(input_hashes=input_hashes,
        cases={n:dict(kind=c['kind'],reference=c['reference'],window=c['window']) for n,c in cases.items()},
        source_files={str(p):sha256(p) for p in source_files},new_DNN=0,new_native_sessions=0,
        no_scheduler_decision_or_profile_changed=True,
        hypothesis='Count service-history reset activations and locate their actors in actual existing model runs.',
        reference_timestamps_must_be_identical=True,
        no_extra_service_sum_as_wait_penalty=True,
        mixed_prior_scheduler_versions_are_not_a_cross_network_performance_comparison=True))
    summary=[];files={}
    for name,case in cases.items():
        path=OUT/(name+'.json')
        if path.exists():
            saved=read(path)
            if saved['plan_sha256']!=sha256(OUT/'PLAN.json'):raise ValueError('audit replay plan differs')
        else:
            with isolated_inputs_only():
                kw=dict(quanta=case['quanta'],placement='wake_boundary',completion_model='independent',
                    completion_point='midpoint',runnable_completion_wait=True,controls=None,
                    gate='off',host_order='before',**case['window'])
                if case['kind']=='compiled_output':
                    p=case['params'];model=IsolatedOutputFinite(case['stages'],wake=True,sleeper=True,
                        wakeup_granularity_ns=p['wakeup_granularity_ns'],sleeper_credit_ns=p['sleeper_credit_ns'],
                        isolated_output_cost=case['output_cost'],collector_mask=case['collector_mask'],**kw)
                else:model=CompletionFinite(case['stages'],**kw)
                observer=ObserveHistory(model.scheduler,lambda key:dict(model.active[key]['meta']),lambda:model.time)
                model.scheduler=observer;result=model.run()
                if sha256(case['reference']['path'])!=case['reference']['sha256']:
                    raise ValueError('reference changed')
                reference=read(case['reference']['path'])['result']
                if result!=reference:raise ValueError('observation changed frozen replay')
                saved=dict(result=result,events=observer.result(),plan_sha256=sha256(OUT/'PLAN.json'),
                    complete_result_exactly_reproduced=True)
                freeze(path,saved)
        events=saved['events'];window=case['window']
        kept=[e for e in events if window['warmup']<=e['selected_task'].get('frame',-1)<window['warmup']+window['scored']]
        role_counts=Counter(str(e['actor']) for e in kept)
        summary.append(dict(network=name,replay_kind=case['kind'],total_activations=len(events),
            selected_task_in_scored_frames=len(kept),
            prior_service_exceeds_new_budget=sum(e['prior_service_exceeds_new_budget'] for e in kept),
            maximum_service_before_new_grant_ms=max((e['service_since_observed_dispatch_ns'] for e in kept),default=0)/1e6,
            selected_actor_counts=dict(role_counts),complete_result_exactly_reproduced=True,
            empirical_budget_not_actual_kernel_slice=True))
        files[name]=dict(path=str(path),sha256=sha256(path))
        print('只读模型运行历史核查',name,len(events),'次触发，结果完全一致',flush=True)
    freeze(OUT/'FREEZE.json',dict(files=files,plan_sha256=sha256(OUT/'PLAN.json')))
    output=dict(summary=summary,new_DNN=0,new_native_sessions=0,offline_seconds=time.monotonic()-start,
        no_scheduler_correction_tested=True,no_static_or_Top20_gain_claim=True,
        event_presence_does_not_prove_native_DNN_error_cause=True)
    write(OUT/'RESULTS.json',output,replace=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    lines=['CPU运行历史重置：现有DNN回放中的触发核查','',
        '只增加独立日志，不改变调度、成本、设备时序或任何时间戳。两个既有回放的完整结果逐字段完全恢复。',
        'ResNet使用先前编译内核常量＋隔离输出成本的版本；YOLO使用与旧完整调度轨迹匹配的原有限服务版本。两者不同，不构成性能公平比较。',
        '原近似：一个线程先独占核心，之后出现竞争且仍被选中时，直接给它新的经验服务预算，没有扣除该轮已运行量。',
        '运行内核的tick检查会考虑当前调度段已经运行的历史；但实际tick相位与Yield/重新调度仍未被复原。', '',
        '网络 / 全部触发 / 当前选中工作属于计入帧的触发 / 已运行量不少于新预算 / 最长既有运行量ms']
    for s in summary:lines.append(f"{s['network']} / {s['total_activations']} / {s['selected_task_in_scored_frames']} / {s['prior_service_exceeds_new_budget']} / {s['maximum_service_before_new_grant_ms']:.3f}")
    lines+=['','触发不等于已证明造成实测误差。多个核心或重叠事件不能直接相加成丢失时间，也不能据此拟合一个惩罚。',
        '日志中的经验预算不是实际内核time slice。若考虑修正，须保留调度段运行历史，同时明确tick和主动Yield语义的近似，而非直接以已运行量超额值加到阶段耗时。',
        '本轮没有新模型、没有新推理或新Top-20。下一步由触发位置和关键worker的对应关系判断是否值得验证最小修正。']
    (REPORT/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(output,ensure_ascii=False,indent=2));return output


if __name__=='__main__':run()
