"""Bounded offline test of one service-history correction on five known cases."""
from pathlib import Path
import json
import statistics as st
import time

from audit_service_history_activation import inputs,ROOT
from retained_service_history import RetainedServiceHistory
from isolated_output_control_model import IsolatedOutputFinite
from device_completion_replay import CompletionFinite
from phase_model.simulator import rational
from cpu_vta_partitioning.common import read,write,sha256
from finite_service.common import freeze
from policy_branch_eval.common import isolated_inputs_only
from scheduler_gap_audit.curves import csv_write

OUT=ROOT/'artifacts/retained_service_history_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_retained_service_history'


def cases():
    original,hashes=inputs();result={n+'/diagnostic':dict(c,network=n) for n,c in original.items()}
    source=ROOT/'artifacts/compiled_gap_validation_01';p=read(source/'INPUTS.json')
    refs=read(source/'SELECTION_FREEZE.json')['prediction_files']
    for network,configs in p['networks'].items():
        for cid,config in configs.items():
            if cid=='cfg_2ac037fa511b37a62f7c43a4':continue
            result[network+'/'+cid]=dict(kind='compiled_output',network=network,
                stages=config['stages'],quanta=p['quanta_ns'],window=p['window'],params=p['parameters'],
                output_cost=p['output_costs'][network+'_ring1'],collector_mask=config['configuration']['control_mask'],
                reference=refs[network+'/'+cid+'/compiled_output_ring1'])
    if len(result)!=5:raise ValueError('expected exactly five predeclared known cases')
    return result,hashes


def construct(case):
    kw=dict(quanta=case['quanta'],placement='wake_boundary',completion_model='independent',
        completion_point='midpoint',runnable_completion_wait=True,controls=None,
        gate='off',host_order='before',**case['window'])
    if case['kind']=='compiled_output':
        p=case['params'];return IsolatedOutputFinite(case['stages'],wake=True,sleeper=True,
            wakeup_granularity_ns=p['wakeup_granularity_ns'],sleeper_credit_ns=p['sleeper_credit_ns'],
            isolated_output_cost=case['output_cost'],collector_mask=case['collector_mask'],**kw)
    return CompletionFinite(case['stages'],**kw)


def callback_work(result):
    return {(r['stage'],r['slot'],r['kernel'],r['generation']):(r['calls'],rational(r['CPU']))
            for r in result['actor_totals'] if r['role']=='stage' and r['operation']=='worker_compute'}


def run():
    start=time.monotonic();selected,hashes=cases()
    source_files=[Path(__file__),ROOT/'retained_service_history.py',ROOT/'tests/test_retained_service_history.py',
        ROOT/'audit_service_history_activation.py',ROOT/'finite_service/scheduler.py',ROOT/'compiled_sleep_model.py',
        ROOT/'finite_service_replay.py',ROOT/'device_completion_replay.py',ROOT/'isolated_output_control_model.py']
    freeze(OUT/'PLAN.json',dict(cases=selected,input_hashes=hashes,
        source_hashes={str(p):sha256(p) for p in source_files},
        rule='Preserve service since model dispatch, subtract it when recomputing the existing empirical budget.',
        source_basis='Matched kernel task_tick_fair and set_next_entity consume existing execution history.',
        missing='Actual tick phase, voluntary Yield and runqueue state remain approximated.',
        parameters_added=0,profiles_refreshed=0,new_DNN=0,new_native_sessions=0,
        hypothesis='A reset that grants excess modeled service could distort worker completion under sharing.',
        acceptance='Only supports further validation if CPU-stage MAE falls in both networks without increasing their worst CPU-stage error; no promotion from these known cases.',
        stop='If this simple correction is inconsistent, preserve the negative result; do not fit granules or add penalties.',
        complete_case_set_frozen_before_native_analysis=True,main_static_score_DP_and_domain_unchanged=True))
    files={}
    for key,case in selected.items():
        path=OUT/'replays'/(key.replace('/','_')+'.json')
        if not path.exists():
            with isolated_inputs_only():
                if sha256(case['reference']['path'])!=case['reference']['sha256']:
                    raise ValueError('original reference changed')
                old=read(case['reference']['path'])['result']
                model=construct(case);model.scheduler=RetainedServiceHistory(model.scheduler)
                result=model.run()
                if callback_work(result)!=callback_work(old):raise ValueError('prescribed callback work changed')
                freeze(path,dict(result=result,plan_sha256=sha256(OUT/'PLAN.json'),
                    no_pipeline_values_read=True,all_prescribed_callback_work_identical=True))
        if read(path)['plan_sha256']!=sha256(OUT/'PLAN.json'):raise ValueError('correction identity differs')
        files[key]=dict(path=str(path),sha256=sha256(path));print('离线运行历史修正',len(files),'/5',key,flush=True)
    freeze(OUT/'REPLAY_FREEZE.json',dict(files=files,plan_sha256=sha256(OUT/'PLAN.json'),
        predictions_frozen_before_native_analysis=True))
    # Existing clean measurements are read only after every candidate freezes.
    native_path=ROOT/'artifacts/compiled_gap_validation_01/RESULTS.json';native=read(native_path)
    trace_path=ROOT/'artifacts/yolo_scheduler_trace_repair_01/analysis/RESULTS.json';yolo=read(trace_path)
    rows=[]
    for key,case in selected.items():
        if key=='yolov3_tiny416/diagnostic':observed=yolo['clean']['stage_service_ms'];native_source=trace_path
        else:
            cid='cfg_2ac037fa511b37a62f7c43a4' if key=='resnet18/diagnostic' else key.split('/')[1]
            ns=[r for r in native['rows'] if r['configuration_id']==cid]
            if not ns or any(r['all_native_stage_service_ms']!=ns[0]['all_native_stage_service_ms'] for r in ns):
                raise ValueError('same-case clean results differ across model versions')
            observed=ns[0]['all_native_stage_service_ms'];native_source=native_path
        old=read(case['reference']['path'])['result'];new=read(files[key]['path'])['result']
        for i,stage in enumerate(case['stages']):
            if stage['device']!='cpu':continue
            before=old['replay']['stage_service_ns'][i]/1e6;after=new['replay']['stage_service_ns'][i]/1e6
            rows.append(dict(case=key,network=case['network'],stage=i,
                program=Path(stage['profile']['kernels'][0]['module']).parent.name,
                native_service_ms=observed[i],before_ms=before,after_ms=after,
                before_error_ms=before-observed[i],after_error_ms=after-observed[i],
                absolute_error_change_ms=abs(after-observed[i])-abs(before-observed[i]),
                native_source=str(native_source),native_source_sha256=sha256(native_source)))
    networks=[]
    for name in ('resnet18','yolov3_tiny416'):
        r=[x for x in rows if x['network']==name]
        before=st.mean(abs(x['before_error_ms']) for x in r);after=st.mean(abs(x['after_error_ms']) for x in r)
        worst0=max(abs(x['before_error_ms']) for x in r);worst1=max(abs(x['after_error_ms']) for x in r)
        networks.append(dict(network=name,CPU_stage_rows=len(r),before_MAE_ms=before,after_MAE_ms=after,
            before_worst_absolute_ms=worst0,after_worst_absolute_ms=worst1,
            development_gate_passed=after<before and worst1<=worst0))
    result=dict(rows=rows,networks=networks,development_gate_passed=all(x['development_gate_passed'] for x in networks),
        new_DNN=0,new_native_sessions=0,correction_promoted=False,
        known_cases_only=True,no_new_Top20_or_throughput_claim=True,seconds=time.monotonic()-start)
    write(OUT/'RESULTS.json',result,replace=True);csv_write(OUT/'stages.csv',rows)
    REPORT.mkdir(parents=True,exist_ok=True)
    text=['CPU调度段运行历史：无新增参数的离线修正对照','',
        '全部五项已知布局事前冻结，只替换调度预算的历史记账：同一调度段已运行量，在核心出现竞争或预算变化时继续保留。',
        '使用原隔离计算量、线程映射、设备完成模型、输入输出成本及经验预算；全部回调计算CPU量逐项守恒。',
        '不是Linux CFS的完整复现，缺少实际tick相位、主动Yield和真实运行队列历史。',
        '两个诊断布局保留各自原版本；另外三项来自先前的四布局验证。它们都是已知数据，不能称本次独立确认。', '',
        '布局 / CPU阶段 / 干净实测ms / 原预测ms / 修正预测ms / 绝对误差变化ms']
    for r in rows:text.append(f"{r['case']} / {r['program']} / {r['native_service_ms']:.3f} / {r['before_ms']:.3f} / {r['after_ms']:.3f} / {r['absolute_error_change_ms']:+.3f}")
    text+=['','预设继续条件：两个网络CPU阶段平均绝对误差都下降，且各自最大误差不增加。仅支持进一步验证，不据此直接接入DP。']
    for n in networks:text.append(f"{n['network']}：MAE {n['before_MAE_ms']:.3f}→{n['after_MAE_ms']:.3f}ms；最差 {n['before_worst_absolute_ms']:.3f}→{n['after_worst_absolute_ms']:.3f}ms；通过 {n['development_gate_passed']}")
    text+=['','整体继续条件：'+str(result['development_gate_passed']),
        '未证明新选点或实际吞吐改善。正式静态评分和完整DP保持原样，不把误差差值回填成慢化参数。',
        '若未通过，则说明仅保留模型运行历史仍不足；不通过调节预算或增加拟合项挽救该结果。',
        f"本轮新增板测0、推理0，离线约{result['seconds']:.1f}秒。"]
    (REPORT/'FINDINGS_zh.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(networks=networks,development_gate_passed=result['development_gate_passed'],new_DNN=0,seconds=result['seconds']),ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':run()
