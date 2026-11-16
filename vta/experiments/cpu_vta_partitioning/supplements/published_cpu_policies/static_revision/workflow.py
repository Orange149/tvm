from collections import Counter,defaultdict
from fractions import Fraction
from pathlib import Path
from statistics import mean
import shutil,time
from cpu_vta_partitioning.common import digest,read,write,sha256,now
from model_semantics_audit.replay import validate_isolated
from model_semantics_audit.simulator import AuditReplay
from phase_model.simulator import rational
from thread_history_audit.simulator import HistoryReplay
from scheduler_gap_audit.curves import csv_write
from .common import ROOT,OUTPUT,REPORT,INPUT,TRACE,freeze,protocol,sources
from .trace_replay import TraceReplay


def verify_delivery(directory):
    directory=Path(directory);delivery=read(directory/'DELIVERY.json');checked=0
    for key in ('sources','files'):
        values=delivery.get(key,{})
        if isinstance(values,dict):
            for path,h in values.items():
                if isinstance(h,str) and len(h)==64 and path.startswith('/'):
                    if sha256(path)!=h:raise ValueError('frozen evidence differs: '+path)
                    checked+=1
    return dict(path=str(directory/'DELIVERY.json'),sha256=sha256(directory/'DELIVERY.json'),checked_hashes=checked)


def audit(output=OUTPUT):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    predecessors=[verify_delivery(ROOT/'artifacts'/n) for n in
        ('scheduler_gap_audit_01','model_semantics_audit_01','thread_history_audit_01',
         'yolo_host_placement_01','yolo_scheduler_trace_01','yolo_scheduler_trace_repair_01')]
    input_freeze=read(INPUT/'INPUT_FREEZE.json')
    for p,h in input_freeze['files'].items():
        if sha256(INPUT/p)!=h:raise ValueError('frozen isolated projection differs')
    iso=validate_isolated(read(INPUT/'ISOLATED_INPUTS.json'))
    if sorted(iso['cases'])!=sorted(protocol()['cases']):raise ValueError('audit cases changed')
    # No pipeline control projection or native timing enters the replay input.
    projection=dict(schema=iso['schema'],origin=iso['origin'],contains_pipeline_data=False,
        cases=iso['cases'],window=protocol()['window'],source_hashes=iso['source_hashes'])
    freeze(output/'ISOLATED_INPUTS.json',projection)
    formulas=[
        dict(term='CPU_per_core',formula='D_c=sum_i,j:m_ij=c U_i*w_ij + sum_i:caller_i=c B_i',
             code='cpu_policies/open_resources/costs.py:score_configuration',
             input='isolated run CPU work, whole-session logical thread shares, boundary CPU work',
             limitation='whole-session shares are rescaled to scored compute; homogeneous core permutation; no scheduling chronology'),
        dict(term='CPU_subsets',formula='max_nonempty_S ceil((sum_c_in_S D_c + H*1[M subset S])/|S|)',
             code='staged_ablation/model.py:score',input='all 15 nonempty core subsets; isolated VTA run plus boundary CPU work',
             limitation='host is ideally movable within its allowed mask; producer and collector CPU not separately profiled'),
        dict(term='CPU_stage',formula='max_i s_i',code='staged_ablation/model.py:score',
             input='isolated full stage service',limitation='isolated service is not shared pipeline service'),
        dict(term='VTA_sum',formula='sum_all_VTA_islands v_i',code='staged_ablation/model.py:score',
             input='isolated VTA stage service at requested host placement',limitation='single accelerator serialized; service includes host elapsed portions'),
        dict(term='DDR_sum',formula='sum_i d_i',code='cpu_policies/open_resources/costs.py:ddr_stage_service',
             input='request DMA bytes plus CPU traffic proxies; frozen bandwidth',limitation='traffic proxy, not measured total DDR utilization'),
        dict(term='serial_group',formula='max_group sum_i:group(i)=group s_i',code='cpu_policies/open_resources/costs.py:score_configuration',
             input='matching resource_group gate',limitation='only used when execution actually serializes repeated exact groups'),
        dict(term='team_service',formula='max_c sum_i:c in A_i s_i',code='resource_envelope/model.py:score',
             input='isolated stage service and selected CPU groups',limitation='planning proxy for overlapping open groups, not Linux latency bound'),
        dict(term='main_score',formula='max(CPU_stage,VTA_sum,DDR_sum,CPU_subsets,serial_group,team_service)',
             code='staged_ablation/common.py:fine_host; staged_ablation/model.py:score',
             input='nonuniform thread shares plus host CPU work plus team service; host concentration OFF',
             limitation='candidate selection estimate; not strict whole-system throughput upper bound'),
        dict(term='Tarnawski_adaptation',formula='max(max_c sum_i:on_c s_i, sum_VTA v_i, sum_DDR d_i)',
             code='staged_ablation/model.py:tarnawski',input='single physical CPU core per stage; resource_group execution',
             limitation='platform adaptation, not unchanged reproduction of original paper input/graph/compiler'),
        dict(term='temporal_replay',formula='event replay of isolated per-launch worker CPU work, fork/join, queues, host CPU/delay',
             code='model_semantics_audit/simulator.py:AuditReplay',input='isolated CPU and VTA phase projections',
             limitation='instant positive fluid share; original host placement reselected each event; diagnostic observed controls not deployable cost')]
    specs=[dict(case=c,placement=p) for c in protocol()['cases'] for p in protocol()['replay_modes']]
    freeze(output/'PLAN.json',dict(protocol=protocol(),specs=specs,sources=sources(),
        isolated_inputs_sha256=sha256(output/'ISOLATED_INPUTS.json'),predecessors=predecessors,
        formulas=formulas,corrected_old_trace_verdict_sha256=sha256(TRACE/'DATA_VERDICT.json'),
        reference_data_known_development=True,pipeline_inputs_used_for_prediction=False))
    csv_write(output/'model_formula_inventory.csv',formulas)
    progress=dict(goal_status='ACTIVE',phase='offline audit and matched-window comparison',
        previous_goal_turn='initial goal activation; no previous goal investigation turn',
        verified_existing=['network-specific real profiles','per-stage thread and affinity choices',
            'main model definition and host-concentration exclusion','corrected scheduler trace'],
        required_remaining=['trace-versus-replay alignment','workload-dependent mechanism diagnosis',
            'literature-backed minimal correction','frozen unseen-configuration validation',
            'conditional complete DP and equal-budget comparison','contribution and cost reporting'],
        current_protocol=protocol(),updated_at=now())
    write(output/'GOAL_PROGRESS.json',progress,replace=True)
    return dict(frozen=True,formula_terms=len(formulas),replays=len(specs),new_native_sessions=0)


def replay(output=OUTPUT):
    output=Path(output);plan=read(output/'PLAN.json')
    if plan['sources']!=sources() or sha256(output/'ISOLATED_INPUTS.json')!=plan['isolated_inputs_sha256']:
        raise ValueError('revision replay code or input changed')
    iso=validate_isolated(read(output/'ISOLATED_INPUTS.json'));files={};tick=time.monotonic()
    for s in plan['specs']:
        case=iso['cases'][s['case']];name='replays/'+digest(s)[:24]+'.json';path=output/name
        identity=dict(spec=s,plan_sha256=sha256(output/'PLAN.json'))
        if path.exists():
            item=read(path)
            if item['input_identity']!=digest(identity) or item['content_sha256']!=digest({k:v for k,v in item.items() if k!='content_sha256'}):
                raise ValueError('revision replay changed')
        else:
            kwargs=dict(placement=s['placement'],runnable_completion_wait=True,controls=None,
                gate=case['configuration']['execution_mode'],host_order='before',**protocol()['window'])
            run=TraceReplay(case['stages'],**kwargs).run()
            original=HistoryReplay(case['stages'],**kwargs).run()
            if run['replay']!=original['replay'] or run['actor_totals']!=original['actor_totals']:
                raise ValueError('observation changed frozen model behavior')
            item=dict(spec=s,result=run,input_identity=digest(identity),pipeline_inputs_read=False,
                      trace_does_not_change_baseline=True,development_not_unseen_validation=True)
            item['content_sha256']=digest(item);freeze(path,item)
        files[name]=sha256(path)
        print('同窗口离线对照',s['case'],s['placement'],flush=True)
    freeze(output/'REPLAY_FREEZE.json',dict(files=files,sources=sources(),plan_sha256=sha256(output/'PLAN.json'),
        all_predictions_before_new_analysis=True,new_native_sessions=0))
    write(output/'REPLAY_COST.json',dict(seconds=time.monotonic()-tick,new_native_sessions=0,new_DNN_inferences=0),replace=True)
    return dict(replays=len(files),new_native_sessions=0)


def intrinsic_summary(case,value):
    result=[]
    for stage,s in enumerate(value['stages']):
        if s['device']!='cpu':continue
        p=s['profile'];per_kernel=[];thread=[Fraction(0) for _ in s['worker_core_map']]
        for k in p['kernels']:
            work=[sum(map(rational,w['compute_segments'])) for w in k['workers']]
            for j,w in enumerate(work):thread[j]+=w
            per_kernel.append(max(work))
        total=sum(per_kernel);largest=sorted(per_kernel,reverse=True)
        result.append(dict(case=case,stage=stage,threads=len(s['worker_core_map']),mask=s['mask'],
            parallel_launches=len(p['kernels']),worker_compute_CPU_ms=float(sum(thread))/1e6,
            sum_launch_max_worker_ms=float(total)/1e6,max_thread_summed_CPU_ms=float(max(thread))/1e6,
            dependency_vs_total_difference_ms=float(total-max(thread))/1e6,
            largest_launch_ms=float(largest[0])/1e6,
            two_largest_launch_share=float(sum(largest[:2])/total) if total else 0,
            per_thread_compute_ms=[float(x)/1e6 for x in thread],
            intrinsic_CPU_only_not_pipeline_elapsed=True))
    return result


def analyze(output=OUTPUT):
    output=Path(output);plan=read(output/'PLAN.json');seal=read(output/'REPLAY_FREEZE.json')
    if seal['sources']!=sources() or seal['plan_sha256']!=sha256(output/'PLAN.json'):
        raise ValueError('revision freeze changed')
    iso=validate_isolated(read(output/'ISOLATED_INPUTS.json'))
    native_path=TRACE/'analysis/RESULTS.json'
    if sha256(native_path)!=read(TRACE/'DELIVERY.json')['results_sha256']:raise ValueError('native corrected trace differs')
    native=read(native_path);frames=set(native['complete_front_frames'])
    comparison=[];summary=[];intrinsics=[]
    for case,value in iso['cases'].items():intrinsics+=intrinsic_summary(case,value)
    for name,h in seal['files'].items():
        if sha256(output/name)!=h:raise ValueError('model replay changed')
        item=read(output/name);case=item['spec']['case'];r=item['result']
        front=[t for t in r['tasks'] if t['role']=='stage' and t['stage']==0 and t['operation']=='worker_compute']
        last=defaultdict(float)
        joins=[k for k in r['kernel_joins'] if k['stage']==0]
        for k in joins:
            for s in k['last_slots']:last[str(s)]+=1/len(k['last_slots'])
        summary.append(dict(case=case,placement=item['spec']['placement'],
            front_service_ms=r['replay']['stage_service_ns'][0]/1e6,
            completion_interval_ms=r['replay']['interval_ns']/1e6,
            front_last_slot_counts=dict(last),
            front_completion_wait_CPU_ms=sum(float(rational(t['CPU'])) for t in r['tasks'] if t['role']=='stage' and t['stage']==0 and t['operation']=='completion_wait')/16/1e6,
            model_first_service_is_instant_for_all_runnable_tasks=True,
            isolated_inputs_only=True))
        if case!='yolov3_tiny416/overlap':continue
        for ordinal in range(12):
            model=[t for t in front if t['kernel']==ordinal and t['frame'] in frames]
            if len(model)!=len(frames)*4:raise ValueError('model worker coverage differs')
            actual=[t for t in native['worker_rows'] if t['kernel']==ordinal]
            for slot in range(4):
                a=[t for t in actual if t['slot']==slot];m=[t for t in model if t['slot']==slot]
                if {t['frame'] for t in a}!={t['frame'] for t in m}:raise ValueError('matched frame set differs')
                comparison.append(dict(placement=item['spec']['placement'],kernel=ordinal,slot=slot,frames=len(a),
                    model_CPU_ms=mean(float(rational(t['CPU'])) for t in m)/1e6,
                    model_ideal_service_deprivation_ms=mean(float(rational(t['ideal_service_deprivation'])) for t in m)/1e6,
                    native_CPU_clock_ms=mean(t['CPU_clock_ns'] for t in a)/1e6,
                    native_preempt_off_ms=mean(t['preempt_off_ns'] for t in a)/1e6,
                    native_TASKSTATS_queue_lower_ms=mean(t['queue_lower_ns'] for t in a)/1e6,
                    native_TASKSTATS_queue_upper_ms=mean(t['queue_upper_ns'] for t in a)/1e6,
                    native_launch_to_callback_ms=mean(t['launch_to_body_ns'] for t in a)/1e6,
                    model_ready_to_first_service_ms=mean(t['first_service_ns']-t['ready_ns'] for t in m)/1e6,
                    model_deprivation_is_not_native_runqueue=True))
    result=dict(protocol=protocol(),formula_inventory=plan['formulas'],model_summary=summary,
        matched_kernel_workers=comparison,intrinsic_workload=intrinsics,
        native_reference_sha256=sha256(native_path),matched_native_frames=sorted(frames),
        new_native_sessions=0,new_DNN_inferences=0,score_changed=False,
        no_correction_accepted_yet=True,goal_complete=False,recorded_at=now())
    dest=output/'analysis';dest.mkdir(exist_ok=True)
    write(dest/'RESULTS.json',result)
    for filename,rows in [('model_summary.csv',summary),('matched_kernel_workers.csv',comparison),('intrinsic_workload.csv',intrinsics)]:
        csv_write(dest/filename,rows)
    render(dest,result)
    REPORT.mkdir(parents=True,exist_ok=True)
    for p in dest.iterdir():
        if p.is_file():shutil.copy2(p,REPORT/p.name)
    write(output/'DELIVERY_AUDIT.json',dict(status='INITIAL_AUDIT_COMPLETE_GOAL_ACTIVE',
        files={str(p):sha256(p) for p in REPORT.iterdir() if p.is_file()},sources=sources(),
        goal_complete=False,new_native_sessions=0))
    progress=read(output/'GOAL_PROGRESS.json');progress.update(phase='matched-window model/native alignment complete',
        last_goal_turn='progress: frozen formula inventory and nine model replays with exact baseline invariance',
        next_action='derive minimal scheduler/host-phase hypothesis using workload and per-kernel evidence',updated_at=now())
    write(output/'GOAL_PROGRESS.json',progress,replace=True)
    return dict(report=str(REPORT/'FINDINGS_zh.txt'),replays=len(summary),compared_worker_groups=len(comparison),goal_complete=False)


def render(dest,r):
    lines=['静态评分与时序回放核查','',
        '本轮仅离线核查，没有新增推理、修改评分或裁剪搜索域。Goal仍在进行。',
        '正式主模型为非均匀线程份额、主机CPU需求及线程组服务约束，主机集中约束关闭。时序回放是另一层候选预测，不等同于正式静态容量公式。','',
        '九项48帧回放与原模型逐项保持时间戳、阶段均值和资源记账相同，只恢复逐任务轨迹。',
        '两个YOLO配置和ResNet参照均使用各自的真实隔离工作量。','',
        '同窗口回放结果：配置，放置规则，前段毫秒，完成间隔毫秒，前段合成完成等待核毫秒']
    for x in r['model_summary']:
        lines.append(f"{x['case']} | {x['placement']} | {x['front_service_ms']:.3f} | {x['completion_interval_ms']:.3f} | {x['front_completion_wait_CPU_ms']:.3f}")
    lines+=['','隔离前段工作量：']
    for x in r['intrinsic_workload']:
        if x['stage']==0:
            lines.append(f"{x['case']}：{x['parallel_launches']}次并行启动，最重启动的最大worker工作{x['largest_launch_ms']:.3f}毫秒，最重两次占逐次最大worker工作之和的{x['two_largest_launch_share']:.1%}。")
    lines+=['','解释与限制：',
        '真实轨迹包含首次排队、抢占与同核竞争；模型中的连续服务缺额不是Linux排队计数，表中分别命名。',
        '所有worker在模型内立即获得正CPU份额；首次排队恒为零来自模型构造，不能据此解释为实际没有唤醒或调度延迟。',
        '不同放置规则仍使用连续公平分摊。仅匹配平均耗时不证明完成次序、主机分布或caller等待正确。',
        '逐kernel/slot采用同一组12个完整帧，未使用截断的旧轨迹做全窗口归因。',
        '控制工作在本轮可部署隔离回放中仍未建模，未把已有流水控制时间混入预测。',
        '已有结果属于开发回放；修正必须另行冻结假设、预测和未见配置验证，不能宣称本轮已提高吞吐。']
    (dest/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')
