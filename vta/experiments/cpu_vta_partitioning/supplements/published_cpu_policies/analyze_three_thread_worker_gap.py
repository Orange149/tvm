"""One critical worker per launch: an exact wall path, not summed thread waits."""
from collections import Counter
from pathlib import Path
import json
import shutil
import statistics as st

from run_three_thread_worker_gap import ROOT,POOL,RUN,selection,protocol
from run_three_thread_worker_gap_compat import verified_receipts
import run_three_thread_host_gap as clean
from cpu_vta_partitioning.common import read,write,sha256,verify_inventory
from dependency_audit.trace import load_trace
from scheduler_gap_audit.curves import csv_write
from scheduler_gap_audit.analysis import call_details
from phase_model.simulator import rational

REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_three_thread_worker_gap'


def critical_path(launch):
    starts=launch['worker_wall_start_ns'];ends=launch['worker_wall_end_ns'];cpu=launch['worker_cpu_ns']
    if not starts or len(starts)!=len(ends) or len(starts)!=len(cpu):raise ValueError('incomplete worker vectors')
    last=max(range(len(ends)),key=lambda i:(ends[i],-i))
    entry=starts[last]-launch['wall_start_ns'];tail=launch['wall_end_ns']-ends[last]
    span=ends[last]-starts[last];nonrunning=span-cpu[last]
    if min(entry,tail,span,cpu[last])<0 or nonrunning < -10000:
        raise ValueError('critical path escaped launch or clocks inconsistent')
    if entry+cpu[last]+nonrunning+tail!=launch['wall_end_ns']-launch['wall_start_ns']:
        raise ValueError('critical path does not conserve launch wall time')
    return dict(last_worker=last,entry_delay_ns=entry,callback_CPU_ns=cpu[last],
        callback_wall_minus_CPU_ns=nonrunning,return_tail_ns=tail,
        signed_clock_residual_not_Linux_queue_measurement=True)


def parse_front(receipt,bundle):
    task=receipt['task'];raw=Path(receipt['raw']);program=read(bundle/'plans/programs.json')['cpu_00_02']
    trace=load_trace(raw,program['files']['module.so']['sha256'])
    total=sum(task[k] for k in ('warmup','scored','tail'))
    if not trace['frame_mapping'] or len(trace['frames'])!=total:raise ValueError('front trace lacks full frame mapping')
    if trace['threads']!=3 or trace['core_map']!=[0,1,2] or len(trace['sequence'])!=12:
        raise ValueError('target group or kernel sequence differs')
    phases=[json.loads(s) for s in (raw/'phase_trace.jsonl').read_text().splitlines()]
    runs={p['call']:p for p in phases if p['stage']=='cpu_00_02' and p['operation']=='run'}
    if set(runs)!=set(range(total)):raise ValueError('missing or duplicate phase run')
    frames=[];kernels=[]
    for f in trace['frames'][task['warmup']:task['warmup']+task['scored']]:
        phase=runs[f['frame_id']];group=f['launches']
        if phase['tid']!=trace['caller_tid'] or any(not phase['wall_start_ns']<=k['wall_start_ns']<=k['wall_end_ns']<=phase['wall_end_ns'] for k in group):
            raise ValueError('ordinal mapping disagrees with exact same-clock run boundary')
        pieces=[]
        for index,k in enumerate(group):
            p=critical_path(k);pieces.append(p)
            kernels.append(dict(frame=f['frame_id'],kernel=index,function_offset=k['function_offset'],
                callback_CPU_ms_by_slot=[v/1e6 for v in k['worker_cpu_ns']],
                callback_wall_ms_by_slot=[v/1e6 for v in k['worker_wall_ns']],
                launch_ms=k['launch_wall_ns']/1e6,caller_completion_CPU_ms=k['caller_tail_cpu_ns']/1e6,
                caller_completion_wall_ms=k['caller_tail_wall_ns']/1e6,**p))
        phase_wall=phase['wall_end_ns']-phase['wall_start_ns']
        outside=phase_wall-sum(k['launch_wall_ns'] for k in group)
        if outside<0:raise ValueError('overlapping launches double counted')
        values={key:sum(k[key] for k in pieces) for key in ('entry_delay_ns','callback_CPU_ns','callback_wall_minus_CPU_ns','return_tail_ns')}
        if outside+sum(values.values())!=phase_wall:raise ValueError('run wall accounting not conserved')
        frames.append(dict(frame=f['frame_id'],run_wall_ms=phase_wall/1e6,
            run_caller_CPU_ms=(phase['cpu_end_ns']-phase['cpu_start_ns'])/1e6,
            outside_launch_wall_ms=outside/1e6,**{k.replace('_ns','_ms'):v/1e6 for k,v in values.items()},
            callback_CPU_ms_by_slot=[sum(k['worker_cpu_ns'][i] for k in group)/1e6 for i in range(3)],
            caller_completion_CPU_ms=sum(k['caller_tail_cpu_ns'] for k in group)/1e6,
            caller_completion_wall_ms=sum(k['caller_tail_wall_ns'] for k in group)/1e6))
    fields=('run_wall_ms','run_caller_CPU_ms','outside_launch_wall_ms','entry_delay_ms','callback_CPU_ms',
        'callback_wall_minus_CPU_ms','return_tail_ms','caller_completion_CPU_ms','caller_completion_wall_ms')
    average={k:st.mean(f[k] for f in frames) for k in fields}
    average['all_worker_callback_CPU_ms']=sum(st.mean(f['callback_CPU_ms_by_slot'][i] for f in frames) for i in range(3))
    average['callback_CPU_ms_by_slot']=[st.mean(f['callback_CPU_ms_by_slot'][i] for f in frames) for i in range(3)]
    return dict(sequence=trace['sequence'],frames=frames,kernels=kernels,average=average,
        last_workers=dict(Counter(k['last_worker'] for k in kernels)),
        kernel_count=len(kernels),exact_same_clock_run_enclosure_checked=True,
        startup_launches_excluded=trace['non_target_launches_by_module'])


def analyze():
    receipts=verified_receipts();s=selection();bundle=POOL/'yolov3_tiny416/capsule'
    observed={};evidence={};baseline={}
    for p,r in receipts:
        evidence[str(p)]=sha256(p)
        if r['task']['spec']['phase']=='observed':
            key=r['task']['spec']['scope']+'/'+r['task']['spec']['case']
            observed[key]=dict(receipt=r,front=parse_front(r,bundle))
    for key,item in s['references'].items():
        p=Path(item['receipt'])
        if sha256(p)!=item['sha256']:raise ValueError('clean reference changed')
        r=read(p);verify_inventory(r['raw'],r['files'])
        if clean.summary(r['raw'],r['task'],clean.POOL/'yolov3_tiny416/capsule')!=r['summary']:
            raise ValueError('clean reference no longer replays')
        if r['boot_id']!=s['expected_boot_id']:raise ValueError('clean and observed boot differ')
        baseline[key]=r;evidence[str(p)]=sha256(p)
    iso=observed['component/3'];iso_clean=baseline['component/3']['summary']['cost']['service_ms']
    iso_probe=iso['receipt']['summary']['cost']['service_ms']
    rows=[];kernels=[];frame_rows=[];queue_rows=[];host_rows=[];model_rows=[]
    for key,data in observed.items():
        role='isolated' if key=='component/3' else s['roles'][key.split('/')[1]]
        front=data['front'];a=front['average'];raw=data['receipt']['summary']
        clean_service=iso_clean if role=='isolated' else baseline[key]['summary']['stage_service_ms'][0]
        probe_service=iso_probe if role=='isolated' else raw['stage_service_ms'][0]
        gap=clean_service-iso_clean
        disturbance=max(abs(probe_service-clean_service),abs(iso_probe-iso_clean))
        full_gap_limit=0.2*abs(gap)
        limit=min(full_gap_limit,protocol()['heavy_case_quantification_limit_ms'])
        calls,_,phases=call_details(data['receipt']['raw'],data['receipt']['task'])
        front_calls=[c for c in calls if c['stage']=='cpu_00_02']
        for c in front_calls:queue_rows.append(dict(role=role,**c))
        fields=('sum_last_finisher_queue_lower_ms','sum_last_finisher_queue_upper_ms',
            'sum_last_finisher_pre_entry_counter_ms','caller_completion_queue_counter_ms')
        row=dict(role=role,clean_front_service_ms=clean_service,probe_front_service_ms=probe_service,
            probe_front_disturbance_ms=probe_service-clean_service,
            max_component_pipeline_disturbance_ms=disturbance,
            quantitative_comparison_limit_ms=limit if role!='isolated' else None,
            quantitative_comparison_supported=role!='isolated' and gap>0 and disturbance<=limit,
            original_full_gap_20pct_limit_ms=full_gap_limit if role!='isolated' else None,
            full_gap_evidence_quality_supported=role!='isolated' and gap>0 and disturbance<=full_gap_limit,
            disturbance_over_clean_slowdown=disturbance/gap if gap>0 else None,
            observed_FPS=raw.get('fps'),clean_FPS=baseline[key]['summary'].get('fps'),
            observed_frames=len(front['frames']),observed_kernels=front['kernel_count'],
            last_workers=front['last_workers'],**a,
            **{k:st.mean(c[k] for c in front_calls) for k in fields})
        rows.append(row)
        if role!='isolated':
            t=data['receipt']['task'];selected=[v for v in phases if t['warmup']<=v['call']<t['warmup']+t['scored']]
            model_path=clean.POOL/'existing_model_diagnostic'/(t['configuration']['id']+'.json')
            old=read(model_path)['result']
            for stage,index in [('vta_03_06',1),('vta_09_12',3),('_producer',None),('_collector',None)]:
                group=[v for v in selected if v['stage']==stage]
                modeled=sum(float(rational(v['CPU'])) for v in old['actor_totals'] if v['stage']==index)/t['scored']/1e6 if index is not None else None
                host_rows.append(dict(role=role,stage=stage,
                    observed_thread_CPU_ms_per_frame=sum(v['cpu_end_ns']-v['cpu_start_ns'] for v in group)/t['scored']/1e6,
                    model_CPU_ms_per_frame=modeled,control_cost_unmodeled=index is None,
                    thread_clock_not_identical_to_scheduler_switch_span=True))
            tasks={(v['frame'],v['kernel'],v['slot']):v for v in old['tasks'] if v['stage']==0 and v['operation']=='worker_compute' and v['generation']==0}
            joins={(v['frame'],v['kernel']):v for v in old['kernel_joins'] if v['stage']==0}
            ready=sum(float(rational(tasks[(k['frame'],k['kernel'],k['last_worker'])]['ready_not_running'])) for k in front['kernels'])/t['scored']/1e6
            agreement=sum(k['last_worker'] in joins[k['frame'],k['kernel']]['last_slots'] for k in front['kernels'])/len(front['kernels'])
            model_rows.append(dict(role=role,source=str(model_path),source_sha256=sha256(model_path),
                model_front_ms=old['replay']['stage_service_ns'][0]/1e6,
                same_native_chosen_slot_model_ready_not_running_ms=ready,
                last_worker_identity_agreement=agreement,
                native_callback_queue_lower_ms=row['sum_last_finisher_queue_lower_ms'],
                native_callback_queue_upper_ms=row['sum_last_finisher_queue_upper_ms'],
                native_pre_entry_queue_counter_ms=row['sum_last_finisher_pre_entry_counter_ms'],
                model_ready_wait_and_native_queue_have_different_markers=True,
                differences_not_added_as_static_penalties=True))
        if front['sequence']!=iso['front']['sequence']:raise ValueError('pipeline kernel sequence differs from isolation')
        for index in range(12):
            matching=[v for v in front['kernels'] if v['kernel']==index]
            old=[v for v in iso['front']['kernels'] if v['kernel']==index]
            vector=[st.mean(v['callback_CPU_ms_by_slot'][i] for v in matching) for i in range(3)]
            reference=[st.mean(v['callback_CPU_ms_by_slot'][i] for v in old) for i in range(3)]
            kernels.append(dict(role=role,kernel=index,function_offset=matching[0]['function_offset'],
                CPU_ms_by_slot=vector,isolated_CPU_ms_by_slot=reference,
                CPU_difference_ms_by_slot=[a-b for a,b in zip(vector,reference)],
                launch_ms=st.mean(v['launch_ms'] for v in matching),
                entry_delay_ms=st.mean(v['entry_delay_ns'] for v in matching)/1e6,
                last_worker_counts=dict(Counter(v['last_worker'] for v in matching))))
        frame_rows += [dict(role=role,**f) for f in front['frames']]
    rows.sort(key=lambda r:('isolated','host_on_heavier_worker','host_outside_front_group').index(r['role']))
    out=dict(rows=rows,kernels=kernels,queue_calls=queue_rows,host_and_control=host_rows,model_diagnostic=model_rows,
        source_receipts=evidence,source_sha256=sha256(__file__),
        boot_id=s['expected_boot_id'],new_DNN=58,new_CPU_stage_calls=11,new_native_sessions=6,
        clean_references_not_new_measurements=True,no_pipeline_FPS_fit=True,
        exact_one_critical_worker_wall_path=True,
        CPU_clocks_are_not_pure_arithmetic=True,
        callback_wall_minus_CPU_is_not_CPU_queue_counter=True,
        completion_tail_overlaps_critical_worker_path_do_not_add=True,
        no_new_static_model=True,no_complete_search=True)
    dest=POOL/'analysis';write(dest/'RESULTS.json',out,replace=True)
    csv_write(dest/'front_comparison.csv',rows);csv_write(dest/'kernels.csv',kernels);csv_write(dest/'frames.csv',frame_rows)
    csv_write(dest/'queue_calls.csv',queue_rows);csv_write(dest/'host_and_control.csv',host_rows)
    csv_write(dest/'model_comparison.csv',model_rows)
    labels=dict(isolated='前段隔离',host_on_heavier_worker='主机固定在重worker核心1',host_outside_front_group='主机固定在前段核组外核心3')
    by={r['role']:r for r in rows};a=by['isolated'];b=by['host_on_heavier_worker']
    lines=['YOLO三线程前段慢化：实际CPU工作、CPU排队与完成等待','',
        '本轮是已知布局的机制诊断，不是新评分、DP或Top20验证。没有重启开发板、修改运行库或执行机制。',
        '固定CPU0–2→VTA3–6→CPU7–8→VTA9–12→CPU13，CPU配置3线程012／1线程3／1线程3，门控关闭，控制线程共享四核。',
        '复用上一轮同启动、同程序的关闭探针参照；新增6个原生会话、58次完整推理和11次隔离CPU调用，包括观察模式自身的资格检查。观察窗口未增加。','',
        '一、最主要的新证据',
        f"三个worker的回调CPU核时合计：隔离{a['all_worker_callback_CPU_ms']:.3f}，重worker共核{b['all_worker_callback_CPU_ms']:.3f}核毫秒，变化{100*(b['all_worker_callback_CPU_ms']/a['all_worker_callback_CPU_ms']-1):+.3f}%。这不是纯算术指令计数，但没有看到与约65ms墙钟慢化相当的CPU工作量增长。",
        f"每次只看最后完成的worker：回调期间CPU排队估计范围从隔离{a['sum_last_finisher_queue_lower_ms']:.3f}–{a['sum_last_finisher_queue_upper_ms']:.3f}ms，增至{b['sum_last_finisher_queue_lower_ms']:.3f}–{b['sum_last_finisher_queue_upper_ms']:.3f}ms/帧。它与回调墙钟减CPU钟的约{b['callback_wall_minus_CPU_ms']:.3f}ms相符，支持CPU竞争导致关键worker拖尾。",
        '这里没有把所有worker等待相加：不同并行启动顺序执行，每次只保留一名最后完成者。入口前的CPU排队计数另列，因包含较宽读取窗口与可能先前未结算等待，不能直接当作同一时间段的精确拆分。','',
        '二、探针质量限制必须保留',
        '方案 | 关闭探针前段ms | 开启探针前段ms | 差ms | 原全慢化20%条件 | 剩余误差精确归因条件']
    for r in rows:
        lines.append(f"{labels[r['role']]} | {r['clean_front_service_ms']:.3f} | {r['probe_front_service_ms']:.3f} | {r['probe_front_disturbance_ms']:+.3f} | {r['full_gap_evidence_quality_supported']} | {r['quantitative_comparison_supported']}")
    lines += [f"重worker方案探针扰动{b['max_component_pipeline_disturbance_ms']:.3f}ms，小于原全部慢化64.583ms的20%，支持上述粗粒度机制证据；但超过预先冻结的剩余22.897ms×20%＝4.579ms阈值，不精确声称本批已经归完剩余23ms。", 
        '核组外参照的原慢化只有4.171ms，探针扰动占比更大，其细分数据只作为辅助线索。没有删除正式性能或自动追加测量。','',
        '三、沿一条完成路径拆分run墙钟（有探针记录）',
        '每个并行启动取最后完成worker，严格使用：入口延迟＋该worker CPU时间＋该worker墙钟减CPU时间＋最后worker完成后的返回时间。',
        '再加并行启动之外的run墙钟，五项严格重建run总墙钟；不把调用线程的完成等待再次相加。',
        '方案 | 启动外墙钟 | 关键worker入口延迟 | 关键worker CPU | 关键worker墙钟减CPU | 返回尾部 | run总墙钟']
    for r in rows:
        lines.append(f"{labels[r['role']]} | {r['outside_launch_wall_ms']:.3f} | {r['entry_delay_ms']:.3f} | {r['callback_CPU_ms']:.3f} | {r['callback_wall_minus_CPU_ms']:.3f} | {r['return_tail_ms']:.3f} | {r['run_wall_ms']:.3f}")
    lines += ['最后完成者的CPU时间会受其身份变化影响，不能把其差值直接叫作工作量增加；工作量比较另按相同内核、相同逻辑线程列在kernels.csv。',
        f"调用线程的完成尾部约{b['caller_completion_wall_ms']:.3f}ms，CPU约{b['caller_completion_CPU_ms']:.3f}核毫秒，主要与其他worker执行及排队重叠，已经处于同一个run窗口内。不能再往阶段耗时加约69ms。",'',
        '四、为什么现有时序回放仍偏乐观',
        '方案 | 回放前段ms | 同原生最后worker槽位的模型ready等待ms | 原生回调CPU排队范围ms | 原生入口前计数增长ms | 最后worker身份一致率']
    for r in model_rows:
        lines.append(f"{labels[r['role']]} | {r['model_front_ms']:.3f} | {r['same_native_chosen_slot_model_ready_not_running_ms']:.3f} | {r['native_callback_queue_lower_ms']:.3f}–{r['native_callback_queue_upper_ms']:.3f} | {r['native_pre_entry_queue_counter_ms']:.3f} | {100*r['last_worker_identity_agreement']:.1f}%")
    lines += ['模型已包含部分等待，不能再把实测约58ms加到它上面。模型ready等待、原生回调排队与入口前计数的起止标记不完全相同，上表用于定位差别，不把三者相减后直接作为修正系数。',
        '现有有限服务模型是根据受控任务标定的近似，不是实际Linux CFS。其短周期主机唤醒、队列状态和任务历史处理仍可能让关键worker过早得到CPU；本批尚不足以确认哪一条调度规则造成全部剩余误差。','',
        '五、也检查了主机与控制工作，避免把问题全推给调度',
        '方案 | 工作 | 实测线程CPU/帧 | 回放CPU/帧']
    for r in host_rows:
        v='未建模' if r['model_CPU_ms_per_frame'] is None else f"{r['model_CPU_ms_per_frame']:.3f}"
        lines.append(f"{labels[r['role']]} | {r['stage']} | {r['observed_thread_CPU_ms_per_frame']:.3f} | {v}")
    lines += ['主机时序profile、主机实际CPU工作及结果收集开销仍有近似。线程CPU钟、调度切换区间的计量边界也不同，不能直接以表中差额证明少算了同样多的物理CPU服务，更不能将结果收集CPU时间直接加到完成间隔。',
        '六、本轮解决和没有解决的部分',
        '已支持：在这个重worker共核配置中，隔离到流水的较大慢化主要表现为关键worker的CPU排队和完成拖尾，而不是整体回调CPU工作增加几十毫秒。',
        '未完成：精确归因剩余约23ms、建立可用于未测候选的共享调度修正、验证完整域Top20选点收益。没有采用新的静态模型。',
        '下一步优先核对回放中关键worker的可运行与获得CPU时序，以及主机CPU与未建模控制工作的口径；不能重新用总核时除核心数解释已观测的拖尾，也不因这次归因继续无止境增加调度模型。','',
        '方法依据与范围：Pipe-it用固定计算核组组织流水；GCAPS区分CPU主机工作与独立设备工作。本项目复用这些资源语义，但普通Linux线程的共享调度并不自动等于论文假设下的可预测调度。CPU排队来自已通过能力核验的TASKSTATS计数，不把普通睡眠称为CPU排队。',
        'Pipe-it：https://arxiv.org/pdf/1903.05898',
        'GCAPS：https://drops.dagstuhl.de/storage/00lipics/lipics-vol298-ecrts2024/LIPIcs.ECRTS.2024.14/LIPIcs.ECRTS.2024.14.pdf',
        'Linux delay accounting：https://www.kernel.org/doc/html/latest/accounting/delay-accounting.html','',
        '验收器恢复说明：原验收器未接受冻结执行器已有的_producer/feed和_collector/collect事件；复用既有严格验收逻辑后重新接收已结束的资格会话，未重复推理。任务身份和原生命令逐项核对不变，恢复凭据见VALIDATOR_COMPAT.json。',
        '所有原始记录和旧成果保持原身份。已有ResNet18约13FPS成果没有被本轮局部YOLO诊断替代，整体goal仍在继续。']
    (dest/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')
    REPORT.mkdir(parents=True,exist_ok=True)
    for p in dest.iterdir():
        if p.is_file():shutil.copy2(p,REPORT/p.name)
    return out


if __name__=='__main__':
    x=analyze();print(json.dumps({k:x[k] for k in ('rows','model_diagnostic','new_DNN','new_CPU_stage_calls')},ensure_ascii=False,indent=2))
