"""Only after all scenarios freeze: join old observations, never choose a fit."""
from collections import Counter,defaultdict
from fractions import Fraction
from pathlib import Path
import json
import shutil
import statistics as st
from cpu_vta_partitioning.common import read,sha256,write
from scheduler_gap_audit.curves import csv_write
from .common import PARENT,OUTPUT,REPORT,protocol
from .replay import validate


def ms(value):
    if isinstance(value,dict):value=Fraction(value['numerator'],value['denominator'])
    return float(value)/1e6


def kernel_summary(result):
    joins=[j for j in result['kernel_joins'] if j['stage']==0]
    # The last generation is the parallel-launch completion join.
    last={}
    for j in joins:
        key=(j['frame'],j['kernel'])
        if key not in last or last[key]['generation']<j['generation']:last[key]=j
    credit=defaultdict(Fraction);ties=0
    for j in last.values():
        ties+=len(j['last_slots'])>1
        for slot in j['last_slots']:credit[slot]+=Fraction(1,len(j['last_slots']))
    totals=result['actor_totals']
    tail=sum((Fraction(t['CPU']['numerator'],t['CPU']['denominator']) for t in totals
              if t['role']=='stage' and t['stage']==0 and t['operation']=='completion_wait'),Fraction(0))
    return dict(launches=len(last),tied_last_finishers=ties,
        last_finisher_credit={str(k):float(v) for k,v in sorted(credit.items())},
        max_ready_skew_ns=max((j['ready_skew_ns'] for j in last.values()),default=0),
        max_first_service_skew_ns=max((j['first_service_skew_ns'] for j in last.values()),default=0),
        synthetic_completion_CPU_ms_per_frame=ms(tail/64),
        simultaneous_ties_not_forced_to_highest_slot=True)


def analyze(output):
    output=Path(output);validate(output);seal=read(output/'REPLAY_FREEZE.json')
    # First normal decode of actual FPS/phase service in this package.
    delivery=read(PARENT/'DELIVERY.json')
    if sha256(PARENT/'analysis/RESULTS.json')!=delivery['results_sha256']:raise ValueError('old result identity changed')
    measured=read(PARENT/'analysis/RESULTS.json');actual={r['network']+'/'+r['case']:r for r in measured['cases']}
    observed_stages={(r['network']+'/'+r['case'],r['stage']):r for r in measured['stages']}
    observed_kernels=read(PARENT/'analysis/CRITICAL_TIMELINES.json')
    tails=read(PARENT/'analysis/COMPLETION_TAILS.json');iso=read(output/'ISOLATED_INPUTS.json')
    diag=read(output/'DIAGNOSTIC_CONTROL_INPUTS.json');items=[read(output/p) for p in seal['files']]
    flat=[];stage_table=[];primary=[];sensitivities=[];thread_table=[];control_scope=[]
    indexed={}
    for item in items:
        spec=item['spec'];case=spec['case'];r=item['result']['replay'];seen=actual[case]
        hosts=json.dumps(spec['host_assignment'],sort_keys=True)
        indexed[case,spec['control_mode'],hosts]=item
        front=seen['configuration']['stages'][0]['program'];obs=observed_stages[case,front]
        flat.append(dict(id=spec['id'],case=case,control_mode=spec['control_mode'],host_mode=spec['host_mode'],
            host_assignment=spec['host_assignment'],predicted_interval_ms=r['interval_ns']/1e6,
            predicted_front_ms=r['stage_service_ns'][0]/1e6,observed_front_ms=obs['clean_pipeline_service_ms'],
            front_error_ms=r['stage_service_ns'][0]/1e6-obs['clean_pipeline_service_ms'],
            observed_interval_ms=seen['clean_interval_ms'],interval_error_ms=r['interval_ns']/1e6-seen['clean_interval_ms'],
            synthetic_wait_CPU_ms=ms(item['result']['accounting']['synthetic_wait_CPU']),
            control_CPU_ms=ms(item['result']['accounting']['control_CPU']),
            diagnostics_not_deployable_prediction=spec['control_mode']!='zero'))
    pairs=[]
    for key,item in indexed.items():
        case,mode,hosts=key
        if mode!='zero':continue
        other=indexed[case,'diagnostic_observed',hosts]
        a=item['result']['replay'];b=other['result']['replay']
        pairs.append(dict(case=case,host_assignment=item['spec']['host_assignment'],
            front_change_ms=(b['stage_service_ns'][0]-a['stage_service_ns'][0])/1e6,
            interval_change_ms=(b['interval_ns']-a['interval_ns'])/1e6,
            not_causal_measured_contribution=True))
    for case,value in iso['cases'].items():
        seen=actual[case];base=indexed[case,'zero','null'];added=indexed[case,'diagnostic_observed','null']
        a=base['result']['replay'];b=added['result']['replay'];cfg=value['configuration']
        for index,s in enumerate(cfg['stages']):
            old=observed_stages[case,s['program']]
            stage_table.append(dict(case=case,stage=s['program'],device=s['device'],threads=s['threads'],mask=s['mask'],
                old_frozen_isolation_ms=old['isolated_service_ms'],base_prediction_ms=a['stage_service_ns'][index]/1e6,
                diagnostic_control_prediction_ms=b['stage_service_ns'][index]/1e6,
                measured_clean_service_ms=old['clean_pipeline_service_ms'],
                base_error_ms=a['stage_service_ns'][index]/1e6-old['clean_pipeline_service_ms'],
                diagnostic_error_ms=b['stage_service_ns'][index]/1e6-old['clean_pipeline_service_ms'],
                quantitative_probe_attribution_allowed=seen['quantitative_attribution_allowed'],
                CPU_probe_values_are_auxiliary_for_other_two_cases=not seen['quantitative_attribution_allowed']))
        observed_front=observed_stages[case,cfg['stages'][0]['program']]['clean_pipeline_service_ms']
        base_error=abs(a['stage_service_ns'][0]/1e6-observed_front);added_error=abs(b['stage_service_ns'][0]/1e6-observed_front)
        primary.append(dict(case=case,base_front_ms=a['stage_service_ns'][0]/1e6,diagnostic_front_ms=b['stage_service_ns'][0]/1e6,
            measured_front_ms=observed_front,front_absolute_error_reduction_ms=base_error-added_error,
            base_interval_ms=a['interval_ns']/1e6,diagnostic_interval_ms=b['interval_ns']/1e6,
            measured_interval_ms=seen['clean_interval_ms'],old_measured_FPS=seen['clean_FPS'],
            diagnostic_is_not_new_FPS=True,probe_quality=seen['quantitative_attribution_allowed']))
        controls=diag['cases'][case]
        for role,c in controls.items():control_scope.append(dict(case=case,role=role,CPU_ms_per_frame=ms(c['CPU_ns']),
            observed_wall_ms_per_frame=st.mean(c['wall_samples_ns'])/1e6,scope=c['scope'],
            unknown_other_control_work=True,not_isolated_profile=True))
        for mode in ('zero','diagnostic_observed'):
            fixed=[row for row in flat if row['case']==case and row['control_mode']==mode and row['host_mode']=='fixed_trajectory']
            sensitivities.append(dict(case=case,control_mode=mode,assignments=len(fixed),
                front_min_ms=min(r['predicted_front_ms'] for r in fixed),front_max_ms=max(r['predicted_front_ms'] for r in fixed),
                interval_min_ms=min(r['predicted_interval_ms'] for r in fixed),interval_max_ms=max(r['predicted_interval_ms'] for r in fixed),
                measured_front_ms=observed_front,
                measured_front_inside_sensitivity_range=min(r['predicted_front_ms'] for r in fixed)<=observed_front<=max(r['predicted_front_ms'] for r in fixed),
                strict_bound_claimed=False,no_closest_fit_assignment_selected=True))
            selected=indexed[case,mode,'null'];model=kernel_summary(selected['result'])
            network,name=case.split('/')
            real=[r for r in observed_kernels if r['network']==network and r['case']==name and r['scope']=='pipeline' and r['stage']==cfg['stages'][0]['program']]
            counts=Counter(r['last_worker'] for r in real)
            real_tail=st.mean(r['completion_tail_CPU_ms'] for r in tails if r['network']==network and r['scope']=='pipeline' and r['case']==name)
            thread_table.append(dict(case=case,control_mode=mode,**model,
                observed_last_finisher_counts=dict(sorted(counts.items())),observed_completion_tail_CPU_ms_per_frame=real_tail,
                observed_launch_skew_sum_ms_per_frame=st.mean(sum(r['launch_skew_ms'] for r in real if r['frame']==f) for f in range(8,72)),
                comparison_scopes='model synthetic waiting excludes real return overhead; ready skew is not Linux runqueue counter',
                probe_quality=seen['quantitative_attribution_allowed']))
    target=output/'analysis';target.mkdir(parents=True,exist_ok=True)
    for name,rows in [('all_198_scenarios.csv',flat),('paired_controls.csv',pairs),('stage_errors.csv',stage_table),
                      ('primary_comparison.csv',primary),('host_sensitivity.csv',sensitivities),
                      ('worker_completion.csv',thread_table),('control_scope.csv',control_scope)]:csv_write(target/name,rows)
    result=dict(primary=primary,host_sensitivity=sensitivities,worker_completion=thread_table,control_scope=control_scope,
        stage_errors=stage_table,scenario_count=len(items),new_native_sessions=0,new_inferences=0,
        no_static_predictor_correction_claimed=True,no_real_throughput_gain_claimed=True,
        performance_decoded_only_after_replay_freeze=True,replay_freeze_sha256=sha256(output/'REPLAY_FREEZE.json'))
    write(target/'ANALYSIS.json',result,replace=True)
    render(output,result,flat)
    REPORT.mkdir(parents=True,exist_ok=True)
    for p in target.iterdir():shutil.copy2(p,REPORT/p.name)
    for leaf in ('AUDIT.json','PLAN.json','INPUT_FREEZE.json','REPLAY_FREEZE.json','REPLAY_COST.json'):
        shutil.copy2(output/leaf,REPORT/leaf)
    return dict(report=str(REPORT/'README.md'),primary=primary,scenario_count=198,new_native_sessions=0)


def render(output,result,flat):
    target=Path(output)/'analysis';audit=read(Path(output)/'AUDIT.json');cost=read(Path(output)/'REPLAY_COST.json')
    lines=['# 结果处理与主机放置：198项离线语义回放','',
        '本轮未连接开发板、未新增推理，未修改执行器、隔离成本、评分、DP、Top-20或线程域。全部情形提前确定，回放冻结后才解码旧性能值。','',
        '## 直接结果','',
        '| 固定配置 | 原前段预测ms | 加控制CPU后的诊断预测ms | 旧流水实测ms | 前段绝对误差减少ms |',
        '|---|---:|---:|---:|---:|']
    names={'resnet18/static_first':'ResNet原模型静态第一名','yolov3_tiny416/overlap':'YOLO四线程重叠','yolov3_tiny416/split':'YOLO三线程分组'}
    for r in result['primary']:lines.append(f"| {names[r['case']]} | {r['base_front_ms']:.3f} | {r['diagnostic_front_ms']:.3f} | {r['measured_front_ms']:.3f} | {r['front_absolute_error_reduction_ms']:+.3f} |")
    main=next(r for r in result['primary'] if r['case']=='yolov3_tiny416/overlap')
    lines+=['',f"YOLO重叠方案加入已观测控制CPU后，前段预测变化{main['diagnostic_front_ms']-main['base_front_ms']:+.3f}ms，完成间隔预测变化{main['diagnostic_interval_ms']-main['base_interval_ms']:+.3f}ms。它衡量该理想回放对遗漏控制工作的敏感性，不是实际移除输出核验后的收益。",'',
        '**诊断输入包含旧流水中观测的控制CPU，不能作为可部署预测器、通用子图成本或DP转移。没有产生新的实测FPS。**','',
        '## 源码与计时范围','',
        '已核验实际冻结执行器源码和运行库SHA。defer-result-write只延后文件写入；ResultJSON仍在write_completed内逐帧调用输出遍历、哈希、统计及JSON构造。没有把这些成本都归给哈希。','',
        '| 对象 | 生产CPU ms/帧 | write_completed CPU ms/帧 |','|---|---:|---:|']
    for case in names:
        producer=next(r for r in result['control_scope'] if r['case']==case and r['role']=='_producer')
        collector=next(r for r in result['control_scope'] if r['case']==case and r['role']=='_collector')
        lines.append(f"| {names[case]} | {producer['CPU_ms_per_frame']:.4f} | {collector['CPU_ms_per_frame']:.4f} |")
    lines+=['','控制段外的析构、回收及其他开销保留未知。输入队列满、输出队列满及消费者等待均不作为CPU工作。结果完成时间仍在最后阶段完成、推入输出队列之前记录；收集CPU仅经共享资源和反压影响未来帧。','',
        '旧FPS保留逐帧输出统计开启口径；不能直接扣除9.5ms推算新FPS，也不据此宣布旧比较无效。两个网络的输出数量和尺寸不同，不能全部归因于网络结构。','',
        '## 主机固定轨迹敏感性','',
        '| 配置 | 控制口径 | 固定轨迹数 | 前段预测范围ms | 完成间隔范围ms | 实测前段在范围内 |',
        '|---|---|---:|---|---|---|']
    for r in result['host_sensitivity']:
        mode='原零控制' if r['control_mode']=='zero' else '诊断性补入'
        lines.append(f"| {names[r['case']]} | {mode} | {r['assignments']} | {r['front_min_ms']:.3f}–{r['front_max_ms']:.3f} | {r['interval_min_ms']:.3f}–{r['interval_max_ms']:.3f} | {r['measured_front_inside_sensitivity_range']} |")
    lines+=['','全部主机固定组合都在原允许集合内，配置执行身份未改变。它们是可能轨迹假设，不是实际绑核干预或严格上下界；没有选择最接近旧FPS的组合作为新参数。所有198项见all_198_scenarios.csv，99组控制匹配差见paired_controls.csv。','',
        '## 启动与完成依赖','',
        '| 配置 | 控制口径 | 模型就绪/首次服务错位 | 模型合成完成等待CPU ms/帧 | 旧观测完成尾部CPU ms/帧 | 同时最后完成的启动数 |',
        '|---|---|---|---:|---:|---:|']
    for r in result['worker_completion']:
        lines.append(f"| {names[r['case']]} | {r['control_mode']} | {r['max_ready_skew_ns']} / {r['max_first_service_skew_ns']} ns | {r['synthetic_completion_CPU_ms_per_frame']:.3f} | {r['observed_completion_tail_CPU_ms_per_frame']:.3f} | {r['tied_last_finishers']}/{r['launches']} |")
    lines+=['','模型仍让同一次启动的worker同时就绪并立即取得正比例服务，无法产生Linux意义上的首次排队。增加控制工作与改变主机轨迹没有改变这一结构。连续共享的服务缺额与TASKSTATS排队计数分别命名；模型等待也不包含实测尾部的全部返回开销。','',
        '模型并列最后完成者采用均分计数，不强行归给编号最大的线程。完整分布见worker_completion.csv；六项主对照的任务与join明细保留在各回放JSON中。','',
        '## 结论边界与下一步','',
        '- 结果处理开销是否重要，依据上述匹配回放差判断；模型中的变化不能代替真实测量干预。',
        '- 主机轨迹敏感性较大时，下一步需要真实落点／迁移证据；不能从这些范围挑出一个最佳拟合核心。',
        '- 若仍低估，下一项缺口是就绪、唤醒、调度粒度及线程历史状态；本轮没有声称重现CFS，也没有套用猜测时间片。',
        '- YOLO重叠方案的探针质量支持主要定量讨论；ResNet及YOLO分组的细分观测仅作辅助。',
        '- 三项均为已知配置的开发回放，不是未见配置预测验证，更没有新的真实吞吐增益。',
        '', 'Linux参考：[5.4 CFS说明](https://www.kernel.org/doc/html/v5.4/scheduler/sched-design-CFS.html)。供应商内核源码身份及实际调度粒度仍未知；历史运行库构建凭据没有完整源码哈希列表，已如实记录，不把当前源码直接认定为二进制的唯一来源。','',
        '## 成本与复现','',f"198项回放原始计算累计{cost['original_replay_seconds']:.2f}秒；本命令用时见REPLAY_COST.json。新增板端会话0、推理0、编译0。",'',
        '```bash','python run_model_semantics_audit.py run','python run_model_semantics_audit.py validate','```','',
        '新代码独立存放，旧共同Top-20和隔离组件胶囊仍通过校验。阶段、控制范围、198项敏感性和线程完成表均随报告交付。']
    (target/'README.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels=['ResNet reference','YOLO overlap','YOLO grouped'];x=np.arange(3);w=.24
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    for i,(field,label) in enumerate([('base_front_ms','Original replay'),('diagnostic_front_ms','With observed control CPU (diagnostic)'),('measured_front_ms','Existing clean pipeline measurement')]):
        ax.bar(x+(i-1)*w,[r[field] for r in result['primary']],w,label=label)
    ax.set(xticks=x,xticklabels=labels,ylabel='Front CPU-stage service ms',title='Offline diagnostic replay; no new hardware measurements')
    ax.legend(fontsize=8);fig.savefig(target/'front_stage_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
    for ax,(case,label) in zip(axes,zip(names,labels)):
        data=[[r['predicted_front_ms'] for r in flat if r['case']==case and r['control_mode']==mode and r['host_mode']=='fixed_trajectory'] for mode in ('zero','diagnostic_observed')]
        ax.boxplot(data,tick_labels=['Zero control','Diagnostic control'],showfliers=True)
        observed=next(r['measured_front_ms'] for r in result['primary'] if r['case']==case)
        ax.axhline(observed,color='tab:red',linestyle='--',label='Existing measurement');ax.set(title=label,ylabel='Front stage ms');ax.tick_params(axis='x',labelsize=8)
    axes[0].legend(fontsize=7);fig.savefig(target/'host_placement_sensitivity.png',dpi=170);plt.close(fig)
