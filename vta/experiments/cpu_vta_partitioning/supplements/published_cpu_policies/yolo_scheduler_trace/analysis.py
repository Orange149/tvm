from collections import Counter,defaultdict
from pathlib import Path
from statistics import mean
import json,shutil
from cpu_vta_partitioning.common import digest,read,sha256,verify_inventory,write,now
from scheduler_gap_audit.analysis import call_details,queue_bounds
from scheduler_gap_audit.curves import csv_write
from .common import ROOT,PARENT,REPORT,protocol,specs,sources
from .prepare import validate
from .runtime import make_task,summary
from .events import clip,Attribution


def measure_callback(kernel,worker,stream):
    body=clip(stream,worker['wall_start_ns'],worker['wall_end_ns'])
    before=clip(stream,kernel['start_ns'],worker['wall_start_ns'])
    lo,hi=queue_bounds(worker)
    return dict(frame=kernel['frame'],kernel=kernel['kernel'],launch=kernel['launch'],slot=worker['task'],tid=worker['tid'],
        module=kernel['module'].split('/programs/',1)[-1],function_offset=kernel['function_offset'],
        last=worker['task']==kernel['last_worker'],wall_ns=worker['wall_end_ns']-worker['wall_start_ns'],
        CPU_clock_ns=worker['cpu_end_ns']-worker['cpu_start_ns'],
        barrier_CPU_ns=worker['barrier_cpu_ns'],**{k:v for k,v in body.items() if k not in ('on','off')},
        CPU_clock_difference_ns=body['on_CPU_ns']-(worker['cpu_end_ns']-worker['cpu_start_ns']) if body['fully_covered'] else None,
        callback_start_ns=worker['wall_start_ns'],callback_end_ns=worker['wall_end_ns'],
        queue_lower_ns=lo,queue_upper_ns=hi,launch_to_body_ns=worker['wall_start_ns']-kernel['start_ns'],
        pre_body_on_CPU_ns=before['on_CPU_ns'],pre_body_off_CPU_ns=before['off_CPU_ns'],pre_body_unknown_ns=before['unknown_ns'],
        pre_body_includes_observer_and_entry_work=True,off_CPU_is_not_all_runnable_wait=True),body


def analyze(output,run_dir):
    output,run_dir=Path(output),Path(run_dir);validate(output);state=read(run_dir/'state.json')
    if state['status']!='COMPLETE' or state['accepted_sessions']!=5 or state['DNN_inferences']!=146:raise ValueError('native scheduler diagnostic incomplete')
    bundle=output/'capsule';items={};proofs={}
    for spec in specs():
        task=make_task(bundle,spec);p=run_dir/'accepted'/state['boot_id']/(task['task_id']+'.json');item=read(p)
        if item['task']!=task or item['boot_id']!=state['boot_id']:raise ValueError('scheduler receipt identity changed')
        verify_inventory(item['raw'],item['files'])
        if (Path(item['raw'])/'task_hash').read_text().strip()!=digest(task) or (Path(item['raw'])/'boot_id').read_text().strip()!=state['boot_id']:raise ValueError('raw native identity changed')
        if summary(item['raw'],task,bundle)!=item['summary']:raise ValueError('scheduler receipt no longer replays')
        items[spec['runner'],spec['phase']]=item;proofs[str(p)]=sha256(p)
    clean=items['phase','clean']['summary'];observed=items['phase','observed']['summary'];bridge=items['old','clean']['summary']
    if any(x['summary']['output_fingerprints']!=clean['output_fingerprints'] for x in items.values()):raise ValueError('full network output fingerprints differ')
    item=items['phase','observed'];doc=item['task'];scheduler=observed['scheduler'];streams=scheduler['streams']
    _,kernels,phases=call_details(item['raw'],doc);front=[k for k in kernels if k['stage']=='cpu_00_02']
    if len(front)!=doc['scored']*12:raise ValueError('expected all 12 front launches in each scored frame')
    attribute=Attribution(streams);workers=[];tails=[];off_rows=[];critical=[]
    covered_roles=defaultdict(int);covered_preempt=0;unobserved=0
    for kernel in front:
        for worker in kernel['worker_events']:
            row,body=measure_callback(kernel,worker,streams[str(worker['tid'])]);workers.append(row)
            if row['last']:
                critical.append(row)
                if row['fully_covered']:
                    for off in body['off']:
                        allocation=attribute.other_work(off,worker['task'])
                        off_rows.append(dict(frame=row['frame'],kernel=row['kernel'],last_slot=row['slot'],**off,**allocation))
                        if off['preempt']:
                            covered_preempt+=off['end_ns']-off['start_ns'];unobserved+=allocation['unobserved_ns']
                            for role,ns in allocation['by_role_ns'].items():covered_roles[role]+=ns
        event=kernel['launch_event'];a,b=event['caller_tail_wall_start_ns'],event['wall_end_ns']
        if not kernel['start_ns']<=a<=b:raise ValueError('caller completion-tail scope invalid')
        tail=clip(streams[str(event['caller'])],a,b)
        tails.append(dict(frame=kernel['frame'],kernel=kernel['kernel'],wall_ns=b-a,
            CPU_clock_ns=event['cpu_end_ns']-event['caller_tail_cpu_start_ns'],start_ns=a,end_ns=b,
            **{k:v for k,v in tail.items() if k not in ('on','off')},not_all_on_CPU_is_pure_spinning=True))
    complete_frames=[]
    for frame in range(doc['warmup'],doc['warmup']+doc['scored']):
        group=[w for w in workers if w['frame']==frame]
        if len(group)==48 and all(w['fully_covered'] for w in group):complete_frames.append(frame)
    parent_results=PARENT/'analysis/RESULTS.json';parent_delivery=read(PARENT/'DELIVERY.json')
    if sha256(parent_results)!=parent_delivery['results_sha256']:raise ValueError('original isolated reference changed')
    prior=read(parent_results);iso=next(r for r in prior['stages'] if r['network']=='yolov3_tiny416' and r['case']=='overlap' and r['stage']=='cpu_00_02')
    disturbance=abs(observed['stage_service_ms'][0]-clean['stage_service_ms'][0]);gap=clean['stage_service_ms'][0]-iso['isolated_service_ms']
    good=gap>0 and disturbance<=protocol()['probe_quality_fraction']*gap
    by_kernel=[]
    for ordinal in range(12):
        crit=[r for r in critical if r['kernel']==ordinal and r['fully_covered']]
        tail=[r for r in tails if r['kernel']==ordinal and r['fully_covered']]
        launch=[r for r in front if r['kernel']==ordinal]
        by_kernel.append(dict(kernel=ordinal,scored_calls=len(launch),covered_last_callbacks=len(crit),covered_tails=len(tail),
            launch_wall_ms=mean(r['end_ns']-r['start_ns'] for r in launch)/1e6,
            critical_on_ms=mean(r['on_CPU_ns'] for r in crit)/1e6 if crit else None,
            critical_preempt_off_ms=mean(r['preempt_off_ns'] for r in crit)/1e6 if crit else None,
            critical_nonpreempt_off_ms=mean(r['nonpreempt_off_ns'] for r in crit)/1e6 if crit else None,
            critical_TASKSTATS_queue_lower_ms=mean(r['queue_lower_ns'] for r in crit)/1e6 if crit else None,
            critical_TASKSTATS_queue_upper_ms=mean(r['queue_upper_ns'] for r in crit)/1e6 if crit else None,
            caller_tail_on_ms=mean(r['on_CPU_ns'] for r in tail)/1e6 if tail else None))
    result=dict(status='COMPLETE',boot_id=state['boot_id'],clean=clean,observed=observed,bridge=bridge,
        callback_rows=workers,critical_rows=critical,tail_rows=tails,kernel_means=by_kernel,
        covered_frames=complete_frames,representative_frame=complete_frames[0] if complete_frames else None,
        last_slot_counts=dict(Counter(r['slot'] for r in critical)),
        preempted_critical_off_ns=covered_preempt,coincident_work_by_role_ns=dict(covered_roles),unobserved_preempt_off_ns=unobserved,
        fully_covered_workers=sum(r['fully_covered'] for r in workers),fully_covered_critical=sum(r['fully_covered'] for r in critical),
        probe_front_disturbance_ms=disturbance,probe_FPS_change_percent=100*(observed['fps']/clean['fps']-1),
        quantitative_attribution_allowed=good,reference_isolated_front_ms=iso['isolated_service_ms'],
        reference_isolation_is_old_same_boot_not_new_calibration=True,
        event_coverage_does_not_provide_exact_wakeup=True,no_off_CPU_sum_added_to_II=True,
        new_native_sessions=5,DNN_inferences=146,new_component_calibrations=0,
        sources=sources(),source_receipts=proofs,parent_results_sha256=sha256(parent_results),state=state,recorded_at=now())
    dest=output/'analysis';dest.mkdir(parents=True,exist_ok=True);write(dest/'RESULTS.json',result)
    csv_write(dest/'workers.csv',workers);csv_write(dest/'critical_callbacks.csv',critical);csv_write(dest/'caller_tails.csv',tails)
    csv_write(dest/'kernel_means.csv',by_kernel);csv_write(dest/'critical_off_cpu_activity.csv',off_rows)
    render(dest,result);plot(dest,result,front,streams)
    REPORT.mkdir(parents=True,exist_ok=True)
    for p in dest.iterdir():shutil.copy2(p,REPORT/p.name)
    write(output/'DELIVERY.json',dict(status='COMPLETE',sources=sources(),new_native_sessions=5,DNN_inferences=146,
        results_sha256=sha256(dest/'RESULTS.json'),files={str(p):sha256(p) for p in REPORT.iterdir() if p.is_file()},
        no_reboot=True,score_changed=False,DP_changed=False,Top20_changed=False,recorded_at=now()))
    return dict(report=str(REPORT/'README.md'),events=scheduler['total_events'],covered_last_callbacks=result['fully_covered_critical'],
        scored_launches=len(critical),complete_frames=complete_frames,quantitative_attribution_allowed=good,
        coincident_work_by_role_ns=dict(covered_roles),unobserved_ns=unobserved)


def render(dest,r):
    lines=['# YOLO重叠配置：真实调度与并行完成时间线','',
        '本轮固定CPU0–2四线程核心0–3、CPU7–8三线程核心1–3、CPU13单线程核心0，主机及控制线程共享四核、CPU门控关闭。仅新增可选观测；不修改模块、运行库、原等待机制、静态评分、DP或Top-20。','',
        '5个原生会话、146次完整推理；两次单帧输出资格检查，三次8帧预热＋16个完成时间戳＋24帧尾部。16个时间戳按15个间隔计算FPS。未重启、未新增组件标定、未自动加测。','',
        '| 会话 | FPS | 完成间隔ms | 前段服务ms |', '|---|---:|---:|---:|']
    for label,key in [('新程序，探针关闭','clean'),('同一程序，探针开启','observed'),('原冻结程序桥接','bridge')]:
        v=r[key];lines.append(f"| {label} | {v['fps']:.4f} | {v['ii_ms']:.4f} | {v['stage_service_ms'][0]:.4f} |")
    lines+=['',f"探针FPS变化{r['probe_FPS_change_percent']:+.3f}%，前段服务扰动{r['probe_front_disturbance_ms']:.3f}ms；质量条件通过={r['quantitative_attribution_allowed']}。质量条件只影响诊断解释，不拒收正式性能或触发重测。",'',
        '## 调度覆盖','',f"捕获{r['observed']['scheduler']['total_events']}条切换记录。16帧各12次前段并行启动，共192次；完整覆盖最后完成者的回调{r['fully_covered_critical']}次，完整覆盖全部四个回调的帧为{r['covered_frames']}。边界缺口保留未知，不补成运行或排队。",
        f"最后完成者槽位分布：{r['last_slot_counts']}。例图使用第一个所有前段回调均被覆盖的帧{r['representative_frame']}，不选择最慢帧。",'',
        '| 内核序号 | 完整覆盖关键回调数 | 启动墙钟ms | 关键回调运行ms | 抢占离核ms | 非抢占离核ms | TASKSTATS排队范围ms | caller尾部运行ms |',
        '|---|---:|---:|---:|---:|---:|---|---:|']
    def f(v):return '未知' if v is None else f'{v:.3f}'
    for v in r['kernel_means']:
        q='未知' if v['critical_TASKSTATS_queue_lower_ms'] is None else f"{v['critical_TASKSTATS_queue_lower_ms']:.3f}–{v['critical_TASKSTATS_queue_upper_ms']:.3f}"
        lines.append(f"| {v['kernel']} | {v['covered_last_callbacks']} | {v['launch_wall_ms']:.3f} | {f(v['critical_on_ms'])} | {f(v['critical_preempt_off_ms'])} | {f(v['critical_nonpreempt_off_ms'])} | {q} | {f(v['caller_tail_on_ms'])} |")
    lines+=['', '每次启动只取一个最后完成者，不把四个worker的等待相加。表中不同内核完整覆盖数量可能不同，均值不能当作严格逐帧加法分解。caller尾部与worker运行／离核重叠，不能再相加到阶段耗时。',
        '', '## 关键worker被抢占期间，同核上运行了谁','',
        '只统计最后完成者且完整覆盖的回调中、带抢占标记的OUT→IN区间。同核其他目标线程的已观测运行片段按角色列出；它们是同时占用证据，不自动证明全部等待都由该角色造成。','',
        '| 角色 | 累计同核运行ms | 占所统计抢占离核时间 |','|---|---:|---:|']
    total=r['preempted_critical_off_ns']
    for role,ns in sorted(r['coincident_work_by_role_ns'].items(),key=lambda p:(-p[1],p[0])):
        lines.append(f"| {role} | {ns/1e6:.3f} | {100*ns/total if total else 0:.2f}% |")
    lines.append(f"| 未观测／边界等未知部分 | {r['unobserved_preempt_off_ns']/1e6:.3f} | {100*r['unobserved_preempt_off_ns']/total if total else 0:.2f}% |")
    lines+=['','## 解释限制','',
        '事件接口没有精确唤醒时刻。非抢占离核可能来自主动让出CPU、阻塞等，不能全部称为睡眠；离核也不能全部当作CPU排队。TASKSTATS排队单独列出并保留读取包络。',
        '回调记录体之前包含入口及TASKSTATS读数；caller完成尾部也包含返回及观察代码。运行片段不等于纯算术，尾部在CPU上也不等于全部忙等。',
        '这里只记录12个已识别目标TID，不是全系统任务与中断的完整轨迹。未知部分不强行归到某个竞争者。',
        '正式FPS取探针关闭会话。探针扰动超出质量条件时只保留线索，不作定量归因、不自动重测。单输入、单启动、单配置不支持稳定收益或显著性结论。',
        '本轮不将实测调度片段、等待或慢化比例写回通用静态成本，不自动修改模型或启动Top-20。',
        '', '全部原始文件、角色／绑核、8个输出、帧、程序与boot身份均在验收中核验。接口依据：[perf_event_open](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)。']
    (dest/'README.md').write_text('\n'.join(lines)+'\n')


def plot(dest,r,kernels,streams):
    if r['representative_frame'] is None:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=r['representative_frame'];chosen=[k for k in kernels if k['frame']==frame]
    start=min(k['start_ns'] for k in chosen);end=max(k['end_ns'] for k in chosen)
    fig,axes=plt.subplots(2,1,figsize=(13,6),layout='constrained')
    colors={'CPU_stage':'tab:blue','VTA_host':'tab:red','control':'tab:green'}
    for stream in streams.values():
        role=stream['role']
        for span in stream['on']:
            a=max(start,span['start_ns']);b=min(end,span['end_ns'])
            color='tab:orange' if role['kind']=='CPU_stage' and role['stage']!=0 else colors[role['kind']]
            if a<b:axes[0].broken_barh([((a-start)/1e6,(b-a)/1e6)],(span['cpu']-.3,.6),facecolors=color)
    axes[0].set(yticks=range(4),ylabel='CPU core',xlabel='ms since selected front launch',title=f'Frame {frame}: blue front CPU, orange other CPU, red VTA host, green control')
    for k in chosen:
        for worker in k['worker_events']:
            slot=worker['task'];a=worker['wall_start_ns'];b=worker['wall_end_ns']
            axes[1].broken_barh([((a-start)/1e6,(b-a)/1e6)],(slot-.3,.6),facecolors='#dce0e5')
            for span in clip(streams[str(worker['tid'])],a,b)['on']:
                axes[1].broken_barh([((span['start_ns']-start)/1e6,(span['end_ns']-span['start_ns'])/1e6)],(slot-.3,.6),facecolors='tab:blue')
        axes[1].text((k['start_ns']-start)/1e6,3.4,str(k['kernel']),fontsize=7)
    axes[1].set(yticks=range(4),ylabel='Front logical worker',xlabel='ms since selected front launch',title='Callback envelopes (gray) and running intervals (blue); kernel ordinals above')
    axes[0].set_ylim(-.7,3.8);axes[1].set_ylim(-.7,4)
    fig.savefig(dest/'representative_timeline.png',dpi=180);plt.close(fig)
