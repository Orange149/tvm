#!/usr/bin/env python3
"""Analyze a sealed three-session recorder repair without changing its controller."""
from pathlib import Path
from collections import Counter,defaultdict
from statistics import mean
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,verify_inventory,now
from scheduler_gap_audit.analysis import call_details
from scheduler_gap_audit.curves import csv_write
from yolo_scheduler_trace.analysis import measure_callback
from yolo_scheduler_trace.events import Attribution,clip
from yolo_scheduler_trace_repair.common import ROOT,ORIGINAL,OUTPUT,RUN,specs,protocol
from yolo_scheduler_trace_repair.prepare import validate
from yolo_scheduler_trace_repair.runtime import make_task,summary


def analyze(output=OUTPUT,run_dir=RUN):
    output,run_dir=Path(output),Path(run_dir);validate(output)
    state=read(run_dir/'state.json')
    if state['status']!='COMPLETE' or state['accepted_sessions']!=3 or state['DNN_inferences']!=97:
        raise ValueError('three-session repair batch incomplete')
    items={};receipts={};bundle=output/'capsule'
    for spec in specs():
        task=make_task(bundle,spec)
        path=run_dir/'accepted'/state['boot_id']/(task['task_id']+'.json');item=read(path)
        if item['task']!=task or item['boot_id']!=state['boot_id']:raise ValueError('receipt identity differs')
        verify_inventory(item['raw'],item['files'])
        if summary(item['raw'],task,bundle)!=item['summary']:raise ValueError('receipt no longer replays')
        items[spec['phase']]=item;receipts[str(path)]=sha256(path)
    clean=items['clean']['summary'];obs=items['observed']['summary']
    if any(i['summary']['output_fingerprints']!=clean['output_fingerprints'] for i in items.values()):
        raise ValueError('eight-output fingerprints differ')
    raw=items['observed']['raw'];doc=items['observed']['task'];scheduler=obs['scheduler']
    streams=scheduler['streams'];profile=read(Path(raw)/'profile.json')
    _,kernels,_=call_details(raw,doc);front=[k for k in kernels if k['stage']=='cpu_00_02']
    if len(front)!=192:raise ValueError('front launch coverage differs')
    if not all(s['coverage_ns'] for s in streams.values()):raise ValueError('no common interior coverage')
    start=max(s['coverage_ns'][0] for s in streams.values());end=min(s['coverage_ns'][1] for s in streams.values())
    selected=[]
    for f in range(doc['warmup'],doc['warmup']+doc['scored']):
        group=[k for k in front if k['frame']==f]
        if len(group)==12 and all(start<=k['start_ns']<=k['end_ns']<=end for k in group):selected.append(f)
    if not selected:raise ValueError('no complete front frame across all target TIDs')
    at=Attribution(streams);workers=[];critical=[];tails=[];role_totals=defaultdict(int);unknown=0
    for k in front:
        if k['frame'] not in selected:continue
        for w in k['worker_events']:
            r,body=measure_callback(k,w,streams[str(w['tid'])])
            if not r['fully_covered']:raise ValueError('selected callback not covered')
            workers.append(r)
            if r['last']:
                roles=defaultdict(int);unseen=0
                for off in body['off']:
                    if not off['preempt']:continue
                    a=at.other_work(off,w['task']);unseen+=a['unobserved_ns']
                    for name,ns in a['by_role_ns'].items():roles[name]+=ns;role_totals[name]+=ns
                critical.append({**r,'coincident_CPU_by_role_ns':dict(roles),'unobserved_preempt_off_ns':unseen})
                unknown+=unseen
        ev=k['launch_event'];a,b=ev['caller_tail_wall_start_ns'],ev['wall_end_ns']
        t=clip(streams[str(ev['caller'])],a,b)
        if not t['fully_covered']:raise ValueError('selected caller tail not covered')
        tails.append(dict(frame=k['frame'],kernel=k['kernel'],wall_ns=b-a,
            CPU_clock_ns=ev['cpu_end_ns']-ev['caller_tail_cpu_start_ns'],
            **{key:v for key,v in t.items() if key not in ('on','off')}))
    by_kernel=[]
    for ordinal in range(12):
        rows=[r for r in critical if r['kernel']==ordinal]
        tail=[r for r in tails if r['kernel']==ordinal]
        by_kernel.append(dict(kernel=ordinal,calls=len(rows),
            mean_callback_wall_ms=mean(r['wall_ns'] for r in rows)/1e6,
            mean_callback_on_CPU_ms=mean(r['on_CPU_ns'] for r in rows)/1e6,
            mean_preempt_off_ms=mean(r['preempt_off_ns'] for r in rows)/1e6,
            mean_nonpreempt_off_ms=mean(r['nonpreempt_off_ns'] for r in rows)/1e6,
            mean_TASKSTATS_queue_lower_ms=mean(r['queue_lower_ns'] for r in rows)/1e6,
            mean_TASKSTATS_queue_upper_ms=mean(r['queue_upper_ns'] for r in rows)/1e6,
            mean_caller_tail_on_CPU_ms=mean(r['on_CPU_ns'] for r in tail)/1e6,
            last_slot_counts=dict(Counter(r['slot'] for r in rows))))
    ring=[]
    for s in profile['streams']:
        chunks=s['drain_chunks'];capacity=s['data_size']
        ring.append(dict(tid=s['tid'],role=streams[str(s['tid'])]['role']['role'],
            events=len(streams[str(s['tid'])]['events']),drains=len(chunks),retained_bytes=s['retained_bytes'],
            physical_capacity_bytes=capacity,
            peak_retained_since_drain_bytes=max(c['head_after_publish']-c['tail'] for c in chunks),
            minimum_guard_remaining_bytes=min(capacity-(c['head_after_publish']-c['tail']) for c in chunks),
            drain_wall_envelope_ns=sum(c['end_ns']-c['begin_ns'] for c in chunks),
            boundary_unknown_ns=streams[str(s['tid'])]['boundary_unknown_ns']))
    original_result_path=ORIGINAL/'analysis/RESULTS.json'
    if sha256(original_result_path)!=read(ORIGINAL/'DELIVERY.json')['results_sha256']:
        raise ValueError('isolated motivation source changed')
    ref=read(original_result_path)['reference_isolated_front_ms']
    gap=clean['stage_service_ms'][0]-ref
    disturbance=abs(obs['stage_service_ms'][0]-clean['stage_service_ms'][0])
    quality=gap>0 and disturbance<=protocol()['probe_quality_fraction']*gap
    per_frame=[]
    for frame in selected:
        rows=[r for r in critical if r['frame']==frame]
        per_frame.append(dict(frame=frame,critical_callbacks=len(rows),
            critical_on_CPU_ms=sum(r['on_CPU_ns'] for r in rows)/1e6,
            critical_preempt_off_ms=sum(r['preempt_off_ns'] for r in rows)/1e6,
            critical_nonpreempt_off_ms=sum(r['nonpreempt_off_ns'] for r in rows)/1e6,
            caller_tail_on_CPU_ms=sum(r['on_CPU_ns'] for r in tails if r['frame']==frame)/1e6))
    result=dict(status='ANALYZED',boot_id=state['boot_id'],clean=clean,observed=obs,
        protocol=protocol(),source_receipts=receipts,analysis_source_sha256=sha256(__file__),
        original_capacity_correction_sha256=sha256(output/'DATA_VERDICT.json'),
        common_interior_ns=[start,end],complete_front_frames=selected,
        selected_launches=len(critical),all_scored_launches=len(front),kernel_means=by_kernel,
        critical_rows=critical,worker_rows=workers,tail_rows=tails,per_frame=per_frame,ring=ring,
        coincident_CPU_by_role_ns=dict(role_totals),unobserved_preempt_off_ns=unknown,
        total_selected_preempt_off_ns=sum(r['preempt_off_ns'] for r in critical),
        reference_isolated_front_ms=ref,reference_is_not_new_calibration=True,
        probe_FPS_change_percent=100*(obs['fps']/clean['fps']-1),
        probe_front_service_disturbance_ms=disturbance,probe_quality_passed=quality,
        quantitative_attribution_allowed_for_selected_frames_only=quality,
        all_kernel_means_use_same_complete_frame_set=True,
        off_CPU_not_all_runnable_wait=True,coincident_CPU_not_complete_causal_attribution=True,
        caller_tail_not_all_pure_spinning=True,caller_tail_overlaps_worker_intervals=True,
        no_wait_or_slowdown_added_to_static_cost=True,score_changed=False,DP_changed=False,Top20_changed=False,
        new_native_sessions=3,DNN_inferences=97,automatic_additional_measurements=0,recorded_at=now())
    dest=output/'analysis';dest.mkdir(exist_ok=True)
    write(dest/'RESULTS.json',result)
    for filename,rows in [('ring_capacity.csv',ring),('critical_callbacks.csv',critical),('workers.csv',workers),
                          ('caller_tails.csv',tails),('kernel_means.csv',by_kernel),('per_frame.csv',per_frame)]:
        csv_write(dest/filename,rows)
    plot(dest,result,front,streams)
    findings(dest,result)
    write(output/'DELIVERY.json',dict(status='ANALYZED',results_sha256=sha256(dest/'RESULTS.json'),
        analysis_source_sha256=sha256(__file__),files={str(p):sha256(p) for p in dest.iterdir() if p.is_file()},
        no_reboot=True,score_changed=False,DP_changed=False,Top20_changed=False,recorded_at=now()))
    return dict(complete_front_frames=selected,events=scheduler['total_events'],
        clean_FPS=clean['fps'],observed_FPS=obs['fps'],probe_quality_passed=quality,
        kernel_means=by_kernel,coincident_CPU_by_role_ms={k:v/1e6 for k,v in role_totals.items()},
        report=str(dest/'FINDINGS_zh.txt'))


def findings(dest,r):
    a=r['clean'];b=r['observed'];frames=len(r['complete_front_frames'])
    lines=['YOLO重叠配置调度诊断与记录器修复','',
        '本轮只修复观测数据回收，没有优化执行策略、静态评分或DP。',
        f"3个新会话、97次完整推理，未重启。关闭探针：{a['fps']:.4f} FPS，前段{a['stage_service_ms'][0]:.3f} ms；开启探针：{b['fps']:.4f} FPS，前段{b['stage_service_ms'][0]:.3f} ms。",
        f"探针FPS变化{r['probe_FPS_change_percent']:+.3f}%，前段服务扰动{r['probe_front_service_disturbance_ms']:.3f} ms，原约定质量条件通过={r['probe_quality_passed']}。",'',
        '记录完整性',
        f"12个目标线程共{r['observed']['scheduler']['total_events']}条切换记录；每次回收均检查真实内核环形缓冲容量、游标与连续保留字节，没有用扩大虚构容量来通过检查。",
        f"所有线程共同覆盖的内部区间用于归因，完整前段帧为{r['complete_front_frames']}。以下各内核均取相同{frames}帧，不混用截断前缀和不同worker的样本。",
        '启用之前、禁用之后及各线程首末事件以外的状态仍未知；接口不给出精确唤醒时刻。','',
        '最后完成worker的逐内核对照，单位毫秒',
        '内核  回调墙钟  实际在CPU上  抢占后离核  TASKSTATS排队范围']
    for x in r['kernel_means']:
        lines.append(f"{x['kernel']:2d}  {x['mean_callback_wall_ms']:9.3f}  {x['mean_callback_on_CPU_ms']:10.3f}  {x['mean_preempt_off_ms']:10.3f}  {x['mean_TASKSTATS_queue_lower_ms']:.3f}—{x['mean_TASKSTATS_queue_upper_ms']:.3f}")
    lines+=['','最后完成worker被抢占期间，同核可见的其他工作（每个完整前段帧的平均核毫秒）']
    for role,ns in sorted(r['coincident_CPU_by_role_ns'].items(),key=lambda x:-x[1]):
        lines.append(f'{role}: {ns/frames/1e6:.3f}')
    lines += [f"未观测到的同核工作或覆盖差额：{r['unobserved_preempt_off_ns']/frames/1e6:.3f}。",'',
        '可支持的结论',
        '若质量条件通过，这些记录直接显示哪些最后完成worker存在拖尾，以及它们离核时同核上运行的已识别工作。它们支持继续核查模型的理想共享与真实调度时序差异。',
        '这不等于已将约38毫秒模型误差完全分解；原模型已经包含一部分共享竞争。不能再把全部排队时间或caller尾部核时加到原预测上。',
        '同核同时出现是资源占用证据，不足以独立证明移走该角色即可获得相同数量的吞吐收益。非抢占离核不等于纯睡眠，caller尾部在CPU上不等于全是忙等。',
        '本轮仅一个输入、一次启动、一个资源配置；不能直接推广到其他切图或网络，也未修改任何未测候选评分。',
        '下一步应使用隔离输入检查模型是否能重现这些完成顺序和同核竞争时段，再决定最小模型修改；实测轨迹只作诊断，不回填通用成本。',
        '本批结束，不自动补测、重启或启动Top-20。']
    (dest/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')


def plot(dest,r,front,streams):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    frame=r['complete_front_frames'][0];group=[k for k in front if k['frame']==frame]
    a=min(k['start_ns'] for k in group);b=max(k['end_ns'] for k in group)
    colors=dict(CPU_stage='#4c78a8',VTA_host='#f58518',control='#54a24b')
    fig,ax=plt.subplots(figsize=(13,4))
    for s in streams.values():
        role=s['role'];color=colors[role['kind']]
        for span in s['on']:
            x,y=max(a,span['start_ns']),min(b,span['end_ns'])
            if y>x:ax.broken_barh([((x-a)/1e6,(y-x)/1e6)],(span['cpu']-.35,.7),facecolors=color)
    for k in group:ax.axvline((k['start_ns']-a)/1e6,color='black',alpha=.15,lw=.5)
    ax.set(yticks=range(4),yticklabels=['core '+str(i) for i in range(4)],xlabel='Time from front stage launch (ms)',
        title=f'First jointly covered front frame {frame}: observed target TIDs (blank = unobserved)',xlim=(0,(b-a)/1e6),ylim=(-.6,3.6))
    ax.legend(handles=[Patch(color=v,label=k) for k,v in colors.items()],loc='upper right')
    fig.tight_layout();fig.savefig(dest/'timeline.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--run-dir',type=Path,default=RUN);a=p.parse_args()
    print(json.dumps(analyze(a.output,a.run_dir),ensure_ascii=False))
