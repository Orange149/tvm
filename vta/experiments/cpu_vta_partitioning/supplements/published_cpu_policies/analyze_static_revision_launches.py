#!/usr/bin/env python3
"""Diagnostic launch-entry and same-thread workload comparisons; no model fitting."""
from pathlib import Path
from statistics import mean
from collections import defaultdict
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,now
from phase_model.simulator import rational
from scheduler_gap_audit.curves import csv_write
from static_revision.common import ROOT,OUTPUT,REPORT,TRACE
from static_revision.workflow import intrinsic_summary


def run():
    p=TRACE/'analysis/RESULTS.json'
    if sha256(p)!=read(TRACE/'DELIVERY.json')['results_sha256']:raise ValueError('corrected trace differs')
    n=read(p);frames=n['complete_front_frames'];rows=[]
    for k in range(12):
        by_frame=[];skew=[];before=[]
        for f in frames:
            w=[x for x in n['worker_rows'] if x['frame']==f and x['kernel']==k]
            if len(w)!=4:raise ValueError('four-worker scope incomplete')
            late=max(w,key=lambda x:(x['callback_start_ns'],x['slot']))
            by_frame.append(late['launch_to_body_ns']);before.append(late['pre_body_off_CPU_ns'])
            skew.append(max(x['callback_start_ns'] for x in w)-min(x['callback_start_ns'] for x in w))
        rows.append(dict(kernel=k,frames=len(frames),mean_latest_callback_entry_ms=mean(by_frame)/1e6,
            mean_callback_start_skew_ms=mean(skew)/1e6,
            mean_late_worker_pre_entry_off_CPU_ms=mean(before)/1e6,
            last_worker_body_CPU_ms=n['kernel_means'][k]['mean_callback_on_CPU_ms'],
            last_worker_body_preempt_off_ms=n['kernel_means'][k]['mean_preempt_off_ms'],
            entry_interval_includes_dispatch_observation_and_unknown_wakeup=True))
    intr_path=ROOT/'artifacts/cross_network_model_01/pool_temporal/INTRINSICS.json'
    audit=read(ROOT/'artifacts/model_semantics_audit_01/AUDIT.json')
    if sha256(intr_path)!=audit['intrinsics_sha256']:raise ValueError('isolated cache differs')
    profiles=read(intr_path);matched=[]
    for x in profiles['CPU'].values():
        i=x['identity']
        if i['program']!='cpu_00_02':continue
        if not x['no_pipeline_performance']:raise ValueError('profile is not isolated')
        cfg=dict(device='cpu',mask=i['mask'],worker_core_map=i['mask'],profile=x['average'])
        s=intrinsic_summary(i['network'],dict(stages=[cfg]))[0]
        s['program']=i['program'];s['module_sha256']=i['module_sha256'];matched.append(s)
    # The body work, total input size and resource mapping all differ. The two
    # network labels do not establish an architectural cause by themselves.
    observation=dict(source_sha256=sha256(p),analysis_source_sha256=sha256(__file__),
        frame_set=frames,kernel_entry=rows,
        sum_mean_latest_entry_ms=sum(x['mean_latest_callback_entry_ms'] for x in rows),
        sum_mean_start_skew_ms=sum(x['mean_callback_start_skew_ms'] for x in rows),
        these_sums_are_not_an_additive_stage_error_decomposition=True,
        same_thread_front_intrinsics=matched,isolated_cache_sha256=sha256(intr_path),
        long_compute_and_short_launches_both_require_explanation=True,
        probe_front_service_change_ms=n['probe_front_service_disturbance_ms'],
        no_pipeline_delay_added_to_predictor=True,no_new_board_session=True,recorded_at=now())
    write(OUTPUT/'LAUNCH_ENTRY_AUDIT.json',observation)
    csv_write(OUTPUT/'analysis/launch_entry.csv',rows)
    csv_write(OUTPUT/'analysis/same_thread_front_intrinsics.csv',matched)
    notes=dict(sources=[
        dict(name='Linux 5.4 CFS design',url='https://www.kernel.org/doc/html/v5.4/scheduler/sched-design-CFS.html',
             supported='Ideal simultaneous service is a reference; actual execution selects one runnable entity by virtual runtime with granularity.',
             application='Need separate ready, first scheduled, computation and completion events; do not infer a time quantum from one off-CPU span.',
             limitation='Vendor kernel source and effective granularity not yet established; no full CFS reproduction claim.'),
        dict(name='Daydream, USENIX ATC 2020, section 4 Algorithm 1',url='https://www.cs.toronto.edu/~pekhimenko/Papers/Daydream-ATC_20.pdf',
             supported='Per-task execution resource, dependencies and otherwise omitted CPU gaps affect replay; scheduling updates thread progress.',
             application='Compare dependency and resource timelines, beyond a global throughput match.',
             limitation='The paper does not establish an isolated-profile predictor for Linux contention among overlapping pinned CPU groups.'),
        dict(name='Packrat, ICML 2025, appendix A',url='https://proceedings.mlr.press/v267/bhardwaj25a.html',
             supplied_pdf=str(ROOT.parents[4]/'paper/VTA/Packrat.pdf'),supplied_pdf_sha256=sha256(ROOT.parents[4]/'paper/VTA/Packrat.pdf'),
             supported='Authors analyze isolation/concurrency gaps and leave interference out of the optimizer after checking that accounting for it did not change chosen configurations in their experiments.',
             application='Test whether interference preserves candidate order, rather than requiring perfect absolute-time agreement.',
             limitation='Their CPU-instance domain and x86 frequency/memory findings are not directly the mechanism on this four-core ARM CPU–VTA system.')],
        exact_reproduction_claim=False,new_scoring_formula_adopted=False)
    write(OUTPUT/'LITERATURE_NOTES.json',notes)
    lines=['并行启动与工作负载核查补充','',
        '新增板端会话0，未修改评分。以下使用同一组12个完整帧。',
        f"每帧12次调用的最晚回调进入延迟之和，平均{observation['sum_mean_latest_entry_ms']:.3f}毫秒；回调开始错位之和平均{observation['sum_mean_start_skew_ms']:.3f}毫秒。",
        '这是描述性合计，包含唤醒、调度、入口和探针等未完全区分的部分，不能当作可直接加入模型的延迟。','',
        '短调用也会拖尾：']
    for k in (1,5,9):
        x=rows[k];lines.append(f"第{k+1}次并行调用：最后完成者回调CPU约{x['last_worker_body_CPU_ms']:.4f}毫秒，最晚进入回调平均{x['mean_latest_callback_entry_ms']:.3f}毫秒。")
    lines+=['','同为四线程的前段隔离工作量：']
    for x in matched:
        if x['threads']==4:lines.append(f"{x['case']}：{x['parallel_launches']}次并行启动；最重一次的最大worker工作{x['largest_launch_ms']:.3f}毫秒；最重两次占逐次最大worker工作和{x['two_largest_launch_share']:.1%}。")
    lines+=['YOLO前段的并行启动次数并不比ResNet更多。差异还包括单次工作长度、线程数收益、输入尺寸、切图和并发阶段；不能只按网络名称或残差/双头结构解释。','',
        '文献适用边界：',
        'Packrat明确分析了隔离与并发性能偏差，但其优化器未直接建模该干扰，因为作者验证中选点未改变。这里需要检验的是候选排序是否被改变，不能照搬其不建模干扰的前提。',
        'Daydream提供依赖和执行资源时间线的依据；它没有替我们解决重叠核组的Linux争用预测。',
        'CFS的理想公平参考与实际执行存在粒度差别。已有板端记录未暴露调度粒度参数，下一步不能用猜测默认值或流水FPS拟合时间片。','',
        '下一项可检验方向：先核实平台可提供的调度/线程池调度信息，再用最小的非DNN平台测量检验有限执行粒度与线程身份历史；只有通过后才建立候选模型和冻结未见配置验证。']
    (OUTPUT/'analysis/LAUNCH_FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')
    for name in ('LAUNCH_ENTRY_AUDIT.json','LITERATURE_NOTES.json'):
        import shutil
        shutil.copy2(OUTPUT/name,REPORT/name)
    for name in ('launch_entry.csv','same_thread_front_intrinsics.csv','LAUNCH_FINDINGS_zh.txt'):
        shutil.copy2(OUTPUT/'analysis'/name,REPORT/name)
    write(OUTPUT/'LAUNCH_AUDIT_DELIVERY.json',dict(files={str(REPORT/n):sha256(REPORT/n) for n in
        ('LAUNCH_ENTRY_AUDIT.json','LITERATURE_NOTES.json','launch_entry.csv','same_thread_front_intrinsics.csv','LAUNCH_FINDINGS_zh.txt')},
        analysis_source_sha256=sha256(__file__),goal_complete=False))
    progress=read(OUTPUT/'GOAL_PROGRESS.json')
    progress.update(next_action='test platform-derived finite service and wakeup semantics; account for both short-entry delay and long-callback preemption',
        new_findings=['same four-thread YOLO front has longer dominant isolated kernels, not more launches',
          'short callbacks also have multi-millisecond entry skew',
          'predicted overall interval can be close while front stage and competing core attribution are wrong'],
        updated_at=now())
    write(OUTPUT/'GOAL_PROGRESS.json',progress,replace=True)
    return dict(entry_sum_ms=observation['sum_mean_latest_entry_ms'],same_thread_profiles=len(matched),
                new_board_sessions=0,report=str(REPORT/'LAUNCH_FINDINGS_zh.txt'),goal_complete=False)


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False))
