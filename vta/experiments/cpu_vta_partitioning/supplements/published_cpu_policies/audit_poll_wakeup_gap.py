"""Offline check of the real host-switch cadence and its validation coverage.

No board commands, new predictor or cost fitting. Non-preemptive off-CPU
intervals include both blocking and subsequent CPU wait, not exact wake times.
"""
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
import json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,verify_inventory,now
from yolo_scheduler_trace.events import clip,intersection
from scheduler_gap_audit.curves import csv_write

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'artifacts/poll_wakeup_gap_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_poll_wakeup_gap'


def host_cycles(stream,phases,start,end):
    phases=sorted([p for p in phases if p['operation']=='run' and p['tid']==stream['tid']],
        key=lambda p:p['wall_start_ns'])
    begins=[p['wall_start_ns'] for p in phases];events=stream['events'];rows=[]
    for a,b,c in zip(events,events[1:],events[2:]):
        if (a['direction'],b['direction'],c['direction'])!=('in','out','in') or b['preempt']:continue
        if not start<=a['time_ns']<=b['time_ns']<=c['time_ns']<=end:continue
        i=bisect_right(begins,a['time_ns'])-1
        if i<0 or c['time_ns']>phases[i]['wall_end_ns']:continue
        if a['cpu']!=b['cpu']:raise ValueError('host moved while continuously on CPU')
        rows.append(dict(tid=stream['tid'],stage=stream['role']['role'],frame=phases[i]['call'],
            on_start_ns=a['time_ns'],out_ns=b['time_ns'],in_ns=c['time_ns'],
            on_CPU_ns=b['time_ns']-a['time_ns'],nonpreempt_off_ns=c['time_ns']-b['time_ns'],
            cycle_ns=c['time_ns']-a['time_ns'],from_core=b['cpu'],to_core=c['cpu'],
            migrated=b['cpu']!=c['cpu'],exact_wakeup_time_observed=False))
    return rows


def distribution(values):
    if not values:return dict(count=0)
    return dict(count=len(values),minimum_us=min(values)/1000,median_us=float(np.median(values))/1000,
        p90_us=float(np.percentile(values,90))/1000,p99_us=float(np.percentile(values,99))/1000,
        maximum_us=max(values)/1000)


def run():
    original=ROOT/'artifacts/yolo_scheduler_trace_repair_01';path=original/'analysis/RESULTS.json'
    if sha256(path)!=read(original/'DELIVERY.json')['results_sha256']:raise ValueError('corrected trace digest differs')
    valid=read(original/'FINAL_VALIDATION.json');trace=read(path)
    if valid['status']!='PASSED' or not trace['probe_quality_passed'] or trace['complete_front_frames']!=list(range(12,24)):
        raise ValueError('complete trace/probe quality not established')
    observed=None
    for receipt,h in trace['source_receipts'].items():
        if sha256(receipt)!=h:raise ValueError('trace source receipt differs')
        item=read(receipt)
        if item['task']['spec']['phase']=='observed':observed=item
    if observed is None:raise ValueError('observed trace source missing')
    verify_inventory(observed['raw'],observed['files'])
    phases=[json.loads(s) for s in Path(observed['raw'],'phase_trace.jsonl').read_text().splitlines()]
    streams=trace['observed']['scheduler']['streams'];a,b=trace['common_interior_ns']
    hosts=[s for s in streams.values() if s['role']['kind']=='VTA_host']
    cycles=[c for s in hosts for c in host_cycles(s,phases,a,b)]
    summaries=[]
    for s in hosts:
        rows=[c for c in cycles if c['tid']==s['tid']]
        summaries.append(dict(stage=s['role']['role'],tid=s['tid'],cycles=len(rows),
            on_CPU=distribution([r['on_CPU_ns'] for r in rows]),
            off_CPU_after_nonpreempt_out=distribution([r['nonpreempt_off_ns'] for r in rows]),
            combined_cycle=distribution([r['cycle_ns'] for r in rows]),
            migrated_cycles=sum(r['migrated'] for r in rows),
            sleep_and_runnable_wait_not_separated=True))
    thresholds=[20_000,50_000,100_000,200_000];coincident={n:dict(short=0,long=0) for n in thresholds}
    host_rows={c:[] for c in range(4)}
    for s in hosts:
        for r in s['on']:host_rows[r['cpu']].append(r)
    for rows in host_rows.values():rows.sort(key=lambda r:r['start_ns'])
    begins={c:[r['start_ns'] for r in rows] for c,rows in host_rows.items()}
    for row in trace['critical_rows']:
        body=clip(streams[str(row['tid'])],row['callback_start_ns'],row['callback_end_ns'])
        for off in body['off']:
            if not off['preempt']:continue
            core=off['from_cpu']
            if core!=off['to_cpu']:raise ValueError('fixed critical worker migrated')
            rows=host_rows[core];j=max(0,bisect_right(begins[core],off['start_ns'])-1)
            while j<len(rows) and rows[j]['start_ns']<off['end_ns']:
                r=rows[j];n=intersection(off['start_ns'],off['end_ns'],r['start_ns'],r['end_ns'])
                for threshold in thresholds:
                    coincident[threshold]['short' if r['end_ns']-r['start_ns']<=threshold else 'long']+=n
                j+=1
    expected=sum(v for k,v in trace['coincident_CPU_by_role_ns'].items() if k.endswith('/host'))
    if any(v['short']+v['long']!=expected for v in coincident.values()):
        raise ValueError('coincident-host CPU split does not conserve existing attribution')
    split=[dict(on_CPU_span_threshold_us=k/1000,
        short_host_coincident_core_ms_per_frame=v['short']/12/1e6,
        longer_host_coincident_core_ms_per_frame=v['long']/12/1e6,
        total_host_coincident_core_ms_per_frame=expected/12/1e6,
        not_causal_wait_decomposition=True) for k,v in coincident.items()]
    old=read(ROOT/'artifacts/finite_service_01/PLAN.json')['protocol']['heldout_case']
    result=dict(hosts=summaries,coincident_split=split,
        old_platform_heldout=old,source_trace=str(path),source_trace_sha256=sha256(path),
        native_boot=trace['boot_id'],new_native_sessions=0,new_DNN_inferences=0,
        selected_front_frames=trace['complete_front_frames'],all_counts_within_common_trace_run_spans=True,
        nonpreempt_out_not_proof_of_nanosleep=True,exact_wakeup_events_unavailable=True,
        no_scoring_or_DP_change=True,source_sha256=sha256(__file__),recorded_at=now())
    write(OUTPUT/'RESULTS.json',result,replace=True);csv_write(OUTPUT/'host_cycles.csv',cycles)
    csv_write(OUTPUT/'critical_host_span_split.csv',split)
    lines=['隔离profile与流水误差：主机短唤醒节奏核查','',
        '本轮仅使用已修复、无丢失且满足原探针质量条件的已有轨迹；没有连接开发板，没有新增推理、评分或DP变化。',
        '统计限制在全部目标TID共同覆盖区间中、完全处于真实VTA run调用内的连续IN→非抢占OUT→IN片段，排除跨run/队列边界的片段。',
        '非抢占离开到恢复包含阻塞和之后等待CPU，不能称为精确睡眠或唤醒延迟；短运行片段与轮询一致，但不保证每段只对应一次查询。','']
    for s in summaries:
        lines.append(f"{s['stage']}：{s['cycles']}个完整片段，运行中位数{s['on_CPU']['median_us']:.3f}us，离开CPU后恢复中位数{s['off_CPU_after_nonpreempt_out']['median_us']:.3f}us，合计周期中位数{s['combined_cycle']['median_us']:.3f}us；周期P99 {s['combined_cycle']['p99_us']:.3f}us，跨核心恢复{s['migrated_cycles']}次。")
    lines+=['','关键worker被抢占期间，同核心上实际观测的VTA主机CPU工作：']
    for r in split:lines.append(f"按完整on-CPU片段不超过{r['on_CPU_span_threshold_us']:.0f}us划分：短片段{r['short_host_coincident_core_ms_per_frame']:.3f}核ms/帧，较长片段{r['longer_host_coincident_core_ms_per_frame']:.3f}核ms/帧。")
    lines+=['这些是重合区间内的主机CPU工作，不是可直接加到阶段耗时的惩罚，不证明全部拖尾都由主机导致。',
        '原有限CPU服务模型的混合负载验证使用1ms睡眠、30us或1ms忙段，且误差容限允许达到最大标定服务粒度；这不足以证明模型准确表达上述短片段和恢复时序。',
        '不过，同核重合CPU工作主要来自较长片段；先定位这些长片段的set/run/get、物理核心和关键worker关系，再决定是否需要短唤醒自测，不能先把主要原因归给短轮询。',
        '已有13FPS基准差异保留，不继续桥接或全域搜索。后续逐phase定位见20261111_host_burst_gap报告。']
    REPORT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')
    import shutil
    for p in OUTPUT.iterdir():
        if p.is_file():shutil.copy2(p,REPORT/p.name)
    return dict(hosts=summaries,coincident_split=split,report=str(REPORT/'FINDINGS_zh.txt'))

if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
