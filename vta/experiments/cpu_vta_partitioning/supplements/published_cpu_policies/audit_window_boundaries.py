"""Check a queue-derived tail bound and observed pre/post-exhaustion gaps.

All existing scores, windows and FPS are preserved. Subwindow rates describe
the accepted run; they do not estimate a corrected FPS or rank candidates.
"""
from pathlib import Path
import json
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,verify_inventory
from audit_latest_dp_window import ns
from scheduler_gap_audit.curves import csv_write

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/window_boundaries_audit_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261112_latest_dp_window_audit'


def sufficient_tail(stages,depth):
    """Conservative frame bound, not a warmup or stationarity guarantee.

    Downstream input queues hold at most depth*(stages-1) distinct frames.
    Each downstream stage can hold one frame while executing or pushing.
    One more completed-front frame may be waiting to enter the next queue.
    At C_last, network completions are warmup+scored, so tail>ahead_bound
    prevents the front from having completed all planned frames already.
    """
    if type(stages) is not int or stages<1 or type(depth) is not int or depth<1:
        raise ValueError('invalid single-frame-stage queue contract')
    ahead=(stages-1)*(depth+1)+1
    return dict(max_ahead_conservative=ahead,sufficient_tail=ahead+1,
                ensures_only_no_early_source_exhaustion=True,
                does_not_prove_stationarity=True)


def partition_gaps(times,exhaustion):
    if len(times)<2 or any(type(t) is not int for t in times):
        raise ValueError('integer completion timestamps required')
    if any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('completion timestamps must increase')
    rows=[]
    for i,(begin,end) in enumerate(zip(times,times[1:])):
        kind=('before' if end<=exhaustion else 'after' if begin>=exhaustion else 'crossing')
        rows.append(dict(interval_index=i,begin_ns=begin,end_ns=end,duration_ns=end-begin,part=kind))
    groups={}
    for kind in ('before','crossing','after'):
        amounts=[r['duration_ns'] for r in rows if r['part']==kind]
        total=sum(amounts)
        groups[kind]=dict(intervals=len(amounts),duration_ns=total,
            mean_interval_ns=None if not amounts else dict(numerator=total,denominator=len(amounts)),
            observed_subwindow_FPS=None if not amounts else 1e9*len(amounts)/total)
    if sum(g['intervals'] for g in groups.values())!=len(times)-1 or sum(g['duration_ns'] for g in groups.values())!=times[-1]-times[0]:
        raise ValueError('completion gaps not conserved')
    return dict(intervals=rows,groups=groups)


def verify_native_source():
    bundle=ROOT/'artifacts/bundles/abbcccc62af230ee844745ac'
    manifest_path=bundle/'manifest.json';manifest=read(manifest_path)
    source_path=bundle/'sources/vta/apps/native_deploy/vta_stage_pipeline_runner.cc'
    proof_path=bundle/'plans/source_provenance.json';proof=read(proof_path)
    binary_path=bundle/'runner/vta_stage_pipeline_runner_pinned'
    for path in (source_path,proof_path,binary_path):
        if sha256(path)!=manifest['files'][str(path.relative_to(bundle))]['sha256']:
            raise ValueError('native source/binary capsule provenance changed')
    if (proof['source_policy_executor_sha256']!=sha256(binary_path)
            or proof['source_pinned_native_sha256']!=sha256(source_path)):
        raise ValueError('archived source-to-executor provenance differs')
    text=source_path.read_text()
    tokens=['not_full_.wait(lock, [&]() { return closed_ || queue_.size() < capacity_; });',
            'std::shared_ptr<Frame> frame;','while (queues[stage_index]->Pop(&frame))',
            'frame->done_ms = NowMillis();',
            'if (!queues[stage_index + 1]->Push(frame, frame->QueueTimingFor(stage_index + 1)))']
    lines={token:text[:text.index(token)].count('\n')+1 for token in tokens}
    if not lines[tokens[2]]<lines[tokens[3]]<lines[tokens[4]]:
        raise ValueError('completion/forwarding order changed')
    return dict(executor_sha256=sha256(binary_path),source_sha256=sha256(source_path),
        semantic_lines=lines,applies_to_ordinary_unmanaged_single_frame_stage_loop=True,
        source_to_binary_relation_is_archived_build_provenance=True),{
            str(p):sha256(p) for p in (manifest_path,source_path,proof_path,binary_path)}


def main():
    if OUT.exists():raise FileExistsError('preserve the boundary audit')
    parent=ROOT/'artifacts/latest_dp_window_audit_01'
    for path,want in read(parent/'MANIFEST.json')['files'].items():
        if sha256(path)!=want:raise ValueError('accepted inventory audit changed')
    previous=read(parent/'RESULTS.json')
    source_semantics,sources=verify_native_source()
    sources[str(parent/'RESULTS.json')]=sha256(parent/'RESULTS.json')
    source_receipts={}
    for path,want in previous['sources'].items():
        if '/accepted/' not in path:continue
        if sha256(path)!=want:raise ValueError('accepted native receipt changed')
        receipt=read(path);t=receipt['task']
        source_receipts[t['network'],t['configuration']['id']]=(path,receipt)
    results=[];gap_rows=[]
    for case in previous['cases']:
        network,key=case['network'],case['configuration']
        path,receipt=source_receipts[network,key];task=receipt['task'];raw=Path(receipt['raw'])
        verify_inventory(raw,receipt['files'])
        if task['executor_sha256']!=source_semantics['executor_sha256']:
            raise ValueError('queue proof belongs to another executor')
        launcher=(raw/'execute.sh').read_text()
        if '--queue-depth 2 ' not in launcher or '--p8-managed-slots' in launcher:
            raise ValueError('native queue mode differs from the proven ordinary pipeline')
        rows=[json.loads(s) for s in (raw/'result.jsonl').read_text().splitlines()]
        indices=[i for i,stage in enumerate(task['stages']) if stage['device']=='cpu']
        first=min(ns(rows[-1]['stage'+str(i)+'_end_ms']) for i in indices)
        kept=rows[task['warmup']:task['warmup']+task['scored']]
        times=[ns(r['completion_ms']) for r in kept]
        split=partition_gaps(times,first)
        before,after=[split['groups'][k]['observed_subwindow_FPS'] for k in ('before','after')]
        ratio=None if before is None or after is None else 100*(after/before-1)
        value=dict(network=network,configuration=key,original_FPS=case['FPS'],
            first_CPU_exhaustion_relative_to_window_ms=(first-times[0])/1e6,
            window_duration_ms=(times[-1]-times[0])/1e6,groups=split['groups'],
            after_vs_before_observed_FPS_percent=ratio,
            queue_tail_bound=sufficient_tail(len(task['stages']),2),
            subwindow_rates_not_used_for_ranking_or_costs=True)
        results.append(value)
        gap_rows.extend(dict(network=network,configuration=key,**r) for r in split['intervals'])
        sources[path]=sha256(path)
    summaries={}
    for network in ('resnet18','yolov3_tiny416'):
        cases=[r for r in results if r['network']==network]
        comparable=[r for r in cases if r['after_vs_before_observed_FPS_percent'] is not None]
        changes=[r['after_vs_before_observed_FPS_percent'] for r in comparable]
        summaries[network]=dict(cases=len(cases),with_complete_gaps_on_both_sides=len(comparable),
            after_faster=sum(v>0 for v in changes),after_slower=sum(v<0 for v in changes),
            median_observed_change_percent=None if not changes else statistics.median(changes),
            no_independence_or_causal_effect_claimed=True)
    OUT.mkdir()
    write(OUT/'RESULTS.json',dict(source_semantics=source_semantics,sources=sources,
        summaries=summaries,cases=results,maximum_domain_tail_bound=sufficient_tail(7,2),
        no_new_native_sessions=True,new_DNN=0,no_corrected_steady_FPS_inferred=True,
        no_automatic_tail_change=True))
    csv_write(OUT/'gaps.csv',gap_rows)
    csv_write(OUT/'cases.csv',results)
    lines=['窗口边界核查：只使用48项已有记录；新增板测和推理为0。',
        '每项原16时间戳的15个间隔按最早CPU阶段全部完成时刻分为之前、跨越和之后。所有间隔保留，原正式FPS不变。']
    for network,s in summaries.items():
        lines.append('%s：%d项中%d项有边界前后完整间隔；之后更快%d项、更慢%d项，描述性中位变化%s%%。'%
            (network,s['cases'],s['with_complete_gaps_on_both_sides'],s['after_faster'],s['after_slower'],
             '无' if s['median_observed_change_percent'] is None else '%.3f'%s['median_observed_change_percent']))
    lines.extend([
        '这不是因果干预：两个区段还可能具有不同在途分布及调度历史，不能将差额全部归给前段退出；更不能挑选较慢区段重新评估Top20。',
        '原执行器源码/二进制来源已核对。对普通线性传递、每阶段单个在执行帧、队列深度q的s阶段流水，前段领先最后完成数的保守容量界为(s-1)(q+1)+1。',
        '理由：下游s-1个输入队列各最多q帧，下游s-1个工作线程各最多持有1个未完成帧，加前段已完成但仍在Push的1帧。共享指针的重复引用不会增加不同帧数。',
        '最大7阶段、深度2时界为19帧。在最后计入时间戳C23处，已完成24帧；尾部至少20帧足以排除“前段已做完全部帧”这一边界。已准备的尾部24由队列结构支持，不是按旧FPS拟合的数值。',
        '这个界仅排除源任务提前耗尽，不保证足够预热、阶段负载平稳或模型准确；不作为新的拒收规则。',
        '尾部验证仍待用户确认；保持8帧预热和16时间戳，四个既定代表、8会话、196次推理，不自动执行。',
    ])
    report=REPORT/'BOUNDARY_GAPS_AND_TAIL_BOUND_zh.txt';report.write_text('\n'.join(lines)+'\n')
    write(OUT/'MANIFEST.json',dict(source_sha256=sha256(__file__),
        files={str(p):sha256(p) for p in [*OUT.iterdir(),report] if p.is_file()},sources=sources))
    print(json.dumps(dict(summaries=summaries,tail_bound=sufficient_tail(7,2),report=str(report)),ensure_ascii=False))


if __name__=='__main__':main()
