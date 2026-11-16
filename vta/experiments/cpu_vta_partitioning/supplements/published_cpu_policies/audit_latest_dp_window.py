"""Audit finite-window work inventory in the completed 48-layout DP retest.

Reports existing performance and isolated costs without changing either.
An exhausted stage proves a boundary condition, not a corrected steady FPS.
"""
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,verify_inventory
from scheduler_gap_audit.curves import csv_write
from run_scope_rank import assert_plan,inputs_for

ROOT=Path(__file__).resolve().parent
PARENT=ROOT/'artifacts/run_scope_retest_20261112_01'
OUT=ROOT/'artifacts/latest_dp_window_audit_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261112_latest_dp_window_audit'


def ns(value):
    return int((Decimal(str(value))*1_000_000).to_integral_value(rounding=ROUND_HALF_UP))


def window_inventory(rows, stages, warmup, scored):
    if len(rows)<warmup+scored or scored<2:
        raise ValueError('incomplete completion window')
    if [r['frame_id'] for r in rows]!=list(range(len(rows))):
        raise ValueError('missing, duplicate or reordered frame')
    times=[ns(r['completion_ms']) for r in rows]
    if any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('nonmonotone network completions')
    begin,end=times[warmup],times[warmup+scored-1]
    nfinal=[sum(t<=bound for t in times) for bound in (begin,end)]
    gap_count=scored-1
    if nfinal[1]-nfinal[0]!=gap_count:
        raise ValueError('completion interval count differs')
    result=[]
    for i,stage in enumerate(stages):
        finished=[ns(r['stage'+str(i)+'_end_ms']) for r in rows]
        started=[ns(r['stage'+str(i)+'_start_ms']) for r in rows]
        if (any(b<a for a,b in zip(started,finished))
            or any(finished[k]>started[k+1] for k in range(len(rows)-1))):
            raise ValueError('stage invocation order invalid')
        counts=[sum(t<=bound for t in finished) for bound in (begin,end)]
        ahead=[counts[k]-nfinal[k] for k in (0,1)]
        inventory_change=ahead[1]-ahead[0]
        if counts[1]-counts[0]-gap_count!=inventory_change or min(ahead)<0:
            raise ValueError('work-in-flight conservation failed')
        after_last=max(0,end-max(begin,finished[-1]))
        result.append(dict(stage_index=i,stage=stage['program'],device=stage['device'],
            completed_at_begin=counts[0],completed_at_end=counts[1],
            completed_in_window=counts[1]-counts[0],network_completions_in_window=gap_count,
            ahead_at_begin=ahead[0],ahead_at_end=ahead[1],inventory_change=inventory_change,
            stage_exhausted_before_window_end=finished[-1]<end,
            window_after_last_stage_call_ns=after_last,
            window_after_last_stage_call_percent=100*after_last/(end-begin)))
    return dict(begin_ns=begin,end_ns=end,duration_ns=end-begin,gaps=gap_count,stages=result)


def main():
    if OUT.exists() or REPORT.exists():
        raise FileExistsError('preserve this completed-window audit')
    result_path=PARENT/'analysis/RESULTS.json'
    for path,want in read(PARENT/'analysis/MANIFEST.json')['files'].items():
        if sha256(path)!=want:raise ValueError('accepted retest report changed')
    result=read(result_path);pool=read(PARENT/'POOL.json');lists=read(PARENT/'LISTS.json')
    plan=assert_plan();source_hashes={str(result_path):sha256(result_path),
        str(PARENT/'FREEZE.json'):sha256(PARENT/'FREEZE.json')}
    inputs={network:inputs_for(plan,network,'original') for network in pool}
    for spec in plan['networks'].values():
        for leaf in ('cost','programs','ddr'):
            source_hashes[spec[leaf+'_path']]=spec[leaf+'_sha256']
    stage_rows=[];cases=[]
    for observed in result['rows']:
        network,key=observed['network'],observed['configuration']
        cfg=pool[network][key];path=Path(observed['receipt'])
        if sha256(path)!=observed['receipt_sha256']:
            raise ValueError('native acceptance receipt changed')
        receipt=read(path)
        if receipt['boot_id']!=result['boot_id'] or receipt['task']['configuration']!=cfg:
            raise ValueError('boot or execution configuration mismatch')
        raw=Path(receipt['raw']);verify_inventory(raw,receipt['files'])
        rows=[json.loads(line) for line in (raw/'result.jsonl').read_text().splitlines()]
        task=receipt['task']
        if len(rows)!=task['warmup']+task['scored']+task['tail']:
            raise ValueError('native frame contract changed')
        inventory=window_inventory(rows,cfg['stages'],task['warmup'],task['scored'])
        for i,r in enumerate(inventory['stages']):
            existing=observed['stage_completion_counts_at_window'][i]
            if existing!={'stage':r['stage'],'counts':[r['completed_at_begin'],r['completed_at_end']]}:
                raise ValueError('raw stage boundary counts differ from accepted summary')
            cost=inputs[network][1].get(cfg['stages'][i],cfg)
            current=observed['stage_service_ms'][i]
            stage_rows.append(dict(network=network,configuration=key,**r,
                threads=cfg['stages'][i]['threads'],mask=cfg['stages'][i]['mask'],
                isolated_service_ms=cost['service_ms'],native_service_ms=current,
                native_over_isolated=current/cost['service_ms'],
                isolated_source_key=cost['source_key'],cost_assumptions=cost['assumptions'],
                isolated_is_not_a_prediction_of_contended_stage_service=True))
        cpu_exhaust=[r for r in inventory['stages'] if r['device']=='cpu' and r['stage_exhausted_before_window_end']]
        selected={m:next((i+1 for i,r in enumerate(rows) if r['id']==key),None)
                  for m,rows in lists[network].items()}
        pred=[r for r in result['predictions'] if r['network']==network and r['configuration']==key]
        cases.append(dict(network=network,configuration=key,method_ranks=selected,
            FPS=observed['FPS'],II_ms=observed['II_ms'],context=cfg['context_id'],
            CPU_gate=cfg['execution_mode'],exhausted_CPU_stages=[r['stage'] for r in cpu_exhaust],
            any_CPU_exhausted=bool(cpu_exhaust),
            max_window_after_last_CPU_percent=max([r['window_after_last_stage_call_percent'] for r in cpu_exhaust] or [0]),
            predictions=pred,inventory=inventory))
        source_hashes[str(path)]=sha256(path)
    if len(cases)!=48:raise ValueError('latest shared cohort no longer has48 layouts')
    totals={}
    for network in pool:
        group=[r for r in cases if r['network']==network]
        totals[network]=dict(configurations=len(group),
            with_early_CPU_exhaustion=sum(r['any_CPU_exhausted'] for r in group),
            maximum_window_after_last_CPU_percent=max(r['max_window_after_last_CPU_percent'] for r in group))
    OUT.mkdir();REPORT.mkdir(parents=True)
    csv_write(OUT/'stage_cost_and_inventory.csv',stage_rows)
    csv_write(OUT/'cases.csv',[{k:v for k,v in r.items() if k not in ('inventory','predictions')} for r in cases])
    output=dict(sources=source_hashes,boot_id=result['boot_id'],summary=totals,cases=cases,
        no_corrected_steady_FPS_inferred=True,old_performance_not_replaced=True,
        windows_not_rejected_and_no_remeasurement=True,new_native_sessions=0,new_DNN=0,
        historical_isolation_vs_current_shared_comparison_not_a_same_boot_causal_intervention=True)
    write(OUT/'RESULTS.json',output)
    lines=[
        '结论：短窗口存在工作量库存变化，不能把其FPS不加限定地称为持续满载稳态吞吐。',
        '本轮检查11月12日已完成的48项共同实测，原始帧和哈希重新核验；未新增板测或修改任何旧结果。',
        '计时仍为第8至23帧的16个完成时间戳，对应15个间隔；以下使用同一时钟重建各阶段在两端的累计完成数。',
    ]
    for network,summary in totals.items():
        lines.append('%s: %d项中%d项存在CPU阶段在正式窗口结束前做完本次全部28帧；最大后续无该阶段DNN调用的窗口占比%.2f%%。' %
            (network,summary['configurations'],summary['with_early_CPU_exhaustion'],summary['maximum_window_after_last_CPU_percent']))
    for network in pool:
        first=lists[network]['original_complete'][0]['id']
        case=next(c for c in cases if c['network']==network and c['configuration']==first)
        lines.append('\n%s完整域静态第一名 %s，原短窗口%.3f FPS：'%(network,first,case['FPS']))
        for st in case['inventory']['stages']:
            row=next(r for r in stage_rows if r['network']==network and r['configuration']==first and r['stage_index']==st['stage_index'])
            lines.append('  %s: 隔离/流水服务 %.3f/%.3f ms；窗口完成%d个阶段调用，末段完成15帧；在途领先量%d→%d；全部阶段调用完成后的窗口占比%.2f%%。' %
                (st['stage'],row['isolated_service_ms'],row['native_service_ms'],st['completed_in_window'],
                 st['ahead_at_begin'],st['ahead_at_end'],st['window_after_last_stage_call_percent']))
    lines.extend([
        '\n解释边界：',
        '前段在窗口中完成11帧、末段完成15帧，意味着4帧差额由先前在途库存补给，不代表这15帧都在相同持续竞争条件下完成。',
        '这不证明所有预测误差来自窗口，更不能按15/11等比例修正FPS。阶段调用跨越窗口边界，其墙钟时长不是窗口CPU工作量。',
        '不同CPU阶段的隔离服务用于资源需求记账，本来就不等于共享执行后的各阶段服务预测；逐阶段慢化和候选II误差必须分开报告。',
        '原短窗口结果保留为本次有限帧流观察；本审计不撤销旧数据、不新增拒收门槛、不自动延长或重测。',
        '下一项应先复核已有匹配长窗口是否消除了阶段提前耗尽，再决定最小尾部长度验证。保持16个时间戳也可以单独检验尾部长度，不必默认扩成大规模64帧重测。',
        '尚未获得修正后稳态FPS，不宣称本轮提高吞吐或解决调度误差。',
    ])
    (REPORT/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')
    write(OUT/'MANIFEST.json',dict(source_sha256=sha256(__file__),
        files={str(p):sha256(p) for p in [*OUT.iterdir(),REPORT/'FINDINGS_zh.txt'] if p.is_file()},
        source_hashes=source_hashes))
    print(json.dumps(dict(summary=totals,report=str(REPORT/'FINDINGS_zh.txt')),ensure_ascii=False))


if __name__=='__main__':main()
