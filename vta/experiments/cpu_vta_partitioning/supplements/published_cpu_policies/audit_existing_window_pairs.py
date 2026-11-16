"""Replay work-in-flight accounting for nine accepted short/long pairs.

Same configuration, program, runtime and boot; separate processes and both
scored count/tail changed. This cannot isolate the causal effect of tail alone.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,verify_inventory
from scheduler_gap_audit.curves import csv_write
from audit_latest_dp_window import window_inventory

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/existing_window_pairs_audit_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261112_latest_dp_window_audit'


def main():
    if OUT.exists():raise FileExistsError('preserve the accepted window-pair audit')
    result_path=ROOT/'artifacts/cross_network_model_01/FINAL_LONG_RESULTS.json'
    old=read(result_path);source_hashes={str(result_path):sha256(result_path)}
    parents={str(Path(r['path']).parent) for r in old['sources'].values()}
    index={}
    for parent in parents:
        for path in Path(parent).glob('*.json'):
            receipt=read(path);task=receipt['task']
            if task.get('scored')==16:
                index.setdefault(task['configuration']['id'],[]).append((path,receipt))
    pairs=[]
    for identity,source in old['sources'].items():
        path=Path(source['path'])
        if sha256(path)!=source['sha256']:raise ValueError('accepted long source changed')
        long=read(path);task=long['task'];cfg=task['configuration']
        matches=index.get(cfg['id'],[])
        if len(matches)!=1:raise ValueError('matching short task ambiguous')
        short_path,short=matches[0]
        if short['boot_id']!=long['boot_id'] or long['boot_id']!=old['boot_id']:
            raise ValueError('short/long boot mismatch')
        allowed={'task_id','scored','tail','phase'}
        if {k:v for k,v in task.items() if k not in allowed}!={k:v for k,v in short['task'].items() if k not in allowed}:
            raise ValueError('short/long execution differs beyond window')
        row=dict(network=identity.split('/')[0],configuration=cfg['id'],same_boot=long['boot_id'])
        for label,receipt,receipt_path in [('short',short,short_path),('long',long,path)]:
            raw=Path(receipt['raw']);verify_inventory(raw,receipt['files'])
            frames=[json.loads(s) for s in (raw/'result.jsonl').read_text().splitlines()]
            t=receipt['task'];window=window_inventory(frames,cfg['stages'],t['warmup'],t['scored'])
            affected=[r for r in window['stages'] if r['device']=='cpu' and r['stage_exhausted_before_window_end']]
            row.update({label+'_FPS':1e9*window['gaps']/window['duration_ns'],
                label+'_window':[t[k] for k in ('warmup','scored','tail')],
                label+'_exhausted_CPU_stages':[r['stage'] for r in affected],
                label+'_maximum_exhausted_window_percent':max([r['window_after_last_stage_call_percent'] for r in affected] or [0]),
                label+'_stage_counts':[{k:r[k] for k in ('stage','completed_at_begin','completed_at_end','inventory_change')}
                                      for r in window['stages']]})
            if abs(row[label+'_FPS']-receipt['summary']['FPS'])>1e-7:
                raise ValueError('raw window does not reproduce accepted FPS')
            source_hashes[str(receipt_path)]=sha256(receipt_path)
        row['long_vs_short_FPS_percent']=100*(row['long_FPS']/row['short_FPS']-1)
        pairs.append(row)
    if len(pairs)!=9:raise ValueError('accepted long review scope changed')
    OUT.mkdir()
    summary={network:dict(pairs=len(group),short_with_exhaustion=sum(bool(r['short_exhausted_CPU_stages']) for r in group),
        long_with_exhaustion=sum(bool(r['long_exhausted_CPU_stages']) for r in group))
        for network in sorted({r['network'] for r in pairs})
        for group in [[r for r in pairs if r['network']==network]]}
    write(OUT/'RESULTS.json',dict(sources=source_hashes,summary=summary,pairs=pairs,new_native_sessions=0,new_DNN=0,
        not_the_latest_run_scope_configs=True,no_causal_tail_only_effect_claimed=True,
        windows_not_used_to_fit_static_cost_or_FPS=True))
    csv_write(OUT/'pairs.csv',pairs)
    lines=['已有匹配短/长窗口复核：9个旧选点，全部来自同一旧boot；不是11月12日新Top20的重测。',
           '每对除phase、任务哈希和窗口(8/16/4→8/64/32)外，任务执行定义完全相同；重新校验全部原始文件并重建FPS。']
    for network,counts in summary.items():
        lines.append('%s: %d对中，短窗口%d项有CPU提前做完全部任务，长窗口%d项。'%
                     (network,counts['pairs'],counts['short_with_exhaustion'],counts['long_with_exhaustion']))
    for r in pairs:
        lines.append('%s %s: 短%.3f FPS、提前耗尽CPU%s；长%.3f FPS、提前耗尽CPU%s。'%
            (r['network'],r['configuration'],r['short_FPS'],r['short_exhausted_CPU_stages'],
             r['long_FPS'],r['long_exhausted_CPU_stages']))
    lines.extend(['该证据支持“足够尾部可避免源工作在窗口内耗尽”的可行性，但长窗口同时改变计入帧数和进程运行历史，不能将FPS差额全部归因于尾部。',
        '下一项最小验证应保持预热8和计入16，只改变尾部，先验证少数冻结代表；不重搜DP、不改变公式、不直接扣除或缩放旧FPS。',
        '本轮不自动启动补测。旧13FPS成果保留；此项审计没有推出新的修正后FPS或稳定性结论。'])
    report=REPORT/'EXISTING_WINDOW_PAIRS_zh.txt';report.write_text('\n'.join(lines)+'\n')
    write(OUT/'MANIFEST.json',dict(source_sha256=sha256(__file__),window_accounting_source_sha256=sha256(ROOT/'audit_latest_dp_window.py'),
        files={str(p):sha256(p) for p in [*OUT.iterdir(),report] if p.is_file()},source_hashes=source_hashes))
    print(json.dumps(dict(summary=summary,report=str(report)),ensure_ascii=False))


if __name__=='__main__':main()
