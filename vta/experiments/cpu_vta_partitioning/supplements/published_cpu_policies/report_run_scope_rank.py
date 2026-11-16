"""Report frozen full-domain input sensitivity without reading pipeline FPS."""
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256
from cpu_policies.open_resources.domain import context
from policy_branch_eval.common import isolated_inputs_only
from scheduler_gap_audit.curves import csv_write
from run_scope_rank import ROOT,OUT,NETWORKS

REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_run_scope_rank'


def topology(row):
    return ' → '.join(('CPU' if r['device']=='cpu' else 'VTA')+
                     str(r['begin'])+'–'+str(r['end']-1) for r in row['stages'])


def CPU_resources(row):
    return '; '.join('CPU'+str(r['begin'])+'–'+str(r['end']-1)+': '+str(r['threads'])+
        '线程/'+str(r['worker_core_map']) for r in row['stages'] if r['device']=='cpu')


def run():
    seal=read(OUT/'SELECTION_FREEZE.json')
    for leaf,wanted in seal['files'].items():
        if sha256(OUT/leaf)!=wanted:
            raise ValueError('frozen rank comparison changed')
    if seal['plan_sha256']!=sha256(OUT/'PLAN.json'):
        raise ValueError('rank plan changed')
    with isolated_inputs_only():
        compared=read(OUT/'COMPARE.json');lists=read(OUT/'LISTS.json');scores=read(OUT/'SCORES.json')
        rows=[]
        for network in NETWORKS:
            old={r['id']:r for r in lists[network+'/original']}
            new={r['id']:r for r in lists[network+'/run_scope']}
            union={**old,**new}
            for key,row in sorted(union.items(),key=lambda p:(
                    old.get(p[0],{}).get('static_rank',999),new.get(p[0],{}).get('static_rank',999),p[0])):
                a,b=(scores[network][variant][key] for variant in ('original','run_scope'))
                rows.append(dict(network=network,configuration=key,original_rank=old.get(key,{}).get('static_rank'),
                    run_scope_rank=new.get(key,{}).get('static_rank'),
                    original_estimated_II_ms=a['score_ns']/1e6,run_scope_estimated_II_ms=b['score_ns']/1e6,
                    original_binding_terms=a['binding_terms'],run_scope_binding_terms=b['binding_terms'],
                    topology=topology(row),CPU_configuration=CPU_resources(row),
                    context=context(row)['label'],CPU_gate='关闭' if row['execution_mode']=='off' else '同组串行',
                    host_mask=row['host_mask'],control_mask=row['control_mask'],
                    measured_FPS_not_read=True))
    REPORT.mkdir(parents=True,exist_ok=True)
    csv_write(REPORT/'static_candidate_comparison.csv',rows)
    lines=['隔离线程份额修正：完整候选域静态检验','',
        '本轮只检验输入修正，不修改评分公式或运行机制，不读取新旧流水FPS。',
        '两边均采用已有最强完整模型：非均匀CPU份额、VTA主机CPU需求、线程组服务约束、主机集中约束全部开启。',
        '旧输入使用原隔离会话线程份额；修正输入使用可重建的全部run区间份额，无法重建者保留原值或回退。',
        '两个网络沿用各自真实成本；阶段总服务、总run核时、交接、VTA和DDR均不变。',
        '所有九种资源上下文、合法核心集合、开放并发及同组串行均保留。YOLO两项仅用于原子线程曲线的模块继续排除在切图域外，与此前合法域一致。','',
        '网络 / 原第一名是否相同 / 前五项重合 / 前二十项重合 / 名单顺序完全相同 / 去重并集']
    for r in compared['summaries']:
        lines.append(f"{r['network']} / {r['same_first']} / {r['top5_overlap']}/5 / {r['overlap20']}/20 / {r['same_ordered20']} / {r['union']}")
    lines+=['','全域证明与准备成本：']
    for network,spec in NETWORKS.items():
        lines.append(f"{network}：{spec['topologies']}种切图、{spec['allocations']:,}个资源组合/版本，九个上下文全部经独立C++枚举核对；最终Top-20另经Python标量公式复核。")
        for variant in ('original','run_scope'):
            path=OUT/network/variant
            proof=read(path/'PROOF.json');progress=read(path/'PROGRESS.json');source=read(path/'INPUTS.json')
            lines.append(f"  {variant}：本地耗时{progress['elapsed_seconds']:.2f}s，精确复用{len(source['cache'])}个原检查点，网络尝试{proof['guard']['network_attempts']}次。")
    lines+=['',
        '详细表列出各项的切图、CPU线程映射、主机/控制位置、门控、两个静态分数及主导约束。估计完成间隔不是实测FPS，也不是严格的整机性能上限。',
        '如果两份有序名单完全相同，说明这项输入修正在本域的预算1/5/20均未改变选点，无需为了该零差异再重复板测。',
        '如果名单或顺序变化，只能说明输入修正影响了搜索；是否改善预算1/5/20下的实测选点，仍需冻结后共同测量。',
        '额外全域组件标定和本轮板测均为0。输入作用域验证的既有三签名实验成本另列，不称为免费取得了直接逐线程时钟。',
        '共享执行时序模型仍有约32.8ms诊断误差。这项输入修正未修改其逐内核CPU工作，不能据此宣布该误差已解决。',
        '原ResNet18约13FPS历史结果保持原身份和口径，不把本轮静态分数转换为对它的性能复测。']
    (REPORT/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')
    write(REPORT/'EVIDENCE.json',dict(source_selection_sha256=sha256(OUT/'SELECTION_FREEZE.json'),
        source_sha256=sha256(__file__),files={str(p):sha256(p) for p in (
            REPORT/'FINDINGS_zh.txt',REPORT/'static_candidate_comparison.csv')},
        no_pipeline_performance_read=True,new_native_sessions=0))
    print(json.dumps(dict(summaries=compared['summaries'],report=str(REPORT/'FINDINGS_zh.txt')),
                     ensure_ascii=False,indent=2))


if __name__=='__main__':
    run()
