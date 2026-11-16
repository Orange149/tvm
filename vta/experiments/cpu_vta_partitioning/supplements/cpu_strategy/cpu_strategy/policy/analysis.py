"""Plain-language, boot-paired policy and Top20 evidence reports."""
import csv
from pathlib import Path
import math
import statistics

from cpu_vta_partitioning.common import read,write
from ..analysis import aggregate,comparison,write_csv
from .domain import CUTS,FAMILIES
from .campaign import validate_state,validated_rounds,load_costs

NAMES={'current':'现有共享配置','local_fastest':'各阶段单独最快','fixed_groups':'固定核组负载平衡',
       'same_groups_pinned':'相同分配逐线程固定','pinned_reallocated':'逐线程固定后重新分配',
       'reserved_host':'三核计算、一核主机与控制'}
METHODS={'single_core':'单线程物理核心映射','prefix':'原前缀核组静态DP','enhanced':'资源规则增强DP'}
CUT_NAMES={'current':'当前切图','historical_cut':'历史切点的共同图版本','light_tail':'尾段很轻的切图'}
MECHANISMS={'pinning':'相同分配下逐线程固定','host_placement':'固定CPU分配下的主机与控制放置',
            'gate':'固定分配下同组串行','placement':'固定两线程的核心分开放置'}


def analyze(root):
    root=Path(root).resolve();state=validate_state(root);rounds=validated_rounds(root);costs=load_costs(root,state)
    selection=read(root/'policy_selection.json') if (root/'policy_selection.json').exists() else None
    rule=read(root/'resource_rule.json') if (root/'resource_rule.json').exists() else None
    final=read(root/'final_selection.json') if (root/'final_selection.json').exists() else None
    confirmed=aggregate([r for r in rounds if r['role']=='confirmation'])
    rows=[];pairs={};policy=[];results={};versus_local={}
    if selection and confirmed:
        for cut in CUTS:
            for family,key in selection['families'][cut].items():
                if key in confirmed:
                    v=confirmed[key];policy.append({'cut':cut,'strategy':NAMES.get(family,family),'configuration':key,
                        'fps':v['fps'],'ii_ms':v['ii_ms'],'core_ms':v['core_ms_per_completion'],'boots':len(v['boots'])})
            for name,(a,b) in selection['pairs'][cut].items():
                if a in confirmed and b in confirmed:pairs[cut+'_'+name]=comparison(confirmed,a,b)
            if rule:
                a=selection['families'][cut][rule['family']];b=selection['families'][cut]['current']
                if a in confirmed and b in confirmed:results[cut]=comparison(confirmed,a,b)
                local=selection['families'][cut]['local_fastest']
                if a in confirmed and local in confirmed:versus_local[cut]=comparison(confirmed,a,local)
    if final and confirmed:
        for method,budgets in final['methods'].items():
            for budget,key in budgets.items():
                if key not in confirmed:continue
                v=confirmed[key];rows.append({'method':METHODS[method],'budget':int(budget),'configuration':key,
                    'fps':v['fps'],'ii_ms':v['ii_ms'],'core_ms':v['core_ms_per_completion'],'boots':len(v['boots'])})
        for budget in ('1','5'):
            for other in ('single_core','prefix'):
                a=final['methods']['enhanced'][budget];b=final['methods'][other][budget]
                if a in confirmed and b in confirmed:pairs['增强DP对'+METHODS[other]+'_预算'+budget]=comparison(confirmed,a,b)
    complete=state['status']=='COMPLETE'
    if complete:
        if not final or set(confirmed)!=set(final['pool']) or any(len(v['boots'])!=3 or any(x['rounds']!=2 for x in v['boots'].values()) for v in confirmed.values()):
            raise ValueError('complete run lacks three independent, two-round confirmation boots')
    support=False;boot_gains={}
    if complete and len(results)==3:
        boots=set(next(iter(results.values()))['paired_throughput_gains'])
        for boot in boots:boot_gains[boot]=math.exp(statistics.mean(math.log1p(p['paired_throughput_gains'][boot]) for p in results.values()))-1
        support=(statistics.mean(boot_gains.values())>=.05 and min(boot_gains.values())>0
                 and all(min(p['paired_throughput_gains'].values())>=-.05 for p in results.values()))
    mechanisms={}
    if complete:
        for name,label in MECHANISMS.items():
            relevant={cut:pairs[cut+'_'+name] for cut in CUTS if cut+'_'+name in pairs}
            if not relevant:continue
            boots=set(next(iter(relevant.values()))['paired_throughput_gains'])
            gains={boot:math.exp(statistics.mean(math.log1p(p['paired_throughput_gains'][boot]) for p in relevant.values()))-1 for boot in boots}
            positive=statistics.mean(gains.values())>=.05 and min(gains.values())>0 and all(min(p['paired_throughput_gains'].values())>=-.05 for p in relevant.values())
            mechanisms[name]={'name':label,'cuts':list(relevant),'paired_boot_gains':gains,'supported_5pct_consistent_direction':positive,
                              'conclusion':'达到本轮实际收益条件' if positive else '未达到本轮实际收益条件，不判定等效'}
    screen=[r for r in rounds if r['role']=='top20_screening'];top20=[];ranking_stats={}
    if screen:
        measured=aggregate(screen);ranked=read(root/'rankings.json')['enhanced']['rankings']
        for i,r in enumerate(ranked):
            if r['id'] not in measured:raise ValueError('enhanced Top20 incomplete')
            v=measured[r['id']];top20.append({'static_rank':i+1,'configuration':r['id'],'predicted_ms':r['score_ms'],'measured_ms':v['ii_ms'],'fps':v['fps']})
        by_speed=sorted(top20,key=lambda r:(r['measured_ms'],r['configuration']))
        order={r['configuration']:i+1 for i,r in enumerate(by_speed)}
        for r in top20:r['measured_rank']=order[r['configuration']]
        from scipy.stats import spearmanr
        ranking_stats={'best_static_rank':by_speed[0]['static_rank'],
                       'spearman':float(spearmanr([r['static_rank'] for r in top20],[r['measured_rank'] for r in top20]).statistic),
                       'scope':'one screening boot, two rounds; selection only'}
    report={'status':'COMPLETE' if complete else 'INCOMPLETE','formal_significance':False,
            'resource_rule':rule,'policy_support':support if complete else None,'policy_boot_gains':boot_gains,
            'policy_results':policy,'policy_vs_current':results,'comparisons':pairs,'methods':rows,
            'policy_vs_local_fastest':versus_local,'mechanism_conclusions':mechanisms,
            'top20':top20,'ranking_stats':ranking_stats,'completed_boots':state['boot_index'],
            'component_records':len(costs),'imported_component_records':sum(v['kind']=='imported' for v in costs.values()),
            'cpu_time_scope':'whole_process_between_scored_completions',
            'historical_11511':'Different float32 programs/boundaries and input; not a same-program FPS comparison'}
    dest=root/'analysis';dest.mkdir(exist_ok=True)
    write(dest/'analysis.json',report,replace=True);write_csv(dest/'policy_results.csv',policy);write_csv(dest/'methods.csv',rows);write_csv(dest/'top20.csv',top20)
    component_rows=[{'signature':k,'program':v['configuration']['stages'][0]['program'],
                     'threads':v['configuration']['stages'][0]['threads'],'mode':v['configuration']['affinity_mode'],
                     'mask':','.join(map(str,v['configuration']['stages'][0]['mask'])),'source':v['kind'],**v['cost']} for k,v in costs.items()]
    write_csv(dest/'components.csv',component_rows)
    text=['ResNet18 成熟资源分配与完整DP验证','状态：'+report['status'],
          f"完成独立启动：{state['boot_index']}/6；组件成本{len(costs)}项，其中复用{report['imported_component_records']}项。",
          '正式结论采用最后三个启动；同启动两轮先平均，不将帧作为独立样本。']
    if rule:text.append('冻结的资源策略：'+NAMES[rule['family']])
    if complete:text.append('资源规则结论：'+('达到5%实际改善门槛且三个启动方向一致。' if support else '未达到预设的跨启动实际改善条件；不将结果解释为等效。'))
    for item in mechanisms.values():text.append(item['name']+'：'+item['conclusion']+'；范围：'+'、'.join(CUT_NAMES[c] for c in item['cuts'])+'。')
    for r in policy:text.append(f"{CUT_NAMES[r['cut']]}，{r['strategy']}：{r['fps']:.3f}张/秒，完成间隔{r['ii_ms']:.3f}ms。")
    for r in rows:text.append(f"{r['method']}，预算{r['budget']}：{r['fps']:.3f}张/秒，完成间隔{r['ii_ms']:.3f}ms。")
    text.extend(['历史11.511 FPS来自不同程序、浮点边界及输入，未作为本轮必须复现的目标。',
                 '不同绑核、门控和主机策略的收益分别由匹配干预报告；不把全部收益归为评分公式。'])
    (dest/'REPORT_zh.txt').write_text('\n'.join(text)+'\n')
    plots(dest,component_rows,policy,top20)
    return report


def plots(dest,components,policy,top20):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if top20:
        fig,ax=plt.subplots(figsize=(8,4));ax.plot([r['static_rank'] for r in top20],[r['fps'] for r in top20],'o-')
        ax.set(xlabel='Static DP rank',ylabel='Measured frames/s',title='All 20 enhanced-DP candidates');fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(dest/('top20.'+ext),dpi=160)
        plt.close(fig)
    programs=sorted({r['program'] for r in components if r['program'].startswith('cpu_')})
    programs=[p for p in programs if p in ('cpu_00_02','cpu_17_20','cpu_18_20','cpu_20_20')]
    if programs:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for program in programs:
            for mode in ('group','pinned'):
                data=[r for r in components if r['program']==program and r['mode']==mode and r['mask']==','.join(map(str,range(r['threads']))) ]
                data.sort(key=lambda r:r['threads'])
                if not data:continue
                for ax,f in zip(axes,('run_ms','run_core_ms')):ax.plot([r['threads'] for r in data],[r[f] for r in data],marker='o',label=program+' '+mode)
        axes[0].set(xlabel='Threads',ylabel='Computation wall ms');axes[1].set(xlabel='Threads',ylabel='CPU core-ms')
        axes[1].legend(fontsize=6);fig.tight_layout()
        for ext in ('png','svg'):fig.savefig(dest/('thread_wall_core_curves.'+ext),dpi=160)
        plt.close(fig)
