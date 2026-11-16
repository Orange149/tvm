"""Post-freeze evaluation; never imported by search or candidate selection."""
import csv
from itertools import combinations
import json
import math
from pathlib import Path
import statistics

from .common import MODELS,settings,api

def evaluate(lists, measurements):
    if set(lists)!={row[0] for row in MODELS}:raise ValueError('eight lists required')
    methods={}
    for key,ids in lists.items():
        if len(ids)!=20 or len(set(ids))!=20:raise ValueError('twenty different candidates required')
        if any(i not in measurements or not math.isfinite(measurements[i]['ii_ms']) or measurements[i]['ii_ms']<=0 for i in ids):
            raise ValueError('missing or invalid performance')
        budgets={}
        for n in (1,5,20):
            selected=min(ids[:n],key=lambda i:(measurements[i]['ii_ms'],i))
            ii=measurements[selected]['ii_ms']
            budgets[str(n)]=dict(selected=selected,static_position=ids.index(selected)+1,ii_ms=ii,fps=1000/ii)
        methods[key]=dict(label=settings(key)['label'],members=ids,budgets=budgets)
    deletion=[]
    for name,key in [('线程负载份额','host_team'),('主机集中约束','weights_team'),('线程组服务约束','weights_host')]:
        for n in ('1','5','20'):
            complete=methods['complete']['budgets'][n];removed=methods[key]['budgets'][n]
            deletion.append(dict(factor=name,without_model=key,budget=int(n),complete_fps=complete['fps'],without_fps=removed['fps'],
                removal_cost_percent=(complete['fps']/removed['fps']-1)*100,
                selected_configuration_same=complete['selected']==removed['selected']))
    effects=[]
    for n in ('1','5','20'):
        for mask in range(1,8):
            coefficient=0.
            for key,_,w,h,t in MODELS:
                signs=[1 if w else -1,1 if h else -1,1 if t else -1]
                sign=math.prod(signs[i] for i in range(3) if mask&(1<<i))
                coefficient+=sign*math.log(methods[key]['budgets'][n]['fps'])/8
            effects.append(dict(budget=int(n),factor_combination='×'.join(['线程份额','主机约束','线程组约束'][i]
                for i in range(3) if mask&(1<<i)),coefficient_log_fps=coefficient,
                factorial_effect_log_fps=2*coefficient,factorial_effect_percent=math.expm1(2*coefficient)*100,
                descriptive_only=True))
    overlaps=[]
    for a,b in combinations(lists,2):
        for n in (1,5,20):
            aa,bb=set(lists[a][:n]),set(lists[b][:n]);intersection=len(aa&bb)
            overlaps.append(dict(model_a=a,model_b=b,budget=n,intersection=intersection,
                Jaccard=intersection/len(aa|bb),ordered_lists_identical=lists[a][:n]==lists[b][:n]))
    return dict(methods=methods,deletion=deletion,factorial_effects=effects,overlaps=overlaps,
        single_boot_observation_only=True,independent_confirmation=False)

def tied_metrics(predicted,observed):
    ties=concordant=discordant=0
    for a,b in combinations(predicted,2):
        if predicted[a]==predicted[b]:ties+=1;continue
        direction=(predicted[a]-predicted[b])*(observed[a]-observed[b])
        concordant+=direction>0;discordant+=direction<0
    return dict(distinct_predictions=len(set(predicted.values())),prediction_tied_pairs=ties,
        concordant_pairs=concordant,discordant_pairs=discordant,
        MAE_ms=statistics.mean(abs(predicted[i]-observed[i]) for i in predicted),
        mean_signed_error_ms=statistics.mean(predicted[i]-observed[i] for i in predicted))

def analyze(root):
    from .campaign import validate_state,tasks,Recorder,freeze_document
    root=Path(root).resolve();state=validate_state(root);bundle=Path(state['bundle']);c=api(bundle)
    from .prepare import validate_bundle
    validate_bundle(bundle)
    if c.read(root/'selection_freeze.json')!=freeze_document(bundle):raise ValueError('selection freeze differs')
    ranked=c.read(bundle/'plans/selection_rankings.json');pool=c.read(bundle/'plans/static_pool.json')
    scores=c.read(bundle/'plans/model_scores.json');recorder=Recorder(root,state);measurements={};receipts=[]
    for task in tasks(bundle):
        result=recorder.saved(task)
        receipt=c.read(root/'accepted'/state['boot_id']/(task['task_id']+'.json'));receipts.append(receipt)
        if task['kind']=='performance':
            rows=[json.loads(line) for line in (root/receipt['raw']/'result.jsonl').read_text().splitlines() if line][8:24]
            gaps=[b['completion_ms']-a['completion_ms'] for a,b in zip(rows,rows[1:])]
            if len(gaps)!=15 or abs(statistics.mean(gaps)-result['ii_ms'])>1e-8:
                raise ValueError('completion interval reconstruction differs')
            measurements[task['candidate_id']]={**result,'gap_min_ms':min(gaps),'gap_max_ms':max(gaps),
                'gap_cv':statistics.pstdev(gaps)/statistics.mean(gaps)}
    if set(measurements)!=set(pool):raise ValueError('incomplete candidate union')
    lists={key:[r['id'] for r in rows] for key,rows in ranked.items()};decision=evaluate(lists,measurements)
    out=root/'analysis';out.mkdir(exist_ok=True)
    def write(name,rows):
        with (out/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    budgets=[dict(model=key,label=row['label'],budget=int(n),**v)
        for key,row in decision['methods'].items() for n,v in row['budgets'].items()]
    write('selection_budgets.csv',budgets);write('deletion_effects.csv',decision['deletion'])
    write('factorial_effects.csv',decision['factorial_effects']);write('list_overlap.csv',decision['overlaps'])
    positions={key:{r['id']:i+1 for i,r in enumerate(rows)} for key,rows in ranked.items()}
    from cpu_policies.open_resources.costs import Costs
    from .model import capacity_binding
    records=c.read(bundle/'plans/isolated_costs.json')['costs'];ddr=c.read(bundle/'plans/ddr.json')
    programs=c.read(bundle/'plans/programs.json')
    tables={flag:Costs(records,programs=programs,ddr=ddr,distribution='session_weights' if flag else 'balanced')
        for flag in (False,True)}
    configs=[];constraints=[]
    for name,cfg in pool.items():
        cpu=[s for s in cfg['stages'] if s['device']=='cpu'];m=measurements[name]
        configs.append(dict(id=name,partition=' → '.join(('CPU' if s['device']=='cpu' else 'VTA')+
            str(s['begin'])+'–'+str(s['end']-1) for s in cfg['stages']),
            context=cfg['context_id'],gate=cfg['execution_mode'],CPU_threads='+'.join(str(s['threads']) for s in cpu),
            CPU_masks=str([s['mask'] for s in cpu]),worker_maps=str([s['worker_core_map'] for s in cpu]),
            host_mask=str(cfg['host_mask']),control_mask=str(cfg['control_mask']),
            measured_ii_ms=m['ii_ms'],fps=1000/m['ii_ms'],
            kept_timestamps=m['kept_timestamps'],completion_intervals=m['gaps'],
            gap_min_ms=m['gap_min_ms'],gap_max_ms=m['gap_max_ms'],gap_cv=m['gap_cv'],
            p95_gap_ms=m['p95_gap_ms'],average_cpu_cores=m['average_cpu_cores'],
            cpu_time_scope=m['cpu_time_scope'],
            **{key+'_position':positions[key].get(name,'') for key,*_ in MODELS},
            **{key+'_predicted_ms':scores[key][name]['score_ns']/1e6 for key,*_ in MODELS}))
        for key,*_ in MODELS:
            v=scores[key][name]
            constraints.append(dict(id=name,model=key,label=settings(key)['label'],
                predicted_ii_ms=v['score_ns']/1e6,measured_ii_ms=m['ii_ms'],
                capacity_ms=v['terms_ns']['capacity']/1e6,
                host_constraint_ms=v['terms_ns'].get('host_concentration',0)/1e6 if settings(key)['host_guard'] else '',
                team_constraint_ms=v['terms_ns'].get('team_service',0)/1e6 if settings(key)['team_guard'] else '',
                binding_terms=';'.join(v['binding_terms']),
                capacity_binding_terms=';'.join(capacity_binding(cfg,tables[settings(key)['thread_weights']]))))
    write('configurations.csv',configs);write('constraint_accounting.csv',constraints)
    observed={i:m['ii_ms'] for i,m in measurements.items()};metrics=[]
    for key,*_ in MODELS:
        for scope,ids in [('common_union',list(pool)),('own_top20',lists[key])]:
            metrics.append(dict(model=key,label=settings(key)['label'],scope=scope,count=len(ids),
                **tied_metrics({i:scores[key][i]['score_ns']/1e6 for i in ids},observed)))
    write('prediction_quality.csv',metrics)
    proofs=c.read(bundle/'plans/search_proofs.json')
    costs=dict(configurations=len(pool),native_sessions=len(receipts),complete_DNN_inferences=29*len(pool),
        new_component_profiles=0,compiled_modules=0,compiled_runners=0,runtime_changes=False,
        measurement_and_collection_seconds=sum(r['measurement_and_collection_seconds'] for r in receipts),
        raw_files=sum(len(r['files']) for r in receipts),
        preparation=c.read(bundle/'plans/preparation_cost.json'),
        search_verification_and_compute_seconds=sum(p['elapsed_seconds'] for p in proofs.values()),
        search_per_model={key:dict(elapsed_seconds=p['elapsed_seconds'],cache_replayed=p['cache_replayed']) for key,p in proofs.items()},
        saved_configurations_by_deduplication=160-len(pool),
        saved_native_sessions_by_deduplication=2*(160-len(pool)),
        search_time_note='per-model worker wall times sum; parallel execution means this is not total elapsed wall time')
    result=dict(boot_id=state['boot_id'],decision=decision,costs=costs,prediction_quality=metrics,
        pool_best=min(configs,key=lambda row:(row['measured_ii_ms'],row['id'])),
        scope='one input, one boot, short windows; model-selection ablation, not causal decomposition of OS behavior',
        unresolved=['whole-session thread shares approximate scored compute shares',
            'same-thread-count core permutations are a model assumption',
            'host/team constraints are planning proxies, not latency bounds',
            'control work, scheduling, synchronization and cache effects remain incompletely modeled'])
    c.write(out/'analysis.json',result,replace=True)
    lines=['# ResNet18三因素消融：八组Top-20共同实测','',
        f'八组严格Top-20去重为{len(pool)}项，全部使用输入0，在同一次启动共同测量。',
        '每项一次单帧资格检查，再测8帧预热＋16完成时间戳＋4尾帧；16时间戳对应15间隔。','',
        '|模型|一次预算FPS|五次预算FPS|二十次预算FPS|','|---|---:|---:|---:|']
    for key,*_ in MODELS:
        row=decision['methods'][key]
        lines.append('|'+row['label']+'|'+'|'.join(f"{row['budgets'][n]['fps']:.3f}" for n in ('1','5','20'))+'|')
    lines+=['','删除某因素的代价＝完整模型FPS／删项模型FPS−1。负值表示删项版本在本次观察中更快。','',
        '|删除因素|一次预算代价|五次预算代价|二十次预算代价|','|---|---:|---:|---:|']
    for factor in ('线程负载份额','主机集中约束','线程组服务约束'):
        values=[r['removal_cost_percent'] for r in decision['deletion'] if r['factor']==factor]
        lines.append('|'+factor+'|'+'|'.join(f'{v:+.2f}%' for v in values)+'|')
    lines+=['','相同执行配置共用测量，选择相同如实报告零选点差异；未出现改善不解释为统计等效。',
        '八组合使用x∈{−1,+1}编码，系数为sum(sign×log FPS)/8，因素效应为系数的两倍。',
        '主效应与交互只作描述性分析；不将不同配置、帧或八种方法当作独立重复，不报告显著性。','',
        f"共{costs['native_sessions']}会话、{costs['complete_DNN_inferences']}次完整推理；测量与回收{costs['measurement_and_collection_seconds']:.1f}秒。",
        '没有新增组件标定或改动执行器。继承的775项成本及87项DDR记录保留原来源，旧FPS不计入本次比较。',
        '基础模型是本项目均分核时容量模型，不是Tarnawski原作者实现。',
        '本批考察评分改变后的选点效果；实际切图与资源配置可随版本改变，不宣称是同一配置的调度因果实验。','',
        '[预算](selection_budgets.csv) · [删项效果](deletion_effects.csv) · [交互](factorial_effects.csv) · [全部配置](configurations.csv) · [约束主导](constraint_accounting.csv)']
    (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    c.write(out/'VERIFICATION.json',dict(rankings_frozen_before_measurement=True,eight_models=True,
        full_domain_count_per_model=966240306,all_qualifications_and_raw_hashes_replayed=True,
        same_boot=True,automatic_remeasurement=False,historical_special_candidates_added=False,**costs),replace=True)
    plot(out,decision)
    return dict(report=str(out/'REPORT_zh.md'),pool_size=len(pool),accepted_sessions=len(receipts))

def plot(output,decision):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5.6),constrained_layout=True)
    keys=[r[0] for r in MODELS]
    short=['Base','Thread\nshares','Host\nconstraint','Team\nconstraint','Shares\n+host','Shares\n+team','Host\n+team','Complete']
    for i,n in enumerate(('1','5','20')):
        axes[0].bar([j+(i-1)*.26 for j in range(8)],
            [decision['methods'][key]['budgets'][n]['fps'] for key in keys],width=.26,label='Budget '+n)
    axes[0].set(xticks=range(8),xticklabels=short,ylabel='Best observed FPS',title='Eight formulas, common measurement union')
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.27),ncol=3)
    axes[0].tick_params(axis='x',rotation=35)
    for j,factor in enumerate(('线程负载份额','主机集中约束','线程组服务约束')):
        rows=[r for r in decision['deletion'] if r['factor']==factor]
        axes[1].plot([1,5,20],[r['removal_cost_percent'] for r in rows],'o-',
            label=['Remove thread shares','Remove host constraint','Remove team constraint'][j])
    axes[1].axhline(0,color='gray',linewidth=.8);axes[1].set(xticks=[1,5,20],xlabel='Measurement budget',
        ylabel='Complete / without factor FPS - 1 (%)',title='Leave-one-factor-out effect')
    axes[1].legend();fig.suptitle('ResNet18 / single input / single boot / descriptive results only')
    for suffix in ('png','pdf'):fig.savefig(Path(output)/('ablation_summary.'+suffix),dpi=170)
    plt.close(fig)
