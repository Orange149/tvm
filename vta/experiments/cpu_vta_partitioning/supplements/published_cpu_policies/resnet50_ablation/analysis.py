"""Shared-pool selection effects, interactions, prediction errors and costs."""
from itertools import combinations
from pathlib import Path
import csv
import math
import statistics

from cpu_vta_partitioning.common import read, sha256, write
from resnet50_validation import analysis as old_analysis, runtime
from .common import ROOT, SOURCE, PERFORMANCE, MODELS, LABELS, settings, assert_controller
from .model import costs_for, score
from .workflow import validate_bundle

ZH_LABELS=dict(zip(MODELS,('粗粒度CPU，不计主机需求','粗粒度CPU，计入主机需求',
    '细化CPU，不计主机需求','细化CPU，计入主机需求','当前完整模型（含主机集中约束）')))


def selection_budgets(lists, results):
    rows=[]
    for key in MODELS:
        ids=[r['id'] for r in lists[key]]
        for budget in (1,5,20):
            name=min(ids[:budget],key=lambda n:(results[n]['II_ms'],n))
            rows.append(dict(model=key,budget=budget,static_rank=ids.index(name)+1,
                configuration=name,II_ms=results[name]['II_ms'],FPS=results[name]['FPS']))
    return rows


def effects(budgets):
    lookup={(r['model'],r['budget']):r['FPS'] for r in budgets}
    comparisons=(('CPU refinement, host absent','fine_no_host','coarse_no_host'),
        ('CPU refinement, host included','fine_host','coarse_host'),
        ('Host demand, coarse CPU','coarse_host','coarse_no_host'),
        ('Host demand, fine CPU','fine_host','fine_no_host'),
        ('Host concentration','complete','fine_host'),
        ('All scoring additions','complete','coarse_no_host'))
    result=[];interactions=[]
    for b in (1,5,20):
        for label,new,base in comparisons:
            x,y=lookup[new,b],lookup[base,b]
            result.append(dict(comparison=label,budget=b,model=new,reference=base,
                FPS=x,reference_FPS=y,FPS_gain_percent=100*(x/y-1),
                II_reduction_percent=100*(1-y/x)))
        d=(math.log(lookup['fine_host',b])-math.log(lookup['fine_no_host',b])
           -math.log(lookup['coarse_host',b])+math.log(lookup['coarse_no_host',b]))
        interactions.append(dict(budget=b,CPU_host_log_interaction=d,
                                 multiplicative_ratio=math.exp(d),descriptive_only=True))
    return result,interactions


def cross_network_comparison(contributions):
    """Within-batch effects only; different protocols remain separate evidence."""
    base=ROOT.parents[4]/'paper/my_paper/analysis'
    staged=base/'20261109_staged_ablation/selection_budgets.csv'
    host=base/'20261109_host_work_ablation/selection_budgets.csv'
    def load(path):
        with path.open(encoding='utf-8-sig') as stream:
            return {(r['model'],int(r['budget'])):float(r['fps']) for r in csv.DictReader(stream)}
    a,b=load(staged),load(host);rows=[]
    for item in contributions:
        if item['comparison']=='All scoring additions':
            continue  # No same-batch ResNet18 full-five-group Top20 comparison.
        concentrated=item['comparison']=='Host concentration'
        lookup=b if concentrated else a
        first,last=('complete','included') if concentrated else (item['model'],item['reference'])
        key=item['budget']
        rows.append(dict(comparison=item['comparison'],budget=key,
            ResNet50_within_batch_FPS_gain_percent=item['FPS_gain_percent'],
            ResNet18_within_batch_FPS_gain_percent=100*(lookup[first,key]/lookup[last,key]-1),
            ResNet18_source=str(host if concentrated else staged),
            ResNet18_source_sha256=sha256(host if concentrated else staged),
            ResNet18_tail_frames=4,ResNet50_tail_frames=24,
            interpretation='separate within-batch effects; different protocols/boots, not workload-only causality'))
    return rows


def figures(target,budgets,bridge):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lookup={(r['model'],r['budget']):r['FPS'] for r in budgets}
    fig,ax=plt.subplots(figsize=(11,4.8))
    names=['Coarse CPU\nNo host','Coarse CPU\nHost','Fine CPU\nNo host',
           'Fine CPU\nHost','Complete']
    for j,b in enumerate((1,5,20)):
        xs=[i+(j-1)*.24 for i in range(5)]
        bars=ax.bar(xs,[lookup[k,b] for k in MODELS],width=.24,label=f'Budget {b}')
        ax.bar_label(bars,fmt='%.3f',fontsize=7,padding=2,rotation=90)
    ax.set_xticks(range(5),names);ax.set_ylabel('Throughput (FPS)')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.16),ncol=3,frameon=False)
    ax.set_ylim(0,max(r['FPS'] for r in budgets)*1.22)
    fig.text(.5,.01,'One input, one boot; each policy has a strict top-20. Baseline bridges are reported separately.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.95))
    for ext in ('png','pdf','svg'):fig.savefig(target/('five_policy_budgets.'+ext),dpi=240)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),sharey=True)
    for ax,b in zip(axes,(1,5,20)):
        for label,keys in [('No host demand',('coarse_no_host','fine_no_host')),
                           ('Host demand',('coarse_host','fine_host'))]:
            vals=[lookup[k,b] for k in keys];ax.plot([0,1],vals,marker='o',label=label)
        ax.set_xticks([0,1],['Coarse CPU','Fine CPU']);ax.set_title(f'Budget {b}')
    axes[0].set_ylabel('Throughput (FPS)');axes[-1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(target/('CPU_host_interaction.'+ext),dpi=240)
    plt.close(fig)


def analyze(output,run_dir):
    output,run_dir=Path(output),Path(run_dir)
    assert_controller(output);bundle=output/'pipeline_bundle';validate_bundle(bundle)
    state=read(run_dir/'pipeline/state.json')
    if state['status']!='COMPLETE' or not state['post_deployment_hashes_passed']:
        raise ValueError('the entire common pool and post-run hashes must pass')
    boot=read(run_dir/'run.json')['boot_id']
    if state['boot_id']!=boot:
        raise ValueError('performance boot mixed')
    pool=read(bundle/'plans/pool.json');lists=read(bundle/'plans/rankings.json')['lists']
    records=read(bundle/'provenance/static_inputs/isolated_costs.json')['costs']
    programs=read(bundle/'plans/programs.json');ddr=read(bundle/'provenance/static_inputs/ddr.json')
    results={};receipts=[];hashes={}
    for name,cfg in pool.items():
        expected=None
        for phase in ('qualify','short'):
            task=runtime.task(bundle,cfg,phase)
            path=run_dir/'pipeline/accepted'/boot/(task['task_id']+'.json')
            item=old_analysis.receipt(path,task,boot,bundle=bundle)
            receipts.append(item);hashes[str(path)]=sha256(path)
            if phase=='qualify':expected=item['summary']['output_fingerprints']
            elif item['summary']['output_fingerprints']!=expected:
                raise ValueError('qualification and timed outputs differ')
            else:results[name]=item['summary']
    target=output/'analysis';target.mkdir(parents=True,exist_ok=True)
    budgets=selection_budgets(lists,results);contributions,interactions=effects(budgets)
    bridges=[]
    with (PERFORMANCE/'analysis/budgets.csv').open(encoding='utf-8-sig') as stream:
        historical={int(r['budget']):r for r in csv.DictReader(stream) if r['method']=='tarnawski'}
    for row in read(bundle/'plans/bridges.json'):
        name=row['configuration']['id'];r=results[name]
        prior=historical[row['previous_budget']]
        if prior['configuration']!=name:
            raise ValueError('historical baseline selection no longer matches the fixed bridge')
        bridges.append(dict(previous_budget=row['previous_budget'],original_static_rank=row['original_static_rank'],
            configuration=name,FPS=r['FPS'],II_ms=r['II_ms'],
            historical_FPS=float(prior['FPS']),
            FPS_change_vs_historical_percent=100*(r['FPS']/float(prior['FPS'])-1),
            interpretation='new measurement of old selected configuration; not a new baseline budget optimum'))
    all_scores=[];constraints=[];prediction=[]
    for key in MODELS:
        costs=costs_for(key,records,programs,ddr);errors=[]
        ranks={r['id']:i+1 for i,r in enumerate(lists[key])}
        for name,cfg in pool.items():
            value=score(cfg,costs,key);actual=results[name]['II_ms'];pred=value['score_ns']/1e6
            error=100*(pred/actual-1);errors.append(abs(error))
            all_scores.append(dict(model=key,configuration=name,static_rank=ranks.get(name),
                predicted_II_ms=pred,actual_II_ms=actual,FPS=results[name]['FPS'],
                signed_error_percent=error,absolute_error_percent=abs(error),
                context=cfg['context_id'],gate=cfg['execution_mode'],
                stages=[s['program'] for s in cfg['stages']],
                CPU_threads=[s['threads'] for s in cfg['stages'] if s['device']=='cpu'],
                CPU_core_sets=[s['mask'] for s in cfg['stages'] if s['device']=='cpu'],
                host_mask=cfg['host_mask'],control_mask=cfg['control_mask'],
                CPU_core_demand_ns=value['CPU_core_demand_ns'],
                observed_host_CPU_ns=value['observed_host_cpu_ns'],
                scored_host_CPU_ns=value['scored_host_cpu_ns'],
                binding_terms=value['binding_terms'],capacity_binding_terms=value['capacity_binding_terms']))
            for scope,terms in [('score',value['terms_ns']),('capacity',value['capacity_terms_ns'])]:
                for term,demand in terms.items():
                    constraints.append(dict(model=key,configuration=name,scope=scope,term=term,demand_ns=demand,
                                            equals_final_score=demand==value['score_ns']))
        prediction.append(dict(model=key,common_pool_count=len(pool),MAPE_percent=statistics.mean(errors),
            Top20_distinct_scores=len({r['score_ns'] for r in lists[key]}),
            first_score_ties=sum(r['score_ns']==lists[key][0]['score_ns'] for r in lists[key])))
    overlap=[]
    for a,b in combinations(MODELS,2):
        x={r['id'] for r in lists[a]};y={r['id'] for r in lists[b]}
        overlap.append(dict(model_a=a,model_b=b,intersection=len(x&y),Jaccard=len(x&y)/len(x|y)))
    tables={'budgets.csv':budgets,'bridge_results.csv':bridges,'contributions.csv':contributions,
        'CPU_host_interaction.csv':interactions,'common_pool_predictions.csv':all_scores,
        'cross_network_effects.csv':cross_network_comparison(contributions),
        'constraint_scores.csv':constraints,'prediction_summary.csv':prediction,'list_overlap.csv':overlap,
        'stage_isolated_pipeline.csv':old_analysis.stage_rows(pool,results,records,programs,ddr),
        'CPU_thread_curves.csv':old_analysis.thread_curve_rows(records)}
    for name,rows in tables.items():old_analysis._csv(target/name,rows)
    search_costs={}
    for key in MODELS:
        doc=read(output/'ranking/models'/key/'RESULTS.json');p=doc['proof']
        search_costs[key]=dict(reused_original_ranking=key=='complete',
            new_search_seconds=0 if key=='complete' else p['elapsed_seconds'],
            statistics=p.get('statistics'),verification_and_preparation_seconds=doc['preparation_seconds'])
    cost=dict(new_calibration=0,new_DMA=0,new_native_builds=0,
        native_build_scope='stage modules, runtime and board runner; local C++ verification builds are separate',
        offline_small_validation_seconds=read(output/'acceptance/small/ACCEPTANCE.json')['elapsed_seconds'],
        offline_ResNet18_replay_seconds=read(output/'acceptance/resnet18/ACCEPTANCE.json')['elapsed_seconds'],
        performance=old_analysis.measurement_cost(receipts),search=search_costs,
        referenced_source_cost=read(PERFORMANCE/'analysis/COSTS.json')['referenced_source_work'],
        referenced_cost_report_sha256=sha256(PERFORMANCE/'analysis/COSTS.json'),
        deployment=read(run_dir/'deployment_cost.json'),
        new_raw_bytes=sum(sum(f['bytes'] for f in r['files'].values()) for r in receipts))
    report=dict(status='COMPLETE',network='resnet50_v2',performance_boot_id=boot,
        profile_boot_ids=sorted({r['source']['boot_id'] for r in records.values()}),
        unique_configurations=len(pool),native_sessions=len(receipts),complete_DNN_inferences=49*len(pool),
        budgets=budgets,contributions=contributions,interactions=interactions,bridge_results=bridges,
        receipt_sha256=hashes,source_freeze_sha256=sha256(output/'ranking/FREEZE.json'),
        significance_claimed=False,pure_thread_domain_contribution_identified=False,
        whole_session_thread_share_approximation=True,CPU_refinement_groups_two_factors=True,
        control_and_scheduler_overhead_completely_explained=False)
    write(target/'RESULTS.json',report,replace=True);write(target/'COSTS.json',cost,replace=True)
    write(target/'METHOD_DEFINITIONS.json',
          {k:dict(settings(k),Chinese_label=ZH_LABELS[k]) for k in MODELS},replace=True)
    lines=['ResNet50：五组同域消融与三项基线桥接','',
        f'五组Top-20共同测量，去重{len(pool)}项；{len(receipts)}个会话，{49*len(pool)}次完整推理。',
        '输入0；同一性能启动；资格1帧，正式8预热＋16时间戳＋24尾帧，按15间隔计算。','']
    for key in MODELS:
        vals=[next(r['FPS'] for r in budgets if r['model']==key and r['budget']==b) for b in (1,5,20)]
        lines.append(ZH_LABELS[key]+'：预算1/5/20 FPS '+ ' / '.join(f'{v:.6f}' for v in vals))
    lines+=['','同预算条件贡献：']
    for row in contributions:
        lines.append(f"{row['comparison']}，预算{row['budget']}：{row['FPS_gain_percent']:+.3f}%。")
    lines+=['','基线三项为旧预算选点的新桥接测量，不是重新测完基线Top-20后的预算最优。',
        'CPU细化为非均匀线程份额＋线程组服务的组合；不单独量化纯线程域扩展。',
        '切图、线程、核组和门控均可能随评分改变；结果评价选点，不代表同配置运行机制被改变。',
        '同分与名单选择相同时如实报告零选点差异；不报告显著性、统计等效或全域实测最优。',
        '隔离与流水阶段差异可能包含跨启动条件及未建模竞争；不相加并发阶段时间或等待。',
        '旧ResNet18的消融窗口不同，不跨批相减分解本轮收益。']
    (target/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')
    figures(target,budgets,bridges)
    return dict(status='COMPLETE',candidates=len(pool),budgets=budgets,results=str(target/'RESULTS.json'))
