"""Post-freeze three-model comparison. Search never imports this module."""
import csv
from itertools import combinations
import json
import math
import os
from pathlib import Path
import statistics

from .common import MODELS, settings, api
from model_ablation.analysis import tied_metrics

COMPARISONS = (
    ('计入主机CPU需求的贡献', 'included', 'ignored'),
    ('主机集中约束的额外贡献', 'complete', 'included'),
    ('主机建模的组合贡献', 'complete', 'ignored'),
)

def evaluate(lists, measurements):
    if set(lists) != {r[0] for r in MODELS}:
        raise ValueError('three complete Top-20 lists required')
    methods = {}
    for key, ids in lists.items():
        if len(ids)!=20 or len(set(ids))!=20:
            raise ValueError('twenty unique configurations required')
        if any(i not in measurements or not math.isfinite(measurements[i]['ii_ms'])
               or measurements[i]['ii_ms']<=0 for i in ids):
            raise ValueError('missing or invalid pipeline measurement')
        budgets = {}
        for n in (1, 5, 20):
            selected = min(ids[:n], key=lambda i: (measurements[i]['ii_ms'], i))
            ii = measurements[selected]['ii_ms']
            budgets[str(n)] = dict(selected=selected, static_position=ids.index(selected)+1,
                                   ii_ms=ii, fps=1000/ii)
        methods[key] = dict(label=settings(key)['label'], members=ids, budgets=budgets)
    comparisons = []
    for label, treatment, baseline in COMPARISONS:
        for n in ('1', '5', '20'):
            a=methods[treatment]['budgets'][n]; b=methods[baseline]['budgets'][n]
            comparisons.append(dict(comparison=label, treatment=treatment, baseline=baseline,
                budget=int(n), treatment_configuration=a['selected'], baseline_configuration=b['selected'],
                treatment_fps=a['fps'], baseline_fps=b['fps'],
                fps_improvement_percent=100*(a['fps']/b['fps']-1),
                ii_reduction_percent=100*(1-a['ii_ms']/b['ii_ms']),
                selected_configuration_same=a['selected']==b['selected']))
    overlaps=[]
    for a,b in combinations(lists,2):
        for n in (1,5,20):
            aa,bb=set(lists[a][:n]),set(lists[b][:n])
            overlaps.append(dict(model_a=a,model_b=b,budget=n,intersection=len(aa&bb),
                Jaccard=len(aa&bb)/len(aa|bb),ordered_lists_identical=lists[a][:n]==lists[b][:n]))
    return dict(methods=methods, comparisons=comparisons, overlaps=overlaps,
        thread_weights_enabled=True, team_constraint_enabled=True,
        single_boot_observation_only=True, independent_confirmation=False)

def analyze(root):
    from .campaign import validate_state, tasks, Recorder, freeze_document
    from .prepare import validate_bundle
    root=Path(root).resolve(); state=validate_state(root); bundle=Path(state['bundle']); c=api(bundle)
    validate_bundle(bundle)
    if c.read(root/'selection_freeze.json')!=freeze_document(bundle):
        raise ValueError('selection freeze changed')
    rankings=c.read(bundle/'plans/selection_rankings.json'); pool=c.read(bundle/'plans/static_pool.json')
    scores=c.read(bundle/'plans/model_scores.json'); recorder=Recorder(root,state)
    measurements={}; receipts=[]
    for task in tasks(bundle):
        result=recorder.saved(task)
        receipt=c.read(root/'accepted'/state['boot_id']/(task['task_id']+'.json')); receipts.append(receipt)
        if task['kind']=='performance':
            rows=[json.loads(x) for x in (root/receipt['raw']/'result.jsonl').read_text().splitlines() if x][8:24]
            gaps=[b['completion_ms']-a['completion_ms'] for a,b in zip(rows,rows[1:])]
            if len(gaps)!=15 or abs(statistics.mean(gaps)-result['ii_ms'])>1e-8:
                raise ValueError('16 timestamps / 15 intervals reconstruction failed')
            measurements[task['candidate_id']]={**result, 'gap_min_ms':min(gaps), 'gap_max_ms':max(gaps),
                'gap_cv':statistics.pstdev(gaps)/statistics.mean(gaps)}
    if set(measurements)!=set(pool):raise ValueError('candidate union not complete')
    lists={key:[r['id'] for r in rows] for key,rows in rankings.items()}
    decision=evaluate(lists,measurements)
    out=root/'analysis'; out.mkdir(exist_ok=True)
    def write(name, rows):
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    budgets=[dict(model=k,label=v['label'],budget=int(n),**b)
             for k,v in decision['methods'].items() for n,b in v['budgets'].items()]
    write('selection_budgets.csv',budgets)
    write('host_effects.csv',decision['comparisons']); write('list_overlap.csv',decision['overlaps'])
    positions={key:{r['id']:i+1 for i,r in enumerate(rows)} for key,rows in rankings.items()}
    configs=[]; constraints=[]
    for name,cfg in pool.items():
        m=measurements[name];cpu=[s for s in cfg['stages'] if s['device']=='cpu']
        configs.append(dict(id=name,partition=' → '.join(('CPU' if s['device']=='cpu' else 'VTA')+
            str(s['begin'])+'–'+str(s['end']-1) for s in cfg['stages']),
            context=cfg['context_id'],gate=cfg['execution_mode'],CPU_threads='+'.join(str(s['threads']) for s in cpu),
            CPU_masks=str([s['mask'] for s in cpu]),worker_maps=str([s['worker_core_map'] for s in cpu]),
            host_mask=str(cfg['host_mask']),control_mask=str(cfg['control_mask']),
            measured_ii_ms=m['ii_ms'],fps=1000/m['ii_ms'],kept_timestamps=m['kept_timestamps'],
            completion_intervals=m['gaps'],gap_min_ms=m['gap_min_ms'],gap_max_ms=m['gap_max_ms'],gap_cv=m['gap_cv'],
            p95_gap_ms=m['p95_gap_ms'],average_cpu_cores=m['average_cpu_cores'],cpu_time_scope=m['cpu_time_scope'],
            **{k+'_position':positions[k].get(name,'') for k,*_ in MODELS},
            **{k+'_predicted_ms':scores[k][name]['score_ns']/1e6 for k,*_ in MODELS}))
        for key,*_ in MODELS:
            s=scores[key][name]
            constraints.append(dict(id=name,model=key,label=settings(key)['label'],
                predicted_ii_ms=s['score_ns']/1e6,measured_ii_ms=m['ii_ms'],
                observed_host_cpu_ms=s['observed_host_cpu_ns']/1e6,
                scored_host_cpu_ms=s['scored_host_cpu_ns']/1e6,
                excluded_host_cpu_ms=s['excluded_host_cpu_ns']/1e6,
                CPU_core_demand_ms=str([x/1e6 for x in s['CPU_core_demand_ns']]),
                VTA_service_sum_ms=s['VTA_service_sum_ns']/1e6,DDR_service_ms=s['DDR_service_ns']/1e6,
                capacity_ms=s['terms_ns']['capacity']/1e6,team_service_ms=s['terms_ns']['team_service']/1e6,
                host_concentration_ms=s['terms_ns'].get('host_concentration',0)/1e6 if settings(key)['host_guard'] else '',
                binding_terms=';'.join(s['binding_terms']),capacity_binding_terms=';'.join(s['capacity_binding_terms'])))
    write('configurations.csv',configs);write('constraint_accounting.csv',constraints)
    observed={i:v['ii_ms'] for i,v in measurements.items()};metrics=[]
    for key,*_ in MODELS:
        for scope,ids in [('common_union',list(pool)),('own_top20',lists[key])]:
            metrics.append(dict(model=key,label=settings(key)['label'],scope=scope,count=len(ids),
                **tied_metrics({i:scores[key][i]['score_ns']/1e6 for i in ids},observed)))
    write('prediction_quality.csv',metrics)
    proofs=c.read(bundle/'plans/search_proofs.json')
    costs=dict(configurations=len(pool),native_sessions=len(receipts),complete_DNN_inferences=29*len(pool),
        new_component_profiles=0,compiled_modules=0,compiled_runners=0,runtime_changes=False,
        measurement_and_collection_seconds=sum(r['measurement_and_collection_seconds'] for r in receipts),
        raw_files=sum(len(r['files']) for r in receipts),preparation=c.read(bundle/'plans/preparation_cost.json'),
        search_verification_and_compute_seconds=sum(p['elapsed_seconds'] for p in proofs.values()),
        search_per_model={k:dict(elapsed_seconds=p['elapsed_seconds'],cache_replayed=p['cache_replayed']) for k,p in proofs.items()},
        saved_configurations_by_deduplication=60-len(pool),saved_native_sessions_by_deduplication=2*(60-len(pool)),
        search_time_note='per-model worker wall-time sum; actual total rank elapsed is recorded separately')
    result=dict(boot_id=state['boot_id'],decision=decision,costs=costs,prediction_quality=metrics,
        pool_best=min(configs,key=lambda r:(r['measured_ii_ms'],r['id'])),
        scope='single input, single boot, short windows; conditional host-model selection effects with thread shares and team constraint enabled',
        unresolved=['VTA wall service still includes host-side elapsed work in all three models',
            'whole-session thread shares and homogeneous core permutations remain approximations',
            'host concentration and team service are planning proxies, not proven latency bounds',
            'producer, collector, synchronization and cache effects are not separately fully modeled'])
    c.write(out/'analysis.json',result,replace=True)
    lines=['# ResNet18：VTA主机CPU需求三组消融','',
        f'三个完整域Top-20去重为{len(pool)}项，在同一次启动、输入0下共同测量。',
        '每项单帧资格检查，再测8帧预热＋16完成时间戳＋4尾帧；16时间戳对应15间隔。','',
        '|模型|一次预算FPS|五次预算FPS|二十次预算FPS|','|---|---:|---:|---:|']
    for key,*_ in MODELS:
        m=decision['methods'][key]
        lines.append('|'+m['label']+'|'+'|'.join(f"{m['budgets'][n]['fps']:.3f}" for n in ('1','5','20'))+'|')
    lines+=['','|比较|一次预算FPS变化|五次预算FPS变化|二十次预算FPS变化|','|---|---:|---:|---:|']
    for label,_,_ in COMPARISONS:
        rows=[r for r in decision['comparisons'] if r['comparison']==label]
        lines.append('|'+label+'|'+'|'.join(f"{r['fps_improvement_percent']:+.2f}%" for r in rows)+'|')
    lines+=['','FPS变化＝所比较版本FPS／对照版本FPS−1；负值表示本轮所比较版本更慢。完成间隔下降另列在host_effects.csv，避免混淆分母。',
        '本轮固定开启线程负载份额和线程组服务约束。计入需求／忽略需求比较显式主机CPU占用；完整／计入需求比较集中约束；完整／忽略需求比较组合效果。',
        '忽略版本只排除所有VTA阶段的运行及交接CPU核时，不改变真实执行，不移除VTA墙钟服务中的主机影响。',
        '原始775项组件成本及87项DDR记录只读，不新增标定；CPU自身交接、单物理VTA累计和DDR约束三组一致。',
        '评分版本不进入执行身份；相同配置共用测量，选点相同报告零差异。不同配置与帧不作为独立重复，不报告显著性或等效性。',
        '本轮全部FPS来自名单冻结后的共同实测，不复用旧批次性能，也不据此完成操作系统行为的因果归因。','',
        f"共{costs['native_sessions']}个会话、{costs['complete_DNN_inferences']}次完整推理；测量与回收{costs['measurement_and_collection_seconds']:.1f}秒。",
        '', '[预算](selection_budgets.csv) · [主机选点贡献](host_effects.csv) · [全部配置](configurations.csv) · [约束记账](constraint_accounting.csv) · [名单重合](list_overlap.csv)']
    (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    c.write(out/'VERIFICATION.json',dict(rankings_frozen_before_measurement=True,three_models=True,
        full_domain_count_per_model=966240306,all_qualifications_and_raw_hashes_replayed=True,same_boot=True,
        automatic_remeasurement=False,historical_special_candidates_added=False,old_pipeline_performance_reused=False,**costs),replace=True)
    plot(out,decision)
    return dict(report=str(out/'REPORT_zh.md'),pool_size=len(pool),accepted_sessions=len(receipts))

def plot(output,decision):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/cpu-vta-host-work-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family']=['DejaVu Sans',font_manager.FontProperties(fname=str(font)).get_name()]
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(12,5.2))
    keys=[r[0] for r in MODELS]
    for j,n in enumerate(('1','5','20')):
        axes[0].bar([i+(j-1)*.25 for i in range(3)],
            [decision['methods'][key]['budgets'][n]['fps'] for key in keys],width=.25,label='预算'+n)
    axes[0].set(xticks=range(3),xticklabels=['忽略主机CPU需求','计入主机CPU需求','完整模型'],
        ylabel='最佳实测吞吐量（FPS）',title='三份严格Top-20，共同实测')
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=3)
    for label,_,_ in COMPARISONS:
        rows=[r for r in decision['comparisons'] if r['comparison']==label]
        axes[1].plot([1,5,20],[r['fps_improvement_percent'] for r in rows],'o-',label=label)
    axes[1].axhline(0,color='gray',linewidth=.8)
    axes[1].set(xticks=[1,5,20],xlabel='完整流水选点预算',ylabel='相对对照版本的FPS变化（%）',title='分别报告两层主机建模作用')
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.17),fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('ResNet18：单输入、单启动、短窗口；固定线程份额及线程组约束')
    fig.subplots_adjust(bottom=.28,top=.84,wspace=.28)
    for suffix in ('png','pdf'):fig.savefig(Path(output)/('host_work_summary.'+suffix),dpi=170)
    plt.close(fig)
