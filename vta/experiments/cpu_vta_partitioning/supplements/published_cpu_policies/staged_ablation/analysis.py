"""Analyze the common measurement only after frozen selection is verified."""
import csv
from itertools import combinations
import math
import os
from pathlib import Path
import statistics
import json
from .common import MODELS,settings,api
from model_ablation.analysis import tied_metrics

COMPARISONS=(
    ('CPU细化贡献：不计主机需求','fine_no_host','coarse_no_host'),
    ('CPU细化贡献：计入主机需求','fine_host','coarse_host'),
    ('主机需求贡献：粗粒度CPU','coarse_host','coarse_no_host'),
    ('主机需求贡献：细化CPU','fine_host','fine_no_host'),
    ('本文主模型相对Tarnawski适配','fine_host','tarnawski'),
    ('增加线程组约束：已有线程份额','fine_host','weights_only'),
    ('增加线程份额：已有线程组约束','fine_host','team_only'),
    ('主机集中约束的额外作用','concentrated','fine_host'),
)

def evaluate(lists,measurements):
    if set(lists)!={r[0] for r in MODELS}:raise ValueError('all declared model lists required')
    methods={}
    for key,ids in lists.items():
        if len(ids)!=settings(key)['measurement_budget'] or len(set(ids))!=len(ids):raise ValueError('unique declared budget required')
        if any(i not in measurements or not math.isfinite(measurements[i]['ii_ms']) or measurements[i]['ii_ms']<=0 for i in ids):
            raise ValueError('missing valid performance')
        budgets={}
        for n in (1,5,20):
            if n>len(ids):continue
            selected=min(ids[:n],key=lambda i:(measurements[i]['ii_ms'],i));ii=measurements[selected]['ii_ms']
            budgets[str(n)]=dict(selected=selected,static_position=ids.index(selected)+1,ii_ms=ii,fps=1000/ii)
        methods[key]=dict(label=settings(key)['label'],members=ids,budgets=budgets)
    effects=[]
    for label,treatment,baseline in COMPARISONS:
        for n in ('1','5','20'):
            if n not in methods[treatment]['budgets'] or n not in methods[baseline]['budgets']:continue
            a=methods[treatment]['budgets'][n];b=methods[baseline]['budgets'][n]
            effects.append(dict(comparison=label,treatment=treatment,baseline=baseline,budget=int(n),
                treatment_configuration=a['selected'],baseline_configuration=b['selected'],
                treatment_fps=a['fps'],baseline_fps=b['fps'],fps_improvement_percent=100*(a['fps']/b['fps']-1),
                ii_reduction_percent=100*(1-a['ii_ms']/b['ii_ms']),selected_configuration_same=a['selected']==b['selected']))
    interactions=[]
    for n in ('1','5','20'):
        f={k:methods[k]['budgets'][n]['fps'] for k in ('coarse_no_host','coarse_host','fine_no_host','fine_host')}
        value=math.log(f['fine_host'])-math.log(f['fine_no_host'])-math.log(f['coarse_host'])+math.log(f['coarse_no_host'])
        interactions.append(dict(budget=int(n),CPU_host_log_interaction=value,
            multiplicative_interaction_percent=100*math.expm1(value),
            interpretation='descriptive interaction of selection budgets; not a causal attribution or significance test'))
    overlaps=[]
    for a,b in combinations(lists,2):
        for n in (1,5,20):
            if n>min(len(lists[a]),len(lists[b])):continue
            aa,bb=set(lists[a][:n]),set(lists[b][:n])
            overlaps.append(dict(model_a=a,model_b=b,budget=n,intersection=len(aa&bb),
                Jaccard=len(aa&bb)/len(aa|bb),ordered_lists_identical=lists[a][:n]==lists[b][:n]))
    return dict(methods=methods,comparisons=effects,interactions=interactions,overlaps=overlaps,
        single_boot_observation_only=True,independent_confirmation=False)

def analyze(root):
    from .campaign import validate_state,tasks,Recorder,freeze_document
    from .prepare import validate_bundle
    from cpu_policies.open_resources.costs import Costs
    root=Path(root).resolve();state=validate_state(root);bundle=Path(state['bundle']);c=api(bundle)
    validate_bundle(bundle)
    if c.read(root/'selection_freeze.json')!=freeze_document(bundle):raise ValueError('selection freeze changed')
    rankings=c.read(bundle/'plans/selection_rankings.json');pool=c.read(bundle/'plans/static_pool.json')
    scores=c.read(bundle/'plans/model_scores.json');recorder=Recorder(root,state)
    costs=Costs(c.read(bundle/'plans/isolated_costs.json')['costs'],programs=c.read(bundle/'plans/programs.json'),ddr=c.read(bundle/'plans/ddr.json'))
    measurements={};receipts=[];stages=[]
    for task in tasks(bundle):
        summary=recorder.saved(task);receipt=c.read(root/'accepted'/state['boot_id']/(task['task_id']+'.json'));receipts.append(receipt)
        if task['kind']!='performance':continue
        frames=[json.loads(s) for s in (root/receipt['raw']/'result.jsonl').read_text().splitlines() if s][8:24]
        gaps=[b['completion_ms']-a['completion_ms'] for a,b in zip(frames,frames[1:])]
        if len(gaps)!=15 or abs(statistics.mean(gaps)-summary['ii_ms'])>1e-8:raise ValueError('16 timestamps must reconstruct 15 intervals')
        measurements[task['candidate_id']]={**summary,'gap_min_ms':min(gaps),'gap_max_ms':max(gaps),
            'gap_cv':statistics.pstdev(gaps)/statistics.mean(gaps)}
        cfg=pool[task['candidate_id']]
        for i,row in enumerate(cfg['stages']):
            iso=costs.get(row,cfg);wall=statistics.mean(f['stage'+str(i)+'_ms'] for f in frames)
            stages.append(dict(configuration=cfg['id'],program=row['program'],device=row['device'],threads=row['threads'],
                core_mask=str(row['mask']),host_mask=str(cfg['host_mask']),gate=cfg['execution_mode'],
                isolated_service_ms=iso['service_ms'],pipeline_service_ms=wall,service_ratio=wall/iso['service_ms'],
                isolated_run_ms=iso['run_ms'],pipeline_run_ms=statistics.mean(f['stage'+str(i)+'_run_ms'] for f in frames),
                scope='same scored frame indices; stage wall times exclude gate waiting and must not be added across concurrent stages'))
    if set(measurements)!=set(pool):raise ValueError('candidate union incomplete')
    lists={k:[r['id'] for r in rows] for k,rows in rankings.items()};decision=evaluate(lists,measurements)
    out=root/'analysis';out.mkdir(exist_ok=True)
    def write(name,rows):
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    budgets=[dict(model=k,label=v['label'],budget=int(n),**b) for k,v in decision['methods'].items() for n,b in v['budgets'].items()]
    write('selection_budgets.csv',budgets);write('contribution_effects.csv',decision['comparisons'])
    write('CPU_host_interaction.csv',decision['interactions']);write('list_overlap.csv',decision['overlaps'])
    write('stage_profile_comparison.csv',stages)
    positions={k:{r['id']:i+1 for i,r in enumerate(rows)} for k,rows in rankings.items()};configs=[];constraints=[]
    for name,cfg in pool.items():
        m=measurements[name];cpu=[s for s in cfg['stages'] if s['device']=='cpu']
        configs.append(dict(id=name,partition=' → '.join(('CPU' if s['device']=='cpu' else 'VTA')+str(s['begin'])+'–'+str(s['end']-1) for s in cfg['stages']),
            selected_by=';'.join(settings(k)['label'] for k in positions if name in positions[k]),
            context=cfg['context_id'],gate=cfg['execution_mode'],CPU_threads='+'.join(str(s['threads']) for s in cpu),
            CPU_masks=str([s['mask'] for s in cpu]),worker_maps=str([s['worker_core_map'] for s in cpu]),
            host_mask=str(cfg['host_mask']),control_mask=str(cfg['control_mask']),measured_ii_ms=m['ii_ms'],fps=1000/m['ii_ms'],
            gap_min_ms=m['gap_min_ms'],gap_max_ms=m['gap_max_ms'],gap_cv=m['gap_cv'],average_cpu_cores=m['average_cpu_cores'],cpu_time_scope=m['cpu_time_scope'],
            **{k+'_position':positions[k].get(name,'') for k,*_ in MODELS},
            **{k+'_predicted_ms':scores[k][name]['score_ns']/1e6 if scores[k][name]['applicable'] else '' for k,*_ in MODELS}))
        for key,*_ in MODELS:
            s=scores[key][name]
            if not s['applicable']:continue
            constraints.append(dict(id=name,model=key,label=settings(key)['label'],predicted_ii_ms=s['score_ns']/1e6,
                measured_ii_ms=m['ii_ms'],observed_host_cpu_ms=s['observed_host_cpu_ns']/1e6,scored_host_cpu_ms=s['scored_host_cpu_ns']/1e6,
                CPU_core_demand_ms=str([v/1e6 for v in s['CPU_core_demand_ns']]),CPU_device_service_ms=str([v/1e6 for v in s['CPU_device_service_ns']]),
                VTA_service_sum_ms=s['VTA_service_sum_ns']/1e6,DDR_service_ms=s['DDR_service_ns']/1e6,
                binding_terms=';'.join(s['binding_terms']),capacity_binding_terms=';'.join(s['capacity_binding_terms'])))
    write('configurations.csv',configs);write('constraint_accounting.csv',constraints)
    observed={i:v['ii_ms'] for i,v in measurements.items()};metrics=[]
    for key,*_ in MODELS:
        for scope,ids in [('applicable_common_pool',[i for i in pool if scores[key][i]['applicable']]),('own_measured_list',lists[key])]:
            metrics.append(dict(model=key,label=settings(key)['label'],scope=scope,count=len(ids),
                **tied_metrics({i:scores[key][i]['score_ns']/1e6 for i in ids},observed)))
    write('prediction_quality.csv',metrics)
    mechanism=c.read(bundle/'plans/mechanism_audit.json');pair={r['label']:r['configuration']['id'] for r in mechanism['required_new_configurations']}
    mechanism_result=dict(source='current frozen executor, input0; two adjustable CPU stages of a five-stage full pipeline',
        configurations=pair,overlap_ii_ms=measurements[pair['two_thread_overlap']]['ii_ms'],
        split_ii_ms=measurements[pair['two_thread_split']]['ii_ms'])
    mechanism_result['split_FPS_gain_percent']=100*(mechanism_result['overlap_ii_ms']/mechanism_result['split_ii_ms']-1)
    c.write(out/'mechanism_results.json',mechanism_result,replace=True)
    proofs=c.read(bundle/'plans/search_proofs.json');nominal=sum(settings(r[0])['measurement_budget'] for r in MODELS)+len(pair)
    accounting=dict(configurations=len(pool),native_sessions=len(receipts),complete_DNN_inferences=29*len(pool),
        new_component_profiles=0,compiled_modules=0,compiled_runners=0,runtime_changes=False,
        measurement_and_collection_seconds=sum(r['measurement_and_collection_seconds'] for r in receipts),
        raw_files=sum(len(r['files']) for r in receipts),preparation=c.read(bundle/'plans/preparation_cost.json'),
        search_per_model={k:dict(elapsed_seconds=p['elapsed_seconds'],cache_replayed=p['cache_replayed'],candidate_count=p['selected_count']) for k,p in proofs.items()},
        saved_configurations_by_deduplication=nominal-len(pool),saved_native_sessions_by_deduplication=2*(nominal-len(pool)))
    inherited=bundle/'provenance/evidence/INHERITED_PROFILE_COST.json'
    if inherited.is_file():accounting['inherited_profile_cost']=c.read(inherited)
    rank_cost=bundle/'plans/rank_cost.json'
    if rank_cost.is_file():accounting['ranking_total_wall_seconds']=c.read(rank_cost)['elapsed_seconds']
    result=dict(boot_id=state['boot_id'],decision=decision,costs=accounting,prediction_quality=metrics,
        mechanism=mechanism_result,pool_best=min(configs,key=lambda r:(r['measured_ii_ms'],r['id'])),
        scope='single input, single boot, short window; all frozen lists measured together',
        unresolved=['thread shares are whole-session approximations; three tiny signatures use uniform fallback',
            'homogeneous core permutation is assumed, not a per-mask calibration',
            'control, cache and synchronization costs are not fully separated',
            'team/concentration guards are planning proxies, not response-time bounds',
            'same-score Top20 order is canonical identity, not demonstrated fine-grained performance ranking'])
    c.write(out/'analysis.json',result,replace=True)
    lines=['# ResNet18：Tarnawski基线、规律与分阶段消融','',
        f"本批共同实测{len(pool)}个配置，{len(receipts)}个会话；全部名单在计时前冻结。",'',
        '## 一、论文基线与本文主模型','', '|模型|预算一次FPS|预算五次FPS|预算二十次FPS|','|---|---:|---:|---:|']
    for key in ('tarnawski','fine_host'):
        m=decision['methods'][key];lines.append('|'+m['label']+'|'+'|'.join(f"{m['budgets'][n]['fps']:.3f}" for n in ('1','5','20'))+'|')
    lines+=['','Tarnawski为设备负载目标的平台适配：融合区间成本、单VTA复用、DDR及Top-K均明确列出；不是原作者完整系统复现。','',
        '## 二、统一主要消融','', '|模型|预算一次FPS|预算五次FPS|预算二十次FPS|','|---|---:|---:|---:|']
    for key in ('coarse_no_host','coarse_host','fine_no_host','fine_host'):
        m=decision['methods'][key];lines.append('|'+m['label']+'|'+'|'.join(f"{m['budgets'][n]['fps']:.3f}" for n in ('1','5','20'))+'|')
    lines+=['','CPU细化和主机需求分别在另一因素关闭／开启条件下比较，见contribution_effects.csv；交互用对数FPS描述，不相加删项百分比。','',
        '## 三、次要消融','', '|模型|预算一次FPS|预算五次FPS|','|---|---:|---:|']
    for key in ('weights_only','team_only','concentrated'):
        m=decision['methods'][key];lines.append('|'+m['label']+'|'+'|'.join(f"{m['budgets'][n]['fps']:.3f}" for n in ('1','5'))+'|')
    lines+=['','次要名单只测前五项，不补造二十次预算结果。旧八组消融为独立开发证据。','',
        '## 四、机制及剩余误差','',f"两线程分开放置相对重叠的本批FPS变化为{mechanism_result['split_FPS_gain_percent']:+.2f}%。",
        '阶段服务对照见stage_profile_comparison.csv；并发阶段不能相加。选点改善不表示每阶段耗时已预测准确。',
        '同分数量和有效预测范围见prediction_quality.csv；单线程基线不对多线程域给出伪预测。','',
        '## 五、范围与成本','',f"正式DNN推理{accounting['complete_DNN_inferences']}次，测量与回收{accounting['measurement_and_collection_seconds']:.1f}秒。",
        '单输入、单启动、短窗口观察；不同配置和帧不是独立重复，不报告显著性或等效性。',
        '历史正反例见部署包provenance/evidence/REPORT_zh.md；旧FPS没有作为本批测量。',
        'Pipe-it和GCAPS为执行组织与资源拆分依据，不作为独立性能柱。2023年TVM模型、YOLO和贝叶斯优化未纳入。']
    (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    c.write(out/'VERIFICATION.json',dict(rankings_frozen_before_measurement=True,all_qualifications_and_raw_hashes_replayed=True,
        same_boot=True,automatic_remeasurement=False,old_pipeline_performance_reused=False,**accounting),replace=True)
    curves=bundle/'provenance/evidence/thread_curves.csv'
    if curves.is_file():(out/'thread_curves.csv').write_bytes(curves.read_bytes())
    plot(out,decision)
    return dict(report=str(out/'REPORT_zh.md'),pool_size=len(pool),accepted_sessions=len(receipts))

def plot(output,decision):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/cpu-vta-staged-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
    if font.is_file():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=['DejaVu Sans',font_manager.FontProperties(fname=str(font)).get_name()]
    plt.rcParams['axes.unicode_minus']=False
    groups=[('method_comparison',['tarnawski','fine_host'],'论文基线与本文主模型'),
        ('CPU_host_ablation',['coarse_no_host','coarse_host','fine_no_host','fine_host'],'CPU细化与主机CPU需求：共同实测')]
    labels={'tarnawski':'Tarnawski\n平台适配','fine_host':'细化CPU\n计入主机','coarse_no_host':'粗粒度CPU\n不计主机',
        'coarse_host':'粗粒度CPU\n计入主机','fine_no_host':'细化CPU\n不计主机'}
    for filename,keys,title in groups:
        fig,ax=plt.subplots(figsize=(9,5))
        for j,n in enumerate(('1','5','20')):
            ax.bar([i+(j-1)*.25 for i in range(len(keys))],[decision['methods'][k]['budgets'][n]['fps'] for k in keys],
                width=.25,label='预算'+n)
        ax.set(xticks=range(len(keys)),xticklabels=[labels[k] for k in keys],ylabel='最佳实测吞吐（FPS）',title=title)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(ncol=3,loc='lower center',bbox_to_anchor=(.5,-.3))
        fig.suptitle('ResNet18：输入0、单启动、短窗口；相同测量协议',fontsize=10)
        fig.subplots_adjust(bottom=.28,top=.85)
        for suffix in ('png','pdf'):fig.savefig(Path(output)/(filename+'.'+suffix),dpi=170)
        plt.close(fig)
    curve_path=Path(output)/'thread_curves.csv'
    if curve_path.is_file():
        with curve_path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        names=sorted({r['program'] for r in rows});fig,axes=plt.subplots(1,len(names),figsize=(12,4.5),squeeze=False)
        for ax,name in zip(axes[0],names):
            values=sorted([r for r in rows if r['program']==name],key=lambda r:int(r['threads']))
            for field,label in [('service_ms','隔离服务时间'),('run_core_ms','计算CPU核时')]:
                first=float(values[0][field]);ax.plot([int(r['threads']) for r in values],
                    [float(r[field])/first for r in values],'o-',label=label)
            ax.axhline(1,color='gray',linewidth=.7);ax.set(title=name,xticks=[1,2,3,4],xlabel='配置线程数',ylabel='相对单线程')
            ax.grid(alpha=.2);ax.legend(fontsize=9)
        fig.suptitle('已有隔离profile：服务时间缩短与CPU核时变化分别报告')
        fig.tight_layout(rect=(0,0,1,.92))
        for suffix in ('png','pdf'):fig.savefig(Path(output)/('thread_curves.'+suffix),dpi=170)
        plt.close(fig)
