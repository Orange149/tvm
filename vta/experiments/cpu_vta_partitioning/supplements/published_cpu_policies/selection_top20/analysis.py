import csv
from pathlib import Path
from .common import api
from .selector import execution_class

LABELS={'pure_static20':'原纯静态20项','diverse20':'原差异化20项','coverage20':'AutoTVM覆盖筛选20项'}

def evaluate(lists, measurements):
    if any(len(v)!=20 or len(set(v))!=20 for v in lists.values()):raise ValueError('invalid twenty-entry list')
    if lists['coverage20'][:5]!=lists['pure_static20'][:5]:raise ValueError('preserved first five differ')
    methods={}
    for key,ids in lists.items():
        budgets={}
        for n in (1,5,20):
            best=min(ids[:n],key=lambda i:(measurements[i]['ii_ms'],i))
            budgets[str(n)]={'selected':best,'ii_ms':measurements[best]['ii_ms'],'fps':measurements[best]['fps']}
        methods[key]={'budgets':budgets,'members':ids}
    new=methods['coverage20']['budgets']['20']['ii_ms']
    baseline={k:methods[k]['budgets']['20']['ii_ms'] for k in ('pure_static20','diverse20')}
    outcome=('improved_over_both' if all(new<v for v in baseline.values()) else
             'same_as_best_baseline' if new==min(baseline.values()) else 'no_additional_value')
    return {'methods':methods,'outcome':outcome,'positive_single_boot_observation':outcome=='improved_over_both',
        'improvement_percent_vs_baseline':{k:(v/new-1)*100 for k,v in baseline.items()},
        'first1_first5_preserved_by_design':True,'independent_confirmation':False}

def analyze(root):
    from .campaign import validate_state,tasks,Recorder,freeze_document
    from cpu_policies.open_resources.domain import contexts
    root=Path(root);state=validate_state(root);c=api(state['bundle']);bundle=Path(state['bundle'])
    if c.read(root/'selection_freeze.json')!=freeze_document(bundle):raise ValueError('selection not frozen before analysis')
    ranked=c.read(bundle/'plans/selection_rankings.json');pool=c.read(bundle/'plans/static_pool.json')
    recorder=Recorder(root,state);measurements={};receipts=[]
    for doc in tasks(bundle):
        value=recorder.saved(doc)
        if doc['kind']=='performance':measurements[doc['candidate_id']]=value
        receipts.append(c.read(root/'accepted'/state['boot_id']/(doc['task_id']+'.json')))
    if set(measurements)!=set(pool):raise ValueError('incomplete measurement union')
    lists={key:[r['id'] for r in ranked[key]] for key in LABELS}
    decision=evaluate(lists,measurements);output=root/'analysis';output.mkdir(exist_ok=True)
    def write(name,rows):
        with (output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    positions={key:{name:i+1 for i,name in enumerate(ids)} for key,ids in lists.items()}
    new={r['id']:r for r in ranked['coverage20']};rows=[]
    for name,cfg in pool.items():
        cpu=[s for s in cfg['stages'] if s['device']=='cpu'];m=measurements[name]
        rows.append({'id':name,'partition':' → '.join(('CPU' if s['device']=='cpu' else 'VTA')+
                str(s['begin'])+'–'+str(s['end']-1) for s in cfg['stages']),
            'context':cfg['context_id'],'gate':cfg['execution_mode'],
            'CPU_execution_class':execution_class(cfg['execution_mode'],[s['mask'] for s in cpu]),
            'CPU_threads':'+'.join(str(s['threads']) for s in cpu),'CPU_masks':str([s['mask'] for s in cpu]),
            'caller_cores':str([s['caller_core'] for s in cpu]),'host_mask':str(cfg['host_mask']),
            'control_mask':str(cfg['control_mask']),'predicted_ii_ms':cfg['score_ns']/1e6,
            'measured_ii_ms':m['ii_ms'],'prediction_error_ms':cfg['score_ns']/1e6-m['ii_ms'],'fps':m['fps'],
            **{key+'_position':index.get(name,'') for key,index in positions.items()},
            'new_selection_reason':new.get(name,{}).get('selection_reason','')})
    budgets=[{'method':key,'label':LABELS[key],'budget':budget,**value}
             for key,v in decision['methods'].items() for budget,value in v['budgets'].items()]
    coverage=[]
    for ctx in contexts():
        for key,ids in lists.items():
            configs=[pool[i] for i in ids if pool[i]['context_id']==ctx['id']]
            coverage.append({'context':ctx['id'],'context_label':ctx['label'],'method':key,'selected':len(configs),
                'measured':len(configs),'unique_topologies':len({tuple((s['device'],s['begin'],s['end']) for s in v['stages']) for v in configs}),
                'execution_classes':str(sorted({execution_class(v['execution_mode'],[s['mask'] for s in v['stages'] if s['device']=='cpu']) for v in configs}))})
    write('configurations.csv',rows);write('selection_budgets.csv',budgets);write('context_coverage.csv',coverage)
    costs={'pipeline_configurations':len(pool),'accepted_sessions':len(receipts),'complete_DNN_inferences':len(pool)*29,
        'new_component_profiles':0,'CPU_frame_stream_sessions':0,
        'measurement_and_collection_seconds':sum(r['measurement_and_collection_seconds'] for r in receipts),
        'raw_files':sum(len(r['files']) for r in receipts),'raw_bytes':sum(v['bytes'] for r in receipts for v in r['files'].values()),
        'preparation':c.read(bundle/'plans/preparation_cost.json')}
    overlap={key:{'shared_with_new':len(set(ids)&set(lists['coverage20'])),
        'removed':sorted(set(ids)-set(lists['coverage20'])),'added':sorted(set(lists['coverage20'])-set(ids))}
        for key,ids in lists.items() if key!='coverage20'}
    result={'decision':decision,'costs':costs,'overlap':overlap,'boot_id':state['boot_id'],
        'pool_best':min(rows,key=lambda r:(r['measured_ii_ms'],r['id'])),
        'scope':'one input, one boot, short windows, no FPS feedback, fixed original model'}
    c.write(output/'analysis.json',result,replace=True)
    lines=['# AutoTVM覆盖筛选 Top-20 实测结果','',
        f"三份冻结名单并集{len(pool)}个配置，同一次启动各测一轮；输入0，8帧预热＋16个完成时间戳＋4帧尾部。",'',
        '|名单|一次预算FPS|五次预算FPS|二十次预算FPS|','|---|---:|---:|---:|']
    for key,v in decision['methods'].items():
        lines.append('|'+LABELS[key]+'|'+'|'.join(f"{v['budgets'][b]['fps']:.3f}" for b in ('1','5','20'))+'|')
    conclusion={'improved_over_both':'新名单的二十次预算最佳结果超过两份旧名单，属于本次单启动的正向观察。',
        'same_as_best_baseline':'新名单覆盖更广，但二十次预算没有获得额外吞吐收益。',
        'no_additional_value':'新名单的二十次预算未超过旧名单最佳结果，本轮覆盖策略没有额外选点价值。'}
    lines+=['',conclusion[decision['outcome']],'一次、五次预算共用原前五名；相同结果来自设计，不能记为改善。',
        '',f"完整推理{costs['complete_DNN_inferences']}次，测量及回收约{costs['measurement_and_collection_seconds']:.1f}秒；无新增组件或CPU共同帧流标定。",
        '等待机制核验仅确认二进制路径，不证明它是性能瓶颈。',
        '本次只报告单图、单启动、短窗口观察；不宣称稳定收益，不评价名单之外配置，不自动启动补测。','',
        '[配置实测表](configurations.csv) · [预算对照](selection_budgets.csv) · [上下文覆盖](context_coverage.csv)']
    (output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    c.write(output/'VERIFICATION.json',{'rankings_frozen_before_measurement':True,'same_boot_union_measurements':True,
        'all_qualifications_passed':True,'all_raw_hashes_replayed':True,'first5_preserved':True,
        'full_DP_formula_changed':False,'runtime_rebuilt':False,'historical_special_candidates_added':False,**costs},replace=True)
    return {'report':str(output/'REPORT_zh.md'),'outcome':decision['outcome'],
        'pool_size':len(pool),'accepted_sessions':len(receipts)}
