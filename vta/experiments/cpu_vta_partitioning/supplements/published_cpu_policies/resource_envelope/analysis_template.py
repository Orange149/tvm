"""Frozen campaign analysis; ranking modules never import this evaluator."""
import csv
from itertools import combinations
from pathlib import Path
from .common import api,protocol

LABELS=protocol()['method_labels']

def execution_class(cfg):
    if cfg['execution_mode']=='resource_group':return 'fixed_groups_serial'
    masks=[set(s['mask']) for s in cfg['stages'] if s['device']=='cpu']
    return 'overlap_open' if any(a&b for a,b in combinations(masks,2)) else 'disjoint_open'

def evaluate(lists,measurements):
    methods={}
    for key,ids in lists.items():
        if len(ids)!=20 or len(set(ids))!=20:raise ValueError('invalid twenty-entry list')
        budgets={}
        for n in (1,5,20):
            best=min(ids[:n],key=lambda i:(measurements[i]['ii_ms'],i))
            budgets[str(n)]={'selected':best,'ii_ms':measurements[best]['ii_ms'],'fps':measurements[best]['fps']}
        methods[key]={'budgets':budgets,'members':ids}
    base=methods['pure_static20']['budgets']
    comparisons={key:{n:(v['fps']/base[n]['fps']-1)*100 for n,v in row['budgets'].items()}
        for key,row in methods.items() if key!='pure_static20'}
    return {'methods':methods,'FPS_improvement_percent_vs_original':comparisons,
        'independent_confirmation':False,'single_boot_observation_only':True}

def analyze(root):
    from .campaign import validate_state,tasks,Recorder,freeze_document
    root=Path(root);state=validate_state(root);c=api(state['bundle']);bundle=Path(state['bundle'])
    if c.read(root/'selection_freeze.json')!=freeze_document(bundle):raise ValueError('selection freeze differs')
    ranked=c.read(bundle/'plans/selection_rankings.json');pool=c.read(bundle/'plans/static_pool.json')
    known=c.read(bundle/'plans/history_labels.json');recorder=Recorder(root,state);measurements={};receipts=[]
    for doc in tasks(bundle):
        value=recorder.saved(doc)
        if doc['kind']=='performance':measurements[doc['candidate_id']]=value
        receipts.append(c.read(root/'accepted'/state['boot_id']/(doc['task_id']+'.json')))
    if set(measurements)!=set(pool):raise ValueError('incomplete measurement union')
    lists={key:[r['id'] for r in ranked[key]] for key in LABELS};decision=evaluate(lists,measurements)
    positions={key:{name:i+1 for i,name in enumerate(ids)} for key,ids in lists.items()}
    rows=[]
    for name,cfg in pool.items():
        cpu=[s for s in cfg['stages'] if s['device']=='cpu'];m=measurements[name]
        original=cfg['original_score_ns']/1e6;new=cfg['score_ns']/1e6
        rows.append({'id':name,'partition':' → '.join(('CPU' if s['device']=='cpu' else 'VTA')+
            str(s['begin'])+'–'+str(s['end']-1) for s in cfg['stages']),
            'context':cfg['context_id'],'gate':cfg['execution_mode'],'CPU_execution_class':execution_class(cfg),
            'CPU_threads':'+'.join(str(s['threads']) for s in cpu),'CPU_masks':str([s['mask'] for s in cpu]),
            'caller_cores':str([s['caller_core'] for s in cpu]),'host_mask':str(cfg['host_mask']),
            'control_mask':str(cfg['control_mask']),'original_predicted_ii_ms':original,
            'new_model_predicted_ii_ms':new,'measured_ii_ms':m['ii_ms'],'fps':m['fps'],
            'original_error_ms':original-m['ii_ms'],'new_model_error_ms':new-m['ii_ms'],
            'history_label':known['labels'][name],
            **{key+'_position':index.get(name,'') for key,index in positions.items()}})
    costs={'pipeline_configurations':len(pool),'accepted_sessions':len(receipts),'complete_DNN_inferences':len(pool)*29,
        'new_component_profiles':0,'runtime_changes':False,'compiled_runners':0,
        'measurement_and_collection_seconds':sum(r['measurement_and_collection_seconds'] for r in receipts),
        'raw_files':sum(len(r['files']) for r in receipts),'raw_bytes':sum(v['bytes'] for r in receipts for v in r['files'].values()),
        'preparation':c.read(bundle/'plans/preparation_cost.json'),
        'search':c.read(bundle/'plans/source_provenance.json')['static_search_cost']}
    output=root/'analysis';output.mkdir(exist_ok=True)
    def write(name,values):
        with (output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    budgets=[{'method':key,'label':LABELS[key],'budget':n,**v}
        for key,row in decision['methods'].items() for n,v in row['budgets'].items()]
    write('configurations.csv',rows);write('selection_budgets.csv',budgets)
    ranking=[]
    for key,ids in lists.items():
        best=min(ids,key=lambda i:(measurements[i]['ii_ms'],i));score_field='original_score_ns' if key=='pure_static20' else 'score_ns'
        inversions=sum(measurements[a]['ii_ms']>measurements[b]['ii_ms'] for a,b in combinations(ids,2))
        ranking.append({'method':key,'best_id':best,'best_static_position':ids.index(best)+1,
            'first_id':ids[0],'first_fps':measurements[ids[0]]['fps'],
            'mean_absolute_II_error_ms':sum(abs(pool[i][score_field]/1e6-measurements[i]['ii_ms']) for i in ids)/20,
            'strict_inversions_in_frozen_order':inversions,'pair_count':190,
            'known_exact':sum(known['labels'][i]=='previously_completed_exact_configuration' for i in ids),
            'physical_mirrors':sum(known['labels'][i]=='physical_mirror_of_completed_configuration' for i in ids)})
    write('ranking_quality.csv',ranking)
    result={'decision':decision,'costs':costs,'ranking':ranking,'boot_id':state['boot_id'],
        'pool_best':min(rows,key=lambda r:(r['measured_ii_ms'],r['id'])),
        'model_settings':c.read(bundle/'plans/source_provenance.json')['model_settings'],
        'scope':'one input, same boot, short windows; isolated-profile model; known cases are development replay',
        'unresolved':['OS scheduling and synchronization are not explicitly predicted',
            'CPU per-thread shares are approximations: uniform or whole-session shares','production/collection costs remain unprofiled',
            'reservation and host guards are conservative planning proxies, not latency bounds',
            'behavior-family compression can discard useful light-stage and physical-core alternatives']}
    c.write(output/'analysis.json',result,replace=True)
    lines=['# 静态资源模型 Top-20 同批比较','',
        f'三份名单并集{len(pool)}项，各做单帧资格检查及8帧预热＋16完成时间戳＋4尾帧计时；未重启开发板。','',
        '|名单|一次预算FPS|五次预算FPS|二十次预算FPS|','|---|---:|---:|---:|']
    for key,row in decision['methods'].items():
        lines.append('|'+LABELS[key]+'|'+'|'.join(f"{row['budgets'][n]['fps']:.3f}" for n in ('1','5','20'))+'|')
    lines+=['','所有名单先冻结再实测；各方法按自己的名单顺序计算预算。',
        '严格前20项与按行为筛选20项分开报告，后者压缩物理镜像及低贡献阶段变体，并非完整行为族最优搜索。',
        f"完整推理{costs['complete_DNN_inferences']}次；测量与回收{costs['measurement_and_collection_seconds']:.1f}秒；全域搜索{costs['search']['seconds']:.1f}秒。", 
        '无新增组件profile，无运行库及执行器变更；继承的775项成本与87项DDR记录的原始成本另见来源记录。',
        '本次是单图、单启动短窗口观察；精确已测配置及物理镜像分别标记，不宣称独立稳定收益或完整解决竞争建模。','',
        '[全部配置](configurations.csv) · [选点预算](selection_budgets.csv) · [排序误差](ranking_quality.csv)']
    (output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    c.write(output/'VERIFICATION.json',{'rankings_frozen_before_measurement':True,'same_boot_union_measurements':True,
        'all_qualifications_passed':True,'all_raw_hashes_replayed':True,'full_DP_formula_changed':True,
        'runtime_rebuilt':False,'historical_special_candidates_added':False,'board_rebooted':False,**costs},replace=True)
    return {'report':str(output/'REPORT_zh.md'),'pool_size':len(pool),'accepted_sessions':len(receipts)}
