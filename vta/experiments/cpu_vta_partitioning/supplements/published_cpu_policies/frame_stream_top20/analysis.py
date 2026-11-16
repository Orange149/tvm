from pathlib import Path
import csv
import statistics
from .common import api

LABELS={'pure_static':'原纯静态20项','diverse':'原差异化20项','frame_stream':'CPU共同帧流重排20项'}

def evaluate(lists,measurements):
    output={}
    for method,names in lists.items():
        if len(names)!=20 or len(set(names))!=20:raise ValueError('invalid twenty-entry evaluation list')
        budgets={}
        for budget in (1,5,20):
            best=min(names[:budget],key=lambda n:(measurements[n]['ii_ms'],n))
            budgets[str(budget)]={'selected':best,'ii_ms':measurements[best]['ii_ms'],'fps':measurements[best]['fps']}
        output[method]={'budgets':budgets,'members':names}
    new=output['frame_stream']['budgets'];baselines=('pure_static','diverse')
    nonworse=all(new['20']['ii_ms']<=output[b]['budgets']['20']['ii_ms'] for b in baselines)
    improvements={b:[k for k in ('1','5','20') if new[k]['ii_ms']<output[b]['budgets'][k]['ii_ms']] for b in baselines}
    return {'methods':output,'improved_budgets_vs_baseline':improvements,'budget20_not_worse_than_both':nonworse,
            'positive_single_boot_observation':nonworse and any(improvements.values()),
            'independent_confirmation':False,'extra_CPU_profile_cost_must_be_reported':True}

def analyze(root):
    from .campaign import validate_state, tasks, Recorder, cpu_records
    root=Path(root);state=validate_state(root);c=api(state['bundle']);bundle=Path(state['bundle'])
    if 'prediction_freeze.json' not in state['frozen_files']:raise ValueError('rankings must be frozen before reading pipeline observations')
    predicted=c.read(root/'predictions.json');freeze=c.read(root/'prediction_freeze.json')
    if freeze['predictions_sha256']!=c.sha256(root/'predictions.json'):raise ValueError('prediction freeze differs')
    pool=c.read(bundle/'plans/static_pool.json');static=c.read(bundle/'plans/static_rankings.json')
    recorder=Recorder(root,state);measurements={}
    for doc in tasks(bundle)['pipeline']:
        s=recorder.saved(doc)
        if doc['kind']=='performance':measurements[doc['candidate_id']]=s
    if set(measurements)!=set(pool):raise ValueError('incomplete pipeline coverage')
    lists={'pure_static':[r['id'] for r in static['pure_static20']],
           'diverse':[r['id'] for r in static['diverse20']],
           'frame_stream':[r['id'] for r in predicted['top20']]}
    decision=evaluate(lists,measurements);cpu=cpu_records(root,state)
    output=root/'analysis';output.mkdir(exist_ok=True)
    def write(name,rows):
        with (output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    new={r['id']:r for r in predicted['rankings']};rows=[];stages=[];budgets=[]
    def cut(cfg):return ' → '.join(('CPU' if s['device']=='cpu' else 'VTA')+str(s['begin'])+'–'+str(s['end']-1) for s in cfg['stages'])
    for name,cfg in pool.items():
        p=new[name];m=measurements[name]
        rows.append({'id':name,'partition':cut(cfg),'context':cfg['context_id'],'gate':cfg['execution_mode'],
            'CPU_threads':'+'.join(str(s['threads']) for s in cfg['stages'] if s['device']=='cpu'),
            'CPU_masks':str([s['mask'] for s in cfg['stages'] if s['device']=='cpu']),
            'host_mask':str(cfg['host_mask']),'control_mask':str(cfg['control_mask']),
            'original_prediction_ms':cfg['score_ns']/1e6,'CPU_group_ms':p['cpu_group_ns']/1e6,
            'new_prediction_ms':p['score_ns']/1e6,'measured_ii_ms':m['ii_ms'],'fps':m['fps'],
            **{key+'_position':ids.index(name)+1 if name in ids else '' for key,ids in lists.items()},
            'new_prediction_pool_position':p['rerank_position']})
    for name,value in cpu.items():
        for actor in value['actors']:stages.append({'CPU_group':name,**actor})
    for method,value in decision['methods'].items():
        for budget,v in value['budgets'].items():budgets.append({'method':method,'budget':budget,**v})
    write('configurations.csv',rows);write('cpu_stage_service_wait.csv',stages);write('selection_budgets.csv',budgets)
    errors={method:statistics.mean(abs(float(r[field])-r['measured_ii_ms']) for r in rows)
            for method,field in [('original','original_prediction_ms'),('frame_stream','new_prediction_ms')]}
    receipts=[c.read(p) for p in (root/'accepted').rglob('*.json')]
    costs={'CPU_sessions':len(cpu),'CPU_stage_calls':sum(v['stage_calls'] for v in cpu.values()),
           'pipeline_configurations':len(pool),'complete_DNN_inferences':len(pool)*29,
           'accepted_sessions':len(receipts),'raw_files':sum(len(r['files']) for r in receipts),
           'raw_bytes':sum(v['bytes'] for r in receipts for v in r['files'].values()),
           'preparation':c.read(bundle/'plans/preparation_cost.json'),
           'static_search':c.read(bundle/'plans/source_provenance.json')['elapsed_seconds']}
    overlap={key:{'shared_with_new':len(set(ids)&set(lists['frame_stream'])),
                  'removed':sorted(set(ids)-set(lists['frame_stream'])),
                  'added':sorted(set(lists['frame_stream'])-set(ids))} for key,ids in lists.items() if key!='frame_stream'}
    result={'decision':decision,'mean_absolute_error_ms':errors,'overlap':overlap,'costs':costs,
            'pool_best':min(rows,key=lambda r:(r['measured_ii_ms'],r['id'])),'boot_id':state['boot_id'],
            'scope':'one input, one boot, short windows, frozen DP candidate pool'}
    c.write(output/'analysis.json',result,replace=True)
    lines=['# 完整DP候选重排与Top-20结果','',
           f"完整域9种上下文，冻结{len(pool)}项流水配置；{len(cpu)}项CPU共同帧流。新名单在完整流水计时前冻结。",'',
           '|名单|一次预算FPS|五次预算FPS|二十次预算FPS|','|---|---:|---:|---:|']
    for key,v in decision['methods'].items():
        lines.append('|'+LABELS[key]+'|'+'|'.join(f"{v['budgets'][b]['fps']:.3f}" for b in ('1','5','20'))+'|')
    lines+=['',f"原／新模型池内平均绝对预测误差：{errors['original']:.3f}／{errors['frame_stream']:.3f} ms。",
            '本次观察满足继续研究条件。' if decision['positive_single_boot_observation'] else '本次观察未证明新增CPU标定值得扩大范围。',
            '', '新增CPU标定成本独立计入，不称三种方法总调优成本相同。',
            '共同帧流不含VTA主机并发、真实图依赖或VTA访存；只重排冻结候选池，不保证全域最优。',
            '单输入、单启动短窗口观察，不宣称稳定收益。旧批次及原DP公式保持冻结。','',
            '[全部配置](configurations.csv) · [预算对照](selection_budgets.csv) · [CPU服务与等待](cpu_stage_service_wait.csv)']
    (output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    c.write(output/'VERIFICATION.json',{'pool_size':len(pool),'expected_max_pool':30,
        'same_budget_pipeline_measurement':True,'CPU_profile_extra':True,'predictions_frozen_before_pipeline':True,
        'full_DP_formula_changed':False,'historical_special_candidates_added':False,**costs},replace=True)
    return {'report':str(output/'REPORT_zh.md'),'positive_single_boot_observation':decision['positive_single_boot_observation'],
            'pool_size':len(pool),'CPU_sessions':len(cpu),'accepted_sessions':len(receipts)}
