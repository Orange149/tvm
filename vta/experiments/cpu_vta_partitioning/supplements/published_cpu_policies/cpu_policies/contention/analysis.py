"""Report prediction errors and selected performance without refitting profiles."""
import csv
import json
import statistics
from pathlib import Path
from .common import api
from .domain import configurations, components, groups
from ..core_model.domain import component, component_key

METHODS=('original','cpu_corun','cpu_vta_corun')
LABELS={'original':'原资源容量模型','cpu_corun':'增加CPU共跑服务约束','cpu_vta_corun':'增加CPU与VTA共跑服务约束'}

def evaluate(predictions,measurements):
    result={}
    for topology in ('explanation','validation'):
        names=[n for n in predictions['predictions'] if n.startswith(topology+'/')]
        actual=lambda n:measurements[n]['ii_ms']
        base_first=predictions['rankings'][topology]['original'][0]
        baseline_mae=statistics.mean(abs(predictions['predictions'][n]['estimates_ms']['original']-actual(n)) for n in names)
        result[topology]={}
        for method in METHODS:
            ranking=predictions['rankings'][topology][method]; first=ranking[0]
            estimates={n:predictions['predictions'][n]['estimates_ms'][method] for n in names}
            mae=statistics.mean(abs(estimates[n]-actual(n)) for n in names)
            overlap,split=topology+'/overlap2',topology+'/split2'
            predicted_difference=estimates[overlap]-estimates[split]
            actual_difference=actual(overlap)-actual(split)
            sign=lambda x:(x>0)-(x<0)
            direction=sign(predicted_difference)==sign(actual_difference)
            result[topology][method]={'mae_ms':mae,'mape_percent':statistics.mean(abs(estimates[n]-actual(n))/actual(n)*100 for n in names),
                'static_first':first,'first_ii_ms':actual(first),'first_fps':measurements[first]['fps'],
                'first_fps_change_vs_original_percent':(actual(base_first)/actual(first)-1)*100,
                'matched_placement_direction_correct':direction,'predicted_overlap_minus_split_ms':predicted_difference,
                'observed_overlap_minus_split_ms':actual_difference,
                'worth_followup':method!='original' and mae<baseline_mae and actual(first)<=actual(base_first) and direction}
    supported=[m for m in METHODS[1:] if all(result[t][m]['worth_followup'] for t in result)]
    return {'per_topology':result,'recommended_followup':supported[0] if supported else None,
            'single_boot_observation_only':True,'full_DP_modified':False}

def analyze(root):
    from .campaign import validate_state, validated_records
    root=Path(root); state=validate_state(root); c=api(state['bundle']); results=validated_records(root,state)
    predictions=c.read(root/'predictions.json'); measurements={}
    for name in configurations():
        s=results['pipeline/'+name]; measurements[name]={'ii_ms':s['ii_ms'],'fps':s['fps'],**s}
    decision=evaluate(predictions,measurements)
    out=root/'analysis'; out.mkdir(exist_ok=True)
    def write_csv(name,rows):
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    rows=[]
    for name,cfg in configurations().items():
        m=measurements[name]
        rows.append({'case':name,'configuration_id':cfg['id'],'actual_ii_ms':m['ii_ms'],'fps':m['fps'],
                     **predictions['predictions'][name]['estimates_ms']})
    write_csv('predictions_vs_measured.csv',rows)
    component_rows=[]
    for case,cfg in configurations().items():
        cpu=[(i,s) for i,s in enumerate(cfg['stages']) if s['device']=='cpu'][:2]
        for actor,(stage_index,s) in enumerate(cpu):
            iso=results['isolated/'+component_key(component(s))]['actors']['0']
            pair=results['concurrent/'+case+'/cpu_pair']['actors'][str(actor)]
            full=results['concurrent/'+case+'/cpu_pair_vta']['actors'][str(actor)]
            component_rows.append({'case':case,'program':s['program'],'threads':s['threads'],'cores':str(s['mask']),
                'isolated_service_ms':iso['service_ms'],'cpu_pair_service_ms':pair['service_ms'],
                'cpu_vta_service_ms':full['service_ms'],'pipeline_service_ms':measurements[case]['stage_service_ms'][stage_index],
                'CPU_pair_ratio':pair['service_ms']/iso['service_ms'],'CPU_VTA_ratio':full['service_ms']/iso['service_ms']})
    write_csv('stage_comparison.csv',component_rows)
    diag=[]; thread_rows=[]
    for name,value in results.items():
        if not name.startswith('diagnostic/'): continue
        normal=results['concurrent/'+name.split('/',1)[1]]
        for actor,summary in value['actors'].items():
            diag.append({'case':name,'actor':actor,'program':summary['program'],'normal_service_ms':normal['actors'][actor]['service_ms'],
                         'probe_service_ms':summary['service_ms'],'probe_change_percent':(summary['service_ms']/normal['actors'][actor]['service_ms']-1)*100})
        for sample in value['probe']['records']:
            for t in sample['threads']:
                thread_rows.append({'case':name,'program':sample['program'],'iteration':sample['frame_id'],
                    'tid':t['tid'],'logical_worker':t['logical_worker'],'core':t['core'],'role':t['role'],
                    'runtime_ms':None if t['runtime_delta_ns'] is None else t['runtime_delta_ns']/1e6,
                    'runqueue_wait_ms':None if t['runqueue_wait_delta_ns'] is None else t['runqueue_wait_delta_ns']/1e6,
                    'run_wall_ms':sample['run_end_ms']-sample['run_start_ms'],
                    'probe_envelope_ms':sample['after_end_ms']-sample['before_start_ms'],
                    'zero_wait_is_not_evidence_of_no_wait':t['runqueue_wait_delta_ns']==0})
    write_csv('probe_perturbation.csv',diag)
    write_csv('thread_diagnosis.csv',thread_rows)
    inventory=c.read(Path(state['bundle'])/'plans/full_domain_profile_inventory.json')
    c.write(out/'full_domain_profile_inventory.json',inventory,replace=True)
    provenance={'raw_files':0,'raw_bytes':0}
    for p in (root/'accepted').rglob('*.json'):
        r=c.read(p); provenance['raw_files']+=len(r['files']); provenance['raw_bytes']+=sum(v['bytes'] for v in r['files'].values())
    c.write(out/'analysis.json',{'decision':decision,'measurements':measurements,'provenance':provenance,
             'full_domain_profile_inventory':inventory},replace=True)
    lines=['# 受控并发profile与跨切图验证','',f"完成48项会话；原始凭据{provenance['raw_files']}份。单启动、输入0、短窗口观察。",'',
           '|切图|预测方式|平均误差ms|静态第一名|实测FPS|放置方向正确|', '|---|---|---:|---|---:|---|']
    for topology,methods in decision['per_topology'].items():
        for method,value in methods.items():
            lines.append(f"|{topology}|{LABELS[method]}|{value['mae_ms']:.3f}|{value['static_first'].split('/')[-1]}|{value['first_fps']:.3f}|{value['matched_placement_direction_correct']}|")
    selected=decision['recommended_followup']
    lines+=['',('值得下一阶段验证：'+LABELS[selected]) if selected else '两个修正均未满足全部继续条件；不接入完整DP。',
            '', '持续干扰profile不是完整流水的严格上界；未通过验证时不拟合统一慢化系数。',
            '等待核时是多个线程的累计量，不能直接加到完成间隔。详细线程计数及读取包络保留在诊断原始凭据。',
            '同核组与VTA主机、缓存、访存、同步的影响未必能完全分离；不把单个现象写成唯一原因。',
            '', '[逐阶段对照](stage_comparison.csv) · [预测与实测](predictions_vs_measured.csv) · [探针影响](probe_perturbation.csv)']
    (out/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    c.write(out/'VERIFICATION.json',{'sessions':48,**provenance,'prediction_freeze':c.read(root/'prediction_freeze.json'),
             'receipts_replayed':True,'full_DP_modified':False,'costs_use_pipeline_feedback':False},replace=True)
    return {'report':str(out/'REPORT_zh.md'),'decision':decision['recommended_followup'],**provenance}
