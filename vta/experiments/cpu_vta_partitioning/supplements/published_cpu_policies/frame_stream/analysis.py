import csv
import statistics
from pathlib import Path
from .common import api
from .domain import cases

METHODS=('original','saturated','frame_stream')
LABELS={'original':'原容量模型','saturated':'持续饱和CPU共跑约束','frame_stream':'CPU共同帧流整体约束'}

def evaluate(predictions,reference):
    from scipy.stats import spearmanr
    output={}
    for topology,rankings in predictions['rankings'].items():
        names=[n for n in cases() if n.startswith(topology+'/')]
        best=min(names,key=lambda n:(reference[n]['ii_ms'],n));best_ii=reference[best]['ii_ms']
        row={}
        for method in METHODS:
            order=rankings[method];first=order[0];two=min(order[:2],key=lambda n:(reference[n]['ii_ms'],n))
            field={'original':'original_ms','saturated':'saturated_ms','frame_stream':'frame_stream_ms'}[method]
            predicted=[predictions['predictions'][n][field] for n in names]
            actual=[reference[n]['ii_ms'] for n in names]
            correlation=float(spearmanr(predicted,actual).statistic) if len(set(predicted))>1 else None
            row[method]={'static_first':first,'first_fps':reference[first]['fps'],
                'first_ii_ms':reference[first]['ii_ms'],'best_in_pool':best,
                'budget1_regret_percent':(reference[first]['ii_ms']/best_ii-1)*100,
                'budget2_regret_percent':(reference[two]['ii_ms']/best_ii-1)*100,
                'top2_covers_best':best in order[:2],
                'mae_ms':statistics.mean(abs(a-b) for a,b in zip(predicted,actual)),
                'spearman':correlation}
        output[topology]=row
    nonworse=all(r['frame_stream']['first_ii_ms']<=r['original']['first_ii_ms'] for r in output.values())
    improves=any(r['frame_stream']['first_ii_ms']<r['original']['first_ii_ms'] for r in output.values())
    covers=all(r['frame_stream']['top2_covers_best'] for r in output.values())
    return {'per_topology':output,'worth_unseen_configuration_validation':nonworse and improves and covers,
            'development_replay_only':True,'full_DP_changed':False}

def analyze(root):
    from .campaign import validate_state,validated_records
    root=Path(root);state=validate_state(root);c=api(state['bundle'])
    if 'prediction_freeze.json' not in state['frozen_files']:raise ValueError('predictions must be frozen before opening performance reference')
    predicted=c.read(root/'predictions.json');freeze=c.read(root/'prediction_freeze.json')
    if freeze['predictions_sha256']!=c.sha256(root/'predictions.json'):raise ValueError('prediction freeze differs')
    # First and only performance-reference read in the experiment controller.
    reference=c.read(Path(state['bundle'])/'plans/analysis_reference.json')
    records=validated_records(root,state);decision=evaluate(predicted,reference)
    output=root/'analysis';output.mkdir(exist_ok=True)
    def write(name,rows):
        with (output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    rows=[];stages=[]
    for name,summary in records.items():
        p=predicted['predictions'][name]
        rows.append({'case':name,'CPU_group_ii_ms':summary['cpu_ii_ms'],
            'original_prediction_ms':p['original_ms'],'saturated_prediction_ms':p['saturated_ms'],
            'frame_stream_prediction_ms':p['frame_stream_ms'],
            'old_pipeline_ii_ms':reference[name]['ii_ms'],'old_pipeline_fps':reference[name]['fps']})
        for actor in summary['actors']:stages.append({'case':name,**actor})
    write('predictions.csv',rows);write('stage_service_wait.csv',stages)
    budgets=[]
    for topology,methods in decision['per_topology'].items():
        for method,value in methods.items():budgets.append({'topology':topology,'method':method,**value})
    write('selection_budgets.csv',budgets)
    receipts=[c.read(p) for p in (root/'accepted').rglob('*.json')]
    provenance={'accepted_sessions':len(receipts),'stage_calls':sum(r['summary']['stage_calls'] for r in receipts),
        'raw_files':sum(len(r['files']) for r in receipts),'raw_bytes':sum(v['bytes'] for r in receipts for v in r['files'].values())}
    if provenance['accepted_sessions']!=8 or provenance['stage_calls']!=300:raise ValueError('frame-stream execution coverage differs')
    c.write(output/'analysis.json',{'decision':decision,'measurements':records,'provenance':provenance},replace=True)
    lines=['# CPU共同帧流开发验证结果','', '完成8项CPU微基准、300次阶段调用。没有执行新的完整DNN推理。',
        '每阶段14帧；前2帧预热，8个完成时间戳对应7个间隔，后4帧尾部。',
        '完整流水性能引用上一批结果，属于已知配置的开发回放，不是独立收益确认。','',
        '|切图|方法|静态第一名|旧实测FPS|预算一次损失%|前二覆盖池内最佳|平均误差ms|',
        '|---|---|---|---:|---:|---|---:|']
    titles={'explanation':'五阶段切图','validation':'三阶段切图'}
    for topology,methods in decision['per_topology'].items():
        for method,v in methods.items():lines.append(f"|{titles[topology]}|{LABELS[method]}|{v['static_first'].split('/')[-1]}|{v['first_fps']:.3f}|{v['budget1_regret_percent']:.2f}|{v['top2_covers_best']}|{v['mae_ms']:.3f}|")
    lines+=['', '值得开展未见配置验证。' if decision['worth_unseen_configuration_validation'] else '未满足事前选点条件，不支持直接接入完整DP。',
        '', '五阶段切图同时补入CPU20，变化不能全归因于限量；三阶段切图的CPU集合保持一致。',
        '没有VTA主机并发、真实阶段依赖或VTA访存干扰。该整体成本依赖CPU子图组合，不能作为单阶段局部成本推广。',
        '', '[预测与旧实测](predictions.csv) · [阶段服务和等待](stage_service_wait.csv) · [预算对照](selection_budgets.csv)']
    (output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    c.write(output/'VERIFICATION.json',{**provenance,'predictions_frozen_before_reference_read':True,
            'new_full_pipeline_sessions':0,'new_VTA_sessions':0,'new_isolated_calibrations':0,'full_DP_changed':False},replace=True)
    return {'report':str(output/'REPORT_zh.md'),'worth_unseen_configuration_validation':decision['worth_unseen_configuration_validation'],**provenance}
