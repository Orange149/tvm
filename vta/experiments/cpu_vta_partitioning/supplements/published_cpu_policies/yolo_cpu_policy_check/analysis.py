"""Matched descriptive comparisons; no new selections or fitting of costs."""
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
from cpu_vta_partitioning.common import read, sha256, verify_inventory, write
from cpu_policies.open_resources.costs import Costs
from .bundle import validate_bundle
from .campaign import make_task, summary


def compare(open_result,fixed_result,ungated_result,repeated):
    faster=fixed_result['ii_ms']<open_result['ii_ms']
    return dict(fixed_vs_open_FPS_percent=100*(fixed_result['fps']/open_result['fps']-1),
        serial_vs_same_allocation_open_FPS_percent=100*(fixed_result['fps']/ungated_result['fps']-1),
        fixed_observed_faster=faster,repeated_group=repeated,
        gate_has_shared_group_object=repeated)


def analyze(output,run_dir):
    output,run_dir=Path(output),Path(run_dir);bundle=output/'pipeline_bundle';validate_bundle(bundle)
    state=read(run_dir/'state.json')
    if state['status']!='COMPLETE':raise ValueError('all frozen tasks must complete first')
    boot=state['boot_id'];pool=read(bundle/'plans/pool.json');scores=read(bundle/'plans/scores.json')
    blocks=read(bundle/'plans/blocks.json');members=read(bundle/'plans/membership.json')
    freeze=read(bundle/'plans/ranking_freeze.json');results={};receipts={};total_seconds=0.
    for name in pool:
        for phase in ('qualify','performance'):
            task=make_task(bundle,name,phase);path=run_dir/'accepted'/boot/(task['task_id']+'.json')
            item=read(path)
            if item['task']!=task or item['boot_id']!=boot:raise ValueError('accepted task/boot differs')
            verify_inventory(item['raw'],item['files']);result=summary(item['raw'],task,bundle)
            if result!=item['summary']:raise ValueError('summary no longer replays')
            raw=Path(item['raw'])
            if ((raw/'boot_id').read_text().strip()!=boot or (raw/'task_hash').read_text().strip()!=__import__('cpu_vta_partitioning.common',fromlist=['digest']).digest(task)):
                raise ValueError('raw task/boot differs')
            if datetime.fromisoformat(item['measurement_started_at'])<datetime.fromisoformat(freeze['frozen_at']):
                raise ValueError('pipeline measurement predates selection freeze')
            receipts[str(path)]=sha256(path);total_seconds+=item['measurement_and_collection_seconds']
            if phase=='performance':results[name]={**result,'raw':str(raw)}
    costs=Costs(read(bundle/'provenance/static_inputs/isolated_costs.json')['costs'],
        programs=read(bundle/'provenance/static_inputs/programs.json'),
        ddr=read(bundle/'provenance/ddr.json'),distribution='session_weights')
    comparisons=[]
    for block in blocks:
        ids=block['candidates'];r=compare(results[ids['open']],results[ids['fixed']],
                                        results[ids['fixed_gate_off']],block['repeated_group'])
        r.update(block_id=block['id'],origins=block['origins'],
                 predicted_open_ms=scores[ids['open']]['score_ns']/1e6,
                 predicted_fixed_ms=scores[ids['fixed']]['score_ns']/1e6)
        r['fixed_static_score_worse']=r['predicted_fixed_ms']>r['predicted_open_ms']
        r['fixed_ranked_worse_but_observed_faster']=r['fixed_static_score_worse'] and r['fixed_observed_faster']
        comparisons.append(r)
    target=run_dir/'analysis';target.mkdir(parents=True,exist_ok=True)
    rows=[];stages=[]
    for name,cfg in pool.items():
        measured=results[name];pred=scores[name]['score_ns']/1e6
        desc=' → '.join(('CPU' if s['device']=='cpu' else 'VTA')+str(s['begin'])+'–'+str(s['end']-1)
                       +('：'+str(s['threads'])+'线程，'+str(s['mask']) if s['device']=='cpu' else '') for s in cfg['stages'])
        rows.append(dict(candidate=name,roles=json.dumps(members[name],ensure_ascii=False),
            context=cfg['context_id'],gate=cfg['execution_mode'],configuration=desc,predicted_ii_ms=pred,
            measured_ii_ms=measured['ii_ms'],FPS=measured['fps'],prediction_error_percent=100*(pred/measured['ii_ms']-1),
            process_core_ms_per_completion=measured['process_core_ms_per_completion'],
            average_CPU_cores=measured['average_cpu_cores']))
        rawrows=[json.loads(s) for s in (Path(measured['raw'])/'result.jsonl').read_text().splitlines()]
        window=rawrows[8:72]
        for i,s in enumerate(cfg['stages']):
            c=costs.get(s,cfg)
            stages.append(dict(candidate=name,stage=i,program=s['program'],device=s['device'],threads=s['threads'],
                worker_map=json.dumps(s['worker_core_map']),host_mask=json.dumps(cfg['host_mask']),
                isolated_service_ms=c['service_ms'],pipeline_service_ms=measured['stage_service_ms'][i],
                service_ratio=measured['stage_service_ms'][i]/c['service_ms'],
                isolated_run_ms=c['run_ms'],pipeline_run_ms=statistics.mean(r[f'stage{i}_run_ms'] for r in window),
                isolated_set_ms=c['set_ms'],pipeline_set_ms=statistics.mean(r[f'stage{i}_set_ms'] for r in window),
                isolated_get_ms=c['get_ms'],pipeline_get_ms=statistics.mean(r[f'stage{i}_get_ms'] for r in window)))
    for leaf,data in [('candidates.csv',rows),('stage_comparison.csv',stages),('matched_comparisons.csv',comparisons)]:
        with (target/leaf).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    write(target/'RESULTS.json',dict(scope='single input, one boot, one preselected 64-timestamp session per configuration',
        boot_id=boot,results=results,comparisons=comparisons,candidates=pool,scores=scores,
        blocks=blocks,receipt_hashes=receipts,old_performance_reused=False,model_changed=False),replace=True)
    write(target/'COSTS.json',dict(candidates=len(pool),native_sessions=2*len(pool),
        complete_DNN_inferences=105*len(pool),new_components=0,new_compilations=0,
        measurements_and_collection_seconds=total_seconds,search=read(output/'ranking_cost.json'),
        audit=read(output/'AUDIT.json'),preparation=read(bundle/'plans/preparation_cost.json')),replace=True)
    text=['# YOLOv3-tiny：固定核组与开放共享匹配验证','',
        '评分、运行库及原生执行器不变；全部配置在本批计时前冻结。输入0，单启动，每项8帧预热、64个完成时间戳、32帧尾部，按63个间隔计FPS。','',
        '| 区块 | 开放共享FPS | 固定组串行FPS | 同分配关闭串行FPS | 固定组相对开放共享 |',
        '|---|---:|---:|---:|---:|']
    for block,c in zip(blocks,comparisons):
        ids=block['candidates'];label='／'.join(block['origins'])
        text.append(f"| {block['id']}：{label} | {results[ids['open']]['fps']:.4f} | {results[ids['fixed']]['fps']:.4f} | {results[ids['fixed_gate_off']]['fps']:.4f} | {c['fixed_vs_open_FPS_percent']:+.2f}% |")
    text+=['','每个区块内切图与主机／控制位置相同；不同区块不能直接用来归因绑核或门控。','']
    for block,c in zip(blocks,comparisons):
        text += [f"## {block['id']}：{'／'.join(block['origins'])}",'',
            '切图：'+' → '.join(block['topology'])+'；资源上下文：'+block['context']['label']+'。',
            f"预测完成间隔：开放共享{c['predicted_open_ms']:.3f} ms，固定组{c['predicted_fixed_ms']:.3f} ms。",
            ('固定组被静态评分排后，但本次实测更快，支持存在该家族漏选问题。' if c['fixed_ranked_worse_but_observed_faster']
             else '固定组是否有用需结合本区块的实测方向；本次不据此拟合通用慢化系数。'),
            ('多个阶段复用同组；串行开关配对可用于判断互斥效果。' if block['repeated_group'] else
             '每个阶段使用不同核组，没有同组互斥对象；串行开关配对的差异只能视为本次运行波动。'),'']
    text+=['## 全部执行配置','']
    for row in rows:
        text += [f"- `{row['candidate']}`：{row['configuration']}；上下文{row['context']}，模式{row['gate']}；预测{row['predicted_ii_ms']:.3f} ms，实测{row['measured_ii_ms']:.3f} ms，{row['FPS']:.4f} FPS。"]
    text+=['',f"共{len(pool)}个配置，{2*len(pool)}个原生会话，{105*len(pool)}次完整推理；无新增组件profile或编译。",
        '', 'Tarnawski静态第一名仅作桥接，不代表重新比较其Top-20最佳性能。没有跨启动重复，不报告显著性或等效性。',
        '', '逐阶段隔离／流水时间见stage_comparison.csv；配置与整体CPU核时见candidates.csv；原始凭据、库存与全部完成间隔见RESULTS.json。',
        '', '未修改评分、加入家族保底名额或自动启动新Top-20。']
    (target/'README.md').write_text('\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family']=['AR PL SungtiL GB','DejaVu Sans']
    fig,ax=plt.subplots(figsize=(10,4.8))
    for j,(key,label) in enumerate([('open','开放共享优选'),('fixed','固定核组串行'),('fixed_gate_off','同分配关闭串行')]):
        ax.bar([i+(j-1)*.25 for i in range(len(blocks))],
               [results[b['candidates'][key]]['fps'] for b in blocks],width=.24,label=label)
    ax.set_xticks(range(len(blocks)),[b['id'] for b in blocks]);ax.set_ylabel('FPS（64个时间戳／63个间隔）')
    ax.legend();fig.tight_layout()
    for ext in ('svg','png','pdf'):fig.savefig(target/('matched_comparison.'+ext),dpi=160)
    plt.close(fig)
    return dict(report=str(target/'README.md'),comparisons=comparisons,candidates=len(pool))
