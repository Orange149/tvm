"""Descriptive single-boot analysis; longer checks do not reselect winners."""
import csv
import json
from pathlib import Path
import statistics
from math import prod
from cpu_vta_partitioning.common import read,sha256,verify_inventory,write
from .common import METHODS,LABELS,ASSETS
from .pipeline import make_task,summary

def nominal_macs():
    """Count each unit's useful convolution/dense MACs directly from Relay JSON."""
    def count(path):
        nodes=read(path)['nodes']
        def shape(node):
            tensor=nodes[int(node['attrs']['_checked_type_'])]
            if tensor['type_key']!='relay.TensorType':raise ValueError('non-tensor MAC operand')
            return [int(nodes[i]['attrs']['value']) for i in nodes[int(tensor['attrs']['shape'])]['data']]
        total=0
        for node in nodes:
            if node['type_key']!='relay.Call':continue
            op=nodes[int(node['attrs']['op'])].get('repr_str')
            if op not in ('nn.conv2d','nn.dense'):continue
            args=nodes[int(node['attrs']['args'])]['data'];weight=shape(nodes[args[1]]);out=shape(node)
            attrs=nodes[int(node['attrs']['attrs'])]['attrs']
            if op=='nn.conv2d':
                if (nodes[int(attrs['data_layout'])]['repr_str']!='NCHW'
                        or nodes[int(attrs['kernel_layout'])]['repr_str']!='OIHW'):
                    raise ValueError('MAC counter requires canonical ordinary layout')
                total+=prod(out)*prod(weight[1:])
            else:total+=prod(out)*weight[1]
        return total
    values=[count(ASSETS/'units'/f'unit_{i:02}.json') for i in range(14)]
    if sum(values)!=count(ASSETS/'graph/canonical_qir.json'):raise ValueError('unit MAC ownership differs from graph')
    return values

def load_receipt(run,bundle,name,phase,boot):
    doc=make_task(bundle,name,phase);path=Path(run)/'accepted'/boot/(doc['task_id']+'.json')
    item=read(path)
    if item['task']!=doc or item['boot_id']!=boot:raise ValueError('analysis task/boot mismatch')
    verify_inventory(item['raw'],item['files'])
    result=summary(item['raw'],doc,bundle)
    if result!=item['summary']:raise ValueError('raw performance does not replay')
    return result,path

def analyze(output,run_dir):
    output,run_dir=Path(output),Path(run_dir);run=run_dir/'pipeline';bundle=output/'pipeline_bundle'
    state=read(run/'state.json')
    if state['status']!='COMPLETE':raise ValueError('screen and fixed long checks must finish first')
    boot=state['boot_id'];pool=read(bundle/'plans/pool.json');lists=read(bundle/'plans/rankings.json')['lists']
    winners=read(run/'WINNERS.json')
    if winners['boot_id']!=boot:raise ValueError('winner freeze boot differs')
    screen={};checks={};receipts={}
    for name in pool:
        load_receipt(run,bundle,name,'qualify',boot)
        screen[name],p=load_receipt(run,bundle,name,'screen',boot);receipts[str(p)]=sha256(p)
    for name in winners['candidates']:
        checks[name],p=load_receipt(run,bundle,name,'confirm',boot);receipts[str(p)]=sha256(p)
    budgets={}
    for key in METHODS:
        budgets[key]={}
        for budget in (1,5,20):
            chosen=winners['methods'][key][str(budget)];name=chosen['candidate_id']
            budgets[key][str(budget)]={**chosen,'recheck_ii_ms':checks[name]['ii_ms'],
                'recheck_fps':checks[name]['fps'],'recheck_intervals':63}
    target=run_dir/'analysis';target.mkdir(parents=True,exist_ok=True)
    unit_macs=nominal_macs();peak=256*100_000_000
    nominal={name:dict(VTA_model_MACs=sum(sum(unit_macs[s['begin']:s['end']]) for s in cfg['stages'] if s['device']=='vta'),
        CPU_model_MACs=sum(sum(unit_macs[s['begin']:s['end']]) for s in cfg['stages'] if s['device']=='cpu')) for name,cfg in pool.items()}
    for row in nominal.values():row['VTA_nominal_compute_only_FPS_reference']=peak/row['VTA_model_MACs']
    write(target/'NOMINAL_COMPUTE_REFERENCE.json',dict(unit_MACs=unit_macs,model_MACs=sum(unit_macs),
        nominal_clock_hz=100_000_000,nominal_MAC_lanes=256,configurations=nominal,
        physical_clock_measured=False,strict_system_upper_bound=False,
        scope='fixed mapping useful conv+dense MACs only; relaxes CPU, DMA, ALU, queues and dependencies'),replace=True)
    for key in METHODS:
        for row in budgets[key].values():row.update(nominal[row['candidate_id']])
    write(target/'RESULTS.json',dict(scope='single input, one boot, one screening and one fixed long session; no significance claim',
        budgets=budgets,screening=screen,confirmation=checks,boot_id=boot,receipt_hashes=receipts,
        pipeline_counts=state,old_pipeline_results_used=False),replace=True)
    scores=read(bundle/'plans/scores.json')
    with (target/'candidates.csv').open('w') as f:
        columns=['candidate','methods_ranks','context','gate','stages','CPU_threads_masks','predictions_ms','screen_ii_ms','screen_FPS','recheck_ii_ms','recheck_FPS','VTA_nominal_compute_only_FPS_reference']
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for name,cfg in pool.items():
            membership={key:next((i for i,r in enumerate(lists[key],1) if r['id']==name),None) for key in METHODS}
            w.writerow(dict(candidate=name,methods_ranks=json.dumps(membership),context=cfg['context_id'],gate=cfg['execution_mode'],
                stages='; '.join(s['program'] for s in cfg['stages']),
                CPU_threads_masks='; '.join(str(s['threads'])+':'+str(s['worker_core_map']) for s in cfg['stages'] if s['device']=='cpu'),
                predictions_ms=json.dumps({key:scores[key][name].get('score_ns',0)/1e6 if scores[key][name]['applicable'] else None for key in METHODS}),
                screen_ii_ms=screen[name]['ii_ms'],screen_FPS=screen[name]['fps'],
                recheck_ii_ms=checks.get(name,{}).get('ii_ms'),recheck_FPS=checks.get(name,{}).get('fps'),
                VTA_nominal_compute_only_FPS_reference=nominal[name]['VTA_nominal_compute_only_FPS_reference']))
    records=read(output/'isolated_costs.json')['costs']
    with (target/'component_curves.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['program','begin','end','device','threads','host_context','service_ms','run_ms','run_core_ms','boundary_core_ms','shares','source_boot','receipt'])
        for rec in records.values():
            s=rec['configuration']['stages'][0];c=rec['cost']
            w.writerow([s['program'],s['begin'],s['end'],s['device'],s['threads'],rec['configuration']['context_id'],
                c['service_ms'],c['run_ms'],c['run_core_ms'],c['boundary_core_ms'],json.dumps(rec.get('session_thread_weights')),
                rec['source']['boot_id'],rec['source']['receipt']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4))
    for j,key in enumerate(METHODS):ax.bar([i+(j-1)*.25 for i in range(3)],
        [budgets[key][str(b)]['recheck_fps'] for b in (1,5,20)],width=.24,label=key)
    ax.set_xticks(range(3),['budget 1','budget 5','budget 20']);ax.set_ylabel('FPS (64 timestamps / 63 intervals)')
    ax.legend();fig.tight_layout();fig.savefig(target/'budget_comparison.svg');fig.savefig(target/'budget_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(7,2,figsize=(11,20))
    for i,ax in enumerate(axes.flat):
        recs=sorted([r for r in records.values() if r['configuration']['stages'][0]['program']==f'cpu_{i:02}_{i:02}'],
                    key=lambda r:r['configuration']['stages'][0]['threads'])
        if len(recs)!=4:raise ValueError('incomplete fourteen-unit thread curve')
        times=[r['cost']['run_ms'] for r in recs];work=[r['cost']['run_core_ms'] for r in recs]
        ax.plot(range(1,5),[times[0]/t for t in times],'-o',label='run speedup')
        ax.plot(range(1,5),[w/work[0] for w in work],'-s',label='run CPU work / t1')
        ax.set_title('unit '+str(i));ax.set_xticks(range(1,5));ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(target/'thread_curves.svg');plt.close(fig)
    imported=read(output/'imported_costs.json')['accepted'];new_count=len(records)-imported
    write(target/'COSTS.json',dict(imported_components=imported,new_component_sessions=new_count,
        component_recorded_stage_calls=new_count*10,DMA_sessions=78,DMA_recorded_calls=78,DMA_warmup_calls=78,
        pipeline=state,component_initialization_probe_calls='separate from ten recorded component calls; see affinity report',
        search=read(output/'ranking/COST.json'),bundle_preparation=read(bundle/'plans/preparation_cost.json'),
        source_cohorts=read(output/'isolated_costs.json')['source_boot_ids']),replace=True)
    rows=['# YOLOv3-tiny416 三组 Top-20 验证','',
        '单输入、单启动观察。预算选点来自短筛选；下表统一报告这些固定选点在新会话较长窗口中的结果，不重新选点。','',
        '| 方法 | 一次预算 FPS | 五次预算 FPS | 二十次预算 FPS |','|---|---:|---:|---:|']
    for key in METHODS:rows.append('| '+LABELS[key]+' | '+' | '.join(f"{budgets[key][str(b)]['recheck_fps']:.4f}" for b in (1,5,20))+' |')
    rows += ['', '| 方法／预算 | 静态预测 FPS | 对应划分的VTA标称纯计算参照 FPS |', '|---|---:|---:|']
    for key in METHODS:
        for b in (1,5,20):
            row=budgets[key][str(b)];name=row['candidate_id']
            rows.append(f"| {LABELS[key]}／{b} | {1e9/scores[key][name]['score_ns']:.3f} | {nominal[name]['VTA_nominal_compute_only_FPS_reference']:.3f} |")
    rows += ['',f"共测{len(pool)}个不同方案，复核{len(checks)}个固定选点，{state['complete_DNN_inferences']}次完整网络推理。另计组件和初始化调用。",
        '', '全部八输出及阶段frontier资格检查、每会话实际绑核、原始文件哈希和同一boot身份通过。',
        '', '原始结果、逐阶段节拍及窗口边界工作库存见RESULTS.json；全部配置见candidates.csv；真实线程曲线见component_curves.csv。',
        '', '限制：短窗口筛选可能受波动影响；较长复核与筛选同boot，不能作为跨启动独立确认。CPU线程份额为全隔离会话近似；其他同核数集合为同构核假设；经验评分不是严格整机上限。']
    (target/'README.md').write_text('\n'.join(rows)+'\n')
    return dict(report=str(target/'README.md'),budgets=budgets)
