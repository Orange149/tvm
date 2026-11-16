#!/usr/bin/env python3
"""Analyze only after the frozen prospective native batch has completed."""
from pathlib import Path
from statistics import mean
import json,shutil,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,digest,verify_inventory,now
from scheduler_gap_audit.curves import csv_write
from finite_service.common import ROOT,OUTPUT
import finite_service_holdout_board as board

LABELS=dict(static_fine_host='原资源容量静态评分',fluid_instant='原逐事件选核连续分摊回放',
            fluid_wake_boundary='保留线程落核历史的连续分摊回放',finite_wake_boundary='加入有限CPU服务的回放')
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_finite_service_validation'


def analyze():
    state=read(board.RUN/'state.json');plan=read(board.POOL/'BOARD_PLAN.json')
    if state['status']!='COMPLETE' or state['DNN_inferences']!=plan['DNN_inferences']:
        raise ValueError('prospective native batch not complete')
    if state['boot_id']!=plan['expected_boot_id']:raise ValueError('prospective boot differs')
    selected=board.selection();receipts={};measured={};qualification={}
    for spec in board.specs():
        bundle=board.POOL/spec['network']/'capsule';board.validate(bundle)
        task=board.make_task(bundle,spec);path=board.RUN/'accepted'/state['boot_id']/(task['task_id']+'.json')
        item=read(path)
        if item['task']!=task or item['boot_id']!=state['boot_id']:raise ValueError('prospective receipt identity differs')
        verify_inventory(item['raw'],item['files'])
        if board.summary(item['raw'],task,bundle)!=item['summary']:raise ValueError('prospective evidence no longer replays')
        if (Path(item['raw'])/'task_hash').read_text().strip()!=digest(task):raise ValueError('native task digest differs')
        key=spec['network'],spec['case'];receipts[str(path)]=sha256(path)
        if spec['phase']=='qualify':qualification[key]=item['summary']['output_fingerprints']
        else:
            if item['summary']['output_fingerprints']!=qualification[key]:raise ValueError('performance outputs differ from qualification')
            measured[key]=dict(summary=item['summary'],task=task,raw=item['raw'])
    poolread=read(board.POOL/'KNOWN_EXECUTIONS.json');known=set(poolread['fingerprints'])
    from prepare_finite_service_holdout import fingerprint
    candidates=[];method_rows=[];stage_rows=[];ranking_rows=[]
    for network,data in selected['networks'].items():
        actuals={cid:measured[network,cid]['summary'] for cid in data['selected_ids']}
        winner=max(actuals,key=lambda cid:(actuals[cid]['fps'],cid));best=actuals[winner]['fps']
        for cid in data['selected_ids']:
            score=data['scores'][cid];cfg=score['configuration'];m=actuals[cid]
            if fingerprint(cfg) in known:raise ValueError('selected configuration already in excluded inventory')
            row=dict(network=network,configuration_id=cid,
                CPU_configuration=[dict(program=s['program'],threads=s['threads'],mask=s['mask']) for s in cfg['stages'] if s['device']=='cpu'],
                fps=m['fps'],ii_ms=m['ii_ms'],stage_service_ms=m['stage_service_ms'],
                process_core_ms_per_completion=m['process_core_ms_per_completion'],
                kept_timestamps=m['kept_timestamps'],gaps=m['gaps'],
                relative_loss_to_measured_pool_best_percent=100*(best/m['fps']-1))
            for method in data['rankings']:
                pred=score[method]/1e6
                row[method+'_predicted_II_ms']=pred
                row[method+'_absolute_II_error_ms']=abs(pred-m['ii_ms'])
            candidates.append(row)
            for i,(s,p,a) in enumerate(zip(cfg['stages'],score['finite_stage_ns'],m['stage_service_ms'])):
                stage_rows.append(dict(network=network,configuration_id=cid,stage=i,program=s['program'],device=s['device'],
                    threads=s['threads'],mask=s['mask'],finite_prediction_ms=p/1e6,native_service_ms=a,
                    absolute_error_ms=abs(p/1e6-a)))
        for method,ranking in data['rankings'].items():
            choice=ranking[0]
            if choice not in actuals:raise ValueError('a model first choice was not jointly measured')
            m=actuals[choice]
            method_rows.append(dict(network=network,method=method,label=LABELS[method],selected_id=choice,
                fps=m['fps'],ii_ms=m['ii_ms'],pool_best_FPS=best,pool_best_id=winner,
                loss_to_measured_pool_best_percent=100*(best/m['fps']-1),
                selection_budget=1,full_DP_Top20_comparison=False))
            for rank,cid in enumerate(ranking,1):
                ranking_rows.append(dict(network=network,method=method,rank=rank,configuration_id=cid,
                    predicted_II_ms=data['scores'][cid][method]/1e6,
                    measured_FPS=actuals[cid]['fps'] if cid in actuals else None,
                    candidate_measured=cid in actuals))
    gains=[]
    for network in selected['networks']:
        f=next(x for x in method_rows if x['network']==network and x['method']=='finite_wake_boundary')
        for reference in ('static_fine_host','fluid_instant','fluid_wake_boundary'):
            b=next(x for x in method_rows if x['network']==network and x['method']==reference)
            gains.append(dict(network=network,reference=reference,FPS_gain_percent=100*(f['fps']/b['fps']-1),
                same_configuration=f['selected_id']==b['selected_id']))
    result=dict(state=state,methods=method_rows,candidates=candidates,stage_errors=stage_rows,
        prospective_gain=gains,all_rankings=ranking_rows,source_receipts=receipts,
        prediction_selection_freeze_sha256=sha256(board.POOL/'SELECTION_FREEZE.json'),
        native_boot=state['boot_id'],old_DNN_measurements_not_used_as_new_results=True,
        output_correct=True,all_native_positions_verified=True,
        pool_known_exclusion_scope=poolread['scope'],
        no_new_component_profiles=True,existing_phase_profiles_reused=True,
        new_zero_DNN_platform_session=1,DNN_inferences=state['DNN_inferences'],
        native_sessions=state['total_sessions'],no_reboot=True,
        single_input_single_boot_short_window=True,no_significance_claim=True,
        prototype_pool_has_two_fixed_topologies=True,full_search_domain_not_pruned=True,
        no_new_Top20_claim=True,goal_complete=False,recorded_at=now(),analysis_source_sha256=sha256(__file__))
    dest=board.POOL/'analysis';dest.mkdir(exist_ok=True);write(dest/'RESULTS.json',result)
    for name,rows in [('native_candidates.csv',candidates),('method_top1.csv',method_rows),
                      ('stage_errors.csv',stage_rows),('all_frozen_rankings.csv',ranking_rows),('gains.csv',gains)]:csv_write(dest/name,rows)
    render(dest,result);plot(dest,result)
    REPORT.mkdir(parents=True,exist_ok=True)
    for p in dest.iterdir():
        if p.is_file():shutil.copy2(p,REPORT/p.name)
    for name in ('CALIBRATION.json','MIXED_VALIDATION.json'):
        shutil.copy2(OUTPUT/name,REPORT/name)
    write(OUTPUT/'HOLDOUT_DELIVERY.json',dict(status='SMALL_PROSPECTIVE_CHECK_COMPLETE_GOAL_ACTIVE',
        results_sha256=sha256(dest/'RESULTS.json'),files={str(p):sha256(p) for p in REPORT.iterdir() if p.is_file()},
        analysis_source_sha256=sha256(__file__),goal_complete=False))
    progress=read(ROOT/'artifacts/static_revision_01/GOAL_PROGRESS.json')
    progress.update(phase='finite-service platform calibration and prospective resource-selection check completed',
        last_goal_turn='progress: zero-DNN platform profile, held-out synthetic validation, 15 development replays and frozen prospective native pool',
        finite_service_result=str(dest/'RESULTS.json'),next_action='use prospective gains and stage errors to decide whether full-DP candidate integration is supported',
        current_boot=state['boot_id'],goal_status='ACTIVE',updated_at=now())
    write(ROOT/'artifacts/static_revision_01/GOAL_PROGRESS.json',progress,replace=True)
    return dict(methods=method_rows,gains=gains,report=str(REPORT/'FINDINGS_zh.txt'),goal_complete=False)


def render(dest,r):
    cal=read(OUTPUT/'CALIBRATION.json');mixed=read(OUTPUT/'MIXED_VALIDATION.json')
    lines=['有限CPU服务近似与新核组布局验证','',
        '本轮先用零DNN平台忙负载标定有限CPU服务，再检查未用于参数选择的混合短任务负载，最后冻结两个网络各17项局部候选的预测并共同测量模型选出的并集。',
        'CPU每核一次执行一个逻辑线程，保留线程虚拟运行量及有限服务粒度；唤醒近似借鉴Linux 5.4 fair.c的语义，不声称复现供应商CFS。没有用DNN流水FPS选择参数。',
        '没有修改运行库、执行器、编译模块或原完整DP域；本轮是两个固定切图下的新资源布局验证。','',
        '平台标定（不是恢复出的内核时间片参数）：']
    for n,q in cal['quanta_ns'].items():lines.append(f'{n}个忙线程竞争一核，连续运行片段中位数{q/1e6:.3f}毫秒。')
    lines += [f"未用于参数选择的混合负载检查通过={mixed['finite_service_platform_check_passed']}；该项不等于DNN预测已有效。",'',
        f"本批{r['native_sessions']}个原生会话、{r['DNN_inferences']}次完整推理；输入0，单帧资格检查后8帧预热、16个完成时间戳、4帧尾部。15个间隔计算FPS。未重启、未加锚点或自动重测。",'',
        '一次选点的共同实测结果：']
    for x in r['methods']:
        lines.append(f"{x['network']} | {x['label']} | {x['fps']:.4f} FPS | {x['ii_ms']:.3f}毫秒 | {x['selected_id']}")
    lines+=['','有限服务回放相对于各参照的选点差异：']
    for x in r['prospective_gain']:
        lines.append(f"{x['network']}，相对{LABELS[x['reference']]}：{x['FPS_gain_percent']:+.2f}%；选择同一配置={x['same_configuration']}。")
    lines+=['','解释范围：',
        '原资源容量评分、逐事件选核回放、保留线程历史的连续回放、有限服务回放获得同一个冻结候选池。共享选点只测一次，没有把评分版本放入执行配置身份。',
        '候选从完整合法资源类型中取局部代表，排除了所审计接受凭据和准备清单中的已知完整配置。已知性审计覆盖当前项目，未穷尽项目建立前的全部历史脚本。',
        '各网络固定切图，仅变化前两个CPU子图的线程和核组；后续CPU阶段、VTA主机/控制位置、门控、程序和输入保持不变。不是新切图或新网络的盲测。',
        '只报告一次选点结果，不将3项实测冒充预算20或完整DP Top-20验证。未测的候选没有实测FPS。',
        '阶段误差与整体选点分别报告；更接近某阶段耗时并不能单独证明选点改善。',
        '单输入、单启动、短窗口只提供方向性证据，无显著性或稳定收益结论。',
        '跨启动旧成本保留原身份；平台标定与本批性能来自同一新启动，旧性能未作为本批结果复用。',
        '完整DP接入及与Tarnawski的预算1/5/20比较仍需依本轮结果判断，Goal没有完成。']
    (dest/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')


def plot(dest,r):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=['static_fine_host','fluid_instant','fluid_wake_boundary','finite_wake_boundary']
    labels=['Static resources','Fluid original','Fluid with history','Finite service']
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,network in zip(axes,('resnet18','yolov3_tiny416')):
        ys=[next(x['fps'] for x in r['methods'] if x['network']==network and x['method']==m) for m in methods]
        bars=ax.bar(range(4),ys,color=['#999999','#4c78a8','#72b7b2','#f58518'])
        ax.set_xticks(range(4),labels,rotation=20,ha='right');ax.set_ylabel('FPS');ax.set_title(network+' / frozen top-1 choice')
        ax.set_ylim(0,max(ys)*1.2)
        for b,y in zip(bars,ys):ax.text(b.get_x()+b.get_width()/2,y,f'{y:.3f}',ha='center',va='bottom')
    fig.tight_layout();fig.savefig(dest/'prospective_top1.png',dpi=180);plt.close(fig)


if __name__=='__main__':print(json.dumps(analyze(),ensure_ascii=False))
