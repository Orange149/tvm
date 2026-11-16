"""Post-freeze reporting only; never imported by search or native task builders."""
from pathlib import Path
import shlex
import statistics
import time

from cpu_vta_partitioning.common import locked,now,read,sha256,sync_tree,write
from common_top20.analyze import load_result
from common_top20.campaign import accept
from common_top20.prepare import validate
from common_top20.runtime import task,transport
from policy_branch_eval.analysis import csv_write
from pool_temporal.ranking import validate_rank
from workload_model.common import OUTPUT,ROOT,REPORT,freeze,require_space

LABELS={
    'current':'原完整容量模型',
    'semantics_revision':'去除开放线程组整段累计',
    'fixed_family':'固定不重叠核组家族',
    'mixed':'原模型与固定组混合选点',
    'tarnawski':'Tarnawski单线程平台适配',
    'temporal_replace':'替代整段累计的时序池内重排',
    'temporal_append':'保留整段累计的时序池内重排',
}


def intrinsic_features():
    """Descriptive per-signature evidence; no pipeline results are read."""
    from fractions import Fraction
    def ns(value):return Fraction(value['numerator'],value['denominator'])
    data=read(OUTPUT/'pool_temporal/INTRINSICS.json');cpu=[];vta=[]
    for profile in data['CPU'].values():
        identity=profile['identity'];value=profile['average'];workers={};critical=0
        for kernel in value['kernels']:
            for worker in kernel['workers']:
                slot=worker['logical_worker'];workers[slot]=workers.get(slot,0)+sum(ns(v) for v in worker['compute_segments'])
            for segment in range(kernel['barrier_generations']+1):
                critical+=max(ns(w['compute_segments'][segment]) for w in kernel['workers'])
        serial=sum(ns(v) for v in value['serial_cpu_segments'])+sum(ns(k['launch_prefix_cpu']) for k in value['kernels'])
        cpu.append(dict(**identity,kernels_per_call=len(value['kernels']),
            barrier_generations_per_call=sum(k['barrier_generations'] for k in value['kernels']),
            actual_participant_counts=sorted({k['tasks'] for k in value['kernels']}),
            serial_and_launch_CPU_ms=float(serial/1000000),
            useful_worker_compute_CPU_ms=float(sum(workers.values())/1000000),
            useful_work_per_logical_slot_ms={str(k):float(v/1000000) for k,v in workers.items()},
            ideal_kernel_critical_compute_ms=float(critical/1000000),
            ideal_aggregate_slot_compute_ms=float(max(workers.values(),default=0)/1000000),
            isolated_completion_tail_CPU_ms=float(sum(ns(k['isolated_completion_wait_and_return_cpu']) for k in value['kernels'])/1000000),
            observed_run_wall_ms=float(ns(value['run_wall'])/1000000),
            work_metrics_are_not_pipeline_interval=True))
    for profile in data['VTA'].values():
        value=profile['stages'];host=[ns(v) for v in value['host_serial_CPU']]
        vta.append(dict(network=profile['network'],program=profile['program'],mask=profile['mask'],
            device_calls=len(value['devices']),host_before_CPU_ms=float(host[0]/1000000),
            host_between_CPU_ms=float(sum(host[1:-1])/1000000),host_after_CPU_ms=float(host[-1]/1000000),
            driver_CPU_ms=float(sum(ns(v['CPU']) for v in value['devices'])/1000000),
            device_call_wall_ms=float(sum(ns(v['wall']) for v in value['devices'])/1000000),
            device_call_wall_is_not_hardware_compute=True))
    cpu.sort(key=lambda r:(r['network'],r['program'],r['threads']))
    vta.sort(key=lambda r:(r['network'],r['program'],r['mask']))
    freeze(OUTPUT/'INTRINSIC_FEATURES.json',dict(CPU=cpu,VTA=vta,pipeline_FPS_read=False,
        intrinsic_sha256=sha256(OUTPUT/'pool_temporal/INTRINSICS.json')))
    # Older references retain an extra canonical-graph identity; keep it rather
    # than dropping provenance or assuming every source has identical fields.
    fields=sorted({k for row in cpu for k in row})
    csv_write(OUTPUT/'isolated_phase_CPU_features.csv',[{k:row.get(k) for k in fields} for row in cpu])
    csv_write(OUTPUT/'isolated_phase_VTA_features.csv',vta)
    return dict(CPU_profiles=len(cpu),VTA_profiles=len(vta),pipeline_FPS_read=False)


def budgets(ranked,measured):
    if len(ranked)!=20 or len({r['id'] for r in ranked})!=20:
        raise ValueError('a strict unique twenty-item list is required')
    if any(r['id'] not in measured for r in ranked):
        raise ValueError('cannot report budgets over an incomplete measured list')
    result={}
    for count in (1,5,20):
        best=min(enumerate(ranked[:count],1),key=lambda pair:(measured[pair[1]['id']]['II_ms'],pair[0]))
        index,row=best;value=measured[row['id']]
        result[str(count)]=dict(configuration_id=row['id'],rank=index,II_ms=value['II_ms'],FPS=1000/value['II_ms'],
                                all_prefix_items_measured=True)
    return result


def short():
    temporal=OUTPUT/'pool_temporal';validate_rank(temporal)
    common=OUTPUT/'common_top20';result=read(common/'SHORT_RESULTS.json')
    run=ROOT/'runs/cross_network_common_top20_01'
    state=read(run/'state.json')
    if state['status'] not in ('SHORT_COMPLETE','LONG_COMPLETE') or state['boot_id']!=result['boot_id']:
        raise ValueError('complete formal evidence before supplemental analysis')
    networks={};extra={};rows=[];prediction_rows=[]
    for network,previous in result['networks'].items():
        rank=read(common/network/'RANKINGS.json');new=read(temporal/network/'RANKINGS.json')
        measured={r['configuration_id']:r for r in result['rows'] if r['network']==network}
        if set(measured)!=set(rank['pool']):raise ValueError('not all frozen configurations were measured')
        reports=dict(previous['methods']);scoring={r['configuration']['id']:r for r in new['all_scores']}
        for label in ('replace','append'):
            ranked=new['lists'][label];method='temporal_'+label
            reports[method]=dict(budgets=budgets(ranked,measured),selected_configurations=[r['id'] for r in ranked],
                scope='strict Top20 inside the frozen full-DP seed union; not a full-domain temporal optimum')
        selected={m:r['budgets']['20']['configuration_id'] for m,r in reports.items()}
        extra[network]=dict(configurations=sorted(set(selected.values())),method_winners=selected)
        errors={}
        for label in ('replace','append'):
            errors[label]=statistics.mean(100*abs(scoring[i]['models']['before'][label+'_score_ns']/1e6/v['II_ms']-1)
                                           for i,v in measured.items())
        for identity,value in measured.items():
            projected=scoring[identity]
            prediction_rows.append(dict(network=network,configuration_id=identity,
                measured_II_ms=value['II_ms'],FPS=value['FPS'],
                current_predicted_ms=value['current_predicted_ms'],
                capacity_predicted_ms=projected['capacity_score_ns']/1e6,
                replay_II_ms=projected['models']['before']['replay']['interval_ns']/1e6,
                replay_driver_after_II_ms=projected['models']['after']['replay']['interval_ns']/1e6,
                temporal_replace_ms=projected['models']['before']['replace_score_ns']/1e6,
                temporal_append_ms=projected['models']['before']['append_score_ns']/1e6,
                replay_stage_service_ms=[v/1e6 for v in projected['models']['before']['replay']['stage_service_ns']],
                actual_stage_service_ms=value['pipeline_stage_service_ms'],
                isolated_stage_service_ms=value['isolated_stage_service_ms'],
                assumption_list=projected['assumptions']))
        networks[network]=dict(methods=reports,pool_count=len(measured),temporal_union_MAPE=errors,
            old_union_MAPE=previous['union_MAPE'],best_in_union=previous['best_in_union'])
        for method,report in reports.items():
            order=report['selected_configurations']
            for position,identity in enumerate(order,1):
                cfg=rank['pool'][identity];value=measured[identity]
                prediction=(scoring[identity]['models']['before'][method.removeprefix('temporal_')+'_score_ns']/1e6
                            if method.startswith('temporal_') else next(r['score_ns'] for r in rank['lists'][method] if r['id']==identity)/1e6)
                rows.append(dict(network=network,method=method,method_zh=LABELS[method],rank=position,
                    configuration_id=identity,predicted_II_ms=prediction,measured_II_ms=value['II_ms'],FPS=value['FPS'],
                    stages=cfg['stages'],context=cfg['context_id'],gate=cfg['execution_mode'],
                    host_mask=cfg['host_mask'],control_mask=cfg['control_mask']))
    doc=dict(networks=networks,rows=rows,boot_id=result['boot_id'],
        common_short_sha256=sha256(common/'SHORT_RESULTS.json'),temporal_rank_seal_sha256=sha256(temporal/'RANK_FREEZE.json'),
        old_performance_reused=False,single_boot=True,independent_start_or_significance_claimed=False)
    freeze(OUTPUT/'FINAL_SHORT_RESULTS.json',doc)
    freeze(OUTPUT/'FINAL_LONG_SELECTION.json',dict(networks=extra,boot_id=result['boot_id'],
        short_results_sha256=sha256(OUTPUT/'FINAL_SHORT_RESULTS.json'),
        rule='one budget20 short winner per method; no reselection after long results',window=dict(warmup=8,scored=64,tail=32)))
    csv_write(OUTPUT/'all_methods_top20.csv',rows)
    csv_write(OUTPUT/'all_candidate_prediction_errors.csv',prediction_rows)
    return dict(networks=networks,long_configurations=sum(len(v['configurations']) for v in extra.values()))


def long(*,execute_board=False):
    if not execute_board:raise ValueError('--execute-board is required')
    require_space();common=OUTPUT/'common_top20';run=ROOT/'runs/cross_network_common_top20_01'
    selection=read(OUTPUT/'FINAL_LONG_SELECTION.json');boot=selection['boot_id']
    if selection['short_results_sha256']!=sha256(OUTPUT/'FINAL_SHORT_RESULTS.json'):
        raise ValueError('long selection changed')
    if read(run/'state.json')['status']!='LONG_COMPLETE':raise ValueError('finish the original method long review first')
    ssh=transport(read(OUTPUT/'board.local.json')['board'],execute_board=True)
    if ssh.boot_id()!=boot:raise ValueError('board boot changed; stop without rebooting')
    t0=time.monotonic();completed=0
    with locked(run):
        for network,items in selection['networks'].items():
            bundle=common/network/'capsule';manifest=validate(bundle)
            pool=read(bundle/'plans/rankings.json')['pool']
            remote=ssh.deploy(bundle,run/network);ssh.qualify_environment(remote,boot,run/network,bundle)
            for identity in items['configurations']:
                cfg=pool[identity]
                if ssh.boot_id()!=boot:raise ValueError('board boot changed during frozen long review')
                # Accepted identical long tasks are replayed, never measured twice.
                q=task(bundle,cfg,'qualify');qpath=run/'accepted'/boot/(q['task_id']+'.json')
                if not qpath.exists():raise ValueError('missing full-output qualification for long review')
                qualification=accept(ssh,q,bundle,remote,run,boot)
                accept(ssh,task(bundle,cfg,'long'),bundle,remote,run,boot,qualification['summary']['output_fingerprints'])
                completed+=1
                write(OUTPUT/'FINAL_LONG_STATE.json',dict(status='RUNNING',completed=completed,boot_id=boot,updated_at=now()),replace=True)
                print('冻结长窗口复核',completed,network,identity,flush=True)
            ssh.idle_command('cd '+shlex.quote(remote)+' && sha256sum -c board.sha256 > final_post_verify.log && sync',boot=boot,timeout=600)
        write(OUTPUT/'FINAL_LONG_STATE.json',dict(status='COMPLETE',completed=completed,boot_id=boot,
             elapsed_seconds=time.monotonic()-t0,updated_at=now()),replace=True)
        sync_tree(run)
    return read(OUTPUT/'FINAL_LONG_STATE.json')


def long_results():
    common=OUTPUT/'common_top20';run=ROOT/'runs/cross_network_common_top20_01'
    selection=read(OUTPUT/'FINAL_LONG_SELECTION.json');boot=selection['boot_id'];networks={};sources={};rows=[]
    for network,items in selection['networks'].items():
        bundle=common/network/'capsule';pool=read(bundle/'plans/rankings.json')['pool'];measured={}
        for identity in items['configurations']:
            item,source=load_result(run,bundle,pool[identity],'long',boot)
            qualification,_=load_result(run,bundle,pool[identity],'qualify',boot)
            if item['summary']['output_fingerprints']!=qualification['summary']['output_fingerprints']:
                raise ValueError('long output differs from qualification')
            measured[identity]=item['summary'];sources[network+'/'+identity]=source
            rows.append(dict(network=network,configuration_id=identity,**item['summary']))
        networks[network]={method:dict(configuration_id=identity,**measured[identity])
                           for method,identity in items['method_winners'].items()}
    result=dict(networks=networks,rows=rows,sources=sources,boot_id=boot,
        selection_sha256=sha256(OUTPUT/'FINAL_LONG_SELECTION.json'),DNN_inferences=104*len(rows),native_sessions=len(rows),
        same_boot_review_not_independent_start=True)
    freeze(OUTPUT/'FINAL_LONG_RESULTS.json',result)
    csv_write(OUTPUT/'all_methods_long_winners.csv',[dict(network=n,method=m,FPS=v['FPS'],II_ms=v['II_ms'],configuration_id=v['configuration_id'])
               for n,ms in networks.items() for m,v in ms.items()])
    return result


def report():
    """Render the complete common evidence, including negative observations."""
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    short_doc=read(OUTPUT/'FINAL_SHORT_RESULTS.json')
    long_doc=read(OUTPUT/'FINAL_LONG_RESULTS.json')
    if short_doc['boot_id']!=long_doc['boot_id']:raise ValueError('final report cannot mix boots')
    validate_rank(OUTPUT/'pool_temporal')
    if sha256(OUTPUT/'common_top20/SHORT_RESULTS.json')!=short_doc['common_short_sha256']:
        raise ValueError('common results changed after final selection')
    REPORT.mkdir(parents=True,exist_ok=True)
    english={'current':'Original capacity','semantics_revision':'Capacity without open-team sum',
             'fixed_family':'Disjoint-group family','mixed':'Mixed branches','tarnawski':'Tarnawski adaptation',
             'temporal_replace':'Phase replay: replace team sum','temporal_append':'Phase replay: append constraint'}
    colors=plt.cm.tab10(np.arange(len(LABELS)))
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for ax,(network,values) in zip(axes,short_doc['networks'].items()):
        for color,(method,label) in zip(colors,LABELS.items()):
            result=values['methods'][method]['budgets']
            ax.plot([1,5,20],[result[str(k)]['FPS'] for k in (1,5,20)],'o-',label=english[method],color=color)
        ax.set(title=network,xlabel='Complete-pipeline measurement budget',ylabel='FPS',xticks=[1,5,20])
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8,loc='best')
    for suffix in ('png','svg'):fig.savefig(REPORT/('final_same_budget.'+suffix),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5.8),layout='constrained')
    for ax,(network,methods) in zip(axes,long_doc['networks'].items()):
        y=np.arange(len(LABELS));ax.barh(y,[methods[m]['FPS'] for m in LABELS],color=colors)
        ax.set(yticks=y,yticklabels=[english[m] for m in LABELS],title=network,xlabel='FPS: frozen budget20 winner, long window')
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    for suffix in ('png','svg'):fig.savefig(REPORT/('final_long_review.'+suffix),dpi=180)
    plt.close(fig)
    lines=['# 双网络共同验证：最终性能与适用边界','',
           '所有名单在本批完整流水计时前冻结；以下只使用本批共同实测。短窗口、长窗口分开解释，不把帧或不同配置当独立启动。',
           '',f"启动身份：`{short_doc['boot_id']}`。全程未重启开发板。",'',
           '[方法与公式](METHOD_CURRENT.md)；[开发证据与反例](REPORT_zh.md)。','',
           '## 同预算短窗口结果','',
           '每项先做全输出资格，再测8预热/16完成时间戳/4尾帧。预算1、5、20分别表示在该冻结排序前缀中选出的最好结果。','']
    findings=[]
    for network,data in short_doc['networks'].items():
        methods=data['methods'];long_methods=long_doc['networks'][network]
        comparison=dict(network=network,
            short_budget20_gain_over_current_percent=100*(methods['temporal_replace']['budgets']['20']['FPS']/methods['current']['budgets']['20']['FPS']-1),
            long_winner_gain_over_current_percent=100*(long_methods['temporal_replace']['FPS']/long_methods['current']['FPS']-1),
            long_winner_gain_over_tarnawski_percent=100*(long_methods['temporal_replace']['FPS']/long_methods['tarnawski']['FPS']-1),
            same_budget20_winner_as_fixed_family=long_methods['temporal_replace']['configuration_id']==long_methods['fixed_family']['configuration_id'],
            removing_team_term_alone_better_at_budget20=methods['semantics_revision']['budgets']['20']['FPS']>methods['current']['budgets']['20']['FPS'],
            append_replace_same_budget_winners=all(methods['temporal_replace']['budgets'][str(k)]['configuration_id']==methods['temporal_append']['budgets'][str(k)]['configuration_id'] for k in (1,5,20)))
        findings.append(comparison)
    summary=['## 本轮直接结论','']
    for finding in findings:
        summary.append(f"- {finding['network']}：时序池内重排相对原模型，短窗口预算20差距{finding['short_budget20_gain_over_current_percent']:+.2f}%；冻结优胜配置长窗口差距{finding['long_winner_gain_over_current_percent']:+.2f}%，相对Tarnawski适配为{finding['long_winner_gain_over_tarnawski_percent']:+.2f}%。")
    if all(f['same_budget20_winner_as_fixed_family'] for f in findings):
        summary += ['', '两网络的时序前20优胜配置均与固定核组家族前20优胜相同。它证明本轮能将有效固定组候选纳入选点，但没有证明额外22.7分钟调用级采集优于直接测固定组家族Top-20。不能把全部收益归给新增时序模型。']
    if all(f['append_replace_same_budget_winners'] for f in findings):
        summary += ['', '保留/替代原线程组项的时序版本，在两网络预算1/5/20选中了相同优胜配置；本轮未观察到删除该项的额外选点价值。它不是严格物理下界，但仍可能起到有用的启发式排序作用。只删除该项的版本另有负向结果，不能将“语义更理想”当作性能必然改善。']
    summary += ['', '共享计算与主机活动的先后信息降低了本共同池的平均预测误差，但YOLO仍有明显低估阶段，问题仅得到部分改善。没有改变执行机制；这里比较的是选点贡献。', '']
    lines[8:8]=summary
    write(OUTPUT/'FINAL_FINDINGS.json',dict(findings=findings,short_sha256=sha256(OUTPUT/'FINAL_SHORT_RESULTS.json'),
        long_sha256=sha256(OUTPUT/'FINAL_LONG_RESULTS.json'),single_boot_observations=True),replace=True)
    for network,data in short_doc['networks'].items():
        lines += ['### '+network,'','| 方法 | 预算1 FPS | 预算5 FPS | 预算20 FPS | 预算20最佳名次 |',
                  '|---|---:|---:|---:|---:|']
        for method,label in LABELS.items():
            b=data['methods'][method]['budgets']
            lines.append('| '+label+' | '+' | '.join(f"{b[str(k)]['FPS']:.3f}" for k in (1,5,20))+f" | {b['20']['rank']} |")
        lines += ['',f"共同池共{data['pool_count']}项。时序方法是在该池内重排，不是全域时序目标的全局Top-20。",'']
        original=data['methods']['current']['budgets'];modified=data['methods']['temporal_replace']['budgets']
        changes={k:100*(modified[k]['FPS']/original[k]['FPS']-1) for k in ('1','5','20')}
        lines += ['时序替代相对原模型的短窗口选点差距：'+ '，'.join(f"预算{k}为{v:+.2f}%" for k,v in changes.items())+'。',
                  '这些是本次观测，不是统计显著或稳定收益；若为负值，同样保留。','']
    lines += ['## 冻结选点的较长窗口复核','',
              '每方法仅复核其短窗口预算20优胜配置，8预热/64时间戳/32尾帧；不按复核数据换点。64时间戳按63个间隔计算。','',
              '| 网络 | 方法 | 短窗口优胜 FPS | 长窗口 FPS | 配置 |','|---|---|---:|---:|---|']
    for network,methods in long_doc['networks'].items():
        for method,value in methods.items():
            short=short_doc['networks'][network]['methods'][method]['budgets']['20']
            lines.append(f"| {network} | {LABELS[method]} | {short['FPS']:.3f} | {value['FPS']:.3f} | `{value['configuration_id']}` |")
    lines += ['', '![同预算选点](final_same_budget.png)','', '![长窗口复核](final_long_review.png)','',
              '## 预测误差与消融','',
              '| 网络 | 原模型共同池MAPE | 去开放整组项MAPE | 时序替代MAPE | 时序附加MAPE |',
              '|---|---:|---:|---:|---:|']
    for network,value in short_doc['networks'].items():
        old=value['old_union_MAPE'];new=value['temporal_union_MAPE']
        lines.append(f"| {network} | {old['current']:.2f}% | {old['semantics_revision']:.2f}% | {new['replace']:.2f}% | {new['append']:.2f}% |")
    lines += ['', '同批阶段差距示例（不将并发阶段墙钟相加）：','',
              '| 网络 | 选点 | CPU阶段 | 隔离服务ms | 时序预测服务ms | 流水内服务ms |',
              '|---|---|---|---:|---:|---:|']
    pipeline_rows=read(OUTPUT/'common_top20/SHORT_RESULTS.json')['rows']
    for network,data in short_doc['networks'].items():
        pool=read(OUTPUT/'common_top20'/network/'RANKINGS.json')['pool']
        simulations={r['configuration']['id']:r for r in read(OUTPUT/'pool_temporal'/network/'RANKINGS.json')['all_scores']}
        for label,method,budget in [('原模型静态第一名','current','1'),('时序静态第一名','temporal_replace','1'),('时序前20优选','temporal_replace','20')]:
            identity=data['methods'][method]['budgets'][budget]['configuration_id']
            row=next(r for r in pipeline_rows if r['network']==network and r['configuration_id']==identity)
            indices=[i for i,s in enumerate(pool[identity]['stages']) if s['device']=='cpu']
            selected={indices[0],max(indices,key=lambda i:row['pipeline_stage_service_ms'][i])}
            for index in sorted(selected):
                program=pool[identity]['stages'][index]['program']
                predicted=simulations[identity]['models']['before']['replay']['stage_service_ns'][index]/1e6
                lines.append(f"| {network} | {label} | {program} | {row['isolated_stage_service_ms'][index]:.3f} | {predicted:.3f} | {row['pipeline_stage_service_ms'][index]:.3f} |")
    lines += ['', '此表保留模型仍偏乐观的配置。YOLO时序静态第一名的前段约154ms预测对197ms实测，说明完成依赖和主机调用先后仍不足以恢复实际共享执行。原执行器的CPU门控等待位于阶段服务计时之外，不能把其墙钟直接加到这些阶段服务上。全配置对照见`all_candidate_prediction_errors.csv`。', '',
              '## 优胜配置的负载分组','',
              '下表事后解释已冻结选点，不将其反写为硬编码规则。组负载是相应隔离服务之和，仅因真实同组串行而具有执行语义；不是开放并发阶段墙钟之和。','',
              '| 网络 | 计算核组 | 复用该组的CPU子图 | 隔离累计服务ms |','|---|---|---|---:|']
    for network,data in short_doc['networks'].items():
        identity=data['methods']['temporal_replace']['budgets']['20']['configuration_id']
        cfg=read(OUTPUT/'common_top20'/network/'RANKINGS.json')['pool'][identity]
        row=next(r for r in pipeline_rows if r['network']==network and r['configuration_id']==identity)
        groups={}
        for index,stage in enumerate(cfg['stages']):
            if stage['device']!='cpu':continue
            group=tuple(stage['mask']);entry=groups.setdefault(group,dict(stages=[],service=0))
            entry['stages'].append('单元'+str(stage['begin'])+'–'+str(stage['end']-1)+'（'+str(stage['threads'])+'线程）')
            entry['service']+=row['isolated_stage_service_ms'][index]
        for group,value in sorted(groups.items()):
            lines.append('| '+network+' | {'+','.join(map(str,group))+'} | '+'；'.join(value['stages'])+f" | {value['service']:.3f} |")
    lines += ['', '两例都采用三核组加单核组，但谁进入三核组由实际子图负载决定。ResNet重前段使用三核；YOLO将单元0单独作为单线程阶段，单元2–4等较重子图使用三核。不能从固定CPU0–2切图的失败推断整个网络不适合固定组。',
              '', 'ResNet本轮优胜的VTA主机固定核心2，但计算仍可使用核心2；YOLO优胜主机共享四核。它们都不是全系统核心独占，也不能据此统一指定核心2或要求永久预留主机核。', '']
    lines += ['',
       '原模型与去开放整组项使用同一宏观成本和完整域；时序附加/替代使用同一隔离轨迹、同一158项池，区别只在是否保留原整组启发项。它们用于区分旧项遮蔽与时序信息的选点贡献。固定组和混合分支是既有策略，不能把受限家族收益包装成新建模收益。',
       '',
       '较小MAPE不保证更好的前一项/前五项选点。容量代理不是严格整机上限；未测池外候选不据此判优劣。',
       '', '## 可以支持的机制解释','',
       '两个网络迁移时使用相同评分、搜索和实际流水执行器。YOLO没有漏做最小单元/profile/完整DP。前段并行计算回调CPU量变化较小，但流水完成跨度和调用线程等待尾部增长；只修核时总量不能恢复线程就绪及完成次序。',
       '',
       '本轮开发记录中，YOLO两段VTA的主机CPU量约67核ms，ResNet对应约16核ms，且工作散布在真实设备调用之间。两网络的输入大小、CPU子图形状、切点、frontier和尾段并行性不同。不能仅凭网络名称归因，也不能把全部差异归为DNN结构。',
       '',
       'CPU0–2隔离四线程在两网络都能加速；共享流水中不同阶段同时就绪，固定线程完成汇合会放大竞争和拖尾。主机工作量相近不代表出现时机相同。逐核容量与任务先后互补，但理想公平共享仍未等价恢复Linux调度、缓存和唤醒。',
       '',
       '此前六项未见组合的预设选点标准未通过；本轮任何正向观察都不改写该失败。条件主机集中、持续共跑、阻塞等待和迁移等路线也保留其正反证据，不重复宣称已解决。',
       '', '## 成本、复现与原始证据','']
    features=read(OUTPUT/'INTRINSIC_FEATURES.json')
    lines += ['CPU0–2四线程的隔离调用分解（本Goal同钟轨迹）：','',
              '| 网络 | 每帧并行启动 | 串行/启动CPU ms | worker计算核ms | 逐内核最慢worker之和 ms | 整段最忙worker ms |',
              '|---|---:|---:|---:|---:|---:|']
    for item in features['CPU']:
        if item['program']=='cpu_00_02' and item['threads']==4:
            lines.append(f"| {item['network']} | {item['kernels_per_call']} | {item['serial_and_launch_CPU_ms']:.3f} | {item['useful_worker_compute_CPU_ms']:.3f} | {item['ideal_kernel_critical_compute_ms']:.3f} | {item['ideal_aggregate_slot_compute_ms']:.3f} |")
    lines += ['', '两种汇总的差值很小，说明单独补完成汇合而不改变共享执行模型，不能解释数十毫秒慢化。YOLO本例并行启动数更少，却有更多计算与调用线程串行CPU工作；不能按启动次数或网络名称统一加惩罚。此表也不是两个网络在相同输入形状和相同子图结构下的因果对照。', '']
    component_state=read(ROOT/'runs/cross_network_pool_intrinsics_01/state.json')
    common=read(OUTPUT/'common_top20/SHORT_RESULTS.json')
    search_costs=[];search_sources={}
    for network in short_doc['networks']:
        proofs=read(OUTPUT/'common_top20'/network/'RANKINGS.json')['proofs']
        for method,proof in proofs.items():
            values=[]
            for path,expected in proof['files'].items():
                if Path(path).name not in {'shared4.json',*[f'host_shared_{c}.json' for c in range(4)],*[f'host_reserved_{c}.json' for c in range(4)]}:continue
                if sha256(path)!=expected:raise ValueError('search timing checkpoint changed')
                source=read(path);search_sources[path]=expected
                if 'seconds' in source:values.append(source['seconds'])
            search_costs.append(dict(network=network,method=method,recorded_contexts=len(values),
                search_seconds=sum(values) if values else None,
                scope='original context checkpoint search times; cached reuse, not a fresh cold full search; standalone oracle cost not separately aggregated'))
    runner_build=read(OUTPUT/'phase_profile/build/BUILD.json')
    observer_build=read(OUTPUT/'device_phase_02/build/BUILD.json')
    long_measurement_seconds=0
    for source in long_doc['sources'].values():
        if sha256(source['path'])!=source['sha256']:raise ValueError('long cost receipt changed')
        long_measurement_seconds+=read(source['path'])['measurement_and_collection_seconds']
    common_state=read(ROOT/'runs/cross_network_common_top20_01/state.json')
    costs=dict(search_checkpoints=search_costs,search_sources=search_sources,
        temporal_pool_scoring_seconds=read(OUTPUT/'pool_temporal/RANK_COST.json')['seconds'],
        new_intrinsic_collection_wall_seconds=component_state['elapsed_seconds'],
        common_short_measurement_and_collection_seconds=common['measurement_and_collection_seconds'],
        common_long_measurement_and_collection_seconds=long_measurement_seconds,
        common_long_flow_seconds=common_state['elapsed_seconds'],
        optional_observation_runner_build_seconds=runner_build['seconds'],
        optional_device_observer_build_seconds=observer_build['seconds'],
        new_stage_module_builds=0,new_runtime_builds=0,formal_runner_builds=0,
        deployment_only_timer_available=False,
        build_sources={str(OUTPUT/'phase_profile/build/BUILD.json'):sha256(OUTPUT/'phase_profile/build/BUILD.json'),
                      str(OUTPUT/'device_phase_02/build/BUILD.json'):sha256(OUTPUT/'device_phase_02/build/BUILD.json')})
    write(OUTPUT/'FINAL_COSTS.json',costs,replace=True)
    lines += [f"有限池补测84种隔离签名：168会话、418次CPU阶段调用、506次VTA阶段调用，0次完整DNN；总流程耗时{component_state['elapsed_seconds']/60:.1f}分钟。宏观成本775+714项未改。",
       '',f"共同短窗口：{common['new_common_short_configurations']}项配置、{common['new_short_native_sessions']}会话、{common['new_short_DNN_inferences']}次完整DNN；测量/回收累计{common['measurement_and_collection_seconds']/60:.1f}分钟。长窗口去重{len(long_doc['rows'])}项、{long_doc['DNN_inferences']}次完整DNN。组件调用、初始化、ARM仿真及完整推理分开计数。",
       '',
       f"本Goal可选观察执行器编译约{runner_build['seconds']:.2f}秒，设备调用观察库约{observer_build['seconds']:.2f}秒；本轮补测及正式流水直接复用它们，新增阶段模块、运行库及正式执行器编译均为0。时序池评分{costs['temporal_pool_scoring_seconds']:.2f}秒。长窗口实际测量/回收累计{long_measurement_seconds/60:.1f}分钟；额外入口对相同赢家只复放凭据，没有重复测量。",
       '', '完整搜索检查点中的历史搜索时间如下，不冒充此次冷重搜，也不把缓存核验的几秒钟当作全域搜索成本。固定家族未记录独立计时，记为未知而不是零。纯部署计时未单列；流程总耗时与测量/回收耗时分开保存，不将二者差值直接当成纯传输。',
       '', '| 网络 | 搜索版本 | 有时间记录的上下文 | 原检查点搜索累计秒 |','|---|---|---:|---:|',
       *[f"| {row['network']} | {LABELS[row['method']]} | {row['recorded_contexts']} | "+(f"{row['search_seconds']:.2f}" if row['search_seconds'] is not None else '未单列')+' |' for row in search_costs],
       '', '成本机器记录为`FINAL_COSTS.json`，含搜索检查点及构建来源哈希。原始775/714项宏观profile的历史总准备时间没有从本次回放重新推算。',
       '',
       '入口均在`vta/experiments/cpu_vta_partitioning/supplements/published_cpu_policies/`。现有凭据的resume会校验并复用；不要为了复现表格重测已有有效任务。',
       '', '```bash',
       'python run_pool_temporal.py validate',
       'python run_pool_temporal.py project',
       'python run_pool_temporal.py rank',
       'python run_common_top20.py validate',
       'python run_common_top20.py analyze',
       'python run_cross_network_final.py short',
       'python run_cross_network_final.py long-results',
       'python run_cross_network_final.py report',
       '```','',
       '结果：`artifacts/cross_network_model_01/FINAL_SHORT_RESULTS.json`、`FINAL_LONG_RESULTS.json`、`all_methods_top20.csv`、`all_methods_long_winners.csv`。逐候选原始文件、完整输出、绑核及boot凭据位于`runs/cross_network_common_top20_01/`；组件位于`runs/cross_network_pool_intrinsics_01/`。',
       '', '## 研究结论的边界','',
       '若方法只改善一网或部分预算，应限定为局部有效；若两网均无额外选点收益，应保留负向结论。全域DP→有限池时序重排的预选成本和覆盖限制必须单列。此次单输入、单启动、精简窗口不能证明通用最优或稳定提升。未测生产/收集、具体调度及缓存等仍是限制，不把本轮流水慢化回填为通用组件profile。','']
    (REPORT/'FINAL_REPORT_zh.md').write_text('\n'.join(lines))
    for network in short_doc['networks']:
        for method in LABELS:
            csv_write(REPORT/(network+'_'+method+'_top20.csv'),[r for r in short_doc['rows'] if r['network']==network and r['method']==method])
    delivered=[REPORT/'FINAL_REPORT_zh.md',REPORT/'METHOD_CURRENT.md',*sorted(REPORT.glob('final_*')),
               *[REPORT/(n+'_'+m+'_top20.csv') for n in short_doc['networks'] for m in LABELS]]
    manifest=dict(report=str(REPORT/'FINAL_REPORT_zh.md'),files={str(p):sha256(p) for p in delivered},
                  results={str(OUTPUT/p):sha256(OUTPUT/p) for p in ('FINAL_SHORT_RESULTS.json','FINAL_LONG_RESULTS.json','FINAL_FINDINGS.json','FINAL_COSTS.json')},
                  generated_at=now(),reporter_sha256=sha256(Path(__file__)))
    write(OUTPUT/'FINAL_REPORT_MANIFEST.json',manifest,replace=True)
    return dict(report=manifest['report'],csv_tables=14,complete_common_configurations=158)


def audit_delivery():
    """Offline completeness audit; a negative performance verdict can pass."""
    short=read(OUTPUT/'FINAL_SHORT_RESULTS.json');long=read(OUTPUT/'FINAL_LONG_RESULTS.json')
    selection=read(OUTPUT/'FINAL_LONG_SELECTION.json');manifest=read(OUTPUT/'FINAL_REPORT_MANIFEST.json')
    temporal=validate_rank(OUTPUT/'pool_temporal')
    if len(short['networks'])!=2 or short['boot_id']!=long['boot_id'] or short['boot_id']!=selection['boot_id']:
        raise ValueError('incomplete networks or mixed boot in final evidence')
    for path,expected in {**manifest['files'],**manifest['results']}.items():
        if sha256(path)!=expected:raise ValueError('delivered report or result hash differs: '+path)
    counts={}
    for network,data in short['networks'].items():
        capsule=OUTPUT/'common_top20'/network/'capsule';validate(capsule)
        rank=read(capsule/'plans/rankings.json');pool=set(rank['pool'])
        if set(data['methods'])!=set(LABELS):raise ValueError('not all frozen methods delivered')
        if data['pool_count']!=len(pool):raise ValueError('common pool count differs')
        for method,result in data['methods'].items():
            ids=result['selected_configurations']
            if len(ids)!=20 or len(set(ids))!=20 or not set(ids)<=pool:
                raise ValueError('missing unique measured Top20')
            if set(result['budgets'])!={'1','5','20'}:raise ValueError('missing measurement budget')
            for count,value in result['budgets'].items():
                if value['configuration_id'] not in ids[:int(count)]:raise ValueError('winner outside its frozen prefix')
            if long['networks'][network][method]['configuration_id']!=result['budgets']['20']['configuration_id']:
                raise ValueError('long review reselected a different winner')
        counts[network]=len(pool)
    expected={(n,i) for n,s in selection['networks'].items() for i in s['configurations']}
    if {(r['network'],r['configuration_id']) for r in long['rows']}!=expected or len(expected)>14:
        raise ValueError('frozen long winner coverage differs')
    if sum(counts.values())!=158:raise ValueError('formal common pool was not fully delivered')
    sources=[OUTPUT/'FINAL_SHORT_RESULTS.json',OUTPUT/'FINAL_LONG_RESULTS.json',OUTPUT/'FINAL_REPORT_MANIFEST.json',
             OUTPUT/'pool_temporal/RANK_FREEZE.json',OUTPUT/'common_top20/RANK_FREEZE.json',
             OUTPUT/'FEATURE_FREEZE.json',OUTPUT/'HISTORICAL_EVIDENCE.json',OUTPUT/'EXECUTOR_IDENTITY_CORRECTION.json',
             ROOT/'tests/test_pool_temporal_simulator.py',ROOT/'tests/test_pool_temporal_runtime.py',
             ROOT/'tests/test_cross_network_final.py']
    document=dict(status='EVIDENCE_COMPLETE',ready_for_goal_completion=True,boot_id=short['boot_id'],
        common_configurations=counts,method_lists=14,Top20_items_with_reused_measurements=280,
        short_native_sessions=316,short_DNN_inferences=4582,long_native_sessions=len(expected),
        long_DNN_inferences=104*len(expected),new_isolated_CPU_calls=418,new_isolated_VTA_calls=506,
        new_component_native_sessions=168,old_performance_reused=False,
        performance_verdict_may_be_negative=True,global_temporal_optimum_claimed=False,
        files={str(p):sha256(p) for p in sources},auditor_sha256=sha256(Path(__file__)),updated_at=now())
    write(OUTPUT/'DELIVERY_AUDIT.json',document,replace=True)
    return document
