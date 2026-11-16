"""Common native measurements after fresh resource selection is frozen."""
from pathlib import Path
import json,shutil,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,digest,verify_inventory,now
from scheduler_gap_audit.curves import csv_write
from cpu_policies.open_resources.domain import canonical_tuple
from report_completion_study import describe
import completion_holdout_board as board

ROOT=board.ROOT
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_completion_holdout'
LABELS=dict(static='原评分在本轮未测局部池中的第一名',aggregate='同份新成本、聚合驱动时序',midpoint='同份新成本、设备独立完成')


def analyze():
    state=read(board.RUN/'state.json');plan=read(board.POOL/'BOARD_PLAN.json');selected=board.selection()
    if state['status']!='COMPLETE' or state['DNN_inferences']!=plan['DNN_inferences'] or state['boot_id']!=plan['expected_boot_id']:
        raise ValueError('fresh completion batch incomplete, inference count or boot differs')
    inputs=read(board.POOL/'INPUTS.json')['networks'];measured={};qual={};receipts={}
    for spec in board.specs():
        bundle=board.POOL/spec['network']/'capsule';board.validate(bundle)
        task=board.make_task(bundle,spec);path=board.RUN/'accepted'/state['boot_id']/(task['task_id']+'.json');item=read(path)
        verify_inventory(item['raw'],item['files'])
        if item['task']!=task or item['boot_id']!=state['boot_id'] or board.summary(item['raw'],task,bundle)!=item['summary']:
            raise ValueError('fresh completion receipt no longer validates')
        if Path(item['raw'],'task_hash').read_text().strip()!=digest(task):raise ValueError('native task digest differs')
        key=spec['network'],spec['case'];receipts[str(path)]=sha256(path)
        if spec['phase']=='qualify':qual[key]=item['summary']['output_fingerprints']
        else:
            if item['summary']['output_fingerprints']!=qual[key]:raise ValueError('performance output differs from full qualification')
            measured[key]=item['summary']
    rows=[];methods=[];errors=[];ranks=[]
    for network,data in selected['networks'].items():
        ids=data['selected_ids'];actual={cid:measured[network,cid] for cid in ids}
        winner=min(ids,key=lambda cid:(actual[cid]['ii_ms'],canonical_tuple(inputs[network][cid]['configuration'])))
        for cid in ids:
            m=actual[cid];cfg=inputs[network][cid]['configuration']
            row=dict(network=network,configuration_id=cid,configuration=describe(cfg),
                fps=m['fps'],ii_ms=m['ii_ms'],stage_service_ms=m['stage_service_ms'],
                process_core_ms_per_completion=m['process_core_ms_per_completion'],
                kept_timestamps=m['kept_timestamps'],gaps=m['gaps'],
                loss_to_measured_union_best_percent=100*(actual[winner]['fps']/m['fps']-1))
            for model in ('static','aggregate','midpoint','lower','upper'):
                row[model+'_predicted_II_ms']=data['scores'][cid][model]/1e6
            rows.append(row)
            for mode in ('aggregate','midpoint','lower','upper'):
                p=read(board.POOL/'predictions'/network/(cid+'_'+mode+'.json'))['result']['replay']
                for index,(s,ns,a) in enumerate(zip(cfg['stages'],p['stage_service_ns'],m['stage_service_ms'])):
                    errors.append(dict(network=network,configuration_id=cid,model=mode,stage=index,program=s['program'],
                        predicted_service_ms=ns/1e6,native_service_ms=a,absolute_error_ms=abs(ns/1e6-a)))
        for model in ('static','aggregate','midpoint'):
            choice=data['rankings'][model][0]
            if choice not in actual:raise ValueError('a frozen primary first choice was not measured')
            m=actual[choice]
            methods.append(dict(network=network,method=model,label=LABELS[model],selected_id=choice,
                fps=m['fps'],ii_ms=m['ii_ms'],pool_best_FPS=actual[winner]['fps'],
                loss_to_measured_union_best_percent=100*(actual[winner]['fps']/m['fps']-1),
                selection_budget=1,full_DP_Top20_comparison=False))
        for model,ranking in data['rankings'].items():
            for rank,cid in enumerate(ranking,1):
                ranks.append(dict(network=network,method=model,rank=rank,configuration_id=cid,
                    predicted_II_ms=data['scores'][cid][model]/1e6,
                    measured_FPS=actual[cid]['fps'] if cid in actual else None))
    gains=[]
    for network in selected['networks']:
        new=next(m for m in methods if m['network']==network and m['method']=='midpoint')
        for reference in ('static','aggregate'):
            old=next(m for m in methods if m['network']==network and m['method']==reference)
            gains.append(dict(network=network,reference=reference,FPS_gain_percent=100*(new['fps']/old['fps']-1),
                same_configuration=new['selected_id']==old['selected_id']))
    primary=[g for g in gains if g['reference']=='static']
    positive=all(g['FPS_gain_percent']>=0 for g in primary) and any(g['FPS_gain_percent']>0 for g in primary)
    refresh=read(board.POOL/'COST_REFRESH_AUDIT.json')
    result=dict(state=state,methods=methods,candidates=rows,stage_errors=errors,gains=gains,rankings=ranks,
        cost_refresh_only_counterfactual=refresh,
        comparison_scope='fixed topology, shared-four-core host/control, gate off, previously measured resource layouts excluded',
        global_Top1_or_Top20_result=False,historical_full_search_Top1_not_remeasured=True,
        source_receipts=receipts,selection_sha256=sha256(board.POOL/'SELECTION_FREEZE.json'),
        prospective_static_selection_gate_passed=positive,
        fresh_resource_layouts_under_known_topologies=True,no_new_Top20_claim=True,
        formal_executor_runtime_unchanged=True,old_FPS_not_used_as_new_results=True,
        single_input_single_boot_short_window=True,no_significance_claim=True,
        matching_new_cost_control='aggregate versus midpoint',
        old_static_comparison_also_includes_new_isolated_VTA_profile_acquisition=True,
        goal_complete=False,recorded_at=now(),source_sha256=sha256(__file__))
    dest=board.POOL/'analysis';write(dest/'RESULTS.json',result,replace=True)
    for name,values in [('native_candidates',rows),('method_top1',methods),('stage_errors',errors),('gains',gains),('all_rankings',ranks)]:
        csv_write(dest/(name+'.csv'),values)
    lines=['设备独立完成：未测资源配置共同验证','',
        '重要口径：以下第一名仅指本轮固定切图、主机/控制共享四核、门控关闭且排除已测配置后的14项局部候选第一名，不是完整切图DP的全域第一名。',
        '此前完整模型全域第一名测得13.001 FPS（20261108_model_ablation）和13.043 FPS（20261109_host_work_ablation），对应CPU线程3/1/1/1、核组012/3/0/3、VTA主机固定2、门控关闭。该配置未在本批重测。',
        '本批ResNet局部第一名线程4/2/1/1、核组0123/12/0/3、主机共享四核，测得11.156 FPS。不能把它解释为原13 FPS配置下降至11 FPS；本批也没有检验原13 FPS配置能否在当前启动复现。',
        '在两个固定切图内，每个网络14项尚未执行的资源放置先完成4种冻结预测；只测原静态、同成本聚合时序、设备独立完成三个方法第一名的并集。',
        '原静态使用原冻结隔离成本；另外两种使用完全相同的新VTA隔离成本。两者差异用于识别时序建模贡献，相对原静态的差异包含新标定这一准备变化，不能全部归因于模型。',
        '设备完成模型及CPU调度近似在该验证前已冻结，没有根据本批FPS更新参数、名单或完成区间中点。',
        f"本批{state['total_sessions']}个原生会话、{state['DNN_inferences']}次完整DNN；输入0，每配置1帧资格检查，之后8预热+16时间戳+4尾部，以15个间隔计算FPS。没有观察适配库，没有重启、锚点或自动重测。",'']
    for m in methods:lines.append(f"{m['network']} | {m['label']}：{m['fps']:.6f} FPS，{m['ii_ms']:.3f}ms。")
    lines+=['','相同本批测量的选点差异：']
    for g in gains:lines.append(f"{g['network']}，设备独立完成相对{LABELS[g['reference']]}：{g['FPS_gain_percent']:+.3f}%；相同配置={g['same_configuration']}。")
    lines+=['','测量前单独冻结了“仅更新5项VTA成本、原静态公式与CPU成本不变”的反事实：']
    for network,d in refresh['networks'].items():
        lines.append(f"{network}：更新成本前后静态第一名相同={d['top1_unchanged']}。这只识别本池预算一次的选点作用，不等于成本刷新对全部分数或预算20无影响。")
    lines+=['','逐配置：']
    for r in rows:lines.append(r['network']+' '+r['configuration']+f"：{r['fps']:.6f}FPS，前段{r['stage_service_ms'][0]:.3f}ms。")
    lines+=['','关键阶段的剩余误差：']
    for network in selected['networks']:
        cid=selected['networks'][network]['rankings']['midpoint'][0]
        group=[e for e in errors if e['network']==network and e['configuration_id']==cid and e['stage']==0]
        a=next(e for e in group if e['model']=='aggregate');b=next(e for e in group if e['model']=='midpoint')
        lines.append(f"{network}本轮局部池第一名：前段实测{b['native_service_ms']:.3f}ms，聚合驱动模型{a['predicted_service_ms']:.3f}ms，独立完成模型{b['predicted_service_ms']:.3f}ms。")
    lines+=['部分VTA阶段的误差下降，同时关键CPU前段可能更乐观。语义更细不保证关键线程调度正确，也不保证选点更好，不能用全阶段平均误差代替选点判断。',
        '下一项可辨识缺口：原平台混合负载验证使用1ms睡眠、每段30us或1ms CPU；新驱动记录显示轮询周期约60us、单次CPU仅数us。旧检查没有证明CPU调度近似适用于这种频繁短唤醒。',
        '后续若继续，先用不运行DNN和FPGA的受控短唤醒负载检验CPU服务及落核规则；先冻结程序工作量和判据，不能用本批DNN FPS调参数。尚未执行这项自测。']
    lines+=['','判断：'+('达到预设的小范围正向观察条件；下一步仍需核算完整候选所需profile与接入成本，不能称已完成全域Top-20改善。' if positive else '未达到预设的双网络选点条件，不据此扩大完整DP或新Top-20测量。'),
        '完成区间两端属于隔离设备测量的不确定性，不是共享流水性能上下界；端点改变排名的情况保留在all_rankings.csv。',
        'CPU调度、主机实际落核、访存变化及控制CPU开销仍有近似，不能从本批选择结果推断阶段慢化已全部解决。',
        '新布局的未知性审计覆盖当前项目的接受凭据和配置清单，未穷尽项目建立前的历史脚本。',
        '未测配置的FPS留空，不用少量配置冒充预算5/20或全局Top-20结果。',
        f"成本：原生准备与执行{state['seconds']:.3f}秒；112项离线预测排名{read(board.POOL/'RANK_COST.json')['seconds']:.3f}秒；设备完成区间获取15组件会话、105次VTA调用另计。", 
        '原始凭据索引：'+str(dest/'RESULTS.json'),
        '复现：python run_completion_holdout.py validate；python analyze_completion_holdout.py。已完成board批次不需重新执行。']
    (dest/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')
    plot(dest,methods)
    REPORT.mkdir(parents=True,exist_ok=True)
    for path in dest.iterdir():
        if path.is_file():shutil.copy2(path,REPORT/path.name)
    write(board.POOL/'DELIVERY.json',dict(goal_complete=False,prospective_gate_passed=positive,
        results_sha256=sha256(dest/'RESULTS.json'),files={str(p):sha256(p) for p in REPORT.iterdir() if p.is_file()}),replace=True)
    p=ROOT/'artifacts/static_revision_01/GOAL_PROGRESS.json';progress=read(p)
    progress.update(updated_at=now(),goal_status='ACTIVE',full_goal_completion_proven=False,
        phase='independent device completion frozen and new resource layout check completed',
        current_turn_new_native_sessions=15+state['total_sessions'],current_turn_new_DNN_inferences=state['DNN_inferences'],
        current_turn_VTA_component_calls=105,no_DNN_inferences_this_goal_turn=False,
        new_completion_prospective_results=str(dest/'RESULTS.json'),
        new_completion_prospective_gate_passed=positive,
        next_action='verify finite-service assumptions at measured tens-of-microseconds polling cadence using bounded zero-DNN synthetic controls; old platform validation used millisecond sleepers; no full-DP expansion')
    write(p,progress,replace=True)
    return dict(methods=methods,gains=gains,positive=positive,report=str(REPORT/'FINDINGS_zh.txt'))


def plot(dest,methods):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,network in zip(axes,('resnet18','yolov3_tiny416')):
        values=[next(m['fps'] for m in methods if m['network']==network and m['method']==model)
                for model in ('static','aggregate','midpoint')]
        bars=ax.bar(range(3),values,color=['#999999','#4c78a8','#f58518'])
        ax.set_xticks(range(3),['Static: local pool','Matched aggregate','Independent device'],rotation=15,ha='right')
        ax.set_title(network+' / fixed-topology unseen pool');ax.set_ylabel('FPS');ax.set_ylim(0,max(values)*1.18)
        for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v,f'{v:.3f}',ha='center',va='bottom')
    fig.tight_layout();fig.savefig(dest/'fresh_top1.png',dpi=180);plt.close(fig)

if __name__=='__main__':print(json.dumps(analyze(),ensure_ascii=False,indent=2))
