"""Explain a bounded isolation/pipeline gap experiment; never changes scores."""
from datetime import datetime
from pathlib import Path
import json
import shutil
import statistics

from run_three_thread_host_gap import analyze,POOL,RUN,ROOT,RefreshedCosts
from cpu_vta_partitioning.common import read,write,sha256
from cpu_vta_partitioning.measurement import frames
from scheduler_gap_audit.curves import csv_write
from policy_branch_eval.ranking import inputs
from phase_model.simulator import rational

REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_three_thread_host_gap'
LABELS={
    'host_on_heavier_worker':'主机固定核心1：前段较重worker',
    'host_on_lighter_worker':'主机固定核心2：前段较轻worker',
    'host_outside_front_group':'主机固定核心3：前段核组外',
}


def run():
    result=analyze()
    selected=read(POOL/'ISOLATED_SELECTION.json')
    isolated=read(POOL/'CURRENT_ISOLATED_CPU_ONLY.json')['costs']
    pred=read(POOL/'PIPELINE_PREDICTIONS_FREEZE.json')
    stamp=(POOL/'PIPELINE_PREDICTIONS_FREEZE.json').stat().st_mtime
    clean=[]
    for p in (RUN/'accepted'/result['boot_id']).glob('*.json'):
        r=read(p)
        if r['task']['spec']['scope']=='pipeline':
            started=datetime.fromisoformat(r['measurement_started_at']).timestamp()
            if started<=stamp:raise ValueError('prediction file was not frozen before pipeline qualification')
        clean.append((p,r))
    if len(clean)!=10:raise ValueError('scope changed or session incomplete')
    rows=[];phase_rows=[]
    programs,records,ddr,*_=inputs('yolov3_tiny416')
    costs=RefreshedCosts(records,programs=programs,ddr=ddr,distribution='session_weights',overrides=isolated)
    for r in result['rows']:
        if r['scope']!='pipeline':continue
        cid=r['case'];m=r['measurement'];p=pred['results'][cid]
        row=dict(configuration_id=cid,role=r['role'],label=LABELS[r['role']],
            same_boot_isolated_front_ms=isolated['3']['service_ms'],
            pipeline_front_ms=m['stage_service_ms'][0],
            front_slowdown_ms=m['stage_service_ms'][0]-isolated['3']['service_ms'],
            front_slowdown_percent=100*(m['stage_service_ms'][0]/isolated['3']['service_ms']-1),
            pipeline_II_ms=m['ii_ms'],pipeline_FPS=m['fps'],
            pipeline_all_stage_ms=m['stage_service_ms'],
            process_core_ms_per_completion=m['process_core_ms_per_completion'],
            original_static_II_ms=p['original']['score_ns']/1e6,
            refreshed_static_II_ms=p['refreshed']['score_ns']/1e6,
            refreshed_uniform_static_II_ms=p['refreshed_uniform']['score_ns']/1e6,
            refreshed_capacity_terms=p['refreshed']['capacity_binding_terms'],
            CPU_demand_ns=p['refreshed']['CPU_core_demand_ns'],
            host_CPU_ns=p['refreshed']['scored_host_cpu_ns'],
            receipt=r['receipt'],receipt_sha256=r['receipt_sha256'])
        for mode in ('original','refreshed','refreshed_uniform'):
            row[mode+'_II_error_ms']=row[mode+'_static_II_ms']-row['pipeline_II_ms']
        rows.append(row)
        receipt=read(r['receipt']);task=receipt['task']
        frame_rows=frames(Path(receipt['raw'])/'result.jsonl',sum(task[k] for k in ('warmup','scored','tail')))
        kept=frame_rows[task['warmup']:task['warmup']+task['scored']]
        for index,stage in enumerate(task['stages']):
            component=costs.get(stage,task['configuration'])
            for operation in ('set','run','get'):
                measured=statistics.mean(f['stage'+str(index)+'_'+operation+'_ms'] for f in kept)
                phase_rows.append(dict(configuration_id=cid,role=r['role'],stage=stage['program'],
                    operation=operation,isolated_ms=component[operation+'_ms'],pipeline_ms=measured,
                    difference_ms=measured-component[operation+'_ms'],
                    isolated_source='same_boot_new_component' if index==0 else component['source_key'],
                    assumptions=component['assumptions'],
                    pipeline_wall_only_not_concurrent_process_CPU_attribution=True))
    rows.sort(key=lambda r:list(LABELS).index(r['role']))
    by={r['role']:r for r in rows}
    pairs=[]
    for name,other in [('重worker与轻worker','host_on_lighter_worker'),('重worker与前段核组外','host_outside_front_group')]:
        a,b=by['host_on_heavier_worker'],by[other]
        pairs.append(dict(pair=name,front_difference_ms=a['pipeline_front_ms']-b['pipeline_front_ms'],
            II_difference_ms=a['pipeline_II_ms']-b['pipeline_II_ms'],
            other_FPS_gain_percent=100*(b['pipeline_FPS']/a['pipeline_FPS']-1)))
    metrics={}
    for mode in ('original','refreshed','refreshed_uniform'):
        ordered=sorted(rows,key=lambda r:(r[mode+'_static_II_ms'],r['configuration_id']))
        metrics[mode]=dict(mean_absolute_II_error_ms=sum(abs(r[mode+'_II_error_ms']) for r in rows)/len(rows),
            static_order=[r['role'] for r in ordered],selected_FPS=ordered[0]['pipeline_FPS'])
    replay_path=POOL/'existing_model_diagnostic/RESULTS.json'
    replay=read(replay_path) if replay_path.exists() else None
    if replay and replay['native_source_sha256']!=sha256(POOL/'analysis/GAP_COMPARISON.json'):
        raise ValueError('diagnostic replay comparison source changed')
    intrinsic=ROOT/'artifacts/finite_service_01/development/INPUTS.json'
    profile=read(intrinsic)['isolated']['cases']['yolov3_tiny416/split']['stages'][0]['profile']
    kernel_rows=[]
    for index,k in enumerate(profile['kernels']):
        work=[float(sum((rational(x) for x in w['compute_segments']),0)/1e6) for w in k['workers']]
        kernel_rows.append(dict(kernel=index,function_offset=k['function_offset'],
            isolated_compute_ms_by_slot=work,worker2_over_worker1=work[2]/work[1] if work[1] else None,
            interpretation='isolated input only; no native per-kernel measurements in this batch'))
    out=dict(rows=rows,matched_pairs=pairs,prediction_metrics=metrics,
        phase_comparison=phase_rows,isolated_kernel_work=kernel_rows,
        isolated_CPU=isolated,isolated_real_compute_per_slot_ms=selected['isolated_kernel_compute_per_slot_ms'],
        boot_id=result['boot_id'],native_sessions=10,new_DNN=87,new_CPU_stage_calls=22,
        controls_shared=True,gate='off',no_model_or_runtime_change=True,
        no_new_general_predictor_validated=True,not_complete_domain_or_Top20=True,
        isolated_proxy_shares_not_new_per_window_thread_measurement=True,
        prediction_II_not_stage_service=True,single_boot_short_window=True,
        source_sha256=sha256(__file__),
        evidence={str(POOL/'analysis/RESULTS.json'):sha256(POOL/'analysis/RESULTS.json'),
            str(POOL/'analysis/GAP_COMPARISON.json'):sha256(POOL/'analysis/GAP_COMPARISON.json'),
            str(POOL/'PIPELINE_PREDICTIONS_FREEZE.json'):sha256(POOL/'PIPELINE_PREDICTIONS_FREEZE.json'),
            str(RUN/'state.json'):sha256(RUN/'state.json')})
    dest=POOL/'analysis'
    # The initial summary is already a hashed input of the replay. Preserve it.
    write(dest/'ENRICHED_GAP_COMPARISON.json',out,replace=True)
    csv_write(dest/'isolation_pipeline_gap.csv',rows)
    csv_write(dest/'host_matched_pairs.csv',pairs)
    csv_write(dest/'stage_phase_comparison.csv',phase_rows)
    csv_write(dest/'isolated_kernel_work.csv',kernel_rows)
    lines=['YOLO隔离profile与流水慢化：固定三线程计算组的主机位置匹配对照','',
        '一、这轮做了什么',
        '切图固定为CPU0–2→VTA3–6→CPU7–8→VTA9–12→CPU13。CPU资源固定为3线程012／1线程3／1线程3。两个VTA主机一起固定核心1、2或3，输入生产和结果收集仍共享四核，CPU门控关闭。',
        '同一输入0、同一运行库和执行器，组件2帧预热＋8帧计入；流水8帧预热＋16个完成时间戳＋4帧尾部，15个完成间隔。每种配置先做单帧完整输出和实际绑核资格检查。',
        '共10个原生会话，87次完整网络推理、22次隔离CPU阶段调用。没有重启、全域搜索、评分修改、详细探针或自动追加测量。','',
        '二、先排除隔离成本没有体现多线程收益的误解',
        f"本批前段三线程隔离服务{isolated['3']['service_ms']:.3f}ms，四线程{isolated['4']['service_ms']:.3f}ms；运行核时分别{isolated['3']['run_core_ms']:.3f}和{isolated['4']['run_core_ms']:.3f}核毫秒。运行核时包含运行时CPU工作，不是纯算术。", 
        '三线程的较重/较轻worker由隔离记录选择，未读取本批FPS：逻辑线程1约169.973、线程2约103.247核毫秒。调用线程0另有计算及交接工作，不把它当空闲核心。',
        '核心3仅在前段核组之外，中段和尾段仍使用核心3；它不是全系统专用主机核心。','',
        '三、同一CPU配置，仅改变主机位置的实测',
        '主机位置 | 前段隔离ms | 前段流水ms | 慢化ms | 流水II ms | FPS']
    for r in rows:
        lines.append(f"{r['label']} | {r['same_boot_isolated_front_ms']:.3f} | {r['pipeline_front_ms']:.3f} | {r['front_slowdown_ms']:.3f} | {r['pipeline_II_ms']:.3f} | {r['pipeline_FPS']:.4f}")
    for p in pairs:
        lines.append(f"{p['pair']}：前段差{p['front_difference_ms']:.3f}ms，完成间隔差{p['II_difference_ms']:.3f}ms，后者相对重worker的FPS变化{p['other_FPS_gain_percent']:+.3f}%。")
    lines += ['', '前段慢化发生在哪个阶段内操作：',
        '主机位置 | 隔离run ms | 流水run ms | run差ms | 流水set＋get ms']
    for r in rows:
        entries={p['operation']:p for p in phase_rows if p['configuration_id']==r['configuration_id'] and p['stage']=='cpu_00_02'}
        lines.append(f"{r['label']} | {entries['run']['isolated_ms']:.3f} | {entries['run']['pipeline_ms']:.3f} | {entries['run']['difference_ms']:.3f} | {entries['set']['pipeline_ms']+entries['get']['pipeline_ms']:.3f}")
    lines += ['这些run墙钟包含并行启动、计算、等待worker全部完成及返回，不能把增加的部分全部叫作计算CPU工作或CPU排队。流水内的进程CPU钟覆盖并发阶段，本表没有将其当作单阶段CPU核时。']
    lines += ['', '四、用相同公式区分成本刷新与线程份额的作用',
        '三种预测均在本批任何完整流水调用之前冻结。原表、仅刷新前段标量成本、刷新后将线程份额改为均分，使用同一已有评分函数。其他组件仍引用原隔离成本。',
        '预测是整流水完成间隔评分，不是前段服务时间；下表只与实测II比较，不能用它直接解释每段墙钟差额。',
        '主机位置 | 原表预测II | 刷新预测II | 刷新但均分预测II | 实测II']
    for r in rows:
        lines.append(f"{r['label']} | {r['original_static_II_ms']:.3f} | {r['refreshed_static_II_ms']:.3f} | {r['refreshed_uniform_static_II_ms']:.3f} | {r['pipeline_II_ms']:.3f}")
    lines += ['原非均匀份额是旧会话线程核时份额重缩放；本轮没有重新测逐worker份额，因此这是明确保留的近似。均分是离线输入消融，不是新执行策略。',
        '三项局部排名不等于完整DP或Top20验证；同分按原配置身份确定，不依据实测调整。没有将流水慢化或本批主机位置反填为通用成本。','',
        '轻worker并不在每个内核中都轻：第0个内核线程1／2计算约3.520／3.499ms，第3个内核约2.769／2.836ms。整段核时少，不代表各次完成汇合之前都存在可利用的空闲。',
        '这提示累计容量没有表达工作的发生顺序与局部完成依赖，但本批没有逐内核原生轨迹，不能把这一提示直接定量认定为全部慢化原因。']
    if replay:
        lines += ['', '已有时序模型的事后匹配回放（不是新模型或独立预测验证）：',
            '主机位置 | 时序回放前段ms | 实测前段ms | 差ms']
        for r in replay['rows']:
            lines.append(f"{LABELS[r['role']]} | {r['predicted_stage_ms'][0]:.3f} | {r['native_stage_ms'][0]:.3f} | {r['front_error_ms']:.3f}")
        lines += ['既有模型保留隔离内核工作、完成依赖及有限服务近似，没有重拟合CPU量或参数，且使用与本批相同的28帧窗口。',
            '它能够产生轻worker与核组外之间的前段差别，但重worker配置仍明显偏乐观。共享主机隔离时序迁移到固定主机、旧CPU轨迹跨启动使用、控制工作未建模等限制仍在。该回放只帮助定位容量汇总缺失的信息，没有使完整静态算法自动获得验证。']
    lines += ['',
        '五、这轮能够与不能够说明什么',
        '主机位置之间的差别是在CPU配置、线程数、控制位置相同的条件下取得，可用于检验主机与具体计算worker竞争。单输入、单启动、每项一轮短窗口，不宣称稳定增益、统计显著或等效。',
        '本轮三线程组与此前四线程重叠布局的CPU中段/尾段分配也不同，不能把跨这两个布局的FPS差全部归因于主机或前段线程数。',
        '即使本轮局部对照符合预测，也没有证明共享四核的理想调度模型已经正确，没有解决worker唤醒、CPU排队与完成拖尾的全部时序误差。',
        '本轮没有返回基准桥接或全域搜索。后续仍围绕已有有效轨迹与模型的关键worker、主机执行时序及排队差异核查，不把发现的时间差直接相加到评分。','',
        '文献与复用范围：Pipe-it的固定计算核组与阶段负载平衡、GCAPS的主机CPU需求和独立设备需求分开表示，提供资源语义依据；这里没有完整复现两篇系统，也没有实现它们未给出的Linux共享调度预测器。',
        'Pipe-it：https://arxiv.org/pdf/1903.05898',
        'GCAPS：https://drops.dagstuhl.de/storage/00lipics/lipics-vol298-ecrts2024/LIPIcs.ECRTS.2024.14/LIPIcs.ECRTS.2024.14.pdf',
        '', '已有ResNet18全域静态第一名约13FPS成果保持原身份，未被本轮局部YOLO实验替代。整体研究目标仍未完成。']
    (dest/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')
    REPORT.mkdir(parents=True,exist_ok=True)
    for name in ('GAP_COMPARISON.json','ENRICHED_GAP_COMPARISON.json','isolation_pipeline_gap.csv','host_matched_pairs.csv','stage_phase_comparison.csv','isolated_kernel_work.csv','FINDINGS_zh.txt'):
        shutil.copy2(dest/name,REPORT/name)
    if replay:shutil.copy2(replay_path,REPORT/'existing_model_diagnostic.json')
    return dict(report=str(REPORT/'FINDINGS_zh.txt'),rows=rows,pairs=pairs,prediction_metrics=metrics)


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
