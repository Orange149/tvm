"""Check whole-session versus scored-run thread shares using accepted records.

No cost is changed. Callback-only noncaller CPU may miss runtime bookkeeping;
the difference from process run CPU is reported, never spread across workers.
"""
from fractions import Fraction
from pathlib import Path
import json
import statistics as st
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,verify_inventory
from cpu_policies.open_resources.costs import Costs
from cpu_policies.open_resources.domain import stage
from scheduler_gap_audit.curves import csv_write

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/thread_share_scope_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_thread_share_scope'


def rational(value):return Fraction(value['numerator'],value['denominator'])


def run_work(profile,n):
    if n<1:raise ValueError('invalid participant count')
    work=[Fraction(0)]*n;work[0]=rational(profile['caller_CPU'])
    barrier_generations=0
    for kernel in profile['kernels']:
        barrier_generations+=kernel['barrier_generations']
        if not 1<=kernel['tasks']<=n or len(kernel['workers'])!=kernel['tasks']:
            raise ValueError('callback participant count differs')
        for slot,worker in enumerate(kernel['workers']):
            if slot:
                work[slot]+=sum((rational(x) for x in worker['compute_segments']),Fraction(0))
    if sum(work)!=rational(profile['known_CPU']):
        raise ValueError('known scoped CPU no longer matches caller plus noncaller callbacks')
    if any(x<0 for x in work) or sum(work)<=0:raise ValueError('invalid scoped CPU work')
    return work,barrier_generations


def normalize(values):
    total=sum(values)
    if total<=0 or any(v<0 for v in values):raise ValueError('invalid thread CPU counter')
    return [float(v/total) for v in values]


def run():
    intrinsic_path=ROOT/'artifacts/cross_network_model_01/pool_temporal/INTRINSICS.json'
    intrinsic=read(intrinsic_path);sources={str(intrinsic_path):sha256(intrinsic_path)}
    inputs={
        'resnet18':(ROOT/'artifacts/frame_stream_top20_static_validation/isolated_costs.json',
                    ROOT/'artifacts/bundles/8a942b2047c031339168f9d2/plans/programs.json'),
        'yolov3_tiny416':(ROOT/'artifacts/yolo_validation_01/ranking/inputs/isolated_costs.json',
                         ROOT/'artifacts/yolo_validation_01/ranking/inputs/programs.json')}
    costs={};programs={}
    for network,(cp,pp) in inputs.items():
        costs[network]=Costs(read(cp)['costs'],programs=read(pp),distribution='session_weights')
        programs[network]=read(pp);sources.update({str(p):sha256(p) for p in (cp,pp)})
    rows=[]
    for key,v in intrinsic['CPU'].items():
        identity=v['identity'];n=identity['threads']
        if n<2:continue
        provenance=v['provenance'];path=Path(provenance.get('observed_receipt',provenance.get('receipt','')))
        expected=provenance.get('observed_sha256',provenance.get('sha256'))
        if not expected or sha256(path)!=expected:raise ValueError('accepted component receipt changed')
        receipt=read(path);verify_inventory(receipt['raw'],receipt['files']);sources[str(path)]=expected
        if receipt['task']['spec']['scope']!='component':raise ValueError('pipeline data cannot define isolated shares')
        affinity=read(Path(receipt['raw'])/'affinity.json')['stages'][0]
        tids={w['slot']:w['tid'] for w in affinity['worker_slots']}
        if set(tids)!=set(range(n)):raise ValueError('logical thread identity incomplete')
        ledger={r['tid']:r for r in receipt['summary']['thread_ledger']}
        whole=[Fraction(str(ledger[tids[i]]['cpu_time_delta_ms'])) for i in range(n)]
        if any(ledger[tid]['scope']!='whole_session_including_warmup_and_tail' for tid in tids.values()):
            raise ValueError('whole-session counter scope changed')
        work,barriers=run_work(v['average'],n);known=sum(work);phase=normalize(work)
        network=identity['network'];program=identity['program'];p=programs[network][program]
        if p['files']['module.so']['sha256']!=identity['module_sha256']:
            raise ValueError('profile module differs from formal static-cost program')
        row=stage('cpu',p['begin'],p['end'],n,identity['mask'])
        formal=costs[network].get(row);old=formal['cpu_core_weights']
        if v.get('observed_cost'):
            total=Fraction(str(v['observed_cost']['run_core_ms']))*1_000_000
            residual=float(total-known)/1e6
            residue_percent=100*float((total-known)/total) if total else None
        else:residual=residue_percent=None
        if any(v<0 for v in whole):raise ValueError('whole-session CPU counter reversed')
        actual=normalize(whole) if sum(whole)>0 else None
        rows.append(dict(network=network,program=program,threads=n,source_boot=receipt['boot_id'],
            source_receipt=str(path),formal_source_key=formal['source_key'],
            original_formal_weights=old,same_session_whole_weights=actual,scored_run_known_weights=phase,
            run_known_CPU_ms=[float(x)/1e6 for x in work],
            scoped_known_CPU_ms=float(known)/1e6,unassigned_run_CPU_ms=residual,
            unassigned_run_CPU_percent=residue_percent,
            caller_whole_minus_run_percentage_points=100*(actual[0]-phase[0]) if actual else None,
            original_caller_minus_run_percentage_points=100*(old[0]-phase[0]),
            barrier_generations=barriers,callback_based_run_vector_available=barriers==0,
            complete_per_thread_run_budget_not_proven=True,
            noncaller_runtime_CPU_not_distributed=True,
            observed_service_ratio=v.get('observed_service_ratio'),
            serial_CPU_ms=float(sum((rational(x) for x in v['average']['serial_cpu_segments']),Fraction(0)))/1e6))
    summaries=[]
    for network in inputs:
        chosen=[r for r in rows if r['network']==network]
        visible=[r for r in chosen if r['same_session_whole_weights'] is not None]
        summaries.append(dict(network=network,matched_multithread_signatures=len(chosen),
            whole_session_counters_unobservable=len(chosen)-len(visible),
            mean_abs_caller_whole_vs_run_difference_percentage_points=st.mean(abs(r['caller_whole_minus_run_percentage_points']) for r in visible),
            max_abs_caller_whole_vs_run_difference_percentage_points=max(abs(r['caller_whole_minus_run_percentage_points']) for r in visible),
            barrier_free_signatures=sum(r['barrier_generations']==0 for r in chosen)))
    result=dict(rows=rows,summaries=summaries,sources=sources,
        source_sha256=sha256(__file__),
        no_pipeline_FPS_or_timing_read=True,new_native_sessions=0,new_DNN=0,
        no_formal_cost_or_formula_changed=True,full_DNN_gap_not_explained_by_this_audit=True,
        complete_scored_per_thread_clock_coverage_missing=True,
        no_all_signature_imputation_or_automatic_global_profile=True)
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    write(OUT/'RESULTS.json',result,replace=True);csv_write(OUT/'thread_share_scopes.csv',rows)
    text=['线程份额的测量作用域核查','',
        '只读取两个网络已有隔离profile与原始凭据；不读取流水FPS、改写成本或修改评分。',
        '同一观察会话中比较整会话线程计数与计入帧run区间：调用线程用run CPU时钟，非调用线程用已记录计算回调CPU之和。',
        '调用线程的回调CPU不重复相加；未分配的非调用线程运行时工作单列，不按均分或比例补造。',
        '整会话份额包含预热、池初始化、资格/空回调等实际发生的工作，以及较粗的线程计数口径；本表不能仅凭份额差异把它们进一步精确分摊。',
        '原正式成本也单列比较，其记录来自不同隔离批次，不冒充同进程直接因果比较。','',
        '网络 / 子图 / 线程 / 同会话整段caller份额 / 计入run已知caller份额 / run未分配核时ms']
    for r in rows:
        whole_label=f"{100*r['same_session_whole_weights'][0]:.1f}%" if r['same_session_whole_weights'] is not None else 'unobservable'
        text.append(f"{r['network']} / {r['program']} / {r['threads']} / {whole_label} / {100*r['scored_run_known_weights'][0]:.1f}% / {r['unassigned_run_CPU_ms'] if r['unassigned_run_CPU_ms'] is not None else 'unknown'}")
    text+=['','结论与边界：',
        '整会话CPU份额不能未经核查就视为稳态run阶段的线程份额。差异在短子图上尤其可能改变具体核心负载的归属。',
        '这不否定非均匀负载建模本身，但说明测量窗口和初始化开销必须与被缩放的run总核时匹配。',
        '当前按回调记录构造的份额仍有非调用线程运行时CPU残差，不等同于完整逐线程run时钟。存在屏障的签名单列，不用该向量冒充包含等待CPU的完整需求。',
        '尚未证明换份额会改善阶段预测或全域选点；尤其不能因看到差异就覆盖已取得13FPS结果的原成本。',
        '下一项最小验证应为三个已有子图（YOLO前段4线程、中段3线程、ResNet前段3线程）的run前后逐线程CPU时钟，不在每个内核插入详细探针。',
        '该小验证预计3次资格+3次2预热/8计入组件会话，共33次CPU子图调用、0次完整DNN；仅在单独冻结新身份后执行。',
        '完整域逐线程时钟补齐属于单独准备成本，本轮不自动启动，也不将已测份额缩放到缺失子图。']
    (REPORT/'FINDINGS_zh.txt').write_text('\n'.join(text)+'\n')
    examples=[r for r in rows if (r['network'],r['program'],r['threads']) in
        [('yolov3_tiny416','cpu_07_08',3),('yolov3_tiny416','cpu_00_02',4),('resnet18','cpu_00_02',3)]]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(examples),figsize=(12,4))
    for ax,r in zip(axes,examples):
        n=r['threads'];positions=list(range(n))
        ax.bar([i-.18 for i in positions],[100*v for v in r['same_session_whole_weights']],width=.36,label='Whole session',color='#8b959b')
        ax.bar([i+.18 for i in positions],[100*v for v in r['scored_run_known_weights']],width=.36,label='Scored run: known CPU',color='#3386aa')
        ax.set_xticks(positions,['Caller']+['Worker '+str(i) for i in range(1,n)])
        ax.set_ylim(0,60);ax.set_ylabel('Share (%)')
        ax.set_title(r['network']+'\n'+r['program']+', '+str(n)+' threads',fontsize=10)
        ax.text(.5,.94,f"Unassigned run CPU: {r['unassigned_run_CPU_percent']:.2f}%",ha='center',transform=ax.transAxes,fontsize=8)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False,fontsize=9)
    fig.suptitle('Measurement scope changes inferred thread load',y=.99)
    fig.text(.5,.015,'Same accepted observation within each pair. Noncaller runtime residual is reported, not distributed. No FPS or cost-table change.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.05,right=.99,bottom=.19,top=.70,wspace=.23)
    fig.savefig(REPORT/'thread_shares.png',dpi=150);plt.close(fig)
    write(OUT/'MANIFEST.json',dict(files={str(p):sha256(p) for p in
        [OUT/'RESULTS.json',OUT/'thread_share_scopes.csv',REPORT/'FINDINGS_zh.txt',REPORT/'thread_shares.png']}),replace=True)
    print(json.dumps(dict(summaries=summaries,examples=examples,report=str(REPORT/'FINDINGS_zh.txt')),ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':run()
