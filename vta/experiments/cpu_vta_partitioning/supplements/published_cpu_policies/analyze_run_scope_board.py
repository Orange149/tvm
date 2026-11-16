"""Analyze only accepted measurements of the frozen common-validation batch."""
from pathlib import Path
import argparse
import json
import math
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256,verify_inventory
from common_top20.runtime import task,summary
from scheduler_gap_audit.curves import csv_write
from run_scope_board import ROOT,OUT,RUN,selection,assert_sources,validate
from run_scope_rank import freeze

METHOD_LABELS=dict(tarnawski='Tarnawski平台适配',original_complete='已有最强完整模型',
                   run_scope_complete='计算区间线程份额修正')


def budget_choices(lists,measurements):
    result=[]
    if not set(lists).issubset(METHOD_LABELS):
        raise ValueError('unknown comparison method')
    for method in METHOD_LABELS:
        if method not in lists:
            continue
        rows=lists[method]
        if len(rows)!=20 or len({r['id'] for r in rows})!=20:
            raise ValueError('twenty distinct frozen configurations required')
        for budget in (1,5,20):
            candidates=rows[:budget]
            for row in candidates:
                value=measurements[row['id']]
                if (type(value['II_ms']) not in (int,float) or not math.isfinite(value['II_ms']) or
                        value['II_ms']<=0 or type(value['FPS']) not in (int,float) or
                        not math.isfinite(value['FPS']) or value['FPS']<=0 or
                        abs(value['FPS']-1000/value['II_ms'])>1e-8):
                    raise ValueError('invalid measured interval or FPS')
            chosen=min(candidates,key=lambda r:(measurements[r['id']]['II_ms'],r['id']))
            value=measurements[chosen['id']]
            result.append(dict(method=method,method_label=METHOD_LABELS[method],budget=budget,
                configuration=chosen['id'],static_rank=chosen['static_rank'],
                FPS=value['FPS'],II_ms=value['II_ms'],same_measurement_shared_between_methods=True,
                one_input_one_boot_one_short_window_not_independent_confirmation=True))
    return result


def require_complete(state,plan):
    if (state['status']!='COMPLETE' or state['completed_configurations']!=plan['configurations'] or
            state['native_sessions']!=plan['native_sessions'] or state['DNN_inferences']!=plan['DNN_inferences']):
        raise ValueError('common native measurements are incomplete')


def analyze(run_dir=RUN):
    run_dir=Path(run_dir);prepared=assert_sources();plan,pool,order=selection()
    state=read(run_dir/'state.json');require_complete(state,plan)
    identity=read(run_dir/'run.json');boot=identity['boot_id']
    if (boot!=state['boot_id'] or identity['bundles']!=prepared['bundles'] or
            identity['selection_sha256']!=sha256(OUT/'FREEZE.json') or
            identity['preparation_sha256']!=sha256(OUT/'PREPARATION.json')):
        raise ValueError('measurement run identity differs')
    output=OUT/'analysis'
    if (output/'MANIFEST.json').exists():
        cached=read(output/'RESULTS.json')
        if (cached['source_sha256']!=sha256(__file__) or
                cached['selection_sha256']!=sha256(OUT/'FREEZE.json')):
            raise ValueError('sealed analysis identity changed; use an independent analysis revision')
        for path,wanted in read(output/'MANIFEST.json')['files'].items():
            if sha256(path)!=wanted:raise ValueError('sealed analysis changed')
        for path,wanted in cached['sources'].items():
            if sha256(path)!=wanted:raise ValueError('source receipt changed after analysis')
            receipt=read(path);verify_inventory(receipt['raw'],receipt['files'])
        return cached
    lists=read(OUT/'LISTS.json');measurements={n:{} for n in pool};sources={};all_rows=[];expected_receipts=set()
    session_seconds=0
    for network,candidates in pool.items():
        bundle=OUT/network/'capsule';validate(bundle)
        for key,cfg in candidates.items():
            qualified=None
            for phase in ('qualify','short'):
                document=task(bundle,cfg,phase)
                path=run_dir/'accepted'/boot/(document['task_id']+'.json');expected_receipts.add(path.resolve())
                receipt=read(path)
                if receipt['task']!=document or receipt['boot_id']!=boot:
                    raise ValueError('accepted task or boot differs')
                verify_inventory(receipt['raw'],receipt['files'])
                value=summary(receipt['raw'],document,bundle)
                if value!=receipt['summary']:
                    raise ValueError('accepted measurement no longer replays')
                sources[str(path)]=sha256(path)
                session_seconds+=receipt['measurement_and_collection_seconds']
                if phase=='qualify':
                    qualified=value['output_fingerprints']
                else:
                    if value['output_fingerprints']!=qualified:
                        raise ValueError('performance output differs from complete qualification')
                    measurements[network][key]=value
                    all_rows.append(dict(network=network,configuration=key,FPS=value['FPS'],II_ms=value['II_ms'],
                        CPU_ms_per_completion=value['process_CPU_ms_per_completion'],
                        stage_service_ms=value['stage_service_ms'],context=cfg['context_id'],
                        CPU_gate=cfg['execution_mode'],CPU_stages=[s for s in cfg['stages'] if s['device']=='cpu'],
                        host_mask=cfg['host_mask'],control_mask=cfg['control_mask'],
                        measurement_receipt=str(path),measurement_receipt_sha256=sha256(path)))
    actual={p.resolve() for p in (run_dir/'accepted'/boot).glob('*.json')}
    if expected_receipts!=actual:
        raise ValueError('missing, duplicate, or foreign accepted session')
    budgets=[]
    for network in pool:
        budgets.extend(dict(network=network,**r) for r in budget_choices(lists[network],measurements[network]))
    output.mkdir(parents=True,exist_ok=True)
    freeze(output/'STATE_SNAPSHOT.json',state)
    result=dict(rows=all_rows,budgets=budgets,sources=sources,boot_id=boot,
        new_native_sessions=plan['native_sessions'],DNN_inferences=plan['DNN_inferences'],
        method_versions_do_not_duplicate_measurements=True,old_FPS_not_reused=True,
        input_scope_correction_not_a_new_scoring_formula=True,
        unmodeled_shared_timing_not_claimed_solved=True,source_sha256=sha256(__file__),
        selection_sha256=sha256(OUT/'FREEZE.json'),state_sha256=sha256(run_dir/'state.json'),
        state_snapshot_sha256=sha256(output/'STATE_SNAPSHOT.json'),
        accepted_session_measurement_and_collection_seconds=session_seconds,
        last_controller_invocation_seconds=state['seconds'],
        deployment_time_not_separately_measured=True)
    freeze(output/'RESULTS.json',result)
    csv_write(output/'candidates.csv',all_rows);csv_write(output/'budgets.csv',budgets)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    colors=['#8a9399','#377fa8','#d39535'];labels=['Tarnawski adaptation','Existing complete model','Run-scoped shares']
    for ax,network in zip(axes,pool):
        for j,method in enumerate(METHOD_LABELS):
            values=[next(r['FPS'] for r in budgets if r['network']==network and r['method']==method and r['budget']==k)
                    for k in (1,5,20)]
            bars=ax.bar([i+(j-1)*.24 for i in range(3)],values,width=.24,color=colors[j],label=labels[j])
            ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
        ax.set_xticks(range(3),['1','5','20']);ax.set_xlabel('Full-pipeline selection budget')
        ax.set_ylabel('Observed FPS');ax.set_title(network)
        ax.set_ylim(0,ax.get_ylim()[1]*1.12)
    handles,legend=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='upper center',ncol=3,frameon=False,fontsize=9)
    fig.text(.5,.015,'One input, one boot, one short window per configuration. Shared configurations use the same measurement.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.065,1,.90]);fig.savefig(output/'budget_FPS.png',dpi=180);fig.savefig(output/'budget_FPS.svg');plt.close(fig)
    lines=['计算区间线程份额修正：共同测量结果','',
        '同批输入0、相同原执行器/模块/运行库，单启动、每配置一次8预热+16完成时间戳+4尾部；16个时间戳对应15个间隔。',
        '每项性能之前均做单帧全阶段输出资格检查。原始输出、实际绑定、帧、boot及文件哈希均经回放核对。',
        '三组方法共同实测去重并集，旧FPS未作为本批结果复用。新版本只修改CPU线程份额输入，其他公式及总成本保持不变。',
        '预算内最佳结果取相应静态名单前K项的实测最佳，不用这些结果修改名单或成本。','',
        '网络 / 方法 / 完整流水预算 / FPS / II ms / 选中静态名次']
    for r in budgets:
        lines.append(f"{r['network']} / {r['method_label']} / {r['budget']} / {r['FPS']:.4f} / {r['II_ms']:.3f} / {r['static_rank']}")
    lines+=['',
        'ResNet18新旧有序Top-20相同；两条性能曲线共用逐项测量，选择相同应报告零选点差异。',
        '输入作用域修正贡献、已有完整模型对Tarnawski的贡献分开报告。不得把静态输入修正描述为已经修复全部流水阶段慢化。',
        '本批没有独立启动重复、统计显著性或等效性结论；帧和配置不作为独立重复。',
        f"实际候选{plan['configurations']}项，原生会话{plan['native_sessions']}个，完整网络推理{plan['DNN_inferences']}次。",
        f"已验收会话的测量/回收累计{session_seconds:.2f}s；最后一次控制器调用{state['seconds']:.2f}s，后者不冒称跨中断累计总耗时。",
        '部署耗时未单独采样；无中断时最后一次调用包含部署、测量和回收。此前profile、静态搜索及本地准备成本另列。',
        '启动空回调探针开启，未加载详细线程/排队探针；未逐次采样初始化耗时，初始化不计为完整DNN推理。',
        '原ResNet18约13FPS历史成果保留原身份；本批同配置结果可作为共同环境桥接，不能混同成原程序的第二次历史测量。']
    (output/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')
    write(output/'MANIFEST.json',dict(files={str(p):sha256(p) for p in output.iterdir() if p.is_file() and p.name!='MANIFEST.json'}))
    print(json.dumps(dict(budgets=budgets,report=str(output/'FINDINGS_zh.txt')),ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,default=RUN);args=parser.parse_args()
    analyze(args.run_dir)
