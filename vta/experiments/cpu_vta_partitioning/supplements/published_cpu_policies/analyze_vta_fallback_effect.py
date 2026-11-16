"""Join isolated and pipeline effects without importing them as static costs."""
from pathlib import Path
from types import FunctionType
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256
from scheduler_gap_audit.curves import csv_write
import run_vta_fallback_threads as component
import run_vta_fallback_pipeline as pipeline

ROOT=component.ROOT
OUT=ROOT/'artifacts/vta_fallback_effect_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261111_vta_fallback_effect'


def run():
    c=component.analyze()
    # The frozen controller omitted its CSV-writer import in the analysis-only
    # branch. Supply that existing writer here; native code and receipts stay
    # unchanged, and the complete raw verifier still runs before reporting.
    p=FunctionType(pipeline.analyze.__code__,{**pipeline.analyze.__globals__,'csv_write':csv_write})()
    baseline=next(r for r in p['rows'] if r['VTA_CPU_threads']==[1,1]);rows=[]
    estimates=read(pipeline.POOL/'PLAN.json')['estimates']
    for row in p['rows']:
        e=next(x for x in estimates if x['configuration']==row['configuration'])
        rows.append(dict(configuration=row['configuration'],VTA_CPU_threads=row['VTA_CPU_threads'],
            FPS=row['FPS'],II_ms=row['II_ms'],FPS_change_percent=100*(row['FPS']/baseline['FPS']-1),
            front_CPU_service_ms=row['stage_service_ms'][0],
            middle_CPU_service_ms=row['stage_service_ms'][2],
            final_CPU_service_ms=row['stage_service_ms'][4],
            first_VTA_pipeline_service_ms=row['stage_service_ms'][1],
            second_VTA_pipeline_service_ms=row['stage_service_ms'][3],
            isolated_VTA_service_sum_ms=e['isolated_VTA_service_sum_ms'],
            isolated_VTA_run_and_boundary_core_ms=e['isolated_VTA_run_and_boundary_core_ms'],
            whole_process_core_ms_per_completion=row['whole_process_core_ms_per_completion']))
    best=min(rows,key=lambda r:(r['II_ms'],r['configuration']))
    result=dict(components=c['rows'],component_comparisons=c['comparisons'],pipelines=rows,
        component_source_receipts=c['source_receipts'],pipeline_source_receipts=p['source_receipts'],
        complete_DNN_inferences=87,isolated_VTA_component_calls=44,native_sessions=14,
        shared_boot_id=read(pipeline.RUN/'state.json')['boot_id'],
        no_new_runtime_modules_or_executor=True,no_static_model_or_DP_change=True,
        analysis_only_overlay='supply missing existing csv_write import; no measurement or verifier predicate changed',
        new_Top20_gain_not_proven=True,single_boot_short_window=True,
        observed_best=best['configuration'],
        observed_best_FPS_change_percent=best['FPS_change_percent'],
        observed_best_front_service_reduction_ms=baseline['stage_service_ms'][0]-best['front_CPU_service_ms'],
        front_CPU_isolation_not_remeasured_in_this_batch=True,
        isolation_to_pipeline_gap_not_resolved=True,
        no_new_global_profile_or_automatic_extra_measurement=True)
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    write(OUT/'RESULTS.json',result,replace=True);csv_write(OUT/'pipelines.csv',rows)
    text=['VTA子图内CPU线程：隔离收益与共享流水的匹配对照','',
        '背景：冻结YOLO VTA模块包含CPU回调，当前原配置的VTA运行时线程数为1。只改变该线程数，复用全部编译模块、运行库、执行器和输入。',
        '先测两个模块1/4线程，各资格1帧、标定2预热+8计入，共8会话44次子图调用。',
        '再冻结三个完整流水配置：VTA线程对1/1、4/1、4/4。CPU线程数4/3/1及核心集合0123/123/0不变，VTA主机和控制仍共享四核，门控关闭。',
        '完整流水每配置资格1帧，再8预热+16完成时间戳+4尾部；共6会话87次完整DNN。16个时间戳按15个间隔计算FPS。',
        '两批分别冻结、共同使用同一个boot；数据不是跨启动独立重复。没有重启、位流重载、漂移拒收或追加测量。', '',
        '一、隔离组件', '子图 / 1线程服务ms / 4线程服务ms / 加速比 / 4线程运行核时相对1线程']
    for row in c['comparisons']:text.append(f"{row['program']} / {row['one_thread_service_ms']:.3f} / {row['four_thread_service_ms']:.3f} / {row['service_speedup']:.4f} / {row['run_core_time_ratio']:.4f}")
    text+=['','四线程的辅助worker确有CPU增量，配置NumThreads与实际允许集合均通过检查。允许在组内迁移，不声称逐线程固定。',
        '二、完整流水', 'VTA线程对 / FPS / 相对1/1变化 / CPU前段ms / VTA前段ms / VTA后段ms / 窗口整体CPU核时每完成间隔ms']
    for row in rows:text.append(f"{row['VTA_CPU_threads']} / {row['FPS']:.4f} / {row['FPS_change_percent']:+.2f}% / {row['front_CPU_service_ms']:.3f} / {row['first_VTA_pipeline_service_ms']:.3f} / {row['second_VTA_pipeline_service_ms']:.3f} / {row['whole_process_core_ms_per_completion']:.3f}")
    text+=['','三、能支持的结论',
        f"本批相同条件下，VTA线程4/1取得{best['FPS']:.3f} FPS，比原1/1的{baseline['FPS']:.3f} FPS高{best['FPS_change_percent']:.2f}%；CPU前段服务少{baseline['stage_service_ms'][0]-best['front_CPU_service_ms']:.3f} ms。",
        '两个VTA阶段都开四线程的隔离服务之和最低，但整流水并未取得本批最高FPS。局部更快不能直接推出整体更快。',
        '差距尚未解决：前段流水服务仍约216 ms。本批没有重测该CPU前段的隔离成本，不能把其他批次的参考差值当作本批精确误差。',
        '上述改善是一次启动、每配置一次短窗口的观察值，不是稳定收益、完整搜索Top-20收益或新的静态模型贡献。',
        'VTA阶段包含可以使用TVM CPU线程池的工作；实际1/4线程输出完全一致，证明这个资源设置确实可执行。',
        '隔离服务下降与运行CPU核时增长必须同时考虑。节点级耗时仍未知，不能将设备调用外CPU核时全部归给激活或布局变换。',
        '线程设置会同时改变CPU竞争和执行时序。因此配置之间是这个资源策略的整体效果，不能把全部差异归给某一条内核或某一种等待。',
        '原成本表已经包含VTA run的CPU时间。本批不是发现以前将其记为零，也不将新核时再次叠加。',
        '配置仍是固定切图的小池，不是完整搜索Top-20或方法通用性验证。不能根据本批最好的点改写全域选点结果。',
        '四、后续边界',
        '若考虑接回搜索，须将VTA子图内CPU回调的worker资源与单物理加速器区分，保留原1线程选项，准备受影响的匹配profile。',
        '本批不自动扩大到全部VTA区间，不从两个模块缩放补点，也不将本批流水FPS或慢化回填为静态成本。',
        '下一步先复用已有VTA CPU回调轨迹，分开其计算与回调之外的主机工作，再检查隔离时序输入的缺口；不追加配置追逐更高FPS。']
    (REPORT/'FINDINGS_zh.txt').write_text('\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    ordered=sorted(rows,key=lambda x:x['VTA_CPU_threads']);labels=['/'.join(map(str,r['VTA_CPU_threads'])) for r in ordered]
    for ax,field,title,unit in [(axes[0],'FPS','Whole pipeline throughput','FPS'),
                               (axes[1],'front_CPU_service_ms','Unchanged front CPU stage','Service (ms)')]:
        values=[r[field] for r in ordered];ax.bar(labels,values,color=['#777777','#3d8caa','#bf7b42']);ax.set_title(title);ax.set_ylabel(unit)
        ax.set_xlabel('CPU threads inside the two VTA stages');ax.set_ylim(0,max(values)*1.17)
        for i,v in enumerate(values):ax.text(i,v+max(values)*.02,f'{v:.3f}',ha='center')
    fig.suptitle('YOLO: isolated VTA-graph CPU tuning under shared execution')
    fig.text(.5,.01,'Same boot, one short window per configuration. No new static ranking or stable-gain claim.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.93));fig.savefig(REPORT/'pipeline_effect.png',dpi=160);fig.savefig(REPORT/'pipeline_effect.svg');plt.close(fig)
    files=[OUT/'RESULTS.json',OUT/'pipelines.csv',REPORT/'FINDINGS_zh.txt',REPORT/'pipeline_effect.png',
        Path(__file__),component.POOL/'PLAN.json',pipeline.POOL/'PLAN.json']
    write(OUT/'MANIFEST.json',dict(files={str(p):sha256(p) for p in files},
        component_program_sha256=read(component.BUNDLE/'manifest.json')['phase_runner_sha256'],
        pipeline_program_sha256=read(pipeline.POOL/'yolov3_tiny416/capsule/manifest.json')['phase_runner_sha256']),replace=True)
    print(json.dumps(dict(pipelines=rows,report=str(REPORT/'FINDINGS_zh.txt')),ensure_ascii=False,indent=2));return result


if __name__=='__main__':run()
