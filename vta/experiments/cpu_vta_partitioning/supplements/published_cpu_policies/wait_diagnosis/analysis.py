"""Replay evidence and separate official throughput from observer-only metrics."""
import csv
import json
from pathlib import Path
from cpu_vta_partitioning.common import digest,read,sha256,verify_inventory,write
from .common import assert_prepared
from .prepare import validate_bundle
from .runtime import make_task,summary
from .campaign import key,decide


def load_results(output,run_dir):
    output,run_dir=Path(output),Path(run_dir);bundle=output/'capsule';assert_prepared(output);validate_bundle(bundle)
    state=read(run_dir/'state.json')
    if state['status']!='COMPLETE':raise ValueError('waiting experiment not complete')
    boot=state['boot_id'];decision=read(run_dir/'BLOCKING_DECISION.json');results={};evidence={};cost=0.
    full=components=0
    for spec in read(bundle/'plans/task_order.json'):
        if spec['stage']==2 and not decision['execute_blocking']:continue
        for qual in (True,False):
            doc=make_task(bundle,spec,qual);receipt=run_dir/'accepted'/boot/(doc['task_id']+'.json');item=read(receipt)
            if item['task']!=doc or item['boot_id']!=boot:raise ValueError('accepted wait task differs')
            verify_inventory(item['raw'],item['files']);raw=Path(item['raw'])
            if (raw/'boot_id').read_text().strip()!=boot or (raw/'task_hash').read_text().strip()!=digest(doc):
                raise ValueError('raw wait boot or task changed')
            result=summary(raw,doc,bundle)
            if result!=item['summary']:raise ValueError('wait summary no longer replays')
            if not qual:results[key(spec)]={**result,'raw':str(raw)}
            evidence[str(receipt)]=sha256(receipt);cost+=item['measurement_and_collection_seconds']
            n=sum(doc[k] for k in ('warmup','scored','tail'))
            if spec['scope']=='pipeline':full+=n
            else:components+=n
    recomputed=decide(results)
    if any(decision[k]!=v for k,v in recomputed.items()):raise ValueError('pre-intervention decision does not replay')
    if full!=state['DNN_inferences'] or components!=state['component_calls']:
        raise ValueError('inference/stage call count differs')
    return results,dict(boot_id=boot,decision=decision,receipt_hashes=evidence,
        native_sessions=len(evidence),DNN_inferences=full,isolated_stage_calls=components,
        measurement_and_collection_seconds=cost)


def analyze(output,run_dir):
    output,run_dir=Path(output),Path(run_dir);results,evidence=load_results(output,run_dir)
    target=run_dir/'analysis';target.mkdir(parents=True,exist_ok=True)
    rows=[];workers=[];groups=[];comparisons=[]
    for (scope,case,policy,probe),value in results.items():
        row=dict(scope=scope,case=case,wait_policy=policy,probe=probe,
            stage0_service_ms=value['stage_service_ms'][0],
            official_FPS=value.get('fps') if not probe and scope=='pipeline' else None,
            observer_FPS=value.get('fps') if probe and scope=='pipeline' else None,
            whole_process_core_ms=value.get('process_core_ms_per_completion'),
            completion_ii_ms=value.get('ii_ms'),raw=value['raw'])
        rows.append(row)
        if probe:
            for worker in value['wait']['workers']:workers.append(dict(scope=scope,case=case,wait_policy=policy,**worker))
            for stage,metrics in value['wait']['stages'].items():
                groups.append(dict(scope=scope,case=case,wait_policy=policy,stage=stage,**metrics))
    if evidence['decision']['execute_blocking']:
        for case in ('overlap4','split3'):
            a=results[('pipeline',case,'original',False)];b=results[('pipeline',case,'blocking',False)]
            ao=results[('pipeline',case,'original',True)]['wait']['stages']['0']
            bo=results[('pipeline',case,'blocking',True)]['wait']['stages']['0']
            # CPU times can be summed across threads. Their wall waits cannot.
            def wait_cpu_per_launch(s):return (s['barrier_cpu_ns']+s['completion_tail_cpu_ns'])/s['launches']/1e6
            comparisons.append(dict(case=case,original_FPS=a['fps'],blocking_FPS=b['fps'],
                FPS_change_percent=100*(b['fps']/a['fps']-1),
                original_stage0_ms=a['stage_service_ms'][0],blocking_stage0_ms=b['stage_service_ms'][0],
                original_wait_cpu_ms_per_launch=wait_cpu_per_launch(ao),
                blocking_wait_cpu_ms_per_launch=wait_cpu_per_launch(bo),
                stage_and_throughput_improved=b['fps']>a['fps'] and b['stage_service_ms'][0]<a['stage_service_ms'][0]))
    for leaf,data in [('sessions.csv',rows),('workers.csv',workers),('parallel_groups.csv',groups),('comparisons.csv',comparisons)]:
        if data:
            with (target/leaf).open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    serializable={'|'.join(map(str,k)):v for k,v in results.items()}
    write(target/'RESULTS.json',dict(results=serializable,comparisons=comparisons,evidence=evidence,
        official_performance_uses_probe_off=True,offCPU_not_separated=True,
        thread_wall_waits_not_summed_into_II=True,scoring_unchanged=True),replace=True)
    write(target/'COSTS.json',dict(**evidence,preparation=read(output/'capsule/plans/preparation_cost.json'),
        old_performance_reused=False,whole_session_traces_not_new_static_costs=True),replace=True)
    labels={'overlap4':'四线程重叠共享','split3':'三线程计算分组'}
    text=['# YOLO共享慢化：等待诊断与阻塞等待对照','',
        '单输入、单启动观察；正式FPS来自关闭探针的会话。原运行库、执行器、模块及评分保持原身份。','',
        '## 第一阶段：是否复现慢化','',
        '| CPU分配 | 原等待隔离服务 ms | 原等待流水服务 ms | 原等待流水 FPS |',
        '|---|---:|---:|---:|']
    for case,label in labels.items():
        a=results[('component',case,'original',False)];b=results[('pipeline',case,'original',False)]
        text.append(f"| {label} | {a['stage_service_ms'][0]:.3f} | {b['stage_service_ms'][0]:.3f} | {b['fps']:.4f} |")
    d=evidence['decision'];text+=['',f"四线程慢化差值{d['gap_ms']:.3f} ms，探针扰动{d['probe_perturbation_ms']:.3f} ms；目标等待Yield计数{d['target_wait_yield_calls']}。",
        '条件判断：`'+json.dumps(d['conditions'],ensure_ascii=False)+'`。',
        '第二阶段：'+('已按预设条件执行。' if d['execute_blocking'] else '未触发；没有追加试验。'),'']
    if comparisons:
        text+=['## 正式性能：只比较探针关闭的匹配会话','',
            '| CPU分配 | 原等待FPS | 阻塞等待FPS | FPS变化 | 前段服务：原／阻塞 ms |',
            '|---|---:|---:|---:|---:|']
        for c in comparisons:
            text.append(f"| {labels[c['case']]} | {c['original_FPS']:.4f} | {c['blocking_FPS']:.4f} | {c['FPS_change_percent']:+.2f}% | {c['original_stage0_ms']:.3f}／{c['blocking_stage0_ms']:.3f} |")
        text+=['', '等待CPU核时来自单独开启探针的会话；不将其逐项相加解释正式完成间隔，不用探针FPS作为正式收益。']
    text+=['','## 观测范围与限制','',
        '- 当前内核未启用schedstats和ftrace。线程非运行时间是墙钟减线程CPU时间，不能细分成精确的调度等待或阻塞。',
        '- 函数轨迹覆盖整段原生会话，包含预热、尾部及可能的初始化调用；它不是64个正式完成时间戳的逐帧成本。',
        '- 同步屏障内CPU时间、屏障外CPU时间及调用线程尾部时间分别报告；调用尾部也含返回和观测开销。',
        '- 各线程墙钟等待相互重叠，不可相加成阶段服务或完成间隔。',
        '- 观察模式仍调用原函数；阻塞模式改变等待机制，其适配开销包含在正式性能中。',
        '- 不报告显著性或稳定收益，不回填静态成本，不自动启动其他修复、全域标定或Top-20。','',
        f"共{evidence['native_sessions']}个原生会话、{evidence['DNN_inferences']}次完整DNN推理及{evidence['isolated_stage_calls']}次隔离CPU阶段调用；初始化探针另计。",
        '', '原始会话与费用见RESULTS.json、COSTS.json；线程分解见workers.csv，阶段内并行组见parallel_groups.csv。']
    (target/'README.md').write_text('\n'.join(text)+'\n')
    if comparisons:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import font_manager
        font_manager.fontManager.addfont('/usr/share/fonts/truetype/arphic-gbsn00lp/gbsn00lp.ttf')
        import matplotlib.pyplot as plt
        plt.rcParams['font.family']=['AR PL SungtiL GB','DejaVu Sans']
        fig,ax=plt.subplots(figsize=(7,4.5))
        for j,(name,label) in enumerate([('original_FPS','原等待'),('blocking_FPS','阻塞等待')]):
            bars=ax.bar([i+(j-.5)*.3 for i in range(2)],[c[name] for c in comparisons],width=.28,label=label)
            ax.bar_label(bars,fmt='%.3f',padding=3)
        ax.set_xticks(range(2),[labels[c['case']] for c in comparisons]);ax.set_ylabel('FPS（63个完成间隔）')
        ax.legend();fig.tight_layout()
        for ext in ('svg','png','pdf'):fig.savefig(target/('waiting_comparison.'+ext),dpi=180)
        plt.close(fig)
    return dict(report=str(target/'README.md'),decision=d['execute_blocking'],comparisons=comparisons)
