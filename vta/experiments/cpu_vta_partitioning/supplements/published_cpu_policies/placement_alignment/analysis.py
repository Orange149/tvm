"""Match identical kernels and logical slots, not whole-stage wall averages."""
from collections import Counter,defaultdict
from pathlib import Path
from statistics import mean
import json
import shutil
from cpu_vta_partitioning.common import digest,read,sha256,verify_inventory,write
from scheduler_gap_audit.analysis import queue_bounds
from scheduler_gap_audit.curves import csv_write
from model_semantics_audit.common import offline_guard
from .common import PARENT,SEMANTICS,REPORT,sources


def overlap_bounds(start,end,other_start,other_end,other_CPU):
    """Single-window temporal CPU bounds; neither per-core nor additive."""
    if end<start or other_end<other_start or not 0<=other_CPU<=other_end-other_start:
        raise ValueError('invalid overlap/CPU envelope')
    wall=max(0,min(end,other_end)-max(start,other_start))
    return wall,max(0,other_CPU-((other_end-other_start)-wall)),min(other_CPU,wall)


def collect(item):
    task=item['task'];raw=Path(item['raw']);verify_inventory(raw,item['files'])
    if (raw/'boot_id').read_text().strip()!=item['boot_id'] or (raw/'task_hash').read_text().strip()!=digest(task):
        raise ValueError('raw boot/task identity mismatch')
    phases=[json.loads(s) for s in (raw/'phase_trace.jsonl').read_text().splitlines()]
    trace=[json.loads(s) for s in (raw/'wait_trace.jsonl').read_text().splitlines()]
    callbacks=defaultdict(list)
    for r in trace:
        if r['kind']==1:callbacks[r['launch']].append(r)
    launches=[r for r in trace if r['kind']==0];kernels=[]
    for phase in phases:
        if phase['stage']!='cpu_00_02' or phase['operation']!='run' or not task['warmup']<=phase['call']<task['warmup']+task['scored']:continue
        selected=sorted([r for r in launches if r['caller']==phase['tid'] and phase['wall_start_ns']<=r['wall_start_ns']<=r['wall_end_ns']<=phase['wall_end_ns']],key=lambda r:r['wall_start_ns'])
        for ordinal,launch in enumerate(selected):
            workers=sorted(callbacks[launch['launch']],key=lambda r:r['task'])
            if [w['task'] for w in workers]!=list(range(launch['tasks'])):raise ValueError('worker coverage incorrect')
            if workers[0]['tid']!=phase['tid']:raise ValueError('logical slot zero is not caller')
            if len({w['tid'] for w in workers})!=len(workers):raise ValueError('logical workers share a TID')
            for w in workers:
                if w['caller']!=phase['tid'] or not launch['wall_start_ns']<=w['wall_start_ns']<=w['wall_end_ns']<=launch['wall_end_ns']:
                    raise ValueError('callback escaped its launch/caller')
                if not 0<=w['barrier_cpu_ns']<=w['cpu_end_ns']-w['cpu_start_ns']:
                    raise ValueError('negative callback body CPU')
            last=max(workers,key=lambda w:(w['wall_end_ns'],w['task']))
            exposure=[]
            for host in phases:
                if not host['stage'].startswith('vta_') or host['operation']!='run':continue
                cpu=host['cpu_end_ns']-host['cpu_start_ns']
                wall,lower,upper=overlap_bounds(launch['wall_start_ns'],launch['wall_end_ns'],host['wall_start_ns'],host['wall_end_ns'],cpu)
                if wall:exposure.append(dict(stage=host['stage'],frame=host['call'],wall_overlap_ns=wall,
                    host_CPU_lower_ns=lower,host_CPU_upper_ns=upper,start_core=host['core_start'],end_core=host['core_end'],
                    not_actual_per_core_CPU=True,not_additive_across_kernel_windows=True))
            kernels.append(dict(frame=phase['call'],kernel=ordinal,module=launch['module'].split('/programs/',1)[-1],
                function_offset=launch['function_offset'],tasks=launch['tasks'],
                launch_wall_ns=launch['wall_end_ns']-launch['wall_start_ns'],workers=workers,last_slot=last['task'],
                launch_start_skew_ns=max(w['wall_start_ns'] for w in workers)-min(w['wall_start_ns'] for w in workers),
                exposure=exposure))
    signatures=defaultdict(set)
    for k in kernels:signatures[k['frame']].add((k['kernel'],k['module'],k['function_offset'],k['tasks']))
    if len(signatures)!=task['scored'] or any(s!=next(iter(signatures.values())) for s in signatures.values()):
        raise ValueError('per-frame kernel signature sequence incomplete')
    return kernels,phases


def align_kernel(isolated,pipeline):
    sig=lambda r:(r['module'],r['function_offset'],r['tasks'])
    if len({sig(r) for r in isolated+pipeline})!=1:raise ValueError('kernel/module/thread count mismatch')
    slots=[]
    for slot in range(isolated[0]['tasks']):
        def values(rows):return [next(w for w in k['workers'] if w['task']==slot) for k in rows]
        a,b=values(isolated),values(pipeline)
        def cpu(v):return v['cpu_end_ns']-v['cpu_start_ns']-v['barrier_cpu_ns']
        slots.append(dict(slot=slot,isolated_CPU_ms=mean(cpu(v) for v in a)/1e6,pipeline_CPU_ms=mean(cpu(v) for v in b)/1e6,
            isolated_body_queue_lower_ms=mean(queue_bounds(v)[0] for v in a)/1e6,
            pipeline_body_queue_lower_ms=mean(queue_bounds(v)[0] for v in b)/1e6,
            isolated_last_count=sum(k['last_slot']==slot for k in isolated),pipeline_last_count=sum(k['last_slot']==slot for k in pipeline)))
    aw=mean(r['launch_wall_ns'] for r in isolated)/1e6;bw=mean(r['launch_wall_ns'] for r in pipeline)/1e6
    def last_queue(rows,index):
        return mean(queue_bounds(next(w for w in r['workers'] if w['task']==r['last_slot']))[index] for r in rows)/1e6
    return dict(kernel=isolated[0]['kernel'],module=isolated[0]['module'],function_offset=isolated[0]['function_offset'],tasks=isolated[0]['tasks'],
        isolated_launch_ms=aw,pipeline_launch_ms=bw,launch_wall_growth_ms=bw-aw,
        isolated_callback_CPU_ms=sum(s['isolated_CPU_ms'] for s in slots),pipeline_callback_CPU_ms=sum(s['pipeline_CPU_ms'] for s in slots),
        pipeline_start_skew_ms=mean(r['launch_start_skew_ns'] for r in pipeline)/1e6,
        isolated_last_queue_lower_ms=last_queue(isolated,0),isolated_last_queue_upper_ms=last_queue(isolated,1),
        pipeline_last_queue_lower_ms=last_queue(pipeline,0),pipeline_last_queue_upper_ms=last_queue(pipeline,1),
        isolated_last_counts=dict(Counter(k['last_slot'] for k in isolated)),pipeline_last_counts=dict(Counter(k['last_slot'] for k in pipeline)),
        slots=slots,matched_exact_function_and_logical_slot=True)


def analyze(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with offline_guard() as ledger:
        delivery=read(PARENT/'DELIVERY.json');source=PARENT/'analysis/RESULTS.json'
        if sha256(source)!=delivery['results_sha256']:raise ValueError('parent measurement results changed')
        semantic_delivery=read(SEMANTICS/'DELIVERY.json')
        endpoint_source=next(Path(p) for p in semantic_delivery['files'] if Path(p).name=='HOST_ENDPOINT_EVIDENCE.json')
        if sha256(endpoint_source)!=semantic_delivery['files'][str(endpoint_source)]:raise ValueError('model host evidence changed')
        model_hosts=read(endpoint_source)['model']
        data=read(source);observed={};evidence={}
        for name,h in data['receipts'].items():
            p=Path(name)
            if sha256(p)!=h:raise ValueError('receipt source changed')
            item=read(p);s=item['task']['spec']
            if s['phase']!='observed':continue
            observed[s['network'],s['scope'],s['case']]=item;evidence[str(p)]=h
        kernels=[];slot_rows=[];exposures=[];endpoints=[]
        for case in data['cases']:
            network=case['network'];name=case['case'];n=case['configuration']['stages'][0]['threads']
            bundle=PARENT/network/'capsule';m=read(bundle/'manifest.json')
            if m['bundle_id']!=delivery['bundles'][network] or digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]!=m['bundle_id']:
                raise ValueError('source module capsule seal changed')
            module='programs/cpu_00_02/module.so'
            if sha256(bundle/module)!=m['files'][module]['sha256']:raise ValueError('CPU module changed')
            component=observed[network,'component','front_t'+str(n)];pipeline=observed[network,'pipeline',name]
            a,_=collect(component);b,phases=collect(pipeline)
            if component['task']['runtime_sha256']!=pipeline['task']['runtime_sha256'] or component['task']['executor_sha256']!=pipeline['task']['executor_sha256']:
                raise ValueError('isolated and pipeline executable identity mismatch')
            if component['task']['stages'][0]!=pipeline['task']['stages'][0] or component['task']['adapter_sha256']!=pipeline['task']['adapter_sha256']:
                raise ValueError('CPU mapping or observer identity differs')
            for ordinal in sorted({r['kernel'] for r in a}):
                paired=align_kernel([r for r in a if r['kernel']==ordinal],[r for r in b if r['kernel']==ordinal])
                paired.update(case=network+'/'+name,probe_quality=case['quantitative_attribution_allowed'])
                kernels.append(paired)
                slot_rows += [dict(case=paired['case'],kernel=ordinal,function_offset=paired['function_offset'],probe_quality=paired['probe_quality'],**s) for s in paired['slots']]
            for k in b:
                exposures += [dict(case=network+'/'+name,front_frame=k['frame'],kernel=k['kernel'],last_slot=k['last_slot'],**e) for e in k['exposure']]
            for stage in (s for s in case['configuration']['stages'] if s['device']=='vta'):
                spans=[r for r in phases if r['stage']==stage['program'] and r['operation']=='run' and 8<=r['call']<72]
                endpoints.append(dict(case=network+'/'+name,stage=stage['program'],spans=len(spans),
                    starts=dict(Counter(r['core_start'] for r in spans)),ends=dict(Counter(r['core_end'] for r in spans)),
                    endpoint_core0=sum(r['core_start']==0 for r in spans)+sum(r['core_end']==0 for r in spans),
                    endpoints_are_not_CPU_residency=True))
        result=dict(kernels=kernels,slot_comparison=slot_rows,host_endpoints=endpoints,source_receipts=evidence,
            parent_results_sha256=sha256(source),sources=sources(),new_board_sessions=0,new_inferences=0,
            semantic_delivery_sha256=sha256(SEMANTICS/'DELIVERY.json'),model_host_source_sha256=sha256(endpoint_source),
            model_host_CPU=model_hosts,no_new_prediction=True)
        write(output/'ALIGNMENT.json',result,replace=True);csv_write(output/'per_kernel.csv',[{k:v for k,v in r.items() if k!='slots'} for r in kernels])
        csv_write(output/'per_slot.csv',slot_rows);csv_write(output/'host_overlap_envelopes.csv',exposures)
        render(output,result)
        write(output/'OFFLINE_GUARD.json',ledger,replace=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    for leaf in ('ALIGNMENT.json','per_kernel.csv','per_slot.csv','host_overlap_envelopes.csv','README.md','OFFLINE_GUARD.json','kernel_slowdown.png'):
        shutil.copy2(output/leaf,REPORT/leaf)
    return dict(report=str(REPORT/'README.md'),matched_kernel_signatures=len(kernels),new_inferences=0,new_board_sessions=0)


def render(output,result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[r for r in result['kernels'] if r['case']=='yolov3_tiny416/overlap']
    ordered=sorted(chosen,key=lambda r:(-r['launch_wall_growth_ms'],r['kernel']))
    total=sum(r['launch_wall_growth_ms'] for r in chosen)
    lines=['# 主机端点与worker逐内核对齐','',
        '本轮只复用已有记录，未连接开发板、未新增推理、未修改模型或排名。核对同一模块、同一函数偏移、相同线程数及逻辑槽位，避免用不同kernel平均值解释完成顺序。','',
        '这里分析的是隔离到流水的已有慢化，不是新性能优化。上一轮回放补入已观测控制工作后，YOLO前段预测约184.214ms，正式流水服务约221.423ms，仍差约37.209ms；本轮不声称已消除该误差。下面并行启动的75.704ms增长属于有探针的隔离／流水对照，不能视为这37.209ms的可加分解。','',
        '## YOLO四线程重叠：全部12个并行内核','',
        '| kernel序号 | 隔离启动ms | 流水启动ms | 增长ms | 隔离回调CPU核ms | 流水回调CPU核ms | 流水最后完成者回调内排队ms | 启动错位ms |',
        '|---|---:|---:|---:|---:|---:|---|---:|']
    for r in chosen:lines.append(f"| {r['kernel']} | {r['isolated_launch_ms']:.3f} | {r['pipeline_launch_ms']:.3f} | {r['launch_wall_growth_ms']:+.3f} | {r['isolated_callback_CPU_ms']:.3f} | {r['pipeline_callback_CPU_ms']:.3f} | {r['pipeline_last_queue_lower_ms']:.3f}–{r['pipeline_last_queue_upper_ms']:.3f} | {r['pipeline_start_skew_ms']:.3f} |")
    top=ordered[:3];growth=sum(r['launch_wall_growth_ms'] for r in top)
    cpu_growth=100*(sum(r['pipeline_callback_CPU_ms'] for r in top)/sum(r['isolated_callback_CPU_ms'] for r in top)-1)
    host=next(r['model_host_CPU_ms_per_frame_by_core'] for r in result['model_host_CPU'] if r['case']=='yolov3_tiny416/overlap' and r['control_mode']=='zero')
    host_total=sum(host.values());host0=host.get('0',0)
    endpoints=[r for r in result['host_endpoints'] if r['case']=='yolov3_tiny416/overlap']
    count=sum(2*r['spans'] for r in endpoints);core0=sum(r['endpoint_core0'] for r in endpoints)
    lines+=['',f'12次顺序启动的平均墙钟增长累计{total:.3f}ms。它与阶段总差值范围不同，还包含启动/返回与观测包络；不能与worker排队或调用线程尾部相加。',
        '', '按墙钟增长排序的前三项为：'+', '.join(str(r['kernel']) for r in top)+f'，占12项增长的{100*growth/total:.2f}%；对应回调计算CPU量仅增加{cpu_growth:.2f}%。这是事后诊断展示，全部项均交付，不据此重新定义计算单元或搜索候选。','',
        '## 主机位置证据的边界','',
        f'YOLO重叠方案两个VTA主机共{count}个run首尾端点，其中{core0}个采到核心0；原理想回放将约{host0:.2f}/{host_total:.2f}核ms、约{100*host0/host_total:.1f}%的主机CPU分配到核心0。这是值得核实的位置假设差异；端点出现频率不是CPU占用率，仍不能证明主机从不在核心0运行。',
        '', 'host_overlap_envelopes.csv只给每个CPU启动窗口与VTA运行窗口的交集，以及该窗口可能容纳的主机CPU上下量。它们不是逐核心CPU，独立窗口上界不能累加；VTA墙钟内还包含设备等待。',
        '', '## 下一项最小缺口','',
        '需要验证能否直接读取同一主机TID在各核心上的CPU时间，并验证各核心计数之和与线程CPU时钟守恒。CPU排队已能观测，尚不能定位所有竞争者的CPU去向。接口验证应先用迁移忙任务和睡眠任务，不运行DNN，也不改系统等待或线程配置。',
        '准备好的计数程序仅代表接口可行性准备；未在开发板运行时必须标为未验证。旧ResNet及YOLO分组的探针质量不足，细分对比仅为辅助线索。']
    (Path(output)/'README.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    x=[r['kernel'] for r in chosen]
    axes[0].bar(x,[r['launch_wall_growth_ms'] for r in chosen]);axes[0].set(xlabel='Kernel ordinal',ylabel='Pipeline minus isolation launch ms',title='YOLO front: existing trace diagnosis')
    axes[1].plot(x,[r['isolated_callback_CPU_ms'] for r in chosen],'o-',label='Isolated callback CPU')
    axes[1].plot(x,[r['pipeline_callback_CPU_ms'] for r in chosen],'s--',label='Pipeline callback CPU')
    axes[1].set(xlabel='Kernel ordinal',ylabel='Core ms');axes[1].legend(fontsize=8)
    fig.savefig(Path(output)/'kernel_slowdown.png',dpi=170);plt.close(fig)
