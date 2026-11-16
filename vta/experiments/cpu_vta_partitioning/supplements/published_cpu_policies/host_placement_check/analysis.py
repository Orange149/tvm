from pathlib import Path
import json,shutil
from cpu_vta_partitioning.common import digest,inventory,now,read,sha256,verify_inventory,write
from scheduler_gap_audit.curves import csv_write
from .common import PARENT,ROOT,REPORT,protocol,sources
from .campaign import full_specs
from .runtime import make_task,summary
from .prepare import validate


def analyze(output,run_dir):
    output,run_dir=Path(output),Path(run_dir);validate(output)
    state=read(run_dir/'state.json')
    if state['status']!='COMPLETE' or state['accepted_sessions']!=5 or state['DNN_inferences']!=314:
        raise ValueError('bounded native study incomplete')
    bundle=output/'capsule';receipts={};evidence={}
    for spec in full_specs():
        task=make_task(bundle,spec);p=run_dir/'accepted'/state['boot_id']/(task['task_id']+'.json');item=read(p)
        if item['task']!=task or item['boot_id']!=state['boot_id']:raise ValueError('native receipt task or boot mismatch')
        raw=Path(item['raw']);verify_inventory(raw,item['files'])
        if (raw/'boot_id').read_text().strip()!=state['boot_id'] or (raw/'task_hash').read_text().strip()!=digest(task):raise ValueError('raw native identity changed')
        if summary(raw,task,bundle)!=item['summary']:raise ValueError('native receipt summary does not replay')
        receipts[spec['runner'],spec['phase']]=item;evidence[str(p)]=sha256(p)
    clean=receipts['phase','clean']['summary'];observed=receipts['phase','observed']['summary'];bridge=receipts['old','clean']['summary']
    reference=receipts['phase','qualify']['summary']['output_fingerprints']
    if any(r['summary']['output_fingerprints']!=reference for r in receipts.values()):raise ValueError('executors did not produce identical full network outputs')
    placement=observed['placement'];roles=placement['roles'];gaps=protocol()['pipeline']['scored']-1
    raw=Path(receipts['phase','observed']['raw'])
    frames=[json.loads(s) for s in (raw/'result.jsonl').read_text().splitlines()]
    formal_ns=round((frames[71]['completion_ms']-frames[8]['completion_ms'])*1e6)
    anchor_ns=placement['completion_end_ns']-placement['completion_start_ns']
    if abs(formal_ns-anchor_ns)>1000:raise ValueError('counter anchors do not match formal completion interval')
    for r in roles:
        r['all_CPU_ms_per_completion']=r['all_CPU_ns']/gaps/1e6
        r['core_ms_per_completion']=[v/gaps/1e6 for v in r['by_core_ns']]
    groups={}
    for label,kinds in [('VTA_host',('vta',)),('control',('control',)),('CPU_computation',('cpu','cpu_worker')),('unclassified',('unclassified',))]:
        selected=[r for r in roles if r['kind'] in kinds]
        core=[sum(r['core_ms_per_completion'][c] for r in selected) for c in range(4)]
        groups[label]=dict(by_core_ms_per_completion=core,all_CPU_ms_per_completion=sum(r['all_CPU_ms_per_completion'] for r in selected))
    semantic=ROOT/'artifacts/model_semantics_audit_01';old=read(semantic/'DELIVERY.json')
    model_path=next(Path(p) for p in old['files'] if Path(p).name=='HOST_ENDPOINT_EVIDENCE.json')
    if sha256(model_path)!=old['files'][str(model_path)]:raise ValueError('source model distribution changed')
    model=next(r['model_host_CPU_ms_per_frame_by_core'] for r in read(model_path)['model'] if r['case']=='yolov3_tiny416/overlap' and r['control_mode']=='zero')
    parent=read(PARENT/'DELIVERY.json');old_results=PARENT/'analysis/RESULTS.json'
    if sha256(old_results)!=parent['results_sha256']:raise ValueError('historical diagnostic baseline changed')
    prior=next(r for r in read(old_results)['stages'] if r['network']=='yolov3_tiny416' and r['case']=='overlap' and r['stage']=='cpu_00_02')
    disturbance=abs(clean['stage_service_ms'][0]-observed['stage_service_ms'][0]);gap=clean['stage_service_ms'][0]-prior['isolated_service_ms']
    valid=gap>0 and disturbance<=.2*gap
    h=groups['VTA_host']['by_core_ms_per_completion']
    result=dict(status='COMPLETE',boot_id=state['boot_id'],clean=clean,observed=observed,bridge=bridge,
        roles=roles,groups=groups,model_host_CPU_by_core_ms_per_frame=model,
        actual_host_core0_fraction=h[0]/sum(h),model_host_core0_fraction=model['0']/sum(model.values()),
        probe_front_service_disturbance_ms=disturbance,probe_FPS_change_percent=100*(observed['fps']/clean['fps']-1),
        reference_isolation_source='previous same-boot accepted component measurement, not remeasured',
        reference_isolated_front_ms=prior['isolated_service_ms'],clean_to_reference_isolation_gap_ms=gap,
        quantitative_attribution_allowed=valid,probe_quality_does_not_reject_performance=True,
        bridge_clean_FPS_change_percent=100*(clean['fps']/bridge['fps']-1),
        window=placement,formal_completion_window_ns=formal_ns,anchor_window_ns=anchor_ns,
        CPU_distribution_not_a_migration_trace=True,CPU_distribution_not_added_to_static_scores=True,
        new_native_sessions=5,DNN_inferences=314,new_component_calibrations=0,source_receipts=evidence,
        model_source_sha256=sha256(model_path),parent_results_sha256=sha256(old_results),sources=sources(),
        state=state,recorded_at=now())
    dest=output/'analysis';dest.mkdir(parents=True,exist_ok=True)
    write(dest/'RESULTS.json',result)
    csv_write(dest/'roles.csv',[{k:v for k,v in r.items() if k not in ('before_read_ns','after_read_ns')} for r in roles])
    render(dest,result)
    REPORT.mkdir(parents=True,exist_ok=True)
    for p in dest.iterdir():shutil.copy2(p,REPORT/p.name)
    write(output/'DELIVERY.json',dict(status='COMPLETE',new_native_sessions=5,DNN_inferences=314,
        results_sha256=sha256(dest/'RESULTS.json'),sources=sources(),files={str(p):sha256(p) for p in REPORT.iterdir() if p.is_file()},
        no_reboot=True,no_score_change=True,no_DP_change=True,no_Top20_change=True,recorded_at=now()))
    return dict(report=str(REPORT/'README.md'),actual_host_core0_percent=100*result['actual_host_core0_fraction'],
        model_host_core0_percent=100*result['model_host_core0_fraction'],quantitative_attribution_allowed=valid,
        native_sessions=5,DNN_inferences=314)


def render(dest,r):
    lines=['# YOLO重叠配置：真实主机与控制工作落核','',
        '固定CPU0–2（四线程，核心0–3）→VTA3–6→CPU7–8（三线程，核心1–3）→VTA9–12→CPU13（单线程，核心0）。主机与控制线程共享四核，CPU门控关闭。',
        '', '本轮只改变可选计数观测。模块、运行库、切图、线程配置、原等待机制及输出统计口径均复用。5个会话、314次完整推理；不重启、不更新公式、DP或Top-20。','',
        '## 同批性能及探针扰动','',
        '| 执行方式 | FPS | 平均完成间隔ms | CPU0–2服务ms |',
        '|---|---:|---:|---:|']
    for name,key in [('原冻结执行器桥接','bridge'),('新诊断执行器，计数关闭','clean'),('同一新执行器，计数开启','observed')]:
        v=r[key];lines.append(f"| {name} | {v['fps']:.4f} | {v['ii_ms']:.4f} | {v['stage_service_ms'][0]:.4f} |")
    lines+=['',f"计数开启相对关闭：FPS变化{r['probe_FPS_change_percent']:+.3f}%，前段服务差值{r['probe_front_service_disturbance_ms']:.3f}ms。探针质量条件：{r['quantitative_attribution_allowed']}。参照隔离值来自此前同boot记录，未在本批重测。",
        f"新程序关闭探针相对原程序的桥接FPS差异{r['bridge_clean_FPS_change_percent']:+.3f}%，属于单会话观察，不宣布等效或稳定收益。",
        '', '## 逐核心CPU分布','',
        '下表是各TID计数窗口内的原始CPU增量除以63个完成间隔，单位为核毫秒／完成间隔；包含读取包络，不按enabled/running进行放大。主机行含该TID在窗口内的全部CPU工作，不将设备等待墙钟填入CPU时间。','',
        '| 角色 | 核心0 | 核心1 | 核心2 | 核心3 | 全核心总计 |',
        '|---|---:|---:|---:|---:|---:|']
    for v in r['roles']:
        if v['kind']=='unclassified' and v['all_CPU_ns']==0:continue
        c=v['core_ms_per_completion'];lines.append(f"| {v['role']} | {c[0]:.4f} | {c[1]:.4f} | {c[2]:.4f} | {c[3]:.4f} | {v['all_CPU_ms_per_completion']:.4f} |")
    h=r['groups']['VTA_host'];m=r['model_host_CPU_by_core_ms_per_frame']
    lines+=['',f"两个真实VTA主机在核心0上的CPU份额为{100*r['actual_host_core0_fraction']:.3f}%；原理想回放为{100*r['model_host_core0_fraction']:.3f}%。原回放各核主机CPU为{[round(m[str(c)],4) for c in range(4)]}核ms／模型帧；本批计数为{[round(v,4) for v in h['by_core_ms_per_completion']]}核ms／完成间隔。口径不同，份额可用于检查放置假设，不直接相减当作阶段误差。",
        '', '## 窗口与解释边界','',
        '计数沿用执行器原有窗口触发点：收集线程取出完成编号8时开始、取出71时停止。正式FPS仍取末阶段完成时间戳。计数记录保留了这两个末阶段完成时刻与实际读数之间的偏移，不把收集窗口冒充严格同步的末阶段窗口。',
        f"起点读数包络相对末阶段完成的偏移为{[round(v/1e6,4) for v in r['window']['start_after_completion_ns']]}ms；终点为{[round(v/1e6,4) for v in r['window']['stop_after_completion_ns']]}ms。启动设置及首读共{r['window']['setup_and_start_read_ns']/1e6:.4f}ms，终读{r['window']['stop_read_ns']/1e6:.4f}ms。",
        '本数据支持核对该配置实际CPU需求的落核分布；不重建逐次唤醒或迁移轨迹，也不测量其他进程与中断负载。未分类辅助线程单列，不能自动归到某个主机或CPU阶段。',
        '若探针质量条件未通过，角色分布只作为带扰动线索；不自动补测。即使落核差异明确，也不声称它独自解释全部约37ms阶段残差，更不能据此宣称已改善吞吐。',
        '后续若修改回放，应提出可由隔离输入或通用执行规则确定的放置模型，再用未参与修正的配置验证；本轮不把实测落核比例或慢化系数回填为通用成本。','',
        '接口依据：[Linux perf_event_open](https://man7.org/linux/man-pages/man2/perf_event_open.2.html)。逐TID计数、输出、实际绑核、帧完整性、boot及原始文件哈希均经核验。']
    (dest/'README.md').write_text('\n'.join(lines)+'\n')
