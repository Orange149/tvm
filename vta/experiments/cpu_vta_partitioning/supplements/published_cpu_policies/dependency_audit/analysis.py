"""Diagnostic comparisons are downstream of the isolated replay seal."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import csv
import json

from cpu_vta_partitioning.common import read, sha256, verify_inventory, write
from .common import require_stage
from .evidence import diagnostic_sources
from .trace import load_trace

LABELS = dict(overlap4='四线程', split3='三线程')


def mean(values):
    return Fraction(sum(values), len(values))


def fraction(value):
    return dict(numerator=value.numerator, denominator=value.denominator, unit='ns')


def milliseconds(value):
    if isinstance(value, dict):
        value = Fraction(value['numerator'], value['denominator'])
    return float(value / 1000000)


def csv_table(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v,(list,dict)) else v
                         for k,v in row.items()} for row in rows)


def metrics(trace, first, count):
    frames = trace['frames'][first:first+count]
    if len(frames) != count or not trace['frame_mapping']:
        raise ValueError('requested frame metrics lack a reliable frame mapping')
    kernels = []
    for k in range(len(trace['sequence'])):
        rows = [f['launches'][k] for f in frames]
        names = ('largest_task_cpu_ns','launch_wall_ns','body_span_ns','start_skew_ns',
                 'after_last_callback_ns','caller_tail_cpu_ns','caller_tail_wall_ns')
        values = {name: fraction(mean([r[name] for r in rows])) for name in names}
        kernels.append(dict(kernel_index=k,function_offset=trace['sequence'][k],means=values,
            worker_CPU_means_ns=[fraction(mean([r['worker_cpu_ns'][j] for r in rows])) for j in range(trace['threads'])],
            last_workers=dict(Counter(r['last_worker'] for r in rows))))
    names = ('largest_task_cpu_ns','launch_wall_ns','body_span_ns','start_skew_ns',
             'after_last_callback_ns','caller_tail_cpu_ns','caller_tail_wall_ns')
    totals = {name: fraction(mean([sum(r[name] for r in f['launches']) for f in frames])) for name in names}
    totals['total_callback_CPU_ns'] = fraction(mean([sum(sum(r['worker_cpu_ns']) for r in f['launches']) for f in frames]))
    return dict(first_frame=first, frame_count=count,kernels=kernels,totals=totals,
        native_service_ns=fraction(mean([f['native_service_ns'] for f in frames])),
        native_run_wall_ns=fraction(mean([f['native_run_wall_ns'] for f in frames])))


def analyze(output):
    output = Path(output)
    require_stage(output, 'replay')
    audit, extraction, replay = [read(output/name) for name in ('AUDIT.json','EXTRACTION.json','REPLAY.json')]
    references = diagnostic_sources()
    diagnostic = {}
    for case in ('overlap4','split3'):
        source = references['cases'][case+'_observed']
        diagnostic[case] = load_trace(source['raw'],audit['module_sha256'])
    target = output/'analysis'
    target.mkdir(parents=True, exist_ok=True)
    comparisons, kernels, frame_rows, accounting, coverage = [], [], [], [], []
    for case in ('overlap4','split3'):
        isolated, pipeline = extraction['cases'][case], diagnostic[case]
        coverage.append(dict(case=case,program='cpu_00_02',threads=isolated['threads'],
            isolation_frame_mapping=isolated['frame_mapping'],pipeline_frame_mapping=pipeline['frame_mapping'],
            isolation_launches=len(isolated['launches']),pipeline_launches=len(pipeline['launches']),
            kernel_sequence=isolated['sequence'],exact_cross_clock_anchor=False,
            isolated_non_target_launches=isolated['non_target_launches_by_module'],
            pipeline_non_target_launches=pipeline['non_target_launches_by_module']))
        if not isolated['frame_mapping'] or not pipeline['frame_mapping']:
            continue
        if isolated['sequence'] != pipeline['sequence']:
            raise ValueError('pipeline and isolated modules have different kernel sequences')
        iso, pipe = metrics(isolated,2,8),metrics(pipeline,8,64)
        replay_case = replay['cases'][case]
        scored = replay_case['phases']['scored']['means']
        clean_iso = audit['cases'][case+'_clean']['cost']['service_ms']
        clean_pipe = references['cases'][case+'_clean']['summary']['stage_service_ms'][0]
        observed_iso, observed_pipe = milliseconds(iso['native_service_ns']),milliseconds(pipe['native_service_ns'])
        comparisons.append(dict(case=case,threads=isolated['threads'],
            aggregate_ms=milliseconds(scored['aggregate_ns']),join_ms=milliseconds(scored['join_ns']),
            additional_dependency_ms=milliseconds(scored['additional_dependency_ns']),
            observed_interlaunch_caller_CPU_ms=milliseconds(scored['observed_gap_CPU_ns']),
            callback_plus_observed_gap_ms=milliseconds(scored['callback_plus_observed_gap_ns']),
            clean_isolated_service_ms=clean_iso,clean_pipeline_service_ms=clean_pipe,
            clean_slowdown_gap_ms=clean_pipe-clean_iso,
            observed_isolated_service_ms=observed_iso,observed_pipeline_service_ms=observed_pipe,
            probe_component_difference_ms=observed_iso-clean_iso,probe_pipeline_difference_ms=observed_pipe-clean_pipe,
            observed_isolated_callback_CPU_ms=milliseconds(iso['totals']['total_callback_CPU_ns']),
            observed_pipeline_callback_CPU_ms=milliseconds(pipe['totals']['total_callback_CPU_ns']),
            isolated_body_span_ms=milliseconds(iso['totals']['body_span_ns']),
            pipeline_body_span_ms=milliseconds(pipe['totals']['body_span_ns']),
            isolated_last_worker_CPU_ms=milliseconds(iso['totals']['largest_task_cpu_ns']),
            pipeline_last_worker_CPU_ms=milliseconds(pipe['totals']['largest_task_cpu_ns']),
            isolated_start_skew_ms=milliseconds(iso['totals']['start_skew_ns']),
            pipeline_start_skew_ms=milliseconds(pipe['totals']['start_skew_ns'])))
        for k,(a,b) in enumerate(zip(iso['kernels'],pipe['kernels'])):
            row=dict(case=case,kernel=k,function_offset=a['function_offset'],
                     module_sha256=audit['module_sha256'],isolated_last_workers=a['last_workers'],pipeline_last_workers=b['last_workers'])
            for prefix,value in (('isolated',a),('pipeline',b)):
                row.update({prefix+'_'+key.replace('_ns','_ms'):milliseconds(v) for key,v in value['means'].items()})
                row[prefix+'_worker_CPU_ms']=[milliseconds(v) for v in value['worker_CPU_means_ns']]
            kernels.append(row)
        for frame in replay_case['frames']:
            observed = isolated['frames'][frame['frame_id']]
            names = ('aggregate_ns','join_ns','additional_dependency_ns','observed_gap_CPU_ns','callback_plus_observed_gap_ns')
            frame_rows.append(dict(case=case,frame_id=frame['frame_id'],phase=frame['phase'],
                **{k.replace('_ns','_ms'):frame[k]/1e6 for k in names},
                observed_run_wall_ms=observed['native_run_wall_ns']/1e6))
            accounting.append(dict(case=case,frame_id=frame['frame_id'],phase=frame['phase'],
                run_process_CPU_ns=observed['native_run_cpu_ns'],accounted_launches_and_gaps_CPU_ns=observed['accounted_run_cpu_ns'],
                unlocated_CPU_residual_ns=observed['unlocated_CPU_residual_ns'],
                residual_not_assigned_to_prefix_or_suffix=True))
    missing = [
        dict(item='其他CPU阶段的并行结构',status='缺少同粒度隔离轨迹',existing='714项成本具有阶段汇总，未提供每次并行调用的依赖及worker工作量',
             why='CPU7–8和CPU13的可运行任务会与前段交错',minimum_future_data='只检查当前切图的CPU7–8一／三线程和CPU13单线程，不做全域采集'),
        dict(item='阶段首尾及串行工作的位置',status='仅有汇总与部分间隙',existing='可测相邻启动间调用线程CPU；run总核时可计算未定位差额',
             why='相同总工作量位于不同位置会改变竞争时序',minimum_future_data='未来需要同一时钟下的阶段run边界及串行区段标记，不能任意均分剩余量'),
        dict(item='VTA主机工作的可运行时序',status='只有汇总需求',existing='隔离set/run/get墙钟与CPU核时，未分解提交、轮询、阻塞及唤醒',
             why='CPU需求总量不能确定它何时与计算线程竞争',minimum_future_data='先明确可观测的主机事件与CPU时间口径；不能把整个设备等待墙钟当CPU工作'),
        dict(item='可运行状态与CPU调度规则',status='无法从现有轨迹恢复',existing='回调开始／结束、线程CPU、固定affinity；无schedstats和ftrace',
             why='非运行时间不等于调度等待，未知时间片或放置不能靠FPS反推',minimum_future_data='另行选择可验证的受控调度假设或补齐运行状态观测；本轮不重刷内核'),
        dict(item='生产与结果收集开销',status='未独立标定',existing='整体进程CPU包含这些工作，但不能逐项归属',
             why='不能在完整模型中将控制工作默认设为零',minimum_future_data='后续仅对实际控制路径定义独立测量，不用并发阶段的进程CPU相加'),
        dict(item='缓存、访存与动态频率影响',status='未完全区分',existing='回调CPU工作较接近、频率设置固定，不构成完全排除证据',
             why='隔离计算成本是否可直接迁移仍须检验',minimum_future_data='需要能区分活跃计算变化与非运行变化的匹配证据，不按统一慢化系数回填'),
        dict(item='核组内迁移的详细轨迹',status='本原型不支持',existing='最新批次有允许集合与正式性能，没有调度迁移轨迹',
             why='不能把迁移线程仍按确定物理核处理',minimum_future_data='未来另行定义迁移模式的资源语义；不缩小原完整搜索域')]
    results = dict(comparisons=comparisons,coverage=coverage,missing_inputs=missing,
        isolated_model_replay_sha256=sha256(output/'REPLAY.json'),
        replay_freeze_sha256=sha256(output/'REPLAY_FREEZE.json'),
        diagnostic_receipts=references['receipts'],affinity_background=references['affinity_observations'],
        full_shared_pipeline_prediction_supported=False,full_pipeline_FPS=None,
        conclusions=dict(single_stage_reconstructed_cases=len(replay['cases']),
            aggregation_delta_smaller_than_observed_gap={r['case']:r['additional_dependency_ms']<r['clean_slowdown_gap_ms'] for r in comparisons},
            only_join_aggregation_is_not_a_supported_fix=True,
            no_unseen_configuration_validation=True,no_DNN_performance_improvement_claim=True,
            no_followup_board_tasks=True,unknown_data_not_zero=True))
    write(target/'RESULTS.json',results,replace=True)
    for name,rows in (('comparisons',comparisons),('kernels',kernels),('frames',frame_rows),
                      ('CPU_accounting',accounting),('coverage',coverage),('missing_inputs',missing)):
        csv_table(target/(name+'.csv'),rows)
    report(target,comparisons,missing,coverage,output)
    from .plots import plot
    plot(target,extraction,diagnostic,replay,comparisons)
    return dict(status='OFFLINE_AUDIT_COMPLETE',report=str(target/'README.md'),
        single_stage_replay_cases=len(replay['cases']),full_shared_pipeline_prediction_supported=False,
        new_native_sessions=0,new_DNN_inferences=0)


def report(target,comparisons,missing,coverage,output):
    mapped=sum(c['isolation_frame_mapping'] and c['pipeline_frame_mapping'] for c in coverage)
    lines=['# YOLO并行完成依赖：离线核查结果','',
        f'**本轮只做已有记录的离线核验和单阶段工作回放，没有连接开发板、执行推理或改变静态模型。{mapped}/2个配置可进行隔离／流水逐帧对照；解析／回放成功不等于共享流水预测已经修好。**','',
        '## 两种核时汇总到底差多少','',
        '下表使用原等待、隔离探针会话的计入帧2–9，共8帧。全部先按整数纳秒计算，平均值保留有理数；只在表格中换算毫秒。','',
        '| CPU前段 | 整段取最大 ms | 逐次完成依赖 ms | 差值 ms | 已观测间隙CPU ms | 计算＋间隙回放 ms | 无探针流水慢化差值 ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for c in comparisons:
        lines.append(f"| {LABELS[c['case']]} | {c['aggregate_ms']:.6f} | {c['join_ms']:.6f} | {c['additional_dependency_ms']:.6f} | {c['observed_interlaunch_caller_CPU_ms']:.6f} | {c['callback_plus_observed_gap_ms']:.6f} | {c['clean_slowdown_gap_ms']:.3f} |")
    lines += ['','整段取最大是每worker核时先累计，再取最大；逐次完成依赖是每次并行启动取最大，再累加。它们仅表示计算工作，不等于阶段完整服务时间或流水II。已观测间隙也只有相邻启动之间、同一调用线程的CPU增量，不含首尾未知工作和完整启动开销。','',
        '**现有隔离阶段墙钟已经包含这些并行完成依赖。因此不能把本表再加到原评分里当作新惩罚项。**','',
        '## 隔离与流水：计算工作和墙钟跨度分别看','',
        '| CPU前段 | 回调CPU：隔离／流水 ms | 回调覆盖跨度：隔离／流水 ms | 最慢任务CPU累计：隔离／流水 ms | 启动错位累计：隔离／流水 ms |',
        '|---|---:|---:|---:|---:|']
    for c in comparisons:
        lines.append(f"| {LABELS[c['case']]} | {c['observed_isolated_callback_CPU_ms']:.3f}／{c['observed_pipeline_callback_CPU_ms']:.3f} | {c['isolated_body_span_ms']:.3f}／{c['pipeline_body_span_ms']:.3f} | {c['isolated_last_worker_CPU_ms']:.3f}／{c['pipeline_last_worker_CPU_ms']:.3f} | {c['isolated_start_skew_ms']:.3f}／{c['pipeline_start_skew_ms']:.3f} |")
    lines += ['','此表来自独立开启探针的会话：隔离取帧2–9，流水取帧8–71。各列包含重叠区间，不能横向相加解释II；非运行时间也不能直接叫作调度等待。逐内核及最后完成者见kernels.csv。','',
        '| CPU前段 | 无探针隔离／流水服务 ms | 开探针隔离／流水服务 ms | 探针差异：隔离／流水 ms |','|---|---:|---:|---:|']
    for c in comparisons:
        lines.append(f"| {LABELS[c['case']]} | {c['clean_isolated_service_ms']:.3f}／{c['clean_pipeline_service_ms']:.3f} | {c['observed_isolated_service_ms']:.3f}／{c['observed_pipeline_service_ms']:.3f} | {c['probe_component_difference_ms']:+.3f}／{c['probe_pipeline_difference_ms']:+.3f} |")
    lines += ['','探针数据没有按无探针FPS缩放或拟合。最新绑定对照只作为背景证据保存在RESULTS.json中，不提供回放时长。','',
        '## 哪些东西重建了，哪些没有','',
        '- 可映射配置中，每帧12次计算启动的函数偏移序列、logical worker/TID、回调核时及完成汇合均经原始凭据核验。预热、计入帧、整段统计分别保留。','- coverage.csv的non_target_launches包含所有非目标调用，流水中还包括其他CPU阶段；不能把它们统称为初始化或闲置辅助工作。','- 轨迹以完成次序写入，因此按同一调用线程的开始时刻重建顺序；不是要求JSONL按开始时间排列。','- 帧对应依据完整重复序列和顺序执行，另核验存在一致的钟偏移区间；没有捏造精确绝对钟起点，也不把区间中点用作边界。','- 回放只模拟单阶段、固定一worker一核心的理想执行，显示计算节点和join；没有模拟Linux时间片、可运行队列、VTA并发或线程迁移。','- 首尾串行段、启动开销及未定位差额保留未知。CPU_accounting.csv的剩余量是未定位CPU工作，不是可以随意分到开头或结尾的墙钟。','',
        '## 完整共享预测器的最小缺口','',
        '| 缺口 | 当前状态 | 为什么需要 | 后续最小工作 |','|---|---|---|---|']
    for row in missing:
        lines.append(f"| {row['item']} | {row['status']} | {row['why']} | {row['minimum_future_data']} |")
    lines += ['','## 本轮决策','',
        ('1. **可以重建单阶段完成依赖，但仅修改核时求和顺序不是当前误差的充分修复。** 小差值不能直接解释共享流水中大得多的慢化；这也不排除完成依赖与外部调度交互的作用。' if comparisons else '1. **未得到可靠的隔离／流水逐帧对照，保留整段记录，暂不能判断求和修正的量级。**'),
        '2. **现有数据不足以建立经过验证的完整共享流水预测器。** 不能把缺少的主机、控制、串行和可运行时序写成零，或用已知流水耗时填回这些位置。',
        '3. **是否补测需要另行决定。** 下一阶段首先应提出能被独立验证的CPU调度／资源可用性假设，并说明各缺口怎样被测到；不宜直接开展全域细粒度profile或新的Top-20。',
        '', '本轮是已知案例开发回放，没有未见配置验证，不报告预测泛化、吞吐改善、显著性或跨启动结论。',
        '', '方法依据：[Daydream第4节](https://www.cs.toronto.edu/~pekhimenko/Papers/Daydream-ATC_20.pdf)的细粒度任务与依赖回放；本实现为TVM记录的独立适配，不是其完整系统复现。',
        '', f'离线产物与源记录索引：`{Path(output).resolve()}`。输入、回放先冻结，诊断后读取；见INPUT_FREEZE.json、REPLAY_FREEZE.json及各来源哈希。',
        '', '图表：dependency_work.png、dependency_graph_overlap4.png、dependency_graph_split3.png、callback_timeline_overlap4.png、callback_timeline_split3.png；完整依赖图及节点时序见交付目录graphs/下的DOT和REPLAY.json。时间线中的w表示逻辑线程，c表示物理核心；例如w0/c0即逻辑线程0固定于核心0。']
    (target/'README.md').write_text('\n'.join(lines)+'\n')
