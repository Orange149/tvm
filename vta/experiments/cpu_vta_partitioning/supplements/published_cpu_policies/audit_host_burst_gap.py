"""Attribute existing host CPU bursts without changing or fitting a predictor.

All native inputs are the accepted, loss-free scheduler trace. Model inputs
are the frozen finite-service replay of exactly the same execution layout.
Coincidence with a preempted critical worker is evidence, not causal latency.
"""
from bisect import bisect_right
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cpu_vta_partitioning.common import read, write, sha256, verify_inventory
from scheduler_gap_audit.curves import csv_write
from yolo_scheduler_trace.events import clip, intersection

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'artifacts/host_burst_gap_01'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261111_host_burst_gap'


def fraction(x):
    return Fraction(x['numerator'], x['denominator'])


class Phases:
    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda r: r['wall_start_ns'])
        self.starts = [r['wall_start_ns'] for r in self.rows]
        previous = None
        for r in self.rows:
            a, b = r['wall_start_ns'], r['wall_end_ns']
            if b < a or (previous is not None and a < previous):
                raise ValueError('phase intervals overlap or run backwards')
            previous = b

    def split(self, start, end):
        if end < start:
            raise ValueError('negative observation interval')
        result = []
        cursor = start
        i = max(0, bisect_right(self.starts, start) - 1)
        while i < len(self.rows) and self.rows[i]['wall_start_ns'] < end:
            row = self.rows[i]
            a, b = max(start, row['wall_start_ns']), min(end, row['wall_end_ns'])
            if a < b:
                if cursor < a:
                    result.append(('outside_recorded_phases', cursor, a))
                result.append((row['operation'], a, b))
                cursor = b
            i += 1
        if cursor < end:
            result.append(('outside_recorded_phases', cursor, end))
        if sum(b - a for _, a, b in result) != end - start:
            raise ValueError('phase split does not conserve observed CPU')
        return result


def critical_overlap(trace, hosts, phase_indexes, threshold=50_000):
    by_core = {c: [] for c in range(4)}
    for host in hosts:
        for span in host['on']:
            by_core[span['cpu']].append((span, host))
    for rows in by_core.values():
        rows.sort(key=lambda r: r[0]['start_ns'])
        if any(a[0]['end_ns'] > b[0]['start_ns'] for a, b in zip(rows, rows[1:])):
            raise ValueError('two native hosts occupy the same CPU simultaneously')
    starts = {c: [r[0]['start_ns'] for r in rows] for c, rows in by_core.items()}
    totals = defaultdict(int)
    for critical in trace['critical_rows']:
        stream = trace['observed']['scheduler']['streams'][str(critical['tid'])]
        for off in clip(stream, critical['callback_start_ns'], critical['callback_end_ns'])['off']:
            if not off['preempt']:
                continue
            core = off['from_cpu']
            if core != off['to_cpu']:
                raise ValueError('fixed CPU worker changed core')
            rows = by_core[core]
            i = max(0, bisect_right(starts[core], off['start_ns']) - 1)
            while i < len(rows) and rows[i][0]['start_ns'] < off['end_ns']:
                span, host = rows[i]
                a = max(off['start_ns'], span['start_ns'])
                b = min(off['end_ns'], span['end_ns'])
                if a < b:
                    size = 'short' if span['end_ns'] - span['start_ns'] <= threshold else 'long'
                    for op, begin, end in phase_indexes[host['tid']].split(a, b):
                        totals[host['role']['role'], core, op, size, critical['kernel']] += end - begin
                i += 1
    expected = sum(n for role, n in trace['coincident_CPU_by_role_ns'].items() if role.endswith('/host'))
    if sum(totals.values()) != expected:
        raise ValueError('host attribution differs from accepted original trace')
    nframes = len(trace['complete_front_frames'])
    return [dict(host=k[0], core=k[1], operation=k[2], span_type=k[3], critical_kernel=k[4],
                 coincident_CPU_ns=n, core_ms_per_front_frame=n/nframes/1e6)
            for k, n in sorted(totals.items())]


def run():
    base = ROOT / 'artifacts/yolo_scheduler_trace_repair_01'
    trace_path = base / 'analysis/RESULTS.json'
    if sha256(trace_path) != read(base / 'DELIVERY.json')['results_sha256']:
        raise ValueError('accepted native trace changed')
    trace = read(trace_path)
    if read(base / 'FINAL_VALIDATION.json')['status'] != 'PASSED' or not trace['probe_quality_passed']:
        raise ValueError('native trace acceptance or probe quality missing')
    if trace['complete_front_frames'] != list(range(12, 24)):
        raise ValueError('critical callback population changed')
    source_files = {str(trace_path): sha256(trace_path)}
    for receipt, expected in trace['source_receipts'].items():
        if sha256(receipt) != expected:
            raise ValueError('native receipt changed')
        source_files[receipt] = expected
        value = read(receipt)
        if value['task']['spec']['phase'] == 'observed':
            observed = value
    verify_inventory(observed['raw'], observed['files'])
    raw_phases = Path(observed['raw']) / 'phase_trace.jsonl'
    source_files[str(raw_phases)] = sha256(raw_phases)
    phases = [json.loads(s) for s in raw_phases.read_text().splitlines()]
    hosts = [s for s in trace['observed']['scheduler']['streams'].values() if s['role']['kind'] == 'VTA_host']
    indexes = {h['tid']: Phases([p for p in phases if p['tid'] == h['tid']]) for h in hosts}
    rows = critical_overlap(trace, hosts, indexes)

    dev = ROOT / 'artifacts/finite_service_01/development'
    seal = read(dev / 'REPLAY_FREEZE.json')
    for path, expected in seal['sources'].items():
        if sha256(path) != expected:
            raise ValueError('frozen finite-service source changed')
    inp = read(dev / 'INPUTS.json')
    if read(dev / 'PLAN.json')['inputs_sha256'] != sha256(dev / 'INPUTS.json'):
        raise ValueError('frozen isolated input changed')
    case = inp['isolated']['cases']['yolov3_tiny416/overlap']
    if [s['worker_core_map'] for s in case['configuration']['stages'] if s['device'] == 'cpu'] != [[0,1,2,3], [1,2,3], [0]]:
        raise ValueError('model is not the diagnostic execution layout')
    replay_path = None
    for leaf, expected in seal['files'].items():
        item = read(dev / leaf)
        if item['spec'] == dict(case='yolov3_tiny416/overlap', mode='finite', placement='wake_boundary'):
            if sha256(dev / leaf) != expected:
                raise ValueError('frozen replay changed')
            replay_path, model = dev / leaf, item['result']
    if replay_path is None:
        raise ValueError('exact frozen model case missing')
    source_files[str(replay_path)] = sha256(replay_path)
    source_files[str(dev / 'INPUTS.json')] = sha256(dev / 'INPUTS.json')

    # Compare complete native phase calls, not censored phases at trace edges.
    eligible = []
    a, b = trace['common_interior_ns']
    for frame in trace['complete_front_frames']:
        group = [p for p in phases if p['tid'] in indexes and p['call'] == frame]
        if len(group) == 6 and all(a <= p['wall_start_ns'] <= p['wall_end_ns'] <= b for p in group):
            eligible.append(frame)
    if eligible != list(range(12, 22)):
        raise ValueError('complete native host-call population changed')
    phase_rows = []
    for host in hosts:
        stage = host['role']['stage']
        for op in ('set', 'run', 'get'):
            group = [p for p in phases if p['tid'] == host['tid'] and p['call'] in eligible and p['operation'] == op]
            cpu = [0]*4
            for p in group:
                clipped = clip(host, p['wall_start_ns'], p['wall_end_ns'])
                if clipped['unknown_ns']:
                    raise ValueError('complete host phase has uncovered scheduler time')
                for on in clipped['on']:
                    cpu[on['cpu']] += on['end_ns'] - on['start_ns']
            names = {'host_serial', 'device_host_CPU'} if op == 'run' else {op}
            tasks = [t for t in model['tasks'] if t['role'] == 'stage' and t['stage'] == stage
                     and t['frame'] in eligible and t['operation'] in names]
            predicted = sum((fraction(t['CPU']) for t in tasks), Fraction(0))
            model_core = [Fraction(0)]*4
            for t in model['actor_totals']:
                if t['role'] == 'stage' and t['stage'] == stage and t['operation'] in names:
                    for core, work in t['core_CPU'].items():
                        model_core[int(core)] += fraction(work)/inp['windows']['yolov3_tiny416/overlap']['scored']
            phase_rows.append(dict(host=host['role']['role'], operation=op, native_calls=len(group),
                native_frames=eligible, native_thread_clock_core_ms_per_call=sum(p['cpu_end_ns']-p['cpu_start_ns'] for p in group)/len(group)/1e6,
                native_switch_core_ms_per_call=sum(cpu)/len(group)/1e6,
                model_core_ms_per_call=float(predicted/len(eligible)/1_000_000),
                native_core_distribution_ms_per_call=[n/len(group)/1e6 for n in cpu],
                model_core_distribution_ms_per_call=[float(n/1_000_000) for n in model_core],
                model_core_distribution_frames=list(range(8,24))))
            phase_rows[-1]['thread_clock_minus_switch_core_ms_per_call'] = (
                phase_rows[-1]['native_thread_clock_core_ms_per_call'] -
                phase_rows[-1]['native_switch_core_ms_per_call'])

    # Graph inventory is evidence of the VTA subgraph's composition, not a
    # per-function attribution: existing trace does not timestamp graph nodes.
    graphs = []
    programs = read(base / 'capsule/plans/programs.json')
    for program in ('vta_03_06', 'vta_09_12'):
        path = base / 'capsule/programs' / program / 'graph.json'
        if sha256(path) != programs[program]['files']['graph.json']['sha256']:
            raise ValueError('compiled graph identity differs from native capsule')
        source_files[str(path)] = sha256(path)
        graphs.append(dict(program=program, graph_sha256=sha256(path),
            functions=[n['attrs']['func_name'] for n in read(path)['nodes'] if n['op'] != 'null'],
            individual_function_times_observed=False))
    result = dict(source_files=source_files, source_sha256=sha256(__file__),
        critical_host_overlap=rows, host_phase_costs=phase_rows, graph_inventory=graphs,
        native_boot=trace['boot_id'], critical_front_frames=trace['complete_front_frames'],
        phase_frames=eligible, coincident_CPU_is_not_causal_latency=True,
        fixed_execution_layout=True, compared_model='frozen finite-service aggregate device model; not latest independent device model',
        source_profiles_not_refreshed=True, new_native_sessions=0, new_DNN=0,
        no_scoring_DP_or_Top20_change=True, no_model_parameter_fitted=True,
        long_span_threshold_ns=50_000)
    result['critical_last_worker_counts'] = {
        str(slot): sum(r['slot'] == slot for r in trace['critical_rows']) for slot in range(4)}
    result['host_CPU_clock_vs_switch_discrepancy_unresolved'] = True
    result['stage_service_comparison'] = [dict(
        program=s['program'], predicted_service_ms=model['replay']['stage_service_ns'][i]/1e6,
        native_clean_service_ms=trace['clean']['stage_service_ms'][i],
        prediction_minus_clean_ms=model['replay']['stage_service_ns'][i]/1e6-trace['clean']['stage_service_ms'][i],
        prediction_is_not_latest_model=True)
        for i,s in enumerate(case['configuration']['stages'])]
    from audit_finite_service_alignment import summarize_model, summarize_native
    frames, native_kernels = summarize_native(trace)
    model_kernels = summarize_model(model, frames)
    kernel_rows = []
    for m, n in zip(model_kernels, native_kernels):
        kernel_rows.append(dict(kernel=m['kernel'],
            native_launch_to_last_callback_end_ms=n['launch_to_last_callback_end_ms'],
            model_ready_to_join_ms=m['join_span_ms'],
            native_callback_entry_skew_ms=n['callback_entry_skew_ms'],
            model_first_service_skew_ms=m['first_service_skew_ms'],
            host_coincident_core_ms_per_front_frame=sum(r['core_ms_per_front_frame'] for r in rows if r['critical_kernel']==m['kernel']),
            native_entry_and_model_ready_envelopes_distinct=True,
            not_a_stage_latency_decomposition=True))
    result['kernel_timing_comparison'] = kernel_rows
    result['alignment_source_sha256'] = sha256(ROOT/'audit_finite_service_alignment.py')
    write(OUTPUT / 'RESULTS.json', result, replace=True)
    csv_write(OUTPUT / 'critical_host_phase_core_kernel.csv', rows)
    csv_write(OUTPUT / 'host_phase_costs.csv', phase_rows)
    csv_write(OUTPUT / 'stage_service_comparison.csv', result['stage_service_comparison'])
    csv_write(OUTPUT / 'kernel_timing_comparison.csv', kernel_rows)
    lines = ['隔离profile与流水差距：主机长片段定位', '',
        '本轮只核查已有原始轨迹和冻结回放，新增板测/推理为0，未修改模型、DP或候选。',
        '本次推进：将关键worker等待期间同核重合的主机工作，具体定位到VTA阶段、set/run/get和物理核心。',
        '关键观察：VTA3–6的主机在旧回放中全部使用核心1；轨迹中约九成CPU工作使用核心3。总核时已计入，不代表具体落核及发生时机正确。',
        '局限：run范围的线程CPU钟与调度轨迹重建量尚有差额，不能以任一数字直接认定计算工作量不变或增加；也未完成38ms误差的因果分解。',
        '关键worker重合统计：前段帧12–23、144次并行启动；完整主机phase成本：帧12–21，另两帧受轨迹边界截断，不补造。',
        '对照模型为同一执行布局的旧有限服务、聚合设备时序版本，不冒充最新设备独立完成模型。', '']
    lines += ['同一固定布局的五阶段服务对照（关闭探针的正式实测）：']
    for row in result['stage_service_comparison']:
        lines.append(f"  {row['program']}：旧有限服务模型{row['predicted_service_ms']:.3f}ms，流水{row['native_clean_service_ms']:.3f}ms，预测减实测{row['prediction_minus_clean_ms']:+.3f}ms。")
    lines += ['前段仍偏快，但其他阶段已预测过慢；因此统一增加主机或共享慢化惩罚并不是有证据的修正。', '']
    for host in hosts:
        name = host['role']['role']
        lines.append(name + '：')
        subset = [r for r in phase_rows if r['host'] == name]
        for p in subset:
            lines.append(f"  {p['operation']}：隔离输入所定义的CPU工作{p['model_core_ms_per_call']:.3f}核ms/次；流水线程时钟{p['native_thread_clock_core_ms_per_call']:.3f}核ms/次；调度轨迹重建{p['native_switch_core_ms_per_call']:.3f}核ms/次。")
        actual = [sum(p['native_core_distribution_ms_per_call'][c] for p in subset) for c in range(4)]
        predicted = [sum(p['model_core_distribution_ms_per_call'][c] for p in subset) for c in range(4)]
        lines.append('  核心0–3平均CPU工作：实测' + str([round(n,3) for n in actual]) + '；旧回放' + str([round(n,3) for n in predicted]) + '核ms/次。')
        grouped = defaultdict(float)
        for r in rows:
            if r['host'] == name:
                grouped[r['core'], r['operation'], r['span_type']] += r['core_ms_per_front_frame']
        for (core, op, size), n in sorted(grouped.items()):
            lines.append(f'  关键worker被抢占时，同核主机重合：核心{core}，{op}，{size}片段，{n:.3f}核ms/前段帧。')
    lines += ['', '含义：需同时检查CPU工作量、出现时序和落核；计入主机总核时并不保证预测了关键worker受到的实际竞争。',
        '长片段主要落在VTA子图的run范围，不能将run等同于纯FPGA计算或把全部主机工作归为短轮询。',
        '图中存在转置/布局、类型转换、激活、池化等函数；当前记录没有其独立时间，不能将长片段具体归因给某一个函数。',
        '两个run范围的线程CPU钟分别比调度轨迹多约4.95和3.35核ms/次，此差额保留为计时口径待核查项，未归因给额外计算、等待或某个算子。',
        '核心分布的模型使用16个计入帧、原始记录使用10个完整帧，仅作同布局描述性核对，不作为因果干预或效果提升。',
        '同核重合核时不能加到阶段服务或完成间隔上；当前仍未完成约38ms误差的无重叠定量分解。',
        '旧流体模型剩余约38ms不应当作所有后续版本的固定残差：本文件的有限服务回放对前段仍低估约11.35ms，但已在其他阶段高估，且其前瞻选点验证未通过，未成为正式修正。',
        '逐内核对照：第7次启动的实测跨度约71.08ms、回放约63.28ms；第11次则实测约63.91ms、回放约72.69ms。两者均有明显同核主机重合，但误差方向相反，不能把重合核时直接当作惩罚。',
        '下一步先核对主机run中长CPU工作的隔离/流水工作量及其相对关键计算的出现时机，再决定是否需要最小化采集节点级CPU时间。']
    (OUTPUT / 'FINDINGS_zh.txt').write_text('\n'.join(lines) + '\n')
    REPORT.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT.iterdir():
        if path.is_file():
            shutil.copy2(path, REPORT / path.name)
    return dict(report=str(REPORT / 'FINDINGS_zh.txt'), phase_costs=phase_rows,
                total_coincident_core_ms_per_front_frame=sum(r['core_ms_per_front_frame'] for r in rows),
                new_native_sessions=0)


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
