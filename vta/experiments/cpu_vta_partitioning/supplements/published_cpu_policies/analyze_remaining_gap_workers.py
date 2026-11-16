"""Match logical workers before comparing isolated CPU and shared wall time.

Uses sealed isolated-input predictions and an already accepted diagnostic
trace. It does not rerun a predictor, change its inputs, or connect to a board.
"""
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cpu_vta_partitioning.common import read, write, sha256, verify_inventory
from scheduler_gap_audit.curves import csv_write

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/remaining_gap_same_worker_01'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261111_remaining_gap'


def model_index(tasks):
    result = {}
    for task in tasks:
        if task['stage'] != 0 or task['operation'] != 'worker_compute':
            continue
        key = (task['frame'], task['kernel'], task['slot'])
        if key in result:
            raise ValueError('duplicate logical worker task')
        if Fraction(task['CPU']['numerator'], task['CPU']['denominator']) < 0:
            raise ValueError('negative isolated worker CPU')
        result[key] = task
    return result


def match_workers(frame, kernel, native, predicted, tids):
    if sorted(w['task'] for w in native) != list(range(4)):
        raise ValueError('native callback set incomplete or duplicated')
    tasks = [predicted[(frame, kernel, slot)] for slot in range(4)]
    native_last = max(native, key=lambda w: (w['wall_end_ns'], -w['task']))['task']
    model_last = max(tasks, key=lambda t: (t['end_ns'], -t['slot']))['slot']
    rows = []
    for worker in native:
        slot = worker['task']
        if worker['tid'] != tids[slot]:
            raise ValueError('native logical worker TID differs')
        task = predicted[(frame, kernel, slot)]
        cpu = worker['cpu_end_ns'] - worker['cpu_start_ns']
        span = worker['wall_end_ns'] - worker['wall_start_ns']
        if not 0 <= cpu <= span:
            raise ValueError('native callback clock envelope inconsistent')
        rows.append(dict(frame=frame, kernel=kernel, slot=slot,
            native_callback_CPU_ns=cpu, native_callback_span_ns=span,
            model_isolated_compute_CPU=task['CPU'],
            model_same_worker_span_ns=task['end_ns']-task['first_service_ns'],
            native_last=slot == native_last, model_last=slot == model_last,
            caller=slot == 0,
            CPU_clock_boundaries_differ=True))
    return rows


def mean_ms(rows, field):
    values = [r[field] for r in rows]
    values = [Fraction(v['numerator'], v['denominator']) if isinstance(v, dict) else Fraction(v)
              for v in values]
    return float(sum(values)/len(values)/1_000_000)


def run():
    if OUT.exists():
        raise FileExistsError('preserve the accepted offline analysis')
    geometry_path = ROOT / 'artifacts/current_gap_geometry_01/RESULTS.json'
    geometry = read(geometry_path)
    sources = {str(geometry_path):sha256(geometry_path)}
    model_path = ROOT / 'artifacts/current_model_gap_01/independent.json'
    expected = geometry['sources'][str(model_path)]
    if sha256(model_path) != expected:
        raise ValueError('sealed isolated-input replay changed')
    model = read(model_path)
    if model['no_pipeline_performance_read'] is not True:
        raise ValueError('model input isolation not established')
    predicted = model_index(model['result']['tasks'])
    sources[str(model_path)] = expected
    receipt = None
    for path, expected in geometry['sources'].items():
        if '/accepted/' not in path:
            continue
        if sha256(path) != expected:
            raise ValueError('accepted diagnostic receipt changed')
        value = read(path)
        if value['task']['spec']['phase'] == 'observed':
            if receipt is not None:
                raise ValueError('ambiguous diagnostic source')
            receipt = value
            sources[path] = expected
    if receipt is None:
        raise ValueError('accepted diagnostic source missing')
    verify_inventory(receipt['raw'], receipt['files'])
    raw = Path(receipt['raw'])
    affinity = read(raw/'affinity.json')
    tids = {w['slot']:w['tid'] for w in affinity['stages'][0]['worker_slots']}
    if set(tids) != set(range(4)) or tids[0] != affinity['stages'][0]['caller_tid']:
        raise ValueError('native caller mapping incomplete')
    trace = [json.loads(s) for s in (raw/'wait_trace.jsonl').read_text().splitlines()]
    phases = [json.loads(s) for s in (raw/'phase_trace.jsonl').read_text().splitlines()]
    launches = [r for r in trace if r['kind'] == 0]
    workers = defaultdict(list)
    for record in trace:
        if record['kind'] == 1:
            workers[record['launch']].append(record)
    frames = geometry['matched_frames']
    rows = []
    for frame in frames:
        runs = [p for p in phases if p['stage']=='cpu_00_02' and p['operation']=='run' and p['call']==frame]
        if len(runs) != 1:
            raise ValueError('native run boundary ambiguous')
        phase = runs[0]
        group = sorted([l for l in launches if l['caller']==phase['tid'] and
            phase['wall_start_ns'] <= l['wall_start_ns'] <= l['wall_end_ns'] <= phase['wall_end_ns']],
            key=lambda l:l['wall_start_ns'])
        if len(group) != 12:
            raise ValueError('native launch sequence incomplete')
        for kernel, launch in enumerate(group):
            rows.extend(match_workers(frame, kernel, workers[launch['launch']], predicted, tids))
    kernels = []
    for kernel in range(12):
        chosen = [r for r in rows if r['kernel']==kernel]
        last = [r for r in chosen if r['native_last']]
        kernels.append(dict(kernel=kernel,
            native_last_worker_histogram=dict(Counter(r['slot'] for r in last)),
            model_last_worker_histogram=dict(Counter(r['slot'] for r in chosen if r['model_last'])),
            last_worker_agreement=sum(r['native_last'] and r['model_last'] for r in chosen)/len(frames),
            same_native_last_worker={k:mean_ms(last,k) for k in (
                'native_callback_CPU_ns','model_isolated_compute_CPU',
                'native_callback_span_ns','model_same_worker_span_ns')},
            clock_boundaries_differ_no_causal_decomposition=True))
    # CPU work across all workers may be summed; wall times are not summed.
    native_cpu = sum(r['native_callback_CPU_ns'] for r in rows)/len(frames)/1e6
    isolated_cpu = float(sum(Fraction(r['model_isolated_compute_CPU']['numerator'],
        r['model_isolated_compute_CPU']['denominator']) for r in rows)/len(frames)/1e6)
    result = dict(rows=rows, kernels=kernels, matched_frames=frames,
        all_worker_CPU_ms_per_front_frame=dict(native_callback=native_cpu,isolated_model_compute=isolated_cpu),
        CPU_scope_difference_not_pure_instruction_work=True,
        prior_nonoverlapping_stage_gap=geometry['model_minus_native_observed_ms']['independent'],
        new_native_sessions=0,new_DNN=0,new_replay=0,model_or_cost_or_Top20_modified=False,
        sources=sources,source_sha256=sha256(__file__),
        scope='same diagnostic frames and same logical worker; diagnostic comparison only')
    OUT.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True)
    write(OUT/'RESULTS.json',result);csv_write(OUT/'workers.csv',rows);csv_write(OUT/'kernels.csv',kernels)
    lines=['隔离profile与流水慢化：当前重点与同线程核对','',
        '本轮只读取原始记录；0次新推理、0次板端连接，不修改评分、DP或Top-20。',
        '基准和版本差异已经厘清。当前目标是解释计算工作进入共享流水后为什么变慢。','',
        '同一组诊断帧12–23：前段run观测225.355ms，已有隔离输入回放192.527ms，仍低估32.828ms。',
        '非重叠路径分解：最后完成worker回调体跨度低估20.882ms；其外调用线程及返回/串行区间低估12.091ms；进入前区间总量多估0.145ms。',
        '这是诊断窗口，不用探针开启的时间替代正式FPS，也不把不同批次的隔离时间硬接成因果等式。','',
        '以下只比较实际最后完成的同一个逻辑线程，避免模型换了最后完成者导致CPU工作比较失真。',
        '并行计算代码序号 / 隔离计算CPU ms / 流水回调CPU ms / 流水回调跨度ms / 模型同线程跨度ms / 最后完成者匹配率']
    for r in kernels:
        v=r['same_native_last_worker']
        lines.append(f"{r['kernel']} / {v['model_isolated_compute_CPU']:.3f} / {v['native_callback_CPU_ns']:.3f} / {v['native_callback_span_ns']:.3f} / {v['model_same_worker_span_ns']:.3f} / {100*r['last_worker_agreement']:.1f}%")
    lines += ['',f'全部worker计算核时之和：模型{isolated_cpu:.3f}核毫秒，流水回调{native_cpu:.3f}核毫秒。',
        '两者CPU时钟边界不同，差额包含回调包络与探针等开销；不能全部当作算术工作、缓存损失或调度成本。',
        '可以确认：单看总CPU工作量接近，仍不能保证关键线程按模型预期的时刻完成；关键worker的具体核上竞争与先后次序很重要。',
        '不能确认：壁钟减CPU时间全部是调度排队，或移除某个竞争线程就能等量恢复这段时间。',
        '与线程份额作用域审计分开：整会话份额可被输出统计污染，但此处细粒度回放使用隔离内核CPU工作，未用该整会话份额。',
        '下一步围绕误差最大的计算序号2、6及调用线程区间核对实际与预测的同核竞争、就绪/唤醒及服务次序；先解释这些位置，不再扩展全域候选或添加通用慢化系数。']
    (REPORT/'FINDINGS_zh.txt').write_text('\n'.join(lines)+'\n')
    files=[OUT/'RESULTS.json',OUT/'workers.csv',OUT/'kernels.csv',REPORT/'FINDINGS_zh.txt']
    write(OUT/'MANIFEST.json',dict(files={str(p):sha256(p) for p in files}))
    print(json.dumps(dict(all_worker_CPU_ms=result['all_worker_CPU_ms_per_front_frame'],
        representative_kernels=[r for r in kernels if r['kernel'] in (2,6,10)],
        report=str(REPORT/'FINDINGS_zh.txt')),ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    run()
