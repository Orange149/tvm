"""Account for existing software resource holds in one completion window.

Mutex hold time is not physical VTA compute time. An input-readiness gap is
proved only when every VTA stage's next FIFO head lacks its predecessor result.
No new execution or prediction is performed.
"""
from decimal import Decimal
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cpu_vta_partitioning.common import read, write, sha256, verify_inventory
from scheduler_gap_audit.curves import csv_write
from audit_formal_gap_coverage import ROOT

OUT = ROOT / 'artifacts/formal_window_resources_01'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261111_formal_window_resources'


def ns(value):
    value = Decimal(str(value)) * 1_000_000
    if value != value.to_integral_value():
        raise ValueError('recorded millisecond clock has sub-nanosecond digits')
    return int(value)


def complement(start, end, intervals):
    clipped = sorted((max(a, start), min(b, end)) for a, b in intervals if min(b, end) > max(a, start))
    cursor = start
    gaps = []
    for a, b in clipped:
        if a < cursor:
            raise ValueError('exclusive software resource holds overlap')
        if cursor < a:
            gaps.append((cursor, a))
        cursor = b
    if cursor < end:
        gaps.append((cursor, end))
    held = sum(b - a for a, b in clipped)
    if held + sum(b - a for a, b in gaps) != end - start:
        raise ValueError('resource window lost time')
    return held, gaps


def dependency_unready_prefix(start, end, next_heads):
    """Earliest possible input-ready time across all unfinished VTA stages.

    A predecessor result alone does not prove that a host is runnable: queue
    push and wakeup may follow. Consequently a ready predecessor disables the
    no-input claim rather than being treated as a ready host.
    """
    if not next_heads:
        return 0
    earliest = min(head['predecessor_end_ns'] for head in next_heads)
    return max(0, min(end, earliest) - start)


def run():
    if OUT.exists():
        raise FileExistsError('preserve the accepted resource-window audit')
    source = ROOT / 'artifacts/formal_gap_coverage_01'
    for path, expected in read(source / 'MANIFEST.json')['files'].items():
        if sha256(path) != expected:
            raise ValueError('formal-gap evidence changed')
    cases = read(source / 'RESULTS.json')['focused_cases']
    results, gap_rows = [], []
    for case in cases:
        ref = next(p for p in case['raw_proof'] if p['phase'] == 'short')
        if sha256(ref['path']) != ref['sha256']:
            raise ValueError('focused native receipt changed')
        receipt = read(ref['path'])
        verify_inventory(receipt['raw'], receipt['files'])
        rows = [json.loads(line, parse_float=Decimal) for line in
                (Path(receipt['raw']) / 'result.jsonl').read_text().splitlines()]
        doc = receipt['task']
        a, b = doc['warmup'], doc['warmup'] + doc['scored'] - 1
        start, end = ns(rows[a]['completion_ms']), ns(rows[b]['completion_ms'])
        duration, gaps_count = end - start, b - a
        all_holds, per_stage = [], {}
        vta = [i for i, stage in enumerate(doc['stages']) if stage['device'] == 'vta']
        if not vta or min(vta) == 0:
            raise ValueError('CPU-start alternating pipeline contract differs')
        for i in vta:
            holds = []
            for frame, row in enumerate(rows):
                requested, acquired, released = [ns(row[f'stage{i}_vta_mutex_{key}_ms'])
                                                for key in ('request', 'acquire', 'release')]
                if not requested <= acquired <= released:
                    raise ValueError('native VTA resource clocks reverse')
                holds.append((acquired, released))
            if any(x[1] > y[0] for x, y in zip(holds, holds[1:])):
                raise ValueError('VTA stage violates FIFO execution')
            per_stage[i] = holds
            all_holds.extend(holds)
        held, gaps = complement(start, end, all_holds)
        dependency_ns = 0
        for x, y in gaps:
            heads = []
            for i in vta:
                pending = [(f, hold) for f, hold in enumerate(per_stage[i]) if hold[0] >= x]
                if pending:
                    frame, _ = pending[0]
                    heads.append(dict(stage=i, frame=frame,
                        predecessor_end_ns=ns(rows[frame][f'stage{i-1}_end_ms'])))
            proven = dependency_unready_prefix(x, y, heads)
            dependency_ns += proven
            gap_rows.append(dict(network=case['network'], configuration_id=case['configuration_id'],
                start_ns=x, end_ns=y, software_mutex_unheld_ns=y-x,
                every_VTA_next_input_unready_prefix_ns=proven, next_heads=heads,
                remainder_is_not_proven_scheduler_delay=True))
        group_holds = []
        if doc['configuration']['execution_mode'] == 'resource_group':
            groups = {}
            for i, stage in enumerate(doc['stages']):
                if stage['device'] != 'cpu':
                    continue
                group = tuple(stage['mask'])
                pairs = groups.setdefault(group, [])
                for row in rows:
                    request, acquire, release = [ns(row[f'stage{i}_cpu_gate_{key}_ms'])
                                                 for key in ('request', 'acquire', 'release')]
                    if not request <= acquire <= release:
                        raise ValueError('CPU lease clock reverses')
                    pairs.append((acquire, release))
            for group, pairs in groups.items():
                busy, _ = complement(start, end, pairs)
                group_holds.append(dict(mask=list(group), lease_held_ns=busy,
                    lease_occupancy_percent=100*busy/duration, lease_is_not_CPU_utilization=True))
        result = dict(network=case['network'], kind=case['kind'], configuration_id=case['configuration_id'],
            window_start_ns=start, window_end_ns=end, completion_gaps=gaps_count,
            actual_II_ms=duration/gaps_count/1e6, formal_predicted_II_ms=case['complete_predicted_II_ms'],
            VTA_software_mutex_held_ns=held, VTA_software_mutex_occupancy_percent=100*held/duration,
            software_mutex_unheld_ms_per_completion=(duration-held)/gaps_count/1e6,
            all_next_inputs_unready_ms_per_completion=dependency_ns/gaps_count/1e6,
            unclassified_unheld_ms_per_completion=(duration-held-dependency_ns)/gaps_count/1e6,
            CPU_group_leases=group_holds, raw_receipt=ref,
            physical_device_compute_utilization_unknown=True,
            input_unready_not_claimed_to_explain_all_recoverable_latency=True)
        results.append(result)
    OUT.mkdir(parents=True)
    write(OUT / 'RESULTS.json', dict(cases=results, source_sha256=sha256(__file__),
        coverage_source_sha256=sha256(source / 'RESULTS.json'), new_native_sessions=0, new_DNN=0,
        no_model_or_score_changed=True, no_sum_of_concurrent_CPU_wall_times=True))
    csv_write(OUT / 'resources.csv', results)
    csv_write(OUT / 'VTA_unheld_intervals.csv', gap_rows)
    REPORT.mkdir(parents=True, exist_ok=True)
    text = ['同一完成窗口内的软件资源占用与输入就绪', '',
        '只使用4个已完整验收的旧案例，不连接开发板、不推理、不改模型。',
        '每例使用第8至第23个完成时间戳之间的15个间隔；资源区间按同一窗口裁剪，整数纳秒守恒。',
        'VTA软件互斥区间包含主机、复制、设备调用和输出；不是物理加速器纯计算利用率。',
        'CPU组租约占用也不等于各CPU核心实际运行率。', '',
        '网络/案例 / 预测II毫秒 / 实测II毫秒 / VTA软件互斥占用% / 未持有毫秒每间隔 / 所有VTA下一输入均未就绪毫秒每间隔']
    for row in results:
        text.append(f'{row["network"]}/{row["kind"]} / {row["formal_predicted_II_ms"]:.3f} / '
            f'{row["actual_II_ms"]:.3f} / {row["VTA_software_mutex_occupancy_percent"]:.3f} / '
            f'{row["software_mutex_unheld_ms_per_completion"]:.3f} / {row["all_next_inputs_unready_ms_per_completion"]:.3f}')
    text += ['',
        '输入未就绪证明仅使用每个VTA阶段下一个未服务帧的前驱阶段完成时刻和FIFO顺序。若有前驱已经完成，则不推断主机已经可运行，也不把余下间隙全部叫调度等待。',
        '其余间隙还可能包含队列交接、唤醒、互斥释放/获取包络、调度和控制工作；没有在本轮进一步分摊。',
        '不要把逐阶段帧8–23的平均墙钟相加当作该完成窗口的严格占用：它们可能实际执行于窗口之外。',
        '本表帮助区分CPU慢化导致输入供应不足，以及软件VTA资源接近饱和的布局；不据此直接推算优化后的FPS。']
    path = REPORT / 'FINDINGS_zh.txt'
    path.write_text('\n'.join(text)+'\n')
    files = [OUT/'RESULTS.json', OUT/'resources.csv', OUT/'VTA_unheld_intervals.csv', path]
    write(OUT/'MANIFEST.json', dict(files={str(p):sha256(p) for p in files}))
    print(json.dumps([{k:r[k] for k in ('network','kind','VTA_software_mutex_occupancy_percent',
        'software_mutex_unheld_ms_per_completion','all_next_inputs_unready_ms_per_completion')} for r in results], ensure_ascii=False))


if __name__ == '__main__':
    run()
