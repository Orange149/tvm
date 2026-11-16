"""Identify serial work and directly recorded observer work in an old trace.

This audit does not remove time from a performance result or modify a model.
ELF inspection is restricted to the authenticated module used by that trace.
"""
from collections import defaultdict
from pathlib import Path
import json
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cpu_vta_partitioning.common import read, sha256, verify_inventory, write
from audit_caller_gap_scope import caller_intervals
from scheduler_gap_audit.curves import csv_write
from yolo_scheduler_trace.events import clip, intersection
from yolo_scheduler_trace_repair.events import validate_profile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/serial_gap_work_01'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261111_serial_gap_work'


def disjoint_queries(launches):
    rows = []
    for launch in launches:
        for index in (0, 1):
            rows.append(dict(start=launch[f'queue_read_start{index}_ns'],
                end=launch[f'queue_read_end{index}_ns'], kind=f'launch_query_{index}'))
        for item in launch['worker_queue_baselines']:
            if item['tid']:
                rows.append(dict(start=item['read_start_ns'], end=item['read_end_ns'],
                                 kind='worker_queue_baseline'))
    rows.sort(key=lambda row: (row['start'], row['end']))
    if any(row['start'] <= 0 or row['end'] < row['start'] for row in rows):
        raise ValueError('counter query clock invalid')
    if any(a['end'] > b['start'] for a, b in zip(rows, rows[1:])):
        raise ValueError('counter queries overlap; cannot sum them')
    return rows


def query_occupancy(stream, start, end, queries):
    wall = cpu = 0
    for query in queries:
        if not intersection(start, end, query['start'], query['end']):
            continue
        a, b = max(start, query['start']), min(end, query['end'])
        span = clip(stream, a, b)
        if not span['fully_covered']:
            raise ValueError('query scheduler coverage incomplete')
        wall += b - a
        cpu += span['on_CPU_ns']
    return dict(query_wall_ns=wall, query_on_CPU_ns=cpu)


def serial_call_proof(disassembly, relocations):
    """Check the two indirect calls in this frozen pool/quantize function.

    Match their literal GOT addressing instructions, rather than interpreting
    the absence of a printed ParallelLaunch symbol as proof of no call.
    """
    instructions = []
    for line in disassembly.splitlines():
        match = re.match(r'^\s*([0-9a-f]+):\s+[0-9a-f]{8}\s+(.+)$', line)
        if match:
            instructions.append((int(match[1], 16), match[2]))
    calls = [(i, a, text) for i, (a, text) in enumerate(instructions)
             if re.match(r'bl(?:r)?\s+', text)]
    if len(calls) != 2 or any(not re.match(r'blr\s+x8\b', text) for _, _, text in calls):
        raise ValueError('frozen serial function call structure changed')
    result = []
    for (index, address, _), offset, name in zip(calls, (4064, 4024),
            ('__TVMBackendAllocWorkspace', '__TVMBackendFreeWorkspace')):
        context = '\n'.join(text for _, text in instructions[max(0, index - 24):index + 1])
        if (not re.search(r'adrp\s+x8,\s*2e000\b', context)
                or not re.search(rf'ldr\s+x8,\s*\[x8,\s*#{offset}\]', context)
                or not re.search(r'ldr\s+x8,\s*\[x8\]', context)):
            raise ValueError('serial indirect call address not established')
        got = 0x2e000 + offset
        if not re.search(rf'^0*{got:x}\s+.*\b{name}\s+\+', relocations, re.M):
            raise ValueError('serial indirect call relocation differs')
        result.append(dict(call_offset=address, GOT_offset=got, target=name))
    return result


def run():
    if OUT.exists():
        raise FileExistsError('preserve the serial-work evidence')
    previous = ROOT / 'artifacts/caller_gap_scope_01'
    seal = read(previous / 'MANIFEST.json')
    for path, expected in seal['files'].items():
        if sha256(path) != expected:
            raise ValueError('accepted caller evidence changed')
    prior = read(previous / 'RESULTS.json')
    sources = {str(previous / 'RESULTS.json'): sha256(previous / 'RESULTS.json')}
    model_plan = read(ROOT / 'artifacts/current_model_gap_01/PLAN.json')
    host_source = ROOT / 'device_completion_replay.py'
    if sha256(host_source) != model_plan['original_model_sources'][str(host_source)]:
        raise ValueError('sealed host timing implementation changed')
    sources[str(host_source)] = sha256(host_source)
    chosen = []
    for path, expected in prior['sources'].items():
        if '/accepted/' not in path:
            continue
        if sha256(path) != expected:
            raise ValueError('accepted diagnostic source changed')
        receipt = read(path)
        if receipt['task']['spec']['phase'] == 'observed':
            chosen.append(receipt)
            sources[path] = expected
    if len(chosen) != 1:
        raise ValueError('ambiguous diagnostic receipt')
    receipt = chosen[0]
    verify_inventory(receipt['raw'], receipt['files'])
    raw = Path(receipt['raw'])
    affinity = read(raw / 'affinity.json')
    caller = affinity['stages'][0]['caller_tid']
    streams = validate_profile(read(raw / 'profile.json'), receipt['task'], affinity)['streams']
    stream = streams[str(caller)]
    lines = lambda path: [json.loads(s) for s in path.read_text().splitlines()]
    trace, phases = lines(raw / 'wait_trace.jsonl'), lines(raw / 'phase_trace.jsonl')
    launches = [x for x in trace if x['kind'] == 0 and x['caller'] == caller]
    callbacks = defaultdict(list)
    for row in trace:
        if row['kind'] == 1:
            callbacks[row['launch']].append(row)
    queries = disjoint_queries(launches)
    rows = []
    for frame in prior['matched_frames']:
        phase = next(p for p in phases if p['stage'] == 'cpu_00_02'
                     and p['operation'] == 'run' and p['call'] == frame)
        selected = sorted([x for x in launches if phase['wall_start_ns'] <= x['wall_start_ns']
            <= x['wall_end_ns'] <= phase['wall_end_ns']], key=lambda x: x['wall_start_ns'])
        if len(selected) != 12:
            raise ValueError('parallel sequence incomplete')
        intervals = caller_intervals(phase['wall_start_ns'], phase['wall_end_ns'],
            [dict(start=x['wall_start_ns'], end=x['wall_end_ns'],
                  last_callback_end=max(w['wall_end_ns'] for w in callbacks[x['launch']])) for x in selected])
        for interval in intervals:
            span = clip(stream, interval['start'], interval['end'])
            if not span['fully_covered']:
                raise ValueError('serial interval scheduler coverage incomplete')
            rows.append(dict(frame=frame, **interval, wall_ns=interval['end'] - interval['start'],
                on_CPU_ns=span['on_CPU_ns'], off_CPU_ns=span['off_CPU_ns'],
                **query_occupancy(stream, interval['start'], interval['end'], queries)))
    serial_rows = [row for row in rows if row['kind'] != 'callback_end_to_return']
    count = len(prior['matched_frames'])
    mean = lambda records, key: sum(row[key] for row in records) / count / 1e6
    observed_serial_CPU = mean(serial_rows, 'on_CPU_ns')
    known_query_CPU = mean(serial_rows, 'query_on_CPU_ns')
    isolated_serial_CPU = prior['model']['CPU_ms']
    if mean(rows, 'on_CPU_ns') != prior['native']['on_CPU_ms']:
        raise ValueError('serial audit no longer reproduces caller CPU scope')
    build_dir = ROOT / 'artifacts/yolo_scheduler_trace_repair_01/build'
    build = read(build_dir / 'BUILD.json')
    if (build['runner_sha256'] != receipt['task']['executor_sha256']
            or build['adapter_sha256'] != receipt['task']['adapter_sha256']):
        raise ValueError('observer source/build identity differs')
    for name in ('adapter.cc', 'phase_probe.h'):
        path = build_dir / 'sources' / name
        if sha256(path) != build['source_files'][str(path)]:
            raise ValueError('archived observer source changed')
        sources[str(path)] = sha256(path)
    capsule = ROOT / 'artifacts/yolo_scheduler_trace_repair_01/capsule'
    inventory = read(capsule / 'plans/programs.json')['cpu_00_02']
    program = capsule / 'programs/cpu_00_02'
    for name in ('graph.json', 'module.so'):
        path = program / name
        if sha256(path) != inventory['files'][name]['sha256']:
            raise ValueError('executed graph/module changed')
        sources[str(path)] = sha256(path)
    graph = read(program / 'graph.json')
    graph_functions = [n['attrs']['func_name'] for n in graph['nodes'] if n['op'] == 'tvm_op']
    if len(graph_functions) != 9:
        raise ValueError('compiled CPU graph sequence changed')
    OUT.mkdir(parents=True)
    reloc = subprocess.run(['aarch64-linux-gnu-readelf', '-rW', str(program / 'module.so')],
        capture_output=True, text=True, check=True).stdout
    proofs = []
    positions = []
    for kernel, graph_index in ((4, 2), (8, 5)):
        function = graph_functions[graph_index]
        if 'max_pool2d_multiply_round_clip_cast' not in function:
            raise ValueError('serial graph operation differs')
        symbol = function + '_compute_'
        asm = subprocess.run(['aarch64-linux-gnu-objdump', '-d', '--disassemble=' + symbol,
            str(program / 'module.so')], capture_output=True, text=True, check=True).stdout
        calls = serial_call_proof(asm, reloc)
        asm_path = OUT / (function + '.asm')
        asm_path.write_text(asm)
        proofs.append(dict(symbol=symbol, graph_index=graph_index, call_targets=calls,
                           disassembly=str(asm_path), sha256=sha256(asm_path)))
        actual = [x for x in serial_rows if x['kernel'] == kernel]
        model = next(x for x in prior['serial_positions'] if x['kernel'] == kernel)['model']['CPU_ms']
        positions.append(dict(before_parallel_kernel=kernel, intervening_graph_function=function,
            graph_attribution_includes_executor_and_observer_envelope=True,
            actual_on_CPU_ms=mean(actual, 'on_CPU_ns'), known_query_CPU_ms=mean(actual, 'query_on_CPU_ns'),
            model_serial_CPU_ms=model,
            remaining_CPU_envelope_difference_ms=mean(actual, 'on_CPU_ns') - mean(actual, 'query_on_CPU_ns') - model))
    result = dict(actual_serial_CPU_ms=observed_serial_CPU, directly_recorded_query_CPU_ms=known_query_CPU,
        model_serial_CPU_ms=isolated_serial_CPU,
        remaining_serial_CPU_envelope_difference_ms=observed_serial_CPU - known_query_CPU - isolated_serial_CPU,
        callback_end_to_return_on_CPU_ms=mean([x for x in rows if x['kind'] == 'callback_end_to_return'], 'on_CPU_ns'),
        positions=positions, binary_proofs=proofs, matched_frames=prior['matched_frames'], sources=sources,
        source_sha256=sha256(__file__), query_subtraction_is_CPU_scope_label_only=True,
        no_counterfactual_FPS_or_corrected_wall_estimated=True,
        whole_probe_perturbation_not_equal_to_direct_query_CPU=True,
        graph_location_is_not_function_level_time_measurement=True,
        new_board_connections=0, new_DNN=0, new_native_sessions=0,
        static_model_DP_profile_and_Top20_unchanged=True)
    write(OUT / 'RESULTS.json', result)
    csv_write(OUT / 'intervals.csv', rows)
    csv_write(OUT / 'serial_positions.csv', positions)
    REPORT.mkdir(parents=True, exist_ok=True)
    text = ['YOLO前段剩余误差：串行计算与计数读取范围', '',
        '本轮只读已验收轨迹与实际模块。无板端连接、无新推理，不修改静态评分或DP。',
        '所有数字使用相同诊断帧12–23；探针窗口不是正式FPS。', '',
        f'串行间隙实际运行CPU {observed_serial_CPU:.3f}ms，模型串行CPU {isolated_serial_CPU:.3f}ms。',
        f'其中直接记录到的计数接口读取占用CPU {known_query_CPU:.3f}ms；它不是DNN算术工作。',
        f'按CPU作用域剥离这些已记录查询后，剩余包络差 {result["remaining_serial_CPU_envelope_difference_ms"]:.3f}ms。',
        '这只是观测CPU工作分类，不能从流水墙钟直接扣除，更不能推算无探针FPS。', '',
        '位置 / 实际CPUms / 已知查询CPUms / 模型CPUms / 其余包络差ms']
    for item in positions:
        text.append(f'第{item["before_parallel_kernel"] + 1}次并行启动之前 / '
            + ' / '.join(f'{item[key]:.3f}' for key in ('actual_on_CPU_ms', 'known_query_CPU_ms',
                'model_serial_CPU_ms', 'remaining_CPU_envelope_difference_ms')))
        text.append('相邻编译图操作：' + item['intervening_graph_function'])
    text += ['',
        '实际ELF中两处池化/量化计算函数各有两处间接调用，字面GOT地址及重定位分别指向workspace分配和释放；没有TVM并行启动调用。',
        '因此“阶段配置四线程”不表示该阶段所有工作都并行。图序列能定位这两处串行工作，但当前区间也包含图执行器、探针等包络，不冒充函数级计时。',
        '原隔离输入已经包含这些串行CPU工作；不是发现了两个从未建模的新算子，也不应重复添加池化成本。',
        f'完成回调结束到启动返回的另一部分实际运行CPU {result["callback_end_to_return_on_CPU_ms"]:.3f}ms，另有约6.203ms未运行，仍不等于可删除的返回代价。',
        '串行包络差约1.5ms，不足以单独解释整体32.8ms残差；不能把它直接归为缓存、带宽或算术指令增加。', '',
        '主机时序源码核查：现有CompletionMixin.actions已在每个真实设备调用前后插入host_serial_CPU，并保留提交和轮询段；不是将所有主机CPU工作塞在阶段开头。',
        '仍待解释的是这些工作在共享核心上何时得到执行、关键worker如何排队，以及实际完成次序为何与模型不同。',
        '结论：暂不重新标定所有子图或再加核时惩罚；下一步只核查关键并行调用的服务次序与模型可观测缺口。',
        '已有换核、阻塞等待和迁移结果继续作为反证，不重复执行。']
    report = REPORT / 'FINDINGS_zh.txt'
    report.write_text('\n'.join(text) + '\n')
    files = [OUT / 'RESULTS.json', OUT / 'intervals.csv', OUT / 'serial_positions.csv', report]
    files += [Path(proof['disassembly']) for proof in proofs]
    write(OUT / 'MANIFEST.json', dict(files={str(p): sha256(p) for p in files}))
    print(json.dumps({k: result[k] for k in ('directly_recorded_query_CPU_ms',
        'remaining_serial_CPU_envelope_difference_ms', 'positions', 'new_DNN')}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    run()
