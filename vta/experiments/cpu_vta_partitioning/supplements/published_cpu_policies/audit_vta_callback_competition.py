"""Classify already-recorded VTA CPU callbacks, without another native session.

The output is diagnostic evidence, never a component cost or FPS predictor.
Callback wall intervals are intersected with recorded on-CPU intervals before
being compared with the critical CPU worker's off-CPU intervals.
"""
from bisect import bisect_right
from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import statistics as st
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cpu_vta_partitioning.common import read, sha256, verify_inventory, write
from scheduler_gap_audit.curves import csv_write
from yolo_scheduler_trace.events import clip, intersection
from yolo_scheduler_trace_repair.events import validate_profile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/vta_callback_competition_01'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261111_vta_callback_competition'
PROGRAMS = ('vta_03_06', 'vta_09_12')


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def callback_addresses(assembly):
    """Track literal callback addresses passed in AArch64 x0 to indirect calls.

    This matches observed callback addresses to graph helpers. It does not
    classify an arbitrary indirect call as a parallel launch without its trace.
    """
    target = None
    result = []
    for line in assembly.splitlines():
        match = re.match(r'\s*([0-9a-f]+):\s+[0-9a-f]+\s+([a-z0-9.]+)\s*(.*)', line)
        if not match:
            continue
        pc, op, operands = match.groups()
        address = re.match(r'x0,\s*([0-9a-f]+)\b', operands)
        add = re.match(r'x0,\s*x0,\s*#(0x[0-9a-f]+|[0-9]+)\b', operands)
        if op in ('adrp', 'adr') and address:
            target = int(address[1], 16)
        elif op == 'add' and add:
            if target is not None:
                target += int(add[1], 0)
        elif op == 'blr':
            if target is not None:
                result.append((target, int(pc, 16)))
            target = None
        elif op == 'bl' or re.match(r'[wx]0(?:,|$)', operands):
            # Calls may clobber x0; never carry a stale address across them.
            target = None
    return result


def function_family(function):
    if 'leaky_relu' in function:
        return 'activation_and_fused_conversion'
    if 'max_pool' in function:
        return 'pooling_and_fused_conversion'
    if 'upsampling' in function or 'concatenate' in function:
        return 'upsampling_route_and_fused_conversion'
    if 'strided_slice' in function:
        return 'slice_and_layout'
    return 'layout_and_other_CPU_callback'


def match_frame(phase, launches, callbacks, mapping, expected):
    selected = sorted((event for event in launches
        if event['caller'] == phase['tid'] and
        phase['wall_start_ns'] <= event['wall_start_ns'] <= event['wall_end_ns'] <= phase['wall_end_ns']),
        key=lambda event: event['wall_start_ns'])
    if len(selected) != expected:
        raise ValueError('missing or extra VTA CPU launch')
    result = []
    previous_end = phase['wall_start_ns']
    for event in selected:
        workers = callbacks.get(event['launch'], [])
        if (event['tasks'] != 1 or len(workers) != 1 or workers[0]['task'] != 0
                or workers[0]['tasks'] != 1 or workers[0]['tid'] != phase['tid']
                or workers[0]['caller'] != phase['tid']):
            raise ValueError('VTA single-thread callback identity mismatch')
        worker = workers[0]
        if (worker['module'] != event['module'] or worker['function_offset'] != event['function_offset']
                or event['function_offset'] not in mapping):
            raise ValueError('callback module or function differs')
        if not previous_end <= event['wall_start_ns'] <= worker['wall_start_ns'] <= worker['wall_end_ns'] <= event['wall_end_ns']:
            raise ValueError('callback intervals overlap or are out of order')
        if worker['result'] != 0 or event['result'] != 0 or worker['cpu_end_ns'] < worker['cpu_start_ns']:
            raise ValueError('callback failed or CPU clock reversed')
        if worker['barrier_calls'] or worker['barrier_cpu_ns'] or worker['barrier_wall_ns']:
            raise ValueError('unexpected barrier inside single-thread CPU callback')
        result.append((event, worker, mapping[event['function_offset']]))
        previous_end = event['wall_end_ns']
    return result


class IntervalIndex:
    def __init__(self, rows):
        self.rows = sorted(rows, key=lambda row: row['start_ns'])
        if any(a['end_ns'] > b['start_ns'] for a, b in zip(self.rows, self.rows[1:])):
            raise ValueError('overlapping intervals cannot be summed as CPU work')
        self.ends = [row['end_ns'] for row in self.rows]

    def overlaps(self, start, end):
        index = bisect_right(self.ends, start)
        while index < len(self.rows) and self.rows[index]['start_ns'] < end:
            row = self.rows[index]
            yield row, max(start, row['start_ns']), min(end, row['end_ns'])
            index += 1


def split_on_CPU(start, end, host_index, callback_index):
    total = inside = 0
    families = defaultdict(int)
    for host, left, right in host_index.overlaps(start, end):
        total += right - left
        for callback, a, b in callback_index.overlaps(left, right):
            inside += b - a
            families[callback['family']] += b - a
    if not 0 <= inside <= total <= end - start:
        raise ValueError('on-CPU attribution does not conserve time')
    return dict(total_ns=total, CPU_callback_ns=inside,
        outside_CPU_callback_ns=total-inside, families=dict(families))


def run():
    sources = {}
    index_path = ROOT / 'artifacts/yolo_scheduler_trace_repair_01/analysis/RESULTS.json'
    old = read(index_path)
    sources[str(index_path)] = sha256(index_path)
    observed = []
    for path, expected in old['source_receipts'].items():
        if sha256(path) != expected:
            raise ValueError('accepted pipeline receipt changed')
        receipt = read(path)
        verify_inventory(receipt['raw'], receipt['files'])
        sources[path] = expected
        if receipt['task']['spec']['phase'] == 'observed':
            observed.append(receipt)
    if len(observed) != 1:
        raise ValueError('expected exactly one accepted repaired observation')
    receipt = observed[0]
    task = receipt['task']; raw = Path(receipt['raw'])
    capsule = ROOT / 'artifacts/yolo_scheduler_trace_repair_01/capsule'
    manifest = read(capsule / 'manifest.json')
    verify_inventory(capsule, manifest['files'])
    if (manifest['bundle_id'] != task['bundle_id'] or manifest['runtime_sha256'] != task['runtime_sha256']
            or manifest['phase_runner_sha256'] != task['executor_sha256']
            or manifest['adapter_sha256'] != task['adapter_sha256']):
        raise ValueError('trace binary identity differs from frozen capsule')
    sources[str(capsule / 'manifest.json')] = sha256(capsule / 'manifest.json')
    programs = read(capsule / 'plans/programs.json')
    affinity = read(raw / 'affinity.json')
    streams = validate_profile(read(raw / 'profile.json'), task, affinity)['streams']
    trace = lines(raw / 'wait_trace.jsonl'); phases = lines(raw / 'phase_trace.jsonl')
    launches = [row for row in trace if row['kind'] == 0]
    callbacks = defaultdict(list)
    for row in trace:
        if row['kind'] == 1:
            callbacks[row['launch']].append(row)
    OUT.mkdir(parents=True, exist_ok=True); REPORT.mkdir(parents=True, exist_ok=True)
    frame_rows = []; callback_rows = []; node_rows = []; stage_summaries = []
    callback_indexes = {}; VTA_slots = {}
    for program, expected in zip(PROGRAMS, (14, 8)):
        directory = capsule / 'programs' / program
        for name, info in programs[program]['files'].items():
            path = directory / name
            if sha256(path) != info['sha256']:
                raise ValueError('program or graph identity differs')
            sources[str(path)] = info['sha256']
        mapping = {}
        for node_index, node in enumerate(read(directory / 'graph.json')['nodes']):
            if node['op'] != 'tvm_op':
                continue
            function = node['attrs']['func_name']
            assembly = subprocess.run(['aarch64-linux-gnu-objdump', '-d',
                '--disassemble=' + function + '_compute_', str(directory / 'module.so')],
                text=True, capture_output=True, check=True, timeout=15).stdout
            for address, launch_site in callback_addresses(assembly):
                if address in mapping:
                    raise ValueError('ambiguous callback-to-helper association')
                mapping[address] = dict(node=node_index, function=function,
                    family=function_family(function), callback_offset=address, launch_site=launch_site)
        if len(mapping) != expected:
            raise ValueError('graph helper CPU callback coverage changed')
        VTA_slots[program] = next(i for i, stage in enumerate(task['stages']) if stage['program'] == program)
        slot = VTA_slots[program]; entry = affinity['stages'][slot]
        if task['stages'][slot]['threads'] != 1 or entry['active_runtime_threads'] != 1:
            raise ValueError('this audit requires the original one-thread VTA pool')
        selected = sorted((p for p in phases if p['stage'] == program and p['operation'] == 'run'), key=lambda p:p['call'])
        if [p['call'] for p in selected] != list(range(task['warmup'] + task['scored'] + task['tail'])):
            raise ValueError('VTA phase frames are incomplete')
        VTA_launches = [r for r in launches if r['module'].endswith('/programs/' + program + '/module.so')]
        if any(task['bundle_id'] not in r['module'] for r in VTA_launches):
            raise ValueError('callback path points to a different frozen bundle')
        sequence = None; intervals = []
        for phase in selected:
            if phase['tid'] != entry['caller_tid']:
                raise ValueError('phase caller is not the VTA host')
            matches = match_frame(phase, VTA_launches, callbacks, mapping, expected)
            current = [event['function_offset'] for event, worker, node in matches]
            if sequence is None:
                sequence = current
            if current != sequence or len(set(current)) != expected:
                raise ValueError('per-frame CPU callback sequence changed or duplicated')
            cpu = wall = 0
            for event, worker, node in matches:
                cpu_ns = worker['cpu_end_ns'] - worker['cpu_start_ns']
                wall_ns = worker['wall_end_ns'] - worker['wall_start_ns']
                cpu += cpu_ns; wall += wall_ns
                callback_rows.append(dict(program=program, frame=phase['call'], launch=event['launch'],
                    **node, CPU_ns=cpu_ns, wall_ns=wall_ns))
                intervals.append(dict(start_ns=worker['wall_start_ns'], end_ns=worker['wall_end_ns'], **node))
            run_cpu = phase['cpu_end_ns'] - phase['cpu_start_ns']
            if not 0 <= cpu <= run_cpu:
                raise ValueError('callback CPU exceeds entire VTA run CPU')
            frame_rows.append(dict(program=program, frame=phase['call'], callbacks=len(matches),
                callback_CPU_ms=cpu/1e6, callback_wall_ms=wall/1e6,
                run_CPU_ms=run_cpu/1e6, outside_callback_CPU_ms=(run_cpu-cpu)/1e6,
                run_wall_ms=(phase['wall_end_ns']-phase['wall_start_ns'])/1e6))
        if len(VTA_launches) != len(selected)*expected:
            raise ValueError('unassigned VTA CPU launch outside known phase')
        callback_indexes[program] = IntervalIndex(intervals)
        kept = [r for r in frame_rows if r['program'] == program and r['frame'] in old['complete_front_frames']]
        means = {key:st.mean(row[key] for row in kept) for key in
            ('callback_CPU_ms','callback_wall_ms','run_CPU_ms','outside_callback_CPU_ms','run_wall_ms')}
        stage_summaries.append(dict(program=program, frames=[r['frame'] for r in kept],
            total_session_CPU_launches=len(VTA_launches), CPU_launches_per_frame=expected,
            callback_fraction_of_run_CPU_percent=100*means['callback_CPU_ms']/means['run_CPU_ms'], **means))
        for address, node in mapping.items():
            values = [r for r in callback_rows if r['program']==program and
                r['frame'] in old['complete_front_frames'] and r['callback_offset']==address]
            node_rows.append(dict(program=program, **node, mean_CPU_ms=st.mean(r['CPU_ns'] for r in values)/1e6,
                mean_wall_ms=st.mean(r['wall_ns'] for r in values)/1e6,
                callback_family_is_semantic_group_not_individual_operator_attribution=True))
    coincidence = []
    front_slot = affinity['stages'][0]
    tids = {w['slot']:w['tid'] for w in front_slot['worker_slots']}
    host_indexes = {(program,core):IntervalIndex([r for r in streams[str(affinity['stages'][slot]['caller_tid'])]['on']
        if r['cpu']==core]) for program,slot in VTA_slots.items() for core in range(4)}
    for phase in phases:
        if phase['stage']!='cpu_00_02' or phase['operation']!='run' or phase['call'] not in old['complete_front_frames']:
            continue
        group = sorted((r for r in launches if r['caller']==phase['tid'] and
            phase['wall_start_ns']<=r['wall_start_ns']<=r['wall_end_ns']<=phase['wall_end_ns']), key=lambda r:r['wall_start_ns'])
        if len(group)!=12:
            raise ValueError('front CPU kernel sequence differs')
        for launch in group:
            workers = callbacks[launch['launch']]
            if sorted(w['task'] for w in workers)!=list(range(4)) or any(w['tid']!=tids[w['task']] for w in workers):
                raise ValueError('front CPU logical worker mapping differs')
            last = max(workers, key=lambda w:(w['wall_end_ns'],-w['task']))
            core = task['stages'][0]['worker_core_map'][last['task']]
            for part,start,end in [('entry',launch['wall_start_ns'],last['wall_start_ns']),
                                   ('body',last['wall_start_ns'],last['wall_end_ns'])]:
                window=clip(streams[str(last['tid'])],start,end)
                if not window['fully_covered']:
                    raise ValueError('critical CPU path lacks scheduler coverage')
                for program in PROGRAMS:
                    total=inside=0;families=defaultdict(int)
                    for off in window['off']:
                        if off['from_cpu']!=core or off['to_cpu']!=core:
                            raise ValueError('fixed critical worker unexpectedly migrated')
                        split=split_on_CPU(off['start_ns'],off['end_ns'],host_indexes[(program,core)],callback_indexes[program])
                        total+=split['total_ns'];inside+=split['CPU_callback_ns']
                        for name,ns in split['families'].items():families[name]+=ns
                    coincidence.append(dict(frame=phase['call'],launch=launch['launch'],part=part,program=program,
                        host_CPU_ns=total,callback_CPU_ns=inside,outside_callback_CPU_ns=total-inside,families=dict(families)))
    coincidence_summary=[]
    for part in ('entry','body'):
        for program in PROGRAMS:
            selected=[r for r in coincidence if r['part']==part and r['program']==program]
            n=len(old['complete_front_frames'])
            coincidence_summary.append(dict(part=part,program=program,
                **{key.replace('_ns','_ms_per_front_frame'):sum(r[key] for r in selected)/n/1e6
                   for key in ('host_CPU_ns','callback_CPU_ns','outside_callback_CPU_ns')},
                is_temporal_coincidence_not_causal_penalty=True))
    intrinsic_path=ROOT/'artifacts/cross_network_model_01/pool_temporal/INTRINSICS.json'
    intrinsics=read(intrinsic_path);sources[str(intrinsic_path)]=sha256(intrinsic_path)
    isolation_coverage=[]
    for program in PROGRAMS:
        intrinsic=intrinsics['VTA']['yolov3_tiny416/'+program+'/a0-1-2-3']
        reference=Path(intrinsic['observed_receipt']);r=read(reference);verify_inventory(r['raw'],r['files'])
        sources[str(reference)]=sha256(reference)
        if intrinsic['module_sha256']!=programs[program]['files']['module.so']['sha256']:
            raise ValueError('isolated VTA module differs from pipeline module')
        isolation_coverage.append(dict(program=program,receipt=str(reference),boot=r['boot_id'],
            wait_trace_exists=(Path(r['raw'])/'wait_trace.jsonl').exists(),
            device_trace_exists=(Path(r['raw'])/'device_trace.jsonl').exists(),
            exact_isolated_CPU_callback_cost_comparison_available=False))
    sources[str(Path(__file__))]=sha256(__file__)
    body_rows=[r for r in coincidence_summary if r['part']=='body']
    body_split={key:sum(r[key] for r in body_rows) for key in
        ('host_CPU_ms_per_front_frame','callback_CPU_ms_per_front_frame','outside_callback_CPU_ms_per_front_frame')}
    reclassified=sum(r['total_session_CPU_launches'] for r in stage_summaries)
    if reclassified!=1056:
        raise ValueError('frozen reclassification coverage differs')
    result=dict(source_receipts=sources,pipeline_boot=receipt['boot_id'],stage_summaries=stage_summaries,
        coincidence_summary=coincidence_summary,nodes=node_rows,isolated_coverage=isolation_coverage,
        critical_worker_body_coincident_host_split=body_split,
        VTA_callback_launches_reclassified=reclassified,previously_mixed_with_non_CPU_stage_launches=True,
        no_new_board_connection=True,new_native_sessions=0,new_DNN_inferences=0,
        no_static_cost_formula_DP_or_Top20_change=True,diagnostic_not_predictor=True,
        original_front_probe_quality_passed=old['probe_quality_passed'],
        VTA_specific_probe_disturbance_not_independently_measured=True,
        outside_callback_includes_driver_commands_buffers_and_other_work=True,
        callback_CPU_demand_was_already_in_total_host_cost=True)
    write(OUT/'RESULTS.json',result,replace=True)
    csv_write(OUT/'frames.csv',frame_rows);csv_write(OUT/'nodes.csv',node_rows)
    csv_write(OUT/'critical_coincidence.csv',coincidence)
    report=['已有VTA CPU回调的再归类：服务误差与共享竞争','',
        '只读取已经验收、修复缓冲截断后的旧轨迹；新增开发板连接、原生会话和DNN推理均为0。',
        '按原始模块路径、文件哈希、回调函数偏移、阶段调用线程及帧序号归类。',
        '图节点到匿名回调的对应由冻结AArch64计算helper实际传入的地址核验，不只凭函数名字猜测。',
        '全部48帧重复序列完整：VTA3–6每帧14次CPU启动，VTA9–12每帧8次，共1056次。',
        '以下均值取原前段调度覆盖完整的帧12–23；VTA与前段可能处于不同帧，不强行对齐成同一帧。','',
        '子图 / CPU回调核时ms / 整个run核时ms / 回调占比 / CPU回调外核时ms']
    for r in stage_summaries:
        report.append(f"{r['program']} / {r['callback_CPU_ms']:.3f} / {r['run_CPU_ms']:.3f} / {r['callback_fraction_of_run_CPU_percent']:.1f}% / {r['outside_callback_CPU_ms']:.3f}")
    report+=['','前段最后完成worker离开CPU期间，同核VTA主机的实际运行时间：',
        '片段 / VTA子图 / 主机CPU总ms / 处于CPU回调内ms / 回调外ms']
    for r in coincidence_summary:
        report.append(f"{r['part']} / {r['program']} / {r['host_CPU_ms_per_front_frame']:.3f} / {r['callback_CPU_ms_per_front_frame']:.3f} / {r['outside_callback_CPU_ms_per_front_frame']:.3f}")
    report+=['','结论与限制：',
        f"前段最后完成worker的回调体离开CPU时，同核VTA主机合计运行{body_split['host_CPU_ms_per_front_frame']:.3f} ms；其中CPU回调内{body_split['callback_CPU_ms_per_front_frame']:.3f} ms，回调外{body_split['outside_callback_CPU_ms_per_front_frame']:.3f} ms。",
        'VTA主机不是只有轮询：可以识别实际CPU激活、池化、布局及路由等融合计算回调。',
        '但这些回调只占本批VTA run CPU时间的一部分；不能把设备调用之外的全部CPU时间称为激活，也不能指望只改这些回调的线程数解决全部拖尾。',
        '剩余工作包括设备驱动、指令构造、缓冲处理和其他开销，本轨迹没有同时捕获所有底层边界，暂不进一步分摊。',
        '同核时间相遇证明具体资源重叠；并不证明移除这些工作后前段就会等量变快，也不能重复加到已有容量评分里。',
        '原模型已计入这部分CPU总核时。新证据是工作类型、具体时段及执行位置，不是发现以前将其填零。',
        '两个已有隔离VTA记录包含设备调用轨迹，但没有CPU回调等待轨迹；目前无法以匹配的逐节点隔离成本复原其精确CPU工作时序。',
        '本结果不改评分、DP、候选域或Top-20；不把流水观测回填为未测候选的通用成本。',
        '下一步优先补清VTA主机回调之外的可观测工作边界与现有时序输入如何分配CPU工作，先核对源码和隔离设备记录，不自动扩大板测。']
    (REPORT/'FINDINGS_zh.txt').write_text('\n'.join(report)+'\n')
    write(OUT/'MANIFEST.json',dict(files={str(p):sha256(p) for p in
        [OUT/'RESULTS.json',OUT/'frames.csv',OUT/'nodes.csv',OUT/'critical_coincidence.csv',REPORT/'FINDINGS_zh.txt']}),replace=True)
    print(json.dumps(dict(stages=stage_summaries,coincidence=coincidence_summary,
        report=str(REPORT/'FINDINGS_zh.txt'),new_DNN=0),ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    run()
