"""Observe the sealed replay's phase chain, then compare accepted native logs.

No scheduling rule, cost, rank or native session is changed. The observer must
reproduce the entire sealed replay before opening the native timing files.
Absolute clocks are never joined: each source is aligned to its own front run.
"""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cpu_vta_partitioning.common import read, write, sha256, digest, verify_inventory
from device_completion_replay import CompletionFinite
from phase_model.simulator import ceiling
from policy_branch_eval.common import isolated_inputs_only
from scheduler_gap_audit.curves import csv_write

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/interframe_phase_chain_01'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261112_interframe_phase_chain'


def phase_index(rows, names, frames):
    """Validate the complete set/run/get identity and per-stage frame order."""
    expected = {(name, frame, op) for name in names for frame in frames
                for op in ('set', 'run', 'get')}
    result = {}
    for row in rows:
        if row['stage'] not in names or row['call'] not in frames:
            continue
        key = row['stage'], row['call'], row['operation']
        if key not in expected or key in result:
            raise ValueError('duplicate or unexpected stage phase')
        if not 0 <= row['wall_start_ns'] <= row['wall_end_ns']:
            raise ValueError('invalid phase extent')
        result[key] = row
    if set(result) != expected:
        raise ValueError('missing stage phase')
    for name in names:
        last = None
        for frame in sorted(frames):
            parts = [result[name, frame, op] for op in ('set', 'run', 'get')]
            for before, after in zip(parts, parts[1:]):
                if before['wall_end_ns'] > after['wall_start_ns']:
                    raise ValueError('phase order overlaps')
            if last is not None and last > parts[0]['wall_start_ns']:
                raise ValueError('stage frames overlap or run backwards')
            last = parts[-1]['wall_end_ns']
    return result


def phase_chain(index, names, frame):
    """A telescoping same-clock path; no sums of concurrent thread waits."""
    rows = []
    for i, name in enumerate(names):
        begin = index[name, frame, 'set']['wall_start_ns']
        run = index[name, frame, 'run']
        end = index[name, frame, 'get']['wall_end_ns']
        previous = None if i == 0 else index[names[i-1], frame, 'get']['wall_end_ns']
        if previous is not None and begin < previous:
            raise ValueError('stage begins before its predecessor output')
        rows.append(dict(stage=name, frame=frame, start_ns=begin,
                         run_start_ns=run['wall_start_ns'], run_end_ns=run['wall_end_ns'],
                         end_ns=end, service_ns=end-begin,
                         run_ns=run['wall_end_ns']-run['wall_start_ns'],
                         predecessor_to_start_ns=0 if previous is None else begin-previous))
    elapsed = rows[-1]['end_ns']-rows[0]['start_ns']
    if sum(r['service_ns']+r['predecessor_to_start_ns'] for r in rows) != elapsed:
        raise ValueError('phase-chain conservation failed')
    return rows


def relative_pair(index, names, frame, offset=-1):
    """Compare phases within each source; never align absolute native/model clocks."""
    anchor = index[names[0], frame, 'run']['wall_start_ns']
    return [dict(stage=name, frame=frame+offset,
                 relative_run_start_ns=index[name, frame+offset, 'run']['wall_start_ns']-anchor,
                 relative_run_end_ns=index[name, frame+offset, 'run']['wall_end_ns']-anchor)
            for name in names]


class PhaseRecorder(CompletionFinite):
    """Generator observation only; no extra tasks, clock reads or event boundaries."""
    def __init__(self, *args, **kwargs):
        self.phase_rows = []
        self.queue_exposure = defaultdict(Fraction)
        self.queue_maximum = defaultdict(int)
        super().__init__(*args, **kwargs)

    def actions(self, stage):
        index = next(i for i, value in enumerate(self.stages) if value is stage)
        frame = self.state[index]['frame']
        phase, begin = 'set', self.time
        for action in super().actions(stage):
            if action[0] == 'cpu':
                op = action[3]['operation']
                if phase == 'set' and op != 'set':
                    self._phase(index, frame, phase, begin, self.time)
                    phase, begin = 'run', self.time
                if phase == 'run' and op == 'get':
                    self._phase(index, frame, phase, begin, self.time)
                    phase, begin = 'get', self.time
            yield action
        if phase != 'get':
            raise ValueError('observer failed to identify a full stage')
        self._phase(index, frame, phase, begin, self.time)

    def _phase(self, stage, frame, operation, begin, end):
        self.phase_rows.append(dict(stage=stage, call=frame, operation=operation,
                                   wall_start_ns=ceiling(begin), wall_end_ns=ceiling(end)))


class QueueRecorder:
    """Delegates every scheduler operation, integrating existing queue state."""
    def __init__(self, replay):
        self.replay, self.delegate = replay, replay.scheduler

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def consume(self, tasks, duration):
        r = self.replay
        for i, stage in enumerate(r.stages):
            r.queue_maximum[i] = max(r.queue_maximum[i], len(r.queues[i]))
            waiting = (stage['device'] == 'vta' and r.state[i] is None
                       and r.pending_output[i] is None and r.queues[i]
                       and r.device_owner is not None)
            if waiting:
                r.queue_exposure[i, 'queued_while_other_VTA_holds_lock'] += duration
                if len(r.queues[i]) == r.depth:
                    r.queue_exposure[i, 'queue_full_while_waiting_for_lock'] += duration
                    if i > 0 and r.pending_output[i-1] is not None:
                        r.queue_exposure[i, 'predecessor_output_blocked_by_locked_full_queue'] += duration
        return self.delegate.consume(tasks, duration)


def native_receipt():
    accepted_path = ROOT / 'artifacts/yolo_switch_without_queries_01/analysis/RESULTS.json'
    manifest = read(accepted_path.parent / 'MANIFEST.json')
    for path, expected in manifest['files'].items():
        if sha256(path) != expected:
            raise ValueError('accepted counter-free evidence changed')
    accepted = read(accepted_path)
    receipts = []
    for path, expected in accepted['sources'].items():
        if sha256(path) != expected:
            raise ValueError('native source receipt changed')
        if '/accepted/' in path:
            item = read(path)
            if item['task']['spec']['phase'] == 'observed':
                receipts.append((path, item))
    if len(receipts) != 1:
        raise ValueError('native diagnostic source ambiguous')
    return accepted_path, accepted, receipts[0]


def main():
    if OUT.exists() or REPORT.exists():
        raise FileExistsError('preserve the frozen phase audit')
    parent = ROOT / 'artifacts/current_model_gap_01'
    plan, inputs = read(parent / 'PLAN.json'), read(parent / 'INPUTS.json')
    if sha256(parent / 'INPUTS.json') != plan['input_sha256']:
        raise ValueError('isolated replay input changed')
    sources = {str(parent / 'INPUTS.json'): sha256(parent / 'INPUTS.json')}
    for path, expected in plan['original_model_sources'].items():
        if sha256(path) != expected:
            raise ValueError('frozen replay source changed')
        sources[path] = expected
    sealed_path = parent / 'independent.json'
    if sha256(sealed_path) != read(parent / 'REPLAY_FREEZE.json')['files'][sealed_path.name]:
        raise ValueError('sealed replay changed')
    sealed = read(sealed_path)['result']
    started = time.monotonic()
    with isolated_inputs_only():
        replay = PhaseRecorder(inputs['case']['stages'], quanta=inputs['quanta_ns'],
            placement=inputs['placement'], completion_model='independent',
            completion_point=inputs['point'], runnable_completion_wait=True, controls=None,
            gate=inputs['case']['configuration']['execution_mode'], host_order='before', **inputs['window'])
        replay.scheduler = QueueRecorder(replay)
        result = replay.run()
    if digest(result) != digest(sealed):
        raise ValueError('phase observation changed sealed replay')
    OUT.mkdir(parents=True)
    sources[str(sealed_path)] = sha256(sealed_path)
    names = [s['program'] for s in inputs['case']['configuration']['stages']]
    model_rows = [{**r, 'stage': names[r['stage']]} for r in replay.phase_rows]
    write(OUT / 'MODEL_FREEZE.json', dict(exact_sealed_result_reproduced=True,
        original_sha256=sha256(sealed_path), original_result_digest=digest(sealed),
        observed_result_digest=digest(result), model_phases=model_rows,
        queue_maximum=dict(replay.queue_maximum),
        queue_exposure_ns=[dict(stage=names[i], kind=kind, duration_ns=ceiling(v))
                           for (i, kind), v in sorted(replay.queue_exposure.items())],
        replay_seconds=time.monotonic()-started, no_native_timing_read_before_freeze=True))
    print(json.dumps({'sealed_replay_unchanged': True, 'model_phase_rows': len(model_rows)}), flush=True)

    # Native timing is only opened after exact replay verification and freeze.
    accepted_path, accepted, (receipt_path, receipt) = native_receipt()
    raw = Path(receipt['raw'])
    verify_inventory(raw, receipt['files'])
    if read(raw / 'queue_status.json')['reads']:
        raise ValueError('unexpected CPU delay counter query overhead')
    rows = [json.loads(s) for s in (raw / 'phase_trace.jsonl').read_text().splitlines()]
    count = receipt['task']['warmup']+receipt['task']['scored']+receipt['task']['tail']
    frames = list(range(count))
    native = phase_index(rows, names, frames)
    model = phase_index(model_rows, names, list(range(inputs['window']['frames'])))
    chain_rows, relative_rows = [], []
    for source, index in (('native_observed', native), ('sealed_model', model)):
        for frame in frames:
            chain_rows.extend(dict(source=source, **r) for r in phase_chain(index, names, frame))
        for frame in accepted['complete_front_frames']:
            relative_rows.extend(dict(source=source, anchor_frame=frame, **r)
                                 for r in relative_pair(index, names, frame))
    summary = []
    for name in names:
        item = dict(stage=name)
        for source in ('native_observed', 'sealed_model'):
            chosen = [r for r in relative_rows if r['source']==source and r['stage']==name]
            for field in ('relative_run_start_ns', 'relative_run_end_ns'):
                item[source+'_'+field] = sum(r[field] for r in chosen)/len(chosen)
        summary.append(item)

    build_path = ROOT / 'artifacts/yolo_scheduler_trace_repair_01/build/BUILD.json'
    build = read(build_path)
    source_path = next(Path(p) for p in build['source_files'] if p.endswith('vta_stage_pipeline_runner_phase.cc'))
    if sha256(source_path) != build['source_files'][str(source_path)]:
        raise ValueError('native source snapshot changed')
    text = source_path.read_text()
    pop = text.index('while (queues[stage_index]->Pop(&frame))')
    lock = text.index('std::unique_lock<std::mutex> lock(vta_run_mutex);', pop)
    run = text.index('frame->stage_outputs[stage_index] = RunStage(', lock)
    if not pop < lock < run:
        raise ValueError('native queue/lock source order changed')
    sources.update({str(accepted_path): sha256(accepted_path), receipt_path: sha256(receipt_path),
                    str(build_path): sha256(build_path), str(source_path): sha256(source_path)})
    output = dict(configuration=receipt['task']['configuration'], native_boot=receipt['boot_id'] if 'boot_id' in receipt else read(raw/'boot_id'),
        sources=sources, compared_frames=accepted['complete_front_frames'],
        native_frames_audited=frames, relative_previous_frame_summary=summary,
        source_semantics=dict(native_pop_before_VTA_lock=True, model_pop_after_VTA_lock=True,
            native_source_lines={k:text[:p].count('\n')+1 for k,p in (('pop',pop),('lock',lock),('run',run))},
            no_evidence_yet_this_difference_caused_stage_error=True),
        diagnostic_only=True, same_absolute_clock_not_assumed=True,
        no_new_predictor=True, no_DP_or_Top20_change=True,
        new_native_sessions=0, new_DNN_inferences=0, offline_replays=1)
    write(OUT/'RESULTS.json', output)
    csv_write(OUT/'phase_chain.csv', chain_rows)
    csv_write(OUT/'previous_frame_relative.csv', relative_rows)
    REPORT.mkdir(parents=True)
    findings = [
        '本轮为离线阶段时序核查；新增板测0，新增推理0，评分和DP均未修改。',
        '观察器完整恢复冻结回放，所有返回结果及时间戳的摘要一致。',
        '原生轨迹只在回放冻结后读取；两套时钟各自以当前帧前段run起点为零，不拼接绝对时钟。',
        '同一帧沿图依赖分解：阶段set到get服务＋前驱get结束到本阶段set开始的间隙；其和严格等于该帧经过整图的时长。',
        '间隙可能包括排队、互斥和线程唤醒，不能仅凭此表将其称为调度等待。',
        '源码差别：真实执行器先Pop，再申请VTA锁；旧回放等锁空闲后才Pop。队列暴露量详见MODEL_FREEZE.json；不能仅凭源码不同就认定它解释了误差。',
        '以下为上一帧各阶段相对于当前前段run起点的位置，单位ms：',
    ]
    for row in summary:
        findings.append(row['stage']+': 原生 %.3f～%.3f；模型 %.3f～%.3f' % tuple(row[k]/1e6 for k in (
            'native_observed_relative_run_start_ns','native_observed_relative_run_end_ns',
            'sealed_model_relative_run_start_ns','sealed_model_relative_run_end_ns')))
    findings.extend([
        '这是已知配置上的诊断；不能用原生相位或等待量回填静态成本。',
        '当前模型服务和候选II不同：此处前段run回放偏差不能直接当成最强完整模型的II偏差。',
        '仍需结合首帧到稳态的逐帧链，区分“相位错位造成慢化”与“前段本来偏快导致后续相位错位”，不能从相关性推出因果。',
    ])
    (REPORT/'FINDINGS_zh.txt').write_text('\n'.join(findings)+'\n')
    write(OUT/'MANIFEST.json', dict(source_sha256=sha256(__file__),
        files={str(p):sha256(p) for p in [*OUT.iterdir(), REPORT/'FINDINGS_zh.txt'] if p.is_file()},
        source_files=sources, no_board_connection=True))
    print(json.dumps(dict(report=str(REPORT/'FINDINGS_zh.txt'), phase_chain_rows=len(chain_rows),
                         relative_rows=len(relative_rows)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
