"""Post-freeze comparison of model first service and corrected callback traces.

These observations never enter a predictor. Callback entry is not the kernel's
first scheduler service, so its envelope is kept distinct from model readiness.
"""
from collections import Counter
from fractions import Fraction
from pathlib import Path
from statistics import mean
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cpu_vta_partitioning.common import digest, read, sha256, write
from scheduler_gap_audit.curves import csv_write

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'artifacts/yield_semantics_01/alignment'


def fraction(value):
    return Fraction(value['numerator'], value['denominator'])


def summarize_model(result, frames):
    joins = [r for r in result['kernel_joins'] if r['stage'] == 0 and r['frame'] in frames]
    keys = {(r['frame'], r['kernel'], r['generation']) for r in joins}
    expected = {(frame, kernel, 0) for frame in frames for kernel in range(12)}
    if keys != expected or len(joins) != len(expected):
        raise ValueError('model and native callback populations differ')
    workers = [r for r in result['tasks'] if r['role'] == 'stage' and r['stage'] == 0
               and r['operation'] == 'worker_compute' and r['frame'] in frames]
    # A barrier-free callback contributes one task per logical slot.
    indexed = {(r['frame'], r['kernel'], r['slot']): r for r in workers}
    if len(indexed) != 4*len(expected) or len(workers) != len(indexed):
        raise ValueError('model callback accounting incomplete')
    rows = []
    for kernel in range(12):
        group = [r for r in joins if r['kernel'] == kernel]
        last = Counter()
        for row in group:
            for slot in row['last_slots']:
                last[str(slot)] += Fraction(1, len(row['last_slots']))
        tails = [r for r in result['tasks'] if r['role'] == 'stage' and r['stage'] == 0
                 and r['frame'] in frames and r['operation'] == 'completion_wait' and r['kernel'] == kernel]
        rows.append(dict(kernel=kernel, calls=len(group),
            first_service_skew_ms=mean(r['first_service_skew_ns'] for r in group)/1e6,
            latest_first_service_from_ready_ms=mean(max(indexed[r['frame'], kernel, slot]['first_service_ns']
                    for slot in range(4))-r['ready_ns'] for r in group)/1e6,
            join_span_ms=mean(r['join_ns']-r['ready_ns'] for r in group)/1e6,
            caller_last_fraction=float(last['0']/len(group)),
            last_slot_fractional_counts={k: float(v) for k,v in sorted(last.items())},
            caller_wait_CPU_ms=float(sum((fraction(r['CPU']) for r in tails),Fraction(0))/len(group)/1_000_000)))
    return rows


def summarize_native(trace):
    frames = trace['complete_front_frames']
    if frames != list(range(12,24)) or trace['probe_quality_passed'] is not True:
        raise ValueError('corrected trace quality or complete-frame population differs')
    rows = []
    for kernel in range(12):
        workers = [r for r in trace['worker_rows'] if r['kernel'] == kernel]
        if len(workers) != 4*len(frames):
            raise ValueError('corrected native callback population incomplete')
        by_frame = {frame: [r for r in workers if r['frame'] == frame] for frame in frames}
        if any(sorted(r['slot'] for r in v) != list(range(4)) for v in by_frame.values()):
            raise ValueError('corrected native logical slots incomplete')
        last = Counter(str(r['slot']) for r in workers if r['last'])
        if sum(last.values()) != len(frames):
            raise ValueError('last native callback is ambiguous')
        tails = [r for r in trace['tail_rows'] if r['kernel'] == kernel]
        rows.append(dict(kernel=kernel, calls=len(frames),
            callback_entry_skew_ms=mean(max(r['callback_start_ns'] for r in v)-min(r['callback_start_ns'] for r in v)
                                        for v in by_frame.values())/1e6,
            latest_callback_entry_from_launch_ms=mean(max(r['launch_to_body_ns'] for r in v) for v in by_frame.values())/1e6,
            launch_to_last_callback_end_ms=mean(max(r['launch_to_body_ns']+r['wall_ns'] for r in v)
                                                for v in by_frame.values())/1e6,
            caller_last_fraction=last['0']/len(frames), last_slot_counts=dict(last),
            caller_tail_CPU_ms=mean(r['on_CPU_ns'] for r in tails)/1e6,
            entry_envelope_includes_observer_and_dispatch=True,
            caller_tail_includes_wait_return_and_probe=True))
    return frames, rows


def run():
    dev = ROOT / 'artifacts/finite_service_01/development'
    seal = read(dev / 'REPLAY_FREEZE.json')
    for path, expected in seal['sources'].items():
        if sha256(path) != expected:
            raise ValueError('sealed model implementation changed')
    path = ROOT / 'artifacts/yolo_scheduler_trace_repair_01/analysis/RESULTS.json'
    delivery = read(path.parent.parent / 'DELIVERY.json')
    if sha256(path) != delivery['results_sha256']:
        raise ValueError('corrected native reference changed')
    trace = read(path)
    validation = read(path.parent.parent / 'FINAL_VALIDATION.json')
    if validation['status'] != 'PASSED' or not validation['attribution_conservation_passed']:
        raise ValueError('corrected trace validity unproven')
    frames, native = summarize_native(trace)
    rows = []
    evidence = {str(path): sha256(path)}
    summaries = []
    for name, expected in seal['files'].items():
        file = dev / name
        if sha256(file) != expected:
            raise ValueError('sealed replay changed')
        item = read(file)
        if item['content_sha256'] != digest({k:v for k,v in item.items() if k != 'content_sha256'}):
            raise ValueError('sealed replay content differs')
        spec = item['spec']
        if spec['case'] != 'yolov3_tiny416/overlap':
            continue
        projected = item['result']
        if not projected['tasks']:
            # The old fluid runner intentionally suppressed task storage. Add
            # observation only and require every frozen output timestamp and
            # service statistic to remain exactly identical.
            from static_revision.trace_replay import TraceReplay
            inputs = read(dev / 'INPUTS.json')
            case = inputs['isolated']['cases'][spec['case']]
            projected = TraceReplay(case['stages'], placement=spec['placement'],
                runnable_completion_wait=True, controls=None,
                gate=case['configuration']['execution_mode'], host_order='before',
                **inputs['windows'][spec['case']]).run()
            if projected['replay'] != item['result']['replay']:
                raise ValueError('observation-only fluid replay changed frozen timing')
        model = summarize_model(projected,frames)
        label = spec['mode'] + '/' + spec['placement']
        for a,b in zip(model,native):
            rows.append(dict(model=label, **a, native_callback_entry_skew_ms=b['callback_entry_skew_ms'],
                native_latest_callback_entry_from_launch_ms=b['latest_callback_entry_from_launch_ms'],
                native_launch_to_last_callback_end_ms=b['launch_to_last_callback_end_ms'],
                native_caller_last_fraction=b['caller_last_fraction'], native_caller_tail_CPU_ms=b['caller_tail_CPU_ms']))
        summaries.append(dict(model=label, matched_frames=len(frames), matched_launches=12*len(frames),
            mean_caller_last_fraction=mean(x['caller_last_fraction'] for x in model),
            sum_mean_model_first_service_skew_ms=sum(x['first_service_skew_ms'] for x in model),
            sum_mean_model_latest_service_from_ready_ms=sum(x['latest_first_service_from_ready_ms'] for x in model),
            model_caller_wait_CPU_ms_per_frame=sum(x['caller_wait_CPU_ms'] for x in model),
            native_caller_last_fraction=mean(x['caller_last_fraction'] for x in native),
            native_sum_callback_entry_skew_ms=sum(x['callback_entry_skew_ms'] for x in native),
            native_caller_tail_CPU_ms_per_frame=sum(x['caller_tail_CPU_ms'] for x in native),
            clock_envelopes_are_distinct=True))
        evidence[str(file)] = expected
    if len(summaries) != 5:
        raise ValueError('the five sealed scheduling comparisons are required')
    DEST.mkdir(parents=True,exist_ok=True)
    result = dict(summaries=summaries, per_kernel=rows, source_files=evidence,
        native_boot=trace['boot_id'], platform_boot=read(ROOT/'artifacts/finite_service_01/PLATFORM_RECEIPT.json')['boot_id'],
        native_trace_is_development_not_independent_validation=True, new_DNN_inferences=0,
        known_probe_disturbance_ms=trace['probe_front_service_disturbance_ms'],
        probe_quality_allows_selected_frame_diagnostic_only=True,
        waiting_CPU_and_worker_wait_not_added_to_static_cost=True,
        model_source_sha256=sha256(__file__))
    write(DEST/'RESULTS.json',result)
    csv_write(DEST/'matched_kernel_alignment.csv',rows)
    csv_write(DEST/'model_completion_summary.csv',summaries)
    return summaries


if __name__ == '__main__':
    print(json.dumps(run(),ensure_ascii=False,indent=2))
