"""Revoke whole-window attribution without rewriting any original receipt."""
from collections import Counter, defaultdict
from pathlib import Path
from cpu_vta_partitioning.common import read, sha256, verify_inventory, write
from scheduler_gap_audit.curves import csv_write
from yolo_scheduler_trace.common import OUTPUT as ORIGINAL, RUN, ROOT
from yolo_scheduler_trace.events import Attribution, clip
from yolo_scheduler_trace.prepare import validate

OUTPUT = ROOT / 'artifacts/yolo_scheduler_trace_repair_01'
KERNEL_SOURCE = 'https://github.com/torvalds/linux/blob/v5.4/kernel/events/ring_buffer.c#L171-L246'


def capacity_row(stream, decoded):
    used = stream['head'] - stream['tail']
    capacity = stream['data_size']
    if used != len(bytes.fromhex(stream['records_hex'])) or not 0 <= used <= capacity:
        raise ValueError('invalid original ring extent')
    # The write-forward ring leaves one byte empty. With 32-byte switch records,
    # capacity-32 bytes is already full. No subsequent successful write means
    # pending PERF_RECORD_LOST need never appear in the saved bytes.
    suspect = capacity - used <= 64
    return dict(tid=stream['tid'],role=decoded['role']['role'],events=len(decoded['events']),
                capacity_bytes=capacity,retained_bytes=used,remaining_bytes=capacity-used,
                capacity_suspect=suspect,coverage_ns=decoded['coverage_ns'],
                trailing_unknown_ns=stream['disable_end_ns']-decoded['coverage_ns'][1]
                    if decoded['coverage_ns'] else None)


def audit(output=OUTPUT):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    validate(ORIGINAL)
    delivery = read(ORIGINAL/'DELIVERY.json')
    result_path = ORIGINAL/'analysis/RESULTS.json'
    if sha256(result_path) != delivery['results_sha256']:
        raise ValueError('original analysis identity differs')
    for p,h in delivery['files'].items():
        if sha256(p) != h: raise ValueError('original report changed')
    r = read(result_path); observed = None
    for p,h in r['source_receipts'].items():
        if sha256(p) != h: raise ValueError('original receipt changed')
        item = read(p); verify_inventory(item['raw'], item['files'])
        if item['task']['spec']['phase'] == 'observed': observed = item
    if observed is None: raise ValueError('observed receipt missing')
    raw_path = Path(observed['raw'])/'profile.json'
    profile = read(raw_path); streams = r['observed']['scheduler']['streams']
    rows = [capacity_row(s,streams[str(s['tid'])]) for s in profile['streams']]
    if not all(s['coverage_ns'] for s in streams.values()):
        raise ValueError('common prefix unavailable')
    a = max(s['coverage_ns'][0] for s in streams.values())
    b = min(s['coverage_ns'][1] for s in streams.values())
    frames=[]
    for f in range(8,24):
        workers = [w for w in r['callback_rows'] if w['frame']==f]
        tails = [t for t in r['tail_rows'] if t['frame']==f]
        if len(workers)==48 and len(tails)==12 and all(
            a<=w['callback_start_ns']<=w['callback_end_ns']<=b for w in workers
        ) and all(a<=t['start_ns']<=t['end_ns']<=b for t in tails): frames.append(f)
    # Every target TID, not only the critical worker, must have coverage over
    # every interval attributed to competing actors.
    attribution = Attribution(streams); critical=[]; totals=defaultdict(int)
    for w in r['critical_rows']:
        if w['frame'] not in frames: continue
        body = clip(streams[str(w['tid'])],w['callback_start_ns'],w['callback_end_ns'])
        if not body['fully_covered']: raise ValueError('prefix callback is incomplete')
        roles=defaultdict(int); unknown=0
        for off in body['off']:
            if not off['preempt']: continue
            x=attribution.other_work(off,w['slot']); unknown+=x['unobserved_ns']
            for role,ns in x['by_role_ns'].items(): roles[role]+=ns; totals[role]+=ns
        critical.append(dict(frame=w['frame'],kernel=w['kernel'],last_slot=w['slot'],
            callback_wall_ns=w['wall_ns'],on_CPU_ns=body['on_CPU_ns'],
            preempt_off_ns=body['preempt_off_ns'],nonpreempt_off_ns=body['nonpreempt_off_ns'],
            TASKSTATS_queue_lower_ns=w['queue_lower_ns'],TASKSTATS_queue_upper_ns=w['queue_upper_ns'],
            coincident_by_role_ns=dict(roles),unobserved_ns=unknown))
    correction = dict(status='WHOLE_WINDOW_ATTRIBUTION_REVOKED',
        original_delivery_status_means_sessions_finished_only=True,
        native_output_and_timing_valid=True,full_window_scheduler_trace_valid=False,
        original_quantitative_attribution_flag_superseded=True,
        original_receipts_and_reports_unchanged=True,
        original_results_sha256=sha256(result_path),raw_profile_sha256=sha256(raw_path),
        raw_profile=str(raw_path),kernel_source=KERNEL_SOURCE,
        kernel_statement='write-forward space failure increments pending lost count; LOST is emitted only with a later successful reservation',
        supplier_kernel_exact_source_verified=False,streams=rows,
        capacity_suspect_TIDs=[x['tid'] for x in rows if x['capacity_suspect']],
        common_interior_ns=[a,b],common_interior_duration_ns=max(0,b-a),
        fully_covered_front_frames=frames,prefix_critical_callbacks=critical,
        prefix_coincident_CPU_ns=dict(totals),
        prefix_is_exploratory_not_representative=True,
        no_full_window_role_percentage_or_aggregate_from_biased_subset=True,
        performance={key:{k:r[key][k] for k in ('fps','ii_ms','stage_service_ms')}
                     for key in ('clean','observed','bridge')},
        previous_native_sessions=5,previous_DNN_inferences=146,
        additional_native_sessions=0,additional_DNN_inferences=0,no_board_connection=True)
    write(output/'DATA_VERDICT.json',correction,replace=True)
    csv_write(output/'ring_capacity.csv',rows)
    csv_write(output/'prefix_critical_callbacks.csv',critical)
    # Add an explicit correction alongside the frozen report, without replacing
    # an originally hashed file or relabelling accepted raw data.
    write(ORIGINAL/'analysis/CAPACITY_CORRECTION.json',correction,replace=True)
    report = ROOT.parents[4]/'paper/my_paper/analysis/20261111_yolo_scheduler_trace'
    write(report/'CAPACITY_CORRECTION.json',correction,replace=True)
    return dict(verdict=correction['status'],suspect_streams=len(correction['capacity_suspect_TIDs']),
                common_interior_ms=(b-a)/1e6,complete_prefix_frames=frames,
                correction=str(output/'DATA_VERDICT.json'))
