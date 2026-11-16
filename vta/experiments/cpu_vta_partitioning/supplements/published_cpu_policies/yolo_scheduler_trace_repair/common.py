from pathlib import Path
from cpu_vta_partitioning.common import sha256
from yolo_scheduler_trace.common import ROOT, CONFIG, CAPABILITY, PARENT, freeze
from yolo_scheduler_trace.common import OUTPUT as ORIGINAL, protocol as previous_protocol

OUTPUT=ROOT/'artifacts/yolo_scheduler_trace_repair_01'
RUN=ROOT/'runs/yolo_scheduler_trace_repair_01'


def protocol():
    p=previous_protocol()
    p.update(experiment='yolo_scheduler_trace_repair_01',revision=1,
        max_native_sessions=3,max_DNN_inferences=97,
        drain_policy='existing collector after each completed frame; retain in memory; write only at stop',
        per_TID_retained_limit_bytes=4*1024*1024,near_full_guard_bytes=64,
        new_sampling_thread=False,new_periodic_timer=False,
        previous_batch_whole_window_attribution_revoked=True)
    return p


def specs():
    return [dict(network='yolov3_tiny416',scope='pipeline',case='overlap',phase=p,runner='phase',probe=p=='observed')
            for p in ('qualify','clean','observed')]


def sources():
    from yolo_scheduler_trace.common import sources as old_sources
    files=[ROOT/'run_yolo_scheduler_trace_repair.py',*sorted(Path(__file__).parent.glob('*.py')),
           *sorted(Path(__file__).parent.glob('*.h')),ROOT/'tests/test_yolo_scheduler_trace_repair.py']
    return {**old_sources(),**{str(p):sha256(p) for p in files}}
