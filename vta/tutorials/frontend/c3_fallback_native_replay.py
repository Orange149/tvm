"""Offline certificate regression on one saved completed call; no target API."""
import argparse
from pathlib import Path
import time
from c3_tps_sync_release_local import read,write,sha,require


def replay(unit, output):
    from c3_tps_window_native import certify_native
    from c3_valid_window import certify_trace
    from c3_five_tradeoff_commands import decode_submissions
    from c3_yolo416_fifo_certificate import certify_action_graph
    from c3_tps_window_progress import LANES
    from c3_official_window_workflow import native_certificate_for_storage
    from c3_fallback_physical import certify_physical
    output.mkdir(exist_ok=False)
    inputs={'events':unit/'prepare/R/events.json','dump':unit/'R_detailed/native_call_01.txt',
            'execution':unit/'R_detailed/call_01_execution.json','hardware':unit/'hardware_archive.json'}
    write(output/'inputs.json',{k:{'path':str(p),'sha256':sha(p)} for k,p in inputs.items()})
    events=read(inputs['events']);execution=read(inputs['execution']);dump=inputs['dump'].read_text()
    require(execution['helper_returned'] is True and execution['counts']==1,'SAVED_HELPER_COMPLETION_REQUIRED')
    t=time.monotonic();native=certify_native(events,dump,execution['submits'])
    write(output/'native_certificate.json',native_certificate_for_storage(native));print('native certified',flush=True)
    sem=certify_trace(events,ancestors=native['ancestors'])
    write(output/'semantic_summary.json',{k:v for k,v in sem.items() if k not in ('output_terms','consumer_versions')})
    print('native semantic certified',flush=True)
    records=decode_submissions(dump)[0];hw=read(inputs['hardware'])['fifo_depths']
    commands={i:hw[k] for i,k in enumerate(('load_queue','gemm_queue','store_queue'),1)}
    tokens=dict(zip(LANES,[hw[k] for k in ('l2g_queue','g2l_queue','g2s_queue','s2g_queue')]))
    progress=certify_action_graph(records,commands,tokens)
    require(progress['status']=='FINITE_FIFO_PROGRESS_PROVED','FINITE_FIFO_UNRESOLVED')
    write(output/'progress.json',progress);print('FIFO certified',flush=True)
    physical=certify_physical(records,native)
    write(output/'physical.json',physical);print('physical certified',flush=True)
    write(output/'result.json',{'status':'SAVED_CALL_NATIVE_FIFO_PHYSICAL_CERTIFIED','new_target_calls':0,
        'saved_completed_call_count':1,'qualification_calls_adopted':0,
        'analysis_seconds_not_performance':time.monotonic()-t})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('unit',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();replay(a.unit.resolve(),a.output.resolve())
