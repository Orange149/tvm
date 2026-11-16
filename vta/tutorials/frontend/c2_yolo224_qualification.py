#!/usr/bin/env python3
"""Independently check the actual board fault and interleaving witnesses."""
import argparse,json
from pathlib import Path
from c2_yolo224_prepare import write,sha


def read(path):return json.loads(path.read_text())


def reconcile(out):
    cases={}
    def result(label,invalid=False,error=None):
        directory=out/'runs'/label;r=read(directory/'result.json')
        code=int((directory/'exitcode').read_text())
        if invalid:
            if not code or r['status']!='INVALID' or (error and error not in r['error']):
                raise ValueError('wrong failure path: '+label)
        elif code or r['status']!='passed' or not r['all_free'] or not (directory/'verified.json').exists():
            raise ValueError('healthy qualification failed: '+label)
        cases[label]={'status':r['status'],'error':r['error'],'result_sha256':sha(directory/'result.json'),
            'native_completed':True,'safe_join_scope':'immutable native Run joins all workers before Save'}
        return r
    def events(r,name,edge=None,frame=None):
        return [x for x in r['events'] if x['event']==name and
            (edge is None or x['edge']==edge) and (frame is None or x['frame']==frame)]
    def event(r,name,edge=None,frame=None):
        values=events(r,name,edge,frame)
        if len(values)!=1:raise ValueError(f'expected exactly one {name}, edge={edge}, frame={frame}')
        return values[0]['time_ms']
    full=result('control_full')
    witness=event(full,'two_full_slots_and_blocked_producer_verified',0,0)
    if not (event(full,'producer_wait',0,2)<=witness<event(full,'input_lease_released',0,0)<=
            event(full,'producer_acquired',0,2)):
        raise ValueError('full/slow-consumer witness does not gate the third producer')
    before=result('control_before');after=result('control_after')
    if not (event(before,'producer_wait',0,1)<=event(before,'before_read_overwrite_blocked',0,0)<
            event(before,'last_reader_return',0,0)<=event(before,'input_lease_released',0,0)<=
            event(before,'producer_acquired',0,1)):
        raise ValueError('producer acquired live input before synchronous last-reader return')
    frames={r['frame_id']:r for r in after['frames']}
    order=[event(after,'last_reader_return',0,0),event(after,'input_lease_released',0,0),
        event(after,'producer_acquired',0,1),frames[1]['stage_begin_ms'][0],frames[1]['stage_end_ms'][0],
        event(after,'producer_published',0,1),event(after,'next_frame_fully_produced_before_consumer_tail',0,0),
        frames[0]['stage_end_ms'][1]]
    if any(a>b for a,b in zip(order,order[1:])):raise ValueError('controlled full-production overlap was not achieved')
    cases['control_after']['controlled_order_ms']=order
    for label,reason in (('control_stale','stale, duplicate or invalid lease transition'),
                         ('control_duplicate','input lease returned twice'),
                         ('control_incomplete','incomplete same-generation bundle cannot publish'),
                         ('control_abort','controlled abort')):
        r=result(label,True,reason)
        if label=='control_stale' and events(r,'consumer_acquired',0,0):raise ValueError('stale token reached consumer')
        if label=='control_duplicate' and len(events(r,'input_lease_released',0,0))!=1:raise ValueError('duplicate lease release succeeded')
        if label=='control_incomplete' and events(r,'producer_published',1,0):raise ValueError('incomplete dual-head bundle published')
        if label=='control_abort':
            if not events(r,'producer_wait',0,2) or not events(r,'two_full_slots_and_blocked_producer_verified',0,0):
                raise ValueError('abort did not exercise a real blocked producer')
    if (out/'runs/control_bad_capacity').exists():
        r=result('control_bad_capacity',True,'before submission')
        if r['runtime_profile']['driver_run_calls']!=0:raise ValueError('undersized backing reached hardware')
    health=result('health_after_faults')
    if len(health['frames'])!=3:raise ValueError('fresh executor health run is incomplete')
    from c2_yolo224_address_observer import verify
    verify(out,'address_observed_E1')
    data={'status':'passed','boot_id':health['boot_id'],'cases':cases,
        'observer_certificate_sha256':sha(out/'dynamic_address_certificate.json'),
        'generator_sha256':sha(Path(__file__)),
        'scope':'board witnesses plus saved byte-exact checks; separate from natural-overlap statistics'}
    target=out/'qualification_reconciliation.json'
    if not target.exists():write(target,data)
    elif read(target)!=data:raise ValueError('qualification reconciliation drift')
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=reconcile(a.out);print(json.dumps({'status':r['status'],'boot_id':r['boot_id'],'cases':len(r['cases'])}))
