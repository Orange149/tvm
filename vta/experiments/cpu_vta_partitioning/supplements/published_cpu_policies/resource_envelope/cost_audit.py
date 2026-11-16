"""Recover inherited profiling cost from source task IDs and timestamped events."""
from collections import Counter
from datetime import datetime
from pathlib import Path
from run_resource_behavior import ROOT,SOURCE,api

def audit():
    c=api();costs=c.read(SOURCE/'isolated_costs.json')['costs'];ddr=c.read(SOURCE/'ddr.json')
    needed={};calls=Counter()
    for value in costs.values():
        for role in ('qualification','measurement'):
            proof=value[role];doc=proof['document'];key=proof['boot_id'],doc['task_id']
            needed[key]=doc;calls[role]+=sum(doc[k] for k in ('warmup','scored','tail'))
    for value in ddr['traffic_sources'].values():
        proof=value['measurement'];doc=proof['document'];needed[(proof['boot_id'],doc['task_id'])]=doc
        calls['DDR_traffic']+=sum(doc[k] for k in ('warmup','scored','tail'))
    intervals={};sources=[]
    for run in sorted((ROOT/'runs').glob('*')):
        path=run/'events.jsonl'
        if not path.is_file():continue
        start={};matched=0
        for line in path.read_text().splitlines():
            row=__import__('json').loads(line);key=row.get('boot_id'),row.get('task_id')
            if key not in needed:continue
            when=datetime.fromisoformat(row['utc'])
            if row['event']=='session_start':start[key]=when
            elif row['event']=='session_accepted' and key in start:
                elapsed=(when-start.pop(key)).total_seconds()
                if elapsed<0:raise ValueError('source event time reversed')
                intervals[key]=elapsed;matched+=1
        if matched:sources.append({'events_path':str(path),'sha256':c.sha256(path),'matched_sessions':matched})
    missing=sorted(set(needed)-set(intervals))
    return {'isolated_signatures':len(costs),'DDR_traffic_signatures':len(ddr['traffic_sources']),
        'source_native_sessions':len(needed),'source_stage_calls':dict(calls),
        'accepted_session_and_collection_seconds':sum(intervals[k] for k in needed if k in intervals),
        'matched_event_sessions':len(set(needed)&set(intervals)),
        'missing_event_sessions':len(missing),'sources':sources,
        'scope':'accepted source sessions and collection only; excludes deployment, search, offline preparation, failed attempts and user idle time',
        'DDR_bandwidth_freshly_calibrated':ddr['provenance']['formal_calibration_ready'],
        'not_zero_if_unrecorded':True}
