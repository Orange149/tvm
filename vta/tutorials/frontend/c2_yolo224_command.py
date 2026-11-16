#!/usr/bin/env python3
"""Independently reconcile complete main-experiment command submissions."""
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path


FIELDS=('reason','insn_bytes','uop_bytes','load_bytes','store_bytes')


def account(run):
    result=json.loads((run/'result.json').read_text())
    contract=json.loads((run/'contract.json').read_text())
    rows=[json.loads(line[len('[VTA_QUEUE] '):]) for line in (run/'stderr').read_text().splitlines()
          if line.startswith('[VTA_QUEUE] ')]
    if not rows or len(rows)!=result['runtime_profile']['synchronize_calls']:
        raise ValueError('incomplete submission ledger')
    queues=defaultdict(list);environment=contract['environment']
    for row in rows:
        if row['timeout'] or row['submit_threshold_configured']:
            raise ValueError('main experiment changed submission scheduling or timed out')
        for name,key in (('insn','VTA_INSN_BUFFER_BYTES'),('uop','VTA_UOP_BUFFER_BYTES')):
            if row[name+'_capacity']!=int(environment[key]) or row[name+'_bytes']>row[name+'_capacity']:
                raise ValueError('submission capacity contract differs')
        queues[row['queue_id']].append(row)
    for queue in queues.values():
        if [r['submit'] for r in queue]!=list(range(1,len(queue)+1)):
            raise ValueError('submission sequence has a gap or duplicate')
        for name in ('insn','uop'):
            peak=0
            for row in queue:
                peak=max(peak,row[name+'_bytes'])
                if row[name+'_peak']!=peak:raise ValueError('complete submission peak accounting differs')
    counts=Counter(tuple(r[k] for k in FIELDS) for r in rows)
    signature=sorted([list(key)+[count] for key,count in counts.items()])
    peaks=[max(r[k+'_bytes'] for r in rows) for k in ('insn','uop')]
    return {'frames':len(result['frames']),'complete_submissions':len(rows),
        'queue_instances':len(queues),'complete_tail_peaks':peaks,
        'Q1_capacities':[(max(4096,2*x)+255)//256*256 for x in peaks],
        'command_signature_sha256':hashlib.sha256(json.dumps(signature,separators=(',',':')).encode()).hexdigest(),
        'signature_fields':[*FIELDS,'multiplicity']}


def reconcile(directory):
    runs={name:account(directory/'runs'/('qual_'+name)) for name in ('M0','M1','M2','M3','manual')
          if (directory/'runs'/('qual_'+name)/'verified.json').exists()}
    if not runs:return None
    baseline=runs.get('M2',next(iter(runs.values())))
    for value in runs.values():
        if any(value[k]!=baseline[k] for k in ('frames','complete_submissions','queue_instances',
                'complete_tail_peaks','Q1_capacities','command_signature_sha256')):
            raise ValueError('main modes changed complete command work or submission multiplicity')
    return {'status':'passed','modes':runs,'command_signature_sha256':baseline['command_signature_sha256']}
