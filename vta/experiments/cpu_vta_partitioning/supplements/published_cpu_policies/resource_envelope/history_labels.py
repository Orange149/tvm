"""Post-freeze provenance labels; reads configuration plans, never FPS files."""
from itertools import permutations
from pathlib import Path
from run_resource_behavior import ROOT,SELECTION,PARENT,api

def mirror_key(cfg):
    values=[]
    for perm in permutations(range(4)):
        remap=lambda mask:tuple(sorted(perm[c] for c in mask))
        stages=tuple((s['device'],s['begin'],s['end'],s['threads'],remap(s['mask']),
                      tuple(perm[c] for c in s.get('worker_core_map',[])),
                      perm[s['caller_core']] if s['device']=='cpu' else -1) for s in cfg['stages'])
        values.append((stages,remap(cfg['compute_pool']),remap(cfg['host_mask']),
                       remap(cfg['control_mask']),remap(cfg['reserved_cores']),cfg['execution_mode']))
    return min(values)

def annotate(pool):
    c=api();known={};sources=[]
    for run in sorted((ROOT/'runs').glob('*')):
        state_file=run/'state.json'
        if not state_file.is_file():continue
        state=c.read(state_file)
        if state.get('status')!='COMPLETE':continue
        bundle=Path(state['bundle'])
        for leaf in ('static_pool.json','configurations.json'):
            path=bundle/'plans'/leaf
            if not path.is_file():continue
            rows=c.read(path)
            values=rows.values() if isinstance(rows,dict) else rows
            accepted=[r for r in values if isinstance(r,dict) and 'stages' in r and 'host_mask' in r]
            for row in accepted:known[row['id']]=row
            if accepted:sources.append({'run_id':run.name,'boot_id':state.get('boot_id'),
                'plan_path':str(path),'sha256':c.sha256(path),'configuration_count':len(accepted)})
    mirrors={mirror_key(v) for v in known.values()}
    labels={name:('previously_completed_exact_configuration' if name in known else
                  'physical_mirror_of_completed_configuration' if mirror_key(cfg) in mirrors else
                  'not_in_completed_configuration_plans') for name,cfg in pool.items()}
    return {'labels':labels,'sources':sources,'history_configuration_count':len(known),
        'annotation_after_selection_freeze':True,'FPS_files_read':False,
        'scope':'configuration novelty only; known topology or resource behavior can still inform new configurations'}
