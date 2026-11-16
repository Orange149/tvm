from copy import deepcopy
from .common import api, protocol
from cpu_policies.open_resources.domain import validate_candidate, configuration, canonical_tuple

def pool_from_rankings(rankings):
    result={}
    for name in ('pure_static20','diverse20'):
        rows=rankings[name]
        if len(rows)!=20 or len({r['id'] for r in rows})!=20:raise ValueError('expected twenty distinct configurations')
        for row in rows:
            validate_candidate(row)
            if type(row.get('score_ns'))!=int or row['score_ns']<=0:raise ValueError('invalid static score')
            if row['id'] in result and (configuration(result[row['id']])!=configuration(row)
                                       or result[row['id']]['score_ns']!=row['score_ns']):
                raise ValueError('configuration/score identity collision')
            result.setdefault(row['id'],deepcopy(row))
    if len(result)>protocol()['max_pool']:raise ValueError('CPU candidate pool exceeds thirty')
    return dict(sorted(result.items()))

def groups(pool,programs,inputs,executor):
    """Only actual CPU work enters the equivalence key; host cost stays in DP."""
    c=api();result={};mapping={};frontier=inputs['records'][0]['frontiers']
    for name,cfg in pool.items():
        validate_candidate(cfg,programs)
        actors=[deepcopy(s) for s in cfg['stages'] if s['device']=='cpu']
        if not 2<=len(actors)<=4:raise ValueError('CPU actor count outside complete DP domain')
        work=[]
        for row in actors:
            p=programs[row['program']]
            work.append({'stage':row,'program_files':p['files'],
                         'input_frontier':[frontier[str(n)] for n in p['input_node_ids']]})
        key={'actors':work,'control_mask':cfg['control_mask'],'gate':cfg['execution_mode'],
             'core_partition':cfg['core_partition'],'executor':executor,'protocol':protocol()}
        group='cpu_'+c.digest(key)[:24];mapping[name]=group
        if group not in result:
            result[group]={'actors':actors,'control_mask':cfg['control_mask'],'gate':cfg['execution_mode'],
                           'core_partition':cfg['core_partition'],'identity':key,'members':[]}
        result[group]['members'].append(name)
    return {'groups':dict(sorted(result.items())),'configuration_to_group':mapping,
            'equivalence':'same CPU work, frontier, worker maps, controls, gate and executor; no VTA background'}

def canonical(row):
    return canonical_tuple(row)
