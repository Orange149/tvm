"""Eight predeclared pipelines and fifteen exact physical component signatures."""
from copy import deepcopy
from ..open_resources.domain import candidate as old_candidate, stage as old_stage
from ..core_model.domain import stage, component, component_key

CORES = [0,1,2,3]
PLACEMENTS = {
    'overlap4': ([0,1,2,3], [0,1,2,3]),
    'overlap2': ([0,1], [0,1]),
    'split2': ([0,1], [2,3]),
    'split31': ([0,1,2], [3]),
}
TOPOLOGIES = {
    'explanation': [('cpu',0,3),('vta',3,5),('cpu',5,8),('vta',8,20),('cpu',20,21)],
    'validation': [('cpu',0,3),('vta',3,17),('cpu',17,21)],
}

def configurations():
    result = {}
    for topology, intervals in TOPOLOGIES.items():
        for placement,(front,back) in PLACEMENTS.items():
            cpu_index, rows = 0, []
            for device,begin,end in intervals:
                mask = CORES if device == 'vta' else (front if cpu_index == 0 else back if cpu_index == 1 else [0])
                rows.append(stage(device,begin,end,len(mask) if device=='cpu' else 1,mask))
                cpu_index += device == 'cpu'
            result[topology+'/'+placement] = old_candidate(rows,'shared4','off')
    return result

def components():
    result = {}
    for cfg in configurations().values():
        for row in cfg['stages']:
            cfg1 = component(row)
            result[component_key(cfg1)] = cfg1
    assert len(result)==15
    return dict(sorted(result.items()))

def groups():
    result = {}
    for name,cfg in configurations().items():
        cpu = [s for s in cfg['stages'] if s['device']=='cpu'][:2]
        vta = [s for s in cfg['stages'] if s['device']=='vta']
        for kind in ('cpu_pair','cpu_pair_vta'):
            result[name+'/'+kind] = {'case':name,'mode':kind,'actors':deepcopy(cpu+(vta if kind.endswith('_vta') else [])),
                                    'targets':[0,1]}
        if name.endswith('/overlap4'):
            for i in range(2):
                result[name+'/vta_only_'+str(i)] = {'case':name,'mode':'vta_only',
                    'actors':deepcopy([cpu[i]]+vta),'targets':[0]}
    assert len(result)==20
    return result

def diagnostics():
    names = ['explanation/overlap4/cpu_pair','explanation/overlap4/cpu_pair_vta',
             'explanation/split2/cpu_pair_vta','validation/overlap4/cpu_pair_vta']
    return {name: groups()[name] for name in names}

def program_names():
    return sorted({s['program'] for cfg in configurations().values() for s in cfg['stages']})
