from copy import deepcopy
from cpu_policies.contention.domain import configurations as source_configurations

def cases():
    result={}
    for name,cfg in source_configurations().items():
        actors=[deepcopy(s) for s in cfg['stages'] if s['device']=='cpu']
        result[name]={'configuration':cfg,'actors':actors}
    assert len(result)==8
    assert sum(len(v['actors'])*15 for v in result.values())==300
    return result

def program_names():
    return sorted({s['program'] for value in cases().values() for s in value['actors']})
