"""Seven fixed diagnostic cases; no ranking or feedback into the old search."""
from copy import deepcopy
from pathlib import Path

from .common import api
from ..open_resources.domain import candidate, component, stage, validate_candidate

PROGRAMS = ('cpu_00_02', 'cpu_05_07', 'cpu_17_17', 'cpu_20_20',
            'vta_03_04', 'vta_08_19', 'vta_08_16', 'vta_18_19')
EXPECTED_IDS = {
    'static_first': 'cfg_b81cf0b8325c03d01989d7a3',
    'history_host2': 'cfg_85c3cc9bfdcdcbae714e259d',
    'history_shared': 'cfg_02dd007a1254a85d113caf58',
}


def configurations(programs=None):
    first = [stage('cpu', 0, 3, 4, [0, 1, 2, 3]),
             stage('vta', 3, 5, 1, [0, 1, 2, 3]),
             stage('cpu', 5, 8, 4, [0, 1, 2, 3]),
             stage('vta', 8, 20, 1, [0, 1, 2, 3]),
             stage('cpu', 20, 21, 1, [0])]
    hist = [stage('cpu', 0, 3, 3, [0, 1, 2]),
            stage('vta', 3, 5, 1, [2]),
            stage('cpu', 5, 8, 1, [3]),
            stage('vta', 8, 17, 1, [2]),
            stage('cpu', 17, 18, 3, [0, 1, 2]),
            stage('vta', 18, 20, 1, [2]),
            stage('cpu', 20, 21, 1, [3])]
    shared = deepcopy(hist)
    for row in shared:
        if row['device'] == 'vta':
            row.update(stage('vta', row['begin'], row['end'], 1, [0, 1, 2, 3]))
    result = {
        'pipelines': {'static_first': candidate(first),
                      'history_host2': candidate(hist, 'host_shared_2', 'resource_group'),
                      'history_shared': candidate(shared, 'shared4', 'resource_group')},
        'components': {
            'cpu00_t4': component(stage('cpu', 0, 3, 4, [0, 1, 2, 3])),
            'cpu00_t3': component(stage('cpu', 0, 3, 3, [0, 1, 2])),
            'cpu05_t4': component(stage('cpu', 5, 8, 4, [0, 1, 2, 3])),
            # Deliberately profile actual core 3, rather than a core-0 alias.
            'cpu05_t1': component(stage('cpu', 5, 8, 1, [3])),
        },
    }
    for name, value in result['pipelines'].items():
        validate_candidate(value, programs)
        if value['id'] != EXPECTED_IDS[name]:
            raise ValueError('diagnostic case differs from frozen original: ' + name)
    for value in result['components'].values():
        validate_candidate(value, programs, component=True)
    return result


def cases(bundle=None):
    if bundle is None:
        return configurations()
    common = api(bundle)
    expected = configurations(common.read(Path(bundle) / 'plans/programs.json'))
    actual = common.read(Path(bundle) / 'plans/diagnosis_configurations.json')
    if actual != expected:
        raise ValueError('frozen diagnostic cases differ')
    return actual
