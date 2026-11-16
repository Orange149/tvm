"""The fixed five-stage graph with independently ordered CPU worker maps.

Masks remain sets; logical worker order is a distinct execution decision.
Only caller choice is searched. Other helpers follow remaining core order.
"""
from copy import deepcopy
from hashlib import sha256
from itertools import combinations, product
import json

CORES = (0, 1, 2, 3)
PROGRAMS = ('cpu_00_02', 'vta_03_04', 'cpu_05_07', 'vta_08_19', 'cpu_20_20')
CONFIGURATION_FIELDS = ('context_id', 'compute_pool', 'reserved_cores', 'stages',
                        'affinity_mode', 'execution_mode', 'host_mask', 'control_mask',
                        'core_partition')


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def stage(device, begin, end, threads, mask, worker_core_map=None):
    mask = list(mask)
    if device not in ('cpu', 'vta') or any(type(n) is not int for n in (begin, end, threads)):
        raise ValueError('invalid stage identity')
    if (not 0 <= begin < end <= 21 or threads < 1 or not mask
            or any(type(c) is not int for c in mask) or mask != sorted(set(mask))
            or not set(mask) <= set(CORES)):
        raise ValueError('invalid core set or stage range')
    mapping = list(mask if worker_core_map is None else worker_core_map) if device == 'cpu' else []
    if device == 'cpu':
        if (threads != len(mask) or len(mapping) != threads
                or any(type(c) is not int for c in mapping) or sorted(mapping) != mask
                or mapping[1:] != sorted(set(mask) - {mapping[0]})):
            raise ValueError('worker map must select caller then ascending remaining helpers')
    elif threads != 1 or mask != list(CORES) or worker_core_map not in (None, [], ()):
        raise ValueError('VTA host is shared four-core, independent of CPU worker maps')
    program = f'{device}_{begin:02}_{end - 1:02}'
    signature = f"{program}_t{threads}_a{'-'.join(map(str, mask))}"
    if device == 'cpu' and mapping != mask:
        signature += '_w' + '-'.join(map(str, mapping))
    return {'device': device, 'begin': begin, 'end': end, 'threads': threads,
            'mask': mask, 'program': program, 'signature': signature,
            'worker_core_map': mapping, 'caller_core': mapping[0] if mapping else None}


def configuration(value):
    return {name: deepcopy(value[name]) for name in CONFIGURATION_FIELDS}


def candidate(stages):
    rows = [stage(r['device'], r['begin'], r['end'], r['threads'], r['mask'],
                  r.get('worker_core_map')) for r in stages]
    result = {'context_id': 'shared4', 'compute_pool': list(CORES), 'reserved_cores': [],
              'stages': rows, 'affinity_mode': 'pinned', 'execution_mode': 'off',
              'host_mask': list(CORES), 'control_mask': list(CORES), 'core_partition': []}
    result['id'] = 'cfg_' + digest(result)[:24]
    return result


def component(row):
    value = candidate([row])
    if row['device'] == 'vta':
        value['affinity_mode'] = 'group'
        value['id'] = 'cfg_' + digest(configuration(value))[:24]
    return value


def component_key(value):
    if len(value.get('stages', [])) != 1:
        raise ValueError('component key requires exactly one stage')
    row = value['stages'][0]
    effective = {k: row[k] for k in ('device', 'begin', 'end', 'threads', 'mask', 'worker_core_map')}
    return 'cmp_' + digest({'stage': effective, 'affinity_mode': value['affinity_mode']})[:24]


def baseline_candidate():
    return candidate([stage('cpu', 0, 3, 4, CORES), stage('vta', 3, 5, 1, CORES),
                      stage('cpu', 5, 8, 4, CORES), stage('vta', 8, 20, 1, CORES),
                      stage('cpu', 20, 21, 1, [0])])


def validate_candidate(value, programs=None, component=False, **bounds):
    expected = {'context_id': 'shared4', 'compute_pool': list(CORES), 'reserved_cores': [],
                'execution_mode': 'off', 'host_mask': list(CORES),
                'control_mask': list(CORES), 'core_partition': []}
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('core model validation uses one fixed shared resource context')
    rows = value.get('stages')
    if not isinstance(rows, list) or not rows or (component and len(rows) != 1):
        raise ValueError('invalid stage coverage')
    mode = 'group' if component and rows[0].get('device') == 'vta' else 'pinned'
    if value.get('affinity_mode') != mode:
        raise ValueError('wrong CPU binding mode')
    for row in rows:
        if row != stage(row.get('device'), row.get('begin'), row.get('end'), row.get('threads'),
                        row.get('mask', []), row.get('worker_core_map')):
            raise ValueError('stage or ordered worker identity differs')
        if row['program'] not in PROGRAMS:
            raise ValueError('stage is outside the frozen five-stage graph')
        if programs is not None:
            program = programs.get(row['program'])
            if program is None or any(program.get(k) != row[k] for k in ('device', 'begin', 'end')):
                raise ValueError('missing or mismatched compiled stage')
    if not component:
        if tuple(r['program'] for r in rows) != PROGRAMS:
            raise ValueError('full candidate changes the fixed graph partition')
        if rows[-1] != stage('cpu', 20, 21, 1, [0]):
            raise ValueError('tail CPU must stay single-thread on core zero')
    if value.get('id') != 'cfg_' + digest(configuration(value))[:24]:
        raise ValueError('candidate identity differs')
    return value


def configurations(programs=None):
    components = {}
    for label, begin, end in (('cpu00', 0, 3), ('cpu05', 5, 8)):
        for count in range(1, 5):
            components[f'{label}_t{count}'] = component(stage('cpu', begin, end, count, CORES[:count]))
    components['cpu20_t1'] = component(stage('cpu', 20, 21, 1, [0]))
    components['vta03_04'] = component(stage('vta', 3, 5, 1, CORES))
    components['vta08_19'] = component(stage('vta', 8, 20, 1, CORES))
    result = {'components': components, 'pipelines': {'static_first': baseline_candidate()}}
    for value in components.values():
        validate_candidate(value, programs, component=True)
    validate_candidate(result['pipelines']['static_first'], programs)
    return result


def requirements(programs=None):
    return {component_key(cfg): cfg for cfg in configurations(programs)['components'].values()}


def cases(bundle=None):
    if bundle is None:
        return configurations()
    from .common import api
    from pathlib import Path
    common = api(bundle)
    expected = configurations(common.read(Path(bundle) / 'plans/programs.json'))
    # Preparation stores requirements and baseline separately, rather than an
    # expanded list of candidates tied to unknown component measurements.
    if common.read(Path(bundle) / 'plans/component_requirements.json') != requirements():
        raise ValueError('canonical component requirement identity differs')
    if common.read(Path(bundle) / 'plans/fixed_baseline.json') != baseline_candidate():
        raise ValueError('frozen bridge baseline differs')
    return expected


def canonical_tuple(value):
    return tuple((r['device'], r['begin'], r['end'], r['threads'], tuple(r['mask']),
                  tuple(r['worker_core_map'])) for r in value['stages'])


def worker_options(movable=True):
    result = []
    for count in range(1, 5):
        for mask in combinations(CORES, count):
            for caller in (mask if movable else mask[:1]):
                mapping = (caller, *sorted(set(mask) - {caller}))
                result.append((mask, mapping))
    return tuple(result)


def allocations(movable=True):
    for (first_mask, first_map), (middle_mask, middle_map) in product(worker_options(movable), repeat=2):
        rows = baseline_candidate()['stages']
        rows[0] = stage('cpu', 0, 3, len(first_mask), first_mask, first_map)
        rows[2] = stage('cpu', 5, 8, len(middle_mask), middle_mask, middle_map)
        yield candidate(rows)


def intervention_candidates():
    result = {}
    for first, middle in ((0, 0), (0, 3), (3, 0), (3, 3)):
        rows = baseline_candidate()['stages']
        for index, caller in ((0, first), (2, middle)):
            old = rows[index]
            rows[index] = stage('cpu', old['begin'], old['end'], 4, CORES,
                                [caller, *[c for c in CORES if c != caller]])
        result[f'caller_{first}_{middle}'] = candidate(rows)
    for label, first_mask, middle_mask in (('disjoint_2_2', [0, 1], [2, 3]),
                                           ('disjoint_3_1', [0, 1, 2], [3])):
        rows = baseline_candidate()['stages']
        rows[0] = stage('cpu', 0, 3, len(first_mask), first_mask)
        rows[2] = stage('cpu', 5, 8, len(middle_mask), middle_mask)
        result[label] = candidate(rows)
    return result
