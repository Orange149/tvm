"""Pure integer-nanosecond DAG replay. No paths, clocks, FPS or file I/O."""
from fractions import Fraction
import heapq


def integer(value, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError('model requires nonnegative integer nanoseconds/indices')
    return value


def validate(data):
    if (not isinstance(data, dict) or set(data) != {'schema', 'source_role', 'cases'}
            or data['schema'] != 1 or data['source_role'] != 'isolated_original_wait_callbacks'):
        raise ValueError('only projected isolation inputs are supported')
    if not isinstance(data['cases'], dict) or not data['cases']:
        raise ValueError('model needs isolated cases')
    for name, case in data['cases'].items():
        if set(case) != {'core_map', 'frames'} or not isinstance(name, str):
            raise ValueError('model case contains unapproved fields')
        cores = case['core_map']
        if (not isinstance(cores, list) or not 1 <= len(cores) <= 4
                or any(type(c) is not int or c not in range(4) for c in cores)
                or len(set(cores)) != len(cores) or not case['frames']):
            raise ValueError('fixed one-worker-per-core mapping required')
        ids = set()
        for frame in case['frames']:
            if set(frame) != {'frame_id', 'phase', 'worker_cpu_ns', 'interlaunch_caller_cpu_ns'}:
                raise ValueError('model frame contains non-isolated data')
            integer(frame['frame_id'])
            if frame['frame_id'] in ids or frame['phase'] not in ('warmup', 'scored'):
                raise ValueError('frame identity or phase differs')
            ids.add(frame['frame_id'])
            kernels, gaps = frame['worker_cpu_ns'], frame['interlaunch_caller_cpu_ns']
            if not isinstance(kernels, list) or not kernels or len(gaps) != len(kernels)-1:
                raise ValueError('kernel sequence/gap coverage differs')
            for kernel in kernels:
                if len(kernel) != len(cores):
                    raise ValueError('kernel callback coverage differs')
                for value in kernel:
                    integer(value)
            for value in gaps:
                integer(value)
    return data


def graph(frame, cores, include_gaps=False):
    nodes, previous = [], []
    for k, values in enumerate(frame['worker_cpu_ns']):
        tasks = []
        for worker, duration in enumerate(values):
            name = f'kernel{k}_worker{worker}'
            nodes.append(dict(id=name, kind='callback', kernel=k, worker=worker,
                core=cores[worker], duration_ns=duration, parents=list(previous)))
            tasks.append(name)
        name = f'kernel{k}_join'
        nodes.append(dict(id=name, kind='join', kernel=k, core=None, duration_ns=0, parents=tasks))
        previous = [name]
        if include_gaps and k < len(frame['worker_cpu_ns'])-1:
            name = f'gap{k}'
            nodes.append(dict(id=name, kind='observed_interlaunch_caller_CPU', kernel=k,
                core=cores[0], duration_ns=frame['interlaunch_caller_cpu_ns'][k], parents=previous))
            previous = [name]
    return nodes


def schedule(nodes):
    """Topological frontier with deterministic ties and per-core progress."""
    indexed = {n['id']: n for n in nodes}
    if len(indexed) != len(nodes):
        raise ValueError('duplicate DAG node')
    children = {name: [] for name in indexed}
    remaining = {}
    for node in nodes:
        if len(node['parents']) != len(set(node['parents'])):
            raise ValueError('duplicate DAG dependency')
        remaining[node['id']] = len(node['parents'])
        for parent in node['parents']:
            if parent not in indexed:
                raise ValueError('unknown DAG parent')
            children[parent].append(node['id'])
    frontier = [name for name in indexed if remaining[name] == 0]
    heapq.heapify(frontier)
    ends, resources, result = {}, {}, []
    while frontier:
        name = heapq.heappop(frontier)
        node = indexed[name]
        start = max([ends[p] for p in node['parents']] + [0])
        if node['core'] is not None:
            start = max(start, resources.get(node['core'], 0))
        end = start + integer(node['duration_ns'])
        ends[name] = end
        if node['core'] is not None:
            resources[node['core']] = end
        result.append(dict(node, start_ns=start, end_ns=end))
        for child in children[name]:
            remaining[child] -= 1
            if remaining[child] == 0:
                heapq.heappush(frontier, child)
    if len(ends) != len(nodes):
        raise ValueError('cyclic dependency graph')
    return max(ends.values(), default=0), result


def rational(values):
    value = Fraction(sum(values), len(values))
    return dict(numerator=value.numerator, denominator=value.denominator, unit='ns')


def replay(data):
    validate(data)
    cases = {}
    for name, case in data['cases'].items():
        frames = []
        for frame in case['frames']:
            kernels = frame['worker_cpu_ns']
            aggregate = max(sum(k[w] for k in kernels) for w in range(len(case['core_map'])))
            join, timeline = schedule(graph(frame, case['core_map']))
            extended, extended_timeline = schedule(graph(frame, case['core_map'], True))
            # Separate scalar implementation checks the graph traversal.
            if join != sum(max(kernel) for kernel in kernels) or extended != join+sum(frame['interlaunch_caller_cpu_ns']):
                raise ValueError('DAG replay disagrees with scalar arithmetic')
            if join < aggregate:
                raise ValueError('join bound cannot be below the aggregate bound')
            frames.append(dict(frame_id=frame['frame_id'], phase=frame['phase'], aggregate_ns=aggregate,
                join_ns=join, additional_dependency_ns=join-aggregate,
                callback_plus_observed_gap_ns=extended, observed_gap_CPU_ns=sum(frame['interlaunch_caller_cpu_ns']),
                total_callback_CPU_ns=sum(sum(k) for k in kernels), timeline=timeline,
                timeline_with_observed_gaps=extended_timeline))
        phases = {}
        for phase in ('warmup', 'scored', 'whole_session'):
            selected = frames if phase == 'whole_session' else [f for f in frames if f['phase'] == phase]
            if selected:
                fields = ('aggregate_ns', 'join_ns', 'additional_dependency_ns',
                          'callback_plus_observed_gap_ns', 'observed_gap_CPU_ns', 'total_callback_CPU_ns')
                phases[phase] = dict(frame_count=len(selected),
                    means={key: rational([f[key] for f in selected]) for key in fields})
        cases[name] = dict(core_map=case['core_map'], frames=frames, phases=phases)
    return dict(schema=1, source_role=data['source_role'], cases=cases,
        scope='ideal single-stage compute-work replay; not stage service or pipeline II',
        Linux_scheduling_simulated=False, full_pipeline_FPS=None,
        repeated_completion_already_in_isolated_stage_service=True)
