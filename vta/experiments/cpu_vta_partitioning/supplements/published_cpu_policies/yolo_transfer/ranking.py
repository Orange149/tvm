"""Transfer the frozen ResNet model to true YOLO components, without pipeline I/O."""
import inspect
import math
from pathlib import Path
import time

from cpu_vta_partitioning.common import digest, now, read, sha256, verify_inventory, write
from cpu_vta_partitioning.prepare.build import copy_checked


def write_frozen(path, document):
    path = Path(path)
    if path.exists():
        if read(path) != document:
            raise ValueError('partial static output changed; use a new ranking directory')
    else:
        write(path, document)


def replay_profiles(bundle, run_dir):
    from . import profile
    bundle, run_dir = Path(bundle), Path(run_dir)
    manifest = profile.validate(bundle)
    data = read(run_dir / 'isolated_costs.json')
    state = read(run_dir / 'run.json')
    configurations = read(bundle / 'plans/component_configurations.json')
    if (data['status'] != 'COMPLETE' or set(data['costs']) != set(configurations)
            or data['accepted'] != len(configurations) or state['bundle_id'] != manifest['bundle_id']):
        raise ValueError('the complete matched component table is required before ranking')
    for key, record in data['costs'].items():
        source = record['source']
        path = Path(source.get('receipt', run_dir / 'accepted' / source['boot_id'] / (source['task_id'] + '.json')))
        if sha256(path) != source['receipt_sha256']:
            raise ValueError('component receipt changed')
        receipt = read(path); document = receipt['task']
        if document['kind'] != 'calibration' or document['component_key'] != key:
            raise ValueError('only isolated calibration evidence is admissible')
        if (document['configuration'] != configurations[key] or record['configuration'] != configurations[key]
                or receipt['boot_id'] != source['boot_id'] or source['boot_id'] != state['boot_id']):
            raise ValueError('profile execution or boot identity differs')
        verify_inventory(receipt['raw'], receipt['files'])
        result = profile.summary(receipt['raw'], document, bundle)
        if result != receipt['summary'] or any(record.get(k) != v for k, v in result.items()):
            raise ValueError('component cost does not replay exactly')
    return data['costs']


def replay_traffic(bundle, run_dir):
    from . import traffic
    bundle, run_dir = Path(bundle), Path(run_dir)
    data = read(run_dir / 'traffic.json')
    state = read(run_dir / 'run.json')
    programs = read(bundle / 'plans/programs.json')
    expected = {name for name, value in programs.items() if value['device'] == 'vta'}
    if data['status'] != 'COMPLETE' or set(data['records']) != expected:
        raise ValueError('all actual VTA DMA payloads must be available')
    for name, record in data['records'].items():
        source = record['source']
        path = run_dir / 'accepted' / source['boot_id'] / (source['task_id'] + '.json')
        if sha256(path) != source['receipt_sha256'] or source['boot_id'] != state['boot_id']:
            raise ValueError('traffic receipt or boot identity differs')
        receipt = read(path); document = receipt['task']
        if document != traffic.task(bundle, document['component_key']) or document['stages'][0]['program'] != name:
            raise ValueError('traffic task differs')
        verify_inventory(receipt['raw'], receipt['files'])
        result = traffic.summary(receipt['raw'], document, bundle)
        if result != receipt['summary'] or any(record.get(k) != v for k, v in result.items()):
            raise ValueError('traffic source does not replay exactly')
    return data['records']


def ddr_table(programs, traffic, hardware):
    """Preserve the existing DDR formula using new shapes and actual DMA payloads."""
    from cpu_policies.open_resources.ddr import tensor_bytes
    def positive(value):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('invalid historical hardware bandwidth parameter')
        return value
    table = {}
    for name, spec in programs.items():
        if spec['device'] == 'cpu':
            inputs = sum(tensor_bytes(t) for t in spec['input_types'])
            outputs = sum(tensor_bytes(t) for t in spec['output_types'])
            table[name] = {str(n): inputs / (positive(hardware['cpu_GBps'][f'read_t{n}']) * 1e6)
                          + outputs / (positive(hardware['cpu_GBps'][f'write_t{n}']) * 1e6) for n in range(1, 5)}
        else:
            record = traffic[name]
            if record.get('hooks_present') is not True or record.get('count') != 1:
                raise ValueError('actual single-run VTA payload required')
            table[name] = {'1': positive(record['load_bytes']) / (positive(hardware['vta_load_GBps']) * 1e6)
                          + positive(record['store_bytes']) / (positive(hardware['vta_store_GBps']) * 1e6)}
    return {'stage_service_ms': table, 'provenance': hardware,
            'formula': 'max_with_sum_DDR_service_not_added_to_wall',
            'assumptions': ['historical hardware bandwidths are provisional, not a fresh calibration',
                'CPU logical frontier traffic is a proxy; aliases, cache and weight traffic are not modeled exactly',
                'VTA requested DMA payload is measured on these modules, not AXI total traffic',
                'no ResNet stage service time, scaling coefficient or pipeline FPS is used']}


def compact_records(programs, checkpoints, unit_count):
    """Retain the audited collector expansion with an explicit graph-size adapter."""
    from selection_top20.static import compact_records as original
    from resource_envelope.search_adapter import _replace
    source = _replace(inspect.getsource(original), 'graphs=search._topologies(programs,21,3)',
                      f'graphs=search._topologies(programs,{unit_count},3)')
    namespace = dict(original.__globals__)
    exec(compile(source, '<YOLO typed graph-size adapter>', 'exec'), namespace)
    return namespace['compact_records'](programs, checkpoints)


def verify_artifacts(output):
    """Bind all nine complete-domain proofs, model inputs and frozen lists."""
    import ast
    from cpu_policies.open_resources import search, domain
    from cpu_policies.open_resources.costs import Costs
    from resource_envelope.selection import profile_scales, select
    from .domain import programs as legal_programs
    from .domain_counts import allocation_counts
    output = Path(output).resolve()
    freeze = read(output / 'FREEZE.json')
    if freeze['rankings_sha256'] != sha256(output / 'rankings.json') or freeze['pool_sha256'] != sha256(output / 'pool.json'):
        raise ValueError('ranking freeze differs')
    inputs = read(output / 'INPUTS.json')
    for entry in inputs.values():
        if sha256(output / entry['frozen_copy']) != entry['sha256']:
            raise ValueError('frozen component/model input changed')
    if (inputs['isolated_costs.json']['sha256'] != freeze['profile_sha256']
            or inputs['traffic.json']['sha256'] != freeze['traffic_sha256']
            or sha256(output / 'ddr.json') != freeze['ddr_sha256']):
        raise ValueError('proof and component-source identities differ')
    data = read(output / 'inputs/isolated_costs.json')
    if data['status'] != 'COMPLETE':
        raise ValueError('incomplete component table cannot support a ranking freeze')
    programs = read(output / 'inputs/programs.json')
    programs = {name: p for name, p in programs.items() if p['eligible_search']}
    expected = {p['name']: p for p in legal_programs() if p['eligible_search']}
    if set(programs) != set(expected) or any(any(programs[n][k] != p[k] for k in ('device', 'begin', 'end'))
                                            for n, p in expected.items()):
        raise ValueError('qualified stages do not cover the complete fourteen-unit domain')
    graphs = search._topologies(programs, 14, 3)
    ddr = read(output / 'ddr.json')
    count_proof = allocation_counts()
    ranked = read(output / 'rankings.json')
    if any(ranked['counts'][k] != count_proof[k] for k in ('topologies', 'contexts', 'enhanced', 'single_core', 'prefix')):
        raise ValueError('search cardinality differs from independent combinatorial counting')
    contexts = {ctx['id']: ctx for ctx in domain.contexts()}
    all_checkpoints = {}
    def function_ast(text, name):
        found = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name]
        if len(found) != 1:
            raise ValueError('missing or ambiguous search helper')
        return ast.dump(found[0], include_attributes=False)
    for variant, distribution in (('original', 'balanced'), ('enhanced', 'session_weights')):
        root = output / variant
        source_identity = read(root / 'source_identity.json')
        package_name = source_identity['generated_package']
        if Path(package_name).name != package_name:
            raise ValueError('generated package path escapes proof directory')
        package = root / 'sources' / package_name
        # Exact source bytes, not cached Python bytecode, define the algorithm.
        generated_files = source_identity['generated_files']
        for name in ('search.py', 'oracle.cc', 'picojson.h'):
            verify_inventory(package, {name: generated_files[name]})
        for helper in ('_prepare', '_topologies'):
            if function_ast((package / 'search.py').read_text(), helper) != function_ast(Path(search.__file__).read_text(), helper):
                raise ValueError('input-table or topology adapter changed unexpectedly')
        implementation = {name: sha256(package / name) for name in ('search.py', 'oracle.cc', 'picojson.h')}
        costs = Costs(data['costs'], programs=programs, ddr=ddr, distribution=distribution)
        checkpoints = []
        counts = {name: 0 for name in ('enhanced', 'single_core', 'prefix')}
        for ctx in contexts.values():
            cp = read(root / 'search' / (ctx['id'] + '.json'))
            if cp['checkpoint_sha256'] != search._digest({k: v for k, v in cp.items() if k != 'checkpoint_sha256'}):
                raise ValueError('static checkpoint content seal differs')
            payload = {'programs': search._prepare(programs, costs, ctx, ddr), 'topologies': graphs,
                       'context': ctx, 'k': 20, 'score_scale': search.SCORE_SCALE, 'implementation': implementation}
            if cp['input_sha256'] != search._digest(payload) or cp['context'] != ctx or cp['topologies'] != len(graphs):
                raise ValueError('checkpoint does not match the full component/model input')
            fields = ('counts', 'gate_counts', 'rankings', 'topology_strata', 'resource_strata')
            proof = cp['verification']
            if (not proof['passed'] or proof['source_sha256'] != implementation['oracle.cc']
                    or proof['compared'] != list(fields)
                    or proof['result_sha256'] != search._digest({k: cp[k] for k in fields})):
                raise ValueError('independent enumeration proof is not bound to the checkpoint')
            for name in counts:
                counts[name] += cp['counts'][name]
            checkpoints.append(cp)
        if any(counts[name] != count_proof[name] for name in counts):
            raise ValueError('per-context search coverage differs from independent domain counting')
        candidates = []
        for cp in checkpoints:
            for item in cp['rankings']['enhanced']:
                terminal = search.Terminal.from_json(item)
                cfg = search._configuration(terminal, graphs, programs, contexts)
                candidates.append({**cfg, 'score_ns': terminal.score})
        candidates.sort(key=lambda cfg: (cfg['score_ns'], domain.canonical_tuple(cfg)))
        key = 'uniform_strict20' if variant == 'original' else 'weighted_strict20'
        simplified = lambda rows: [(r['id'], domain.configuration(r), r['score_ns']) for r in rows]
        if simplified(candidates[:20]) != simplified(ranked['lists'][key]):
            raise ValueError('strict Top20 differs from the independently checked terminal lists')
        all_checkpoints[variant] = checkpoints
    costs = Costs(data['costs'], programs=programs, ddr=ddr, distribution='session_weights')
    records, expand = compact_records(programs, all_checkpoints['enhanced'], 14)
    selected, selection = select(records, expand, profile_scales(programs, costs), preserve_worker_roles=True)
    if simplified(selected) != simplified(ranked['lists']['weighted_behavior20']) or selection != ranked['selection']:
        raise ValueError('behavior list differs from the frozen profile-only selection rule')
    return {'passed': True, 'scope': 'all 9 contexts, both full model domains and all three lists',
            'domain_counts': count_proof, 'pipeline_FPS_read': False}


def run(bundle, profile_run, traffic_run, hardware_source, protocol, output):
    from cpu_policies.open_resources.costs import Costs
    from cpu_policies.open_resources.domain import contexts, validate_candidate
    from resource_envelope.model import score
    from resource_envelope.search_adapter import generated
    from resource_envelope.weighted_search import original_engine
    from resource_envelope.selection import profile_scales, select
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'FREEZE.json').exists():
        raise FileExistsError('candidate lists are frozen; use a new output directory')
    protocol_data = read(protocol)
    source_root = Path(__file__).resolve().parents[1]
    if any(sha256(source_root / name) != expected for name, expected in protocol_data['model_sources'].items()):
        raise ValueError('a frozen model or selection source changed before transfer')
    started = time.monotonic()
    records = replay_profiles(bundle, profile_run)
    traffic = replay_traffic(bundle, traffic_run)
    sources = {'isolated_costs.json': Path(profile_run) / 'isolated_costs.json',
               'traffic.json': Path(traffic_run) / 'traffic.json', 'hardware_source.json': Path(hardware_source),
               'transfer_protocol.json': Path(protocol), 'profile_capsule_manifest.json': bundle / 'manifest.json',
               'programs.json': bundle / 'plans/programs.json'}
    for filename, source in sources.items():
        copy_checked(source, output / 'inputs' / filename)
    write_frozen(output / 'INPUTS.json', {name: {'source_path': str(p.resolve()), 'sha256': sha256(p),
                 'frozen_copy': 'inputs/' + name} for name, p in sources.items()})
    programs = {name: p for name, p in read(bundle / 'plans/programs.json').items() if p['eligible_search']}
    hardware = read(hardware_source)['provenance']
    ddr = ddr_table(programs, traffic, hardware)
    write(output / 'ddr.json', ddr, replace=True)
    uniform = Costs(records, programs=programs, ddr=ddr, distribution='balanced')
    weighted = Costs(records, programs=programs, ddr=ddr, distribution='session_weights')
    original = original_engine(output / 'original')
    enhanced = generated(output / 'enhanced')
    baseline = original.rank_all(programs, uniform, ddr=ddr, output_dir=output / 'original/search',
                                 unit_count=14, max_vta_islands=3, verify=True)
    ranked = enhanced.rank_all(programs, weighted, ddr=ddr, output_dir=output / 'enhanced/search',
                               unit_count=14, max_vta_islands=3, verify=True)
    if baseline['counts'] != ranked['counts'] or ranked['counts']['topologies'] != 2509:
        raise ValueError('the two models did not search the same complete YOLO domain')
    if not baseline['verification']['passed'] or not ranked['verification']['passed']:
        raise ValueError('full-domain independent enumeration must pass')
    checkpoints = [read(output / 'enhanced/search' / (ctx['id'] + '.json')) for ctx in contexts()]
    compact, expand = compact_records(programs, checkpoints, 14)
    behavior, selection = select(compact, expand, profile_scales(programs, weighted), preserve_worker_roles=True)
    lists = {'uniform_strict20': baseline['pure_static20'], 'weighted_strict20': ranked['pure_static20'],
             'weighted_behavior20': behavior}
    pool = {}
    for method, rows in lists.items():
        if len(rows) != 20 or len({r['id'] for r in rows}) != 20:
            raise ValueError('each frozen method requires twenty distinct configurations')
        for row in rows:
            validate_candidate(row, programs, unit_count=14, max_vta_islands=3)
            prediction = score(row, uniform if method == 'uniform_strict20' else weighted, ddr)
            variant = 'original' if method == 'uniform_strict20' else 'combined'
            if row['score_ns'] != prediction[variant]:
                raise ValueError('selected score differs from the scalar formula')
            pool[row['id']] = {**row, 'selected_score_ns': score(row, weighted, ddr)['combined'],
                              'original_score_ns': score(row, uniform, ddr)['original']}
    if len(pool) > 60:
        raise ValueError('frozen union exceeds sixty configurations')
    write_frozen(output / 'rankings.json', {'lists': lists, 'selection': selection,
          'counts': ranked['counts'], 'original_verification': baseline['verification'],
          'enhanced_verification': ranked['verification']})
    write_frozen(output / 'pool.json', dict(sorted(pool.items())))
    write(output / 'FREEZE.json', {'frozen_at': now(), 'rankings_sha256': sha256(output / 'rankings.json'),
          'pool_sha256': sha256(output / 'pool.json'), 'transfer_protocol_sha256': sha256(protocol),
          'profile_sha256': sha256(Path(profile_run) / 'isolated_costs.json'),
          'traffic_sha256': sha256(Path(traffic_run) / 'traffic.json'),
          'hardware_source_sha256': sha256(hardware_source), 'ddr_sha256': sha256(output / 'ddr.json'),
          'ranking_source_sha256': sha256(Path(__file__)), 'candidate_count': len(pool),
          'elapsed_seconds': time.monotonic() - started, 'pipeline_FPS_used': False})
    write(output / 'ARTIFACT_VERIFICATION.json', verify_artifacts(output))
    return {'candidate_count': len(pool), 'counts': ranked['counts'], 'lists_frozen': True}
