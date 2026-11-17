"""Audited source reuse and portable capsule preparation, without old writes."""
import inspect
from pathlib import Path
from types import FunctionType

from cpu_vta_partitioning.common import digest, read, sha256, verify_inventory, write
from cpu_policies.open_resources.domain import configuration, canonical_tuple, validate_candidate
from resnet50_validation import bundle as old_bundle, validation as old_validation
from resnet50_validation.common import assert_controllers
from .common import SOURCE, PERFORMANCE, MODELS, protocol, frozen, controller_sources, assert_controller
from .ranking import inputs, require_acceptance


def audit(output, run_dir=None):
    from resnet50_validation.ranking import verify_search_acceptance
    output = Path(output); controllers = assert_controllers(SOURCE)
    verify_search_acceptance(SOURCE)
    programs, records, ddr, hashes = inputs(output)
    receipts = set(); boots = set()
    for key, record in records.items():
        source = record['source']; path = Path(source['receipt'])
        if sha256(path) != source['receipt_sha256']:
            raise ValueError('isolated provenance differs: ' + key)
        if not record['correct'] or not record['actual_affinity_checked'] or record['frame_count'] != 10:
            raise ValueError('invalid existing isolated session')
        boots.add(source['boot_id']); receipts.add(str(path))
    traffic = read(output / 'ranking/inputs/traffic.json')
    for record in traffic['records'].values():
        source = record['source']
        path = SOURCE.parents[1] / 'runs' / SOURCE.name / 'traffic/accepted' / source['boot_id'] / (source['task_id']+'.json')
        if sha256(path) != source['receipt_sha256']:
            raise ValueError('DMA provenance differs')
    if len(records) != 2751 or len(programs) != 613 or len(traffic['records']) != 299:
        raise ValueError('frozen graph/component domain changed')
    result = dict(passed=True, original_controller_files=len(controllers['sources']),
        input_hashes=hashes, component_records=len(records), source_receipts=len(receipts),
        DMA_records=len(traffic['records']), profile_boots=sorted(boots),
        CPU_costs=sum(r['configuration']['stages'][0]['device']=='cpu' for r in records.values()),
        whole_session_share_scope='includes warmup and startup probe; not compute-only',
        compiler_runtime_executor_changed=False, new_profiles=0, board_accessed=False)
    frozen(output / 'AUDIT.json', result)
    return result


def validate_lists(lists, pool, programs=None, *, unit_count=38, bridges=()):
    if set(lists) != set(MODELS):
        raise ValueError('five policy lists required')
    union = set()
    for key, rows in lists.items():
        if len(rows)!=20 or len({r['id'] for r in rows})!=20:
            raise ValueError('twenty distinct configurations required per policy')
        ordered=[]
        for rank,row in enumerate(rows,1):
            name=row['id']; cfg=configuration(row)
            if name!='cfg_'+digest(cfg)[:24] or name not in pool or cfg!=configuration(pool[name]):
                raise ValueError('execution identity or pool coverage differs')
            if row.get('static_rank',rank)!=rank or type(row['score_ns']) is not int:
                raise ValueError('invalid static rank or noninteger score')
            if programs is not None:
                validate_candidate(row,programs,unit_count=unit_count,max_vta_islands=3)
            ordered.append((row['score_ns'],canonical_tuple(row)));union.add(name)
        if ordered!=sorted(ordered):
            raise ValueError('score and canonical order differ')
    if len(bridges)!=3 or sorted(b['previous_budget'] for b in bridges)!=[1,5,20]:
        raise ValueError('three fixed baseline bridges required')
    for b in bridges:
        cfg=b['configuration'];name=cfg['id']
        if (not b['not_a_new_baseline_Top20'] or name not in pool
                or configuration(cfg)!=configuration(pool[name])):
            raise ValueError('baseline bridge identity differs')
        if cfg['execution_mode']!='resource_group' or any(
                s['threads']!=1 for s in cfg['stages'] if s['device']=='cpu'):
            raise ValueError('baseline physical-core semantics differ')
        union.add(name)
    if set(pool)!=union or len(pool)>103:
        raise ValueError('selection union or approved scope differs')
    return True


def validate_bundle(bundle):
    bundle=Path(bundle)
    namespace={**old_validation.validate_bundle.__globals__, 'protocol':protocol,
        'validate_lists':lambda lists,pool,programs: validate_lists(lists,pool,programs,
            bridges=read(bundle/'plans/bridges.json'))}
    fn=FunctionType(old_validation.validate_bundle.__code__,namespace)
    return fn(bundle)


def prepare(output, run_dir=None):
    output=Path(output).resolve();audit(output);require_acceptance(output)
    frozen(output/'CONTROLLERS.json',dict(sources=controller_sources(),protocol=protocol()))
    namespace=dict(vars(old_bundle))
    for name,fn in vars(old_bundle).items():
        if isinstance(fn,FunctionType) and fn.__globals__ is vars(old_bundle):
            copy=FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__)
            copy.__kwdefaults__=fn.__kwdefaults__;namespace[name]=copy
    namespace.update(assert_controllers=assert_controller,protocol=protocol,
        FROZEN_PROFILE=SOURCE/'profile_capsule',validate_bundle=validate_bundle,
        validate_lists=lambda lists,pool,programs,**kw:validate_lists(lists,pool,programs,
            bridges=read(output/'ranking/bridges.json'),**kw))
    text=inspect.getsource(old_bundle.prepare)
    before="profile = output / 'profile_capsule'"
    if text.count(before)!=1:
        raise ValueError('frozen capsule preparer changed')
    text=text.replace(before,'profile = FROZEN_PROFILE')
    exec(compile(text,__file__+':portable-capsule','exec'),namespace)
    manifest=namespace['prepare'](output)
    return dict(status='PREPARED',bundle_id=manifest['bundle_id'],
                candidates=manifest['candidate_count'],new_profiles=0,new_native_builds=0)


def validate(output,run_dir=None):
    output=Path(output);result=dict(source=audit(output),acceptance=require_acceptance(output))
    if (output/'pipeline_bundle/manifest.json').exists():
        assert_controller(output)
        result['bundle']=validate_bundle(output/'pipeline_bundle')['bundle_id']
    result['status']='VALIDATED'
    return result


def status(output,run_dir=None):
    output=Path(output);result=dict(board_accessed=False,models={})
    for key in MODELS:
        base=output/'ranking/models'/key
        if (base/'RESULTS.json').exists():
            result['models'][key]=dict(status='COMPLETE',first=read(base/'RESULTS.json')['rankings'][0]['id'])
        elif (base/'search/progress.json').exists():
            doc=read(base/'search/progress.json')
            result['models'][key]={k:v for k,v in doc.items() if k not in ('statistics','input_sha256')}
        else:result['models'][key]={'status':'PENDING'}
    if (output/'ranking/FREEZE.json').exists():
        result['candidates']=read(output/'ranking/FREEZE.json')['candidate_count']
    if run_dir is not None and (Path(run_dir)/'pipeline/state.json').exists():
        result['pipeline']=read(Path(run_dir)/'pipeline/state.json')
    if run_dir is not None and (Path(run_dir)/'state.json').exists():
        result['controller']=read(Path(run_dir)/'state.json')
    return result
