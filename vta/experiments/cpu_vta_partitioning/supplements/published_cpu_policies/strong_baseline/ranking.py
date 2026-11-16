"""Transfer the exact historical complete scoring code, with YOLO's own costs."""
from pathlib import Path
import shutil,time
from cpu_vta_partitioning.common import read,write,sha256
from cpu_policies.open_resources.costs import Costs
from staged_ablation.model import score
from policy_branch_eval.ranking import engine_from
from policy_branch_eval.common import isolated_inputs_only
from finite_service.common import freeze
from yolo_validation.ranking import inputs as yolo_inputs
from .audit import ROOT,OUTPUT,PAPER

DEST=OUTPUT/'yolo_complete'
ARCHIVE=PAPER/'20261109_host_work_ablation/bundles/86881512ff1d51fc355e042c'
YOLO=ROOT/'artifacts/yolo_validation_01'


def sources():
    paths=[Path(__file__),ROOT/'run_strong_baseline_rank.py',ROOT/'tests/test_strong_baseline_rank.py',
        ROOT/'staged_ablation/model.py',ROOT/'staged_ablation/common.py',ROOT/'resource_envelope/model.py',
        ROOT/'cpu_policies/open_resources/costs.py']
    return {str(p):sha256(p) for p in paths}


def prepare():
    if (DEST/'INPUT_FREEZE.json').exists():
        assert_inputs();return dict(cached=True)
    proof=read(ARCHIVE/'plans/search_proofs.json')['complete']
    if not all(proof['model'][k] for k in ('thread_weights','host_guard','team_guard')) or proof['pipeline_FPS_used']:
        raise ValueError('historical complete model contract differs')
    directory=DEST/'sources/complete';directory.mkdir(parents=True,exist_ok=True)
    for leaf in ('__init__.py','search.py','oracle.cc','costs.py','domain.py','picojson.h'):
        source=ARCHIVE/'provenance/search/complete/source'/leaf
        if leaf in proof['implementation'] and sha256(source)!=proof['implementation'][leaf]:raise ValueError('historical search source differs')
        shutil.copy2(source,directory/leaf)
    programs,records,ddr,hashes=yolo_inputs(YOLO)
    all_programs=read(YOLO/'ranking/inputs/programs.json')
    excluded={n:p['purpose'] for n,p in all_programs.items() if not p['eligible_search']}
    if (len(all_programs)!=159 or len(programs)!=157 or len(records)!=714
        or excluded!={'cpu_01_01':'atomic_thread_curve_only','cpu_12_12':'atomic_thread_curve_only'}):
        raise ValueError('YOLO original legal/curve-only profile coverage differs')
    freeze(DEST/'inputs/programs.json',programs);freeze(DEST/'inputs/isolated_costs.json',dict(costs=records))
    freeze(DEST/'inputs/ddr.json',ddr)
    freeze(DEST/'INPUT_FREEZE.json',dict(sources=sources(),
        inputs={n:sha256(DEST/'inputs'/n) for n in ('programs.json','isolated_costs.json','ddr.json')},
        legacy_implementation=proof['implementation'],
        copied_sources={p.name:sha256(p) for p in directory.iterdir() if p.is_file()},
        original_YOLO_input_hashes=hashes,historical_complete_proof_sha256=sha256(ARCHIVE/'plans/search_proofs.json'),
        model_switches=dict(thread_weights=True,host_cpu_demand=True,team_guard=True,host_guard=True),
        scalar_key='concentrated',units=14,max_vta_islands=3,expected_topologies=2509,
        expected_allocations=472266548,no_pipeline_FPS_input=True,no_new_component_profile=True,
        compiled_modules=159,eligible_interval_modules=157,original_curve_only_modules=excluded,
        domain_not_restricted=True,kind='existing_complete_model_transfer_not_a_new_correction'))
    return dict(programs=len(programs),component_records=len(records),contexts=9,allocations=472266548)


def assert_inputs():
    p=read(DEST/'INPUT_FREEZE.json')
    if p['sources']!=sources():raise ValueError('strong baseline transfer source changed')
    for n,h in p['inputs'].items():
        if sha256(DEST/'inputs'/n)!=h:raise ValueError('YOLO complete isolated input changed')
    for n,h in p['copied_sources'].items():
        if sha256(DEST/'sources/complete'/n)!=h:raise ValueError('copied historical complete source changed')
    return p


def engine(p):
    return engine_from(dict(source_directory=str(DEST/'sources/complete'),implementation=p['legacy_implementation']),
        'strong_complete_yolo')


def rank():
    p=assert_inputs();started=time.monotonic()
    programs=read(DEST/'inputs/programs.json');records=read(DEST/'inputs/isolated_costs.json');ddr=read(DEST/'inputs/ddr.json')
    m=engine(p);costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
    with isolated_inputs_only():
        result=m.rank_all(programs,costs,ddr=ddr,unit_count=14,max_vta_islands=3,output_dir=DEST/'search',verify=True)
    if result['counts']['enhanced']!=p['expected_allocations'] or result['counts']['topologies']!=2509 or not result['verification']['passed']:
        raise ValueError('complete YOLO domain/oracle verification differs')
    for row in result['pure_static20']:
        if score(row,costs,'concentrated')['score_ns']!=row['score_ns']:raise ValueError('complete scalar score differs from search')
    freeze(DEST/'rankings.json',result)
    freeze(DEST/'RANK_FREEZE.json',dict(inputs_sha256=sha256(DEST/'INPUT_FREEZE.json'),rankings_sha256=sha256(DEST/'rankings.json'),
        sources=sources(),full_domain_count=472266548,verification=result['verification'],
        pipeline_FPS_read=False,no_new_board_sessions=True,all_model_switches_equal_historical_complete=True))
    write(DEST/'COST.json',dict(seconds=time.monotonic()-started),replace=True)
    return dict(count=472266548,first=result['pure_static20'][0]['id'],Top20=20,no_new_board_sessions=True)


def validate():
    p=assert_inputs();seal=read(DEST/'RANK_FREEZE.json')
    if seal['inputs_sha256']!=sha256(DEST/'INPUT_FREEZE.json') or seal['rankings_sha256']!=sha256(DEST/'rankings.json'):
        raise ValueError('complete transfer freeze differs')
    result=read(DEST/'rankings.json');programs=read(DEST/'inputs/programs.json');records=read(DEST/'inputs/isolated_costs.json');ddr=read(DEST/'inputs/ddr.json')
    costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights');m=engine(p)
    graphs=m._topologies(programs,14,3);contexts={c['id']:c for c in m.domain.contexts()};total=0;rows=[]
    for ctx in contexts.values():
        cp=read(DEST/'search'/(ctx['id']+'.json'))
        payload=dict(programs=m._prepare(programs,costs,ctx,ddr),topologies=graphs,context=ctx,k=20,
            score_scale=m.SCORE_SCALE,implementation=p['legacy_implementation'])
        fields=('counts','gate_counts','rankings','topology_strata','resource_strata')
        if (cp['checkpoint_sha256']!=m._digest({k:v for k,v in cp.items() if k!='checkpoint_sha256'}) or
            cp['input_sha256']!=m._digest(payload) or not cp['verification']['passed'] or
            cp['verification']['source_sha256']!=p['legacy_implementation']['oracle.cc'] or
            cp['verification']['result_sha256']!=m._digest({k:cp[k] for k in fields})):
            raise ValueError('complete YOLO checkpoint/oracle mismatch')
        total+=cp['counts']['enhanced']
        for raw in cp['rankings']['enhanced']:
            t=m.Terminal.from_json(raw);rows.append({**m._configuration(t,graphs,programs,contexts),'score_ns':t.score,'score_ms':t.score/1e6})
    rows=sorted(rows,key=m._row_key)[:20]
    for i,row in enumerate(rows,1):row['static_rank']=i
    if total!=472266548 or rows!=result['pure_static20']:raise ValueError('complete YOLO global Top20 differs')
    return dict(validated=True,full_domain_count=total,Top20=len(rows))
