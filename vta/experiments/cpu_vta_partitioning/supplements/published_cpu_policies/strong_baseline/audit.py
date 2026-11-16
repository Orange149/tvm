from copy import deepcopy
from pathlib import Path
import json,statistics,time
from cpu_vta_partitioning.common import read,write,sha256,digest,verify_inventory,now
from cpu_policies.open_resources.costs import Costs
from cpu_policies.open_resources.domain import configuration
from host_work_ablation.model import score as historical_score
from cpu_policies.open_resources.runtime import validate_summary
from staged_ablation.model import score as staged_score
from policy_branch_eval.ranking import engine_from
from finite_service.common import freeze

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'artifacts/strong_baseline_01'
PAPER=ROOT.parents[4]/'paper/my_paper/analysis'
ARCHIVES=('20261108_model_ablation','20261109_host_work_ablation')
CURRENT=ROOT/'artifacts/finite_service_01/prospective/resnet18/capsule'
EXPECTED='cfg_23785ba24ae56fbe3f44642b'


def full_search(bundle):
    proof=read(bundle/'plans/search_proofs.json')['complete']
    if not all(proof['model'][k] for k in ('thread_weights','host_guard','team_guard')):
        raise ValueError('historical strongest model switch disabled')
    if proof.get('pipeline_FPS_used') or not proof['verification']['passed']:
        raise ValueError('historical full search not independently verified')
    projected=deepcopy(proof);projected['source_directory']=str(bundle/'provenance/search/complete/source')
    engine=engine_from(projected,'strong_'+bundle.name)
    records=read(bundle/'plans/isolated_costs.json')['costs'];programs=read(bundle/'plans/full_programs.json');ddr=read(bundle/'plans/ddr.json')
    for leaf,h in proof['component_hashes'].items():
        if sha256(bundle/'plans'/leaf)!=h:raise ValueError('historical complete cost input changed')
    costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
    graphs=engine._topologies(programs,21,3);contexts={c['id']:c for c in engine.domain.contexts()}
    expanded=[];count=0;checkpoints={}
    for ctx in contexts.values():
        path=bundle/'provenance/search/complete/checkpoints'/(ctx['id']+'.json');cp=read(path)
        payload=dict(programs=engine._prepare(programs,costs,ctx,ddr),topologies=graphs,
            context=ctx,k=20,score_scale=engine.SCORE_SCALE,implementation=proof['implementation'])
        fields=('counts','gate_counts','rankings','topology_strata','resource_strata')
        if (cp['checkpoint_sha256']!=engine._digest({k:v for k,v in cp.items() if k!='checkpoint_sha256'})
            or cp['input_sha256']!=engine._digest(payload) or not cp['verification']['passed']
            or cp['verification']['source_sha256']!=proof['implementation']['oracle.cc']
            or cp['verification']['result_sha256']!=engine._digest({k:cp[k] for k in fields})):
            raise ValueError('historical full checkpoint/source/oracle result differs')
        count+=cp['counts']['enhanced'];checkpoints[str(path)]=sha256(path)
        for t in cp['rankings']['enhanced']:
            terminal=engine.Terminal.from_json(t)
            expanded.append({**engine._configuration(terminal,graphs,programs,contexts),
                'score_ns':terminal.score,'score_ms':terminal.score/1e6})
    top=sorted(expanded,key=engine._row_key)[:20]
    for i,r in enumerate(top,1):r['static_rank']=i
    archived=read(bundle/'plans/selection_rankings.json')['complete']
    if top!=archived or count!=966240306 or len(graphs)!=4623 or len(contexts)!=9:
        raise ValueError('historical strongest global Top20/domain differs')
    for r in top:
        if historical_score(r,costs,'complete')['score_ns']!=r['score_ns'] or staged_score(r,costs,'concentrated')['score_ns']!=r['score_ns']:
            raise ValueError('complete scalar formula or equivalent switch contract differs')
    return dict(count=count,topologies=len(graphs),contexts=len(contexts),Top20=top,
        scalar_historical_and_current_equivalent=True,proof=proof,checkpoints=checkpoints,
        cached_full_domain_oracle_validated=True,cold_enumeration_not_repeated=True,
        component_records=len(records),programs=len(programs),
        top1_terms=historical_score(top[0],costs,'complete'))


def archive(root):
    index=read(root/'EVIDENCE_INDEX.json');bundle=root/index['archived_bundle'];run=root/index['archived_run']
    if sha256(bundle/'manifest.json')!=index['bundle_manifest_sha256']:raise ValueError('archive manifest differs')
    manifest=read(bundle/'manifest.json')
    if digest({k:v for k,v in manifest.items() if k!='bundle_id'})[:24]!=manifest['bundle_id']:
        raise ValueError('archive manifest identity invalid')
    verify_inventory(bundle,manifest['files']);search=full_search(bundle)
    if search['Top20'][0]['id']!=EXPECTED:raise ValueError('historical Top1 configuration differs')
    receipts=[];performance=None
    for path in sorted((run/'accepted').glob('*/*.json')):
        item=read(path)
        if item['task'].get('candidate_id')!=EXPECTED:continue
        verify_inventory(run,item['files']);raw=run/item['raw'];task=item['task']
        if Path(raw/'task_hash').read_text().strip()!=digest(task) or Path(raw/'boot_id').read_text().strip()!=item['boot_id']:
            raise ValueError('historical task or boot mismatch')
        summary=validate_summary(raw,task,bundle)
        if task['kind']=='performance':summary['fps']=1000/summary['ii_ms']
        if summary!=item['summary']:raise ValueError('historical full output/affinity/summary no longer replays')
        if task['kind']=='performance':
            rows=[json.loads(s) for s in (raw/'result.jsonl').read_text().splitlines() if s]
            if any(r.get('queue_timing_enabled') is not False or r.get('defer_result_write_enabled') is not True for r in rows):
                raise ValueError('historical detailed logging/deferred output mode differs')
            kept=rows[task['warmup']:task['warmup']+task['scored']]
            interval=(kept[-1]['completion_ms']-kept[0]['completion_ms'])/(len(kept)-1)
            if len(rows)!=28 or len(kept)!=16 or abs(interval-summary['ii_ms'])>1e-8:
                raise ValueError('historical completion window differs')
            if len({r['output_mode'] for r in rows})!=1:raise ValueError('mixed output accounting')
            performance=dict(FPS=1000/interval,II_ms=interval,boot=item['boot_id'],
                output_mode=rows[0]['output_mode'],warmup=task['warmup'],scored=task['scored'],tail=task['tail'],
                environment=task['environment'],executor_sha256=task['executor_sha256'],
                stage_service_ms=[statistics.mean(r[f'stage{i}_ms'] for r in kept) for i in range(7)])
        receipts.append(dict(path=str(path),sha256=sha256(path),kind=task['kind'],raw=str(raw),
            all_raw_hashes_validated=True,full_output_or_timing_affinity_validated=True))
    if len(receipts)!=2 or performance is None:raise ValueError('historical Top1 qualification/performance pair incomplete')
    return dict(archive=str(root),bundle=str(bundle),bundle_sha256=sha256(bundle/'manifest.json'),
        search=search,receipts=receipts,performance=performance,
        frozen_score_sources={str(p):sha256(p) for p in (bundle/'sources').rglob('*.py')
            if str(p).endswith(('/host_work_ablation/model.py','/model_ablation/model.py','/resource_envelope/model.py'))})


def audit():
    start=time.monotonic();cohorts=[archive(PAPER/n) for n in ARCHIVES]
    cfg={**configuration(cohorts[-1]['search']['Top20'][0]),'id':EXPECTED}
    if any(configuration(c['search']['Top20'][0])!=configuration(cfg) for c in cohorts):
        raise ValueError('historical cohorts did not execute the same configuration')
    current=read(CURRENT/'manifest.json');verify_inventory(CURRENT,current['files'])
    old=Path(cohorts[-1]['bundle']);m=read(old/'manifest.json')
    programs=read(old/'plans/programs.json')
    nodes={str(v) for s in cfg['stages'] for key in ('input_node_ids','output_node_ids') for v in programs[s['program']][key]}
    needed=[k for k in m['files'] if k.startswith(('runtime/','firmware/','inputs/','references/'))
        or k.startswith('programs/') and k.split('/')[1] in {s['program'] for s in cfg['stages']}]
    needed=[k for k in needed if not k.startswith(('inputs/frontiers/','references/')) or Path(k).stem in nodes]
    matched={};different=[]
    for leaf in needed:
        if leaf not in current['files'] or current['files'][leaf]['sha256']!=m['files'][leaf]['sha256']:
            different.append(leaf)
        else:matched[leaf]=m['files'][leaf]['sha256']
    if different:raise ValueError('historical/current required input/module/runtime differs: '+str(different))
    historical_executor=read(old/'plans/executors.json')['policy']
    if historical_executor['sha256']!=current['old_runner_sha256']:
        raise ValueError('available old executor is not the historical one')
    doc=dict(cohorts=cohorts,configuration=cfg,required_frontier_nodes=sorted(nodes),required_assets_byte_identical=matched,
        historical_executor=historical_executor,current_phase_executor_sha256=current['phase_runner_sha256'],
        current_parent=str(CURRENT),current_parent_sha256=sha256(CURRENT/'manifest.json'),
        model_switches=dict(thread_weights=True,team_guard=True,host_cpu_demand=True,host_concentration=True),
        model_formula=dict(core_demand='sum computeCPU_i * weight_ij mapped to c plus caller boundaryCPU',
            capacity='max CPU stage, sum all VTA islands, DDR, 15 core subsets, executed serial-group sums',
            host_concentration='max_c (core_demand_c + host_CPU if c allowed)',
            team_service='max_c sum isolated_service_i for CPU groups containing c',
            final='maximum of capacity, host_concentration and team_service'),
        subsequent_fine_host_has_host_concentration_disabled=True,
        historical_output_mode=cohorts[-1]['performance']['output_mode'],
        recent_phase_performance_output_mode='raw',
        bridge_required_to_separate_executor_and_output_accounting=True,
        no_board_connection=True,new_DNN_inferences=0,seconds=time.monotonic()-start,
        source_sha256=sha256(__file__),recorded_at=now(),goal_complete=False)
    freeze(OUTPUT/'BASELINE_FREEZE.json',doc)
    return dict(historical_FPS=[c['performance']['FPS'] for c in cohorts],
        full_domain_count=cohorts[0]['search']['count'],configuration=cfg['id'],
        byte_identical_required_assets=len(matched),historical_output_mode=doc['historical_output_mode'],
        executor_changed=historical_executor['sha256']!=current['phase_runner_sha256'])
