"""Bounded, offline audit of the current comparison and three direct ablations.

Writes a new evidence index only. Native launches, SSH, model changes and new
searches are not part of this entry. Old outputs and snapshots remain read-only.
"""
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean
from unittest.mock import patch
import importlib
import json
import time

import run_scope_rank as ranks
import run_scope_tail_comparison as current
from cpu_vta_partitioning.common import read, write, sha256, digest, verify_inventory, now
from common_top20.runtime import task, summary
from cpu_policies.open_resources.costs import Costs
from cpu_policies.open_resources.domain import canonical_tuple
from resource_envelope.model import score as envelope_score
from staged_ablation.model import score as staged_score
from audit_latest_dp_window import window_inventory
from audit_formal_window_resources import complement
from scheduler_gap_audit.curves import csv_write

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/experiment_gap_audit_20261112_01'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261112_experiment_gap_audit'
OLD = (
    ('resnet18_model_ablation_01', 'model_ablation'),
    ('resnet18_host_work_ablation_01', 'host_work_ablation'),
    ('resnet18_staged_ablation_01', 'staged_ablation'),
)


def execution_key(cfg, omitted=()):
    """A matching intervention may omit only its declared execution field."""
    values = dict(stages=tuple((s['device'], s['begin'], s['end'], s['threads'],
                               tuple(s['mask']), tuple(s['worker_core_map'])) for s in cfg['stages']),
                  affinity=cfg['affinity_mode'], gate=cfg['execution_mode'],
                  host=tuple(cfg['host_mask']), control=tuple(cfg['control_mask']),
                  pool=tuple(cfg['compute_pool']), reserved=tuple(cfg['reserved_cores']))
    return tuple((k, v) for k, v in values.items() if k not in omitted)


def host_guard_and_capacity(demand, host, allowed):
    concentration = max(v + (host if c in allowed else 0) for c, v in enumerate(demand))
    capacities = []
    bits = sum(1 << c for c in allowed)
    for subset in range(1, 16):
        work = sum(v for c, v in enumerate(demand) if subset & (1 << c))
        if subset & bits == bits:
            work += host
        width = subset.bit_count()
        capacities.append((work + width - 1) // width)
    return concentration, max(capacities)


def retain(sources, path):
    path = Path(path).resolve()
    sources[str(path)] = sha256(path)


def load_rows(raw):
    return [json.loads(s) for s in (Path(raw) / 'result.jsonl').read_text().splitlines() if s]


def audit_current(sources):
    prepared = current.assert_sources()
    plan, pool, order = current.selection()
    state = read(current.RUN / 'state.json')
    if state['status'] != 'COMPLETE' or state['completed_configurations'] != 88:
        raise ValueError('the complete approved comparison is required')
    boot = state['boot_id']; layouts = {}; count = inferences = 0
    for network, candidates in pool.items():
        bundle = current.OUT / network / 'capsule'; current.validate(bundle)
        for cid, cfg in candidates.items():
            fingerprints = None
            for phase in ('qualify', 'short'):
                doc = task(bundle, cfg, phase)
                path = current.RUN / 'accepted' / boot / (doc['task_id'] + '.json')
                receipt = read(path)
                if receipt['task'] != doc or receipt['boot_id'] != boot:
                    raise ValueError('current native identity changed')
                verify_inventory(receipt['raw'], receipt['files'])
                calculated = summary(receipt['raw'], doc, bundle)
                if calculated != receipt['summary']:
                    raise ValueError('current qualification or performance cannot be replayed')
                retain(sources, path); count += 1
                inferences += sum(doc[k] for k in ('warmup', 'scored', 'tail'))
                if phase == 'qualify':
                    fingerprints = calculated['output_fingerprints']
                    continue
                if fingerprints != calculated['output_fingerprints']:
                    raise ValueError('current timed output differs from qualification')
                rows = load_rows(receipt['raw'])
                inventory = window_inventory(rows, cfg['stages'], doc['warmup'], doc['scored'])
                holds = [(round(r[f'stage{i}_vta_mutex_acquire_ms'] * 1e6),
                          round(r[f'stage{i}_vta_mutex_release_ms'] * 1e6))
                         for r in rows for i, s in enumerate(cfg['stages']) if s['device'] == 'vta']
                held, gaps = complement(inventory['begin_ns'], inventory['end_ns'], holds)
                if abs(inventory['duration_ns'] / 15 / 1e6 - calculated['II_ms']) > 1e-6:
                    raise ValueError('window and official interval differ')
                layouts[(network, cid)] = dict(configuration=cfg, measurement=calculated,
                    inventory=inventory, receipt=str(path),
                    VTA_software_lock_occupancy_percent=100*held/inventory['duration_ns'],
                    software_lock_not_physical_compute_utilization=True)
    if count != 176 or inferences != 4312:
        raise ValueError('approved common cohort task count differs')
    for p in (current.OUT/'FREEZE.json', current.OUT/'LISTS.json', current.OUT/'PLAN.json',
              current.OUT/'PREPARATION.json', current.RUN/'state.json'):
        retain(sources, p)
    return dict(boot=boot, configurations=88, sessions=count, inferences=inferences,
                summaries_and_complete_outputs_replayed=True, preparation=prepared['bundles']), layouts


def audit_old(run_name, module_name, latest_cost_sha, latest_lists, sources):
    run = ROOT / 'runs' / run_name; state = read(run/'state.json'); bundle = Path(state['bundle'])
    protocol = read(run/'protocol.json'); lists = read(bundle/'plans/selection_rankings.json')
    pool = read(bundle/'plans/static_pool.json'); freeze = read(run/'selection_freeze.json')
    for name, value in (('pool_sha256', bundle/'plans/static_pool.json'),
                        ('rankings_sha256', bundle/'plans/selection_rankings.json')):
        if sha256(value) != freeze[name]:
            raise ValueError('old frozen selection changed')
    if freeze['lists'] != {k: [r['id'] for r in rows] for k, rows in lists.items()}:
        raise ValueError('old frozen ordered lists differ')
    if sha256(bundle/'plans/isolated_costs.json') != latest_cost_sha:
        raise ValueError('old ResNet component table differs from current one')
    runtime = importlib.import_module(module_name + '.runtime')
    expected_tasks = {d['task_id']: d for d in read(run/'tasks.json')}
    receipts = list((run/'accepted'/state['boot_id']).glob('*.json'))
    if len(receipts) != len(expected_tasks):
        raise ValueError('old session count differs from frozen tasks')
    measured, windows = {}, {}; total = 0
    for path in receipts:
        receipt = read(path); doc = receipt['task']
        if doc != expected_tasks[doc['task_id']] or receipt['boot_id'] != state['boot_id']:
            raise ValueError('old native task or boot differs')
        verify_inventory(run, receipt['files'])
        raw = run/receipt['raw']
        calculated = runtime.validate_summary(raw, doc, bundle)
        if calculated != receipt['summary']:
            raise ValueError('old native summary cannot be replayed')
        retain(sources, path); total += sum(doc[k] for k in ('warmup','scored','tail'))
        if doc['kind'] == 'performance':
            cid = doc['candidate_id']; measured[cid] = calculated
            windows[cid] = window_inventory(load_rows(raw), pool[cid]['stages'], doc['warmup'], doc['scored'])
    if set(measured) != set(pool):
        raise ValueError('old candidate pool measurement incomplete')
    decision = read(run/'analysis/analysis.json')['decision']; winners = []; models = []
    settings = importlib.import_module(module_name+'.common').settings
    for key, rows in lists.items():
        models.append(dict(model=key, switches=settings(key), measured_budget=len(rows),
            context_counts=dict(Counter(c['context_id'] for c in rows)),
            current_complete20_coverage=len({c['id'] for c in rows} & {c['id'] for c in latest_lists})))
        for budget, value in decision['methods'][key]['budgets'].items():
            cid = value['selected']; candidates = rows[:int(budget)]
            minimum = min(measured[c['id']]['ii_ms'] for c in candidates)
            if cid not in {c['id'] for c in candidates} or abs(minimum-value['ii_ms']) > 1e-8:
                raise ValueError('old budget winner differs from measured frozen list')
            exhausted = [r for r in windows[cid]['stages'] if r['stage_exhausted_before_window_end']]
            winners.append(dict(model=key, budget=int(budget), configuration=cid,
                FPS=measured[cid]['fps'], II_ms=minimum,
                exhausted_stages=[r['stage'] for r in exhausted],
                max_window_after_stage_exhaustion_percent=max(
                    [r['window_after_last_stage_call_percent'] for r in exhausted] or [0.])))
    proof = read(bundle/'plans/search_proofs.json')
    for model, value in proof.items():
        if value.get('pipeline_FPS_used') is not False:
            raise ValueError('old ranking proof does not exclude pipeline FPS')
    for p in (run/'state.json',run/'protocol.json',run/'selection_freeze.json',
              bundle/'plans/isolated_costs.json',bundle/'plans/selection_rankings.json',
              bundle/'plans/search_proofs.json',bundle/'plans/executors.json'):
        retain(sources,p)
    result = dict(run=run_name, boot=state['boot_id'], configurations=len(pool), sessions=len(receipts),
        DNN_inferences=total, protocol=protocol['performance'], models=models, budget_winners=winners,
        same_775_cost_file_as_current=True, cost_sha256=latest_cost_sha,
        executor=read(bundle/'plans/executors.json'),
        all_complete_outputs_affinities_and_raw_summaries_replayed=True,
        candidates_with_early_stage_exhaustion=sum(any(s['stage_exhausted_before_window_end']
                                                      for s in v['stages']) for v in windows.values()),
        affected_selected_budgets=[w for w in winners if w['exhausted_stages']])
    return result, pool, measured, windows


def audit_plateau(network, rows, costs, native):
    details = []
    for cfg in rows:
        d = envelope_score(cfg,costs); o = d['original_components']
        terms = dict(VTA_sum=o['vta_service_sum_ns'], CPU_stage=o['cpu_service_max_ns'],
                     CPU_capacity=max(o['subset_capacity_ns']), CPU_team=d['CPU_team_guard_ns'],
                     host_concentration=d['host_concentration_guard_ns'], DDR=o['ddr_service_ns'])
        if max(terms.values()) != cfg['score_ns']:
            raise ValueError('complete score differs from frozen Top20')
        h, cap = host_guard_and_capacity(o['cpu_core_demand_ns'],o['host_core_ns'],cfg['host_mask'])
        if h != d['host_concentration_guard_ns'] or cap != max(o['subset_capacity_ns']):
            raise ValueError('independent host/subset accounting differs')
        if len(cfg['host_mask']) == 1 and h != cap:
            raise ValueError('fixed-host concentration redundancy was violated')
        details.append(dict(network=network, configuration=cfg['id'], score_ns=cfg['score_ns'],
            terms_ns=terms, binding=[k for k,v in terms.items() if v==cfg['score_ns']],
            FPS=native[(network,cfg['id'])]['measurement']['FPS'],
            II_ms=native[(network,cfg['id'])]['measurement']['II_ms'],
            core3_CPU_stages=[s['program'] for s in cfg['stages'] if s['device']=='cpu' and 3 in s['mask']]))
    return dict(network=network, records=details,
        distinct_scores=len({r['score_ns'] for r in rows}),
        topology_count=len({tuple((s['device'],s['begin'],s['end']) for s in r['stages']) for r in rows}),
        bindings=dict(Counter(k for r in details for k in r['binding'])),
        term_ranges_ns={k:[min(d['terms_ns'][k] for d in details),max(d['terms_ns'][k] for d in details)]
                        for k in details[0]['terms_ns']},
        measured_FPS_range=[min(d['FPS'] for d in details),max(d['FPS'] for d in details)],
        first_II_loss_to_own20_best_percent=100*(details[0]['II_ms']/min(d['II_ms'] for d in details)-1),
        full_domain_tie_class_size_unknown_from_capped_Top20=True)


def host_only_fixed_pool(network, rows, programs, costs, native):
    balanced = Costs(costs.records,programs=programs,ddr=costs.ddr,distribution='balanced')
    records = []
    for cfg in rows:
        ignored = staged_score(cfg,balanced,'coarse_no_host')['score_ns']
        included = staged_score(cfg,balanced,'coarse_host')['score_ns']
        if ignored != cfg['score_ns']:
            raise ValueError('host-off coarse score is not identical to singleton Tarnawski score')
        d = envelope_score(cfg,costs)
        if included != max(d['original'],d['host_concentration_guard_ns'],d['CPU_team_guard_ns']):
            raise ValueError('other factors are not redundant on this fixed singleton pool')
        measured = native[(network,cfg['id'])]['measurement']
        records.append(dict(configuration=cfg['id'],ignored_ns=ignored,included_ns=included,
                            actual_II_ms=measured['II_ms'],FPS=measured['FPS']))
    summaries=[]
    for name in ('ignored','included'):
        field=name+'_ns'
        best=min(rows,key=lambda cfg:(next(r[field] for r in records if r['configuration']==cfg['id']),canonical_tuple(cfg)))
        summaries.append(dict(host_CPU=name,mean_absolute_percent_error=mean(
            abs(r[field]/1e6/r['actual_II_ms']-1)*100 for r in records),
            first=best['id'],first_actual_FPS=native[(network,best['id'])]['measurement']['FPS'],
            distinct_scores=len({r[field] for r in records})))
    return dict(network=network,pool_size=20,records=records,summary=summaries,
        other_factor_redundancy_checked_for_all20=True,
        retrospective_diagnostic_not_new_full_domain_Top20=True,no_new_formula=True)


def gate_pairs(cohort, pool, measured, windows=None):
    result=[]
    for a,b in combinations(pool.values(),2):
        if a['execution_mode']==b['execution_mode'] or execution_key(a,('gate',))!=execution_key(b,('gate',)):
            continue
        off,on=(a,b) if a['execution_mode']=='off' else (b,a)
        def fps(cid):
            d=measured[cid]; return d.get('FPS',d.get('fps'))
        result.append(dict(cohort=cohort,off=off['id'],on=on['id'],off_FPS=fps(off['id']),on_FPS=fps(on['id']),
                           off_vs_on_FPS_percent=100*(fps(off['id'])/fps(on['id'])-1),
                           same_cut_threads_masks_host_control_affinity=True,
                           on_early_exhaustion=any(r['stage_exhausted_before_window_end']
                              for r in windows[on['id']]['stages']) if windows else False))
    return result


def run():
    if OUT.exists():
        raise FileExistsError('preserve the completed offline audit')
    started=time.monotonic();sources={};rank_plan=ranks.assert_plan()
    with patch('socket.socket.connect',side_effect=AssertionError('offline audit network denied')), \
         patch('subprocess.Popen',side_effect=AssertionError('offline audit subprocess denied')):
        current_result,native=audit_current(sources)
        lists=read(current.OUT/'LISTS.json');old_results=[];pairs=[];plateaus=[];host_pool=[]
        for run_name,module_name in OLD:
            result,pool,measurements,windows=audit_old(run_name,module_name,
                rank_plan['networks']['resnet18']['cost_sha256'],lists['resnet18']['original_complete'],sources)
            old_results.append(result);pairs+=gate_pairs(run_name,pool,measurements,windows)
        for network in lists:
            programs,costs,ddr=ranks.inputs_for(rank_plan,network,'original')
            plateaus.append(audit_plateau(network,lists[network]['original_complete'],costs,native))
            host_pool.append(host_only_fixed_pool(network,lists[network]['tarnawski'],programs,costs,native))
            for variant in ('original','run_scope'):
                ranks.verify_one(network,variant)
                retain(sources,ranks.OUT/network/variant/'PROOF.json')
            if network=='resnet18':
                pool={key[1]:v['configuration'] for key,v in native.items() if key[0]==network}
                ms={key[1]:v['measurement'] for key,v in native.items() if key[0]==network}
                ws={key[1]:v['inventory'] for key,v in native.items() if key[0]==network}
                pairs+=gate_pairs('latest_tail24',pool,ms,ws)
        # Consult the unified trace inventory, not just the earlier14-entry projection.
        inventory=ROOT/'artifacts/isolated_trace_inventory_20261112_01/INVENTORY.json'
        trace_index=read(inventory)
        for path,wanted in trace_index['receipt_sources'].items():
            if sha256(path)!=wanted:raise ValueError('unified isolated receipt changed')
        retain(sources,inventory)
        timeline=ROOT/'artifacts/current_pipeline_timeline_20261112_01/analysis/RESULTS.json'
        trace=read(timeline)
        for path,wanted in trace['sources'].items():
            if sha256(path)!=wanted:raise ValueError('matched timeline source changed')
            if '/accepted/' in path:
                receipt=read(path);verify_inventory(receipt['raw'],receipt['files'])
        retain(sources,timeline)
        curves=ROOT/'artifacts/frozen_workloads_20261112_01/RESULTS.json';cv=read(curves)
        for path,wanted in cv['sources'].items():
            if sha256(path)!=wanted:raise ValueError('network curve source changed')
        retain(sources,curves)
    result=dict(recorded_at=now(),current=current_result,old_ablations=old_results,
        plateaus=plateaus,host_only_same_singleton_pools=host_pool,gate_pairs=pairs,
        unified_isolated_observed_receipts=trace_index['unique_observed_receipts'],
        workload_signatures=cv['total_real_thread_curves'],
        native_receipts_fully_replayed=176+sum(r['sessions'] for r in old_results),
        no_board_connection=True,new_native_sessions=0,new_DNN_inferences=0,
        no_new_search=True,scoring_DP_runtime_executor_unchanged=True,
        subprocess_and_network_denied_during_audit=True,
        seconds=time.monotonic()-started,sources=sources)
    OUT.mkdir(parents=True);REPORT.mkdir(parents=True,exist_ok=True)
    write(OUT/'AUDIT.json',result)
    csv_write(REPORT/'old_ablation_budget_coverage.csv',[
        dict(cohort=r['run'],boot=r['boot'],**w) for r in old_results for w in r['budget_winners']])
    csv_write(REPORT/'gate_pair_cross_cohort.csv',pairs)
    csv_write(REPORT/'current_constraint_terms.csv',[
        dict(network=p['network'],configuration=r['configuration'],score_ns=r['score_ns'],
             FPS=r['FPS'],II_ms=r['II_ms'],binding=r['binding'],**r['terms_ns'])
        for p in plateaus for r in p['records']])
    csv_write(REPORT/'host_only_same_pool.csv',[
        dict(network=p['network'],**r) for p in host_pool for r in p['records']])
    csv_write(REPORT/'current_window_resource.csv',[
        dict(network=n,configuration=cid,FPS=d['measurement']['FPS'],II_ms=d['measurement']['II_ms'],
             VTA_software_lock_occupancy_percent=d['VTA_software_lock_occupancy_percent'],
             stage_counts=d['measurement']['stage_completion_counts_at_window']) for (n,cid),d in native.items()])
    write(OUT/'MANIFEST.json',dict(source_sha256=sha256(__file__),
        files={str(p):sha256(p) for p in [OUT/'AUDIT.json',*REPORT.iterdir()] if p.is_file()}))
    print(json.dumps(dict(fully_replayed_receipts=result['native_receipts_fully_replayed'],
        old_exhaustion={r['run']:dict(total=r['candidates_with_early_stage_exhaustion'],
            affected_selected=r['affected_selected_budgets']) for r in old_results},
        plateaus=[{k:v for k,v in p.items() if k!='records'} for p in plateaus],
        host_only=[dict(network=p['network'],summary=p['summary']) for p in host_pool],
        seconds=result['seconds']),ensure_ascii=False))
    return result


if __name__=='__main__':run()
