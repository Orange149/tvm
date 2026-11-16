"""Read-only cross-batch evidence audit; never a ranking or board controller.

The complete-domain static rankings, finite-pool temporal rankings, historical
diagnostics and prospective resource-layout test remain separate cohorts.
"""
from collections import Counter
from pathlib import Path
from statistics import mean
import csv
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from cpu_vta_partitioning.common import digest, now, read, sha256, verify_inventory, write
from cpu_policies.open_resources.costs import Costs
from cpu_policies.open_resources.domain import contexts
from policy_branch_eval.ranking import engine_from, inputs, replay_original
from scheduler_gap_audit.curves import csv_write
from staged_ablation.model import score

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'artifacts/research_evidence_audit_01'
CROSS=ROOT/'artifacts/cross_network_model_01'


def check_files(files):
    for path, expected in files.items():
        if sha256(path)!=expected:
            raise ValueError('evidence hash differs: '+str(path))


def budget_prefix(rows, count):
    if count not in (1,5,20) or len(rows)!=20 or len({r['configuration_id'] for r in rows})!=20:
        raise ValueError('twenty unique measured ordered items required')
    for r in rows:
        if r['measured_II_ms']<=0 or abs(r['FPS']*r['measured_II_ms']-1000)>1e-8:
            raise ValueError('FPS and completion interval differ')
    best=min(enumerate(rows[:count],1),key=lambda pair:(pair[1]['measured_II_ms'],pair[0]))
    return dict(rank=best[0],configuration_id=best[1]['configuration_id'],
                FPS=best[1]['FPS'],II_ms=best[1]['measured_II_ms'])


def physical_baseline(network,programs,records,ddr,hashes,parent,units):
    root=parent.parent/'tarnawski'
    proof=read(root/'proof.json')
    if proof['component_hashes']!=hashes or proof['selected_method']!='single_core' or proof['pipeline_FPS_used']:
        raise ValueError('physical CPU baseline inputs differ')
    engine=engine_from(proof,'goal_'+network+'_baseline')
    cost=Costs(records,programs=programs,ddr=ddr,distribution='balanced')
    graphs=engine._topologies(programs,units,3)
    context={c['id']:c for c in contexts()};rows=[];count=0;sources={}
    for ctx in context.values():
        path=root/'search'/(ctx['id']+'.json');cp=read(path)
        payload=dict(programs=engine._prepare(programs,cost,ctx,ddr),topologies=graphs,
                     context=ctx,k=20,score_scale=engine.SCORE_SCALE,implementation=proof['implementation'])
        fields=('counts','gate_counts','rankings','topology_strata','resource_strata')
        if (cp['checkpoint_sha256']!=engine._digest({k:v for k,v in cp.items() if k!='checkpoint_sha256'})
            or cp['input_sha256']!=engine._digest(payload) or not cp['verification']['passed']
            or cp['verification']['source_sha256']!=proof['implementation']['oracle.cc']
            or cp['verification']['result_sha256']!=engine._digest({k:cp[k] for k in fields})):
            raise ValueError('baseline complete-domain checkpoint/oracle changed')
        count+=cp['counts']['single_core'];sources[str(path)]=sha256(path)
        for raw in cp['rankings']['single_core']:
            terminal=engine.Terminal.from_json(raw)
            rows.append(dict(**engine._configuration(terminal,graphs,programs,context),
                             score_ns=terminal.score,score_ms=terminal.score/1e6))
    rows=sorted(rows,key=engine._row_key)[:20]
    for i,row in enumerate(rows,1):
        row['static_rank']=i
        if score(row,cost,'tarnawski')['score_ns']!=row['score_ns']:
            raise ValueError('baseline scalar formula disagrees')
    if rows!=read(root/'rankings.json')['methods']['single_core']['rankings'] or count!=proof['selected_count']:
        raise ValueError('baseline full Top20 replay differs')
    return dict(count=count,topologies=len(graphs),contexts=len(context),sources=sources,
                checkpoint_oracle_revalidated=True,cold_enumeration_rerun=False)


def complete_searches():
    results={}
    for network in ('resnet18','yolov3_tiny416'):
        programs,records,ddr,hashes,parent,units=inputs(network)
        proof=read(parent/'proof.json');engine=engine_from(proof,'goal_'+network+'_main')
        cost=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
        rows,graphs,main=replay_original(engine,programs,cost,ddr,hashes,parent,units)
        baseline=physical_baseline(network,programs,records,ddr,hashes,parent,units)
        cpu=Counter(r['configuration']['stages'][0]['device'] for r in records.values())
        results[network]=dict(main=main,tarnawski=baseline,component_hashes=hashes,
            real_component_records=len(records),component_device_counts=dict(cpu),programs=len(programs),units=units)
    return results


def common_measurements():
    from common_top20.analyze import load_result
    from common_top20.prepare import validate
    from pool_temporal.ranking import validate_rank
    manifest=read(CROSS/'FINAL_REPORT_MANIFEST.json')
    check_files(manifest['results']);check_files(manifest['files'])
    validate_rank(CROSS/'pool_temporal')
    short=read(CROSS/'FINAL_SHORT_RESULTS.json');base=read(CROSS/'common_top20/SHORT_RESULTS.json')
    if sha256(CROSS/'common_top20/SHORT_RESULTS.json')!=short['common_short_sha256']:
        raise ValueError('final short results differ from common measurements')
    run=ROOT/'runs/cross_network_common_top20_01';state=read(run/'state.json')
    if state['status']!='LONG_COMPLETE' or short['boot_id']!=state['boot_id'] or base['boot_id']!=state['boot_id']:
        raise ValueError('common measurement cohort incomplete or mixed boot')
    native={};receipt_hashes={};sessions=0
    for network in ('resnet18','yolov3_tiny416'):
        bundle=CROSS/'common_top20'/network/'capsule';validate(bundle)
        ranked=read(bundle/'plans/rankings.json')
        for identity,cfg in ranked['pool'].items():
            qualified,qsource=load_result(run,bundle,cfg,'qualify',state['boot_id'])
            timed,tsource=load_result(run,bundle,cfg,'short',state['boot_id'])
            if qualified['summary']['output_fingerprints']!=timed['summary']['output_fingerprints']:
                raise ValueError('common timing output differs from qualification')
            m=timed['summary']
            if m['kept_timestamps']!=16 or m['gaps']!=15:
                raise ValueError('common short completion window differs')
            native[network,identity]=m;sessions+=2
            receipt_hashes[qsource['path']]=qsource['sha256'];receipt_hashes[tsource['path']]=tsource['sha256']
    budgets=[]
    for network,data in short['networks'].items():
        for method,description in data['methods'].items():
            rows=sorted([r for r in short['rows'] if r['network']==network and r['method']==method],key=lambda r:r['rank'])
            if [r['rank'] for r in rows]!=list(range(1,21)):
                raise ValueError('final method ranking incomplete')
            for r in rows:
                m=native[network,r['configuration_id']]
                if abs(r['measured_II_ms']-m['II_ms'])>1e-10:
                    raise ValueError('final ranking row differs from native receipt')
            for k in (1,5,20):
                recomputed=budget_prefix(rows,k);old=description['budgets'][str(k)]
                if any(recomputed[f]!=old[f] for f in ('rank','configuration_id','II_ms')):
                    raise ValueError('selection budget differs from actual prefix')
                budgets.append(dict(network=network,method=method,budget=k,**recomputed,
                    boot_id=state['boot_id'],scope='same-boot full measured pool; temporal ranking is pool-local'))
    long=read(CROSS/'FINAL_LONG_RESULTS.json');long_rows=[]
    if long['boot_id']!=state['boot_id']:
        raise ValueError('long review cohort differs')
    for row in long['rows']:
        network,identity=row['network'],row['configuration_id']
        cfg=read(CROSS/'common_top20'/network/'capsule/plans/rankings.json')['pool'][identity]
        item,source=load_result(run,CROSS/'common_top20'/network/'capsule',cfg,'long',state['boot_id'])
        if item['summary']!= {k:v for k,v in row.items() if k not in ('network','configuration_id')}:
            raise ValueError('long review summary differs from native receipt')
        if row['kept_timestamps']!=64 or row['gaps']!=63:
            raise ValueError('long window count differs')
        receipt_hashes[source['path']]=source['sha256'];long_rows.append(row)
    return dict(boot_id=state['boot_id'],native_configurations=len(native),short_sessions=sessions,
        short_DNN_inferences=29*len(native),long_sessions=len(long_rows),long_DNN_inferences=104*len(long_rows),
        budgets=budgets,long_results=long_rows,source_receipts=receipt_hashes,
        no_new_measurement=True,cohorts_not_merged=True)


def curves():
    path=CROSS/'isolated_CPU_curves.csv'
    rows=list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')))
    tables=[];evidence={str(path):sha256(path)}
    for network in ('resnet18','yolov3_tiny416'):
        _,records,_,_,_,_=inputs(network)
        group={}
        for row in rows:
            if row['network']!=network:continue
            original=records[row['source_key']]['cost']
            for field in ('service_ms','run_ms','run_core_ms','boundary_core_ms','set_ms','get_ms'):
                if abs(float(row[field])-original[field])>1e-8:
                    raise ValueError('thread curve differs from genuine isolated source')
            key=row['program'];group.setdefault(key,{})[int(row['threads'])]=row
        for program,versions in sorted(group.items()):
            if set(versions)!=set(range(1,5)):
                raise ValueError('real thread curve incomplete')
            base=float(versions[1]['service_ms']);base_CPU=float(versions[1]['run_core_ms'])
            times=[float(versions[t]['service_ms']) for t in range(1,5)]
            for t in range(1,5):
                r=versions[t];service=float(r['service_ms'])
                tables.append(dict(network=network,program=program,threads=t,
                    service_ms=service,compute_CPU_ms=float(r['run_core_ms']),
                    speedup=base/service,compute_CPU_growth=float(r['run_core_ms'])/base_CPU,
                    marginal_wall_reduction_percent=None if t==1 else 100*(times[t-2]-service)/times[t-2],
                    nonmonotonic_step=t>1 and service>times[t-2],
                    input_tensors=int(r['input_tensors']),output_tensors=int(r['output_tensors']),
                    output_frontier_bytes=int(r['output_frontier_bytes']),
                    source_key=r['source_key'],candidates_removed=False))
    return dict(rows=tables,sources=evidence)


def audit():
    started=time.monotonic();OUTPUT.mkdir(parents=True,exist_ok=True)
    main=complete_searches();common=common_measurements();thread_curves=curves()
    # Re-check the prospective batch with its original frozen controller.
    import finite_service_holdout_board as holdout
    holdout_state=read(holdout.RUN/'state.json')
    if holdout_state['status']!='COMPLETE':raise ValueError('prospective cohort incomplete')
    receipts={};count=0
    for spec in holdout.specs():
        bundle=holdout.POOL/spec['network']/'capsule';holdout.validate(bundle)
        task=holdout.make_task(bundle,spec)
        path=holdout.RUN/'accepted'/holdout_state['boot_id']/(task['task_id']+'.json')
        item=read(path)
        if item['task']!=task or item['boot_id']!=holdout_state['boot_id']:
            raise ValueError('prospective task or boot changed')
        verify_inventory(item['raw'],item['files'])
        if holdout.summary(item['raw'],task,bundle)!=item['summary']:
            raise ValueError('prospective output/affinity/window no longer validates')
        if Path(item['raw'],'task_hash').read_text().strip()!=digest(task):
            raise ValueError('prospective task hash changed')
        receipts[str(path)]=sha256(path);count+=1
    prospective=read(holdout.POOL/'analysis/RESULTS.json')
    if prospective['source_receipts']!=receipts or count!=12 or holdout_state['DNN_inferences']!=174:
        raise ValueError('prospective analysis evidence differs')
    # Corrected trace and zero-DNN evidence are distinct from DNN performance.
    from yield_semantics.audit import check as yield_check
    yield_path=ROOT/'artifacts/yield_semantics_01/RECEIPT.json';wait=read(yield_path)
    if sha256(wait['raw'])!=wait['raw_sha256'] or yield_check(read(wait['raw']))!=wait['validation']:
        raise ValueError('Yield interface evidence differs')
    trace_dir=ROOT/'artifacts/yolo_scheduler_trace_repair_01'
    trace=read(trace_dir/'FINAL_VALIDATION.json')
    corrected=read(trace_dir/'analysis/RESULTS.json')
    if sha256(trace_dir/'analysis/RESULTS.json')!=read(trace_dir/'DELIVERY.json')['results_sha256']:
        raise ValueError('corrected trace results changed')
    if trace['status']!='PASSED' or not corrected['probe_quality_passed']:
        raise ValueError('corrected trace not valid for selected-window diagnosis')
    check_files(corrected['source_receipts'])
    results=dict(complete_searches=main,common_comparison=common,
        real_thread_curves=thread_curves,prospective_results=prospective,
        zero_DNN_wait_result=wait['validation']['summary'],
        corrected_trace_validation=trace,
        latest_alignment=read(ROOT/'artifacts/yield_semantics_01/alignment/RESULTS.json'),
        costs=read(CROSS/'FINAL_COSTS.json'),
        audit_seconds=time.monotonic()-started,recorded_at=now(),
        audit_source_sha256=sha256(__file__),no_board_connection=True,no_new_DNN_inferences=True,
        full_goal_completion_claimed=False)
    # This is a generated current-state audit, not an immutable raw receipt.
    write(OUTPUT/'AUDIT.json',results,replace=True)
    csv_write(OUTPUT/'same_cohort_budgets.csv',common['budgets'])
    csv_write(OUTPUT/'thread_configuration_table.csv',thread_curves['rows'])
    return dict(passed=True,full_searches={n:r['main']['count'] for n,r in main.items()},
        historical_common_configurations=common['native_configurations'],
        historical_native_receipts=len(common['source_receipts']),
        prospective_native_receipts=count,thread_curve_rows=len(thread_curves['rows']),
        seconds=results['audit_seconds'],board_connections=0,goal_complete=False)


if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False,indent=2))
