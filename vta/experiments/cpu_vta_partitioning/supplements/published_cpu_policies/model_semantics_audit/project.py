"""Two strictly separate inputs: isolated work and diagnostic control work."""
from fractions import Fraction
from pathlib import Path
import json
from cpu_vta_partitioning.common import digest,read,sha256,verify_inventory
from cpu_policies.open_resources.costs import Costs
from policy_branch_eval.common import isolated_inputs_only
from policy_branch_eval.ranking import inputs
from pool_temporal.ranking import model_stages
from .common import ROOT,PARENT,STUDY,RUN,freeze,metadata_projection,protocol


def project(output):
    output=Path(output);audit=read(output/'AUDIT.json');pred=read(PARENT/'PREDICTIONS_FREEZE.json')
    if sha256(PARENT/'PREDICTIONS_FREEZE.json')!=audit['prediction_sha256']:raise ValueError('prediction source changed')
    isolated={}
    with isolated_inputs_only():
        intrinsic=read(STUDY/'pool_temporal/INTRINSICS.json')
        if sha256(STUDY/'pool_temporal/INTRINSICS.json')!=audit['intrinsics_sha256']:raise ValueError('isolated cache changed')
        for key,p in pred['cases'].items():
            network,_=key.split('/');programs,records,ddr,hashes,_,_=inputs(network)
            if hashes!=p['component_hashes']:raise ValueError('macro input identity differs')
            costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
            stages,assumptions=model_stages(network,p['configuration'],programs,costs,intrinsic)
            isolated[key]=dict(configuration=p['configuration'],stages=stages,assumptions=assumptions,
                               expected_baseline=p['matched_long_replay'])
    iso=dict(schema='isolated-semantics-input-v1',origin='isolated_projection',cases=isolated,
             window=protocol()['window'],contains_pipeline_data=False,
             source_hashes=dict(predictions=audit['prediction_sha256'],intrinsics=audit['intrinsics_sha256']))
    freeze(output/'ISOLATED_INPUTS.json',iso)
    # Verify receipt byte hashes without decoding the performance-bearing
    # cases/stages/summary subtrees. Only raw CPU traces become diagnostic input.
    parent_results=PARENT/'analysis/RESULTS.json';delivery=read(PARENT/'DELIVERY.json')
    if sha256(parent_results)!=delivery['results_sha256']:raise ValueError('parent results anchor changed')
    receipts=metadata_projection(parent_results,{'receipts'})['receipts']
    control={};evidence={}
    for filename,expected in receipts.items():
        path=Path(filename)
        if sha256(path)!=expected:raise ValueError('source receipt hash differs')
        meta=metadata_projection(path,{'task','raw','files','boot_id'})
        task=meta['task'];spec=task['spec']
        if spec['phase']!='observed' or spec['scope']!='pipeline':continue
        raw=Path(meta['raw']);verify_inventory(raw,meta['files'])
        if (raw/'boot_id').read_text().strip()!=meta['boot_id'] or (raw/'task_hash').read_text().strip()!=digest(task):
            raise ValueError('control source boot/task differs')
        key=spec['network']+'/'+spec['case']
        if key not in isolated or task['configuration']!=isolated[key]['configuration']:raise ValueError('control configuration differs')
        trace=raw/'phase_trace.jsonl';rows=[json.loads(s) for s in trace.read_text().splitlines()]
        values={}
        for role,operation in (('_producer','feed'),('_collector','collect')):
            group=sorted([r for r in rows if r['stage']==role and r['operation']==operation
                         and task['warmup']<=r['call']<task['warmup']+task['scored']],key=lambda r:r['call'])
            if [r['call'] for r in group]!=list(range(8,72)) or len({r['tid'] for r in group})!=1:
                raise ValueError('control trace missing, duplicated, or has wrong TID')
            cpu=[r['cpu_end_ns']-r['cpu_start_ns'] for r in group]
            wall=[r['wall_end_ns']-r['wall_start_ns'] for r in group]
            if any(c<0 or w<c-10000 for c,w in zip(cpu,wall)):raise ValueError('control clocks inconsistent')
            avg=Fraction(sum(cpu),len(cpu))
            values[role]=dict(CPU_ns=dict(numerator=avg.numerator,denominator=avg.denominator),
                CPU_samples_ns=cpu,wall_samples_ns=wall,TID=group[0]['tid'],mask=task['configuration']['control_mask'],
                scope='feed or write_completed only; excludes control work outside this span',
                blocking_wall_not_used_as_CPU=True)
        control[key]=values;evidence[key]=dict(receipt=str(path),receipt_sha256=expected,
            phase_trace=str(trace),phase_trace_sha256=sha256(trace),boot_id=meta['boot_id'],
            task_id=task['task_id'],executor_sha256=task['executor_sha256'],adapter_sha256=task['adapter_sha256'])
    if set(control)!=set(isolated):raise ValueError('three control cases required')
    diag=dict(schema='diagnostic-control-input-v1',origin='observed_pipeline_control_CPU',diagnostic_only=True,
              forbidden_for_static_prediction=True,cases=control,evidence=evidence,
              full_performance_values_decoded=False,unknown_deallocation_CPU_not_filled=True)
    freeze(output/'DIAGNOSTIC_CONTROL_INPUTS.json',diag)
    freeze(output/'INPUT_FREEZE.json',dict(files={f:sha256(output/f) for f in ('AUDIT.json','ISOLATED_INPUTS.json','DIAGNOSTIC_CONTROL_INPUTS.json')},
        sources=audit['sources'],contains_static_prediction_and_separate_diagnostic_projection=True))
    return dict(cases=len(isolated),pipeline_FPS_decoded=False,diagnostic_controls_separate=True)
