"""Find already-measured host-only matches before proposing another board run.

Filters only task metadata. Performance summaries cannot influence matching.
"""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256
from finite_service.common import freeze
from model_semantics_audit.common import metadata_projection

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/current_model_gap_01/existing_host_matches'


def matches(c):
    if not isinstance(c,dict):return False
    stages=c.get('stages',[])
    if not isinstance(stages,list) or any(not isinstance(s,dict) for s in stages):return False
    if [(s.get('device'),s.get('begin'),s.get('end')) for s in stages]!=[
        ('cpu',0,3),('vta',3,7),('cpu',7,9),('vta',9,13),('cpu',13,14)]:return False
    cpus=[s for s in stages if s['device']=='cpu']
    if [(s.get('threads'),s.get('mask'),s.get('worker_core_map')) for s in cpus]!=[
        (4,[0,1,2,3],[0,1,2,3]),(3,[1,2,3],[1,2,3]),(1,[0],[0])]:return False
    return (c.get('execution_mode')=='off' and c.get('affinity_mode')=='pinned'
            and c.get('control_mask')==[0,1,2,3])


def run():
    source=ROOT/'artifacts/device_completion_holdout_01/KNOWN_EXECUTIONS.json'
    index=read(source)
    freeze(OUT/'PLAN_VALIDATED.json',dict(index=str(source),index_sha256=sha256(source),
        prior_uncompleted_plan_sha256=sha256(OUT/'PLAN.json') if (OUT/'PLAN.json').exists() else None,
        parser_repair='skip null/non-dict component configurations; require actual full-pipeline stages',
        source_sha256=sha256(__file__),no_FPS_filter=True,
        target_CPU_stages='CPU0-2 four pinned 0123; CPU7-8 three pinned 123; CPU13 one pinned 0',
        controls=[0,1,2,3],gate='off',all_VTA_host_positions_retained=True,
        selection_precedes_any_performance_summary_read=True,
        scope='accepted receipts present in the previously frozen canonical inventory; not all historical scripts'))
    matched=[];unparsed=0;accepted=0
    for filename,h in index['files'].items():
        if '/accepted/' not in filename:continue
        p=Path(filename)
        if sha256(p)!=h:raise ValueError('indexed native receipt changed: '+filename)
        try:meta=metadata_projection(p,{'task','boot_id'})
        except ValueError:unparsed+=1;continue
        accepted+=1;task=meta['task'];c=task.get('configuration',{})
        if not matches(c) or not matches({**c,'stages':task.get('stages')}):continue
        matched.append(dict(receipt=filename,receipt_sha256=h,boot_id=meta.get('boot_id'),
            kind=task.get('kind'),configuration_id=c.get('id'),
            VTA_host_masks=[s['mask'] for s in c['stages'] if s['device']=='vta'],
            executor_sha256=task.get('executor_sha256'),runtime_sha256=task.get('runtime_sha256'),
            inputs=task.get('input_indices'),warmup=task.get('warmup'),scored=task.get('scored'),tail=task.get('tail'),
            scope=task.get('spec',{}).get('scope'),phase=task.get('spec',{}).get('phase'),
            FPS_read=False))
    result=dict(matched=matched,parsed_receipts=accepted,unparsed_receipts=unparsed,
        matches=len(matched),index_sha256=sha256(source),source_sha256=sha256(__file__),
        no_FPS_read=True,new_DNN=0,
        host_placement_pairs=sorted({json.dumps(r['VTA_host_masks']) for r in matched}))
    freeze(OUT/'RESULTS.json',result)
    return dict(matches=len(matched),host_placement_pairs=result['host_placement_pairs'],parsed=accepted)


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,indent=2))
