"""Read and replay existing exact stage profiles without launching measurements."""
import hashlib
import json
from pathlib import Path
import statistics
import sys


def inspect_existing(run_dir):
    root=Path(run_dir).resolve();state=json.loads((root/'state.json').read_text())
    bundle=Path(state['bundle']);manifest=json.loads((bundle/'manifest.json').read_text())
    sys.path[:0]=[str(bundle/'sources/cpu_strategy'),str(bundle/'sources/base_controller/src')]
    from cpu_strategy.policy.evidence import Evidence
    from cpu_vta_partitioning.common import verify_inventory
    evidence=Evidence(root,bundle)
    executor=json.loads((bundle/'plans/executors.json').read_text())['policy']['sha256']
    curves={};sources=[]
    for path in sorted((root/'components').glob('*.json')):
        record=json.loads(path.read_text());cfg=record['configuration'];row=cfg['stages'][0]
        if (record['kind']!='measured' or row['device']!='cpu' or cfg['affinity_mode']!='pinned'
                or row['mask']!=list(range(row['threads']))):continue
        relative=str(path.relative_to(root))
        if relative not in state['frozen_files']:raise ValueError('profile is not in the frozen inventory')
        verify_inventory(root,{relative:state['frozen_files'][relative]})
        for entry in [*record['blocks'],record['qualification']]:
            evidence.summary(entry['summary'],entry['document'],entry['boot_id'])
            if entry['document']['executor_sha256']!=executor:raise ValueError('pinned profile executor differs')
            if entry['document']['configuration']!=cfg:raise ValueError('profile configuration differs')
        if any((b['document']['warmup'],b['document']['scored'],b['document']['tail'],b['document']['input_indices'])
               !=(5,32,0,[0]) for b in record['blocks']):raise ValueError('component profiling protocol differs')
        if record['cost']['service_ms']!=statistics.mean(b['summary']['service_ms'] for b in record['blocks']):
            raise ValueError('profile service cost differs from its raw-backed blocks')
        curves.setdefault(row['program'],{})[str(row['threads'])]=record['cost']['service_ms']
        sources.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'program':row['program'],'threads':row['threads'],
                        'boot_ids':sorted({b['boot_id'] for b in record['blocks']})})
    programs=json.loads((bundle/'plans/programs.json').read_text())
    single_units=sorted(p['begin'] for p in programs.values()
                        if p['device']=='cpu' and p['end']==p['begin']+1)
    return {'scope':'verified_existing_stage_cost_input_inventory; no policy selected or measured',
            'source_run':str(root),'bundle_id':manifest['bundle_id'],'pinned_executor_sha256':executor,
            'service_curves_ms':curves,'verified_profile_count':len(sources),'sources':sources,
            'available_single_unit_cpu_modules':single_units,
            'missing_single_unit_cpu_modules':sorted(set(range(21))-set(single_units)),
            'pipeit_full_layer_matrix_ready':False,
            'input_limit':'Measured fused-stage curves are not a complete per-layer T matrix; do not substitute sums or fill missing profiles.',
            'packrat_limit':'Each curve can serve as a single-subgraph batch1 L(t,1) input only; that does not optimize the dependent pipeline.',
            'board_jobs_started':0}

