"""Freeze new resource combinations before their components and pipeline FPS."""
from copy import deepcopy
from pathlib import Path

from cpu_vta_partitioning.common import read,sha256
from cpu_policies.open_resources.domain import candidate,stage
from phase_profile.prepare import configurations
from policy_branch_eval.common import isolated_inputs_only
from workload_model.common import freeze


def project_known_ids(output):
    """Independent identity projection; the selector never receives FPS fields."""
    output=Path(output);source=output/'historical_candidates.json';values=read(source)
    ids={n:sorted({r['candidate'] for r in values if r['network']==n})
         for n in ('resnet18','yolov3_tiny416')}
    freeze(output/'KNOWN_CONFIGURATION_IDS.json',dict(ids=ids,source_sha256=sha256(source),
        scope='184 audited historical configurations used in this Goal development; not an inventory of every historical run',
        projected_fields=['network','candidate'],performance_values_not_exported=True))


def plan(output):
    output=Path(output)
    with isolated_inputs_only():
        known=read(output/'KNOWN_CONFIGURATION_IDS.json')['ids']
        masks=dict(partial=[[0,1,2],[2,3],[1]],overlap=[[0,1,2,3],[0,1],[2]],
                   disjoint=[[0,1],[2,3],[0]])
        networks={}
        for network in ('resnet18','yolov3_tiny416'):
            old=configurations(network)['pipelines'];ranges=old['overlap']['stages'];cases={}
            for name,groups in masks.items():
                value=None
                for rotation in range(4):
                    group=iter(groups);rows=[]
                    for row in ranges:
                        if row['device']=='cpu':
                            mask=sorted((c+rotation)%4 for c in next(group))
                            rows.append(stage('cpu',row['begin'],row['end'],len(mask),mask))
                        else:rows.append(deepcopy(row))
                    trial=candidate(rows,'shared4','off')
                    if trial['id'] not in known[network] and trial['id'] not in {c['id'] for c in old.values()}:
                        value=trial;break
                if value is None:raise ValueError('all four cyclic mappings were used in development')
                cases[name]=value
            networks[network]=cases
        document=dict(networks=networks,
            primary='measured CPU fork/join plus measured VTA per-device-call host work, runnable completion waiting, driver CPU before offCPU interval',
            host_sensitivity='also report driver CPU after offCPU; do not choose order by FPS',
            comparisons=['frozen current capacity model and old isolated costs',
                         'capacity model with new local component costs only',
                         'same new components plus host/device temporal model'],
            decision='both networks selected FPS no worse than refreshed capacity, at least one improves; '
                     'report original-cost baseline separately; inconsistent host-order selection remains unresolved',
            new_component_signatures=4,component_native_sessions=12,component_CPU_stage_calls=84,
            new_pipeline_configurations=6,pipeline_native_sessions=12,DNN_inferences=174,
            component_window=dict(warmup=2,scored=8),pipeline_window=dict(warmup=8,scored=16,tail=4),
            qualification_frames=1,model_sources_frozen={str(p):sha256(p) for p in
                (Path(__file__),Path(__file__).with_name('simulator.py'),
                 Path(__file__).parents[1]/'device_phase/predict_development.py')},
            no_reboot=True,one_round=True,no_anchors=True,no_automatic_remeasurement=True,
            no_DP_change=True,no_full_domain_profile=True,
            resource_mapping_selection='for each predeclared family choose first cyclic core rotation 0..3 absent from known identity pool; no performance fields',
            unseen_scope='resource combinations not in the audited Goal development pool; same known cuts and modules',
            pipeline_FPS_read=False)
        freeze(output/'phase_validation/PLAN.json',document)
    return dict(new_configurations=6,new_component_signatures=4,pipeline_inferences=174)
