"""Reuse the published fixed-group contract with the existing outer TVM/VTA DP.

The outer interval DP remains this project's algorithm. Neither the dependent
CPU stages nor the FPGA host are disguised as Packrat replica batches. No
post-hoc core-number selection is introduced here. The existing host/control
placement is preserved and explicitly remains a project-level constraint.
"""
import hashlib
import json
from pathlib import Path
import sys

from .runtime_contract import build_runtime_contract


def _source(run_dir):
    root=Path(run_dir).resolve();state=json.loads((root/'state.json').read_text());bundle=Path(state['bundle'])
    sys.path[:0]=[str(bundle/'sources/cpu_strategy'),str(bundle/'sources/base_controller/src')]
    return root,state,bundle


def adapt_existing_candidate(run_dir,candidate_id):
    root,state,bundle=_source(run_dir)
    from cpu_strategy.policy.domain import candidate
    from cpu_strategy.policy.transport import make_task,native_command
    selection=json.loads((root/'final_selection.json').read_text());old=selection['pool'][candidate_id]
    groups=[];indices=[]
    for row in old['stages']:
        if row['device']!='cpu':continue
        if row['threads']!=len(row['mask']):raise ValueError('fixed groups require one compute slot per allocated core')
        mask=list(row['mask'])
        if mask not in groups:groups.append(mask)
        indices.append(groups.index(mask))
    contract=build_runtime_contract(groups,indices,host_mask=old['host_mask'],control_mask=old['control_mask'],
                                    allow_stage_group_reuse=True,allow_host_compute_overlap=True)
    new=candidate(old['stages'],'pinned','resource_group',old['host_mask'],old['control_mask'])
    doc=make_task('qualify_pipeline',new,bundle,indices=[0,16,17,18],
                  campaign_phase='offline_published_runtime_contract_adapter',source_candidate_id=candidate_id)
    argv,_=native_command(doc,bundle,'/verified_bundle','/not_executed_job')
    if '--cpu-worker-pinning' not in argv or argv[argv.index('--cpu-stage-gate')+1]!='resource_group':
        raise ValueError('native command does not implement the fixed-group contract')
    for ordinal,full_index in enumerate(i for i,s in enumerate(new['stages']) if s['device']=='cpu'):
        m=contract['stages'][ordinal];row=new['stages'][full_index]
        if m['mask']!=row['mask'] or m['worker_core_map']!=row['worker_core_map']:
            raise ValueError('CPU ordinal to full-pipeline mapping differs')
    return {'scope':'offline source-contract adapter; no ranking or board result',
            'source_candidate_id':candidate_id,'candidate':new,'runtime_contract':contract,
            'native_argv_preview':argv,'task':doc,'board_jobs_started':0,
            'host_placement_preserved':True,'new_policy_performance_claim':False,
            'method_attribution':'Pipe-it fixed-worker/disjoint-group execution semantics adapted to caller-inclusive TVM; outer partition/resource DP is retained',
            'before_board_execution':'require exact matched profiles, frozen campaign, native qualification and independent comparison'}


def profile_requirements(run_dir):
    root,state,bundle=_source(run_dir)
    from cpu_strategy.policy.domain import complete_components
    from cpu_vta_partitioning.common import verify_inventory
    programs=json.loads((bundle/'plans/programs.json').read_text())
    selection=json.loads((root/'policy_selection.json').read_text())
    mirrors=[x for x in selection['mirror_check']['comparisons'] if x['mode']=='pinned']
    if not mirrors or any(x['relative_difference']>.05 for x in mirrors):
        raise ValueError('pinned representative-core assumption has not passed existing mirror checks')
    required=complete_components(programs,'pinned',False,True)
    matched=[];missing=[]
    for key,configuration in required.items():
        path=root/'components'/(key+'.json')
        if not path.exists():missing.append({'component_key':key,'configuration':configuration});continue
        rel=str(path.relative_to(root))
        if rel not in state['frozen_files']:raise ValueError('existing component not frozen')
        verify_inventory(root,{rel:state['frozen_files'][rel]})
        record=json.loads(path.read_text())
        if record['configuration']!=configuration:raise ValueError('component identity collision')
        matched.append({'component_key':key,'configuration':configuration,'source':str(path),
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    return {'scope':'preparation inventory for fixed-group, pinned CPU policy with original shared-four-core host/control',
            'board_jobs_started':0,'costs_or_rankings_generated':False,
            'policy':'fixed compute groups; caller-inclusive one slot per core; same-group serialization',
            'host_mask':[0,1,2,3],'control_mask':[0,1,2,3],'cpu_binding':'pinned',
            'core_permutation_assumption':'same-type isolated core costs, explicitly inferred for other masks; not individually measured',
            'pinned_mirror_checks':mirrors,'profile_protocol':{'warmup':5,'scored':32,'tail':0,'input_indices':[0]},
            'required_count':len(required),'matched_count':len(matched),'missing_count':len(missing),
            'required_cpu_count':sum(c['stages'][0]['device']=='cpu' for c in required.values()),
            'required_vta_count':sum(c['stages'][0]['device']=='vta' for c in required.values()),
            'matched':matched,'missing':missing,
            'reuse_status':'identity-level match here; replay raw receipts before importing into a new campaign',
            'no_universal_5pct_adoption_gate':'The source-defined policy is a candidate to evaluate, not a newly discovered rule that must first dominate every cut.',
            'outer_dp_status':'source contract prepared; no new matched-cost ranking or board comparison performed'}
