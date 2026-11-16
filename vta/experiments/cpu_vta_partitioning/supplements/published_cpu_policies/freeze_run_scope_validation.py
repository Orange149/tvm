"""Freeze the common validation pool, without preparing or running board jobs."""
from pathlib import Path
import json
import random
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,sha256
from cpu_policies.open_resources.domain import configuration
from policy_branch_eval.common import isolated_inputs_only
from run_scope_rank import ROOT,OUT as RANK,NETWORKS,freeze

OUT=ROOT/'artifacts/run_scope_common_validation_01'
METHODS=('tarnawski','original_complete','run_scope_complete')


def run():
    rank_seal=read(RANK/'SELECTION_FREEZE.json')
    reference_seal=read(RANK/'TARNAWSKI_FREEZE.json')
    sources={str(RANK/n):sha256(RANK/n) for n in ('PLAN.json','SELECTION_FREEZE.json','TARNAWSKI_FREEZE.json')}
    for seal in (rank_seal,reference_seal):
        if seal['plan_sha256']!=sha256(RANK/'PLAN.json'):
            raise ValueError('comparison plan changed')
        for leaf,wanted in seal['files'].items():
            if sha256(RANK/leaf)!=wanted:
                raise ValueError('frozen static list or proof changed')
            sources[str(RANK/leaf)]=wanted
    lists={};pool={};counts={}
    with isolated_inputs_only():
        rankings=read(RANK/'LISTS.json');reference=read(RANK/'TARNAWSKI_LISTS.json')
        for network in NETWORKS:
            selected=dict(tarnawski=reference[network],original_complete=rankings[network+'/original'],
                          run_scope_complete=rankings[network+'/run_scope'])
            candidates={};ranks={}
            for method,rows in selected.items():
                if len(rows)!=20 or len({r['id'] for r in rows})!=20:
                    raise ValueError('each method needs an ordered unique Top20')
                for row in rows:
                    cfg={**configuration(row),'id':row['id']}
                    if row['id'] in candidates and candidates[row['id']]!=cfg:
                        raise ValueError('execution identity collision')
                    candidates[row['id']]=cfg
                    ranks.setdefault(row['id'],{})[method]=row['static_rank']
            if len(candidates)>60:
                raise ValueError('common pool exceeds the three-method budget')
            lists[network]=selected;pool[network]=dict(sorted(candidates.items()))
            counts[network]=dict(configurations=len(candidates),native_sessions=2*len(candidates),
                DNN_inferences=29*len(candidates),rank_membership=ranks)
    order=[dict(network=n,configuration=i) for n in sorted(pool) for i in sorted(pool[n])]
    random.Random(20261111).shuffle(order)
    total=sum(r['configurations'] for r in counts.values())
    res_same=([r['id'] for r in lists['resnet18']['original_complete']]==
              [r['id'] for r in lists['resnet18']['run_scope_complete']])
    protocol=dict(experiment='run_scope_common_validation_01',seed=20261111,input_index=0,
        queue_depth=2,ordinary_copy=True,qualification=dict(warmup=0,scored=1,tail=0),
        performance=dict(warmup=8,scored=16,tail=4),
        short_window=dict(warmup=8,scored=16,tail=4),
        environment=dict(cores=4,frequency_khz=1066666,poll_sleep_ns=1000,post_start_sleep_ns=1000,
                         spin_count=0,exclude_worker0=1,bind_threads=1),
        qualification_output='raw_all_stages',performance_output='raw',
        expected_original_executor_sha256='25833e7f8b47fc8824ff9460afc0b968f24934dcee41c3b1f28619a99b6b6fe6',
        scoring_version_not_in_execution_identity=True,one_boot=True,reboot=False,bitstream_reload=False,
        kernel_or_system_parameter_changes=False,anchors=False,automatic_extra_measurement=False,
        detailed_thread_or_queue_observers=False,new_component_profiles=0,
        explicit_execute_board_required=True,same_process_setup_for_every_method=True,
        additional_native_campaign_requires_user_approval=True,
        current_state='SELECTION_FROZEN_ASSETS_NOT_PREPARED')
    freeze(OUT/'LISTS.json',lists);freeze(OUT/'POOL.json',pool);freeze(OUT/'ORDER.json',order)
    freeze(OUT/'PROTOCOL.json',protocol)
    plan=dict(sources=sources,source_sha256=sha256(__file__),networks=counts,
        configurations=total,native_sessions=2*total,DNN_inferences=29*total,
        new_native_sessions=0,pipeline_FPS_read=False,methods=list(METHODS),
        ResNet_input_repair_has_identical_ordered_lists=res_same,
        ResNet_measurement_purpose=('common strongest-method versus Tarnawski comparison and bridge; not a new run-scope gain'
            if res_same else 'common strongest-method control, isolated-input repair, and Tarnawski comparison'),
        YOLO_measurement_purpose='common strongest-method control, isolated-input repair, and Tarnawski comparison',
        assets_prepared=False,board_execution_approved=False)
    freeze(OUT/'PLAN.json',plan)
    freeze(OUT/'FREEZE.json',dict(files={leaf:sha256(OUT/leaf) for leaf in (
        'PLAN.json','LISTS.json','POOL.json','ORDER.json','PROTOCOL.json')},
        no_pipeline_FPS_input=True,new_native_sessions=0))
    print(json.dumps(dict(counts={n:{k:r[k] for k in ('configurations','native_sessions','DNN_inferences')}
        for n,r in counts.items()},total_configurations=total,total_native_sessions=2*total,
        total_DNN_inferences=29*total,assets_prepared=False,board_started=False),ensure_ascii=False,indent=2))


if __name__=='__main__':
    run()
