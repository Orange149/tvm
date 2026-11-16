"""Read-only replay of the already verified Tarnawski reference Top-20.

This adds a reference list to the new comparison without regenerating old
controllers, changing their files, or reading their pipeline measurements.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256
from cpu_policies.open_resources.costs import Costs
from policy_branch_eval.ranking import engine_from
from policy_branch_eval.common import isolated_inputs_only
from staged_ablation.model import score
from run_scope_rank import ROOT,OUT,NETWORKS,assert_plan,freeze

REFERENCES={
    'resnet18':ROOT/'artifacts/staged_ablation_01/tarnawski',
    'yolov3_tiny416':ROOT/'artifacts/yolo_validation_01/ranking/tarnawski'}


def run():
    plan=assert_plan();lists={};proofs={}
    with isolated_inputs_only():
        for network,root in REFERENCES.items():
            proof=read(root/'proof.json');spec=plan['networks'][network]
            expected_count=6121596 if network=='resnet18' else 3067532
            if (proof['selected_method']!='single_core' or proof['selected_count']!=expected_count or
                    proof['pipeline_FPS_used'] or not proof['verification']['passed']):
                raise ValueError('Tarnawski reference contract differs')
            for leaf in ('isolated_costs.json','ddr.json'):
                wanted=spec['cost_sha256' if leaf=='isolated_costs.json' else 'ddr_sha256']
                if proof['component_hashes'][leaf]!=wanted:
                    raise ValueError('reference component costs differ from the new comparison')
            module=engine_from(proof,'scope_tarnawski_'+network)
            programs=read(spec['programs_path'])
            if network=='yolov3_tiny416':
                programs={n:p for n,p in programs.items() if p['eligible_search']}
            records=read(spec['cost_path'])['costs'];ddr=read(spec['ddr_path'])
            costs=Costs(records,programs=programs,ddr=ddr,distribution='balanced')
            graphs=module._topologies(programs,spec['units'],3)
            contexts={c['id']:c for c in module.domain.contexts()}
            expanded=[];count=0;checkpoints={}
            for ctx in contexts.values():
                path=root/'search'/(ctx['id']+'.json');cp=read(path)
                payload=dict(programs=module._prepare(programs,costs,ctx,ddr),topologies=graphs,
                    context=ctx,k=20,score_scale=module.SCORE_SCALE,implementation=proof['implementation'])
                fields=('counts','gate_counts','rankings','topology_strata','resource_strata')
                if (cp['checkpoint_sha256']!=module._digest({k:v for k,v in cp.items() if k!='checkpoint_sha256'}) or
                        cp['input_sha256']!=module._digest(payload) or not cp['verification']['passed'] or
                        cp['verification']['source_sha256']!=proof['implementation']['oracle.cc'] or
                        cp['verification']['result_sha256']!=module._digest({k:cp[k] for k in fields})):
                    raise ValueError('reference full-domain oracle proof differs')
                count+=cp['counts']['single_core'];checkpoints[str(path)]=sha256(path)
                for raw in cp['rankings']['single_core']:
                    t=module.Terminal.from_json(raw)
                    expanded.append({**module._configuration(t,graphs,programs,contexts),
                        'score_ns':t.score,'score_ms':t.score/1e6})
            rows=sorted(expanded,key=module._row_key)[:20]
            for rank,row in enumerate(rows,1):
                row['static_rank']=rank
                if score(row,costs,'tarnawski')['score_ns']!=row['score_ns']:
                    raise ValueError('Tarnawski scalar formula differs from search')
            if (count!=expected_count or len(graphs)!=spec['topologies'] or
                    rows!=read(root/'rankings.json')['methods']['single_core']['rankings']):
                raise ValueError('reference complete Top20 differs')
            lists[network]=rows
            proofs[network]=dict(count=count,topologies=len(graphs),contexts=9,
                source_proof_sha256=sha256(root/'proof.json'),source_rankings_sha256=sha256(root/'rankings.json'),
                checkpoint_hashes=checkpoints,implementation=proof['implementation'],
                scalar_verified=True,cached_complete_enumeration_verified=True,
                new_board_sessions=0,pipeline_FPS_read=False)
    freeze(OUT/'TARNAWSKI_LISTS.json',lists)
    freeze(OUT/'TARNAWSKI_PROOFS.json',proofs)
    freeze(OUT/'TARNAWSKI_FREEZE.json',dict(files={n:sha256(OUT/n) for n in (
        'TARNAWSKI_LISTS.json','TARNAWSKI_PROOFS.json')},source_sha256=sha256(__file__),
        plan_sha256=sha256(OUT/'PLAN.json'),new_native_sessions=0,no_pipeline_FPS_read=True))
    print(json.dumps({n:dict(count=p['count'],first=lists[n][0]['id']) for n,p in proofs.items()},ensure_ascii=False,indent=2))
    return lists


if __name__=='__main__':
    run()
