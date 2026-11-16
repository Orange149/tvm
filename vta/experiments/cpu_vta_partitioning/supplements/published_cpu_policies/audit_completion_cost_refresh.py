"""Freeze the cost-refresh-only counterfactual before validation FPS exists."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,sha256
from finite_service.common import freeze
from cpu_policies.open_resources.costs import Costs,FIELDS
from cpu_policies.open_resources.domain import canonical_tuple
from staged_ablation.model import score
from policy_branch_eval.ranking import inputs
from policy_branch_eval.common import isolated_inputs_only
import prepare_completion_holdout as w
import run_device_completion_replay as model


class RefreshedCosts(Costs):
    def get(self,row,context_value='shared4'):
        value=super().get(row,context_value)
        if row['device']=='vta':
            if row['mask']!=[0,1,2,3]:raise ValueError('no matched refreshed VTA cost for this mask')
            item=self.refresh[self.network+'/'+row['program']]
            for key in FIELDS:value[key]=item['clean_cost'][key]
            value['assumptions'].append('new_isolated_VTA_cost_only_no_CPU_formula_change')
        return value


def audit():
    doc=read(w.OUTPUT/'INPUTS.json');profiles=read(model.PROFILE)['profiles'];result={}
    for network,items in doc['networks'].items():
        with isolated_inputs_only():
            programs,records,ddr,hashes,_,units=inputs(network)
            costs=RefreshedCosts(records,programs=programs,ddr=ddr,distribution='session_weights')
            costs.network=network;costs.refresh=profiles
            scores={cid:score(item['configuration'],costs,'fine_host')['score_ns'] for cid,item in items.items()}
        order=lambda cid:canonical_tuple(items[cid]['configuration'])
        old=sorted(items,key=lambda cid:(items[cid]['static_score_ns'],order(cid)))
        new=sorted(items,key=lambda cid:(scores[cid],order(cid)))
        result[network]=dict(old_ranking=old,refreshed_ranking=new,refreshed_scores_ns=scores,
            top1_unchanged=old[0]==new[0],component_cost_hashes=hashes,
            changed_only='five matched isolated VTA clean costs, original fine_host formula and CPU costs unchanged')
    freeze(w.OUTPUT/'COST_REFRESH_AUDIT.json',dict(networks=result,
        source_sha256=sha256(__file__),input_sha256=sha256(w.OUTPUT/'INPUTS.json'),
        completion_profile_sha256=sha256(model.PROFILE),no_pipeline_FPS_read=True,
        counterfactual_frozen_before_native=True))
    return {n:dict(top1_unchanged=v['top1_unchanged'],refreshed_first=v['refreshed_ranking'][0]) for n,v in result.items()}

if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False,indent=2))
