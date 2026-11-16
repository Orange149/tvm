"""Diagnostic constraints, frozen before full pipeline observations."""
from copy import deepcopy
from ..open_resources.costs import Costs, score_configuration
from ..open_resources.domain import canonical_tuple
from ..core_model.domain import component, component_key
from .domain import configurations

class ExactCosts(Costs):
    def get(self,row,context_value='shared4'):
        key=component_key(component(row))
        if key not in self.records: raise ValueError('missing exact physical profile: '+key)
        result=deepcopy(self.records[key]['cost'])
        result['cpu_core_weights']=[1/row['threads']]*row['threads'] if row['device']=='cpu' else []
        return result

def predictions(records,concurrent,ddr,programs):
    costs=ExactCosts(records,programs,ddr)
    result={'pipeline_feedback_used':False,'costs_are_saturated_component_observations':True,
            'predictions':{},'rankings':{}}
    for name,cfg in configurations().items():
        baseline=score_configuration(cfg,costs,ddr)
        values={'original':baseline['score_ms']}
        for mode,method in [('cpu_pair','cpu_corun'),('cpu_pair_vta','cpu_vta_corun')]:
            actors=concurrent[name+'/'+mode]['actors']
            services=[actors[str(i)]['service_ms'] for i in (0,1)]
            values[method]=max(baseline['score_ms'],*services)
        result['predictions'][name]={'configuration':cfg,'estimates_ms':values,'baseline_ledger':baseline}
    for topology in ('explanation','validation'):
        result['rankings'][topology]={}
        for method in ('original','cpu_corun','cpu_vta_corun'):
            names=[n for n in result['predictions'] if n.startswith(topology+'/')]
            result['rankings'][topology][method]=sorted(names,key=lambda n:(result['predictions'][n]['estimates_ms'][method],canonical_tuple(result['predictions'][n]['configuration'])))
    return result
