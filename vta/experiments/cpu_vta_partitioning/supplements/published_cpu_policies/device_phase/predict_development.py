"""Retrospective development: actual host/device sequence, not whole-run ordering.

All input costs are isolated components. Four pipeline outcomes are already
known to the researcher; this replay cannot count as unseen validation.
"""
from pathlib import Path

from cpu_vta_partitioning.common import read,sha256
from phase_model.simulator import Replay,rational
from policy_branch_eval.common import isolated_inputs_only
from workload_model.common import freeze,OUTPUT


class DeviceReplay(Replay):
    def actions(self,stage):
        if stage['device']=='cpu':
            yield from super().actions(stage)
            return
        mask=stage['mask'];timing=stage['device_profile']
        yield ('cpu',rational(stage['set_cpu_ns']),mask)
        yield ('delay',rational(stage['set_nonrunning_ns']))
        for index,device in enumerate(timing['devices']):
            yield ('cpu',rational(timing['host_serial_CPU'][index]),mask)
            cpu=rational(device['CPU']);off=rational(device['wall'])-cpu
            if off<0:raise ValueError('device thread CPU exceeds wall')
            tasks=(('cpu',cpu,mask),('delay',off))
            yield from (tasks if self.host_order=='before' else tasks[::-1])
        yield ('cpu',rational(timing['host_serial_CPU'][-1]),mask)
        yield ('cpu',rational(stage['get_cpu_ns']),mask)
        yield ('delay',rational(stage['get_nonrunning_ns']))


def predict(output):
    output=Path(output).resolve()
    with isolated_inputs_only():
        device_path=output/'ISOLATED_DEVICE_PHASES.json';device=read(device_path)
        prior_path=OUTPUT/'phase_profile/temporal_development/PREDICTIONS.json';prior=read(prior_path)
        if not device['no_pipeline_FPS'] or prior['pipeline_performance_read']:
            raise ValueError('predictor inputs must be isolated')
        result={}
        for network,cases in prior['networks'].items():
            result[network]={}
            for case,value in cases.items():
                from copy import deepcopy
                stages=deepcopy(value['stages'])
                for stage,row in zip(stages,value['configuration']['stages']):
                    if row['device']=='vta':
                        profile=device['profiles'][network+'/'+row['program']]
                        if not profile['no_pipeline_performance']:raise ValueError('VTA cost is not isolated')
                        stage['device_profile']=profile['stages']
                        from cpu_policies.open_resources.costs import quantize_ns
                        for op in ('set','get'):
                            cpu=quantize_ns(profile['clean_cost'][op+'_core_ms'])
                            stage[op+'_cpu_ns']=cpu
                            stage[op+'_nonrunning_ns']=max(0,quantize_ns(profile['clean_cost'][op+'_ms'])-cpu)
                simulations={}
                for order in ('before','after'):
                    replay=DeviceReplay(stages,host_order=order,runnable_completion_wait=True).run()
                    simulations[order]={**replay,'score_ns':max(value['original_score']['score_ns'],replay['interval_ns'])}
                result[network][case]=dict(configuration=value['configuration'],simulations=simulations,
                    original_score_ns=value['original_score']['score_ns'])
        document=dict(networks=result,development_after_known_pipeline_measurement=True,
            pipeline_FPS_read_by_predictor=False,primary='before',
            input_hashes={str(device_path):sha256(device_path),str(prior_path):sha256(prior_path)},
            model_sources={str(p):sha256(p) for p in (Path(__file__),Path(__file__).parents[1]/'phase_model/simulator.py')},
            interpretation='Replace arbitrary whole-run host ordering by measured per-device-call serial work; '
                'polling CPU inside a driver call remains a before/after sensitivity, not recovered runnable timing',
            no_full_domain_or_unseen_claim=True)
        freeze(output/'DEVELOPMENT_PREDICTIONS.json',document)
    return {n:{c:{o:v['score_ns']/1e6 for o,v in r['simulations'].items()} for c,r in cases.items()}
            for n,cases in result.items()}


if __name__=='__main__':print(predict(OUTPUT/'device_phase_02'))
