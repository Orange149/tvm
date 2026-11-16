"""Same isolated work and clocks, removing only VTA call-level interleaving."""
from copy import deepcopy
from pathlib import Path

from cpu_vta_partitioning.common import read,sha256
from cpu_policies.open_resources.costs import quantize_ns
from phase_model.simulator import Replay,rational
from policy_branch_eval.common import isolated_inputs_only
from workload_model.common import OUTPUT,freeze


def run(output):
    output=Path(output).resolve()
    with isolated_inputs_only():
        source=OUTPUT/'phase_profile/temporal_development/PREDICTIONS.json'
        prior=read(source);profiles=read(output/'ISOLATED_DEVICE_PHASES.json')
        result={}
        for network,cases in prior['networks'].items():
            result[network]={}
            for case,value in cases.items():
                stages=deepcopy(value['stages'])
                for stage,row in zip(stages,value['configuration']['stages']):
                    if row['device']!='vta':continue
                    profile=profiles['profiles'][network+'/'+row['program']]
                    parts=profile['stages']
                    host=sum(rational(s) for s in parts['host_serial_CPU'])
                    driver_CPU=sum(rational(d['CPU']) for d in parts['devices'])
                    driver_wall=sum(rational(d['wall']) for d in parts['devices'])
                    stage['run_cpu_ns']=host+driver_CPU
                    stage['run_nonrunning_ns']=driver_wall-driver_CPU
                    for op in ('set','get'):
                        cpu=quantize_ns(profile['clean_cost'][op+'_core_ms'])
                        stage[op+'_cpu_ns']=cpu
                        stage[op+'_nonrunning_ns']=max(0,quantize_ns(profile['clean_cost'][op+'_ms'])-cpu)
                result[network][case]={}
                for order in ('before','after'):
                    replay=Replay(stages,host_order=order,runnable_completion_wait=True).run()
                    result[network][case][order]={**replay,'score_ns':max(value['original_score']['score_ns'],replay['interval_ns'])}
        freeze(output/'DEVELOPMENT_AGGREGATE_CONTROL.json',dict(networks=result,
            exactly_same_isolated_host_CPU_and_driver_wall_as_interleaved_variant=True,
            only_VTA_run_order_aggregation_changed=True,pipeline_FPS_read=False,
            development_after_known_measurements=True,
            sources={str(source):sha256(source),str(output/'ISOLATED_DEVICE_PHASES.json'):sha256(output/'ISOLATED_DEVICE_PHASES.json'),
                     str(Path(__file__)):sha256(Path(__file__))}))
    return {n:{c:{o:v['score_ns']/1e6 for o,v in sims.items()} for c,sims in cases.items()} for n,cases in result.items()}


if __name__=='__main__':print(run(OUTPUT/'device_phase_02'))
