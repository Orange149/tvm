"""Pure scoring: no performance-reference paths or throughput inputs."""
from copy import deepcopy
from cpu_policies.open_resources.domain import canonical_tuple
from .domain import cases

def rank(model_inputs,measurements):
    if set(model_inputs)!= {'costs','original_predictions','reference_rankings'}:raise ValueError('unexpected model input')
    if set(measurements)!=set(cases()):raise ValueError('eight CPU measurements required')
    result={'pipeline_feedback_used':False,'predictions':{},'rankings':{}}
    for name,cfg in cases().items():
        source=model_inputs['original_predictions'][name];measured=measurements[name]
        original=source['original_score_ns'];cpu=measured['cpu_ii_ns']
        if type(original)!=int or original<=0 or type(cpu)!=int or cpu<=0 or measured['completion_gaps']!=7:
            raise ValueError('invalid source/group score')
        score=max(original,cpu)
        result['predictions'][name]={'configuration':cfg['configuration'],'original_ns':original,
            'saturated_ms':source['saturated_score_ms'],'frame_stream_ns':score,'cpu_group_ns':cpu,
            'frame_stream_ms':score/1e6,'original_ms':original/1e6}
    for topology in ('explanation','validation'):
        names=[name for name in result['predictions'] if name.startswith(topology+'/')]
        refs=model_inputs['reference_rankings'][topology]
        if any(set(refs[m])!=set(names) for m in ('original','saturated')):raise ValueError('source ranking domain differs')
        result['rankings'][topology]={**deepcopy(refs),'frame_stream':sorted(names,key=lambda n:(
            result['predictions'][n]['frame_stream_ns'],canonical_tuple(cases()[n]['configuration'])))}
    return result
