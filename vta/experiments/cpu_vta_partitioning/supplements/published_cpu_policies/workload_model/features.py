"""Features computed from isolated costs and compiled graphs; no pipeline FPS."""
import math
import re

from cpu_vta_partitioning.common import digest
from cpu_policies.open_resources import domain
from cpu_policies.open_resources.costs import Costs


def tensor_bytes(spec):
    shape=spec['shape'];dtype=spec['dtype']
    if not isinstance(shape,list) or any(type(v) is not int or v<0 for v in shape):
        raise ValueError('invalid tensor shape')
    match=re.fullmatch(r'(?:u?int|float|bfloat)(\d+)(?:x(\d+))?',dtype)
    if not match:raise ValueError('unmodeled dtype byte width: '+str(dtype))
    bits=int(match[1])*int(match[2] or 1)
    return (math.prod(shape)*bits+7)//8


def graph_features(graph):
    attrs=graph['attrs'];shapes=attrs['shape'][1];dtypes=attrs['dltype'][1]
    storage=attrs['storage_id'][1];ptr=graph['node_row_ptr']
    if not len(shapes)==len(dtypes)==len(storage)==ptr[-1]:
        raise ValueError('graph entry metadata lengths differ')
    sizes=[tensor_bytes(dict(shape=s,dtype=d)) for s,d in zip(shapes,dtypes)]
    capacities={}
    for sid,size in zip(storage,sizes):
        if sid>=0:capacities[sid]=max(capacities.get(sid,0),size)
    functions=[];convs=[];unknown_convs=[]
    for i,node in enumerate(graph['nodes']):
        if node['op']!='tvm_op':continue
        function=node['attrs']['func_name'];functions.append(function)
        if 'conv2d' not in function:continue
        inputs=node['inputs'];out=shapes[ptr[i]]
        a=node['attrs']
        if len(inputs)>=2:
            data=shapes[ptr[inputs[0][0]]+inputs[0][1]]
            weight=shapes[ptr[inputs[1][0]]+inputs[1][1]]
        else:data=weight=[]
        if (a.get('data_layout')=='NCHW' and a.get('kernel_layout')=='OIHW'
            and len(data)==len(weight)==len(out)==4 and data[1]==weight[1] and out[1]==weight[0]):
            # Only the recognized dense convolution, not fused pointwise operations.
            macs=math.prod(out)*math.prod(weight[1:])
            convs.append(dict(function=function,input_shape=data,weight_shape=weight,
                              output_shape=out,MACs=macs))
        else:unknown_convs.append(function)
    return dict(graph_function_count=len(functions),graph_functions=functions,
        recognized_dense_convolution_MACs=sum(v['MACs'] for v in convs),
        recognized_dense_convolutions=convs,unmodeled_convolution_functions=unknown_convs,
        graph_entry_logical_bytes=sum(sizes),graph_planned_storage_proxy_bytes=sum(capacities.values()),
        workspace_scope='graph storage_id capacity proxy; excludes native workspaces/cache effects; not DDR traffic',
        graph_functions_are_not_parallel_launches=True)


def isolated_features(network,programs,records,ddr,graphs):
    costs=Costs(records,programs=programs,ddr=ddr,distribution='session_weights')
    result=[]
    for name,program in sorted(programs.items()):
        if program['device']!='cpu':continue
        base=costs.get(domain.stage('cpu',program['begin'],program['end'],1,[0]))
        structural=graph_features(graphs[name])
        for n in range(1,5):
            stage=domain.stage('cpu',program['begin'],program['end'],n,list(range(n)))
            value=costs.get(stage);weights=value['cpu_core_weights']
            identity=dict(network=network,canonical_sha256=program['canonical_sha256'],
                program=name,program_files=program['files'],threads=n,mask=stage['mask'])
            result.append(dict(identity='profile_'+digest(identity)[:24],network=network,
                canonical_sha256=program['canonical_sha256'],program=name,begin=program['begin'],end=program['end'],
                units=program['end']-program['begin'],threads=n,service_ms=value['service_ms'],
                run_ms=value['run_ms'],run_core_ms=value['run_core_ms'],boundary_core_ms=value['boundary_core_ms'],
                set_ms=value.get('set_ms'),get_ms=value.get('get_ms'),
                isolated_speedup=base['service_ms']/value['service_ms'],
                run_core_growth=value['run_core_ms']/base['run_core_ms'] if base['run_core_ms'] else None,
                average_running_cores=value['run_core_ms']/value['run_ms'] if value['run_ms'] else None,
                caller_share=weights[0],max_worker_share=max(weights),
                thread_share_CV=math.sqrt(sum((n*w-1)**2 for w in weights)/n),
                thread_weights=weights,assumptions=value['assumptions'],source_key=value['source_key'],
                input_tensors=len(program['input_types']),output_tensors=len(program['output_types']),
                input_frontier_bytes=sum(map(tensor_bytes,program['input_types'])),
                output_frontier_bytes=sum(map(tensor_bytes,program['output_types'])),
                params_file_bytes=program['files']['params.bin']['bytes'],**structural))
    return result


def placement_features(cfg,costs):
    """Independent of network names/FPS; characterize pressure, not fit slowdown."""
    cpu=[s for s in cfg['stages'] if s['device']=='cpu']; values=[costs.get(s,cfg) for s in cpu]
    pairs=[]
    for i,a in enumerate(cpu):
        for j,b in enumerate(cpu[i+1:],i+1):
            overlap=sorted(set(a['mask'])&set(b['mask']))
            relation='same' if a['mask']==b['mask'] else 'partial' if overlap else 'disjoint'
            pairs.append(dict(first=i,second=j,relation=relation,intersection=overlap,
                              service_product_ms2=values[i]['service_ms']*values[j]['service_ms']))
    return dict(CPU_stages=len(cpu),VTA_islands=len(cfg['stages'])-len(cpu),
        configured_CPU_threads=sum(s['threads'] for s in cpu),
        CPU_service_sum_ms=sum(v['service_ms'] for v in values),
        CPU_compute_core_ms=sum(v['run_core_ms'] for v in values),
        max_stage_service_fraction=max(v['service_ms'] for v in values)/sum(v['service_ms'] for v in values),
        core_thread_memberships=[sum(c in s['mask'] for s in cpu) for c in range(4)],
        pairs=pairs,gate=cfg['execution_mode'],context=cfg['context_id'],
        configured_threads_not_simultaneously_runnable_threads=True)
