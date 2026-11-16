"""Topological reference: independent integer convolutions, shared CPU code.

This is deliberately not a second GraphExecutor.run(). CPU functions are an
explicit trusted common path. All VTA convolutions are replaced by workload
geometry/NumPy references. The caller separately binds unchanged CPU code,
actual parameters, and the native execution/caching evidence.
"""
import hashlib
import numpy as np


def require(value,message):
    if not value:raise ValueError(message)


def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def evaluate(graph,parameters,data,convolutions,cpu_call):
    """cpu_call(symbol, inputs, output_descriptors, flatten) -> output arrays.

    `convolutions` maps exact compiled symbols to ReferenceSpec instances.
    No function name pattern, node number, Tile or instruction trace is used
    to calculate convolution values. Keeping logical entries separate avoids
    reproducing a potentially erroneous runtime storage reuse decision.
    """
    shapes=graph['attrs']['shape'][1];dtypes=graph['attrs']['dltype'][1]
    rowptr=graph['node_row_ptr'];values={};conv_rows=[];cpu_rows=[];used=set()
    require(len(rowptr)==len(graph['nodes'])+1 and rowptr[-1]==len(shapes)==len(dtypes),
            'GRAPH_ENTRY_ATTRIBUTES_CHANGED')
    expected_nulls={n['name'] for n in graph['nodes'] if n['op']=='null'}
    require(set(parameters).isdisjoint(data) and set(parameters)|set(data)==expected_nulls,
            'GRAPH_NULL_INPUT_COVERAGE_CHANGED')

    def descriptor(entry):return {'shape':list(shapes[entry]),'dtype':dtypes[entry]}

    def check(value,entry):
        require(isinstance(value,np.ndarray) and list(value.shape)==list(shapes[entry])
                and str(value.dtype)==dtypes[entry],'GRAPH_REFERENCE_ENTRY_ABI_CHANGED')

    for nid,node in enumerate(graph['nodes']):
        first,last=rowptr[nid:nid+2]
        if node['op']=='null':
            require(last-first==1,'MULTI_OUTPUT_NULL_UNSUPPORTED')
            value=parameters[node['name']] if node['name'] in parameters else data[node['name']]
            check(value,first);values[first]=value.copy();continue
        require(node['op']=='tvm_op','UNSUPPORTED_GRAPH_NODE')
        attrs=node['attrs'];symbol=attrs['func_name']
        require(last-first==int(attrs['num_outputs']) and len(node['inputs'])==int(attrs['num_inputs']),
                'GRAPH_CALL_ARITY_CHANGED')
        entries=[]
        for parent,output,version in node['inputs']:
            require(parent<nid and version==0 and 0<=output<rowptr[parent+1]-rowptr[parent],
                    'GRAPH_INPUT_ORDER_OR_VERSION_UNSUPPORTED')
            entries.append(rowptr[parent]+output)
        inputs=[values[e] for e in entries]
        snapshots=[array_sha(x) for x in inputs]
        if symbol in convolutions:
            spec=convolutions[symbol]
            require(last-first==1 and int(attrs['flatten_data'])==0 and len(inputs)==len(spec.abi)-1,
                    'VTA_GRAPH_ABI_UNSUPPORTED')
            outputs=[spec.reference(inputs)]
            conv_rows.append({'node':nid,'symbol':symbol,'input_entries':entries,
                              'output_entry':first,'input_sha256':snapshots,'output_sha256':array_sha(outputs[0])})
            used.add(symbol)
        elif symbol=='__nop':
            require(len(inputs)==last-first==1 and inputs[0].size==int(np.prod(shapes[first])),
                    'GRAPH_NOP_UNSUPPORTED')
            outputs=[inputs[0].reshape(shapes[first])]
        elif symbol=='__copy':
            require(len(inputs)==last-first==1,'GRAPH_COPY_UNSUPPORTED')
            outputs=[inputs[0].copy().reshape(shapes[first])]
        else:
            outputs=cpu_call(symbol,inputs,[descriptor(e) for e in range(first,last)],bool(int(attrs['flatten_data'])))
            require(isinstance(outputs,(list,tuple)) and len(outputs)==last-first,'CPU_REFERENCE_ARITY_CHANGED')
            cpu_rows.append({'node':nid,'symbol':symbol})
        require([array_sha(x) for x in inputs]==snapshots,'REFERENCE_READONLY_INPUT_CHANGED')
        for entry,value in zip(range(first,last),outputs):
            check(value,entry);values[entry]=value.copy()
    require(used==set(convolutions),'REFERENCE_VTA_FUNCTION_COVERAGE_CHANGED')
    heads=[]
    for nid,output,version in graph['heads']:
        require(version==0 and 0<=output<rowptr[nid+1]-rowptr[nid],'GRAPH_HEAD_UNSUPPORTED')
        heads.append(values[rowptr[nid]+output].copy())
    return heads,{'convolution_calls':conv_rows,'CPU_calls':cpu_rows,
                  'independent_component':'NumPy integer convolution and actual fused postprocess',
                  'trusted_component':'explicitly bound unchanged compiled CPU functions',
                  'GraphExecutor_target_calls':0}
