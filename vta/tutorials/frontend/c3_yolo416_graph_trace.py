"""Compose closed fused traces using actual frozen graph arguments.

Namespace only: never change a uop, loop, DMA descriptor, dependency or Sync.
Shared function symbols keep shared kernel identities; graph entries retain
their own tensor versions. The native certifier must still verify the entire
completed 12-submission call and preserve host cache definitions across it.
"""
import math
import re


def require(value,message):
    if not value:raise ValueError(message)


def compose(graph,functions,expected_calls=12):
    rowptr=graph['node_row_ptr'];shapes=graph['attrs']['shape'][1];dtypes=graph['attrs']['dltype'][1]
    storage=graph['attrs']['storage_id'][1]
    events=[];calls=[];scopes={};used=set()
    for nid,node in enumerate(graph['nodes']):
        symbol=node.get('attrs',{}).get('func_name')
        if symbol not in functions:continue
        source=functions[symbol];abi=source['abi'];trace=source['events']
        require(len(abi)==4 and int(node['attrs']['num_inputs'])==3 and int(node['attrs']['num_outputs'])==1
                and int(node['attrs']['flatten_data'])==0,'GRAPH_FUSED_ABI_UNSUPPORTED')
        require(rowptr[nid+1]-rowptr[nid]==1,'GRAPH_FUSED_OUTPUT_ARITY')
        entries=[]
        for parent,output,version in node['inputs']:
            require(parent<nid and version==0 and 0<=output<rowptr[parent+1]-rowptr[parent],
                    'GRAPH_FUSED_INPUT_ORDER_OR_VERSION')
            entries.append(rowptr[parent]+output)
        require(len(entries)==3,'GRAPH_FUSED_INPUT_ARITY')
        entries.append(rowptr[nid])
        for descriptor,entry in zip(abi,entries):
            require(descriptor['shape']==shapes[entry] and descriptor['dtype']==dtypes[entry],
                    'GRAPH_FUSED_TENSOR_ABI_CHANGED')
        require(storage[entries[-1]] not in [storage[e] for e in entries[:-1]],
                'GRAPH_CONVOLUTION_INPUT_OUTPUT_STORAGE_ALIAS')
        require(trace and trace[-1]['kind']=='sync' and sum(e['kind']=='sync' for e in trace)==1,
                'GRAPH_REQUIRES_ONE_COMPLETED_FUSED_SYNC')
        start=len(events)
        for event in trace:
            e=dict(event)
            if e['kind']=='kernel':
                old=e['scope'];key=(symbol,old)
                scopes.setdefault(key,len(scopes))
                require(e['cache_identity']=={'scope':old,'signature':e['signature']},
                        'GRAPH_SOURCE_CACHE_IDENTITY_UNSUPPORTED')
                e['scope']=scopes[key]
                e['cache_identity']={'scope':scopes[key],'signature':e['signature']}
            elif e['kind'] in ('load','store'):
                require(type(e['tensor']) is int and 0<=e['tensor']<4,'GRAPH_SOURCE_TENSOR_ID')
                e['tensor']=entries[e['tensor']]
            elif e['kind'] not in ('push','pop','sync'):
                raise ValueError('GRAPH_UNKNOWN_FUSED_EFFECT')
            events.append(e)
        calls.append({'node':nid,'symbol':symbol,'tensor_entries':entries,
                      'event_begin':start,'event_end':len(events)})
        used.add(symbol)
    require(len(calls)==expected_calls and used==set(functions),'GRAPH_NATIVE_CALL_COVERAGE_CHANGED')
    return events,{'calls':calls,'scope_bindings':[{'symbol':s,'source_scope':old,'graph_scope':new}
                   for (s,old),new in scopes.items()],
                   'status':'GRAPH_NAMESPACE_COMPOSITION_NATIVE_CERTIFICATION_PENDING',
                   'completion_assumption':'each explicit Sync must be bound to actual successful driver completion'}


def storage_bindings(graph,mapping,native,observed=None):
    """Bind every decoded tensor base to graph storage reuse, in bytes.

    `observed` optionally supplies diagnostic snapshot handles. These are
    DataBuffer identities, never misrepresented as device physical addresses.
    Native bases use memory-type element units and are converted explicitly.
    """
    bytes_per_element={1:256,2:16,3:64,4:16}
    storage=graph['attrs']['storage_id'][1]
    allowed={e for call in mapping['calls'] for e in call['tensor_entries']}
    physical={};rows=[]
    for row in native['tensor_domain_dram_bases']:
        entry,domain=row['tensor'],row['domain']
        require(entry in allowed and domain in bytes_per_element,'GRAPH_NATIVE_UNKNOWN_TENSOR_OR_DOMAIN')
        sid=storage[entry];base=row['dram_base']*bytes_per_element[domain]
        require(sid not in physical or physical[sid]==base,'GRAPH_STORAGE_PHYSICAL_BASE_CHANGED')
        physical[sid]=base;rows.append({'entry':entry,'storage_id':sid,'domain':domain,'physical_base_bytes':base})
    sizes={}
    for sid,shape,dtype in zip(storage,graph['attrs']['shape'][1],graph['attrs']['dltype'][1]):
        match=re.fullmatch(r'(?:u?int|float)(8|16|32|64)',dtype)
        require(match is not None,'GRAPH_STORAGE_DTYPE_UNSUPPORTED')
        sizes[sid]=max(sizes.get(sid,0),math.prod(shape)*int(match[1])//8)
    intervals=sorted((base,base+sizes[sid],sid) for sid,base in physical.items())
    require(all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])),
            'GRAPH_DISTINCT_STORAGE_PHYSICAL_RANGES_OVERLAP')
    handles={};reverse={}
    if observed is not None:
        require(len(observed)==len(mapping['calls']),'GRAPH_OBSERVER_CALL_CENSUS')
        for actual,expected in zip(observed,mapping['calls']):
            require(int(str(actual['node']))==expected['node'] and str(actual['symbol'])==expected['symbol'],
                    'GRAPH_OBSERVER_CALL_ORDER')
            require(len(actual['data_handles'])==len(actual['byte_offsets'])==4,'GRAPH_OBSERVER_PARAMETER_COUNT')
            for entry,handle,offset in zip(expected['tensor_entries'],actual['data_handles'],actual['byte_offsets']):
                sid=storage[entry];handle=int(str(handle))
                require(int(str(offset))==0 and handle>0,'GRAPH_OBSERVER_STORAGE_OFFSET')
                require(sid not in handles or handles[sid]==handle,'GRAPH_STORAGE_HANDLE_CHANGED')
                require(handle not in reverse or reverse[handle]==sid,'GRAPH_UNEXPECTED_POOL_HANDLE_ALIAS')
                handles[sid]=handle;reverse[handle]=sid
    require(set(physical)=={storage[e] for e in allowed},'GRAPH_NATIVE_STORAGE_COVERAGE')
    return {'status':'GRAPH_NATIVE_STORAGE_REUSE_BOUND','native_bases':rows,
            'observed_pool_physical_intervals':[{'storage_id':sid,'begin':begin,'end':end}
                                                for begin,end,sid in intervals],
            'data_buffer_handles':[{'storage_id':k,'data_handle':v} for k,v in sorted(handles.items())],
            'handle_meaning':'DLTensor.data runtime DataBuffer identity; physical bases come only from native requests'}
