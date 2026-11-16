"""Pure checks for R/S board evidence; no device or target calls."""
from c3_tps_sync_release_local import require, read

def validate_completion(execution, index, kind):
    """Empty/stale/partial metadata is never accepted after a successful call."""
    require(execution.get('helper_returned') is True and execution.get('call_index')==index,
            'HELPER_DID_NOT_RETURN_FOR_CALL')
    require(not execution.get('observation_errors'),'HELPER_OBSERVATION_FAILED')
    meta=execution.get('metadata')
    required={'schema','started_calls','completed_calls','packed_calls_per_invocation',
              'adds_synchronize','parameter_count','last_kind'}
    require(isinstance(meta,dict) and required.issubset(meta),'HELPER_METADATA_INCOMPLETE')
    require(type(execution.get('counts')) is int and type(meta['started_calls']) is int and
            type(meta['completed_calls']) is int and
            execution['counts']==meta['started_calls']==meta['completed_calls']==index,
            'HIDDEN_OR_MISSING_TARGET_CALL')
    require(meta['schema']=='c3_official_window_once_helper_v1' and
            kind in ('diagnostic','timer') and meta['last_kind']==kind and meta['packed_calls_per_invocation']==1 and
            meta['adds_synchronize'] is False and meta['parameter_count']==4,
            'HELPER_INVOCATION_CONTRACT_CHANGED')
    return True


def accounting(root):
    """Use helper counters as evidence; a begin marker is only an intention."""
    units=[]
    for directory in sorted(root.glob('session_*')):
        if not directory.is_dir():
            continue
        begins=sorted(directory.glob('call_*_begin.json'))
        if not begins:
            continue
        executions={p.name.replace('_execution.json','_begin.json'):read(p)
                    for p in directory.glob('call_*_execution.json')}
        unit={'unit':directory.name,'registered_attempts':len(begins),
              'fully_certified':len(list(directory.glob('call_*_end.json')))}
        last=executions.get(begins[-1].name,{})
        observations={'started_calls':[0],'completed_calls':[0]}
        issues=[]
        for name,execution in executions.items():
            meta=execution.get('metadata',{})
            values={field:meta.get(field) for field in observations} if isinstance(meta,dict) else {}
            values['live_completed']=execution.get('counts')
            for field,value in values.items():
                if value is None:
                    continue
                if type(value) is not int or value<0:
                    issues.append(name+':INVALID_COUNTER:'+field)
                    continue
                destination='completed_calls' if field=='live_completed' else field
                observations[destination].append(value)
                if value>len(begins):
                    issues.append(name+':COUNTER_EXCEEDS_REGISTERED_ATTEMPTS:'+field)
        # Completion proves a start even when metadata has not been refreshed.
        observations['started_calls']+=observations['completed_calls']
        latest_completed=last.get('counts')
        if type(latest_completed) is int and latest_completed<max(observations['completed_calls']):
            issues.append('LIVE_COMPLETED_COUNTER_REGRESSED')
        for field in ('started_calls','completed_calls'):
            lower=max(observations[field])
            upper=None if issues else len(begins)
            if field=='completed_calls' and not issues and type(latest_completed) is int:
                upper=latest_completed
            # Last-success metadata only supplies a lower bound. It cannot
            # establish that a subsequent failing invocation never started.
            unit[field]={'exact':lower if upper==lower else None,
                         'lower_bound':lower,'upper_bound':upper}
        unit['evidence_issues']=issues
        units.append(unit)
    result={'registered_attempts':sum(u['registered_attempts'] for u in units),
            'fully_certified':sum(u['fully_certified'] for u in units),'units':units}
    for field in ('started_calls','completed_calls'):
        values=[u[field] for u in units]
        result[field]={'exact':sum(v['exact'] for v in values) if all(v['exact'] is not None for v in values) else None,
                       'lower_bound':sum(v['lower_bound'] for v in values),
                       'upper_bound':sum(v['upper_bound'] for v in values) if all(v['upper_bound'] is not None for v in values) else None}
    return result



def verify_native_storage(native, storage):
    # data_address is a host DataBuffer handle, NOT a DRAM physical address.
    require(len(storage)==4 and all(s['data_address'] and s['byte_offset']==0 for s in storage),
            'STORAGE_HANDLE_CONTRACT')
    bases=native['tensor_domain_dram_bases']
    require(sorted((b['tensor'],b['domain']) for b in bases)==[(0,2),(1,1),(2,3),(3,4)],
            'TENSOR_DOMAIN_BINDING_CHANGED')
    # Convert VTA vector-element addresses to physical bytes before comparing.
    widths={1:256,2:16,3:64,4:16}
    sizes={0:25088,1:294912,2:1024,3:50176}
    intervals=sorted((b['dram_base']*widths[b['domain']],
                      b['dram_base']*widths[b['domain']]+sizes[b['tensor']]) for b in bases)
    require(all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])), 'TENSOR_DRAM_REGIONS_OVERLAP')
    return True
