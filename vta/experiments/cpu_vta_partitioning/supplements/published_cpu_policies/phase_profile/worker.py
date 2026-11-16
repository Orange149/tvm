"""Reuse the verified trace checks, replacing its old CPU0-2-only target."""
from functools import lru_cache
import hashlib
import inspect


@lru_cache(maxsize=1)
def parser():
    from wait_diagnosis.runtime import trace_summary
    source=inspect.getsource(trace_summary)
    before="    if ('0' not in stage_stats or stage_stats['0']['program']!='cpu_00_02' or\n            not stage_stats['0']['kernel_launches']):raise ValueError('actual target CPU module not intercepted')"
    after="    if ('0' not in stage_stats or not stage_stats['0']['kernel_launches']):raise ValueError('actual profiled CPU module not intercepted')"
    if source.count(before)!=1:raise ValueError('verified worker parser target anchor changed')
    patched=source.replace(before,after)
    namespace=dict(trace_summary.__globals__)
    exec(compile(patched,'<same-clock profile generic first CPU target>','exec'),namespace)
    return namespace['trace_summary'],dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),
        adapted_sha256=hashlib.sha256(patched.encode()).hexdigest(),
        change='first stage must have its own module kernel records; remove CPU0-2-only name requirement',
        clock_callback_module_TID_and_affinity_checks_changed=False)


def validate_workers(raw,task,bundle,affinity):
    function,proof=parser()
    result=function(raw,{**task,'spec':{**task['spec'],'policy':'original'}},bundle,affinity)
    result['parser_adaptation']=proof
    return result
