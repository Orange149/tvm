"""Validate each physical drain before decoding the retained event sequence."""
from types import FunctionType
from yolo_scheduler_trace import events as previous


def validate_chunks(stream):
    capacity=stream['data_size'];chunks=stream['drain_chunks']
    if type(capacity) is not int or capacity<=64 or capacity&(capacity-1):
        raise ValueError('invalid physical ring capacity')
    if not chunks: raise ValueError('no scheduler drain evidence')
    cursor=offset=0;end=stream['enable_start_ns']
    for c in chunks:
        fields=('tail','head','head_after_copy','head_after_publish','begin_ns','end_ns','offset')
        if any(type(c.get(k)) is not int or c[k]<0 for k in fields):
            raise ValueError('invalid scheduler chunk field')
        if c['tail']!=cursor or c['offset']!=offset or not cursor<=c['head']<=c['head_after_copy']<=c['head_after_publish']:
            raise ValueError('scheduler chunk cursor discontinuity')
        if capacity-(c['head_after_publish']-cursor)<=64:
            raise ValueError('scheduler physical ring near full; pending loss possible')
        if not end<=c['begin_ns']<=c['end_ns']:
            raise ValueError('scheduler drain time invalid')
        offset+=c['head']-cursor;cursor=c['head'];end=c['end_ns']
    if cursor!=stream['head'] or chunks[-1]['tail']!=stream['tail']:
        raise ValueError('scheduler final physical cursor differs')
    raw=bytes.fromhex(stream['records_hex'])
    if offset!=len(raw) or offset!=stream['retained_bytes'] or offset>4*1024*1024:
        raise ValueError('scheduler retained extent differs')
    return raw


def decode_stream(stream,pid):
    raw=validate_chunks(stream)
    # Reuse the frozen strict record parser, replacing only its single-buffer
    # extent precondition with the physical per-drain checks above. Preserve
    # physical capacity in the original object and returned evidence.
    import ast, inspect
    tree=ast.parse(inspect.getsource(previous.decode_stream))
    fn=tree.body[0]
    # These two original checks describe a single snapshot, not the record ABI.
    # Check their exact syntax before removing them from this derived parser.
    expected=["if not 0<=tail<=head or head-tail>stream['data_size']:raise ValueError('scheduler ring head/tail invalid')",
              "if len(raw)!=head-tail:raise ValueError('scheduler ring byte count mismatch')"]
    predicates={ast.dump(ast.parse(s).body[0]) for s in expected}
    found=[n for n in fn.body if ast.dump(n) in predicates]
    if len(found)!=2: raise ValueError('frozen decoder extent hooks changed')
    fn.body=[n for n in fn.body if ast.dump(n) not in predicates]
    space=dict(previous.decode_stream.__globals__)
    exec(compile(ast.fix_missing_locations(tree),'<retained-scheduler-decoder>','exec'),space)
    decoded=space['decode_stream'](stream,pid)
    decoded.update(physical_capacity_bytes=stream['data_size'],drain_chunks=stream['drain_chunks'],
                   retained_bytes=len(raw),physical_capacity_checks_passed=True)
    return decoded


def validate_profile(profile,doc,affinity):
    if profile.get('ring_trace_revision')!=2 or profile.get('drain_policy')!='collector_completion':
        raise ValueError('scheduler drained trace revision missing')
    if type(profile.get('retention_end_ns')) is not int or not profile['stop_end_ns']<=profile['retention_end_ns']:
        raise ValueError('scheduler retention endpoint absent')
    if any(c['end_ns']>profile['retention_end_ns'] for s in profile['streams'] for c in s['drain_chunks']):
        raise ValueError('scheduler chunk extends beyond retention')
    fn=FunctionType(previous.validate_profile.__code__,
        {**previous.validate_profile.__globals__,'decode_stream':decode_stream})
    result=fn(profile,doc,affinity)
    result['physical_drain_evidence_valid']=True
    result['exact_wakeup_times']=False
    return result
