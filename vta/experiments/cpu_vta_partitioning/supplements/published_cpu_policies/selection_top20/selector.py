"""Pure selection: no file IO, TVM imports, throughput feedback or score fitting."""
from itertools import combinations
from .upstream import submodular_pick

CLASSES = ('disjoint_open', 'overlap_open', 'fixed_groups_serial')

def execution_class(gate, masks):
    if gate == 'resource_group':
        if any(a != b and set(a)&set(b) for a,b in combinations(masks, 2)):
            raise ValueError('partially overlapping serial groups')
        return 'fixed_groups_serial'
    if gate != 'off':
        raise ValueError('unknown execution mode')
    return 'overlap_open' if any(set(a)&set(b) for a,b in combinations(masks, 2)) else 'disjoint_open'

def relation(a, b):
    a,b = set(a),set(b)
    return ('same' if a == b else 'disjoint' if not a&b else
            'first_contains_second' if b < a else 'second_contains_first' if a < b else 'crossing')

def features(topology, context, gate, cpu):
    """33 categorical dimensions, fixed before any pipeline observations.

    CPU tuples are (begin,end,threads,mask,caller); absent positions use -1.
    Hashes and observation-derived values are deliberately not accepted.
    """
    masks = [r[3] for r in cpu]
    values = [tuple(topology), context['id'], sum(r[0]=='vta' for r in topology),
              execution_class(gate,masks), gate]
    for i in range(4):
        values.extend((*cpu[i][:3], sum(1<<c for c in cpu[i][3]), cpu[i][4])
                      if i < len(cpu) else (-1,)*5)
    for i,j in combinations(range(4),2):
        values.append(relation(masks[i],masks[j]) if j < len(masks) else -1)
    values.extend(sum(1<<c for c in context[k]) for k in ('host_mask','control_mask'))
    assert len(values) == 33
    return tuple(values)

def seeded_pick(vectors, seeds, count):
    """Reuse the unmodified upstream greedy function with seeded coverage.

    Already-covered values map to zero; other values keep distinct codes.
    An initialization row is picked first and discarded. Its score exceeds
    every possible real marginal gain, without scoring any real candidate.
    """
    if not vectors or not 0 <= count <= len(vectors):
        if count == 0:return []
        raise ValueError('insufficient candidates')
    width = len(vectors[0])
    if not width or any(len(v)!=width for v in [*vectors,*seeds]):
        raise ValueError('inconsistent feature dimensions')
    covered = [{v[i] for v in seeds} for i in range(width)]
    codes = [{} for _ in range(width)]
    encoded = []
    for row in vectors:
        value = []
        for i,x in enumerate(row):
            if x in covered[i]:value.append(0)
            else:
                if x not in codes[i]:codes[i][x] = len(codes[i])+1
                value.append(codes[i][x])
        encoded.append(value)
    picked = submodular_pick([width+1]+[0]*len(vectors), [[0]*width]+encoded, count+1, knob_weight=1)
    if not picked or picked[0]!=0:raise ValueError('coverage initialization failed')
    return [i-1 for i in picked[1:]]

def choose(records, anchors, contexts, expand, *, count=20, fallback=()):
    """Records have compact identity, canonical key, feature vector and score."""
    if len(anchors)!=5 or len({r['key'] for r in anchors})!=5:
        raise ValueError('exactly five distinct static anchors required')
    ordered = sorted(records,key=lambda r:(r['score_ns'],r['canonical']))
    minima = {}
    for row in ordered:
        label = row['context'],row['class']
        minima.setdefault(label,row['score_ns'])
    eligible = [r for r in ordered if r['score_ns']*100 <= minima[(r['context'],r['class'])]*115]
    chosen, seen, reasons = [],set(),{}
    def add(row, reason):
        if row['key'] in seen:return False
        chosen.append(row);seen.add(row['key']);reasons[row['key']]=reason;return True
    for row in anchors:add(row,'preserved_static_first5')
    missing_contexts = set(contexts)-{r['context'] for r in chosen}
    representatives = []
    for ctx in missing_contexts:
        match = next((r for r in eligible if r['context']==ctx),None)
        if match is not None:representatives.append(match)
    for row in sorted(representatives,key=lambda r:(r['score_ns'],r['canonical'])):
        add(row,'uncovered_resource_context')
    missing_classes = set(CLASSES)-{r['class'] for r in chosen}
    representatives = []
    for cls in missing_classes:
        match = next((r for r in eligible if r['class']==cls),None)
        if match is not None:representatives.append(match)
    for row in sorted(representatives,key=lambda r:(r['score_ns'],r['canonical'])):
        add(row,'uncovered_CPU_execution_class')
    remaining = [r for r in eligible if r['key'] not in seen]
    slots = min(max(0,count-len(chosen)),len(remaining))
    for i in seeded_pick([r['features'] for r in remaining],[r['features'] for r in chosen],slots):
        add(remaining[i],'autotvm_greedy_feature_coverage')
    for row in fallback:
        if len(chosen)>=count:break
        add(row,'pure_static_fallback')
    if len(chosen)>count:raise ValueError('coverage seeds exceed budget')
    if len(chosen)!=count:raise ValueError('cannot fill twenty distinct configurations')
    rows = []
    for order,r in enumerate(chosen,1):
        value=expand(r);value.update(selection_order=order,selection_reason=reasons[r['key']],
                                     CPU_execution_class=r['class'])
        rows.append(value)
    return rows, {'candidate_count':len(records),'eligible_count':len(eligible),
        'class_minima_ns':{ctx+':'+cls:value for (ctx,cls),value in sorted(minima.items())},
        'context_coverage':sorted({r['context'] for r in chosen}),
        'class_coverage':sorted({r['class'] for r in chosen}),
        'missing_contexts':sorted(set(contexts)-{r['context'] for r in chosen}),
        'feature_dimensions':33,'all_feature_weights':1,'pipeline_FPS_used':False,
        'global_score_guard':False,'class_guard_ratio':'115/100'}
