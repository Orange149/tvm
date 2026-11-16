"""Zero-execution finite FIFO proof for the existing serial action model.

This independent diagnostic does not change a runtime, a closed attempt, or
qualification policy. It uses the same instruction actions as
c3_tps_window_progress: read, pop-prev, pop-next, execute, push-prev,
push-next. Each action is atomic; enabled actions eventually finish. Queue
reads release capacity. A transfer's cycle latency is deliberately absent.

The graph includes fetch order, actor order, token availability, token
capacity, and command FIFO capacity. For a FIFO of capacity C the (j+C)th
write requires the jth read. This is precisely the missing finite capacity
edge, not a relaxation to infinite queues. An acyclic graph proves progress:
every reachable completed-action set is an ideal; a minimal unfinished node
has every enabling prerequisite satisfied. No enumeration of interleavings
is needed. A cycle is reported unresolved, never baseline illegality.
"""
from collections import deque
from c3_tps_window_progress import LANES


def certify_action_graph(records, command_depths, token_depths):
    if any(type(command_depths.get(s)) is not int or command_depths[s] <= 0
           for s in (1, 2, 3)) or any(
               type(token_depths.get(l)) is not int or token_depths[l] <= 0 for l in LANES):
        raise ValueError('MISSING_FINITE_QUEUE_CAPACITY')
    n = len(records)
    # Seven nodes per instruction: global fetch, followed by six actor actions.
    edges = [[] for _ in range(7*n)]
    indegree = [0]*(7*n)
    counts = {}

    def add(a, b, kind):
        edges[a].append(b); indegree[b] += 1
        counts[kind] = counts.get(kind, 0)+1

    queues = {s: [] for s in (1, 2, 3)}
    producers = {l: [] for l in LANES}
    consumers = {l: [] for l in LANES}
    for i, rec in enumerate(records):
        stage = rec.get('stage'); bits = rec.get('bits', ())
        if rec.get('index') != i or stage not in queues or len(bits) != 4:
            raise ValueError('MALFORMED_NATIVE_INSTRUCTION')
        base = 7*i
        if i: add(base-7, base, 'global_fetch_order')
        for phase in range(6): add(base+phase, base+phase+1, 'instruction_actions')
        if queues[stage]: add(7*queues[stage][-1]+6, base+1, 'actor_order')
        q = queues[stage]; capacity = command_depths[stage]
        if len(q) >= capacity:
            add(7*q[-capacity]+1, base, 'command_fifo_capacity')
        q.append(i)
        lanes = ((stage-1, stage), (stage+1, stage),
                 (stage, stage-1), (stage, stage+1))
        for k, (flag, lane) in enumerate(zip(bits, lanes)):
            if flag not in (0, 1) or flag and lane not in LANES:
                raise ValueError('INVALID_NATIVE_TOKEN_BIT')
            if flag:
                # Actor phases 1/2 pop; phases 4/5 push.
                node = base+((2, 3, 5, 6)[k])
                (consumers if k < 2 else producers)[lane].append(node)
    for lane in LANES:
        ps, cs = producers[lane], consumers[lane]
        if len(ps) != len(cs): raise ValueError('UNMATCHED_NATIVE_TOKENS')
        for p, c in zip(ps, cs): add(p, c, 'token_availability')
        capacity = token_depths[lane]
        for j in range(capacity, len(ps)):
            add(cs[j-capacity], ps[j], 'token_fifo_capacity')
    todo = deque(i for i, d in enumerate(indegree) if d == 0)
    visited = 0
    while todo:
        i = todo.popleft(); visited += 1
        for j in edges[i]:
            indegree[j] -= 1
            if indegree[j] == 0: todo.append(j)
    return {'status': 'FINITE_FIFO_PROGRESS_PROVED' if visited == 7*n
            else 'CYCLIC_ACTION_GRAPH_UNRESOLVED',
            'method': 'finite_fifo_action_graph', 'instructions': n,
            'action_nodes': 7*n, 'topologically_completed_nodes': visited,
            'edge_counts': counts,
            'command_depths': {str(k): v for k, v in command_depths.items()},
            'token_depths': {str(k): v for k, v in token_depths.items()},
            'token_pairs': sum(map(len, producers.values())),
            'initial_and_terminal_tokens': [0]*4,
            'finite_action_completion_assumed': True,
            'same_action_model_as_existing_enumerator': True,
            'proves_cycle_timing': False, 'target_executions': 0}
