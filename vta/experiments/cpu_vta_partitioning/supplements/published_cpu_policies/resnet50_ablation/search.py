"""Private adapter of the sealed exact DP, with policy-correct resource bounds.

All traversal, prefix retention, checkpointing and canonical Top-K code is reused
unchanged. Functions receive a private globals mapping, never monkey-patching the
old module. One checked source substitution generalizes its distribution guard.
The scalar oracle always evaluates all 15 core subsets. For this nine-context
domain the DP uses the equivalent singleton/full-pool host formula.
"""
import inspect
from types import FunctionType

from resnet50_validation import search as original
from .common import scoring_sources, settings
from .model import score


def value(label, ctx, gate, policy, extra=(0, 0, 0, 0, 0)):
    future_vta, future_host, future_ddr, future_cpu_max, future_cpu_work = extra
    demand = label[:4]
    host = label[4] + future_host if policy['host_cpu_demand'] else 0
    allowed = ctx['host_mask']
    result = max(label[5] + future_vta, label[6] + future_ddr,
                 label[7], future_cpu_max, *demand)
    if host:
        if len(allowed) == 1:
            result = max(result, demand[allowed[0]] + host)
        elif tuple(allowed) == (0, 1, 2, 3):
            result = max(result, (sum(demand) + host + 3) // 4)
        else:
            # General masks remain correct even though the frozen nine contexts
            # currently use only all four cores or a singleton for their host.
            bits = sum(1 << c for c in allowed)
            for subset in range(1, 16):
                if subset & bits == bits:
                    work = sum(demand[c] for c in range(4) if subset & (1 << c)) + host
                    result = max(result, (work + subset.bit_count() - 1) // subset.bit_count())
    if policy['team_guard']:
        result = max(result, *label[8:12])
    if gate == 'resource_group':
        result = max(result, *label[12:])
    if policy['host_guard']:
        result = max(result, *(demand[c] + host for c in allowed))
    # Safe relaxed suffix work. Host work can enter this pool bound only if all
    # its allowed cores belong to the compute pool, and only when enabled.
    pool = ctx['compute_pool']
    work = sum(demand[c] for c in pool) + future_cpu_work
    if all(c in pool for c in allowed):
        work += host
    return max(result, (work + len(pool) - 1) // len(pool))


def engine(key):
    policy = settings(key)
    namespace = dict(vars(original))
    # Rebind mutually recursive helper globals to this private namespace.
    for name, fn in vars(original).items():
        if isinstance(fn, FunctionType) and fn.__globals__ is vars(original):
            copy = FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
            copy.__kwdefaults__ = fn.__kwdefaults__
            namespace[name] = copy
    namespace['ALGORITHM'] = 'resnet50_five_policy_exact_resource_labels_v1'
    namespace['FROZEN_DISTRIBUTION'] = policy['distribution']
    namespace['_implementation'] = lambda: dict(scoring_sources(), policy=policy)

    def bound(label, ctx, method, gate, extra=(0, 0, 0, 0, 0)):
        if method != 'complete':
            raise ValueError('internal ablation domain is the full CPU resource domain')
        return value(label, ctx, gate, policy, extra)

    def incumbents(values, programs, costs, methods, unit_count, max_islands, contexts, k):
        allowed = {c['id'] for c in contexts}
        result = {'complete': []}
        for row in (values or {}).get('complete', ()):
            ctx = original.domain.context(row['context_id'])
            if ctx['id'] not in allowed:
                continue
            cfg = original.domain.candidate(row['stages'], ctx, row['execution_mode'])
            original.domain.validate_candidate(cfg, programs, unit_count=unit_count,
                                               max_vta_islands=max_islands)
            original._add_complete(result['complete'], cfg, score(cfg, costs, key)['score_ns'], k)
        return result

    namespace['_value'] = bound
    namespace['_check_incumbents'] = incumbents
    text = inspect.getsource(original.rank)
    before = 'if "complete" in methods and costs.distribution != "session_weights":'
    if text.count(before) != 1:
        raise ValueError('sealed DP distribution guard changed')
    text = text.replace(before, 'if "complete" in methods and costs.distribution != FROZEN_DISTRIBUTION:')
    exec(compile(text, __file__ + ':' + key, 'exec'), namespace)
    return namespace['rank']


def rank(programs, costs, *, key, **kwargs):
    if costs.distribution != settings(key)['distribution']:
        raise ValueError('cost distribution differs from the policy')
    return engine(key)(programs, costs, methods=('complete',), **kwargs)
