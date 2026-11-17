"""Dependency-complete units for the frozen Gluon BottleneckV2 expression DAG.

The structural granularity is the existing residual main/projection/merge
contract, not a newly searched fusion domain. Imports belong to preparation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import time

import tvm
from tvm import relay

from .graph import file_hash, write


def operation(node, name):
    return isinstance(node, relay.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == name


def children(node):
    if isinstance(node, relay.Call):
        return list(node.args)
    if isinstance(node, relay.Tuple):
        return list(node.fields)
    if isinstance(node, relay.TupleGetItem):
        return [node.tuple_value]
    return []


def outputs(body):
    if isinstance(body, relay.Tuple):
        return [leaf for item in body.fields for leaf in outputs(item)]
    if not isinstance(body.checked_type, relay.TensorType):
        raise ValueError('terminal output must be a tensor or tuple of tensors')
    return [body]


class Replace(relay.ExprMutator):
    def __init__(self, replacements):
        super().__init__()
        self.replacements = replacements

    def visit(self, node):
        if node in self.replacements:
            return self.replacements[node]
        return super().visit(node)


class UnitGraph:
    """38 expected structural units, proven from the real typed graph."""
    def __init__(self, mod):
        self.mod = relay.transform.InferType()(mod)
        self.nodes = []
        relay.analysis.post_order_visit(self.mod['main'].body, self.nodes.append)
        if any(isinstance(n, (relay.Let, relay.If, relay.Function)) for n in self.nodes):
            raise ValueError('only a first-order expression DAG is supported')
        self.ids = {node: i for i, node in enumerate(self.nodes)}
        self.users = defaultdict(list)
        for node in self.nodes:
            for child in children(node):
                self.users[child].append(node)
        self.calls = [n for n in self.nodes if isinstance(n, relay.Call)]
        if any(not isinstance(n.op, tvm.ir.Op) for n in self.calls):
            raise ValueError('opaque function calls are not valid unit boundaries')
        self.owner, self.units = {}, []
        self.extract()

    def assign(self, node, owner):
        if node in self.owner or isinstance(node, (relay.Var, relay.Constant)):
            return
        for child in children(node):
            self.assign(child, owner)
        self.owner[node] = owner

    def extend(self, root, owner, *, stem=False):
        pending, seen = [root], {root}
        while pending:
            node = pending.pop()
            for user in self.users[node]:
                allowed = operation(user, 'annotation.stop_fusion') or (
                    operation(user, 'cast') and (stem or str(user.attrs.dtype) == 'int8'))
                if stem:
                    allowed |= operation(user, 'round') or operation(user, 'clip') or (
                        operation(user, 'multiply') and any(isinstance(arg, relay.Constant) for arg in user.args))
                if allowed and user not in seen:
                    seen.add(user)
                    self.assign(user, owner)
                    pending.append(user)

    def add_unit(self, name, kind, root, conv_count, *, block=None, extend=False, stem=False):
        index = len(self.units)
        self.units.append({'index': index, 'name': name, 'kind': kind,
                           'expected_conv_count': conv_count, 'block': block})
        self.assign(root, index)
        if extend:
            self.extend(root, index, stem=stem)

    def branch_convs(self, root):
        result, seen = [], set()
        def visit(node):
            if node in seen or node in self.owner:
                return
            seen.add(node)
            if operation(node, 'nn.conv2d'):
                result.append(node)
            for child in children(node):
                visit(child)
        visit(root)
        return result

    def extract(self):
        pools = [n for n in self.calls if operation(n, 'nn.max_pool2d')]
        adds = [n for n in self.calls if operation(n, 'add')
                and all(isinstance(arg, relay.Call) for arg in n.args)
                and isinstance(n.checked_type, relay.TensorType) and len(n.checked_type.shape) == 4]
        if len(pools) != 1 or len(adds) != 16:
            raise ValueError('expected one stem pool and 16 genuine residual adds; got '
                             + str((len(pools), len(adds))))
        channels = [256] * 3 + [512] * 4 + [1024] * 6 + [2048] * 3
        if [int(n.checked_type.shape[1]) for n in adds] != channels:
            raise ValueError('residual block order/channels differ from Gluon ResNet50-v2')
        self.add_unit('stem', 'stem', pools[0], 1, extend=True, stem=True)
        first_blocks = (0, 3, 7, 13)
        stage_lengths = (3, 4, 6, 3)
        for number, add in enumerate(adds):
            stage = next(i for i, start in enumerate(first_blocks)
                         if start <= number < start + stage_lengths[i])
            prefix = f'layer{stage + 1}_block{number - first_blocks[stage]}'
            branches = [(arg, self.branch_convs(arg)) for arg in add.args]
            main = [pair for pair in branches if len(pair[1]) == 3]
            skip = [pair for pair in branches if len(pair[1]) in (0, 1)]
            if len(main) != 1 or len(skip) != 1:
                raise ValueError('cannot prove three-convolution main / projection-or-identity ownership: '
                                 + str((number, [len(pair[1]) for pair in branches])))
            main_root, main_convs = main[0]
            if [list(map(int, c.attrs.kernel_size)) for c in main_convs] != [[1, 1], [3, 3], [1, 1]]:
                raise ValueError('Gluon bottleneck must contain 1x1, 3x3, 1x1 convolutions')
            expected_stride = 2 if number in (3, 7, 13) else 1
            if [list(map(int, c.attrs.strides)) for c in main_convs] != [
                    [1, 1], [expected_stride, expected_stride], [1, 1]]:
                raise ValueError('expected Gluon v2 downsampling on the middle 3x3 convolution')
            if operation(main_root, 'cast') and str(main_root.attrs.dtype) == 'int32':
                main_root = main_root.args[0]
            self.add_unit(prefix + '_main_preadd', 'main_preadd', main_root, 3, block=number)
            skip_root, skip_convs = skip[0]
            projection = number in first_blocks
            if len(skip_convs) != int(projection):
                raise ValueError('unexpected Gluon v2 projection layout')
            if projection:
                conv = skip_convs[0]
                if list(map(int, conv.attrs.kernel_size)) != [1, 1] or list(map(int, conv.attrs.strides)) != [expected_stride, expected_stride]:
                    raise ValueError('projection shape/stride mismatch')
                if operation(skip_root, 'cast') and str(skip_root.attrs.dtype) == 'int32':
                    skip_root = skip_root.args[0]
                self.add_unit(prefix + '_skip_proj', 'projection', skip_root, 1, block=number)
            # V2 has no post-add ReLU. Preserve the existing expression and only
            # own transparent quantized bridge nodes, never invent an epilogue.
            if any(operation(user, 'nn.relu') for user in self.users[add]):
                raise ValueError('unexpected direct post-add ReLU in the frozen v2 graph')
            self.add_unit(prefix + '_merge', 'residual_merge', add, 0, block=number, extend=True)
        self.add_unit('head', 'head', self.mod['main'].body, 0)
        if len(self.units) != 38 or any(node not in self.owner for node in self.calls):
            raise ValueError('incomplete ResNet50-v2 unit ownership')
        for node in self.nodes:
            if node not in self.owner:
                continue
            if any(child in self.owner and self.owner[child] > self.owner[node]
                   for child in children(node)):
                raise ValueError('unit ownership requires backwards execution')
        for unit in self.units:
            nodes = [node for node in self.calls if self.owner[node] == unit['index']]
            counts = Counter(node.op.name for node in nodes)
            if counts['nn.conv2d'] != unit['expected_conv_count']:
                raise ValueError('unit owns an unintended convolution')
            unit.update(operator_counts=dict(counts), call_node_ids=[self.ids[node] for node in nodes])

    @lru_cache(None)
    def frontier(self, boundary):
        if type(boundary) is not int or not 0 <= boundary <= len(self.units):
            raise ValueError('boundary outside unit graph')
        if boundary == 0:
            return list(self.mod['main'].params)
        if boundary == len(self.units):
            return outputs(self.mod['main'].body)
        return [node for node in self.nodes
                if isinstance(node, (relay.Var, relay.Call, relay.TupleGetItem))
                and isinstance(node.checked_type, relay.TensorType)
                and self.owner.get(node, -1) < boundary
                and any(self.owner.get(user, -1) >= boundary for user in self.users[node])]

    @lru_cache(None)
    def stage(self, begin, end):
        if not 0 <= begin < end <= len(self.units):
            raise ValueError('invalid stage interval')
        incoming, outgoing = self.frontier(begin), self.frontier(end)
        visited = set()
        def check(node):
            if node in visited or node in incoming:
                return
            visited.add(node)
            if isinstance(node, relay.Call) and not begin <= self.owner[node] < end:
                raise ValueError('stage would recompute a call outside its interval')
            for child in children(node):
                check(child)
        for node in outgoing:
            check(node)
        params = [relay.var(f'edge{begin}_{i}', type_annotation=node.checked_type)
                  for i, node in enumerate(incoming)]
        body = Replace(dict(zip(incoming, params))).visit(relay.Tuple(outgoing))
        if set(relay.analysis.free_vars(body)) != set(params):
            raise ValueError('frontier is not dependency-complete')
        mod = relay.transform.InferType()(tvm.IRModule.from_expr(relay.Function(params, body)))
        return mod, incoming, outgoing

    def recompose(self, ranges):
        values, position = list(self.mod['main'].params), 0
        for begin, end in ranges:
            if begin != position:
                raise ValueError('ranges do not form a continuous partition')
            mod, _, _ = self.stage(begin, end)
            function = mod['main']
            body = Replace(dict(zip(function.params, values))).visit(function.body)
            values = list(body.fields)
            position = end
        expected = outputs(self.mod['main'].body)
        if position != len(self.units) or len(values) != len(expected) or any(
                not tvm.ir.structural_equal(a, b) for a, b in zip(values, expected)):
            raise ValueError('unit recomposition changed the canonical outputs')
        return True


def prepare_units(graph_directory, output):
    from .domain import programs, summary
    graph_directory, output = Path(graph_directory).resolve(), Path(output).resolve()
    source = graph_directory / 'canonical_qir.json'
    if output.exists():
        status = json.loads((output / 'UNIT_STATUS.json').read_text())
        if status['source_sha256'] != file_hash(source):
            raise ValueError('existing unit proof uses a different common graph')
        return status
    output.mkdir(parents=True)
    started = time.monotonic()
    graph = UnitGraph(tvm.ir.load_json(source.read_text()))
    n = len(graph.units)
    graph.recompose([(i, i + 1) for i in range(n)])
    checked = 0
    for begin in range(n):
        for end in range(begin + 1, n + 1):
            graph.recompose([(i, i + 1) for i in range(begin)] + [(begin, end)]
                            + [(i, i + 1) for i in range(end, n)])
            checked += 1
        print(f'ResNet50 unit interval checks {checked}/{n*(n+1)//2}', flush=True)
    boundaries = []
    for boundary in range(n + 1):
        tensors = graph.frontier(boundary)
        boundaries.append({'before_unit': boundary, 'tensor_count': len(tensors),
                           'tensors': [{'node_id': graph.ids.get(node),
                                        'shape': list(map(int, node.checked_type.shape)),
                                        'dtype': str(node.checked_type.dtype),
                                        'producer_unit': graph.owner.get(node, -1),
                                        'consumer_units': sorted({graph.owner[u] for u in graph.users[node]
                                                                  if u in graph.owner})}
                                       for node in tensors]})
    domain = summary(graph.units)
    if (n, domain['search_CPU_stages'], domain['VTA_stages'], domain['total_basic_signature_count']) != (38, 314, 299, 2751):
        raise ValueError('actual typed graph exceeds or contradicts the approved conditional domain: ' + str(domain))
    write(output / 'units_and_boundaries.json', {'units': graph.units, 'boundaries': boundaries})
    write(output / 'programs.json', programs(graph.units))
    write(output / 'DOMAIN_COUNTS.json', domain)
    status = {'status': 'STRUCTURALLY_VERIFIED_LOCAL', 'units': n, 'source_sha256': file_hash(source),
              'convolutions': 53, 'residual_blocks': 16, 'projection_units': 4,
              'terminal_output_count': len(outputs(graph.mod['main'].body)),
              'all_interval_recompositions_exact': True, 'interval_checks': checked,
              'unit_recomposition_exact': True, 'extra_quantize_calls': 0,
              'frontier_multi_tensor_boundaries': [b['before_unit'] for b in boundaries if b['tensor_count'] > 1],
              'elapsed_seconds': time.monotonic() - started, 'board_accessed': False,
              'ARM_correctness_verified': False,
              'domain_limit': 'same stem/main/projection/merge/head units; VTA begins only at main_preadd; CPU endpoints; 1-3 islands',
              'source_code': {Path(__file__).name: file_hash(__file__)}}
    write(output / 'UNIT_STATUS.json', status)
    return status
