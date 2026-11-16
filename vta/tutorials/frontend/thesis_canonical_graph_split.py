#!/usr/bin/env python3
"""Prepare 21 typed units from the same quantized ResNet18 expression DAG."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tvm
from tvm import relay
from thesis_supplement_run import sha256, write_new_json


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


class Replace(relay.ExprMutator):
    def __init__(self, replacements):
        super().__init__()
        self.replacements = replacements

    def visit(self, expr):
        if expr in self.replacements:
            return self.replacements[expr]
        return super().visit(expr)


class UnitGraph:
    def __init__(self, mod):
        self.mod = relay.transform.InferType()(mod)
        self.nodes = []
        relay.analysis.post_order_visit(self.mod['main'].body, self.nodes.append)
        if any(isinstance(n, (relay.Let, relay.If, relay.Function)) for n in self.nodes):
            raise ValueError('Only a first-order straight expression DAG is supported')
        self.ids = {node: i for i, node in enumerate(self.nodes)}
        self.users = defaultdict(list)
        for node in self.nodes:
            for child in children(node):
                self.users[child].append(node)
        self.calls = [n for n in self.nodes if isinstance(n, relay.Call)]
        if any(not isinstance(n.op, tvm.ir.Op) for n in self.calls):
            raise ValueError('Unsupported function-call boundary')
        self.owner, self.units = {}, []
        self.extract()

    def assign_ancestors(self, node, owner):
        if not isinstance(node, relay.Call) or node in self.owner:
            return
        for child in node.args:
            self.assign_ancestors(child, owner)
        self.owner[node] = owner

    def extend(self, root, owner, stem=False):
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
                    self.assign_ancestors(user, owner)
                    pending.append(user)

    def add_unit(self, name, root, expected_conv, extend=False, stem=False):
        owner = len(self.units)
        self.units.append({'index': owner, 'name': name, 'expected_conv_count': expected_conv})
        self.assign_ancestors(root, owner)
        if extend:
            self.extend(root, owner, stem)

    def branch_convs(self, branch):
        result, visited = [], set()
        def walk(node):
            if node in visited or node in self.owner:
                return
            visited.add(node)
            if operation(node, 'nn.conv2d'):
                result.append(node)
            for child in children(node):
                walk(child)
        walk(branch)
        return result

    def extract(self):
        pools = [n for n in self.calls if operation(n, 'nn.max_pool2d')]
        adds = [n for n in self.calls if operation(n, 'add')
                and all(isinstance(arg, relay.Call) for arg in n.args)
                and isinstance(n.checked_type, relay.TensorType) and len(n.checked_type.shape) == 4]
        if len(pools) != 1 or len(adds) != 8:
            raise ValueError('Expected one stem pool and eight genuine residual adds')
        expected_channels = [64, 64, 128, 128, 256, 256, 512, 512]
        if [int(n.checked_type.shape[1]) for n in adds] != expected_channels:
            raise ValueError('Unexpected residual block order or channels')
        self.add_unit('stem', pools[0], 1, extend=True, stem=True)
        for number, add in enumerate(adds):
            prefix = f'layer{number // 2 + 1}_block{number % 2}'
            branches = [(arg, self.branch_convs(arg)) for arg in add.args]
            main = [pair for pair in branches if len(pair[1]) == 2]
            skip = [pair for pair in branches if len(pair[1]) in (0, 1)]
            if len(main) != 1 or len(skip) != 1:
                raise ValueError('Residual main/skip ownership could not be proven')
            main_root, main_convs = main[0]
            if any(list(map(int, n.attrs.kernel_size)) != [3, 3] for n in main_convs):
                raise ValueError('Expected two 3x3 main-path convolutions')
            if operation(main_root, 'cast') and str(main_root.attrs.dtype) == 'int32':
                main_root = main_root.args[0]
            self.add_unit(prefix + '_main_preadd', main_root, 2)
            skip_root, skip_convs = skip[0]
            projection = number in (2, 4, 6)
            if len(skip_convs) != int(projection):
                raise ValueError('Unexpected projection layout')
            if projection:
                if list(map(int, skip_convs[0].attrs.kernel_size)) != [1, 1] or list(map(int, skip_convs[0].attrs.strides)) != [2, 2]:
                    raise ValueError('Expected 1x1 stride2 projection')
                if operation(skip_root, 'cast') and str(skip_root.attrs.dtype) == 'int32':
                    skip_root = skip_root.args[0]
                self.add_unit(prefix + '_skip_proj', skip_root, 1)
            relus = [n for n in self.users[add] if operation(n, 'nn.relu')]
            if len(relus) != 1:
                raise ValueError('Expected a unique post-add ReLU')
            clips = [n for n in self.users[relus[0]] if operation(n, 'clip')]
            if number < 7:
                if len(clips) != 1:
                    raise ValueError('Expected a unique internal post-ReLU clip')
                tail_root = clips[0]
            else:
                # The existing once-quantized terminal tail casts directly to int8.
                # Preserve its wrap behaviour; introducing a clip changes the network.
                casts = [n for n in self.users[relus[0]] if operation(n, 'cast') and str(n.attrs.dtype) == 'int8']
                if clips or len(casts) != 1:
                    raise ValueError('Unexpected terminal quantization semantics')
                tail_root = relus[0]
            self.add_unit(prefix + '_add_relu_tail', tail_root, 0, extend=True)
        self.add_unit('head', self.mod['main'].body, 0)
        if len(self.units) != 21 or len(self.owner) != len(self.calls):
            raise ValueError('Incomplete 21-unit ownership')
        for node in self.calls:
            for child in node.args:
                if isinstance(child, relay.Call) and self.owner[child] > self.owner[node]:
                    raise ValueError('Ownership requires backwards execution')
        for unit in self.units:
            calls = [n for n in self.calls if self.owner[n] == unit['index']]
            counts = Counter(n.op.name for n in calls)
            if counts['nn.conv2d'] != unit['expected_conv_count']:
                raise ValueError('Unit convolution ownership differs from the original architecture')
            unit.update(operator_counts=dict(counts), call_node_ids=[self.ids[n] for n in calls])

    def frontier(self, boundary):
        if boundary == 0:
            return list(self.mod['main'].params)
        if boundary == 21:
            return [self.mod['main'].body]
        return [n for n in self.calls if self.owner[n] < boundary and
                any(isinstance(user, relay.Call) and self.owner[user] >= boundary for user in self.users[n])]

    def stage(self, begin, end):
        inputs, outputs = self.frontier(begin), self.frontier(end)
        visited = set()
        def check_range(node):
            if node in visited or node in inputs:
                return
            visited.add(node)
            if isinstance(node, relay.Call) and not begin <= self.owner[node] < end:
                raise ValueError('Cut would recompute an operation outside its assigned units')
            for child in children(node):
                check_range(child)
        for node in outputs:
            check_range(node)
        variables = [relay.var(f'edge{begin}_{i}', type_annotation=node.checked_type) for i, node in enumerate(inputs)]
        body = Replace(dict(zip(inputs, variables))).visit(relay.Tuple(outputs))
        if set(relay.analysis.free_vars(body)) != set(variables):
            raise ValueError('Cut is not dependency closed')
        function = relay.Function(variables, body)
        typed = relay.transform.InferType()(tvm.IRModule.from_expr(function))
        return typed, inputs, outputs

    def recompose(self, ranges):
        values = list(self.mod['main'].params)
        for begin, end in ranges:
            stage, _, _ = self.stage(begin, end)
            function = stage['main']
            body = Replace(dict(zip(function.params, values))).visit(function.body)
            if not isinstance(body, relay.Tuple):
                raise ValueError('Transport contract changed')
            values = list(body.fields)
        if len(values) != 1 or not tvm.ir.structural_equal(values[0], self.mod['main'].body):
            raise ValueError('Stage recomposition changed the canonical expression')
        return True


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    original = tvm.ir.load_json(source.read_text())
    # Apply the same explicit common-expression pass before every arm is split.
    # This does not quantize again; preserve both source and resulting graph hashes.
    canonical = relay.transform.InferType()(relay.transform.EliminateCommonSubexpr()(original))
    (output / 'canonical_qir.json').open('x').write(tvm.ir.save_json(canonical))
    (output / 'canonical_qir.txt').open('x').write(canonical.astext(show_meta_data=False))
    graph = UnitGraph(canonical)
    unit_ranges = [(i, i + 1) for i in range(21)]
    integrated_ranges = [(0, 3), (3, 18), (18, 21)]
    graph.recompose(unit_ranges)
    graph.recompose(integrated_ranges)
    boundaries = []
    for boundary in range(22):
        edges = graph.frontier(boundary)
        boundaries.append({'before_unit': boundary, 'tensor_count': len(edges),
            'tensors': [{'node_id': graph.ids.get(node), 'checked_type': str(node.checked_type),
                        'producer_unit': graph.owner.get(node, -1),
                        'consumer_units': sorted({graph.owner[user] for user in graph.users[node] if user in graph.owner}),
                        'transport': 'native dtype/shape; no inferred float scaling or packing adapter qualification'} for node in edges]})
    write_new_json(output / 'units_and_boundaries.json', {'units': graph.units, 'boundaries': boundaries})
    stages = []
    for kind, ranges in [('units', unit_ranges), ('integrated', integrated_ranges)]:
        directory = output / kind
        directory.mkdir()
        for begin, end in ranges:
            stage, inputs, outputs = graph.stage(begin, end)
            name = f'{begin:02}_{end-1:02}'
            (directory / (name + '.json')).open('x').write(tvm.ir.save_json(stage))
            (directory / (name + '.txt')).open('x').write(stage.astext(show_meta_data=False))
            stages.append({'kind': kind, 'begin': begin, 'end_exclusive': end,
                           'input_node_ids': [graph.ids.get(n) for n in inputs],
                           'output_node_ids': [graph.ids.get(n) for n in outputs],
                           'output_sources': [{'frontier_index': i,
                               'kind': 'live_input' if node in inputs else 'executor_output',
                               'live_input_index': inputs.index(node) if node in inputs else None,
                               'graph_output_index': i if node not in inputs else None}
                               for i, node in enumerate(outputs)]})
    write_new_json(output / 'PREPARATION_STATUS.json', {
        'status': 'STATIC_SPLIT_PREPARATION_ONLY', 'source_qir': str(source), 'source_sha256': sha256(source),
        'canonical_qir_sha256': sha256(output / 'canonical_qir.json'),
        'additional_quantize_calls': 0, 'common_pass': 'EliminateCommonSubexpr then InferType',
        'units': 21, 'unit_recomposition_structurally_equal': True,
        'integrated_recomposition_structurally_equal': True, 'stages': stages,
        'original_to_common_pass_numerical_qualification': False,
        'stage02_qualified': False, 'board_accessed': False, 'accuracy_measured': False,
        'remaining': ['synthetic/real full-output qualification', 'packing and adapters',
                      'stage01 certificate', 'official image subset and accuracy'],
    })
    write_new_json(output / 'artifact_hashes.json', {str(p.relative_to(output)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                                                   for p in sorted(output.rglob('*')) if p.is_file()})
    return {'units': 21, 'structural_recomposition': True,
            'integrated_boundary_arities': [len(graph.frontier(i)) for i in (0, 3, 18, 21)],
            'multitensor_boundaries': [b['before_unit'] for b in boundaries if b['tensor_count'] > 1],
            'stage02_qualified': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--qir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.qir, args.output), indent=2))
