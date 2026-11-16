"""Read-only screening of sealed records. No TVM build, simulator, RPC or board access.

The restricted AST walker evaluates integer loop bounds and DMA descriptors from
two frozen TIR scripts. It does not execute Python/TIR or prove hardware safety.
"""
import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOLD = HERE/'report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout'
STRUCT = HOLD/'20261016_c3_yolo224_sync_structure_readonly03'
PROFILE = HOLD/'20261008_c3_yolo224_incremental_profile_run01'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def integer(n, env):
    if isinstance(n, ast.Constant) and isinstance(n.value, int):
        return n.value
    if isinstance(n, ast.Name):
        return env[n.id]
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        return -integer(n.operand, env)
    if isinstance(n, ast.BinOp):
        a, b = integer(n.left, env), integer(n.right, env)
        if isinstance(n.op, ast.Add): return a+b
        if isinstance(n.op, ast.Sub): return a-b
        if isinstance(n.op, ast.Mult): return a*b
    if isinstance(n, ast.Call) and ast.unparse(n.func) in ('T.max', 'T.min'):
        return (max if ast.unparse(n.func) == 'T.max' else min)(integer(x, env) for x in n.args)
    raise ValueError(f'unsupported integer expression: {ast.dump(n)}')


def union_size(intervals):
    total, end = 0, -1
    for a, b in sorted(intervals):
        total += max(0, b-max(a, end))
        end = max(end, b)
    return total


def inspect_tir(path):
    loops, requests, steps, scopes = [], [], Counter(), Counter()
    active = []

    def walk(body, env):
        for n in body:
            if isinstance(n, ast.For):
                kind = ast.unparse(n.iter.func)
                if kind not in ('range', 'T.grid'): raise ValueError(kind)
                extents = [integer(x, env) for x in n.iter.args]
                names = [n.target.id] if isinstance(n.target, ast.Name) else [x.id for x in n.target.elts]
                if len(names) != len(extents): raise ValueError('loop arity')
                for values in itertools.product(*(range(x) for x in extents)):
                    walk(n.body, dict(env, **dict(zip(names, values))))
            elif isinstance(n, ast.With):
                ctx = n.items[0].context_expr
                is_kernel = any(isinstance(x, ast.Constant) and x.value == 'coproc_uop_scope' for x in ctx.args)
                if is_kernel:
                    if active or loops: raise ValueError('nested or unclosed kernel')
                    active.append(set())
                walk(n.body, env)
                if is_kernel:
                    kinds = active.pop()
                    if len(kinds) != 1 or loops: raise ValueError('kernel mode or loop balance')
                    scopes[next(iter(kinds))] += 1
            elif isinstance(n, ast.Assign):
                if ast.unparse(n.value) != 'T.int32()': raise ValueError('unhandled assignment')
            elif isinstance(n, ast.Expr) and isinstance(n.value, ast.Call):
                call = n.value
                fn = ast.unparse(call.func)
                if fn == 'T.call_extern':
                    name = call.args[1].value
                    if name == 'VTAUopLoopBegin': loops.append(integer(call.args[2], env))
                    elif name == 'VTAUopLoopEnd': loops.pop()
                    elif name == 'VTALoadBuffer2D':
                        a = call.args
                        offset, xs, ys, stride = [integer(x, env) for x in a[4:8]]
                        domain = integer(a[13], env)
                        unit = {1: 256, 2: 16, 3: 64}[domain]
                        requests.append({'op': 'LOAD', 'tensor': ast.unparse(a[3]), 'domain': domain,
                            'offset': offset, 'x': xs, 'y': ys, 'stride': stride,
                            'pads': [integer(x, env) for x in a[8:12]],
                            'sram': integer(a[12], env), 'unit': unit})
                    elif name == 'VTAStoreBuffer2D':
                        a = call.args
                        requests.append({'op': 'STORE', 'tensor': ast.unparse(a[5]), 'domain': integer(a[4], env),
                            'offset': integer(a[6], env), 'x': integer(a[7], env), 'y': integer(a[8], env),
                            'stride': integer(a[9], env), 'sram': integer(a[3], env), 'pads': [], 'unit': 16})
                    else: raise ValueError(f'unsupported extern {name}')
                elif fn == 'T.tir.vta.uop_push':
                    mode, reset = [integer(x, env) for x in call.args[:2]]
                    kind = 'ALU' if mode == 1 else ('RESET' if reset else 'MAC')
                    if not active: raise ValueError('uop without scope')
                    active[-1].add(kind)
                    steps[kind] += math.prod(loops)
                elif fn not in ('T.func_attr', 'T.attr', 'T.tir.vta.coproc_dep_push',
                                 'T.tir.vta.coproc_dep_pop', 'T.tir.vta.coproc_sync'):
                    raise ValueError(f'unhandled call {fn}')
            else: raise ValueError(f'unhandled statement {type(n).__name__}')

    tree = ast.parse(path.read_text())
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef): raise ValueError('function count')
    walk(tree.body[0].body, {})
    groups = defaultdict(list)
    for r in requests: groups[(r['op'], r['tensor'], r['domain'])].append(r)
    traffic = []
    for key, rows in groups.items():
        intervals = [((r['offset']+i*r['stride'])*r['unit'],
                      (r['offset']+i*r['stride']+r['x'])*r['unit']) for r in rows for i in range(r['y'])]
        payload = sum(r['x']*r['y']*r['unit'] for r in rows)
        unique = union_size(intervals)
        traffic.append({'operation': key[0], 'tensor': key[1], 'domain': key[2],
                        'requests': len(rows), 'payload_bytes': payload, 'unique_bytes': unique,
                        'average_address_multiplicity': payload/unique})
    return {'steps': dict(steps), 'dynamic_TIR_kernel_calls': dict(scopes), 'traffic': traffic,
            'tensor_requests': requests, 'scope': 'TIR descriptors only; no runtime or timing proof'}


def main(out):
    out.mkdir(exist_ok=False)
    sources = {str(Path(__file__).resolve()): sha(__file__)}

    def read(path):
        sources[str(path.resolve())] = sha(path)
        return json.loads(path.read_text())

    artifacts = read(PROFILE/'artifact_hashes.json')['artifacts']
    for name, expected in artifacts.items():
        if sha(PROFILE/name) != expected: raise ValueError(f'profile identity: {name}')
    inventory = read(STRUCT/'inventory.json')
    times = {r['node']: r for r in read(PROFILE/'node_times.json')['nodes']}
    rows = []
    for item in inventory:
        if item['variant'] != 'G2': continue
        if sha(item['tir']) != item['sha256']: raise ValueError('TIR identity')
        sources[item['tir']] = item['sha256']
        obj = read(STRUCT/'objects'/f"{item['object_id']}.json")
        w = item['abi'][1]['shape']; y = item['abi'][3]['shape']
        inp = item['abi'][0]['shape']
        ci, co, kh, kw = w[1]*w[4], w[0]*w[5], w[2], w[3]
        row = {'node': item['node'], 'object_id': item['object_id'], 'symbol': item['symbol'],
               'ci': ci, 'co': co, 'h': y[2], 'w': y[3], 'kh': kh, 'kw': kw,
               'input_bytes': math.prod(inp), 'weight_bytes': math.prod(w), 'output_bytes': math.prod(y),
               'expanded_steps': obj['counts']['expanded_uop_steps'],
               'MAC_same_address_waits': obj['MAC_categories'].get('MAC_INPUT_OVERWRITE_PROTECTION',0),
               'MAC_disjoint_candidates': obj['MAC_categories'].get('ADDRESS_DISJOINT_PREFIX_NEEDS_FULL_CERTIFICATE',0),
               'force_serial_requested': obj['force_serial_requested'],
               'historical_host_call_ms': times[item['node']]['mean_host_call_ms'],
               'historical_time_variant': 'G0',
               'structural_variant': 'G2',
               'historical_time_scope': times[item['node']]['completion'],
               'same_geometry_has_separately_qualified_TPS_R': item['node'] == 24,
               'new_optimization_candidate_qualified': False}
        rows.append(row)
        if item['node'] in (34, 46):
            # inventory ends in .tir.json: remove only .json.
            tir = Path(item['tir']).with_suffix('')
            sources[str(tir.resolve())] = sha(tir)
            g0 = tir.parent.parent/'G0'/tir.name
            sources[str(g0.resolve())] = sha(g0)
            if g0.read_bytes() != tir.read_bytes(): raise ValueError('G0/G2 candidate TIR differs')
            row['G0_G2_TIR_text_exactly_equal'] = True
            detail = inspect_tir(tir)
            if sum(detail['steps'].values()) != row['expanded_steps']: raise ValueError('independent step mismatch')
            if detail['steps']['MAC']*256 != y[2]*y[3]*co*ci*kh*kw: raise ValueError('MAC mismatch')
            for tr in detail['traffic']:
                expected = {'p0.data': row['input_bytes'], 'p1.data': row['weight_bytes'],
                            'p2.data': co*4, 'T_cast.data': row['output_bytes']}[tr['tensor']]
                if tr['unique_bytes'] != expected: raise ValueError('tensor range coverage')
            (out/f'node_{item["node"]}_descriptors.json').write_text(json.dumps(detail,indent=2)+'\n')
            row['detail'] = {k:v for k,v in detail.items() if k!='tensor_requests'}
    (out/'screen.json').write_text(json.dumps(rows,indent=2)+'\n')
    fields = [k for k in rows[0] if k != 'detail']
    with (out/'screen.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    (out/'source_sha256.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(json.dumps([r for r in rows if 'detail' in r],indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path); main(p.parse_args().output)
