#!/usr/bin/env python3
"""Build current-qIR component programs before freezing the C1 support."""
import argparse
import json
from pathlib import Path
import tvm
from tvm import autotvm, relay
import vta
from vta.top.graphpack import graph_pack
from thesis_canonical_graph_split import UnitGraph, Replace
from thesis_supplement_run import RunStore, sha256, write_new_json
from deploy_classification_stage_pipeline_native import export_stage_lib


def packed_stage(mod, env):
    function = mod['main']
    replacement = {p: relay.Call(tvm.ir.Op.get('annotation.bitpack_start'), [p]) for p in function.params}
    body = Replace(replacement).visit(function.body)
    body = relay.Tuple([relay.Call(tvm.ir.Op.get('annotation.bitpack_end'), [field]) for field in body.fields])
    wrapped = relay.transform.InferType()(tvm.IRModule.from_expr(relay.Function(list(function.params), body)))
    packed = relay.transform.InferType()(tvm.IRModule.from_expr(graph_pack(
        wrapped['main'], env.BATCH, env.BLOCK_OUT, env.WGT_WIDTH,
        start_name=None, stop_name=None, boundary_bridge=True)))
    if not tvm.ir.structural_equal(mod['main'].checked_type, packed['main'].checked_type):
        raise ValueError('Packing changed complete live frontier types')
    return packed


def build(store, output_name, tune_log=None):
    r = store.directory
    source = Path(store.contract['canonical_source'])
    if sha256(source) != store.contract['canonical_qir_sha256']:
        raise ValueError('Canonical qIR changed')
    if store.contract.get('baseline_history_sha256') and (not tune_log or sha256(tune_log) != store.contract['baseline_history_sha256']):
        raise ValueError('Shared baseline Tile history differs from the frozen contract')
    graph = UnitGraph(tvm.ir.load_json(source.read_text()))
    env = vta.get_env()
    cpu = tvm.target.Target(env.target_vta_cpu, host=env.target_host)
    ext = tvm.target.Target(env.target, host=env.target_host)
    # Current completion qualification is for one persistent VTA worker queue.
    # Multi-island TLS queue handoff is outside its domain, excluded before labels.
    topology_ranges = [(3, 10), (3, 13), (3, 16), (3, 18), (5, 13), (5, 16), (5, 18), (8, 18)]
    topologies = [{'topology_id': f'vta_{begin:02}_{end-1:02}',
                   'ranges': [[0, begin, 'cpu'], [begin, end, 'vta'], [end, 21, 'cpu']],
                   'reasonable_a': begin == 3 and end == 18}
                  for begin, end in topology_ranges]
    specs = sorted({(begin, end, device) for topo in topologies for begin, end, device in topo['ranges']})
    package = r / 'raw' / output_name
    package.mkdir()
    records = []
    for begin, end, device in specs:
        mod, inputs, outputs = graph.stage(begin, end)
        original = mod
        if device == 'vta':
            if any(str(p.checked_type.dtype) != 'int8' for p in mod['main'].params):
                raise ValueError('Unsupported non-int8 packed entry')
            mod = packed_stage(mod, env)
        name = f'{device}_{begin:02}_{end-1:02}'
        directory = package / name
        directory.mkdir()
        (directory / 'lowering_qir.json').write_text(tvm.ir.save_json(mod))
        with autotvm.apply_history_best([tune_log] if tune_log else []), vta.build_config(opt_level=3,
                disabled_pass={'AlterOpLayout', 'tir.CommonSubexprElimTIR'}):
            factory = relay.build(mod, target=ext if device == 'vta' else cpu)
        (directory / 'graph.json').write_text(factory.get_graph_json())
        (directory / 'params.bin').write_bytes(relay.save_param_dict(factory.get_params()))
        export_stage_lib(factory.get_lib(), directory / 'module.so')
        records.append({'name': name, 'begin': begin, 'end_exclusive': end, 'device': device,
            'input_names': [p.name_hint for p in original['main'].params],
            'input_node_ids': [graph.ids[n] for n in inputs], 'output_node_ids': [graph.ids[n] for n in outputs],
            'output_sources': [{'kind': 'live_input' if n in inputs else 'executor_output',
                                'live_input_index': inputs.index(n) if n in inputs else None} for n in outputs],
            'files': {filename: {'sha256': sha256(directory / filename), 'bytes': (directory / filename).stat().st_size}
                      for filename in ('graph.json', 'module.so', 'params.bin', 'lowering_qir.json')}})
        print(name + ': compiled, complete frontier preserved, extra quantize=0', flush=True)
    write_new_json(package / 'component_manifest.json', {'canonical_qir_sha256': sha256(source),
        'programs': records, 'topologies': topologies, 'extra_quantize_calls': 0,
        'excluded_island_categories': {'2': 'cross-worker TLS CommandQueue handoff not completion-qualified',
                                       '3': 'cross-worker TLS CommandQueue handoff not completion-qualified'},
        'tune_log_sha256':sha256(tune_log) if tune_log else None,
        'scope': 'single-input C1 preparation; not formal phase04 qualification'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir', type=Path)
    p.add_argument('--output-name', default='c1_components')
    p.add_argument('--tune-log')
    args = p.parse_args()
    if Path(args.output_name).name != args.output_name:
        p.error('output-name must be one directory name')
    build(RunStore(args.run_dir), args.output_name, args.tune_log)
