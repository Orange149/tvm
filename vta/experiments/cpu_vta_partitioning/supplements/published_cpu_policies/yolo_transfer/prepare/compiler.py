"""SDK worker for the frozen YOLO DAG; no transport or performance inputs."""
import argparse
import os
from pathlib import Path
import time

from cpu_vta_partitioning.common import digest, inventory, read, sha256, verify_inventory, write


def program_ir(graph, spec):
    """Independent full graph, complete frontier probe, or a typed interval."""
    import tvm
    from tvm import relay
    from .units import outputs

    if spec['kind'] == 'interval':
        mod, incoming, outgoing = graph.stage(spec['begin'], spec['end'])
        n = len(graph.units)
        graph.recompose([(i, i + 1) for i in range(spec['begin'])]
                        + [(spec['begin'], spec['end'])]
                        + [(i, i + 1) for i in range(spec['end'], n)])
    else:
        incoming = list(graph.mod['main'].params)
        if spec['kind'] == 'complete_graph':
            outgoing = outputs(graph.mod['main'].body)
        elif spec['kind'] == 'complete_frontier_probe':
            outgoing = list(dict.fromkeys(node for boundary in range(len(graph.units) + 1)
                                          for node in graph.frontier(boundary)))
        else:
            raise ValueError('unknown program kind')
        mod = relay.transform.InferType()(tvm.IRModule.from_expr(
            relay.Function(incoming, relay.Tuple(outgoing))))
    return mod, incoming, outgoing


def packed_stage(mod, env):
    """Same explicit typed frontier bridge as prepare/compiler.py, using this DAG mutator."""
    import tvm
    from tvm import relay
    from .packing import pack_frontiers
    from .units import Replace

    function = mod['main']
    replacements = {p: relay.Call(tvm.ir.Op.get('annotation.bitpack_start'), [p])
                    for p in function.params}
    body = Replace(replacements).visit(function.body)
    body = relay.Tuple([relay.Call(tvm.ir.Op.get('annotation.bitpack_end'), [x])
                        for x in body.fields])
    wrapped = relay.transform.InferType()(tvm.IRModule.from_expr(
        relay.Function(list(function.params), body)))
    packed = relay.transform.InferType()(tvm.IRModule.from_expr(pack_frontiers(
        wrapped['main'], env.BATCH, env.BLOCK_OUT, env.WGT_WIDTH)))
    if not tvm.ir.structural_equal(function.checked_type, packed['main'].checked_type):
        raise ValueError('VTA packing changed frontier types')
    return packed


def compile_program(workspace, name):
    import tvm
    from tvm import autotvm, relay
    from tvm.contrib import cc
    import vta
    from .units import UnitGraph, outputs

    workspace = Path(workspace).resolve()
    identity = read(workspace / 'plans/compiler_identity.json')
    verify_inventory(workspace, identity['files'])
    build_identity = digest(identity)
    specs = read(workspace / 'plans/programs.json')
    matches = [s for s in specs if s['name'] == name]
    if len(matches) != 1:
        raise ValueError('program name not unique in frozen plan')
    spec = matches[0]
    output = workspace / 'programs' / name
    record_path = output / 'record.json'
    if record_path.exists():
        record = read(record_path)
        if record['build_identity'] != build_identity or record['spec'] != spec:
            raise ValueError('cached program build identity differs')
        verify_inventory(output, record['files'])
        return record
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    graph = UnitGraph(tvm.ir.load_json((workspace / 'sources/canonical_qir.json').read_text()))
    original, incoming, outgoing = program_ir(graph, spec)
    env = vta.get_env()
    mod = packed_stage(original, env) if spec['device'] == 'vta' else original
    (output / 'lowering_qir.json').write_text(tvm.ir.save_json(mod))
    target = tvm.target.Target(env.target if spec['device'] == 'vta' else env.target_vta_cpu,
                               host=env.target_host)
    context = vta.build_config if spec['device'] == 'vta' else tvm.transform.PassContext
    with autotvm.apply_history_best([str(workspace / 'sources/tuning_history.log')]), context(
            opt_level=3, disabled_pass={'AlterOpLayout', 'tir.CommonSubexprElimTIR'}):
        factory = relay.build(mod, target=target)
    (output / 'graph.json').write_text(factory.get_graph_json())
    (output / 'params.bin').write_bytes(relay.save_param_dict(factory.get_params()))
    sysroot = os.environ['SDKTARGETSYSROOT']
    factory.get_lib().export_library(str(output / 'module.so'),
        fcompile=cc.cross_compiler('aarch64-xilinx-linux-g++'), options=[
            f'--sysroot={sysroot}', f'-Wl,-rpath-link,{sysroot}/lib',
            f'-Wl,-rpath-link,{sysroot}/usr/lib', f'-L{sysroot}/lib', f'-L{sysroot}/usr/lib'])
    def typed(node):
        return {'dtype': str(node.checked_type.dtype), 'shape': list(map(int, node.checked_type.shape))}
    terminal = outputs(graph.mod['main'].body)
    record = {'spec': spec, 'build_identity': build_identity,
              'canonical_sha256': sha256(workspace / 'sources/canonical_qir.json'),
              'target': str(target), 'input_names': [p.name_hint for p in original['main'].params],
              'input_node_ids': [graph.ids[n] for n in incoming],
              'output_node_ids': [graph.ids[n] for n in outgoing],
              'input_types': [typed(n) for n in incoming], 'output_types': [typed(n) for n in outgoing],
              'terminal_output_indices': [outgoing.index(n) if n in outgoing else None for n in terminal],
              'extra_quantize_calls': 0, 'recomposition_exact': True,
              'compile_seconds': time.monotonic() - started,
              'files': inventory(output, ['record.json'])}
    write(record_path, record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--program', required=True)
    args = parser.parse_args()
    record = compile_program(args.workspace, args.program)
    print(args.program, 'compiled', round(record['compile_seconds'], 2), flush=True)


if __name__ == '__main__':
    main()
