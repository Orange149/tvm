#!/usr/bin/env python3
"""Build monolithic and baseline board packages from the frozen common qIR."""
import argparse
import json
import os
from pathlib import Path
import shutil
import numpy as np
import tvm
from tvm import autotvm, relay
import vta
from vta.top.graphpack import graph_pack
from thesis_canonical_graph_split import UnitGraph
from thesis_supplement_run import RunStore, sha256, write_new_json
from deploy_classification_stage_pipeline_native import export_stage_lib


def build(run, output_name='board_programs', tune_log=None):
    r = run.directory
    source = Path(run.contract['canonical_source'])
    if sha256(source) != run.contract['canonical_qir_sha256']:
        raise ValueError('Canonical qIR drift')
    graph = UnitGraph(tvm.ir.load_json(source.read_text()))
    env = vta.get_env()
    cpu = tvm.target.Target(env.target_vta_cpu, host=env.target_host)
    ext = tvm.target.Target(env.target, host=env.target_host)
    packed_source = Path(run.contract['preparation_run']) / 'raw/canonical_pack_preparation_01/P1/packed_qir.json'
    if not packed_source.exists():
        candidates = list((Path(run.contract['preparation_run']) / 'raw').glob('*/P1/packed_qir.json'))
        if len(candidates) != 1:
            raise ValueError('Ambiguous packed canonical P1')
        packed_source = candidates[0]
    packed = tvm.ir.load_json(packed_source.read_text())
    packed_contract = json.loads((packed_source.parents[1] / 'contract.json').read_text())
    packed_manifest = json.loads((packed_source.parents[1] / 'artifact_hashes.json').read_text())
    if (packed_contract['canonical_qir_sha256'] != sha256(source) or
            packed_manifest[str(packed_source.relative_to(packed_source.parents[1]))]['sha256'] != sha256(packed_source)):
        raise ValueError('Packed P1 provenance differs from the frozen canonical qIR')
    # Official full-graph packing, using the already quantized graph unchanged.
    whole = relay.transform.InferType()(tvm.IRModule.from_expr(graph_pack(
        graph.mod['main'], env.BATCH, env.BLOCK_OUT, env.WGT_WIDTH,
        start_name='nn.max_pool2d', stop_name='nn.global_avg_pool2d',
        device_annot=True, annot_start_name='nn.conv2d', annot_end_name='annotation.stop_fusion')))
    modules = [('Q_board', 0, 21, whole, {'cpu': cpu, 'ext_dev': ext})]
    for label, begin, end in [('A_stage0', 0, 3), ('A_stage1', 3, 18), ('A_stage2', 18, 21)]:
        mod, inputs, outputs = graph.stage(begin, end)
        if begin == 3:
            if not tvm.ir.structural_equal(mod['main'].checked_type, packed['main'].checked_type):
                raise ValueError('Packed P1 external interface drift')
            mod = packed
        modules.append((label, begin, end, mod, ext if begin == 3 else cpu))
    package = r / 'raw' / output_name
    package.mkdir()
    records = []
    for label, begin, end, mod, target in modules:
        directory = package / label
        directory.mkdir()
        (directory / 'lowering_qir.json').write_text(tvm.ir.save_json(mod))
        with autotvm.apply_history_best([tune_log] if tune_log else []), vta.build_config(
            opt_level=3, disabled_pass={'AlterOpLayout', 'tir.CommonSubexprElimTIR'}):
            factory = relay.build(mod, target=target)
        (directory / 'graph.json').write_text(factory.get_graph_json())
        (directory / 'params.bin').write_bytes(relay.save_param_dict(factory.get_params()))
        export_stage_lib(factory.get_lib(), directory / 'module.so')
        _, inputs, outputs = graph.stage(begin, end)
        records.append({'name': label, 'begin': begin, 'end_exclusive': end,
                        'device': 'vta' if label in ('Q_board', 'A_stage1') else 'cpu',
                        'input_names': ['data'] if label == 'Q_board' else [p.name_hint for p in mod['main'].params],
                        'input_node_ids': [graph.ids[n] for n in inputs],
                        'output_node_ids': [graph.ids[n] for n in outputs],
                        'output_sources': [{'kind': 'live_input' if n in inputs else 'executor_output',
                                            'live_input_index': inputs.index(n) if n in inputs else None} for n in outputs],
                        'files': {name: {'sha256': sha256(directory / name), 'bytes': (directory / name).stat().st_size}
                                  for name in ('graph.json', 'params.bin', 'module.so', 'lowering_qir.json')}})
        print(label + ': board build complete from common qIR; extra quantize=0', flush=True)
    write_new_json(package / 'program_manifest.json', {'canonical_qir_sha256': sha256(source),
        'packed_P1_source': str(packed_source), 'packed_P1_sha256': sha256(packed_source),
        'extra_quantize_calls': 0, 'schedule': 'stock geometry/architecture-compatible cached configurations; all measurements must be new' if tune_log else 'frozen AutoTVM fallback, no old tuning history',
        'tune_log_sha256': sha256(tune_log) if tune_log else None, 'programs': records})
    write_new_json(r / 'analysis' / (output_name + '_build_summary.json'), {'status': 'BOARD_PACKAGES_BUILT_PENDING_NUMERIC_CHECKS',
        'extra_quantize_calls': 0, 'program_manifest_sha256': sha256(package / 'program_manifest.json')})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_dir', type=Path)
    p.add_argument('--output-name', default='board_programs')
    p.add_argument('--tune-log')
    args=p.parse_args()
    build(RunStore(args.run_dir), args.output_name, args.tune_log)
