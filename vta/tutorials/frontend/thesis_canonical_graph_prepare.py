#!/usr/bin/env python3
"""Static preparation of a pretrained whole graph; no board or accuracy qualification."""
import argparse
from collections import Counter
import json
from pathlib import Path
import mxnet as mx
from mxnet.gluon.model_zoo import vision
import tvm
from tvm import relay
from thesis_supplement_run import sha256, write_new_json


WEIGHTS_SHA256 = '844464c704f3c4b74428d005f552eed0dc0c1502a92d3efad950f7234678e999'


def describe(mod):
    calls, counts = [], Counter()
    def visit(node):
        if not isinstance(node, relay.Call) or not isinstance(node.op, tvm.ir.Op):
            return
        counts[node.op.name] += 1
        span = node.span
        result_type = node.checked_type
        calls.append({'call_index': len(calls), 'op': node.op.name,
                      'span': str(span.source_name.name) if span else None,
                      'checked_type': str(result_type),
                      'attrs': str(node.attrs) if node.attrs else None})
    relay.analysis.post_order_visit(mod['main'].body, visit)
    return {'operator_counts': dict(counts), 'calls': calls,
            'free_variables': [var.name_hint for var in relay.analysis.free_vars(mod['main'].body)]}


def prepare(weights, output):
    if sha256(weights) != WEIGHTS_SHA256:
        raise ValueError('Pretrained weight bytes differ from the stage00 inventory')
    output.mkdir(parents=True, exist_ok=False)
    model = vision.get_model('resnet18_v1', pretrained=True, root=str(weights.parent), ctx=mx.cpu())
    source, params = relay.frontend.from_mxnet(model, {'data': (1, 3, 224, 224)})
    source = relay.transform.InferType()(source)
    (output / 'fp32_source.json').open('x').write(tvm.ir.save_json(source))
    (output / 'fp32_source.txt').open('x').write(source.astext(show_meta_data=False))
    (output / 'fp32_params.bin').open('xb').write(relay.save_param_dict(params))
    write_new_json(output / 'fp32_calls.json', describe(source))
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]) as qconfig:
            (output / 'qconfig.json').open('x').write(tvm.ir.save_json(qconfig))
            quantized = relay.quantize.quantize(source, params=params)
    quantized = relay.transform.InferType()(quantized)
    if [var.name_hint for var in quantized['main'].params] != ['data']:
        raise ValueError('Canonical quantized graph still contains unbound parameters')
    if sha256(weights) != WEIGHTS_SHA256:
        raise ValueError('Weights changed during preparation')
    (output / 'once_quantized.json').open('x').write(tvm.ir.save_json(quantized))
    (output / 'once_quantized.txt').open('x').write(quantized.astext(show_meta_data=False))
    write_new_json(output / 'once_quantized_calls.json', describe(quantized))
    write_new_json(output / 'PREPARATION_STATUS.json', {
        'status': 'STATIC_PREPARATION_ONLY', 'model': 'pretrained MXNet ResNet18_v1',
        'weights_path': str(weights), 'weights_sha256': WEIGHTS_SHA256,
        'input_shape': [1, 3, 224, 224], 'quantize_calls': 1,
        'global_scale': 8.0, 'skip_conv_layers': [0], 'tvm_version': tvm.__version__,
        'mxnet_version': mx.__version__,
        'real_images_or_accuracy_used': False, 'board_accessed': False,
        'stage02_gate_passed': False,
        'remaining': ['21-unit ownership and dependency-closed split', 'packing/layout adapters',
                      'structural and full-output qualification', 'official image subset and accuracy',
                      'stage01 two-boot completion certificate'],
    })
    write_new_json(output / 'artifact_hashes.json', {p.name: {'sha256': sha256(p), 'bytes': p.stat().st_size}
                                                   for p in sorted(output.iterdir()) if p.is_file()})
    return {'status': 'STATIC_PREPARATION_ONLY', 'quantize_calls': 1,
            'fp32_operator_counts': describe(source)['operator_counts'],
            'quantized_operator_counts': describe(quantized)['operator_counts']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.weights, args.output), indent=2))
