"""Freeze the pretrained Gluon ResNet50-v2 common graph, without compiling it."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[4]
DEFAULT_WEIGHTS = Path('/home/orange/.mxnet/models/resnet50_v2-ecdde353.params')
WEIGHTS_SHA256 = 'e00acf01c0ce0767179e2811e18e46176c40815b55d3b6820694614ed62eac71'
DEFAULT_INPUT = ROOT / 'artifacts/run_scope_tail_comparison_20261112_01/resnet18/capsule/inputs/000.bin'
INPUT_SHA256 = '78f9e1dede477e9f3acba0d4d92aae86daeb9a7d1f1ba1e95f373db41207b222'


def file_hash(path):
    h = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _inventory(root):
    return {str(p.relative_to(root)): {'bytes': p.stat().st_size, 'sha256': file_hash(p)}
            for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'graph_manifest.json'}


def validate_graph(output):
    output = Path(output)
    manifest = json.loads((output / 'graph_manifest.json').read_text())
    for name, record in manifest['files'].items():
        path = output / name
        if path.stat().st_size != record['bytes'] or file_hash(path) != record['sha256']:
            raise ValueError('common graph file identity differs: ' + name)
    return manifest


def prepare_graph(output, *, weights=DEFAULT_WEIGHTS, input_path=DEFAULT_INPUT):
    """Import/quantize once; no network forward pass, Relay build or board access."""
    import numpy as np
    import mxnet as mx
    from mxnet.gluon.model_zoo import vision
    import tvm
    from tvm import relay

    output = Path(output).resolve()
    weights, input_path = Path(weights).resolve(strict=True), Path(input_path).resolve(strict=True)
    if file_hash(weights) != WEIGHTS_SHA256:
        raise ValueError('pretrained Gluon ResNet50-v2 weights differ')
    if file_hash(input_path) != INPUT_SHA256:
        raise ValueError('fixed input0 differs from the audited preprocessing contract')
    if output.exists():
        manifest = validate_graph(output)
        if manifest['weights_sha256'] != WEIGHTS_SHA256 or manifest['input_sha256'] != INPUT_SHA256:
            raise ValueError('existing graph belongs to a different input/model')
        return manifest
    output.mkdir(parents=True)
    started = time.monotonic()
    # A verified local cache is mandatory, so pretrained=True cannot silently
    # replace this model with a downloaded or randomly initialized variant.
    model = vision.get_model('resnet50_v2', pretrained=True, root=str(weights.parent), ctx=mx.cpu())
    source, params = relay.frontend.from_mxnet(model, {'data': (1, 3, 224, 224)})
    source = relay.transform.InferType()(source)
    (output / 'fp32_source.json').write_text(tvm.ir.save_json(source))
    (output / 'fp32_source.txt').write_text(source.astext(show_meta_data=False))
    (output / 'fp32_params.bin').write_bytes(relay.save_param_dict(params))
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]) as config:
            (output / 'qconfig.json').write_text(tvm.ir.save_json(config))
            quantized = relay.quantize.quantize(source, params=params)
    canonical = relay.transform.InferType()(relay.transform.EliminateCommonSubexpr()(quantized))
    if [p.name_hint for p in canonical['main'].params] != ['data']:
        raise ValueError('quantized model has unbound parameters')
    (output / 'canonical_qir.json').write_text(tvm.ir.save_json(canonical))
    (output / 'canonical_qir.txt').write_text(canonical.astext(show_meta_data=False))
    shutil.copy2(input_path, output / 'input0.bin')
    data = np.fromfile(output / 'input0.bin', dtype='float32').reshape((1, 3, 224, 224))
    if not np.isfinite(data).all():
        raise ValueError('fixed input contains nonfinite values')
    np.save(output / 'input0.npy', data, allow_pickle=False)
    calls = []
    def describe_type(typ):
        if isinstance(typ, relay.TensorType):
            return {'dtype': str(typ.dtype), 'shape': [int(x) for x in typ.shape]}
        if isinstance(typ, relay.TupleType):
            return {'fields': [describe_type(x) for x in typ.fields]}
        raise ValueError('unsupported common graph value type')
    def collect(node):
        if isinstance(node, relay.Call):
            if not isinstance(node.op, tvm.ir.Op):
                raise ValueError('common graph must contain primitive operator calls')
            calls.append({'index': len(calls), 'op': node.op.name,
                          'type': describe_type(node.checked_type),
                          'attrs': str(node.attrs) if node.attrs else None})
    relay.analysis.post_order_visit(canonical['main'].body, collect)
    counts = dict(Counter(row['op'] for row in calls))
    if counts.get('nn.conv2d') != 53:
        raise ValueError('expected Gluon ResNet50-v2 53 convolutions including projections')
    if file_hash(weights) != WEIGHTS_SHA256 or file_hash(input_path) != INPUT_SHA256:
        raise ValueError('model or input changed during graph preparation')
    write(output / 'calls.json', calls)
    status = {'status': 'COMMON_GRAPH_PREPARED_LOCAL', 'model': 'MXNet Gluon pretrained resnet50_v2',
              'input_shape': [1, 3, 224, 224], 'output_type': describe_type(canonical['main'].body.checked_type),
              'operator_counts': counts, 'quantize_calls': 1, 'global_scale': 8.0,
              'skip_conv_layers': [0], 'common_pass': 'EliminateCommonSubexpr then InferType',
              'tvm_version': tvm.__version__, 'mxnet_version': mx.__version__,
              'host_tvm_library': str(tvm._ffi.base._LIB._name),
              'weights_path': str(weights), 'weights_sha256': WEIGHTS_SHA256,
              'input_source': str(input_path), 'input_sha256': INPUT_SHA256,
              'canonical_sha256': file_hash(output / 'canonical_qir.json'),
              'elapsed_seconds': time.monotonic() - started,
              'board_accessed': False, 'network_inference_calls': 0,
              'ARM_output_correctness_verified': False, 'minimum_units_defined': False,
              'source_distinction': 'Gluon BottleneckV2: downsampling stride on 3x3 conv2; not relay.testing.resnet',
              'performance_costs_imported': False}
    write(output / 'GRAPH_STATUS.json', status)
    manifest = {'model': 'resnet50_v2', 'weights_sha256': WEIGHTS_SHA256,
                'input_sha256': INPUT_SHA256, 'canonical_sha256': status['canonical_sha256'],
                'graph_preparation_source_sha256': file_hash(__file__), 'files': _inventory(output)}
    write(output / 'graph_manifest.json', manifest)
    return manifest
