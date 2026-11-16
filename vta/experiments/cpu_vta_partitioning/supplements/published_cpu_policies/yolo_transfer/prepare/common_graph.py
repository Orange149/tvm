"""Import local Darknet assets once and freeze a common quantized YOLO416 DAG."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time

WEIGHTS_SHA256='dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c'

def file_hash(path):return sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def prepare(output,*,cfg,weights,darknet_lib,image):
    import numpy as np
    import tvm
    from tvm import relay
    from tvm.relay.testing import darknet
    from tvm.relay.testing.darknet import __darknetffi__
    output=Path(output).resolve();assets={k:Path(v).resolve(strict=True)
        for k,v in dict(cfg=cfg,weights=weights,darknet_lib=darknet_lib,image=image).items()}
    if file_hash(assets['weights'])!=WEIGHTS_SHA256:raise ValueError('YOLO weights identity differs')
    before={k:file_hash(p) for k,p in assets.items()};started=time.monotonic()
    if output.exists():raise FileExistsError('graph preparation is immutable; use a new output directory')
    output.mkdir(parents=True)
    net=__darknetffi__.dlopen(str(assets['darknet_lib'])).load_network(
        str(assets['cfg']).encode(),str(assets['weights']).encode(),0)
    shape=(1,int(net.c),int(net.h),int(net.w))
    if shape!=(1,3,416,416):raise ValueError('this transfer batch fixes the standard 416x416 workload')
    original,params=relay.frontend.from_darknet(net,dtype='float32',shape=shape)
    original=relay.transform.InferType()(original)
    (output/'fp32_source.json').write_text(tvm.ir.save_json(original))
    (output/'params.bin').write_bytes(relay.save_param_dict(params))
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=23.0,skip_conv_layers=[0],store_lowbit_output=True,
                round_for_shift=True) as qconfig:
            (output/'qconfig.json').write_text(tvm.ir.save_json(qconfig))
            quantized=relay.quantize.quantize(original,params=params)
    canonical=relay.transform.InferType()(relay.transform.EliminateCommonSubexpr()(quantized))
    if [p.name_hint for p in canonical['main'].params]!=['data']:
        raise ValueError('all parameters must be bound before graph ownership')
    (output/'canonical_qir.json').write_text(tvm.ir.save_json(canonical))
    (output/'canonical_qir.txt').write_text(canonical.astext(show_meta_data=False))
    data=np.ascontiguousarray(darknet.load_image(str(assets['image']),416,416)[None],dtype='float32')
    if data.shape!=shape or not np.isfinite(data).all():raise ValueError('invalid fixed input')
    np.save(output/'input0.npy',data);data.tofile(output/'input0.bin')
    calls=[]
    def describe_type(typ):
        if isinstance(typ,relay.TensorType):return {'shape':[int(x) for x in typ.shape],'dtype':str(typ.dtype)}
        if isinstance(typ,relay.TupleType):return {'fields':[describe_type(t) for t in typ.fields]}
        raise ValueError('unsupported graph value type: '+str(typ))
    def collect(node):
        if isinstance(node,relay.Call):
            calls.append({'index':len(calls),'op':node.op.name,'type':describe_type(node.checked_type),
                'attrs':str(node.attrs) if node.attrs else None})
    relay.analysis.post_order_visit(canonical['main'].body,collect)
    counts=dict(Counter(r['op'] for r in calls))
    if counts.get('nn.conv2d')!=13:raise ValueError('expected the full 13-convolution YOLOv3-tiny')
    if any(file_hash(p)!=before[k] for k,p in assets.items()):raise ValueError('asset changed during import')
    write(output/'calls.json',calls)
    write(output/'GRAPH_STATUS.json',{'status':'COMMON_GRAPH_PREPARED_LOCAL','input_shape':list(shape),
        'output_type':describe_type(canonical['main'].body.checked_type),'operator_counts':counts,
        'quantize_calls':1,'global_scale':23.0,'skip_conv_layers':[0],
        'tvm_version':tvm.__version__,'host_tvm_library':str(tvm._ffi.base._LIB._name),
        'source_assets':{k:{'path':str(p),'sha256':before[k]} for k,p in assets.items()},
        'elapsed_seconds':time.monotonic()-started,'canonical_sha256':file_hash(output/'canonical_qir.json'),
        'input_sha256':file_hash(output/'input0.bin'),'board_accessed':False,
        'ARM_output_correctness_verified':False,'minimum_units_defined':False,
        'source_quantization':'same full-graph global-scale23 policy used by the historical YOLO frontend; no per-stage requantization',
        'performance_costs_imported':False})
    return output
