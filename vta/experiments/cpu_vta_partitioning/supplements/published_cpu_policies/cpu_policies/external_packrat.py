"""Run two original author methods from an external, hash-pinned checkout.

No original Packrat implementation is redistributed here. Only the two method
bodies are taken from the explicitly supplied file at execution time. Imports,
network-loading constructors, serving processes and benchmarks are not run.
This is a method-level reference check, not a full TorchServe reproduction.
"""
import ast
import hashlib
from pathlib import Path

EXPECTED_SOURCE_SHA256 = "9a52955b1cc290a05a6165f16c06b248cc6e9dbc221179fcb45ce3680ee49666"
COMMIT = "18dc6c77eff5d5b402b2b895d376f0977d5a0dd0"


def run_original_methods(source, latencies, total_threads, total_batch):
    path=Path(source)
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=EXPECTED_SOURCE_SHA256:
        raise ValueError("external Packrat source does not match the reviewed commit")
    parsed=ast.parse(data,filename=str(path))
    original=next(n for n in parsed.body if isinstance(n,ast.ClassDef) and n.name=='Optimizer')
    methods=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in ('min_latency','solution')]
    if len(methods)!=2:raise ValueError("author optimizer method coverage differs")
    tree=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(
        name='ExternalOptimizer',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[]))
    namespace={}
    exec(compile(tree,str(path),'exec'),namespace)
    cls=namespace['ExternalOptimizer'];o=cls.__new__(cls)
    o.latencies={'fixture':latencies};o.opt={'fixture':o.min_latency('fixture')}
    result=[];error=None
    try:o.solution(total_threads,total_batch,'fixture',result)
    except Exception as exc:error=repr(exc)
    return {'source_commit':COMMIT,'source_sha256':EXPECTED_SOURCE_SHA256,
            'scope':'unchanged_author_method_bodies_with_offline_injected_profile',
            'predicted_latency_ms':o.opt['fixture'][total_threads][total_batch],
            'allocations':result,'reconstruction_error':error,
            'used_threads':sum(x[0] for x in result),'covered_batch':sum(x[1] for x in result)}

