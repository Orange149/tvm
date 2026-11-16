"""Compare semantics and executable bindings, not JSON Map traversal order."""
from copy import deepcopy
from c3_tps_sync_release_local import read,sha,require


def comparable_binding(value):
    result=deepcopy(value)
    for row in result['functions'].values():
        # Preserve the digest as evidence in its original artifact. This one
        # diagnostic digest includes transient compiler-buffer metadata. The
        # entire normalized LLVM is compared independently below.
        row['initializer_contract'].pop('llvm_sha256')
    return result


def compare(old,new):
    import tvm,vta
    from c3_official_window_baseline import normalize_llvm
    proof=[]
    for variant in ('TH','P'):
        a,b=old/variant,new/variant
        first=tvm.ir.load_json((a/'codegen_input.json').read_text())
        second=tvm.ir.load_json((b/'codegen_input.json').read_text())
        require(tvm.ir.structural_equal(first,second,map_free_vars=False),
                'EXECUTABLE_IR_CHANGED:'+variant)
        for name in ('graph.json','params.bin','CPU_manifest.json'):
            require(sha(a/name)==sha(b/name),'GRAPH_PARAMETERS_OR_CPU_CHANGED:'+variant+':'+name)
        require(normalize_llvm((a/'module.ll').read_text())==normalize_llvm((b/'module.ll').read_text()),
                'EXECUTABLE_LLVM_CHANGED:'+variant)
        require(comparable_binding(read(a/'LLVM_initializer_binding.json'))==
                comparable_binding(read(b/'LLVM_initializer_binding.json')),
                'INITIALIZER_OR_CACHE_SHARING_CHANGED:'+variant)
        proof.append({'variant':variant,'IR_structurally_equal_without_free_variable_remapping':True,
            'IR_serialized_bytes_equal':sha(a/'codegen_input.json')==sha(b/'codegen_input.json'),
            'normalized_entire_LLVM_equal':True,'all_initializer_and_cache_bindings_equal':True,
            'graph_parameters_and_CPU_manifest_identical':True,
            'ignored_diagnostic_only_field':'per-function initializer_contract.llvm_sha256',
            'normalization':'existing frozen normalize_llvm; no instruction, address, loop or handle normalization added'})
    return proof
