"""Exact named tensor equality independent of serialized Map entry order."""
import numpy as np
from c3_tps_sync_release_local import sha,require


def compare(left,right):
    from tvm import relay
    first=relay.load_param_dict(left.read_bytes());second=relay.load_param_dict(right.read_bytes())
    require(set(first)==set(second),'TRAINED_PARAMETER_NAMES_CHANGED')
    tensors=[]
    from c3_official_window_reference import array_sha
    for name in sorted(first):
        a,b=first[name].numpy(),second[name].numpy()
        require(a.shape==b.shape and a.dtype==b.dtype,'TRAINED_PARAMETER_ABI_CHANGED:'+name)
        require(np.array_equal(a,b) and a.tobytes()==b.tobytes(),'TRAINED_PARAMETER_VALUE_CHANGED:'+name)
        tensors.append({'name':str(name),'shape':list(a.shape),'dtype':str(a.dtype),
            'elements':int(a.size),'all_elements_and_bytes_equal':True,'contents_sha256':array_sha(a)})
    return {'status':'ALL_NAMED_MODEL_PARAMETERS_EXACTLY_EQUAL','tensors':tensors,
            'serialized_sha256':[sha(left),sha(right)],'serialized_bytes_equal':left.read_bytes()==right.read_bytes(),
            'parameter_order_has_no_execution_semantics':True}
