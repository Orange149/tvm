"""Node34 physical storage gate; no compiled target or device is invoked."""
from math import prod
from c3_tps_aggregation_prepare import require


def tensor_sizes(abi):
    require(len(abi)==4,'FOUR_BUFFER_ABI_REQUIRED')
    sizes={}
    for i,b in enumerate(abi):
        require(b['ordinal']==i and b['kind']=='buffer' and b['scope']=='global' and
                b['dtype'] in ('int8','int32') and b['elem_offset']=='0' and
                b['strides']==[] and all(type(x) is int and x>0 for x in b['shape']),
                'UNSUPPORTED_STORAGE_ABI')
        sizes[i]=prod(b['shape'])*(1 if b['dtype']=='int8' else 4)
    require(sizes=={0:25088,1:4718592,2:4096,3:50176},'NODE34_ABI_SIZE_CHANGED')
    return sizes


def verify_native_storage(native, storage, abi):
    # data_address identifies the actual host DataBuffer, not a DMA address.
    require(len(storage)==4 and all(s['data_address'] and s['byte_offset']==0 for s in storage),
            'STORAGE_HANDLE_CONTRACT')
    bases=native['tensor_domain_dram_bases']
    require(sorted((b['tensor'],b['domain']) for b in bases)==[(0,2),(1,1),(2,3),(3,4)],
            'TENSOR_DOMAIN_BINDING_CHANGED')
    widths={1:256,2:16,3:64,4:16}
    sizes=tensor_sizes(abi)
    intervals=sorted((b['dram_base']*widths[b['domain']],
                      b['dram_base']*widths[b['domain']]+sizes[b['tensor']]) for b in bases)
    require(all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:])), 'TENSOR_DRAM_REGIONS_OVERLAP')
    return True
