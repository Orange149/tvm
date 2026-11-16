"""Feature units, conservative scope and cross-network profile identities."""
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]

from workload_model.features import tensor_bytes,graph_features,placement_features


@pytest.mark.parametrize('shape,dtype,expected',[([1,3,224,224],'float32',602112),
    ([1,3,416,416],'float32',2076672),([3],'int8',3),([],'int64',8),([5],'int4',3)])
def test_tensor_byte_units(shape,dtype,expected):
    assert tensor_bytes(dict(shape=shape,dtype=dtype))==expected


@pytest.mark.parametrize('value',[dict(shape=[-1],dtype='int8'),dict(shape=[True],dtype='int8'),dict(shape=[1],dtype='handle')])
def test_unknown_memory_scope_rejected(value):
    with pytest.raises(ValueError):tensor_bytes(value)


def test_graph_storage_reuse_and_convolution_MAC_scope():
    g=dict(nodes=[dict(op='null',name='x'),dict(op='null',name='w'),
        dict(op='tvm_op',name='conv',attrs=dict(func_name='fused_nn_conv2d_add',data_layout='NCHW',kernel_layout='OIHW'),inputs=[[0,0,0],[1,0,0]])],
        node_row_ptr=[0,1,2,3],attrs=dict(shape=['list_shape',[[1,3,4,4],[2,3,3,3],[1,2,2,2]]],
            dltype=['list_str',['int8','int8','int32']],storage_id=['list_int',[0,1,0]]))
    f=graph_features(g)
    assert f['recognized_dense_convolution_MACs']==1*2*2*2*3*3*3
    assert f['graph_planned_storage_proxy_bytes']==48+54
    assert f['graph_entry_logical_bytes']==48+54+32
    assert f['graph_functions_are_not_parallel_launches'] is True


def test_placement_counts_allow_partial_overlap_and_do_not_assume_runnable():
    class Fake:
        def get(self,row,context):return dict(service_ms=row['threads']*2,run_core_ms=row['threads']*3)
    cfg=dict(stages=[dict(device='cpu',threads=4,mask=[0,1,2,3]),dict(device='vta'),
        dict(device='cpu',threads=3,mask=[1,2,3]),dict(device='vta'),dict(device='cpu',threads=1,mask=[0])],
        execution_mode='off',context_id='shared4')
    value=placement_features(cfg,Fake())
    assert value['core_thread_memberships']==[2,2,2,2]
    assert value['configured_CPU_threads']==8
    assert [v['relation'] for v in value['pairs']]==['partial','partial','disjoint']
    assert value['configured_threads_not_simultaneously_runnable_threads']
