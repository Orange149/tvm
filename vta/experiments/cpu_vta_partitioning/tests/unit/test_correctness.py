from pathlib import Path
import numpy as np
import pytest
from cpu_vta_partitioning.common import write
from cpu_vta_partitioning.measurement import correctness, input_sequence


def test_full_multioutput_integer_and_float_correctness(tmp_path):
    bundle=tmp_path/'bundle';result=tmp_path/'result'
    (bundle/'references').mkdir(parents=True);(result/'outputs/run_0').mkdir(parents=True)
    arrays=[np.array([1,-2,3],dtype='int8'),np.array([.5,1.5],dtype='float32')]
    frontiers={}
    files=[]
    for i,array in enumerate(arrays):
        np.save(bundle/f'references/{i}.npy',array)
        frontiers[str(i)]={'npy':f'references/{i}.npy'}
        array.tofile(result/f'outputs/run_0/stage0_output_{i}.bin')
        files.append({'path':f'/remote/outputs/run_0/stage0_output_{i}.bin','dtype':str(array.dtype),'shape':list(array.shape)})
    write(bundle/'plans/programs.json',{'program':{'output_node_ids':[0,1]}})
    write(bundle/'plans/inputs.json',{'records':[{'index':0,'frontiers':frontiers}]})
    task={'kind':'qualify_component','warmup':0,'scored':1,'tail':0,'input_indices':[0],'stages':[{'program':'program'}]}
    rows=[{'frame_id':0,'completion_index':0,'input_index':0,'stage_raw_outputs':[{'output_files':files}]}]
    assert correctness(rows,task,bundle,result)
    np.array([1,-2,4],dtype='int8').tofile(result/'outputs/run_0/stage0_output_0.bin')
    with pytest.raises(ValueError,match='incorrect full output'):
        correctness(rows,task,bundle,result)


def test_protocol_changes_cannot_be_silently_ignored():
    from cpu_vta_partitioning.domain import validate_protocol
    from cpu_vta_partitioning.common import read,PROJECT
    protocol=read(PROJECT/'configs/resnet18.json')
    validate_protocol(protocol)
    protocol['statistics']['alpha']=.1
    with pytest.raises(ValueError,match='unsupported protocol'):
        validate_protocol(protocol)


@pytest.mark.parametrize('indices', [[0,0,0], [0,2,1], [0,1,3], [0,1,-1], [0,1,None]])
def test_repeated_missing_and_wrong_frame_inputs_are_rejected(indices):
    task={'kind':'qualify_component','warmup':0,'scored':3,'tail':0,'input_indices':[64,65,66]}
    rows=[{'frame_id':i,'input_index':index} for i,index in enumerate(indices)]
    with pytest.raises(ValueError,match='input index'):
        input_sequence(rows,task)


def test_input_cycles_follow_frame_identity_and_qualification_requires_full_coverage():
    task={'kind':'calibration','warmup':1,'scored':4,'tail':0,'input_indices':[0,1]}
    rows=[{'frame_id':i,'input_index':i%2} for i in reversed(range(5))]
    assert input_sequence(rows,task)
    task.update(kind='qualify_component',warmup=0,scored=1)
    with pytest.raises(ValueError,match='cover every'):
        input_sequence([{'frame_id':0,'input_index':0}],task)
