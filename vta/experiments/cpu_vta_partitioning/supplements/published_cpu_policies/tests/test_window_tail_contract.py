"""Compare native argument vectors of the old and prepared tail-only tasks."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_window_tail_check as check
from cpu_vta_partitioning.common import read
from common_top20.runtime import task,command


@pytest.mark.parametrize('network',['resnet18','yolov3_tiny416'])
def test_every_representative_keeps_native_execution_except_total_tail_calls(network):
    check.assert_sources()
    new_bundle=check.OUT/network/'capsule'
    old_bundle=check.OLD/network/'capsule'
    old_manifest,new_manifest=[read(b/'manifest.json') for b in (old_bundle,new_bundle)]
    assert old_manifest['runner_sha256']==new_manifest['runner_sha256']
    assert old_manifest['runtime_sha256']==new_manifest['runtime_sha256']
    old_programs,new_programs=[read(b/'plans/programs.json') for b in (old_bundle,new_bundle)]
    for cfg in read(check.OUT/'POOL.json')[network].values():
        for stage in cfg['stages']:
            name=stage['program']
            assert old_programs[name]==new_programs[name]
        for phase in ('qualify','short'):
            docs=[task(b,cfg,phase) for b in (old_bundle,new_bundle)]
            (before,before_inputs),(after,after_inputs)=[command(doc,b,'/frozen','/job') for doc,b in zip(docs,(old_bundle,new_bundle))]
            if phase=='short':
                count=after.index('--runs')+1
                assert before[count]=='28' and after[count]=='48'
                after[count]='28'
                # The native runner cycles its one-file input list for --runs.
                assert before_inputs==after_inputs=='/frozen/inputs/000.bin\n'
            else:
                assert before_inputs==after_inputs
            assert before==after
