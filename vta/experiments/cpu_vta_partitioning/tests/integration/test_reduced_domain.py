"""A failed compilation must narrow every group's domain without falsifying coverage."""
from pathlib import Path
from cpu_vta_partitioning.common import PROJECT, read, write, inventory
from cpu_vta_partitioning.prepare.package import create_plans
from cpu_vta_partitioning.prepare.payload import validate_payload
from cpu_vta_partitioning.domain import signatures


def test_reduced_compilation_domain_has_closed_payload_and_frozen_cost_templates(tmp_path):
    specs=read(PROJECT/'tests/fixtures/full_domain.json')
    programs={}
    for spec in specs:
        directory=tmp_path/'programs'/spec['name'];directory.mkdir(parents=True)
        for name in ('module.so','params.bin','graph.json'): (directory/name).write_text(spec['name']+name)
        programs[spec['name']]={**spec,'input_names':['input'], 'input_node_ids':[spec['begin']],
                              'output_node_ids':[spec['end']], 'input_types':[{'dtype':'float32','shape':[1]}],
                              'output_types':[{'dtype':'float32','shape':[1]}], 'files':inventory(directory)}
    # Leave the failed module's stale cache files in place deliberately.
    failed='vta_01_01';programs.pop(failed)
    write(tmp_path/'plans/programs.json',programs)
    write(tmp_path/'plans/segments.json',specs)
    write(tmp_path/'plans/compile_failures.json',{failed:{'error':'synthetic compiler failure'}})
    write(tmp_path/'plans/protocol.json',{'environment':{'cores':4,'poll_sleep_ns':1000,'post_start_sleep_ns':1000}})
    write(tmp_path/'plans/sensitivity.json',read(PROJECT/'configs/sensitivity.json'))
    write(tmp_path/'plans/inputs.json',{'records':[{'input':'inputs/image.bin',
          'frontiers':{str(i):{'raw':'inputs/frontier.bin'} for i in range(22)}} for _ in range(67)]})
    for name in ('inputs/image.bin','inputs/frontier.bin','runtime/runtime.so','runner/native','firmware/bitstream'):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'test')
    coverage=create_plans(tmp_path)
    assert coverage['compiled_programs']=={'cpu':85,'vta':86}
    assert 0<coverage['qualified_domain']['topologies']<coverage['original_domain']['topologies']
    assert coverage['shared_across_all_groups']
    templates=read(tmp_path/'plans/calibration/cores4_poll1000.json')
    assert len(templates)==len(signatures(programs))==512
    assert all(t['stages'][0]['program']!=failed for t in templates)
    payload=validate_payload(tmp_path)
    assert not any(f.startswith('programs/'+failed+'/') for f in payload['files'])
