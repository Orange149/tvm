"""Selected-module capsule, frozen without rebuilding native code."""
from pathlib import Path
import shutil
import subprocess
import time
from cpu_vta_partitioning.common import digest, inventory, read, sha256, sync_tree, verify_inventory, write
from cpu_vta_partitioning.prepare.build import copy_checked
from cpu_policies.open_resources import domain
from .common import ROOT, protocol, assert_prepared


def validate_selection(output):
    output=Path(output);assert_prepared(output);root=output/'ranking';seal=read(root/'FREEZE.json')
    if seal['preparation_sha256']!=sha256(output/'PREPARATION.json'):raise ValueError('preparation changed')
    for leaf,value in seal['files'].items():
        if sha256(root/leaf)!=value:raise ValueError('selection freeze changed')
    pool=read(root/'pool.json');blocks=read(root/'blocks.json')
    if len(pool)!=seal['candidate_count'] or not 1<=len(pool)<=10:raise ValueError('invalid pool size')
    if set(read(root/'execution_order.json'))!=set(pool):raise ValueError('execution order differs')
    for block in blocks:
        rows={k:pool[v] for k,v in block['candidates'].items()}
        if any(r['context_id']!=block['context']['id'] or [s['program'] for s in r['stages']]!=block['topology']
               for r in rows.values()):raise ValueError('matched block topology/context differs')
        fixed=rows['fixed'];masks=[tuple(s['mask']) for s in fixed['stages'] if s['device']=='cpu']
        if len(set(masks))<2 or not domain.compatible_groups(masks) or fixed['execution_mode']!='resource_group':
            raise ValueError('fixed family requires >=2 disjoint groups')
        twin=domain.candidate(fixed['stages'],fixed['context_id'],'off')
        if twin!=rows['fixed_gate_off'] or rows['open']['execution_mode']!='off':
            raise ValueError('matched gate pair differs beyond its execution mode/identity')
    return pool


def prepare(output):
    output=Path(output);prep=assert_prepared(output);pool=validate_selection(output)
    profile=Path(prep['profile_bundle']);parent=read(profile/'manifest.json')
    dest=output/'pipeline_bundle';started=time.monotonic()
    if (dest/'manifest.json').exists():return validate_bundle(dest)
    full=read(profile/'plans/programs.json')
    programs={s['program']:full[s['program']] for cfg in pool.values() for s in cfg['stages']}
    for cfg in pool.values():domain.validate_candidate(cfg,full,unit_count=14,max_vta_islands=3)
    for name,info in parent['files'].items():
        if name.startswith(('runtime/','runner/','firmware/','references/')) or name=='inputs/000.bin':
            copy_checked(profile/name,dest/name,info['sha256'])
    for name in programs:
        for leaf in ('graph.json','module.so','params.bin','record.json'):
            copy_checked(profile/'programs'/name/leaf,dest/'programs'/name/leaf)
    write(dest/'plans/programs.json',programs)
    images=read(profile/'plans/inputs.json')
    for record in images['records']:
        for frontier in record['frontiers'].values():frontier.pop('raw',None)
    write(dest/'plans/inputs.json',images)
    copy_checked(profile/'plans/expected_outputs.json',dest/'plans/expected_outputs.json')
    seal=read(output/'ranking/FREEZE.json')
    for leaf,value in seal['files'].items():copy_checked(output/'ranking'/leaf,dest/'plans'/leaf,value)
    copy_checked(output/'ranking/FREEZE.json',dest/'plans/ranking_freeze.json')
    copy_checked(output/'PREPARATION.json',dest/'provenance/PREPARATION.json')
    copy_checked(output/'AUDIT.json',dest/'provenance/AUDIT.json')
    shutil.copytree(output/'family_search',dest/'provenance/family_search',dirs_exist_ok=True)
    shutil.copytree(output/'block_search',dest/'provenance/block_search',dirs_exist_ok=True)
    from .common import SOURCE
    shutil.copytree(SOURCE/'ranking/inputs',dest/'provenance/static_inputs',dirs_exist_ok=True)
    copy_checked(SOURCE/'ranking/ddr.json',dest/'provenance/ddr.json')
    import cpu_vta_partitioning
    core=Path(cpu_vta_partitioning.__file__).parent
    for rel,value in prep['sources'].items():
        src=core/rel[6:] if rel.startswith('@base/') else ROOT/rel
        copy_checked(src,dest/'sources'/rel.replace('@base/','base/'),value)
    qemu=shutil.which('qemu-aarch64-static')
    if not qemu:raise RuntimeError('QEMU loader required')
    checked=subprocess.run([qemu,str(dest/'runtime/ld-linux-aarch64.so.1'),'--library-path',str(dest/'runtime'),
        str(dest/'runner/vta_stage_pipeline_runner'),'--help'],capture_output=True,text=True,timeout=45)
    if checked.returncode or '--cpu-worker-pinning' not in checked.stdout+checked.stderr:
        raise ValueError('aarch64 loader or pinning flags unavailable')
    write(dest/'plans/qemu_loader.json',dict(passed=True,stdout=checked.stdout,stderr=checked.stderr))
    write(dest/'plans/board_execution.json',dict(boot_policy='one_boot_no_automatic_reboot',
        udmabuf_bytes=201326592,storage_margin_bytes=268435456))
    files={n:i for n,i in inventory(dest).items() if n.startswith(('programs/','runtime/','runner/','firmware/'))
           and n.rsplit('/',1)[-1]!='record.json'}
    for name in ('inputs/000.bin','plans/board_execution.json'):
        files[name]=dict(bytes=(dest/name).stat().st_size,sha256=sha256(dest/name))
    write(dest/'plans/board_payload.json',dict(files=files,unpacked_bytes=sum(v['bytes'] for v in files.values())))
    write(dest/'plans/preparation_cost.json',dict(seconds=time.monotonic()-started,candidates=len(pool),
        programs=len(programs),new_compilations=0,payload_bytes=sum(v['bytes'] for v in files.values())))
    manifest=dict(model='YOLOv3-tiny416',scope=protocol()['experiment'],runner_sha256=parent['runner_sha256'],
        profile_capsule_id=parent['bundle_id'],candidate_count=len(pool),files=inventory(dest))
    manifest['bundle_id']=digest(manifest)[:24];write(dest/'manifest.json',manifest);sync_tree(dest)
    return validate_bundle(dest)


def validate_bundle(bundle):
    bundle=Path(bundle);m=read(bundle/'manifest.json')
    if m['bundle_id']!=digest({k:v for k,v in m.items() if k!='bundle_id'})[:24]:raise ValueError('bundle seal differs')
    verify_inventory(bundle,m['files'])
    prep=read(bundle/'provenance/PREPARATION.json')
    seal=read(bundle/'plans/ranking_freeze.json')
    if seal['preparation_sha256']!=sha256(bundle/'provenance/PREPARATION.json'):raise ValueError('capsule preparation differs')
    for leaf,value in seal['files'].items():
        if sha256(bundle/'plans'/leaf)!=value:raise ValueError('capsule selection changed')
    if read(bundle/'plans/protocol.json')!=protocol():raise ValueError('capsule protocol differs')
    pool=read(bundle/'plans/pool.json');programs=read(bundle/'plans/programs.json')
    if m['candidate_count']!=len(pool) or len(pool)>10:raise ValueError('capsule pool differs')
    for cfg in pool.values():domain.validate_candidate(cfg,programs,unit_count=14,max_vta_islands=3)
    if len(programs[pool[next(iter(pool))]['stages'][-1]['program']]['output_node_ids'])!=8:
        raise ValueError('all eight terminal outputs required')
    from .common import source_identity
    if source_identity()!=prep['sources']:raise ValueError('controller differs from frozen source')
    return m


def validate(output):
    output=Path(output);pool=validate_selection(output)
    m=validate_bundle(output/'pipeline_bundle') if (output/'pipeline_bundle/manifest.json').exists() else None
    return dict(validated=True,candidates=len(pool),bundle_id=m['bundle_id'] if m else None)
