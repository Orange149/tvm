"""Self-contained selected-module bundle; unchanged executor and runtime."""
from pathlib import Path
import shutil
import subprocess
import time
from cpu_vta_partitioning.common import digest,inventory,read,sha256,sync_tree,write
from cpu_vta_partitioning.prepare.build import copy_checked
from cpu_policies.open_resources.domain import validate_candidate
from .common import ROOT,assert_sources,protocol

def prepare(output):
    output=Path(output);prep=assert_sources(output);profile=Path(prep['profile_bundle'])
    ranked=output/'ranking';freeze=read(ranked/'FREEZE.json')
    for leaf,value in freeze['files'].items():
        if sha256(ranked/leaf)!=value:raise ValueError('candidate freeze changed')
    pool=read(ranked/'pool.json');full=read(profile/'plans/programs.json')
    programs={s['program']:full[s['program']] for cfg in pool.values() for s in cfg['stages']}
    for cfg in pool.values():validate_candidate(cfg,full,unit_count=14,max_vta_islands=3)
    dest=output/'pipeline_bundle'
    if (dest/'manifest.json').exists():
        from .validation import validate_bundle
        return validate_bundle(dest)
    dest.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    parent=read(profile/'manifest.json')
    for name,info in parent['files'].items():
        if name.startswith(('runtime/','runner/','firmware/','references/')) or name=='inputs/000.bin':
            copy_checked(profile/name,dest/name,info['sha256'])
    for name in programs:
        for leaf in ('graph.json','module.so','params.bin','record.json'):
            copy_checked(profile/'programs'/name/leaf,dest/'programs'/name/leaf)
    write(dest/'plans/programs.json',programs)
    inputs=read(profile/'plans/inputs.json')
    for r in inputs['records']:
        for frontier in r['frontiers'].values():frontier.pop('raw',None)
    write(dest/'plans/inputs.json',inputs)
    copy_checked(profile/'plans/expected_outputs.json',dest/'plans/expected_outputs.json')
    for leaf in ('rankings.json','pool.json','execution_order.json','scores.json','proofs.json','protocol.json'):
        copy_checked(ranked/leaf,dest/'plans'/leaf)
    copy_checked(ranked/'FREEZE.json',dest/'plans/ranking_freeze.json')
    copy_checked(output/'PREPARATION.json',dest/'provenance/PREPARATION.json')
    copy_checked(profile/'manifest.json',dest/'provenance/profile_manifest.json')
    copy_checked(ranked/'ddr.json',dest/'provenance/ddr.json')
    shutil.copytree(ranked/'inputs',dest/'provenance/static_inputs',dirs_exist_ok=True)
    # Freeze exactly the source closure used; old snapshots are never edited.
    for rel,expected in prep['sources'].items():copy_checked(ROOT/rel,dest/'sources/policies'/rel,expected)
    import cpu_vta_partitioning
    core=Path(cpu_vta_partitioning.__file__).parent
    for p in core.rglob('*.py'):copy_checked(p,dest/'sources/base_controller/cpu_vta_partitioning'/p.relative_to(core))
    qemu=shutil.which('qemu-aarch64-static')
    if not qemu:raise RuntimeError('QEMU loader required')
    result=subprocess.run([qemu,str(dest/'runtime/ld-linux-aarch64.so.1'),'--library-path',str(dest/'runtime'),
        str(dest/'runner/vta_stage_pipeline_runner'),'--help'],capture_output=True,text=True,timeout=45)
    if result.returncode or '--cpu-worker-pinning' not in result.stdout+result.stderr:
        raise ValueError('aarch64 loader or pinning option check failed')
    write(dest/'plans/qemu_loader.json',dict(passed=True,stdout=result.stdout,stderr=result.stderr))
    write(dest/'plans/board_execution.json',dict(boot_policy='one_boot_no_automatic_reboot',udmabuf_bytes=201326592,
        storage_margin_bytes=268435456))
    files={n:i for n,i in inventory(dest).items() if n.startswith(('programs/','runtime/','runner/','firmware/'))
        and n.rsplit('/',1)[-1]!='record.json'}
    for n in ('inputs/000.bin','plans/board_execution.json'):files[n]=dict(bytes=(dest/n).stat().st_size,sha256=sha256(dest/n))
    write(dest/'plans/board_payload.json',dict(files=files,unpacked_bytes=sum(v['bytes'] for v in files.values())))
    write(dest/'plans/preparation_cost.json',dict(seconds=time.monotonic()-started,candidates=len(pool),
        programs=len(programs),new_compilations=0,payload_bytes=sum(v['bytes'] for v in files.values())))
    manifest=dict(model='YOLOv3-tiny416',scope='three_strict_Top20_short_screen_fixed_long_confirmation',
        runner_sha256=parent['runner_sha256'],profile_capsule_id=parent['bundle_id'],
        candidate_count=len(pool),files=inventory(dest))
    manifest['bundle_id']=digest(manifest)[:24]
    write(dest/'manifest.json',manifest);sync_tree(dest)
    return manifest
