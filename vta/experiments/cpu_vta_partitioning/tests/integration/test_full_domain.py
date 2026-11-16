"""Full frozen-domain algorithm audit; no TVM, SSH or historical costs."""
from pathlib import Path
import os
import pytest
from cpu_vta_partitioning.common import read, PROJECT, write
from cpu_vta_partitioning.domain import signatures
from cpu_vta_partitioning.models import Cost
from cpu_vta_partitioning.search import rank,enumerate_reference


def test_full_frozen_domain_vector_search_matches_scalar_enumeration():
    programs={p['name']:p for p in read(PROJECT/'tests/fixtures/full_domain.json')}
    signatures_=signatures(programs)
    assert len(signatures_)==514
    costs={key:Cost(1+(s.end-s.begin)/(s.threads if s.device=='cpu' else 5),
                    (s.end-s.begin)*(1 if s.device=='cpu' else .3),.2+s.begin/100) for key,s in signatures_.items()}
    actual=rank(programs,costs)
    assert actual['counts']['topologies']==4623
    assert actual['counts']['shared_configurations']==972528
    expected=enumerate_reference(programs,costs)
    for group,rows in actual['rankings'].items():
        assert [(r['score_ms'],r['id']) for r in rows]==expected[group]
    if os.environ.get('CPUVTA_WORKSPACE'):
        write(Path(os.environ['CPUVTA_WORKSPACE'])/'plans/full_domain_search_audit.json',
              {'source':'synthetic_test_only','counts':actual['counts'],'independent_scalar_enumeration_matches':True,
               'groups':sorted(expected),'score_rounding_ms':1e-9},replace=True)


def test_real_compiled_closure_and_frontiers_when_preparing():
    value=os.environ.get('CPUVTA_WORKSPACE')
    if not value:
        pytest.skip('set CPUVTA_WORKSPACE after local compilation')
    from cpu_vta_partitioning.prepare.package import create_plans,elf
    workspace=Path(value)
    coverage=create_plans(workspace)
    programs=read(workspace/'plans/programs.json')
    requested={p['name'] for p in read(workspace/'plans/segments.json')}
    failures=set(coverage['failed_programs'])
    assert set(programs).isdisjoint(failures) and set(programs) | failures == requested
    assert coverage['qualified_domain']['topologies'] > 0
    assert coverage['shared_across_all_groups']
    for name,row in programs.items():
        assert elf(workspace/'programs'/name/'module.so')['architecture']=='aarch64'
        assert row['recomposition_exact'] and row['extra_quantize_calls']==0
        if row['device']=='cpu':
            import subprocess
            symbols=subprocess.check_output(['/usr/bin/readelf','-Ws',str(workspace/'programs'/name/'module.so')],text=True)
            assert not any(' UND ' in line and 'VTA' in line for line in symbols.splitlines())
    # Parameter support is checked without executing an aarch64 binary on x86.
    runner=(workspace/'runner/vta_stage_pipeline_runner').read_bytes()
    for flag in (b'--input-list-multi',b'--session-profile-window',b'--thread-affinity-report',b'--cpu-stage-gate'):
        assert flag in runner
    import shutil,subprocess
    emulator=shutil.which('qemu-aarch64-static')
    if emulator:
        result=subprocess.run([emulator,str(workspace/'runtime/ld-linux-aarch64.so.1'),'--library-path',str(workspace/'runtime'),
                               str(workspace/'runner/vta_stage_pipeline_runner'),'--help'],check=True,capture_output=True,text=True,timeout=30)
        help_text=result.stdout+result.stderr
        assert '--session-profile-window' in help_text
        write(workspace/'plans/qemu_runner_help.json',{'passed':True,'scope':'argument parser and bundled ELF loader; no FPGA execution',
                                                      'help':help_text},replace=True)


@pytest.mark.parametrize('begin,end',[(0,1),(2,3),(20,21)])
def test_qemu_multiinput_cpu_numerics_when_preparing(tmp_path,begin,end):
    value=os.environ.get('CPUVTA_WORKSPACE')
    if not value:
        pytest.skip('set CPUVTA_WORKSPACE after local compilation')
    import shutil,subprocess
    emulator=shutil.which('qemu-aarch64-static')
    if not emulator:
        # QEMU is optional; board correctness remains mandatory.
        return
    from cpu_vta_partitioning.measurement.tasks import task,native_command
    from cpu_vta_partitioning.domain import Stage
    from cpu_vta_partitioning.measurement import frames,correctness
    workspace=Path(value)
    available=read(workspace/'plans/programs.json')
    if f'cpu_{begin:02}_{end-1:02}' not in available:
        replacement=next(p for _,p in sorted(available.items()) if p['device']=='cpu')
        begin,end=replacement['begin'],replacement['end']
    t=task('qualify_component',[Stage('cpu',begin,end,1,(0,))],{'cores':4,'poll_sleep_ns':1000,'post_start_sleep_ns':1000},
           warmup=0,scored=3,tail=0,indices=[64,65,66])
    args,lines=native_command(t,workspace,str(workspace),str(tmp_path))
    (tmp_path/'inputs.txt').write_text(lines)
    i=next(i for i,x in enumerate(args) if x.endswith('ld-linux-aarch64.so.1'))
    args.insert(i,emulator)
    result=subprocess.run(args,capture_output=True,text=True,timeout=45)
    (tmp_path/'native.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr
    rows=frames(tmp_path/'result.jsonl',3)
    assert correctness(rows,t,workspace,tmp_path)
    name=f'cpu_{begin:02}_{end-1:02}'
    write(workspace/f'plans/qemu_numerics_{name}.json',{'passed':True,'program':name,'inputs':[64,65,66],
                                                  'scope':'CPU-only numerical and multi-input list check; no FPGA performance claim'},replace=True)


def test_qemu_repeated_diagnostic_input_is_rejected_when_preparing(tmp_path):
    value=os.environ.get('CPUVTA_WORKSPACE')
    if not value:
        pytest.skip('set CPUVTA_WORKSPACE after local compilation')
    import shutil,subprocess
    emulator=shutil.which('qemu-aarch64-static')
    if not emulator:
        return
    from cpu_vta_partitioning.measurement.tasks import task,native_command
    from cpu_vta_partitioning.domain import Stage
    from cpu_vta_partitioning.measurement import frames,correctness
    workspace=Path(value)
    available=read(workspace/'plans/programs.json')
    p=next(p for _,p in sorted(available.items()) if p['device']=='cpu')
    document=task('qualify_component',[Stage('cpu',p['begin'],p['end'],1,(0,))],
                  {'cores':4,'poll_sleep_ns':1000,'post_start_sleep_ns':1000},
                  warmup=0,scored=3,tail=0,indices=[64,65,66])
    argv,lines=native_command(document,workspace,str(workspace),str(tmp_path))
    (tmp_path/'inputs.txt').write_text(lines.splitlines()[0]+'\n')
    i=next(i for i,x in enumerate(argv) if x.endswith('ld-linux-aarch64.so.1'))
    argv.insert(i,emulator)
    completed=subprocess.run(argv,capture_output=True,text=True,timeout=45)
    assert completed.returncode==0,completed.stderr
    rows=frames(tmp_path/'result.jsonl',3)
    assert [row['input_index'] for row in rows]==[0,0,0]
    with pytest.raises(ValueError,match='input index'):
        correctness(rows,document,workspace,tmp_path)
