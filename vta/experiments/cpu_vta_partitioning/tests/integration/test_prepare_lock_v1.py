"""A competing preparation cannot modify the cache before obtaining ownership."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from cpu_vta_partitioning.common import PROJECT, inventory, locked, read, sha256, write
from cpu_vta_partitioning.prepare import build


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    repo = tmp_path/'repo'
    project = repo/'vta/experiments/cpu_vta_partitioning'
    for name in ('resnet18.json', 'sensitivity.json'):
        target = project/'configs'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT/'configs'/name, target)
    files = {
        '3rdparty/vta-hw/config/vta_config.json': '{}',
        'vta/tutorials/frontend/thesis_canonical_graph_split.py': '# adapter',
        'python/tvm/example.py': '# compiler source',
        'vta/experiments/cpu_vta_partitioning/src/cpu_vta_partitioning/prepare/compiler.py': '# worker',
        'graph.json': '{}', 'history.log': 'history', 'host.so': 'host runtime',
        'sdk-a.sh': f'export CC=/usr/bin/gcc CXX=/usr/bin/g++ SDKTARGETSYSROOT={repo}\n# sdk A',
        'sdk-b.sh': f'export CC=/usr/bin/gcc CXX=/usr/bin/g++ SDKTARGETSYSROOT={repo}\n# sdk B',
        'bitstream.bit': 'bitstream', 'udmabuf.ko': 'driver',
    }
    for name, content in files.items():
        path = repo/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    write(repo/'inputs.json', {'canonical_qir_sha256':sha256(repo/'graph.json')})
    write(repo/'segments.json', {'segments':[{'device':'cpu', 'start_index':0, 'end_index':20}]})
    local = {'canonical_graph':'graph.json', 'inputs_manifest':'inputs.json',
             'tuning_history':'history.log', 'segment_manifest':'segments.json',
             'host_tvm_library':'host.so', 'sdk_setup':'sdk-a.sh', 'bitstream':'bitstream.bit',
             'udmabuf_module':'udmabuf.ko', 'python':sys.executable}
    write(project/'configs/local-a.json', local)
    write(project/'configs/local-b.json', {**local, 'sdk_setup':'sdk-b.sh'})
    monkeypatch.setattr(build, 'PROJECT', project)
    monkeypatch.setattr(build, 'REPO', repo)
    return project, repo, project/'configs/resnet18.json', project/'configs/local-a.json'


def test_competing_process_cannot_rewrite_initialization(preparation):
    project, repo, config, local = preparation
    with build.initialize(config, local) as (workspace, _):
        before = inventory(workspace)
        timestamps = {str(p.relative_to(workspace)):p.stat().st_mtime_ns
                      for p in workspace.rglob('*') if p.is_file()}
        # A changed SDK used to overwrite compiler_identity.json before lock rejection.
        code = '''
import sys
from pathlib import Path
from cpu_vta_partitioning.prepare import build
from cpu_vta_partitioning.cli import main
build.PROJECT, build.REPO = map(Path, sys.argv[1:3])
raise SystemExit(main(['prepare', '--config', sys.argv[3], '--local-config', sys.argv[4]]))
'''
        result = subprocess.run([sys.executable, '-c', code, str(project), str(repo), str(config),
                                 str(project/'configs/local-b.json')], cwd=project,
                                env=dict(os.environ, PYTHONPATH=str(PROJECT/'src')),
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 1 and 'another controller owns' in result.stderr
        assert inventory(workspace) == before
        assert {str(p.relative_to(workspace)):p.stat().st_mtime_ns
                for p in workspace.rglob('*') if p.is_file()} == timestamps
    with build.initialize(config, project/'configs/local-b.json') as (reused, _):
        assert reused == workspace
        assert read(reused/'plans/compiler_identity.json')['sdk_setup'] == sha256(repo/'sdk-b.sh')


@pytest.mark.parametrize('fail', [False, True])
def test_cli_keeps_lock_through_seal_and_releases_on_exit(preparation, monkeypatch, fail):
    import cpu_vta_partitioning.cli as cli
    import cpu_vta_partitioning.prepare.package as package
    project, repo, config, local = preparation
    calls = []
    # Keep this test focused on lock ownership; real SDK inspection runs in the
    # cross-process test above and SDK-switch integration tests.
    monkeypatch.setattr(build, 'inspect_toolchain', lambda *args: {'sdk':'lock-test'})
    def check_lock(workspace, *args):
        calls.append(workspace)
        with pytest.raises(RuntimeError, match='another controller owns'):
            with locked(workspace):
                pytest.fail('preparation released its lock early')
    for name in ('prepare_inputs', 'compile_programs', 'compile_runtime'):
        monkeypatch.setattr(build, name, check_lock)
    for name in ('runtime_closure', 'create_plans'):
        monkeypatch.setattr(package, name, check_lock)
    def seal(workspace, evidence):
        check_lock(workspace)
        assert evidence['passed']
        if fail:
            raise ValueError('simulated seal failure')
        return workspace
    monkeypatch.setattr(package, 'seal', seal)
    def tests(argv, **kwargs):
        report = Path(next(a.split('=', 1)[1] for a in argv if a.startswith('--junitxml=')))
        check_lock(report.parents[1])
        report.write_text('<testsuites><testsuite tests="1" errors="0" failures="0" skipped="0"/></testsuites>')
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(cli.subprocess, 'run', tests)
    if fail:
        with pytest.raises(ValueError, match='simulated seal failure'):
            cli.local_prepare(config, local)
    else:
        assert cli.local_prepare(config, local) == calls[0]
    assert len(calls) == 7 and len(set(calls)) == 1
    with locked(calls[0]):
        pass
