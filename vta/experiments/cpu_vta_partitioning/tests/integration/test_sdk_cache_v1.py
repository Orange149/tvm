"""Toolchain switches must update native builds and the complete SDK library closure."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cpu_vta_partitioning.common import digest, inventory, read, sha256, write
from cpu_vta_partitioning.prepare import build, package, toolchain


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.mark.parametrize('indirect', [False, True])
def test_native_sdk_switch_reconfigures_all_targets_and_reuses_matching_cache(tmp_path, monkeypatch, indirect):
    repo = tmp_path/'repo'; workspace = tmp_path/'workspace'; workspace.mkdir()
    put(repo/'CMakeLists.txt', 'cmake_minimum_required(VERSION 3.16)\nproject(probe CXX)\n'
        'add_library(tvm_runtime SHARED runtime.cc)\nadd_library(vta SHARED driver.cc)\n')
    put(repo/'runtime.cc', 'extern "C" int runtime_generation() { return SDK_GENERATION; }\n')
    put(repo/'driver.cc', 'extern "C" int driver_generation() { return SDK_GENERATION; }\n')
    put(repo/'vta/apps/native_deploy/vta_stage_pipeline_runner.cc', '#include <cstdio>\n'
        'extern "C" int runtime_generation();\nextern "C" int driver_generation();\n'
        'int main() { std::printf("%d %d %d\\n", SDK_GENERATION, runtime_generation(), driver_generation()); }\n')
    put(repo/'3rdparty/picojson/picojson.h', '// unused probe header\n')
    put(tmp_path/'host.so', 'unused host library')
    flags = tmp_path/'selected-toolchain.sh'
    shared_setup = tmp_path/'sdk.sh'
    put(shared_setup, 'source "' + str(flags) + '"\n')
    local = {'host_tvm_library':str(tmp_path/'host.so'), 'python':sys.executable, 'compile_jobs':2}
    monkeypatch.setattr(build, 'REPO', repo)
    roots = {}
    for generation in (1, 2, 1):
        setup = flags if indirect else tmp_path/f'sdk{generation}.sh'
        put(setup, f'export CXX="/usr/bin/g++ -DSDK_GENERATION={generation}"\n'
            f'export CC=/usr/bin/gcc SDKTARGETSYSROOT="{repo}"\n')
        local['sdk_setup'] = str(shared_setup if indirect else setup)
        identity = toolchain.inspect_toolchain(workspace, local)
        if (workspace/'plans/compiler_identity.json').exists():
            with pytest.raises(ValueError, match='toolchain changed'):
                build.compile_runtime(workspace, local)
        write(workspace/'plans/compiler_identity.json', {'toolchain':identity}, replace=True)
        build.compile_runtime(workspace, local)
        output = subprocess.check_output([str(workspace/'runner/vta_stage_pipeline_runner')], text=True,
                                         env=dict(os.environ, LD_LIBRARY_PATH=str(workspace/'runtime')))
        assert output.strip() == f'{generation} {generation} {generation}'
        record = read(workspace/'plans/runtime_build.json')
        root = workspace/record['native_build_directory']
        assert record['toolchain'] == identity
        if generation in roots:
            assert root == roots[generation][0]
            assert (root/'build.ninja').stat().st_mtime_ns == roots[generation][1]
        else:
            roots[generation] = root, (root/'build.ninja').stat().st_mtime_ns
    assert roots[1][0] != roots[2][0]


@pytest.fixture
def closure(tmp_path, monkeypatch):
    workspace = tmp_path/'workspace'
    current = {}
    # ELF dependency topology is synthetic here; the full offline suite checks
    # actual AArch64 ELF files, and the native test above executes real binaries.
    def library(path, generation, needed=()):
        write(path, {'needed':list(needed), 'rpath':[], 'architecture':'aarch64',
                     'generation':generation}, replace=True)
    library(workspace/'runtime/libtvm_runtime.so', 0, ['libdep.so.1'])
    library(workspace/'runtime/libvta.so', 0)
    library(workspace/'runner/vta_stage_pipeline_runner', 0, ['libtvm_runtime.so','libvta.so'])
    library(workspace/'firmware/u-dma-buf.ko', 0)
    write(workspace/'plans/programs.json', {})
    monkeypatch.setattr(package, 'elf', read)
    monkeypatch.setattr(toolchain, 'require_prepared_toolchain', lambda *args:dict(current))
    def select(generation, same_sysroot=False):
        sdk = tmp_path/('sdk' if same_sysroot else f'sdk{generation}')
        library(sdk/'lib/libdep.so.1', generation, ['libchild.so.1'])
        library(sdk/'lib/libchild.so.1', generation)
        library(sdk/'lib/ld-linux-aarch64.so.1', generation)
        current.clear(); current.update(sysroot=str(sdk))
        write(workspace/'plans/runtime_build.json', {'toolchain':dict(current),
            'runtime_sha256':sha256(workspace/'runtime/libtvm_runtime.so'),
            'driver_sha256':sha256(workspace/'runtime/libvta.so')}, replace=True)
        return sdk
    return workspace, select


@pytest.mark.parametrize('same_sysroot', [False, True])
def test_sdk_closure_refreshes_libraries_loader_and_prunes_stale_files(closure, same_sysroot):
    workspace, select = closure
    select(1, same_sysroot)
    package.runtime_closure(workspace, {})
    put(workspace/'runtime/libunused.so.0', 'stale SDK file')
    native_before = {name:sha256(workspace/'runtime'/name) for name in ('libtvm_runtime.so','libvta.so')}
    sdk = select(2, same_sysroot)
    package.runtime_closure(workspace, {})
    provenance = read(workspace/'plans/sdk_runtime.json')
    assert len(provenance['libraries']) == 3
    for name, record in provenance['libraries'].items():
        assert sha256(workspace/name) == record['sha256'] == sha256(sdk/'lib'/Path(name).name)
        assert Path(record['source']).is_relative_to(sdk)
        assert read(workspace/name)['generation'] == 2
    assert all(sha256(workspace/'runtime'/name) == value for name,value in native_before.items())
    assert not (workspace/'runtime/libunused.so.0').exists()


@pytest.mark.parametrize('missing', ['libdep.so.1', 'libchild.so.1', 'ld-linux-aarch64.so.1'])
def test_missing_current_sdk_file_cannot_be_satisfied_by_old_cache(closure, missing):
    workspace, select = closure
    select(1); package.runtime_closure(workspace, {})
    sdk = select(2); (sdk/'lib'/missing).unlink()
    before = inventory(workspace/'runtime')
    with pytest.raises(FileNotFoundError, match='unresolved SDK'):
        package.runtime_closure(workspace, {})
    assert inventory(workspace/'runtime') == before


def test_closure_rejects_native_runtime_from_different_sdk(closure):
    workspace, select = closure
    select(1); package.runtime_closure(workspace, {})
    native = read(workspace/'plans/runtime_build.json')
    select(2); write(workspace/'plans/runtime_build.json', native, replace=True)
    before = inventory(workspace/'runtime')
    with pytest.raises(ValueError, match='runtime and SDK'):
        package.runtime_closure(workspace, {})
    assert inventory(workspace/'runtime') == before


def test_interrupted_closure_publication_is_rebuilt_on_retry(closure, monkeypatch):
    workspace, select = closure
    select(1); package.runtime_closure(workspace, {})
    sdk = select(2)
    replace = os.replace
    count = 0
    def interrupted(source, target):
        nonlocal count
        if Path(source).parent.name.startswith('.runtime-closure-'):
            count += 1
            if count == 2:
                raise OSError('simulated interruption')
        return replace(source, target)
    with monkeypatch.context() as patch:
        patch.setattr(os, 'replace', interrupted)
        with pytest.raises(OSError, match='simulated interruption'):
            package.runtime_closure(workspace, {})
    package.runtime_closure(workspace, {})
    for name, record in read(workspace/'plans/sdk_runtime.json')['libraries'].items():
        assert sha256(workspace/name) == record['sha256'] == sha256(sdk/'lib'/Path(name).name)
