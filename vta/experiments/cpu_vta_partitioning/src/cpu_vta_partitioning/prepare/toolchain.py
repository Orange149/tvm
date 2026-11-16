"""Identify the effective SDK environment before reusing local build artifacts."""
import sys

from cpu_vta_partitioning.common import read, sha256


# Run inside the sourced SDK so compiler lookup and inherited flags are accurate.
PROBE = r'''
import hashlib, json, os, shlex, shutil
from pathlib import Path
keys = ('CC', 'CXX', 'AR', 'AS', 'LD', 'RANLIB', 'STRIP', 'CFLAGS', 'CXXFLAGS',
        'CPPFLAGS', 'LDFLAGS', 'SDKTARGETSYSROOT', 'OECORE_NATIVE_SYSROOT',
        'OECORE_TARGET_SYSROOT', 'OECORE_SDK_VERSION', 'CMAKE_TOOLCHAIN_FILE')
environment = {key: os.environ.get(key, '') for key in keys}
def identify(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
tools = {}
for key in ('CC', 'CXX', 'AR', 'AS', 'LD', 'RANLIB', 'STRIP'):
    command = shlex.split(environment[key])
    if not command:
        if key in ('CC', 'CXX'):
            raise ValueError('SDK must specify ' + key)
        continue
    executable = shutil.which(command[0])
    if executable is None:
        raise ValueError('SDK tool not found: ' + command[0])
    tools[key] = identify(executable)
cross = shutil.which('aarch64-xilinx-linux-g++')
if cross:
    tools['stage_cross_compiler'] = identify(cross)
sysroot = Path(environment['SDKTARGETSYSROOT'])
if not environment['SDKTARGETSYSROOT'] or not sysroot.is_absolute() or not sysroot.is_dir():
    raise ValueError('SDK must specify an existing absolute SDKTARGETSYSROOT')
setup_files = {}
for key in ('SDKTARGETSYSROOT', 'OECORE_NATIVE_SYSROOT'):
    if environment[key]:
        for path in sorted((Path(environment[key]) / 'environment-setup.d').glob('*.sh')):
            setup_files[str(path)] = identify(path)
if environment['CMAKE_TOOLCHAIN_FILE']:
    setup_files['cmake_toolchain'] = identify(environment['CMAKE_TOOLCHAIN_FILE'])
print(json.dumps({'environment': environment, 'tools': tools, 'setup_files': setup_files,
                  'sysroot': str(sysroot.resolve())}, sort_keys=True))
'''


def inspect_toolchain(workspace, local):
    from cpu_vta_partitioning.prepare.build import sdk_command, source_path
    setup = source_path(local['sdk_setup']).resolve()
    before = sha256(setup)
    log = workspace / 'build_logs/sdk_toolchain.log'
    sdk_command(local, [local.get('python', sys.executable), '-c', PROBE], log)
    import json
    identity = json.loads(log.read_text().splitlines()[-1])
    if before != sha256(setup):
        raise ValueError('SDK setup changed while inspecting toolchain')
    return {'sdk_setup': str(setup), 'sdk_setup_sha256': before, **identity}


def require_prepared_toolchain(workspace, local):
    actual = inspect_toolchain(workspace, local)
    if actual != read(workspace / 'plans/compiler_identity.json')['toolchain']:
        raise ValueError('SDK toolchain changed during preparation; rerun prepare')
    return actual
