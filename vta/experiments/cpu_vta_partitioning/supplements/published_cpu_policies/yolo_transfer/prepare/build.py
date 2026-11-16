"""Local, resumable SDK builds with a frozen compiler/source identity."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from cpu_vta_partitioning.common import inventory, locked, read, sha256, verify_inventory, write
from cpu_vta_partitioning.prepare.build import copy_checked, sdk_command, source_path


PROBE_PROGRAMS = [
    {'name': 'reference_full', 'kind': 'complete_graph', 'device': 'cpu'},
    {'name': 'reference_frontiers', 'kind': 'complete_frontier_probe', 'device': 'cpu'},
    {'name': 'vta_bridge_probe', 'kind': 'interval', 'device': 'vta', 'begin': 1, 'end': 13},
]


def initialize(graph_directory, workspace, local_path, programs):
    graph_directory, workspace = Path(graph_directory).resolve(), Path(workspace).resolve()
    local = read(local_path)
    status = read(graph_directory / 'GRAPH_STATUS.json')
    if status['canonical_sha256'] != sha256(graph_directory / 'canonical_qir.json'):
        raise ValueError('common graph identity changed')
    if status['input_sha256'] != sha256(graph_directory / 'input0.bin'):
        raise ValueError('common input identity changed')
    for name in ('canonical_qir.json', 'GRAPH_STATUS.json', 'qconfig.json'):
        copy_checked(graph_directory / name, workspace / 'sources' / name)
    copy_checked(graph_directory / 'input0.bin', workspace / 'inputs/000.bin')
    copy_checked(source_path(local['tuning_history']), workspace / 'sources/tuning_history.log')
    package = Path(__file__).resolve().parents[1]
    for source in package.rglob('*.py'):
        copy_checked(source, workspace / 'sources/yolo_transfer' / source.relative_to(package))
    repo = source_path(local['host_tvm_library']).parents[1]
    copy_checked(repo / 'vta/python/vta/top/graphpack.py', workspace / 'sources/graphpack.source.py')
    copy_checked(repo / '3rdparty/vta-hw/config/vta_config.json', workspace / 'sources/vta_config.json')
    identity = {
        'host_tvm_sha256': sha256(source_path(local['host_tvm_library'])),
        'sdk_setup_sha256': sha256(source_path(local['sdk_setup'])),
        'files': {str(p.relative_to(workspace)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                  for p in sorted((workspace / 'sources').rglob('*')) if p.is_file()
                  and '__pycache__' not in p.parts},
        'input': inventory(workspace / 'inputs'),
        'programs': programs,
        'history_usage': 'exact target/workload schedule lookup only; no prior performance costs',
        'model': 'yolov3-tiny-416-common-quantized',
    }
    path = workspace / 'plans/compiler_identity.json'
    if path.exists():
        if read(path) != identity:
            raise ValueError('YOLO preparation identity changed; use a new workspace')
    else:
        write(path, identity)
        write(workspace / 'plans/programs.json', programs)
    return local


def build(graph_directory, workspace, local_path, *, names=None, full_domain=False):
    workspace = Path(workspace).resolve()
    with locked(workspace):
        if full_domain:
            from yolo_transfer.domain import programs as domain_programs, summary
            programs = domain_programs()
            write(workspace / 'DOMAIN.json', summary(), replace=True)
        else:
            programs = PROBE_PROGRAMS
        local = initialize(graph_directory, workspace, local_path, programs)
        names = list(names or [s['name'] for s in programs])
        if not names or len(names) != len(set(names)) or set(names) - {s['name'] for s in programs}:
            raise ValueError('invalid or duplicate requested program')
        def one(name):
            log = workspace / 'build_logs' / (name + '.log')
            sdk_command(local, [local['python'], '-m', 'yolo_transfer.prepare.compiler',
                '--workspace', workspace, '--program', name], log, cwd=workspace / 'sources')
            record = read(workspace / 'programs' / name / 'record.json')
            verify_inventory(workspace / 'programs' / name, record['files'])
            return {'name': name, 'compile_seconds': record['compile_seconds']}
        result = []
        with ThreadPoolExecutor(max_workers=min(2, int(local['compile_jobs']))) as pool:
            futures = {pool.submit(one, name): name for name in names}
            for future in as_completed(futures):
                row = future.result()
                print(row, flush=True)
                result.append(row)
        write(workspace / 'BUILD_STATUS.json', {
            'status': 'LOCAL_PROGRAMS_COMPILED', 'programs': result,
            'ARM_correctness_verified': False, 'board_accessed': False}, replace=True)
        return result
