"""Portable capsule and memory-accounting checks using synthetic graph fixtures.

No fixture is executed as a DNN. Actual frozen ELF files are used only for
architecture, DT_NEEDED and zero-inference --help checks.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
from types import ModuleType

import pytest

from cpu_vta_partitioning.common import digest, inventory, read, sha256, write
from cpu_policies.open_resources.domain import candidate, canonical_tuple, configuration, stage
from resnet50_validation import bundle as package
from resnet50_validation import prepare as local
from resnet50_validation import validation as v
from resnet50_validation.assets import PARENT
from resnet50_validation.common import ROOT, protocol


def tiny_graph(device=12, size=16):
    return dict(nodes=[dict(op='null', name='edge0_0', inputs=[]),
                       dict(op='tvm_op', name='out', inputs=[[0, 0, 0]])],
                arg_nodes=[0], node_row_ptr=[0, 1, 2], heads=[[1, 0, 0]],
                attrs=dict(shape=['list_shape', [[size], [size]]],
                           dltype=['list_str', ['uint8', 'uint8']],
                           storage_id=['list_int', [0, 1]],
                           device_index=['list_int', [device, device]]))


def selected_lists():
    rows = []
    for context in ('shared4', 'host_shared_0'):
        host = list(range(4)) if context == 'shared4' else [0]
        for first in range(4):
            for last in range(4):
                cfg = candidate([stage('cpu', 0, 1, 1, [first]),
                                 stage('vta', 1, 2, 1, host),
                                 stage('cpu', 2, 38, 1, [last])], context, 'resource_group')
                rows.append(cfg)
    rows = sorted(rows, key=canonical_tuple)[:20]
    pool = {r['id']: r for r in rows}
    ranked = [{**r, 'score_ns': 100, 'static_rank': rank} for rank, r in enumerate(rows, 1)]
    return dict(tarnawski=deepcopy(ranked), complete=deepcopy(ranked)), pool


def test_lists_enforce_exact_budget_identity_order_and_single_core_baseline():
    lists, pool = selected_lists()
    assert v.validate_lists(lists, pool)
    with pytest.raises(ValueError, match='twenty'):
        v.validate_lists({**lists, 'complete': lists['complete'][:-1]}, pool)
    changed = deepcopy(lists)
    changed['complete'][1]['score_ns'] = 99
    with pytest.raises(ValueError, match='ordering'):
        v.validate_lists(changed, pool)
    changed = deepcopy(pool)
    changed.pop(next(iter(changed)))
    with pytest.raises(ValueError, match='pool'):
        v.validate_lists(lists, changed)


def test_visible_storage_counts_reused_slots_once_and_keeps_physical_device():
    graph = tiny_graph(size=32)
    graph['attrs']['storage_id'][1] = [0, 0]
    graph['attrs']['shape'][1] = [[16], [32]]
    result = v.graph_memory(graph, device='vta')
    assert result['visible_VTA_storage_bytes'] == 32
    assert result['storage_slots'] == 1
    graph['attrs']['device_index'][1][1] = 1
    with pytest.raises(ValueError, match='different devices'):
        v.graph_memory(graph, device='vta')


def test_tensor_accounting_rejects_dynamic_shapes_and_handles_dtype_lanes():
    assert v.tensor_bytes([2, 3], 'int8') == 6
    assert v.tensor_bytes([2], 'float32x4') == 32
    assert v.tensor_bytes([2], 'bool') == 2
    with pytest.raises(ValueError, match='static'):
        v.tensor_bytes([-1, 2], 'int8')
    with pytest.raises(ValueError, match='unsupported'):
        v.tensor_bytes([2], 'handle')


def test_all_vta_islands_are_summed_and_queues_are_host_proxies(tmp_path):
    programs = {}
    for name in ('vta_a', 'vta_b'):
        path = tmp_path / 'programs' / name
        path.mkdir(parents=True)
        write(path / 'graph.json', tiny_graph(size=60))
        (path / 'params.bin').write_bytes(b'weights')
        programs[name] = dict(device='vta')
    pool = {'one': dict(stages=[dict(program='vta_a'), dict(program='vta_b')])}
    report = v.memory_report(tmp_path, pool, programs, queue_depth=2, buffer_bytes=200)
    assert report['maximum_visible_VTA_storage_bytes'] == 240
    assert report['visible_estimate_exceeds_existing_buffer']
    assert not report['actual_memory_sufficiency_proved']
    assert report['configurations']['one']['visible_frame_arrays_host_proxy_bytes'] > 0
    assert report['configurations']['one']['parameter_blob_host_bytes'] == 14


@pytest.fixture
def prepared_fixture(tmp_path, monkeypatch):
    if not (PARENT / 'manifest.json').is_file():
        pytest.skip('requires existing immutable original runtime capsule')
    if shutil.which('qemu-aarch64-static') is None or shutil.which('readelf') is None:
        pytest.skip('requires local AArch64 loader and ELF inspection tools')
    output = tmp_path / 'work'
    profile = output / 'profile_capsule'
    parent = read(PARENT / 'manifest.json')
    for name, info in parent['files'].items():
        if name.startswith(('runtime/', 'runner/', 'firmware/')) or name == 'inputs/000.bin':
            target = profile / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(PARENT / name, target)
    elf = next((PARENT / 'programs').glob('*/module.so'))
    programs = {}
    for index, (name, device, begin, end) in enumerate([
            ('cpu_00_00', 'cpu', 0, 1), ('vta_01_01', 'vta', 1, 2), ('cpu_02_37', 'cpu', 2, 38)]):
        target = profile / 'programs' / name
        target.mkdir(parents=True)
        write(target / 'graph.json', tiny_graph(device=12 if device == 'vta' else 1))
        shutil.copy2(elf, target / 'module.so')
        (target / 'params.bin').write_bytes(b'synthetic-test-parameters')
        record = dict(name=name, device=device, begin=begin, end=end, kind='interval',
                      canonical_sha256='a' * 64, input_node_ids=[index], output_node_ids=[index + 1],
                      files=inventory(target))
        programs[name] = record
        write(target / 'record.json', record)
    frontiers, expected = {}, {}
    for i in range(4):
        raw, npy = f'inputs/frontiers/000/{i}.bin', f'references/000/{i}.npy'
        for path, data in [(raw, bytes([i]) * 16), (npy, b'synthetic-reference-placeholder')]:
            target = profile / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        frontiers[str(i)] = dict(raw=raw, npy=npy, dtype='uint8', shape=[16])
        expected[str(i)] = dict(dtype='uint8', shape=[16], sha256=sha256(profile / raw))
    write(profile / 'plans/programs.json', programs)
    write(profile / 'plans/inputs.json', dict(records=[dict(index=0, input='inputs/000.bin', frontiers=frontiers)]))
    write(profile / 'plans/expected_outputs.json', expected)
    manifest = dict(network='resnet50_v2', unit_count=38, canonical_sha256='a' * 64,
                    runner_sha256=parent['runner_sha256'], files=inventory(profile))
    write(profile / 'manifest.json', dict(bundle_id=digest(manifest)[:24], **manifest))
    monkeypatch.setattr(local, 'validate_profile', lambda root: read(Path(root) / 'manifest.json'))
    controller = ROOT / 'cpu_policies/open_resources/domain.py'
    controls = dict(sources={str(controller): sha256(controller)}, protocol=protocol())
    write(output / 'CONTROLLERS.json', controls)
    monkeypatch.setattr(package, 'assert_controllers', lambda root: controls)
    lists, pool = selected_lists()
    ranking = output / 'ranking'
    documents = {'rankings.json': dict(lists=lists), 'pool.json': pool,
                 'scores.json': {}, 'execution_order.json': list(pool),
                 'protocol.json': protocol(),
                 'proof.json': dict(pipeline_FPS_used=False, exact_DP=True,
                     independent_global_enumeration=False,
                     small_domain_and_ResNet18_acceptance=dict(passed=True))}
    for name, value in documents.items():
        write(ranking / name, value)
    write(ranking / 'FREEZE.json', dict(files={name: sha256(ranking / name) for name in documents},
                                       candidate_count=len(pool), pipeline_FPS_used=False))
    (ranking / 'inputs').mkdir()
    write(ranking / 'inputs/test_sources.json', dict(synthetic=True))
    return output


def test_pipeline_capsule_is_portable_and_uses_only_selected_native_dependencies(prepared_fixture, tmp_path):
    output = prepared_fixture
    manifest = package.prepare(output)
    assert manifest['network'] == 'resnet50_v2' and manifest['unit_count'] == 38
    assert manifest['candidate_count'] == 20
    original = output / 'pipeline_bundle'
    assert len(read(original / 'plans/programs.json')) == 3
    moved = tmp_path / 'moved'
    shutil.copytree(original, moved)
    assert v.validate_bundle(moved)['bundle_id'] == manifest['bundle_id']
    assert read(moved / 'plans/qemu_loader.json')['network_inference_calls'] == 0
    assert read(moved / 'plans/ELF_DEPENDENCIES.json')['all_dependencies_bundled']
    assert package.prepare(output)['bundle_id'] == manifest['bundle_id']


def test_capsule_rejects_hash_changes(prepared_fixture):
    package.prepare(prepared_fixture)
    target = prepared_fixture / 'pipeline_bundle/programs/cpu_00_00/params.bin'
    target.write_bytes(b'changed')
    with pytest.raises(ValueError):
        v.validate_bundle(prepared_fixture / 'pipeline_bundle')


def reseal_fixture(root):
    value = read(root / 'manifest.json')
    value.pop('bundle_id')
    value['files'] = inventory(root)
    value['files'].pop('manifest.json')
    write(root / 'manifest.json', {**value, 'bundle_id': digest(value)[:24]}, replace=True)


def test_visible_allocator_overflow_stops_a_capsule_without_resizing(prepared_fixture):
    package.prepare(prepared_fixture)
    root = prepared_fixture / 'pipeline_bundle'
    path = root / 'programs/vta_01_01/graph.json'
    write(path, tiny_graph(size=110_000_000), replace=True)
    programs = read(root / 'plans/programs.json')
    programs['vta_01_01']['files']['graph.json'] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    write(root / 'plans/programs.json', programs, replace=True)
    write(root / 'programs/vta_01_01/record.json', programs['vta_01_01'], replace=True)
    memory = v.memory_report(root, read(root / 'plans/pool.json'), programs)
    write(root / 'plans/memory_estimate.json', memory, replace=True)
    reseal_fixture(root)
    with pytest.raises(MemoryError, match='192 MiB'):
        v.validate_bundle(root)
    assert read(root / 'plans/board_execution.json')['udmabuf_bytes'] == v.BUFFER_BYTES


def test_frontier_cannot_point_back_to_an_external_preparation_directory(prepared_fixture):
    package.prepare(prepared_fixture)
    root = prepared_fixture / 'pipeline_bundle'
    inputs = read(root / 'plans/inputs.json')
    inputs['records'][0]['frontiers']['0']['raw'] = str(prepared_fixture / 'profile_capsule/inputs/frontiers/000/0.bin')
    write(root / 'plans/inputs.json', inputs, replace=True)
    reseal_fixture(root)
    with pytest.raises(ValueError, match='nonportable'):
        v.validate_bundle(root)


def test_capsule_rejects_frozen_list_changes_before_copy(prepared_fixture):
    path = prepared_fixture / 'ranking/pool.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='rankings differ'):
        package.prepare(prepared_fixture)
    assert not (prepared_fixture / 'pipeline_bundle').exists()


def test_existing_capsule_cannot_be_reused_under_another_ranking_seal(prepared_fixture):
    package.prepare(prepared_fixture)
    path = prepared_fixture / 'ranking/FREEZE.json'
    seal = read(path)
    seal['frozen_at'] = 'different-freeze-identity'
    write(path, seal, replace=True)
    with pytest.raises(ValueError, match='different profile or frozen selection'):
        package.prepare(prepared_fixture)


def test_partial_capsule_is_not_silently_overwritten(prepared_fixture):
    path = prepared_fixture / 'pipeline_bundle'
    path.mkdir()
    (path / 'partial').write_text('preserve')
    with pytest.raises(FileExistsError, match='partial'):
        package.prepare(prepared_fixture)
    assert (path / 'partial').read_text() == 'preserve'


def test_zero_inference_loader_checks_the_actual_original_binary():
    if not (PARENT / 'manifest.json').is_file() or shutil.which('qemu-aarch64-static') is None:
        pytest.skip('requires the existing original runtime capsule')
    result = package._loader_check(PARENT, read(PARENT / 'manifest.json'))
    assert result['passed'] and result['network_inference_calls'] == 0
    assert set(result['supported_flags']) == set(v.REQUIRED_FLAGS)


def test_workflow_all_reports_pipeline_pending_without_inventing_rankings(tmp_path, monkeypatch):
    from resnet50_validation import workflow
    graph = ModuleType('resnet50_validation.graph')
    units = ModuleType('resnet50_validation.units')
    graph.prepare_graph = lambda output: dict(canonical_sha256='x')
    units.prepare_units = lambda source, output: dict(units=38, interval_checks=741)
    monkeypatch.setitem(sys.modules, graph.__name__, graph)
    monkeypatch.setitem(sys.modules, units.__name__, units)
    called = []
    def local_prepare(output, config, *, stage, **kwargs):
        called.append(stage)
        return dict(status='test')
    monkeypatch.setattr(local, 'prepare', local_prepare)
    result = workflow.prepare(tmp_path, stage='all')
    assert called == ['modules', 'references', 'profile']
    assert result['status'] == 'PROFILE_PREPARED_PIPELINE_PENDING' and result['pipeline_pending']
    assert not (tmp_path / 'CONTROLLERS.json').exists()


def test_workflow_profile_does_not_load_graph_or_tvm(tmp_path, monkeypatch):
    from resnet50_validation import workflow
    monkeypatch.setitem(sys.modules, 'resnet50_validation.graph', None)
    monkeypatch.setitem(sys.modules, 'resnet50_validation.units', None)
    monkeypatch.setattr(local, 'prepare', lambda output, config, *, stage: dict(stage=stage))
    result = workflow.prepare(tmp_path, stage='profile')
    assert result['stages'] == {'profile': {'stage': 'profile'}}


def test_graph_metadata_failure_does_not_silently_assume_zero_memory():
    graph = tiny_graph()
    graph['attrs']['shape'][1].pop()
    with pytest.raises(ValueError, match='lengths'):
        v.graph_memory(graph, device='vta')
