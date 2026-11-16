"""An x86 reference can pass shape checks while failing ARM quantization boundaries."""
from pathlib import Path
import shutil

import pytest

from cpu_vta_partitioning.common import digest, read, sha256, write
from cpu_vta_partitioning.prepare.references import (
    TARGET, PROOF_STATUS, freeze_reference_source, validate_frozen_reference)


def sealed(path, body):
    body = {k: v for k, v in body.items() if k != 'artifact_sha256'}
    write(path, {**body, 'artifact_sha256': digest(body)}, replace=True)


@pytest.fixture
def reference_source(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    graph = source/'graph.json'; graph.write_text('canonical graph')
    program = source/'independent.so'; program.write_bytes(b'independent ARM program')
    records = [{'id':str(i), 'input':{'sha256':str(i)}} for i in range(67)]
    prior = source/'prior.json'
    original = {'canonical_qir_sha256':sha256(graph), 'real_inputs':records[:64],
                'diagnostic_inputs':records[64:]}
    sealed(prior, original)
    manifest = source/'inputs_manifest.json'
    sealed(manifest, {**original, 'reference_execution_target':TARGET,
                     'inputs_changed':False, 'extra_quantize_calls':0,
                     'prior_x86_reference_manifest':{'path':str(prior),'sha256':sha256(prior)},
                     'reference_artifacts':[{'path':str(program),'sha256':sha256(program)}]})
    proof = source/'reference_proof.json'
    def prove():
        sealed(proof, {'status':PROOF_STATUS, 'new_performance_windows':0,
                       'inputs_manifest':{'path':str(manifest),'sha256':sha256(manifest)},
                       'sessions':{'complete_graph':{},'complete_frontier_probe':{}}})
    prove()
    workspace = tmp_path/'workspace'
    (workspace/'sources').mkdir(parents=True)
    shutil.copy2(graph, workspace/'sources/canonical_qir.json')
    return manifest, workspace, sha256(graph), prove


def test_arm_proof_is_frozen_and_validation_survives_source_removal(reference_source, tmp_path):
    manifest, workspace, graph, _ = reference_source
    _, identity = freeze_reference_source(manifest, workspace, graph)
    relocated = tmp_path/'relocated'; shutil.copytree(workspace, relocated)
    shutil.rmtree(manifest.parent)
    assert validate_frozen_reference(relocated) == identity
    assert identity['historical_performance_used'] is False


@pytest.mark.parametrize('change', ['x86', 'graph', 'input', 'artifact', 'proof'])
def test_wrong_reference_source_cannot_be_prepared(reference_source, change):
    manifest, workspace, graph, prove = reference_source
    data = read(manifest)
    if change == 'x86':
        data.pop('reference_execution_target')
    elif change == 'graph':
        data['canonical_qir_sha256'] = 'different graph'
    elif change == 'input':
        data['real_inputs'][1]['input']['sha256'] = 'changed input'
    elif change == 'artifact':
        Path(data['reference_artifacts'][0]['path']).write_bytes(b'changed program')
    sealed(manifest, data); prove()
    if change == 'proof':
        proof = manifest.with_name('reference_proof.json')
        doc = read(proof); doc['sessions'].pop('complete_graph'); sealed(proof, doc)
    with pytest.raises(ValueError):
        freeze_reference_source(manifest, workspace, graph)


def test_modified_frozen_proof_fails_validation(reference_source):
    manifest, workspace, graph, _ = reference_source
    freeze_reference_source(manifest, workspace, graph)
    (workspace/'sources/reference/reference_proof.json').write_text('{}')
    with pytest.raises(ValueError, match='document hash'):
        validate_frozen_reference(workspace)
