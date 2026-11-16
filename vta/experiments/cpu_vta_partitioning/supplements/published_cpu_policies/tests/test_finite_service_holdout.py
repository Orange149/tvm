from pathlib import Path
from copy import deepcopy
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parents[1]/'src')]
from cpu_policies.open_resources import domain
from prepare_finite_service_holdout import fingerprint


def config():
    return domain.candidate([domain.stage('cpu',0,3,4,[0,1,2,3]),
        domain.stage('vta',3,13,1,[0,1,2,3]),domain.stage('cpu',13,14,1,[0])])


def test_known_fingerprint_ignores_score_version_and_configuration_label():
    a=config();b=deepcopy(a);b['id']='different-old-label';b['score_version']='a'
    assert fingerprint(a)==fingerprint(b)


def test_physical_mapping_gate_and_host_context_are_distinct():
    a=config();b=deepcopy(a);b['stages'][-1]=domain.stage('cpu',13,14,1,[1])
    assert fingerprint(a)!=fingerprint(b)
    b=deepcopy(a);b['execution_mode']='resource_group'
    assert fingerprint(a)!=fingerprint(b)
    b=deepcopy(a);b['host_mask']=[3]
    assert fingerprint(a)!=fingerprint(b)


def test_components_and_noncanonical_shapes_are_not_false_full_configs():
    assert fingerprint(dict(stages=['not-a-stage'])) is None
    assert fingerprint(domain.component(domain.stage('cpu',0,3,4,[0,1,2,3]))) is None
