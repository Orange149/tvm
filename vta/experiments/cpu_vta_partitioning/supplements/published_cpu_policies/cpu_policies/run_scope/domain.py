"""Two frozen pipelines and four CPU components, without changing allocation."""
from copy import deepcopy
from pathlib import Path

from .common import api
from ..diagnosis.domain import PROGRAMS, EXPECTED_IDS as _ALL_IDS
from ..diagnosis.domain import configurations as _diagnosis_configurations

EXPECTED_IDS = {name: _ALL_IDS[name] for name in ('static_first', 'history_host2')}


def configurations(programs=None):
    original = _diagnosis_configurations(programs)
    return {'components': deepcopy(original['components']),
            'pipelines': {name: deepcopy(original['pipelines'][name]) for name in EXPECTED_IDS}}


def cases(bundle=None):
    if bundle is None:
        return configurations()
    common = api(bundle)
    expected = configurations(common.read(Path(bundle) / 'plans/programs.json'))
    actual = common.read(Path(bundle) / 'plans/diagnosis_configurations.json')
    if actual != expected:
        raise ValueError('run-scope frozen cases differ')
    return actual
