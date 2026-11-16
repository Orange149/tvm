from pathlib import Path
import pytest
from cpu_vta_partitioning.common import read
from cpu_vta_partitioning.domain import signatures
from cpu_vta_partitioning.models import Cost


@pytest.fixture
def programs():
    return {p['name']:p for p in read(Path(__file__).parent/'fixtures/small_domain.json')}


def synthetic_costs(programs, cores=4):
    return {key:Cost(1 + (s.end-s.begin)/(s.threads if s.device=='cpu' else 5),
                     (s.end-s.begin) * (1 if s.device=='cpu' else .3), .2 + s.begin/100)
            for key,s in signatures(programs,cores).items()}


@pytest.fixture
def costs(programs):
    return synthetic_costs(programs)
