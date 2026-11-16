from pathlib import Path
from cpu_vta_partitioning.common import read,sha256,write,now,inventory
from workload_model.common import ROOT

OUTPUT=ROOT/'artifacts/placement_alignment_01'
PARENT=ROOT/'artifacts/scheduler_gap_audit_01'
SEMANTICS=ROOT/'artifacts/model_semantics_audit_01'
REPORT=ROOT.parents[4]/'paper/my_paper/analysis/20261110_placement_alignment'


def sources():
    return {str(p):sha256(p) for p in [ROOT/'run_placement_alignment.py',*sorted(Path(__file__).parent.glob('*.py')),
        *sorted(Path(__file__).parent.glob('*.cc')),ROOT/'tests/test_placement_alignment.py']}
