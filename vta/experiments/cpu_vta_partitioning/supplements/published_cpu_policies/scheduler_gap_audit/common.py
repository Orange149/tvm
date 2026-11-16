"""New batch boundaries. Never write into an earlier study or capsule."""
from pathlib import Path
from cpu_vta_partitioning.common import digest, inventory, now, read, sha256, write
from workload_model.common import ROOT, OUTPUT as PARENT

OUTPUT = ROOT / 'artifacts/scheduler_gap_audit_01'
RUN = ROOT / 'runs/cross_network_scheduler_gap_01'
REPORT = ROOT.parents[4] / 'paper/my_paper/analysis/20261110_scheduler_gap'
CONFIG = PARENT / 'board.local.json'


def protocol():
    return dict(experiment='cross_network_scheduler_gap_01', revision=1, input_index=0,
                seed=20261110, qualification=dict(warmup=0, scored=1, tail=0),
                component=dict(warmup=2, scored=8, tail=0),
                pipeline=dict(warmup=8, scored=64, tail=32), max_native_sessions=18,
                max_full_DNN_inferences=627, max_component_calls=63,
                no_reboot=True, no_score_change=True, no_domain_pruning=True,
                original_wait=True, CPU_gate='off', automatic_remeasurement=False,
                probe_attribution_fraction=0.2, startup_calls_counted_separately=True)


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('frozen scheduler evidence changed: ' + str(path))
    else:
        write(path, value)
    return value


def sources():
    paths = [ROOT/'run_scheduler_gap_audit.py', *sorted(Path(__file__).parent.glob('*.py')),
             *sorted(Path(__file__).parent.glob('*.cc')), *sorted(Path(__file__).parent.glob('*.h'))]
    return {str(p): sha256(p) for p in paths}
