"""Small frozen experiment, independent of all archived policy controllers."""
from pathlib import Path
from cpu_vta_partitioning.common import read,sha256
from yolo_validation.common import ROOT,PROFILE,LOCAL_CONFIG,frozen

PARENT=ROOT/'artifacts/yolo_cpu_policy_check_01/pipeline_bundle'
OUTPUT=ROOT/'artifacts/yolo_wait_diagnosis_01'
RUN=ROOT/'runs/yolov3_tiny416_wait_diagnosis_01'
BUILD_RECORD=ROOT/'artifacts/bundles/27b583640470adcc2bbd2194/plans/runtime_build.json'
IDS={'overlap4':'cfg_1e2e8cbca42947bbddd3030e','split3':'cfg_f8a1bc12c610f39904cc1231'}


def protocol():
    return dict(experiment='yolov3_tiny416_wait_diagnosis_01',revision=1,seed=20261109,input_index=0,
        qualification=dict(warmup=0,scored=1,tail=0),component=dict(warmup=2,scored=8,tail=0),
        performance=dict(warmup=8,scored=64,tail=32),queue_depth=2,
        anchors=False,drift_rejection=False,automatic_remeasurement=False,automatic_reboot=False,
        unchanged_scores=True,unchanged_runtime_and_runner=True,max_native_sessions=24,
        max_DNN_inferences=840,max_component_calls=44,
        blocking_trigger=dict(reproduced_positive_slowdown=True,observed_target_yields=True,
                              maximum_probe_perturbation_fraction_of_gap=.2),
        trace_scope='whole native session, not the 64 scored timestamps',
        nonrunning_scope='wall minus thread CPU; scheduling/blocking/other off-CPU not separated',
        kernel_schedstats=False,kernel_ftrace=False)


def configurations():
    from cpu_policies.open_resources.domain import component
    parent=read(PARENT/'plans/pool.json');pipes={k:parent[i] for k,i in IDS.items()}
    if any(c['execution_mode']!='off' or c['host_mask']!=[0,1,2,3] or c['control_mask']!=[0,1,2,3]
           for c in pipes.values()):raise ValueError('fixed environment changed')
    components={k:component(c['stages'][0],'shared4') for k,c in pipes.items()}
    return dict(pipelines=pipes,components=components)


def sources():
    from yolo_validation.common import source_identity
    import cpu_vta_partitioning
    files=source_identity()
    paths=[ROOT/'run_yolo_wait_diagnosis.py',ROOT/'run_yolo_validation_gate_compat.py',
           *sorted(Path(__file__).parent.glob('*.py')),*sorted(Path(__file__).parent.glob('*.cc'))]
    files.update({str(p.relative_to(ROOT)):sha256(p) for p in paths})
    base=Path(cpu_vta_partitioning.__file__).parent
    files.update({'@base/'+str(p.relative_to(base)):sha256(p) for p in sorted(base.rglob('*.py'))})
    return files


def assert_prepared(output):
    p=read(Path(output)/'PREPARATION.json')
    if p['sources']!=sources() or p['protocol']!=protocol():raise ValueError('wait controller changed; use a new batch')
    for path,value in p['source_files'].items():
        if sha256(path)!=value:raise ValueError('archived source input changed')
    return p
