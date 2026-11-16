"""Reuse the existing CPU-only mixed-task record, without rerunning native work."""
from pathlib import Path
from types import FunctionType
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,sha256
from finite_service.common import freeze
from finite_service.synthetic import run,mixed_actions
from compiled_wakeup_model import KernelWakeScheduler

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/compiled_wakeup_01'


def validate():
    source=ROOT/'artifacts/finite_service_01';cal=read(source/'CALIBRATION.json')
    p=ROOT/'artifacts/three_thread_worker_gap_01/kernel_parameter_sources/COMPILED_WAKEUP.json'
    gran=read(p)['wakeup_granularity_ns']
    f=FunctionType(run.__code__,{**run.__globals__,'Scheduler':lambda q:KernelWakeScheduler(q,gran)},run.__name__,run.__defaults__)
    predicted=f(mixed_actions(),cal['quanta_ns'],fluid=False)
    freeze(OUT/'PLATFORM_PREDICTION.json',dict(prediction=predicted,parameter_sha256=sha256(p),
        calibration_sha256=sha256(source/'CALIBRATION.json'),source_sha256=sha256(__file__),
        no_new_native_work=True,parameter_not_chosen_from_heldout=True))
    old=read(source/'MIXED_VALIDATION.json')
    if old['calibration_sha256']!=sha256(source/'CALIBRATION.json'):raise ValueError('original platform proof differs')
    rows=[]
    for r in old['results'][0]['actors']:
        new=predicted['wall_ns'][str(r['actor'])];err=abs(new-r['observed_wall_ns'])
        rows.append(dict(actor=r['actor'],observed_wall_ns=r['observed_wall_ns'],old_wall_ns=r['predicted_wall_ns'],
            new_wall_ns=new,old_error_ns=r['absolute_error_ns'],new_error_ns=err,
            original_allowed_error_ns=r['allowed_error_ns'],passed=err<=r['allowed_error_ns']))
    x=dict(rows=rows,passed=all(r['passed'] for r in rows),source_sha256=sha256(__file__),new_native_sessions=0,
        native_reference_sha256=sha256(source/'MIXED_VALIDATION.json'),
        no_DNN_FPS_used=True,no_general_DNN_validity_claim=True)
    freeze(OUT/'PLATFORM_REGRESSION.json',x);return x


if __name__=='__main__':print(json.dumps(validate(),ensure_ascii=False,indent=2))
