from pathlib import Path
from statistics import median
from cpu_vta_partitioning.common import read,sha256
from .common import OUTPUT,freeze,protocol,sources
from .native import check
from .synthetic import mixed_actions,run as synthetic_run


def service_samples(rows,n):
    group=[r for r in rows if r['case']=='busy'+str(n)]
    if len(group)!=n:raise ValueError('busy calibration case missing')
    a=max(r['segments'][0]['wall_start_ns'] for r in group)
    b=min(r['segments'][0]['wall_end_ns'] for r in group)
    samples=[]
    for r in group:
        for left,right in zip(r['events'],r['events'][1:]):
            if left['direction']=='in' and right['direction']=='out' and right['preempt'] and a<=left['time_ns']<right['time_ns']<=b:
                samples.append(right['time_ns']-left['time_ns'])
    if len(samples)<protocol()['calibration_sample_minimum_per_case']:
        raise ValueError('insufficient interior preempted running spans')
    return samples


def calibrate(output=OUTPUT):
    output=Path(output);receipt=read(output/'PLATFORM_RECEIPT.json')
    if receipt['sources']!=sources() or sha256(receipt['raw'])!=receipt['raw_sha256']:
        raise ValueError('finite platform provenance differs')
    data=check(read(receipt['raw']));rows=[r for r in data['streams'] if r['case'].startswith('busy')]
    samples={str(n):service_samples(rows,n) for n in range(2,7)}
    granules={n:int(median(v)) for n,v in samples.items()}
    for mode in (False,True):
        pred=synthetic_run(mixed_actions(),granules,fluid=mode)
        freeze(output/('MIXED_'+pred['model'].upper()+'_PREDICTION.json'),pred)
    payload=dict(quanta_ns=granules,samples_ns=samples,source_receipt_sha256=sha256(output/'PLATFORM_RECEIPT.json'),
        scope='observed finite CPU service; not a recovered kernel tunable',
        heldout_case_used_for_parameter_choice=False,DNN_values_used=False,protocol=protocol(),sources=sources())
    freeze(output/'CALIBRATION.json',payload)
    return dict(quanta_ns=granules,heldout_predictions_frozen=True,DNN_values_used=False)


def validate_mixed(output=OUTPUT):
    output=Path(output);cal=read(output/'CALIBRATION.json');receipt=read(output/'PLATFORM_RECEIPT.json')
    if sha256(output/'PLATFORM_RECEIPT.json')!=cal['source_receipt_sha256'] or cal['sources']!=sources():
        raise ValueError('finite calibration identity differs')
    group=[r for r in receipt['validation']['streams'] if r['case']=='mixed']
    observed={str(r['index']):r['segments'][-1]['wall_end_ns']-r['segments'][0]['wall_start_ns'] for r in group}
    granule=max(cal['quanta_ns'].values());results=[]
    for mode in ('FINITE_VIRTUAL_RUNTIME','FLUID'):
        pred=read(output/('MIXED_'+mode+'_PREDICTION.json'));errors=[]
        for actor,w in observed.items():
            err=abs(pred['wall_ns'][actor]-w);allowed=max(w/5,granule)
            errors.append(dict(actor=int(actor),predicted_wall_ns=pred['wall_ns'][actor],observed_wall_ns=w,
                absolute_error_ns=err,allowed_error_ns=allowed,passed=err<=allowed))
        results.append(dict(model=mode,actors=errors,passed=all(x['passed'] for x in errors)))
    out=dict(results=results,finite_service_platform_check_passed=results[0]['passed'],
        no_DNN_model_validity_claim=True,calibration_sha256=sha256(output/'CALIBRATION.json'),
        no_repeat_or_fitted_adjustment=True)
    freeze(output/'MIXED_VALIDATION.json',out);return out
