"""Check isolated service reconstruction without reading pipeline outcomes."""
from pathlib import Path
import json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from cpu_vta_partitioning.common import read,write,sha256
from finite_service.common import freeze
from device_completion_replay import CompletionFinite
from policy_branch_eval.common import isolated_inputs_only
from scheduler_gap_audit.curves import csv_write
import run_device_completion_replay as w


def audit():
    out=w.OUTPUT/'isolated_reconstruction';inputs=read(w.OUTPUT/'INPUTS.json');profiles=read(w.PROFILE)
    stages={}
    for item in inputs['cases'].values():
        for cfg,s in zip(item['configuration']['stages'],item['stages']):
            if cfg['device']=='vta':stages[item['network']+'/'+cfg['program']]=s
    predictions={};start=time.monotonic()
    for key,stage in stages.items():
        predictions[key]={}
        for model in ('aggregate','independent'):
            with isolated_inputs_only():
                r=CompletionFinite([stage],quanta=inputs['quanta_ns'],placement='wake_boundary',
                    completion_model=model,completion_point='midpoint',frames=10,warmup=2,scored=8,
                    queue_depth=2,runnable_completion_wait=True,controls=None,gate='off',host_order='before').run()
            predictions[key][model]=r['replay']
    freeze(out/'PREDICTION_FREEZE.json',dict(predictions=predictions,input_sha256=sha256(w.OUTPUT/'INPUTS.json'),
        source_sha256=sha256(__file__),no_pipeline_performance_read=True))
    rows=[]
    for key,values in predictions.items():
        native=profiles['profiles'][key]['clean_cost']['service_ms']
        for model,r in values.items():
            pred=r['stage_service_ns'][0]/1e6
            rows.append(dict(component=key,model=model,predicted_service_ms=pred,
                isolated_clean_service_ms=native,relative_error_percent=100*(pred/native-1)))
    write(out/'RESULTS.json',dict(rows=rows,seconds=time.monotonic()-start,
        no_pipeline_performance_read=True,not_an_independent_profile_prediction_test=True),replace=True)
    csv_write(out/'comparison.csv',rows)
    return rows

if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False,indent=2))
