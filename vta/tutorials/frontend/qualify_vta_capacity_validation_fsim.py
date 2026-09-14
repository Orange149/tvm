"""Three-seed FSim of all successful additional capacity-validation candidates."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fsim-hw-path',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    from run_vta_p7r119_yolo_barrier_pilot import static_result,run_fsim
    candidates=[json.loads(s) for s in a.candidates.read_text().splitlines() if s.strip()]
    baseline={r['candidate_id']:r for r in map(json.loads,a.baseline.read_text().splitlines())}
    a.output.mkdir(parents=True)
    protocol={'scope':'additional same-geometry candidates; numerical FSim qualification only',
              'candidate_sha256':hashlib.sha256(a.candidates.read_bytes()).hexdigest(),
              'baseline_sha256':hashlib.sha256(a.baseline.read_bytes()).hexdigest(),
              'all_baseline_successes_required':True,'no_replacement':True,'board_contacted':False}
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    results=[]
    with (a.output/'results.jsonl').open('x') as stream:
        for c in candidates:
            old=baseline[c['candidate_id']]
            if old['status']!='ok': continue
            static=static_result(c)
            if static['status']!='ok' or static['tir_sha256']!=old['tir_sha256']:
                raise RuntimeError('static identity drift: '+c['candidate_id'])
            row,stderr=run_fsim(c,static['tir_sha256'],static['sync']['residency_drains'],300,a.fsim_hw_path)
            # Discard simulator timing lines, retain numerical/error diagnostics.
            safe='\n'.join('[VTA_QUEUE] <parsed; timing discarded>' if '[VTA_QUEUE] ' in s else s
                           for s in stderr.splitlines())
            (a.output/(c['candidate_id']+'.log')).write_text(safe)
            result={'candidate_id':c['candidate_id'],'static':static,'fsim':row}
            stream.write(json.dumps(result)+'\n');stream.flush();results.append(result)
            print(c['family_id'],c['public_mode'],row['status'],flush=True)
    summary={'count':len(results),'fsim_passed':sum(r['fsim']['status']=='passed' for r in results),
             'board_contacted':False,'performance_labels_collected':False,
             'scope':protocol['scope']}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
