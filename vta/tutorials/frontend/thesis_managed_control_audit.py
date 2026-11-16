#!/usr/bin/env python3
"""Audit real host slot transitions and FIFO traces; export a sourced host Gantt."""
import argparse
import json
from pathlib import Path
from thesis_supplement_run import sha256,write_new_json
from thesis_supplement_audit import reconcile


def verify(directory):
    manifest=json.loads((directory/'artifact_hashes.json').read_text())
    files={str(p.relative_to(directory)) for p in directory.rglob('*')
           if p.is_file() and p.name!='artifact_hashes.json'}
    if files!=set(manifest):raise ValueError('Sealed run inventory drift')
    for relative,expected in manifest.items():
        p=directory/relative
        if sha256(p)!=expected['sha256'] or p.stat().st_size!=expected['bytes']:
            raise ValueError('Sealed artifact drift')
    cost=reconcile(directory)
    if cost!=json.loads((directory/'analysis/cost_audit.json').read_text()) or cost['unfinished_action_ids']:
        raise ValueError('Cost ledger drift')
    return cost


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-run',type=Path,required=True)
    parser.add_argument('--control-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    costs=[verify(args.core_run),verify(args.control_run)]
    actions=list(args.core_run.glob('raw/run_host_lifecycle*.json'))
    if len(actions)!=1:raise ValueError('Wrong source action count')
    action=json.loads(actions[0].read_text())
    events=[json.loads(line) for line in (args.core_run/action['stdout']).read_text().splitlines()]
    states={};aborted={};checked=0;gantt=[]
    for e in events:
        case=e['case']
        if not e['event'].startswith('production_'):continue
        point=e['event'][len('production_'):]
        edge,slot,generation,frame=map(int,e['detail'].split(','))
        if point=='aborted':aborted[(case,edge)]=True;continue
        if point in ('producer_wait_free','consumer_wait_ready'):
            if point=='producer_wait_free':
                if any(v['state']=='FREE' for (c,ed,_),v in states.items() if c==case and ed==edge):
                    raise ValueError('Producer reported wait with an observed FREE slot')
            continue
        key=(case,edge,slot)
        if point=='slot_initialized':
            if key in states:raise ValueError('Slot initialized twice')
            states[key]={'state':'FREE','generation':0,'frame':-1};continue
        prior=states[key]
        required={'producer_acquired':'FREE','published':'WRITING','consumer_acquired':'READY','released':'USING'}[point]
        if prior['state']!=required:raise ValueError('Illegal observed production transition')
        if point=='producer_acquired':
            if generation!=prior['generation']+1 or aborted.get((case,edge)):
                raise ValueError('Wrong generation or acquire after abort')
        elif generation!=prior['generation'] or frame!=prior['frame']:
            raise ValueError('Observed token identity mismatch')
        if point=='published' and aborted.get((case,edge)):raise ValueError('Observed publish after abort')
        new={'producer_acquired':'WRITING','published':'READY','consumer_acquired':'USING','released':'FREE'}[point]
        states[key]={'state':new,'generation':generation,'frame':frame};checked+=1
        if case=='full_slots2_0':gantt.append({'slot':slot,'frame':frame,'state':new,'ns':e['host_monotonic_ns']})
    if checked!=5720:raise ValueError('Unexpected observed manager transition count')
    control_summary=json.loads((args.control_run/'summary.json').read_text())
    if control_summary['control_cases_passed']!=160 or control_summary['control_cases_failed']:
        raise ValueError('Control cases not all passed')
    for r in json.loads((args.control_run/'analysis/native_trace_audit.json').read_text()):
        records=[json.loads(line) for line in (args.control_run/r['path']).read_text().splitlines()]
        if len(records)!=r['events']:raise ValueError('Native trace denominator drift')
        if Path(r['path']).stem.startswith('fifo_release'):
            points=[x['point'] for x in records]
            if points.index('latch_entered')>=points.index('latch_released') or points.index('latch_released')>=points.index('latch_resumed'):
                raise ValueError('FIFO release happened outside real entered→release→resume order')
    write_new_json(args.output/'analysis/production_transition_audit.json',{
        'observed_transitions_verified':checked,'all_token_generations_and_abort_rules_passed':True,
        'source':'actual production observer events; audit rules are not replacement manager code'})
    write_new_json(args.output/'analysis/host_slot_gantt_source.json',gantt)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'WRITING':'tab:orange','READY':'tab:blue','USING':'tab:green','FREE':'lightgray'}
    fig,ax=plt.subplots(figsize=(8,2.8),layout='constrained')
    start=min(r['ns'] for r in gantt); end=max(r['ns'] for r in gantt)
    for slot in (0,1):
        rows=[r for r in gantt if r['slot']==slot]
        for index,r in enumerate(rows):
            stop=rows[index+1]['ns'] if index+1<len(rows) else end
            if stop==r['ns']:continue
            ax.broken_barh([((r['ns']-start)/1000,(stop-r['ns'])/1000)],(slot-.3,.6),facecolors=colors[r['state']])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c,label=s) for s,c in colors.items()],ncol=4,loc='upper center')
    ax.set_yticks([0,1],['slot0','slot1']);ax.set_xlabel('Host observer time since first acquire (µs)')
    ax.set_title('Actual production manager / CPU vectors: 3 frames, 2 slots')
    for extension in ('svg','pdf'):fig.savefig(args.output/'figures'/('host_production_slot_gantt.'+extension))
    plt.close(fig)
    write_new_json(args.output/'analysis/source_costs.json',{
        'core_regression_action_seconds':costs[0]['measured_action_total_seconds'],
        'control_test_action_seconds':costs[1]['measured_action_total_seconds'],
        'charged_again_to_current_audit':False})
    print(json.dumps({'production_transitions':checked,'control_cases':160,'formal_phase03_completed':False}))


if __name__=='__main__':main()
