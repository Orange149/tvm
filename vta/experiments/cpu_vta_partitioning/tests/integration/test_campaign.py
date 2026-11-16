import json
from pathlib import Path
import pytest
from cpu_vta_partitioning.common import write,read
from cpu_vta_partitioning.campaign import create,run_current,controller_identity


class SyntheticBoard:
    def __init__(self):
        self.boot='00000000-0000-0000-0000-000000000001'
        self.executed=set()
    def boot_id(self): return self.boot
    def deploy(self,bundle,root): return '/fake/bundle'
    def qualify_environment(self,*args): pass
    def measure(self,document,bundle,remote,root,boot):
        assert boot==self.boot
        self.executed.add((boot,document['task_id']))
        path=root/'raw'/boot/document['task_id']/'collected';path.mkdir(parents=True,exist_ok=True)
        n=document['warmup']+document['scored']+document['tail']
        rows=[]
        for i in range(n):
            row={'frame_id':i,'input_index':i%len(document['input_indices']),'completion_index':i,'completion_ms':10.*(i+1),'total_latency_ms':20.,
                 'stage_cpu_time_scope':'exclusive_process_cpu_time','stage0_set_ms':1.,'stage0_run_ms':2.,'stage0_get_ms':1.,
                 'stage0_set_process_cpu_ms':.1,'stage0_run_process_cpu_ms':1.,'stage0_get_process_cpu_ms':.1}
            rows.append(row)
        (path/'result.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
        write(path/'affinity.json',{'stages':[{'requested_cpus':s['mask'],'thread_ids':list(range(s['threads'])),
              'active_runtime_threads':s['threads'],'effective_affinities':[{'tid':i,'cpus':s['mask']} for i in range(s['threads'])]}
              for s in document['stages']]},replace=True)
        if document.get('detailed'): write(path/'profile.json',{'synthetic':True},replace=True)
        return path


def test_complete_two_phase_campaign_with_ten_distinct_boots(tmp_path,programs,monkeypatch):
    import cpu_vta_partitioning.campaign as campaign
    import cpu_vta_partitioning.prepare.package as package
    bundle=tmp_path/'bundle';bundle.mkdir()
    write(bundle/'manifest.json',{'bundle_id':'synthetic','controller_identity':controller_identity()})
    write(bundle/'plans/programs.json',programs)
    write(bundle/'plans/protocol.json',{'environment':{'cores':4,'poll_sleep_ns':1000,'post_start_sleep_ns':1000},'seed':13})
    write(bundle/'plans/sensitivity.json',{'cores':[2,4],'poll_sleep_ns':[0,1000,100000],'post_start_sleep_ns':1000})
    monkeypatch.setattr(package,'validate',lambda _:None)
    monkeypatch.setattr(campaign,'correctness',lambda *args:True)  # numerical correctness is separately tested
    root=tmp_path/'campaign';create(bundle,root,{})
    board=SyntheticBoard()
    for i in range(10):
        board.boot=f'00000000-0000-0000-0000-{i+1:012}'
        state=run_current(root,board)
        count=len(board.executed)
        assert run_current(root,board)['status']==state['status']
        assert len(board.executed)==count  # same boot cannot become a new replicate
    assert state['status']=='COMPLETE' and len(state['completed_boots'])==10
    analysis=read(root/'env0/analysis.json')
    assert not analysis['continue_sensitivity']
    assert all(row['n_boots']==6 for row in analysis['comparisons'].values())
    assert len(list((root/'env0/confirmation').glob('boot*/round*.json')))==12
    from cpu_vta_partitioning.analysis.export import export
    result=export(root)
    assert result['environments']['env0']['status']=='ANALYZED'
    assert (root/'env0/performance.pdf').is_file()
    assert (root/'env0/host_work_ablation.csv').is_file()


def test_code_change_blocks_resume(tmp_path,monkeypatch):
    import cpu_vta_partitioning.campaign as campaign
    root=tmp_path/'run';root.mkdir()
    write(root/'state.json',{'bundle':'/unused','controller_identity':'old'})
    with pytest.raises(ValueError,match='controller changed'):
        run_current(root,SyntheticBoard())
