"""Independent fixed-pool CPU-only versus all-stage admission experiment."""
import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import statistics as st
import uuid

import run_cpu_vta_residual_validation as native
import run_cpu_stage_overlap_validation as prior
from analyze_cpu_vta_all_stage_gate_validation import analyze, effect, repeated
from cpu_vta_residual_protocol import ROOT, matrix, verify_assets
from thesis_supplement_run import RunStore, sha256, utc, write_new_json

SOURCE = ROOT/'vta/tutorials/frontend/report_out/stage_tile_cotuning/cpu_stage_overlap_validation/20260916T153204Z_overlap_v1'
VERSION = 'cpu_vta_all_stage_admission_v1'
CONTROLS = (Path(__file__).name, 'analyze_cpu_vta_all_stage_gate_validation.py',
            'validate_cpu_vta_all_stage_gate_host.py', 'test_cpu_vta_all_stage_gate_validation.py',
            'analyze_cpu_stage_overlap_validation.py', 'test_cpu_stage_gate.cc') + native.CONTROL_FILES


def read(path):
    return json.loads(Path(path).read_text())


def derive_runner(source):
    """Derive from immutable prior source, leaving canonical/old controls intact."""
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError('Runner patch anchor drift: '+old[:70])
        source = source.replace(old, new)
    replace('  std::string cpu_stage_gate = "off";', '  std::string cpu_stage_gate = "off";\n  std::string stage_gate_scope = "cpu";')
    replace('    } else if (key == "--cpu-stage-gate") {', '    } else if (key == "--stage-gate-scope") {\n      args.stage_gate_scope = need_value(key);\n    } else if (key == "--cpu-stage-gate") {')
    replace('  if (args.cpu_stage_gate != "off" && args.cpu_stage_gate != "observe" &&',
            '  if (args.stage_gate_scope != "cpu" && args.stage_gate_scope != "all")\n    throw std::runtime_error("--stage-gate-scope must be cpu or all");\n  if (args.cpu_stage_gate != "off" && args.cpu_stage_gate != "observe" &&')
    replace('       << ",\\"cpu_stage_gate_scope\\":\\"CPU set/run/get and output snapshot; wait outside stage service\\"";',
            '       << ",\\"stage_gate_scope\\":\\"" << args.stage_gate_scope << "\\""\n'
            '       << ",\\"cpu_stage_gate_scope\\":\\"" << args.stage_gate_scope\n'
            '       << " set/run/get and output snapshot; wait outside stage service\\"";')
    replace('                if (stages[stage_index].device == "vta") {\n                  StageTiming& timing = frame->stage_timings[stage_index];', '''                auto& timing = frame->stage_timings[stage_index];
                const bool gated = args.cpu_stage_gate != "off" &&
                    (args.stage_gate_scope == "all" || stages[stage_index].device != "vta");
                std::unique_ptr<vta_deploy::CpuStageGate::Lease> admission_lease;
                if (gated) {
                  timing.cpu_gate_timing_valid = true;
                  timing.cpu_gate_request_ms = NowMillis();
                  if (args.cpu_stage_gate == "exclusive") admission_lease = cpu_stage_gate.Acquire();
                  timing.cpu_gate_acquire_ms = NowMillis();
                }
                if (stages[stage_index].device == "vta") {''')
    replace('''                  auto& timing = frame->stage_timings[stage_index];
                  std::unique_ptr<vta_deploy::CpuStageGate::Lease> cpu_gate_lease;
                  if (args.cpu_stage_gate != "off") {
                    timing.cpu_gate_timing_valid = true;
                    timing.cpu_gate_request_ms = NowMillis();
                    if (args.cpu_stage_gate == "exclusive")
                      cpu_gate_lease = cpu_stage_gate.Acquire();
                    timing.cpu_gate_acquire_ms = NowMillis();
                  }
''', '')
    replace('''                  if (args.cpu_stage_gate != "off") {
                    timing.cpu_gate_release_ms = NowMillis();
                    cpu_gate_lease.reset();
                  }
''', '')
    replace('''                }
              }
              if (stage_index + 1 == stages.size()) {''', '''                }
                if (gated) {
                  timing.cpu_gate_release_ms = NowMillis();
                  admission_lease.reset();
                }
              }
              if (stage_index + 1 == stages.size()) {''')
    return source


def freeze(directory):
    old = RunStore(SOURCE)
    verify_assets(old)
    if read(SOURCE/'analysis/FINAL_DELIVERY_STATUS.json')['independent_boots'] != 3:
        raise ValueError('Completed three-boot CPU-only source required')
    keys = ('assets','paths','programs','topologies','udmabuf_bytes','numeric','numeric_input',
            'source_identity','canonical_qir_sha256','bitstream_sha256','completion','board_state')
    c = {k:copy.deepcopy(old.contract[k]) for k in keys}
    c['assets'] = {k:v for k,v in c['assets'].items() if not k.startswith(('raw/controller_sources/','raw/runner_sources/'))}
    for k,v in c['assets'].items():
        v['source_path'] = str(SOURCE/k)
    baseline=SOURCE/'raw/host_validation/cpu_stage_gate_host_runner'
    c['assets']['raw/host_baseline_runner']={'source_path':str(baseline),'sha256':sha256(baseline),'bytes':baseline.stat().st_size}
    for name in prior.SOURCE_NAMES + ('picojson.h',):
        path = SOURCE/'raw/runner_sources'/name
        c['assets']['raw/runner_sources/'+name] = {'source_path':str(path),'sha256':sha256(path),'bytes':path.stat().st_size}
    derived = derive_runner((SOURCE/'raw/runner_sources/vta_stage_pipeline_runner.cc').read_text())
    record = c['assets']['raw/runner_sources/vta_stage_pipeline_runner.cc']
    record.update(derived_from_sha256=record['sha256'], sha256=hashlib.sha256(derived.encode()).hexdigest(), bytes=len(derived.encode()))
    for name in CONTROLS:
        path = Path(__file__).with_name(name)
        c['assets']['raw/controller_sources/'+name] = {'source_path':str(path.resolve()),'sha256':sha256(path),'bytes':path.stat().st_size}
    c.update(protocol_version=VERSION,run_id=Path(directory).name,stage='cpu_vta_all_stage_admission',
             campaign_id='all_stage_'+uuid.uuid4().hex[:12],created_at=utc(),prior_completed_run=str(SOURCE),
             main_topology='03:17',threads=[4,1,4],affinity=[[0,1,2,3]]*3,spin_count=0,queue_depth=2,
             actual_policy='ordinary_copy',actual_fpga_frequency='unknown',formal_static_score_changed=False,
             interventions={'A':'normal overlap','B':'CPU-only whole-stage FIFO admission','C':'all CPU/VTA whole-stage FIFO admission'},
             orders=[['A','B','C'],['C','B','A'],['B','C','A']],replication_orders=[['B','C'],['C','B']],
             calibration_orders=[['off','observe'],['observe','off']],
             primary='B-to-C front CPU mean run reduction; tail/VTA/II secondary',
             trigger='median front reduction >=5%; B CPU/VTA overlap positive; C all leases disjoint; serial front drift <=5%',
             interpretation='No throughput-improvement requirement. C changes resource constraints. Pool path and hardware/cache effects remain conditional.',
             performance_window={'runs':307,'warmup':5,'settling':2,'completions':300,'gaps':299},
             inference='Three independent kernel boot paired means; Student-t df2. Pilot alone exploratory.',
             stop_rule='Correctness/timeout/identity/order failure: preserve partial evidence and stop; no silent retries')
    s = RunStore.create(directory,c)
    for k,v in c['assets'].items():
        target=s.directory/k;target.parent.mkdir(parents=True,exist_ok=True)
        if 'derived_from_sha256' in v:
            with target.open('x') as f:f.write(derived)
        else:
            shutil.copy2(v['source_path'],target)
        if sha256(target)!=v['sha256']:
            raise ValueError('Copy drift '+k)
    bindings={name:{'host_path':str(s.directory/'raw/runtime'/name),'sha256':c['assets']['raw/runtime/'+name]['sha256'],
                    'bytes':c['assets']['raw/runtime/'+name]['bytes']} for name in ('libtvm_runtime.so','libvta.so','tvm_rpc','vta_hpc.bit','u-dma-buf.ko')}
    write_new_json(s.directory/'analysis/board_artifact_bindings.json',bindings)
    s.event('all_stage_protocol_frozen',parent_run=str(SOURCE))
    return s


def load(directory):
    s=RunStore(directory)
    if s.contract['protocol_version']!=VERSION:
        raise ValueError('Wrong protocol')
    verify_assets(s)
    for name in CONTROLS:
        v=s.contract['assets']['raw/controller_sources/'+name]
        if sha256(v['source_path'])!=v['sha256']:
            raise ValueError('Live controller drift '+name)
    return s


def build(s):
    path=s.directory/'analysis/runner_binding.json'
    if path.exists():
        b=read(path)
        if sha256(s.directory/b['path'])!=b['sha256']:
            raise ValueError('Runner drift')
        return b
    d=s.directory/'raw/build';d.mkdir(exist_ok=False)
    source=s.directory/'raw/runner_sources';output=d/'vta_stage_pipeline_runner_all_stage_gate'
    inc=[ROOT/p for p in ('include','3rdparty/dlpack/include','3rdparty/dmlc-core/include','3rdparty/vta-hw/include','vta/include')]
    script=d/'build.sh'
    with script.open('x') as f:
        f.write('unset LD_LIBRARY_PATH\nsource '+shlex.quote(native.SDK)+'\n'+
                'aarch64-xilinx-linux-g++ -std=c++17 -O2 --sysroot="$SDKTARGETSYSROOT" '+
                ' '.join('-I '+shlex.quote(str(p)) for p in inc)+' '+shlex.quote(str(source/'vta_stage_pipeline_runner.cc'))+
                ' -o '+shlex.quote(str(output))+' -L '+shlex.quote(str(s.directory/'raw/runtime'))+
                ' -Wl,-rpath-link,"$SDKTARGETSYSROOT/lib" -Wl,-rpath-link,"$SDKTARGETSYSROOT/usr/lib" -ltvm_runtime -lvta -ldl -pthread\n')
    native.checked(s,['bash',str(script)],'compile_frozen_all_stage_runner',300)
    b={'path':str(output.relative_to(s.directory)),'sha256':sha256(output),'runtime_bindings':read(s.directory/'analysis/board_artifact_bindings.json'),
       'defaults':{'cpu_stage_gate':'off','stage_gate_scope':'cpu'},'derived_source_sha256':sha256(source/'vta_stage_pipeline_runner.cc')}
    write_new_json(path,b)
    return b


@contextmanager
def adapters():
    old_build,old_argv=native.build_runner,native.runner_argv
    def argv(board,output,stages,condition,mode,runs,telemetry=False,profile=False):
        result=old_argv(board,output,stages,condition,mode,runs,telemetry,profile)
        if '--defer-result-write' not in result:result+=['--defer-result-write']
        if mode in ('pipeline','correctness_pipeline'):
            result+=['--cpu-stage-gate',condition['gate_mode'],'--stage-gate-scope',condition['gate_scope']]
        return result
    native.build_runner,native.runner_argv=build,argv
    try:yield
    finally:native.build_runner,native.runner_argv=old_build,old_argv


def session(s,boot,name,mode,block,control='observe'):
    load(s.directory)
    gate,scope={'A':(control,'all'),'B':('exclusive','cpu'),'C':('exclusive','all'),
                'off':('off','all'),'observe':('observe','all'),'serial':('off','cpu')}[name]
    c=copy.deepcopy(matrix()[0]);c.update(condition=name,gate_mode=gate,gate_scope=scope)
    r=native.session(s,boot,c,mode,block,telemetry=False)
    if mode in ('pipeline','correctness_pipeline'):
        rows=[json.loads(line) for line in (s.directory/r['raw_jsonl']).read_text().splitlines()]
        r['gate_metrics']=analyze(rows,gate,scope,7 if mode=='pipeline' else 0)
        if abs(r['gate_metrics']['ii_ms']-r['metrics'].get('ii_ms',r['gate_metrics']['ii_ms']))>1e-7:
            raise ValueError('Independent II mismatch')
    r.update(gate_mode=gate,gate_scope=scope)
    write_new_json(s.directory/'raw/sessions'/r['session_id']/'gate_summary.json',r)
    return r


def pairs(s,boot,control,prefix,orders):
    result=[]
    for i,order in enumerate(orders):
        arms={name:session(s,boot,name,'pipeline',prefix+str(i),control) for name in order}
        result.append({'boot_id':boot['boot_id'],'block':i,'cpu_only':arms['B'],'all_stages':arms['C'],
                       'effect':effect(arms['B'],arms['C']),**({'normal':arms['A']} if 'A' in arms else {})})
    return result


def execute(s,replicate=False):
    if (s.directory/'invalid_run.json').exists():raise ValueError('Failed protocol immutable')
    existing=[read(p) for p in (s.directory/'analysis').glob('boot_*.json')]
    if (not replicate and existing) or len(existing)>=3:raise ValueError('Boot count contract')
    if not read(s.directory/'analysis/host_validation.json')['passed']:raise ValueError('Host checks required')
    if replicate and not read(s.directory/'analysis/pilot.json')['replication_triggered']:raise ValueError('Pilot trigger required')
    boot=native.preflight(s)
    if boot['boot_id'] in {b['boot_id'] for b in existing}:raise ValueError('Actual kernel reboot required')
    boot['boot_index']=len(existing)+1
    with adapters():
        native.deploy(s,boot,build(s))
        try:
            session(s,boot,'serial','correctness_serial','qualification')
            control=read(s.directory/'analysis/pilot.json')['control_gate_mode'] if replicate else 'observe'
            for name in (('B','C') if replicate else ('A','B','C')):
                session(s,boot,name,'correctness_pipeline','qualification',control)
            before=session(s,boot,'serial','serial','matched_before')
            if not replicate:
                calibration=[]
                for i,order in enumerate(s.contract['calibration_orders']):
                    arms={n:session(s,boot,n,'pipeline','calibration'+str(i)) for n in order}
                    calibration.append(arms['observe']['metrics']['ii_ms']/arms['off']['metrics']['ii_ms']-1)
                control='observe' if max(map(abs,calibration))<=.02 else 'off'
                write_new_json(s.directory/'analysis/calibration.json',{'paired_ii_changes':calibration,'control_gate_mode':control,'threshold':.02})
            samples=pairs(s,boot,control,'replicate' if replicate else 'explore',s.contract['replication_orders'] if replicate else s.contract['orders'])
            after=session(s,boot,'serial','serial','matched_after')
            drift=after['metrics']['stages'][0]['run_ms']['mean']/before['metrics']['stages'][0]['run_ms']['mean']-1
            write_new_json(s.directory/f'analysis/serial_drift_boot_{boot["boot_index"]}.json',{'before':before,'after':after,'front_run_drift_fraction':drift})
            if replicate:
                write_new_json(s.directory/f'analysis/paired_boot_{boot["boot_index"]}.json',samples)
            else:
                signal=st.median(p['effect']['front_run_reduction_fraction'] for p in samples)
                triggered=signal>=.05 and all(p['effect']['trace_passed'] for p in samples) and abs(drift)<=.05
                write_new_json(s.directory/'analysis/pilot.json',{'control_gate_mode':control,'exploratory_pairs':samples,'front_reduction_fraction':signal,
                              'serial_drift_fraction':drift,'replication_triggered':triggered,'causal_status':'single-boot exploratory'})
                if triggered:
                    fresh=pairs(s,boot,control,'replicate',s.contract['replication_orders'])
                    write_new_json(s.directory/'analysis/paired_boot_1.json',fresh)
        except Exception as error:
            write_new_json(s.directory/'invalid_run.json',{'error':str(error),'boot_id':boot['boot_id'],'time':utc()})
            raise
        finally:native.cleanup_rpc(s,boot)


def report(s):
    out=s.directory/'analysis'/('report_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]);out.mkdir()
    sessions=[read(p) for p in sorted((s.directory/'raw/sessions').glob('*/gate_summary.json'))]
    support=[r for p in sorted((s.directory/'analysis').glob('paired_boot_*.json')) for r in read(p)]
    boot_count=len({r['boot_id'] for r in support});inference=repeated(support) if boot_count==3 else None
    lines=['# CPU-only与全CPU/VTA阶段互斥','','同一流水阶段线程池、4/1/4和四核mask。CPU/VTA均为whole-stage admission，门等待不计入run。全部阶段互斥改变资源约束，II不要求改善；DDR/cache/调度及共享池路径贡献尚需分辨。',
           '', '| 会话 | 门/范围 | 首CPU/VTA/尾CPU run ms | II ms | CPU/VTA阶段重叠ms |', '|---|---|---:|---:|---:|']
    for r in sessions:
        if r['mode']!='pipeline':continue
        m=r['gate_metrics'];runs='/'.join(f'{v["run_mean_ms"]:.3f}' for v in m['stages']);o=m['pairwise_stage_overlap_ms']
        lines.append(f'| {r["session_id"]} | {m["gate_mode"]}/{m["scope"]} | {runs} | {m["ii_ms"]:.3f} | {o["0:1"]+o["1:2"]:.3f} |')
        rows=[json.loads(line) for line in (s.directory/r['raw_jsonl']).read_text().splitlines()][7:27]
        lo=min(x['stage0_start_ms'] for x in rows);hi=max(x['stage2_end_ms'] for x in rows)
        svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="230"><rect width="100%" height="100%" fill="white"/>']
        for i in range(3):
            svg.append(f'<text x="5" y="{45+i*65}">stage{i}</text>')
            for x in rows:
                def px(t):return 90+(t-lo)/(hi-lo)*1080
                a,b=x[f'stage{i}_start_ms'],x[f'stage{i}_end_ms'];req=x.get(f'stage{i}_cpu_gate_request_ms');acq=x.get(f'stage{i}_cpu_gate_acquire_ms')
                if req is not None:svg.append(f'<rect x="{px(req):.3f}" y="{25+i*65}" width="{max(0,px(acq)-px(req)):.3f}" height="25" fill="#bbb"/>')
                svg.append(f'<rect x="{px(a):.3f}" y="{25+i*65}" width="{px(b)-px(a):.3f}" height="25" fill="{("#2674a5","#dc8c20","#2b9563")[i]}"/>')
        svg.append('</svg>');(out/(r['session_id']+'.svg')).write_text('\n'.join(svg))
    pilot=read(s.directory/'analysis/pilot.json') if (s.directory/'analysis/pilot.json').exists() else {}
    lines+=['',f'独立boot数：{boot_count}；探索触发复现：{pilot.get("replication_triggered",False)}。不足三boot仅探索性报告，不拟合34ms。']
    if inference:lines.append('三boot统计：'+json.dumps(inference,ensure_ascii=False))
    (out/'CONCLUSION_ZH.md').write_text('\n'.join(lines)+'\n')
    write_new_json(out/'analysis.json',{'sessions':sessions,'pilot':pilot,'independent_boot_count':boot_count,'replication':inference})
    print(out,flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('phase',choices=('freeze','pilot','replicate','analyze'))
    parser.add_argument('--run-dir',required=True);parser.add_argument('--execute-board',action='store_true');a=parser.parse_args(argv)
    if a.phase in ('pilot','replicate') and not a.execute_board:parser.error('Explicit --execute-board required')
    s=freeze(a.run_dir) if a.phase=='freeze' else load(a.run_dir)
    if a.phase in ('pilot','replicate'):execute(s,a.phase=='replicate');report(s)
    elif a.phase=='analyze':report(s)
    return s


if __name__=='__main__':main()
