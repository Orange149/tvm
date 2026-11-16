"""Freeze and measure one unchanged YOLO224 graph with two B/A substitutions.

The reused two-slot protocol names its slots TH/P: here they mean B/A only.
There is no TopHub program and no whole-network TPS migration in this pool.
"""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import struct
import sys
import traceback
from c3_tps_sync_release_local import HERE, REPO, HOLD, read, write, sha, verify, require
from c3_yolo416_comparison_protocol import manifest, analyse
from c3_yolo416_board_transport import transport_contract
from c3_tps_broader_baseline_compare import bound, bind_seal

PREP=HOLD/'20261017_c3_yolo224_two_node_prepare_run02'
LOCAL=HOLD/'20261017_c3_yolo224_two_node_local_run02'
COMMON=HOLD/'20261017_c3_yolo416_arm_prepare_run01'
SLOTS={'TH':'B','P':'A'}


def build_helpers(root):
    from tvm.contrib import cc
    from build_vta_yolov3_tiny_relay_residency_dispatch_pair import cross_compiler
    out=root/'arm_helpers';out.mkdir()
    options=['-std=c++17','-O2','-ldl']+['-I'+str(REPO/p) for p in
            ('include','3rdparty/dmlc-core/include','3rdparty/dlpack/include')]
    lifetime=HERE/'c3_yolo416_plugin_lifetime.cc'
    source={'graph_once_helper':PREP/'graph_once.cc','route_helper':HERE/'c3_yolo224_two_node_route.cc'}
    result={}
    for key,path in source.items():
        binary=out/(key+'.so')
        cross_compiler()(str(binary),[str(path),str(lifetime)],options=options)
        blob=binary.read_bytes()
        require(blob[:6]==b'\x7fELF\x02\x01' and struct.unpack_from('<H',blob,18)[0]==183,'HELPER_AARCH64_ELF')
        result[key]=bound(binary)
    write(out/'build_contract.json',{'status':'ARM_HELPERS_USE_SAME_QUALIFIED_SOURCES',
          'source':{k:bound(p) for k,p in source.items()},'lifetime_source':bound(lifetime),
          'compiler_command':os.environ.get('CXX'),'LDFLAGS':os.environ.get('LDFLAGS'),
          'sdk_environment':bound('/home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux'),
          'compiler_executable':bound(shutil.which(shlex.split(os.environ['CXX'])[0])),'outputs':result,
          'timer_body_changed':False,'only_diagnostic_capture_and_capacity_adapted':True})
    return result


def prepare(root,host,known_hosts):
    root.mkdir(exist_ok=False)
    try:
        bindings={}
        require(bind_seal(LOCAL,bindings)['status']=='YOLO224_TWO_NODE_GRAPH_PAIR_LOCAL_QUALIFIED',
                'LOCAL_FULL_GRAPH_QUALIFICATION_REQUIRED')
        bindings.update(read(LOCAL/'freeze.json')['bindings'])
        bind_seal(PREP,bindings); bind_seal(COMMON,bindings)
        require(read(PREP/'contract.json')['expected_submissions']==22,'ALL_ORIGINAL_SYNCS_RETAINED')
        old=read(HOLD/'20261017_c3_node34_uop_identity_board_run01/artifacts/prepared.json')
        artifacts={key:old[key] for key in ('runtime_tvm','rpc_server','storage_helper','bitstream')}
        artifacts.update({'runtime':bound(PREP/'runtime_arm/libvta.so'),
            'base_arm':bound(PREP/'base_arm.so'),'hardware_archive':bound(PREP/'hardware_archive.json'),
            'graph_observer_helper':bound(COMMON/'graph_observer_arm.so'),
            'rpc_observer_helper':bound(COMMON/'rpc_observer_arm.so'),
            'cpu_reference_helper':bound(COMMON/'cpu_reference_arm.so'),**build_helpers(root)})
        replacements=read(PREP/'replacements.json'); programs={};identities={}
        for slot,variant in SLOTS.items():
            modules={n:bound(replacements[variant][n]['arm_module']) for n in ('34','46')}
            descriptor={'kind':'graph','variant':slot,'actual_variant':variant,'object_id':'yolo224_two_node_pair',
                'graph_prepared':str(PREP),'local_unit':str(LOCAL),'replacement_modules':modules,
                'execution_identity':{'base_arm_sha256':sha(PREP/'base_arm.so'),
                    'graph_sha256':sha(PREP/'graph.json'),'params_sha256':sha(PREP/'params.bin'),
                    'replacement_module_sha256':{n:m['sha256'] for n,m in modules.items()}}}
            programs['yolo224_two_node_pair/'+slot]=descriptor
            identities[slot]={'actual_variant':variant,'execution_identity':descriptor['execution_identity']}
        artifacts['programs']=programs;write(root/'artifacts.json',artifacts)
        ledger=manifest(root.name,[{'object_id':'yolo224_two_node_pair','kind':'graph','programs':identities}])
        require(ledger['total_target_calls']==120 and len(ledger['sessions'])==20 and ledger['formal_samples']==36,
                'FIXED_WHOLE_GRAPH_B_A_BUDGET')
        write(root/'call_manifest.json',ledger);write(root/'transport_contract.json',transport_contract(ledger))
        write(root/'comparison_semantics.json',{'internal_slot_mapping':SLOTS,'network':'YOLOv3-tiny-224',
            'unchanged_base_ARM':artifacts['base_arm'],'only_replacements':[34,46],
            'B':'two sealed TPS B functions, other graph programs unchanged',
            'A':'same two functions plus exact UOP cache identity reuse',
            'no_TopHub_program':True,'not_full_network_TPS':True,'expected_submissions':22,
            'formal_samples':36,'GraphExecutor_calls':120,'fresh_RPC_sessions':20,
            'historical_samples_reused':0,'metric':'median of six paired per-round percentage changes',
            'timer':'one whole GraphExecutor.run with all command organization, submissions and waits',
            'outside_timer':['RPC','initialization','parameters','input updates','output checks'],
            'no_hidden_warmup_or_retry':True,'no_SD_write_Linux_reboot_or_external_service_stop':True,
            'formal_route':'original PackedFunc directly returned at graph construction, no per-call route wrapper'})
        def artifact_bind(value):
            if isinstance(value,dict):
                if 'path' in value and 'sha256' in value:bindings[value['path']]=value['sha256']
                for v in value.values():artifact_bind(v)
            elif isinstance(value,list):
                for v in value:artifact_bind(v)
        artifact_bind(artifacts)
        for path in [*HERE.glob('c3_*.py'),HERE/'c3_yolo224_two_node_route.cc',Path(known_hosts)]:
            bindings[str(path.resolve())]=sha(path)
        for path in root.rglob('*'):
            if path.is_file():bindings[str(path.resolve())]=sha(path)
        verify(bindings)
        write(root/'freeze.json',{'status':'TH_P_OPERATORS_AND_GRAPH_READY_FPGA_PENDING',
            'actual_comparison':'YOLO224_FIXED_GRAPH_TWO_NODE_B_A','bindings':bindings,'host':host,
            'known_hosts':str(Path(known_hosts).resolve()),'FPGA_calls':0,'target_calls':120,
            'formal_samples':36,'sessions':20})
        print('FIXED_YOLO224_TWO_NODE_B_A_READY_FPGA_PENDING',flush=True)
        return 0
    except Exception as error:
        write(root/'prepare_failure.json',{'type':type(error).__name__,'message':str(error),'FPGA_calls':0})
        traceback.print_exc();return 1


def report(root):
    from c3_yolo224_two_node_board_execute import accounting
    require(not (root/'completion_manifest.json').exists(),'ALREADY_SEALED')
    summary=read(root/'board/summary.json');account=accounting(root/'board')
    success=summary['status']=='COMPLETE_TH_P_OPERATORS_AND_GRAPH_SINGLE_BOOT_ATTEMPT'
    result={'status':'WHOLE_YOLO224_TWO_NODE_B_A_COMPLETE' if success else 'WHOLE_YOLO224_TWO_NODE_B_A_STOPPED',
        'success':success,'internal_slot_mapping':SLOTS,'accounting':account,'historical_samples_reused':0,
        'only_changed_graph_nodes':[34,46],'not_a_TopHub_comparison':True}
    text=['# YOLO224：仅替换节点34、46的整图B/A对照','']
    if success:
        verify(read(root/'freeze.json')['bindings'])
        rows=read(root/'board/results.json'); stats=analyse(read(root/'call_manifest.json'),rows)
        require(account['certified']==120 and not account['execution_unresolved_units'] and
                not account['invalid_records'],'COMPLETE_120_CALL_ACCOUNT_REQUIRED')
        item=stats['objects'][0]
        measured={'T_ms':{SLOTS[k]:v for k,v in item['T_ms'].items()},
            'round_medians_ms':{SLOTS[k]:v for k,v in item['round_medians_ms'].items()},
            'A_over_B_gain_percent':item['Gain_P_over_TH_percent'],'round_gains_percent':item['round_gains_percent'],
            'A_strict_wins':item['strict_wins'],'median_saved_ms':item['median_absolute_ms'],
            'round_saved_ms':item['round_absolute_ms'],'gain_range_percent':item['gain_range_percent'],
            'gain_MAD_percent':item['gain_MAD_percent'],'round_dispersion_ms':item['round_dispersion_ms']}
        result['measurement']=measured;result['statistics_legacy_slot_names']=stats
        text += ['同一份YOLOv3-tiny-224图，仅将节点34、46的原TPS B替换成身份复用A。',
            '图结构、训练参数、其余CPU/VTA程序和原ARM库均固定。', '',
            '| 整图程序 | 完整调用时延 |','|---|---:|',
            '| B：两节点原TPS | %.6f ms |'%measured['T_ms']['B'],
            '| A：两节点身份复用 | %.6f ms |'%measured['T_ms']['A'],'',
            '六轮配对改善中位数 **%.4f%%**，严格胜 **%d/6**，每轮节省时间中位数 **%.6f ms**。'%(
                measured['A_over_B_gain_percent'],measured['A_strict_wins'],measured['median_saved_ms']),
            '', '| 轮次 | B ms | A ms | 改善 % |','|---|---:|---:|---:|']
        for i,(b,a,g) in enumerate(zip(measured['round_medians_ms']['B'],measured['round_medians_ms']['A'],
                                    measured['round_gains_percent']),1):
            text.append('| %d | %.6f | %.6f | %.4f |'%(i,b,a,g))
        text += ['', '120次整图调用、36个正式样本、20个fresh RPC单元；输出逐元素检查，诊断关闭后正式计时。',
            '详细诊断认证全部22次提交、12次整数算子调用、μop引用、缓存与有限FIFO进度；正式路径匹配各自暖诊断。',
            'G不是两个最终时延中位数直接相除；原始样本、逐轮离散程度见result.json。', '',
            '这是两节点身份复用在固定图上的完整调用增量。其余算子仍是原图程序，不能称为全网TPS对TopHub的结果。',
            '单次Linux启动的受控六轮，不称统计显著性或独立复验；不以算子时延相加代替整图测量。']
    else:
        text += ['本attempt触发停止条件，未形成完整性能对照，不重试、不补样、不拼接。',
                 '停止原因及实际执行证据见board/failure.json与call_accounting.json。']
    write(root/'call_accounting.json',account);write(root/'result.json',result)
    (root/'REPORT.md').write_text('\n'.join(text)+'\n')
    write(root/'completion_manifest.json',{'status':result['status'],'success':success,'paused':True,
        'artifact_sha256':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}})
    print(result.get('measurement',result['status']),flush=True);return 0 if success else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','board','report']);p.add_argument('output',type=Path)
    p.add_argument('--host',default='192.168.1.247');p.add_argument('--known-hosts',default='/home/orange/.ssh/known_hosts')
    a=p.parse_args()
    if a.phase=='prepare':sys.exit(prepare(a.output.resolve(),a.host,a.known_hosts))
    if a.phase=='report':sys.exit(report(a.output.resolve()))
    from c3_yolo224_two_node_board_execute import execute
    sys.exit(execute(a.output.resolve()))
