#!/usr/bin/env python3
"""Prepare, execute and seal the single-attempt node34 B/A FPGA comparison."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback
from c3_node34_uop_identity_arm import (HERE, REPO, LOCAL, PYTHON, prepare_artifacts,
    read, write, sha, require, verify_bindings)
from c3_node34_uop_identity_protocol import manifest, _check_manifest, transport_contract, analyse
from c3_tps_sync_release_support import accounting


def preflight(args):
    out=args.preflight_dir or args.output_dir.with_name(args.output_dir.name+'_preflight')
    out.mkdir(exist_ok=False)
    files=list(HERE.glob('c3_*.py'))+list(HERE.glob('test_c3_node34_uop_identity*.py'))
    bindings={str(p):sha(p) for p in files}
    env=dict(os.environ,PYTHONPATH=':'.join(str(REPO/p) for p in ('python','vta/python','vta/tutorials/frontend')),
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    tests=['test_c3_node34_uop_identity_board','test_c3_node34_uop_identity',
           'test_c3_tps_aggregation_prepare',
           'test_c3_tps_aggregation_transport_v2','test_c3_tps_sync_release_method',
           'test_c3_tps_sync_release_local','test_c3_tps_sync_release_prepare']
    with (out/'tests.log').open('x') as log:
        proc=subprocess.run([str(PYTHON),'-m','unittest','-v',*tests],env=env,
                            stdout=log,stderr=subprocess.STDOUT,timeout=240)
    verify_bindings(bindings)
    write(out/'result.json',{'returncode':proc.returncode,'source_bindings':bindings,
                            'target_calls':0,'board_contacted':False})
    require(proc.returncode==0,'TOOL_PREFLIGHT_FAILED')
    return out


def prepare(args):
    root=args.output_dir
    require(not root.exists(),'ATTEMPT_DIRECTORY_ALREADY_EXISTS')
    checked=preflight(args)
    root.mkdir(exist_ok=False)
    try:
        artifacts=prepare_artifacts(root/'artifacts')
        ids={p:{'sha256':d['sha256'],'symbol':d['symbol'],'abi':d['abi'],
                'runtime':artifacts['runtime'],'helper':artifacts['helper']}
             for p,d in artifacts['programs'].items()}
        ledger=manifest(root.name,ids)
        write(root/'call_manifest.json',ledger)
        write(root/'transport_contract.json',transport_contract(ledger))
        bindings=dict(artifacts['source_bindings'])
        for name in ('runtime','helper','storage_helper','runtime_tvm','rpc_server','bitstream','hardware_archive'):
            item=artifacts[name];bindings[item['path']]=item['sha256']
        for descriptor in artifacts['programs'].values():
            for key,digest in descriptor['sha256'].items():bindings[descriptor[key]]=digest
        for p in list(HERE.glob('c3_*.py'))+list(HERE.glob('test_c3_node34_uop_identity*.py')):
            bindings[str(p)]=sha(p)
        for folder in (root,checked):
            for p in folder.rglob('*'):
                if p.is_file():bindings[str(p)]=sha(p)
        bindings[str(Path(args.known_hosts).resolve())]=sha(args.known_hosts)
        verify_bindings(bindings)
        write(root/'freeze.json',{'status':'B_A_ARM_READY_FPGA_PENDING','source_and_artifact_bindings':bindings,
            'host':args.host,'known_hosts':str(Path(args.known_hosts).resolve()),
            'target_calls':120,'sessions':20,'formal_samples':36,'new_FSim_calls':0,
            'baseline':'B: qualified node34 author-order locally adapted TPS',
            'candidate':'A: same ordered work; MAC closure signature k projected to k%2',
            'primary_comparison':'A/B','engineering_threshold':{'gain_percent':2,'strict_wins':5},
            'stop_on_any_error':True,'automatic_retry':False,'pause_after_attempt':True})
        return {'status':'B_A_ARM_READY_FPGA_PENDING','target_calls_executed':0}
    except BaseException as error:
        write(root/'prepare_failure.json',{'error':str(error),'traceback':traceback.format_exc(),
                                         'target_calls':0,'board_contacted':False})
        raise


def board(args):
    from c3_node34_uop_identity_board_execute import execute
    root=args.output_dir
    require(not (root/'completion_manifest.json').exists(),'ATTEMPT_ALREADY_SEALED')
    require(not (root/'prepare_failure.json').exists(),'PREPARATION_FAILED')
    freeze=read(root/'freeze.json')
    require(args.host==freeze['host'] and str(Path(args.known_hosts).resolve())==freeze['known_hosts'],
            'BOARD_ENDPOINT_CHANGED')
    bindings={**freeze['source_and_artifact_bindings'],str(root/'freeze.json'):sha(root/'freeze.json')}
    verify_bindings(bindings)
    ledger=read(root/'call_manifest.json');_check_manifest(ledger)
    args.transport_contract=read(root/'transport_contract.json')
    require(args.transport_contract==transport_contract(ledger),'TRANSPORT_CHANGED')
    return execute(args,read(root/'artifacts/prepared.json'),ledger,root/'board',bindings)


def report(root):
    require(not (root/'completion_manifest.json').exists(),'REPORT_ALREADY_SEALED')
    summary=read(root/'board/summary.json') if (root/'board/summary.json').exists() else read(root/'prepare_failure.json')
    success=summary.get('status')=='COMPLETE_SUCCESS'
    text=['# 节点34 周期μop缓存身份复用板测结果','',
          '对象：YOLOv3-tiny-224 节点34，512→1024、7×7、3×3；Tile b1h7w7ci1co16、oc2/h1。',
          'B为已认证的作者循环顺序本地适配TPS；A只将两处MAC closure的签名从k投影为k%2。完整四参数语义为卷积→广播ADD→算术右移6位→clip[-127,127]→int8。',
          'GEMM仍为128条；这轮研究缓存身份复用，不是指令聚合。','']
    account=accounting(root/'board') if (root/'board').exists() else {'registered_attempts':0,'fully_certified':0}
    write(root/'call_accounting.json',account)
    if success:
        verify_bindings(read(root/'freeze.json')['source_and_artifact_bindings'])
        results=[json.loads(line) for line in (root/'board/results.jsonl').read_text().splitlines()]
        stats=analyse(read(root/'call_manifest.json'),results)
        write(root/'statistics.json',stats)
        require(account['completed_calls']['exact']==account['started_calls']['exact']==account['fully_certified']==120,'CALL_ACCOUNTING_INCOMPLETE')
        require(len(list((root/'board').glob('session_*/joint_*.json')))==18,'JOINT_CERTIFICATE_COUNT')
        organization_rows=[]
        ledger=read(root/'call_manifest.json')
        for file in sorted((root/'board').glob('session_*/organization_*.json')):
            ordinal=int(file.parent.name.split('_')[1]);position=int(file.stem.split('_')[1])
            organization_rows.append({'session':ordinal,'position':position,
                                      'program':ledger['sessions'][ordinal-1]['program'],**read(file)})
        require(len(organization_rows)==36,'ORGANIZATION_CERTIFICATE_COUNT')
        write(root/'organization.json',organization_rows)
        write(root/'qualification_first_calls.json',[r for r in results if r['phase']=='qualification' and r['position']==1])
        gain=stats['pairwise']['A/B']
        text += ['## 能不能复用','',
            '120次完整FPGA调用通过，20个fresh RPC单元、36个正式样本。输出哨兵、独立参考、只读参数、同实际权重存储W0→W1→W0均通过。',
            '36份详细诊断认证完整μop引用及有限FIFO；18份B/A联合证书保持张量物理访问、原有依赖和逻辑通知配对。正式调用匹配自身暖诊断的提交路径。','',
            '## 能复用到哪里','',
            '范围限定于已证明只依赖k%2的两处MAC initializer；原始k继续控制DMA，kernel内容、顺序、Tile、线程和张量请求保持。运行时及helper为共同冻结底座。','',
            '## 复用后值不值','',
            '|程序|六轮中位时延T ms|轮中位数范围 ms|','|---|---:|---|']
        for p in ('B','A'):
            v=stats['round_medians'][p]
            text.append(f"|{p}|{stats['T'][p]:.6f}|{min(v):.6f}–{max(v):.6f}|")
        text += ['',f"逐轮改善中位数 **{gain['Gain_percent']:.4f}%**，严格胜场 **{gain['wins']}/6**，绝对节省中位数 **{gain['absolute_ms']:.6f} ms**。",
                 f"冻结工程目标（Gain≥2%、严格胜≥5/6）：{'达到' if gain['engineering_goal_met'] else '未达到'}。",'',
                 '|轮次|B ms|A ms|改善 %|','|---|---:|---:|---:|']
        for i in range(6):
            text.append(f"|{i+1}|{stats['round_medians']['B'][i]:.6f}|{stats['round_medians']['A'][i]:.6f}|{gain['round_gains_percent'][i]:.6f}|")
        fields=('MAC_GEMM','RESET_GEMM','ALU','all_instructions','compute_NOP','UOP_LOAD','UOP_LOAD_bytes','serialized_uop_bytes','UOP_SRAM_wraps','submissions')
        text += ['', '|程序|MAC GEMM|RESET|ALU|总指令|compute NOP|UOP LOAD|LOAD字节|序列化字节|回绕|提交|',
                 '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for p in ('B','A'):
            rows=[r for r in organization_rows if r['program']==p]
            require(all(tuple(r[k] for k in fields)==tuple(rows[0][k] for k in fields) for r in rows),'DIAGNOSTIC_ORGANIZATION_CHANGED')
            text.append('|'+p+'|'+'|'.join(str(rows[0][k]) for k in fields)+'|')
        text += ['', 'Gain按六个逐轮比值取中位数，不以两个T相除替代。范围、MAD、标准差见statistics.json。',
            '计时覆盖一次完整融合函数组织、全部提交和必要等待；RPC、参数准备、权重更新、初始化、输出读取和检查在外。最小模式仍有固定模式检查和逐提交记录成本。',
            '详细资格首调用成本单列，不与最小模式相减推断缓存建立成本。',
            '结果限于本次启动和节点34固定TPS对照；不称通用方法完成、统计显著性、TopHub胜出或整网加速。实际成本无法分解时不强行归因。']
    else:
        text += ['本attempt已按停止合同封存，未形成完整性能结论。', '', '停止原因：`'+str(summary.get('error',summary))+'`。',
                 '工具/环境错误与程序正确性失败分别判断；失败样本不重试、不补齐、不与其他attempt拼接。实际调用以helper证据为准。']
    text += ['', '证据：[冻结身份](freeze.json)、[逐调用清单](call_manifest.json)、[调用账目](call_accounting.json)、[板端状态](board/summary.json)。',
             '', '**本轮结束，暂停，不增加候选。**','']
    (root/'REPORT.md').write_text('\n'.join(text))
    files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
    seal={'status':'NODE34_IDENTITY_BOARD_COMPLETE' if success else 'NODE34_IDENTITY_BOARD_STOPPED',
          'paused':True,'performance_conclusion':success,'artifact_sha256':files}
    write(root/'completion_manifest.json',seal)
    return {k:v for k,v in seal.items() if k!='artifact_sha256'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','board','report'))
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--preflight-dir',type=Path)
    parser.add_argument('--host',default='192.168.1.247')
    parser.add_argument('--known-hosts',default='/home/orange/.ssh/known_hosts')
    args=parser.parse_args();args.output_dir=args.output_dir.resolve()
    result=report(args.output_dir) if args.phase=='report' else globals()[args.phase](args)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
