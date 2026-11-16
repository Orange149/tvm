#!/usr/bin/env python3
"""Prepare, execute and seal the single-attempt R/S FPGA comparison."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback
from c3_tps_sync_release_arm import (HERE, REPO, LOCAL, PYTHON, prepare_artifacts,
    read, write, sha, require, verify_bindings)
from c3_tps_sync_release_protocol import manifest, _check_manifest, transport_contract, analyse
from c3_tps_sync_release_support import accounting


def preflight(args):
    out=args.preflight_dir or args.output_dir.with_name(args.output_dir.name+'_preflight')
    out.mkdir(exist_ok=False)
    files=list(HERE.glob('c3_*.py'))+list(HERE.glob('test_c3_tps_sync_release*.py'))
    bindings={str(p):sha(p) for p in files}
    env=dict(os.environ,PYTHONPATH=':'.join(str(REPO/p) for p in ('python','vta/python','vta/tutorials/frontend')),
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    tests=['test_c3_tps_sync_release_board','test_c3_tps_aggregation_prepare',
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
        for p in list(HERE.glob('c3_*.py'))+list(HERE.glob('test_c3_tps_sync_release*.py')):
            bindings[str(p)]=sha(p)
        for folder in (root,checked):
            for p in folder.rglob('*'):
                if p.is_file():bindings[str(p)]=sha(p)
        bindings[str(Path(args.known_hosts).resolve())]=sha(args.known_hosts)
        verify_bindings(bindings)
        write(root/'freeze.json',{'status':'R_S_ARM_READY_FPGA_PENDING','source_and_artifact_bindings':bindings,
            'host':args.host,'known_hosts':str(Path(args.known_hosts).resolve()),
            'target_calls':120,'sessions':20,'formal_samples':36,'new_FSim_calls':0,
            'baseline':'restored author-order locally adapted TPS R',
            'candidate':'S exactly two MAC wait consumers moved past four INP loads',
            'primary_comparison':'S/R','engineering_threshold':{'gain_percent':2,'strict_wins':5},
            'stop_on_any_error':True,'automatic_retry':False,'pause_after_attempt':True})
        return {'status':'R_S_ARM_READY_FPGA_PENDING','target_calls_executed':0}
    except BaseException as error:
        write(root/'prepare_failure.json',{'error':str(error),'traceback':traceback.format_exc(),
                                         'target_calls':0,'board_contacted':False})
        raise


def board(args):
    from c3_tps_sync_release_board import execute
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
    text=['# TPS 两处等待后移板测结果','',
          '对象：YOLOv3-tiny-224 节点24，Tile b1h14w14ci4co4，oc2/h1；完整卷积、ADD、右移7位、clip及int8转换。',
          'R为恢复作者循环顺序的本地适配TPS；S只后移第二归约批次的两处MAC等待。O到R的修正不计作创新收益。','']
    if success:
        verify_bindings(read(root/'freeze.json')['source_and_artifact_bindings'])
        results=[json.loads(line) for line in (root/'board/results.jsonl').read_text().splitlines()]
        stats=analyse(read(root/'call_manifest.json'),results)
        write(root/'statistics.json',stats)
        account=accounting(root/'board')
        require(account['completed_calls']['exact']==account['started_calls']['exact']==account['fully_certified']==120,
                'CALL_ACCOUNTING_INCOMPLETE')
        joints=sorted((root/'board').glob('session_*/joint_*.json'))
        require(len(joints)==18,'JOINT_BOARD_CERTIFICATE_COUNT')
        write(root/'call_accounting.json',account)
        organization_rows=[]
        for file in sorted((root/'board').glob('session_*/organization_*.json')):
            ordinal=int(file.parent.name.split('_')[1])
            position=int(file.stem.split('_')[1])
            session=read(root/'call_manifest.json')['sessions'][ordinal-1]
            organization_rows.append({'session':ordinal,'position':position,
                                     'program':session['program'],**read(file)})
        require(len(organization_rows)==36,'ORGANIZATION_CERTIFICATE_COUNT')
        write(root/'organization.json',organization_rows)
        write(root/'qualification_first_calls.json',[r for r in results if r['phase']=='qualification' and r['position']==1])
        gain=stats['pairwise']['S/R']
        text += ['## 能不能撤除等待边界','',
            '120次完整FPGA调用全部通过，20个fresh RPC单元，36个正式样本。同权重存储W0→W1→W0、输出哨兵、逐元素参考及只读参数检查通过。',
            '36次详细诊断核对逐μop引用、实际通知配对及有限FIFO进度；其中18份S证书联合核对两处等待后移。正式调用按自身暖诊断核对完整提交路径。','',
            '## 能提前到哪里','',
            '两处等待分别越过4次INP装载，落在随后的WGT装载前。当前MAC读取INP [0,1024)，新输入写入[1024,2048)；WGT [0,144)覆盖仍受MAC完成通知保护。单位为VTA向量索引。首批RESET等待保持。',
            '既有死RESET读取的无序关系仍按本地证书保留；没有把它们解释成所有物理访问均有序。','',
            '## 合了值不值','',
            '本轮测的是同步细化S/R，不是旧指令聚合G/B。', '',
            '|程序|六轮中位时延T ms|轮中位数范围 ms|','|---|---:|---|']
        for p in ('R','S'):
            v=stats['round_medians'][p]
            text.append(f"|{p}|{stats['T'][p]:.6f}|{min(v):.6f}–{max(v):.6f}|")
        text += ['',f"逐轮改善中位数 **{gain['Gain_percent']:.4f}%**，严格胜场 **{gain['wins']}/6**，绝对节省中位数 **{gain['absolute_ms']:.6f} ms**。",
                 f"冻结工程目标（Gain≥2%、严格胜≥5/6）：{'达到' if gain['engineering_goal_met'] else '未达到'}。",'',
                 '|轮次|R ms|S ms|改善 %|','|---|---:|---:|---:|']
        for i in range(6):
            text.append(f"|{i+1}|{stats['round_medians']['R'][i]:.6f}|{stats['round_medians']['S'][i]:.6f}|{gain['round_gains_percent'][i]:.6f}|")
        text += ['', '|程序|GEMM|RESET|ALU|指令总数|compute NOP|UOP LOAD|载荷字节|回绕|提交|',
                 '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        fields=('MAC_GEMM','RESET_GEMM','ALU','all_instructions','compute_NOP','UOP_LOAD',
                'UOP_LOAD_bytes','UOP_SRAM_wraps','submissions')
        for p in ('R','S'):
            rows=[r for r in organization_rows if r['program']==p]
            require(all(tuple(r[k] for k in fields)==tuple(rows[0][k] for k in fields) for r in rows),
                    'DIAGNOSTIC_ORGANIZATION_CHANGED')
            text.append('|'+p+'|'+'|'.join(str(rows[0][k]) for k in fields)+'|')
        text += ['', 'host清单身份共2052条基址μop，内容共享后序列化2048条；最大单kernel为504条。装载及提交重置后的重装原因逐调用见organization.json。',
                 '', 'Gain为六个逐轮比值的中位数，不由两个T相除替代。范围、MAD和标准差见statistics.json。',
            '计时包含一次完整融合函数的命令组织、提交和完成等待；参数准备、RPC、初始化和输出校验在外。最小模式仍有固定模式检查与逐提交记录成本。',
            '详细模式资格首调用成本单列，不能与最小模式时延相减解释缓存开销。',
            '结果限于一次启动、当前受限TPS表示，不作为独立复验、统计显著性、TopHub胜出或整图加速证据。']
    else:
        text += ['实验按停止合同结束，未形成完整性能对照。', '', '停止原因：`'+str(summary.get('error',summary))+'`。',
                 '失败样本不补齐、不重试、不与其他attempt拼接。执行计数以helper证据及其上下界为准。']
    text += ['', '证据：[冻结身份](freeze.json)、[调用清单](call_manifest.json)、[板端状态](board/summary.json)、[原始逐调用记录](board/results.jsonl)。',
             '', '**本轮结束后暂停，不追加候选。**','']
    (root/'REPORT.md').write_text('\n'.join(text))
    files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
    seal={'status':'R_S_BOARD_COMPLETE' if success else 'R_S_BOARD_STOPPED','paused':True,
          'performance_conclusion':success,'artifact_sha256':files}
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
