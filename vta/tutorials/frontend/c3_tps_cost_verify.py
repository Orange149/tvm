"""Prepare, execute and report the frozen R boundary-cost board pilot."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
from c3_tps_cost_probe import HERE, HOLD, LOCAL, BOARD, read, write, sha, require, verify
from c3_tps_cost_protocol import manifest, check_manifest, transport, analyse

QUALIFIED = HOLD/'20261017_c3_tps_cost_probe_local_run01'


def prepare(args):
    root=args.output.resolve()
    require(not root.exists(),'ATTEMPT_EXISTS')
    bindings={}
    for source in (QUALIFIED, BOARD):
        seal=source/'completion_manifest.json'
        bindings[str(seal)]=sha(seal)
        bindings.update({str(source/k):v for k,v in read(seal)['artifact_sha256'].items()})
    bindings.update(read(QUALIFIED/'freeze.json')['bindings'])
    verify(bindings)
    require(read(QUALIFIED/'qualification_complete.json')['status']=='FOUR_MODES_R_QUALIFIED','LOCAL_NOT_QUALIFIED')
    root.mkdir()
    try:
        artifacts=deepcopy(read(BOARD/'artifacts/prepared.json'))
        artifacts['runtime']={'path':str(QUALIFIED/'runtime_arm/libvta.so'),
                              'sha256':sha(QUALIFIED/'runtime_arm/libvta.so')}
        artifacts['helper']={'path':str(QUALIFIED/'helper_arm/c3_official_window_runtime_helper.so'),
                             'sha256':sha(QUALIFIED/'helper_arm/c3_official_window_runtime_helper.so')}
        original=artifacts['programs']['R'];artifacts['programs']={}
        for mode in ('off','on'):
            descriptor=deepcopy(original)
            descriptor['variant']=mode
            descriptor['local_calls']=str(QUALIFIED/(mode+'_detailed')/'R_detailed/summary.json')
            artifacts['programs'][mode]=descriptor
        artifacts.update(schema='c3_tps_cost_board_prepared_v1',status='LOCAL_QUALIFIED_ARM_TIMER_FPGA_PENDING',
                         new_FSim_calls=0,adopted_FSim_calls=28,board_contacted=False)
        write(root/'prepared.json',artifacts)
        identities={p:{'sha256':d['sha256'],'symbol':d['symbol'],'abi':d['abi'],
                       'runtime':artifacts['runtime'],'helper':artifacts['helper']} for p,d in artifacts['programs'].items()}
        ledger=manifest(root.name,identities);check_manifest(ledger)
        write(root/'call_manifest.json',ledger)
        write(root/'transport_contract.json',transport(ledger))
        sources=list(HERE.glob('c3_*.py'))+[HERE/'test_c3_tps_cost_board.py',HERE/'test_c3_tps_cost_probe.py']
        snapshots=root/'tool_sources';snapshots.mkdir()
        for p in sources:
            bindings[str(p)]=sha(p)
            if p.name.startswith('c3_tps_cost_') or p.name.startswith('test_c3_tps_cost_'):
                shutil.copyfile(p,snapshots/p.name)
        for key in ('runtime','helper','storage_helper','runtime_tvm','rpc_server','bitstream','hardware_archive'):
            item=artifacts[key];bindings[item['path']]=item['sha256']
        bindings[str(Path(args.known_hosts).resolve())]=sha(args.known_hosts)
        for p in root.rglob('*'):
            if p.is_file():bindings[str(p)]=sha(p)
        verify(bindings)
        write(root/'freeze.json',{'status':'PREPARED_BOARD_PENDING','bindings':bindings,
              'host':args.host,'known_hosts':str(Path(args.known_hosts).resolve()),'complete_calls':72,
              'sessions':12,'formal_samples':24,'instrument_sensitivity':{'median_abs_percent':.5,'max_round_abs_percent':1},
              'no_retries':True,'source_contract':str(QUALIFIED/'NEXT_BOARD_CONTRACT.md')})
    except BaseException as error:
        write(root/'prepare_failure.json',{'error':str(error),'new_target_calls':0});raise


def board(args):
    from c3_tps_cost_board import execute
    root=args.output.resolve()
    require(not (root/'completion_manifest.json').exists() and not (root/'prepare_failure.json').exists(),'ATTEMPT_CLOSED')
    freeze=read(root/'freeze.json')
    require(args.host==freeze['host'] and str(Path(args.known_hosts).resolve())==freeze['known_hosts'],'ENDPOINT_CHANGED')
    bindings={**freeze['bindings'],str(root/'freeze.json'):sha(root/'freeze.json')};verify(bindings)
    ledger=read(root/'call_manifest.json');check_manifest(ledger)
    args.transport_contract=read(root/'transport_contract.json')
    require(args.transport_contract==transport(ledger),'TRANSPORT_CHANGED')
    return execute(args,read(root/'prepared.json'),ledger,root/'board',bindings)


def report(root):
    from c3_tps_sync_release_support import accounting
    require(not (root/'completion_manifest.json').exists(),'ALREADY_SEALED')
    summary=read(root/'board/summary.json') if (root/'board/summary.json').exists() else read(root/'prepare_failure.json')
    success=summary.get('status')=='COMPLETE_SUCCESS'
    lines=['# TPS 完整调用成本归因板测','',
           '对象为封存R：YOLOv3-tiny-224节点24，b1h14w14ci4co4、oc2/h1，完整卷积、ADD、右移7位、clip及int8转换。',
           'off/on执行同一R目标模块和同一runtime/helper，只控制VTADeviceRun前后两次时钟读取。不包含同步或聚合优化变体。','']
    account=accounting(root/'board');write(root/'call_accounting.json',account)
    if success:
        verify(read(root/'freeze.json')['bindings'])
        rows=[json.loads(l) for l in (root/'board/results.jsonl').read_text().splitlines()]
        stats=analyse(read(root/'call_manifest.json'),rows)
        require(account['fully_certified']==account['started_calls']['exact']==account['completed_calls']['exact']==72,'CALL_ACCOUNTING_INCOMPLETE')
        write(root/'statistics.json',stats)
        organizations=[]
        for f in sorted((root/'board').glob('session_*/organization_*.json')):
            organizations.append({'file':str(f.relative_to(root)),**read(f)})
        require(len(organizations)==12,'DIAGNOSTIC_COUNT')
        fields=('MAC_GEMM','RESET_GEMM','ALU','all_instructions','UOP_LOAD','UOP_LOAD_bytes','UOP_SRAM_wraps','submissions')
        require(all(tuple(r[k] for k in fields)==tuple(organizations[0][k] for k in fields) for r in organizations),'ORGANIZATION_CHANGED')
        write(root/'organization.json',organizations)
        lines+=['## 完整调用和仪器敏感性','',
                '72次完整FPGA调用全部通过；12个fresh RPC单元、24个正式样本。三seed、同存储W0→W1→W0、哨兵及只读参数检查通过。',
                '12份详细诊断认证μop、依赖和有限FIFO；正式样本分别匹配自己的暖诊断提交路径。','',
                '|轮次|off ms|on ms|on相对off增加 %|','|---|---:|---:|---:|']
        for i,d in enumerate(stats['round_on_minus_off_percent']):
            lines.append(f"|{i+1}|{stats['round_medians_ms']['off'][i]:.6f}|{stats['round_medians_ms']['on'][i]:.6f}|{d:.6f}|")
        lines+=['',f"四轮中位数：off {stats['T_ms']['off']:.6f} ms，on {stats['T_ms']['on']:.6f} ms。",
                f"逐轮相对差异中位数 {stats['median_on_minus_off_percent']:.6f}%，绝对差中位数 {stats['median_on_minus_off_ms']:.6f} ms。",
                '仪器敏感性工程门禁：'+('通过' if stats['instrument_sensitivity_gate_passed'] else '未通过')+'。门槛为差异中位数绝对值≤0.5%，每轮差异绝对值≤1%；不作为统计等价或零仪器开销证明。','',
                '## 同一次调用内的三段成本','',
                '|区间|12个on正式样本中位数 ms|范围 ms|逐调用占比中位数 %|','|---|---:|---|---:|']
        labels={'before_driver_ms':'驱动调用前','driver_run_and_wait_ms':'驱动运行及完成等待','after_driver_ms':'驱动返回后收尾'}
        for field,label in labels.items():
            p=stats['on_formal_parts'][field]
            lines.append(f"|{label}|{p['median_ms']:.6f}|{p['range_ms'][0]:.6f}–{p['range_ms'][1]:.6f}|{p['median_percent']:.4f}|")
        lines+=['','分项占比来自同一次调用的分项/总时长；分项中位数相加不必等于总时长中位数。逐次整数时间戳均闭合。',
                'VTADeviceRun区间包含驱动启动、设备执行与等待，不等于纯硬件计算，更不能据此分解DMA、GEMM与同步stall。驱动外区间也不全部是命令生成。',
                '若仪器敏感性门禁未通过，保留成本观察，但不据此形成细粒度瓶颈主张。','',
                '## 程序组织','',
                '两模式详细诊断组织相同：'+', '.join(k+'='+str(organizations[0][k]) for k in fields)+'。',
                '原详细profiling在正式路径关闭；输出校验、RPC、输入准备及JSON记录在计时外。','',
                '## 研究边界','',
                '能不能合、能合多少：本轮没有改计算与同步边界，只验证观测保持合格R的执行。',
                '值不值：本轮提供后续选题依据，不是优化收益证明。不与历史R/S时延拼接，不称统计显著性或整网加速。']
    else:
        lines+=['按停止合同结束，未形成完整归因对照。','',str(summary),'',
                '失败样本不重试、不补齐、不与其他attempt拼接；调用计数按helper证据报告。']
    lines+=['','证据：[冻结身份](freeze.json)、[调用清单](call_manifest.json)、[调用账目](call_accounting.json)、[板端状态](board/summary.json)。',
            '本阶段结束后暂停，不追加优化候选。','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    write(root/'completion_manifest.json',{'status':'R_COST_BOARD_COMPLETE' if success else 'R_COST_BOARD_STOPPED',
          'paused':True,'complete_comparison':success,'optimization_gain_claim':False,
          'artifact_sha256':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','board','report'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--host',default='192.168.1.247')
    parser.add_argument('--known-hosts',default='/home/orange/.ssh/known_hosts')
    args=parser.parse_args()
    if args.phase=='report':report(args.output.resolve())
    else:globals()[args.phase](args)


if __name__=='__main__':main()
