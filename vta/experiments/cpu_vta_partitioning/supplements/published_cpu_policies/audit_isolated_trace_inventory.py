"""Unify existing isolated trace indexes without acquiring or changing data.

An entry is a projection, not an independent measurement.  Affinity aliases
remain explicit: equal thread counts do not make two placements identical.
"""
from pathlib import Path
from collections import Counter
import json

import run_scope_rank  # Establish the experiment's isolated Python path.
from cpu_vta_partitioning.common import read, write, sha256, verify_inventory

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts/isolated_trace_inventory_20261112_01'
INDEXES = (
    ROOT / 'artifacts/cross_network_model_01/phase_validation/ISOLATED_CPU_INPUTS.json',
    ROOT / 'artifacts/cross_network_model_01/pool_temporal/INTRINSICS.json',
    ROOT / 'artifacts/current_isolated_phases_20261112_01/projection/ISOLATED_PHASES.json',
)


def capabilities(profile, device):
    if device == 'cpu':
        return dict(fork_join_structure=bool(profile.get('average', {}).get('kernels')),
                    paired_clean_cost='clean_cost' in profile,
                    host_cpu_split=False, device_completion_brackets=False)
    stages = profile.get('stages', {})
    devices = stages.get('devices', [])
    return dict(fork_join_structure=False, paired_clean_cost='clean_cost' in profile,
                host_cpu_split=bool(devices) and len(stages.get('host_serial_CPU', [])) == len(devices) + 1,
                device_completion_brackets=bool(devices) and all(
                    'completion_lower_ns' in v and 'completion_upper_ns' in v for v in devices))


def match(row, request):
    required = ('network', 'program', 'module_sha256', 'threads')
    if any(row[k] != request[k] for k in required):
        return 'different_module_or_thread_signature'
    if row['worker_core_map'] == request['worker_core_map'] and row['mask'] == request['mask']:
        return 'matching_placement'
    return 'structure_only_requires_declared_core_permutation'


def declared_hash(document, path, seen=None):
    sources = document.get('sources', {})
    if path in sources:
        return sources[path]
    seen = set() if seen is None else seen
    for source, expected in sources.items():
        if source in seen or '/accepted/' in source or not source.endswith('.json'):
            continue
        seen.add(source)
        if sha256(source) != expected:
            raise ValueError('parent isolated projection changed')
        value = declared_hash(read(source), path, seen)
        if value is not None:
            return value
    return None


def receipt_for(profile, document):
    provenance = profile.get('provenance', {})
    path = provenance.get('observed_receipt', provenance.get('receipt', profile.get('observed_receipt')))
    expected = provenance.get('observed_sha256', provenance.get('sha256'))
    if path and expected is None:
        expected = declared_hash(document, path)
    if path:
        if sha256(path) != expected:
            raise ValueError('isolated projection receipt hash changed')
        return Path(path)
    candidates = []
    for source in document['sources']:
        if '/accepted/' not in source:
            continue
        receipt = read(source)
        doc = receipt['task']
        spec = doc.get('spec', {})
        if spec.get('case') == profile['program'] and spec.get('mode') == 'observed':
            candidates.append(Path(source))
    if len(candidates) != 1:
        raise ValueError('an unambiguous observed component receipt is required')
    return candidates[0]


def run():
    if OUT.exists():
        raise FileExistsError('preserve the frozen inventory')
    rows, sources, receipts = [], {}, {}
    for index in INDEXES:
        document = read(index)
        if (document.get('no_pipeline_FPS') is not True and
                document.get('pipeline_FPS_read') is not False and
                document.get('no_pipeline_performance_input') is not True):
            raise ValueError('only isolated indexes are accepted')
        sources[str(index)] = sha256(index)
        for path, expected in document['sources'].items():
            if sha256(path) != expected:
                raise ValueError('isolated index source changed: ' + path)
        sections = (('CPU', document['CPU']), ('VTA', document['VTA'])) if 'CPU' in document else (
            ('profiles', document['profiles']),)
        for section, profiles in sections:
            for key, value in profiles.items():
                receipt_path = receipt_for(value, document)
                receipt = read(receipt_path)
                task = receipt['task']
                if task['kind'] != 'calibration' or len(task['stages']) != 1 or task['input_indices'] != [0]:
                    raise ValueError('a single-stage input0 calibration is required')
                if receipt_path not in receipts:
                    verify_inventory(receipt['raw'], receipt['files'])
                    receipts[receipt_path] = sha256(receipt_path)
                stage = task['stages'][0]
                identity = value.get('identity', value)
                network = identity.get('network', task.get('profile_spec', {}).get('network', 'yolov3_tiny416'))
                row = dict(index=str(index), projection_key=key, device=stage['device'],
                           network=network, program=stage['program'], threads=stage['threads'],
                           mask=stage['mask'], worker_core_map=stage['worker_core_map'],
                           module_sha256=identity['module_sha256'],
                           executor_sha256=task['executor_sha256'], runtime_sha256=task['runtime_sha256'],
                           observer_sha256=task.get('observer_sha256', task.get('adapter_sha256')),
                           source_boot=receipt['boot_id'], receipt=str(receipt_path),
                           receipt_sha256=sha256(receipt_path),
                           capabilities=capabilities(value, stage['device']))
                rows.append(row)
    unique = {r['receipt']: r for r in rows}
    old = [r for r in rows if r['index'] == str(INDEXES[1])]
    current = [r for r in rows if r['index'] == str(INDEXES[2])]
    matches = []
    for r in current:
        candidates = [dict(receipt=v['receipt'], result=match(v, r), capabilities=v['capabilities'])
                      for v in old if match(v, r) != 'different_module_or_thread_signature']
        matches.append(dict(program=r['program'], threads=r['threads'], requested_mask=r['mask'],
                            previous_support=candidates, new_capabilities=r['capabilities']))
    result = dict(rows=rows, current_support=matches, projection_count=len(rows),
                  unique_observed_receipts=len(unique),
                  unique_by_device=dict(Counter(r['device'] for r in unique.values())),
                  sources=sources, receipt_sources={str(p): h for p, h in receipts.items()},
                  raw_hashes_verified=True, new_native_sessions=0, pipeline_FPS_read=False,
                  separate_projections_are_not_independent_measurements=True,
                  correction='The earlier audit only consulted the14-profile index. The52-CPU/50-VTA pool index already contained CPU2-4t3 on cores012 and host serial splits for the three current VTA modules. Current CPU2-4t3 uses cores123; CPU12-13t3 was missing; current VTA profiles add host3 placement and completion brackets. No automatic affinity alias is introduced.')
    OUT.mkdir(parents=True)
    write(OUT / 'INVENTORY.json', result)
    report = ROOT.parents[4] / 'paper/my_paper/analysis/20261112_matched_device_profile/INVENTORY_CORRECTION_zh.txt'
    report.write_text('\n'.join([
        '隔离轨迹清点更正（只读核查，新增板端会话0）',
        '此前输入覆盖核查只读取14项索引，漏看候选池的52项CPU及50项VTA索引。“CPU2–4三线程结构完全缺失”的说法不准确。',
        '旧CPU2–4三线程已有核心012的结构，新测为核心123；两者应注明具体放置，不能称同一身份，也不能无条件替代。',
        '旧三个VTA模块已有主机首尾/中间CPU分段，但没有当前主机核心3的匹配记录及设备完成读回区间。新记录补的是这些能力。',
        'CPU12–13三线程在旧索引中确实缺失。',
        f'本次统一索引含{len(rows)}个投影，去重为{len(unique)}项观测会话；同源投影不计为独立重复。全部来源及原始文件哈希已核验。',
        '旧报告和原始凭据保留；本更正应与旧INPUT_COVERAGE_zh.txt一起阅读。未来补测前须查询统一索引。',
        '本次没有重新标定、改公式、读取流水FPS、假定核心置换等价或触发开发板连接。',
    ]) + '\n')
    write(OUT / 'MANIFEST.json', dict(files={str(p): sha256(p) for p in (OUT / 'INVENTORY.json', report)},
                                         source_sha256=sha256(__file__)))
    print(json.dumps({k: result[k] for k in ('projection_count', 'unique_observed_receipts', 'unique_by_device')}, ensure_ascii=False))
    return result


if __name__ == '__main__':
    run()
