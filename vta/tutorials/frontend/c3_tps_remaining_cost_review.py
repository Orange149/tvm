"""Read sealed R artifacts only; no TVM import, compilation, RPC or target calls."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
HOLD = HERE / 'report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout'
LOCAL = HOLD / '20261016_c3_tps_sync_release_local_run04'
BOARD = HOLD / '20261016_c3_tps_sync_release_board_run01'
WIDTH = {2: 16, 1: 256, 3: 64, 4: 16}
LABEL = {2: 'INP', 1: 'WGT', 3: 'ADD', 4: 'OUT'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(p):
    return json.loads(p.read_text())


def native_requests(text):
    """Decode tensor requests in native vector-index units, omitting UOP."""
    rows = []
    domains = {'LOAD INP': 2, 'LOAD WGT': 1, 'LOAD ACC': 3, 'STORE:': 4}
    for block in re.split(r'(?=INSTRUCTION \d+:)', text)[1:]:
        label = block.splitlines()[0].split(': ', 1)[1]
        if label not in domains:
            continue
        addr = re.search(r'DRAM: (0x[0-9a-fA-F]+), SRAM:(0x[0-9a-fA-F]+)', block)
        y = re.search(r'y: size=(\d+), pad=\[(\d+), (\d+)\]', block)
        x = re.search(r'x: size=(\d+), stride=(\d+), pad=\[(\d+), (\d+)\]', block)
        require(all((addr, y, x)), 'incomplete native request')
        rows.append(dict(domain=domains[label], offset=int(addr[1], 16),
                         sram=int(addr[2], 16), y=int(y[1]), x=int(x[1]),
                         stride=int(x[2]), pad=[int(x[3]), int(x[4]), int(y[2]), int(y[3])]))
    bases = {d: min(r['offset'] for r in rows if r['domain'] == d) for d in WIDTH}
    for row in rows:
        row['offset'] -= bases[row['domain']]
    return rows


def audit():
    sources = {}

    def sealed(root, name):
        manifest = root / 'completion_manifest.json'
        p = root / name
        require(sha(p) == read(manifest)['artifact_sha256'][name], 'sealed source changed: ' + str(p))
        sources[str(manifest)] = sha(manifest)
        sources[str(p)] = sha(p)
        return p

    events = read(sealed(LOCAL, 'prepare/R/events.json'))
    calls = [json.loads(s) for s in sealed(BOARD, 'board/results.jsonl').read_text().splitlines()]
    requests = [e for e in events if e['kind'] in ('load', 'store')]
    fields = ['domain', 'offset', 'sram', 'y', 'x', 'stride', 'pad']
    descriptors = [{k: e[k] for k in fields} for e in requests]
    traffic = []
    for domain in WIDTH:
        subset = [e for e in requests if e['domain'] == domain]
        counts = Counter(e['offset'] + y * e['stride'] + x
                         for e in subset for y in range(e['y']) for x in range(e['x']))
        require(set(counts) == set(range(len(counts))), 'non-contiguous tensor coverage')
        traffic.append(dict(tensor=LABEL[domain], domain=domain, request_count=len(subset),
                            vector_bytes=WIDTH[domain], payload_bytes=sum(counts.values()) * WIDTH[domain],
                            distinct_bytes=len(counts) * WIDTH[domain],
                            multiplicity_histogram=dict(Counter(counts.values()))))
    expected = {'INP': (50176, 25088), 'WGT': (294912, 294912),
                'ADD': (1024, 1024), 'OUT': (50176, 50176)}
    require(all((r['payload_bytes'], r['distinct_bytes']) == expected[r['tensor']] for r in traffic),
            'fixed R traffic differs')
    work, alu = Counter(), Counter()
    for e in events:
        if e['kind'] != 'kernel':
            continue
        steps = sum(math.prod(loop[0] for loop in u['loops']) for u in e['uops'])
        require(steps == e['expanded_uop_steps'], 'expanded-step disagreement')
        kind = ('RESET' if e['mode_reset'][1] else 'MAC') if e['mode'] == 'VTAPushGEMMOp' else 'ALU'
        work[kind] += steps
        if kind == 'ALU':
            alu[str(e['uops'][0]['args'][5:])] += 1
    require(work['MAC'] * 256 == 128 * 256 * 14 * 14 * 9, 'MAC work differs')
    keys = ['ALU', 'MAC_GEMM', 'RESET_GEMM', 'UOP_LOAD', 'UOP_LOAD_bytes',
            'UOP_SRAM_wraps', 'all_instructions', 'compute_NOP', 'serialized_uop_bytes',
            'single_kernel_max_base_uops', 'host_identity_base_uops_total',
            'host_content_unique_base_uops_total', 'submissions']
    organizations, native_counts = [], []
    for session in (3, 5, 7):
        for position in range(1, 7):
            prefix = f'board/session_{session:02d}'
            raw = sealed(BOARD, f'{prefix}/native_{position}.txt').read_text()
            require(native_requests(raw) == descriptors, 'native tensor request order differs from R events')
            org = read(sealed(BOARD, f'{prefix}/organization_{position}.json'))
            organizations.append({k: org[k] for k in keys})
            labels = re.findall(r'^INSTRUCTION \d+: (.+)$', raw, re.M)
            native_counts.append(dict(Counter(labels)))
    require(all(o == organizations[0] for o in organizations), 'R organization changes between diagnostic paths')
    require(all(n == native_counts[0] for n in native_counts), 'native instruction census differs')
    formal = [r for r in calls if r['program'] == 'R' and r['phase'] == 'formal']
    require(len(formal) == 18, 'R formal budget differs')
    for r in formal:
        require(r['status'] == 'PASS' and r['target_calls'] == 1 and r['timed'], 'invalid formal call')
        require(math.isfinite(r['elapsed_ms']) and r['elapsed_ms'] > 0, 'invalid time')
        require(r['collection_status']['detailed_clock_reads'] == 0, 'formal detailed clocks enabled')
        require(r['collection_status']['replay']['mode'] == 'disabled', 'formal replay enabled')
        require(r['submit_ledger']['recorded_submits'] == 1, 'submission count differs')
        submit = r['submit_ledger']['submits'][0]
        require(submit['completion'] and not submit['timeout'] and submit['insns'] == 73
                and submit['serialized_uop_bytes'] == 8192, 'formal submission differs')
    rounds = []
    for r in range(1, 7):
        rows = [v for v in formal if v['round'] == r]
        require(len(rows) == 3 and {v['seed'] for v in rows} == {0, 20250901, 20260910}, 'round coverage differs')
        rounds.append(statistics.median(v['elapsed_ms'] for v in rows))
    total = sum(t['payload_bytes'] for t in traffic)
    distinct = sum(t['distinct_bytes'] for t in traffic)
    for p in [Path(__file__), HERE / 'c3_official_window_runtime.py',
              HERE.parents[2] / 'vta/runtime/runtime.cc', HERE.parents[2] / '3rdparty/vta-hw/config/vta_config.json']:
        sources[str(p)] = sha(p)
    return dict(schema='c3_tps_remaining_cost_readonly_v1', status='PASS',
                new_target_calls=0, sources=sources, traffic=traffic,
                tensor_payload_bytes=total, distinct_tensor_payload_bytes=distinct,
                repeated_input_bytes=total-distinct, repeated_payload_fraction=(total-distinct)/total,
                vector_steps=dict(work), alu_opcode_imm_histogram=dict(alu),
                organization=organizations[0], native_instruction_counts=native_counts[0],
                ordered_native_request_comparisons=18, R_formal_calls_checked=18,
                round_medians_ms=rounds, T_R_ms=statistics.median(rounds),
                timing_limit='Complete-call only; no measured host/driver/device split in these formal records.',
                scope='Frozen node24 R; payload excludes UOP, instruction traffic, host copies and bus transaction overhead.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'refuse to overwrite output')
    result = audit()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: result[k] for k in ['status', 'new_target_calls', 'tensor_payload_bytes',
                                            'repeated_input_bytes', 'T_R_ms']}))


if __name__ == '__main__':
    main()
