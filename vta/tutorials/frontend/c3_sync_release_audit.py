#!/usr/bin/env python3
"""Read sealed traces and classify synchronization hypotheses without executing them.

The graph hypotheses are independent cuts/moves from each original B, never
new TIR, instruction binaries or combined transformations. Access constraints
include independent OUT storage and separately flagged dead operand reads.
An audit candidate is not an executable, a hardware qualification or a timing.
"""
import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import re
import time

from c3_five_tradeoff_commands import decode_submissions

HERE = Path(__file__).resolve().parent
HOLD = HERE / 'report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout'
OFFICIAL = HOLD / '20261014_c3_official_valid_window_run01'
DOMAINS = {'INP': 2048, 'WGT': 1024, 'ACC': 2048, 'OUT': 2048, 'UOP': 8192}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def bits(mask):
    while mask:
        low = mask & -mask
        yield low.bit_length()-1
        mask ^= low


def address_mask(values, domain):
    result = 0
    for value in values:
        require(type(value) is int and 0 <= value < DOMAINS[domain], 'ADDRESS_OUTSIDE_' + domain)
        result |= 1 << value
    return result


@dataclass
class Access:
    reads: dict = field(default_factory=dict)
    writes: dict = field(default_factory=dict)
    dead_reads: dict = field(default_factory=dict)


class Graph:
    """Retain edge causes: removing a token must never remove a FIFO edge."""
    def __init__(self, records):
        self.records = records
        self.causes = defaultdict(set)
        self.tokens = []
        last, producers, consumers = {}, defaultdict(list), defaultdict(list)
        for i, row in enumerate(records):
            require(row['index'] == i, 'NONCONTIGUOUS_INSTRUCTION_ID')
            stage = row['stage']
            if stage in last:
                self.causes[last[stage], i].add('fifo')
            last[stage] = i
            a, b, c, d = row['bits']
            for flag, lane, push in ((a, (stage-1, stage), False), (b, (stage+1, stage), False),
                                     (c, (stage, stage-1), True), (d, (stage, stage+1), True)):
                if flag:
                    require(lane in ((1,2), (2,1), (2,3), (3,2)), 'UNKNOWN_TOKEN_LANE')
                    (producers if push else consumers)[lane].append(i)
        for lane in sorted(set(producers) | set(consumers)):
            require(len(producers[lane]) == len(consumers[lane]), 'UNMATCHED_TOKEN_' + str(lane))
            for ordinal, (source, target) in enumerate(zip(producers[lane], consumers[lane])):
                cause = 'token:%d:%d:%d' % (*lane, ordinal)
                self.causes[source, target].add(cause)
                self.tokens.append({'lane': list(lane), 'ordinal': ordinal,
                                    'producer': source, 'consumer': target, 'cause': cause})
        self.pred, self.succ = defaultdict(set), defaultdict(set)
        for source, target in self.causes:
            self.pred[target].add(source)
            self.succ[source].add(target)
        indegree = [len(self.pred[i]) for i in range(len(records))]
        queue = deque(i for i, n in enumerate(indegree) if not n)
        self.order = []
        while queue:
            i = queue.popleft()
            self.order.append(i)
            for j in self.succ[i]:
                indegree[j] -= 1
                if not indegree[j]:
                    queue.append(j)
        require(len(self.order) == len(records), 'BASELINE_GRAPH_CYCLE')
        self.ancestors = [0]*len(records)
        for i in self.order:
            for p in self.pred[i]:
                self.ancestors[i] |= self.ancestors[p] | (1 << p)

    def hypothesis(self, token, new_consumer=None):
        """Recompute only changed descendants; never mutate the original graph."""
        source, old = token['producer'], token['consumer']
        require(token['cause'] in self.causes[source, old], 'TOKEN_IDENTITY_NOT_IN_ORIGINAL')
        keep_old_edge = len(self.causes[source, old]) > 1
        result = list(self.ancestors)
        affected = {old}
        if new_consumer is not None:
            require(new_consumer > old and self.records[new_consumer]['stage'] == token['lane'][1],
                    'MOVE_NOT_LATER_ON_SAME_STAGE')
            require(not any(t['lane'] == token['lane'] and old < t['consumer'] <= new_consumer
                            for t in self.tokens), 'MOVE_CROSSES_OR_COLLIDES_WITH_TOKEN_CONSUMER')
            affected.add(new_consumer)
        for i in self.order:
            if i not in affected:
                continue
            value = 0
            for p in self.pred[i]:
                if p == source and i == old and not keep_old_edge:
                    continue
                value |= result[p] | (1 << p)
            if i == new_consumer:
                value |= result[source] | (1 << source)
            if value != result[i]:
                result[i] = value
                affected.update(self.succ[i])
        return result


def expand_contract(contract):
    loops = contract['loops']
    require(len(loops) <= 2 and all(0 < loop[0] < 16384 for loop in loops), 'UNKNOWN_LOOP_EXTENT')
    require(math.prod(loop[0] for loop in loops)*len(contract['args']) <= 5_000_000,
            'AUDIT_EXPANDED_KERNEL_LIMIT')
    for indices in product(*(range(loop[0]) for loop in loops)):
        a,b,c = (sum(i*loop[axis] for i,loop in zip(indices,loops)) for axis in (1,2,3))
        for row in contract['args']:
            yield row, row[2]+a, row[3]+b, row[4]+c


def kernel_access(contract):
    live, dead, writes = defaultdict(set), defaultdict(set), defaultdict(set)
    mode, reset = contract['args'][0][:2]
    for row, dst, src, wgt in expand_contract(contract):
        require(row[:2] == [mode, reset], 'MIXED_KERNEL_MODE')
        writes['ACC'].add(dst)
        writes['OUT'].add(dst)
        if mode == 0:
            target = dead if reset else live
            target['INP'].add(src)
            target['WGT'].add(wgt)
            target['ACC'].add(dst)
        else:
            require(mode == 1 and not reset, 'UNKNOWN_ALU_RESET')
            live['ACC'].add(dst)
            (dead if row[6] else live)['ACC'].add(src)
    return Access({d: address_mask(v, d) for d,v in live.items()},
                  {d: address_mask(v, d) for d,v in writes.items()},
                  {d: address_mask(v, d) for d,v in dead.items()})


def access_model(records, certificate):
    require(certificate['submission_count'] == 1, 'SINGLE_SUBMISSION_AUDIT_SUPPORT_ONLY')
    defs = certificate['cache_definitions']
    contracts = defs['contracts']
    require(set(contracts) == set(defs['definitions']), 'CONTRACT_DEFINITION_IDENTITIES_DIFFER')
    for key,contract in contracts.items():
        require(all(len(row) == 8 for row in contract['args']) and
                [row[2:5] for row in contract['args']] == defs['definitions'][key],
                'CONTRACT_DEFINITION_CONTENT_DIFFER')
    models = {key: kernel_access(value) for key,value in contracts.items()}
    checks = certificate['uop_values']['checks']
    expanded_steps = sum(len(contracts[c['kernel_identity']]['args']) *
                         math.prod(loop[0] for loop in contracts[c['kernel_identity']]['loops']) for c in checks)
    require(expanded_steps <= 5_000_000, 'AUDIT_COMPLETE_CALL_EXPANDED_STEP_LIMIT')
    materials = sorted(certificate['material_map'], key=lambda x: x['instruction'])
    result = [Access() for _ in records]
    filtered = [i for i,r in enumerate(records) if r['description'] == 'LOAD UOP' or
                r['description'] == 'GEMM' or r['description'].startswith('ALU -')]
    fifo_indices = {index:i for i,index in enumerate(filtered)}
    compute = 0
    for material in materials:
        i = material['instruction']
        row, native = records[i], material['native']
        require(row['description'] == material['description'], 'MATERIAL_BINDING_DESCRIPTION_CHANGED')
        if material['kind'] == 'kernel':
            check = checks[compute]
            require(check['values_match'] and check['compute_index'] == compute
                    and check['submission'] == 1 and check['fifo_index'] == fifo_indices[i]
                    and native['begin'] == check['uop_begin'] and native['end'] == check['uop_end'],
                    'KERNEL_BINDING_REFERENCE_CHANGED')
            key = check['kernel_identity']
            model = models[key]
            actual_mode = 0 if row['description'] == 'GEMM' else 1
            require(contracts[key]['args'][0][0] == actual_mode, 'KERNEL_BINDING_MODE_CHANGED')
            require(contracts[key]['args'][0][1] == int(re.search(r'reset_out: (\d+)', row['detail'])[1]),
                    'KERNEL_BINDING_RESET_CHANGED')
            loops = [list(loop) for loop in contracts[key]['loops']]
            default = [1,0,0,0]
            if actual_mode == 0:
                actual = [[int(n),int(dst),int(src),int(wgt)] for n,wgt,src,dst in
                          re.findall(r'loop - iter: (\d+), wgt: (\d+), inp: (\d+), acc: (\d+)', row['detail'])]
                expected = [loops[0] if loops else default, loops[1] if len(loops)==2 else default]
            else:
                actual = [[int(n),int(dst),int(src),0] for n,dst,src in
                          re.findall(r'loop - iter: (\d+), dst: (\d+), src: (\d+)', row['detail'])]
                expected = [loops[0] if len(loops)==2 else default, loops[-1] if loops else default]
            require(actual == expected, 'KERNEL_BINDING_HARDWARE_LOOP_CHANGED')
            result[i] = Access({**model.reads, 'UOP': address_mask(range(native['begin'], native['end']), 'UOP')},
                               dict(model.writes), dict(model.dead_reads))
            compute += 1
        elif material['kind'] in ('load', 'store'):
            if material['kind'] == 'store':
                domain = 'OUT'
                cells = (native['sram'] + y*native['x'] + x
                         for y in range(native['y']) for x in range(native['x']))
                result[i] = Access(reads={domain: address_mask(cells, domain)})
            else:
                domain = row['description'].split()[1]
                require(domain in ('INP', 'WGT', 'ACC'), 'UNKNOWN_MATERIAL_LOAD_DOMAIN')
                left, top, right, bottom = native['padding']
                width = left + native['x'] + right
                cells = range(native['sram'], native['sram'] + width*(top+native['y']+bottom))
                result[i] = Access(writes={domain: address_mask(cells, domain)})
        else:
            raise ValueError('UNKNOWN_MATERIAL_KIND')
    require(compute == len(checks), 'UNBOUND_COMPUTE_REFERENCE')
    for i,row in enumerate(records):
        if row['description'] == 'LOAD UOP':
            begin = int(re.search(r'SRAM:0x([0-9a-fA-F]+)', row['detail'])[1], 16)
            size = int(re.search(r'x: size=(\d+)', row['detail'])[1])
            result[i] = Access(writes={'UOP': address_mask(range(begin, begin+size), 'UOP')})
    # RESET operand values are dead only while its temporary OUT is unobserved.
    # Candidate constraints retain all OUT WAW/RAW/WAR orders from this trace.
    out_writer = [-1]*DOMAINS['OUT']
    for i,access in enumerate(result):
        for address in bits(access.reads.get('OUT',0)):
            writer = out_writer[address]
            require(writer >= 0 and not (records[writer]['description']=='GEMM' and
                    'reset_out: 1' in records[writer]['detail']), 'RESET_TEMPORARY_OUT_OBSERVED')
        for address in bits(access.writes.get('OUT',0)):
            out_writer[address] = i
    return result


def requirements(accesses, include_dead=False):
    """Exact vector-cell RAW/WAR/WAW; retain a concrete witness per pair/domain."""
    last = {d: [-1]*n for d,n in DOMAINS.items()}
    readers = {d: [0]*n for d,n in DOMAINS.items()}
    obligations = {}
    def add(a, b, domain, address, kind):
        if a >= 0 and a != b:
            obligations.setdefault((a, b, domain), {'source': a, 'target': b, 'domain': domain,
                                                  'address': address, 'kind': kind})
    for i,access in enumerate(accesses):
        reads = dict(access.reads)
        if include_dead:
            for domain,mask in access.dead_reads.items():
                reads[domain] = reads.get(domain, 0) | mask
        for domain,mask in reads.items():
            for address in bits(mask):
                add(last[domain][address], i, domain, address, 'RAW')
                readers[domain][address] |= 1 << i
        for domain,mask in access.writes.items():
            for address in bits(mask):
                add(last[domain][address], i, domain, address, 'WAW')
                for a in bits(readers[domain][address]):
                    add(a, i, domain, address, 'WAR')
                last[domain][address] = i
                readers[domain][address] = 0
    return list(obligations.values())


class Obligations:
    def __init__(self, requirements, original_ancestors, must_hold):
        self.masks = defaultdict(int)
        self.witnesses = {}
        self.original_unordered = []
        for row in requirements:
            a,b = row['source'],row['target']
            if not original_ancestors[b] >> a & 1:
                self.original_unordered.append(row)
                continue
            self.masks[b] |= 1 << a
            self.witnesses.setdefault((a,b), row)
        if must_hold:
            require(not self.original_unordered, 'ORIGINAL_LIVE_MEMORY_ORDER_UNRESOLVED:' +
                    str(self.original_unordered[:1]))
        self.targets = sorted(self.masks)

    def missing(self, ancestors):
        for b in self.targets:
            lost = self.masks[b] & ~ancestors[b]
            if lost:
                a = (lost & -lost).bit_length()-1
                return self.witnesses[a,b]
        return None


def classify(graph, live, physical, token, new_consumer=None):
    changed = graph.hypothesis(token, new_consumer)
    witness = live.missing(changed)
    if witness:
        return {'category': 'NECESSARY_PROTECTION', 'witness': witness}
    witness = physical.missing(changed)
    if witness:
        return {'category': 'HARDWARE_OBSERVABILITY_UNRESOLVED', 'witness': witness}
    alt = bool(changed[token['consumer']] >> token['producer'] & 1)
    if alt:
        return {'category': 'UNCHANGED_ORDER', 'producer_to_old_consumer_still_ordered': True}
    return {'category': 'POSSIBLE_EARLIER_RELEASE',
            'all_original_live_and_ordered_physical_conflicts_preserved': True,
            'executable_or_combined_transform_certified': False}


def later_weight_target(graph, token):
    """Same LOAD FIFO prefix, without passing another consumer on this lane."""
    old = token['consumer']
    if graph.records[old]['description'] != 'LOAD INP':
        return None
    blocked = {t['consumer'] for t in graph.tokens if t['lane'] == token['lane']}
    for i in range(old+1, len(graph.records)):
        row = graph.records[i]
        if row['stage'] != 1:
            continue
        if i in blocked:
            return None
        if row['description'] == 'LOAD WGT':
            return i
        if row['description'] != 'LOAD INP':
            return None
    return None


def analyze(records, certificate):
    graph = Graph(records)
    accesses = access_model(records, certificate)
    live_rows = requirements(accesses)
    physical_rows = requirements(accesses, include_dead=True)
    live = Obligations(live_rows, graph.ancestors, True)
    physical = Obligations(physical_rows, graph.ancestors, False)
    first_mac = next(r['index'] for r in records if r['description'] == 'GEMM' and 'reset_out: 0' in r['detail'])
    census, cuts, moves = Counter(), [], []
    for token in graph.tokens:
        if token['lane'] != [2,1]:
            continue
        p,c = token['producer'],token['consumer']
        producer,consumer = records[p],records[c]
        reset = producer['description'] == 'GEMM' and 'reset_out: 1' in producer['detail']
        if reset and consumer['description'] in ('LOAD INP', 'LOAD WGT'):
            census['reset_to_operand'] += 1
            census['startup' if p < first_mac else 'later'] += 1
        row = {**token, 'producer_description': producer['description'],
               'consumer_description': consumer['description'], 'reset_producer': reset,
               'phase': 'startup' if p < first_mac else 'after_first_MAC',
               **classify(graph, live, physical, token)}
        cuts.append(row)
        target = later_weight_target(graph, token)
        if target is not None:
            move = {**token, 'new_consumer': target, 'reset_producer': reset,
                    'phase': row['phase'], **classify(graph, live, physical, token, target)}
            prefix = [i for i in range(c,target) if records[i]['stage'] == 1]
            move['released_INP_instructions'] = prefix
            move['prefix_vector_cells'] = sum(accesses[i].writes.get('INP',0).bit_count() for i in prefix)
            moves.append(move)
    stages = Counter(r['stage'] for r in records)
    lanes = Counter(tuple(t['lane']) for t in graph.tokens)
    batch_fits = all(stages[s] <= 512 for s in (1,2,3)) and all(n <= 1024 for n in lanes.values())
    for row in moves:
        row['finite_fifo'] = 'SUFFICIENT_FULL_BATCH_FITS' if batch_fits else 'UNRESOLVED_LARGE_BATCH'
    reset_rows = [r for r in cuts if r['reset_producer'] and r['consumer_description'] in ('LOAD INP','LOAD WGT')]
    return {'status': 'READONLY_GRAPH_AND_ACCESS_PREAUDIT_COMPLETE', 'instructions': len(records),
            'census': dict(census), 'g2l_tokens': len(cuts), 'cuts': cuts, 'moves': moves,
            'cut_categories': dict(Counter(r['category'] for r in cuts)),
            'reset_cut_categories': dict(Counter(r['category'] for r in reset_rows)),
            'move_categories': dict(Counter(r['category'] for r in moves)),
            'live_obligations': len(live_rows), 'physical_obligations': len(physical_rows),
            'original_dead_read_unordered_count': len(physical.original_unordered),
            'original_dead_read_unordered_examples': physical.original_unordered[:4],
            'finite_full_batch_fits': batch_fits,
            'limits': ['Each hypothesis starts from original B; combinations are unproved',
                       'No new program, command encoding, cycle model or timing',
                       'Existing numerically dead RESET read races are not called baseline errors',
                       'Candidate gate checks every original live order and originally ordered physical conflict',
                       'Only one actual explicit submission supported; no ABI/DRAM alias relaxation',
                       'Every STORE reads a non-RESET final OUT writer; all OUT ordering is retained',
                       'No proof of a more faithful author dthread baseline']}


def inventory():
    rows = []
    seal_path = OFFICIAL / 'completion_manifest.json'
    sealed = read(seal_path)['artifact_sha256']
    # Use the final, frozen evidence index, not partially written failed workers.
    for record in read(OFFICIAL / 'closure_final02/capacity_report.json'):
        dump = Path(record['native_dump'])
        cert = dump.with_name('native_certificate_01.json')
        rows.append({'object_id': dump.parts[-3], 'family': 'official',
                     'dump': str(dump), 'certificate': str(cert),
                     'expected_source_sha256': {str(p):sealed[str(p.relative_to(OFFICIAL))] for p in (dump,cert)},
                     'source_seal': str(seal_path), 'source_seal_sha256': sha(seal_path)})
    require(len(rows) == 26 and len({r['object_id'] for r in rows}) == 26, 'OFFICIAL_INVENTORY_NOT_26')
    directory = HOLD / '20261014_c3_tps_window_complete_run03/TPS_B_detailed'
    seal_path = HOLD / '20261015_c3_tps_aggregation_run02/freeze.json'
    sealed = read(seal_path)['source_and_artifact_bindings']
    dump,cert = directory/'native_call_01.txt',directory/'native_certificate_01.json'
    rows.append({'object_id': 'TPS_node24_adapted', 'family': 'TPS_adapted',
                 'dump': str(dump), 'certificate': str(cert),
                 'expected_source_sha256': {str(p):sealed[str(p)] for p in (dump,cert)},
                 'source_seal': str(seal_path), 'source_seal_sha256': sha(seal_path)})
    return rows


def run_one(row, output):
    started = time.monotonic()
    for path,digest in row['expected_source_sha256'].items():
        require(sha(path) == digest, 'SEALED_SOURCE_IDENTITY_CHANGED:' + path)
    require(sha(row['source_seal']) == row['source_seal_sha256'], 'SOURCE_SEAL_CHANGED')
    text = Path(row['dump']).read_text()
    cert = read(row['certificate'])
    require(sha(row['dump']) == cert['dump_sha256'], 'DUMP_CERTIFICATE_IDENTITY_MISMATCH')
    require('CERTIFIED' in cert['status'], 'SOURCE_CERTIFICATE_NOT_SUCCESSFUL')
    blocks = decode_submissions(text)
    require(len(blocks) == 1, 'SINGLE_SUBMISSION_AUDIT_SUPPORT_ONLY')
    result = analyze(blocks[0], cert)
    result.update(object_id=row['object_id'], family=row['family'], sources=row,
                  source_sha256={p: sha(p) for p in (row['dump'],row['certificate'])},
                  elapsed_seconds=time.monotonic()-started)
    write(output, result)
    print(json.dumps({k:result[k] for k in ('object_id','status','census','reset_cut_categories','move_categories','elapsed_seconds')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', action='store_true')
    parser.add_argument('--row', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.inventory:
        print(json.dumps(inventory(), indent=2))
    else:
        run_one(read(args.row), args.output)


if __name__ == '__main__':
    main()
