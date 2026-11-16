"""Read-only source provenance and descriptor equality for RESET candidates."""
from pathlib import Path
import json
from collections import Counter
from c3_yolo224_sync_structure import HERE, HOLD, read, write, sha, require, classify_events

REVIEW = HOLD / '20261016_c3_reset_opportunity_review_readonly01'
AUTHOR = HOLD / '20261012_c3_banerjee_double_buffer_local_run01'
TPS = HOLD / '20261016_c3_tps_sync_release_local_run04'


def static_inputs(fn):
    import tvm
    from tvm import tir
    from c3_valid_window import call_info, integer
    found = []
    def walk(n, loops=(), branches=()):
        if isinstance(n, tir.For):
            walk(n.body, loops + (n,), branches)
        elif isinstance(n, tir.IfThenElse):
            walk(n.then_case, loops, branches + ((n.condition, True),))
            if n.else_case is not None: walk(n.else_case, loops, branches + ((n.condition, False),))
        elif isinstance(n, tir.SeqStmt):
            for c in n.seq: walk(c, loops, branches)
        elif hasattr(n, 'body'):
            walk(n.body, loops, branches)
        elif isinstance(n, tir.Evaluate):
            c = call_info(n)
            if c and c[0] == 'VTALoadBuffer2D' and integer(c[1][-1]) == 2:
                found.append((c[1], loops, branches))
    walk(fn.body)
    require(len(found) == 2, 'EXPECTED_TWO_INPUT_LOAD_SITES')
    a, b = found
    require(not a[2] and not b[2], 'CONDITIONAL_INPUT_SITE')
    require(len(a[1]) == len(b[1]) and all(x.same_as(y) for x, y in zip(a[1], b[1])), 'DIFFERENT_LOOP_CONTEXT')
    fields = [tvm.ir.structural_equal(a[0][i], b[0][i]) for i in list(range(1, 10)) + [11]]
    require(all(fields), 'INPUT_REQUESTS_NOT_IDENTICAL')
    return {'same_DDR_tensor_offset_extent_stride_padding_domain': True,
            'same_enclosing_loop_objects': True, 'no_conditional_context': True,
            'source_sram': integer(a[0][10]), 'duplicate_sram': integer(b[0][10]),
            'descriptor': [str(x) for x in a[0][1:10]],
            'loops': [{'var': str(n.loop_var), 'min': str(n.min), 'extent': str(n.extent)} for n in a[1]]}


def main():
    import tvm, vta
    seal = read(REVIEW / 'completion_manifest.json')
    for name, digest in seal['artifacts'].items(): require(sha(REVIEW / name) == digest, 'REVIEW_CHANGED')
    out = HOLD / '20261016_c3_reset_opportunity_provenance_readonly01'
    out.mkdir(exist_ok=False)
    inventory = read(REVIEW / 'inventory.json')
    source_files = [Path(__file__), HERE / 'c3_banerjee_completion.py', HERE / 'c3_banerjee_local.py',
                    HERE / 'c3_tps_sync_release_prepare.py', HERE.parents[2] / 'src/tir/transforms/coproc_sync.cc',
                    HERE.parents[2] / 'vta/python/vta/build_module.py']
    author_manifest = read(AUTHOR / 'local_manifest.json')['artifacts']
    for name in ['build_module.py', 'coproc_sync.cc', 'inject_virtual_thread.cc', 'vta_conv2d.py']:
        p = AUTHOR / 'author_sources' / name
        require(sha(p) == author_manifest['author_sources/' + name], 'AUTHOR_SOURCE_CHANGED')
        source_files.append(p)
    tps_manifest = read(TPS / 'completion_manifest.json')['artifact_sha256']
    require(sha(TPS / 'prepare/R/events.json') == tps_manifest['prepare/R/events.json'], 'TPS_EVENTS_CHANGED')
    source_files.append(TPS / 'prepare/R/events.json')
    for name in ['prepare/R/022_before_tir_c3_ImprovedDoubleBuffer.tir',
                 'prepare/R/023_after_tir_c3_ImprovedDoubleBuffer.tir']:
        require(sha(TPS / name) == tps_manifest[name], 'TPS_PASS_EVIDENCE_CHANGED')
        source_files.append(TPS / name)
    write(out / 'freeze.json', {'source_hashes': {str(p): sha(p) for p in source_files},
          'review_manifest_sha256': sha(REVIEW / 'completion_manifest.json'), 'target_calls': 0})
    unique = {r['object_id']: r for r in inventory['graph_calls']}
    proofs = {}
    for key, row in unique.items():
        require(sha(row['tir']) == row['sha256'], 'TIR_CHANGED')
        proof = static_inputs(tvm.ir.load_json(Path(row['tir']).read_text()))
        dynamic = read(REVIEW / 'objects' / (key + '.json'))
        require(all(len(r['input_ranges']) == 1 and r['input_ranges'][0][0] == proof['duplicate_sram']
                    for r in dynamic['rows']), 'CANDIDATE_NOT_SECOND_INPUT')
        proof.update(source=row, dynamic_candidates_bound=dynamic['candidate_count'])
        proofs[key] = proof
    tps = classify_events(read(TPS / 'prepare/R/events.json'))
    reset = [r for r in tps['waits'] if r['producer_kind'] == 'RESET']
    rows = []
    for entry in inventory['graph_calls']:
        obj = read(REVIEW / 'objects' / (entry['object_id'] + '.json'))
        rr = obj['rows']
        rows.append({'node': entry['node'], 'candidates': len(rr),
              'duplicate_input_candidates': proofs[entry['object_id']]['dynamic_candidates_bound'],
              'phase_counts': obj['phases'], 'class_counts': obj['structural_classes'],
              'RESET_steps_per_candidate': sorted({r['producer_vector_steps'] for r in rr}),
              'input_bytes_total': sum(r['input_DDR_bytes'] for r in rr),
              'reset_steps_total': sum(r['producer_vector_steps'] for r in rr),
              'store_bytes_total': sum(m.get('bytes', 0) for r in rr for m in r['all_reset_moves']['materials'] if m['kind'] == 'STORE'),
              'new_physical_conflicts': obj['direct_physical_conflict_count'],
              'released_MAC_count': obj['released_MAC_count'],
              'single_joint_all_equal': all(r['single_and_joint_material_loss_equal'] for r in rr)})
    write(out / 'descriptor_proofs.json', proofs)
    write(out / 'tps_comparison.json', {'R_RESET_waits': reset, 'R_RESET_categories': tps['RESET_categories'],
          'R_MAC_categories': tps['MAC_categories'], 'source': str(TPS / 'prepare/R/events.json'),
          'full_author_compiler_equivalence_claim': False})
    write(out / 'summary.json', {'nodes': rows, 'all_candidates_duplicate_input': sum(r['duplicate_input_candidates'] for r in rows),
          'R_RESET_input_disjoint_candidates': tps['RESET_categories'].get('ADDRESS_DISJOINT_PREFIX_NEEDS_FULL_CERTIFICATE', 0),
          'paper': 'https://arxiv.org/html/2111.15024#S4.SS4.SSS2',
          'source_conclusion': 'All reviewed prefixes repeat the first input request in the same loop iteration; redundant alternate-thread loads are already addressed by Banerjee; sealed adapted TPS R has no RESET disjoint INP prefix.',
          'scope_limit': 'No full-network author compiler port, native FIFO proof or performance bound in milliseconds.'})
    for p, digest in read(out / 'freeze.json')['source_hashes'].items(): require(sha(p) == digest, 'SOURCE_CHANGED')
    write(out / 'completion_manifest.json', {'status': 'SOURCE_PROVENANCE_REVIEW_COMPLETE',
          'artifacts': {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}})
    print(out)


if __name__ == '__main__': main()
