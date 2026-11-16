"""Offline arithmetic audit of sealed TPS R evidence; never executes a target.

Input clock_snapshot.txt is a separately captured, read-only SSH observation.
The issue-slot model is not an RTL cycle simulation or a measured utilization.
"""
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HOLD = HERE / 'report_out/stage_tile_cotuning/c3_dma_residency_autotune/07_grouped_holdout'
HW = ROOT / '3rdparty/vta-hw/build/hardware/xilinx/hls/axu5evb_1x16_i8w8a32_15_15_18_17'
LOCAL = HOLD / '20261016_c3_tps_sync_release_local_run04'
BOARD = HOLD / '20261017_c3_tps_cost_board_run01'
PROBE = HOLD / '20261017_c3_tps_cost_probe_local_run01'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode(text):
    result = []
    for block in re.split(r'(?=INSTRUCTION \d+:)', text):
        head = re.match(r'INSTRUCTION (\d+): (GEMM|ALU[^\n]*)\n', block)
        if not head:
            continue
        lo, hi = map(int, re.search(r'range \((\d+), (\d+)\)', block).groups())
        outer = int(re.search(r'outer loop - iter: (\d+)', block)[1])
        inner = int(re.search(r'inner loop - iter: (\d+)', block)[1])
        reset = int(re.search(r'reset_out: (\d+)', block)[1])
        kind = 'ALU' if head[2].startswith('ALU') else ('RESET' if reset else 'MAC')
        result.append({'instruction': int(head[1]), 'kind': kind,
                       'steps': (hi-lo)*outer*inner, 'ii': 2 if kind == 'ALU' else 1})
    return result


def audit(out):
    require(not (out / 'audit.json').exists(), 'refuse overwrite')
    sources = {}

    def bind(path):
        sources[str(path)] = sha(path)
        return Path(path)

    old_manifest = json.loads(bind(BOARD / 'completion_manifest.json').read_text())
    for name, expected in old_manifest['artifact_sha256'].items():
        require(sha(BOARD/name) == expected, f'sealed board artifact changed: {name}')
    archive = json.loads(bind(LOCAL/'hardware_archive.json').read_text())
    for name, expected in archive['source_bindings'].items():
        require(sha(name) == expected, f'hardware source changed: {name}')
    for bit in archive['bitstreams']:
        data = Path(bit['path']).read_bytes()
        require(hashlib.sha256(data[bit['payload_offset']:]).hexdigest() == bit['payload_sha256'], 'bit payload')

    hwh = bind(HW/'vta.srcs/sources_1/bd/vta/hw_handoff/vta.hwh')
    require(sha(hwh) == archive['hwh_sha256'], 'HWH identity')
    with ZipFile(HW/'vta_wrapper.xsa') as z:
        names = [n for n in z.namelist() if n == 'vta.hwh']
        require(len(names) == 1 and z.read(names[0]) == hwh.read_bytes(), 'embedded HWH')
    modules = {m.get('INSTANCE'): m for m in ET.parse(hwh).getroot().iter('MODULE')}
    ports = {name: {p.get('NAME'): p.attrib for p in modules[name].iter('PORT')}
             for name in ['processing_system', 'pll_clk', 'compute_0']}
    require(ports['processing_system']['pl_clk0']['SIGNAME'] == ports['pll_clk']['clk_in1']['SIGNAME'], 'PS PLL wiring')
    require(ports['pll_clk']['clk_out1']['SIGNAME'] == ports['compute_0']['ap_clk']['SIGNAME'], 'compute clock wiring')
    params = {p.get('NAME'): p.get('VALUE') for p in modules['pll_clk'].iter('PARAMETER')}
    ratio = float(params['C_MMCM_CLKFBOUT_MULT_F']) / float(params['C_MMCM_DIVCLK_DIVIDE']) / float(params['C_MMCM_CLKOUT0_DIVIDE_F'])
    require(ratio == 1 and params['C_USE_DYN_RECONFIG'] == '0', 'frozen clock ratio')
    snap = bind(out/'clock_snapshot.txt').read_text()
    rate = int(re.search(r'^\s*pl0_ref\s+\d+\s+\d+\s+\d+\s+(\d+)', snap, re.M)[1])
    preflight = json.loads(bind(BOARD/'board/preflight.json').read_text())
    require(snap.splitlines()[0] == preflight['boot_id'], 'boot changed')
    require(archive['bitstreams'][0]['whole_sha256'] in snap and '\noperating\n' in snap, 'firmware file or FPGA state')

    report = bind(HW/'vta_compute/soln/syn/report/compute_csynth.rpt').read_text()
    for label, latency, ii in [('GEMM', 4, 1), ('ALU', 5, 2)]:
        line = next(l for l in report.splitlines() if f'EXE_OUT_LOOP_EXE_IN_LOOP_READ_{label}_UOP' in l)
        cells = [c.strip() for c in line.split('|')]
        require(int(cells[4]) == latency and int(cells[5]) == ii, f'HLS {label} schedule')

    events = json.loads(bind(LOCAL/'prepare/R/events.json').read_text())
    expected = Counter()
    for e in events:
        if e['kind'] == 'kernel':
            kind = 'ALU' if e['mode'] == 'VTAPushALUOp' else ('RESET' if e['mode_reset'][1] else 'MAC')
            expected[kind] += e['expanded_uop_steps']
    trace_evidence = []
    first = None
    for session in [3, 4]:
        for position in [1, 2, 3, 4, 5, 6]:
            path = BOARD / f'board/session_{session:02d}/native_{position}.txt'
            if not path.exists():
                # Prefix calls do not emit detailed dumps; warm calls must exist.
                require(position < 4, 'missing warm diagnostic')
                continue
            seq = decode(bind(path).read_text())
            totals = Counter()
            for row in seq:
                totals[row['kind']] += row['steps']
            require(totals == expected, f'TIR versus board work mismatch: {path}')
            if first is None:
                first = seq
            require(first == seq, 'diagnostic compute organization mismatch')
            trace_evidence.append(str(path))
    require(len(trace_evidence) >= 6, 'insufficient warm diagnostic traces')
    require(expected == {'MAC': 225792, 'RESET': 3136, 'ALU': 12544}, 'unexpected R work')
    stats = json.loads(bind(BOARD/'statistics.json').read_text())
    driver_ms = stats['on_formal_parts']['driver_run_and_wait_ms']['median_ms']
    slots = sum(row['steps'] * row['ii'] for row in first)
    spans = sum((row['steps']-1)*row['ii']+1 for row in first)
    scenarios = []
    for label, hz in [('design_nominal', 100000000), ('current_CCF_rate_inference', rate*ratio)]:
        ms = 1000*slots/hz
        scenarios.append({'basis': label, 'clock_hz': hz, 'issue_budget_ms': ms,
                          'issue_budget_div_driver_percent': 100*ms/driver_ms,
                          'driver_minus_model_ms': driver_ms-ms,
                          'not_measured_idle_time': True})

    driver = bind(PROBE/'runtime_arm/c3_official_window_runtime_driver_generated.cc').read_text()
    for key in ['POST_START', 'POLL']:
        require(f'GetEnvUInt64OrDefault("AXU5EVB_DRIVER_{key}_SLEEP_NS", 1000)' in driver, 'sleep default')
    require('std::this_thread::yield();' in driver and 'nanosleep(&ts, nullptr);' in driver, 'sleep implementation')
    env_records = []
    for s in range(1, 13):
        start = json.loads(bind(BOARD/f'board/session_{s:02d}/start.json').read_text())
        env_records.append({k: v for k, v in start['environment'].items() if 'SLEEP_NS' in k})
    bind(HW/'vta_compute/soln/.autopilot/db/vta.pragma.0.cc')
    bind(HW/'vta.runs/impl_1/vta_wrapper_timing_summary_routed.rpt')
    bind(Path(__file__).resolve())
    result = {
        'status': 'READ_ONLY_AUDIT_COMPLETE', 'new_target_calls': 0,
        'sealed_board_artifacts_verified': len(old_manifest['artifact_sha256']),
        'hardware_source_bindings_verified': len(archive['source_bindings']),
        'clock': {'boot_id': preflight['boot_id'], 'current_pl0_ref_hz': rate,
                  'mmcm_ratio': ratio, 'ports': ports,
                  'historical_sample_clock_attested': False, 'physical_frequency_measured': False},
        'work': dict(expected), 'instructions': first, 'diagnostic_sources': trace_evidence,
        'issue_slots': slots, 'within_instruction_issue_spans': spans,
        'model_omits': ['pipeline fill/drain and command transitions', 'DMA and AXI stalls',
                       'FIFO waits', 'MMIO launch', 'host completion observation delay'],
        'driver_interval_median_ms': driver_ms, 'scenarios': scenarios,
        'driver_wait': {'default_post_start_ns': 1000, 'default_poll_ns': 1000,
                        'zero_means': 'thread yield, not busy polling',
                        'wait_cycles_means': 'maximum polling iterations',
                        'frozen_session_sleep_overrides': env_records,
                        'effective_historical_sleep_environment_attested': False,
                        'sleep_latency_or_completion_tail_measured': False},
        'external_api_reference': 'https://man7.org/linux/man-pages/man2/nanosleep.2.html',
        'source_sha256': sources,
    }
    (out/'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k: result[k] for k in ['status', 'work', 'issue_slots', 'scenarios']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    audit(parser.parse_args().output.resolve())
