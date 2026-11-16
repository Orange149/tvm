"""Retain the DDR resource bound with current program traffic and dated rates.

Bandwidth rates are explicitly historical component parameters. They are not
promoted to a fresh calibration or fitted to complete-pipeline throughput.
"""
import math
from pathlib import Path
import re
import shutil

from .common import api

_REPO = Path('/home/orange/code/tvm')
_ARCHIVE = _REPO / 'vta/tutorials/frontend/report_out/resource_aware_maxplus'


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('invalid ' + name)
    return float(value)


def freeze_hardware_parameters(bundle, archive=None):
    """Copy source evidence so a relocated bundle never needs historical paths."""
    bundle, archive = Path(bundle), Path(archive or _ARCHIVE)
    common = api()
    files = {'cpu_bandwidth.json': archive / 'v1_local_cost_table.json',
             'vta_bandwidth.json': archive / 'stage4a_q_qualification/session1/dma_identification_summary.json'}
    provenance = {}
    for name, source in files.items():
        if not source.is_file() or source.is_symlink():
            raise ValueError('missing historical DDR component source: ' + str(source))
        target = bundle / 'provenance/ddr' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        provenance[name] = {'path': str(target.relative_to(bundle)), 'sha256': common.sha256(target),
                            'original_path': str(source), 'remeasured_this_campaign': False}
    cpu, dma = (common.read(bundle / provenance[k]['path']) for k in files)
    if not dma.get('contiguous_load_store_identified'):
        raise ValueError('historical DMA LOAD/STORE parameters unidentified')
    rates = {}
    for row in cpu['shared_ddr']:
        if row['operation'] in ('read', 'write'):
            if row.get('correct') is not True:
                raise ValueError('historical bandwidth correctness failed')
            rates[f"{row['operation']}_t{row['threads']}"] = _positive(row['bandwidth_GBps_median'], 'CPU bandwidth')
    if set(rates) != {f'{op}_t{n}' for op in ('read', 'write') for n in range(1, 5)}:
        raise ValueError('incomplete historical CPU read/write rates')
    result = {'cpu_GBps': rates,
              'vta_load_GBps': _positive(dma['fitted_load_bandwidth_GBps'], 'DMA LOAD bandwidth'),
              'vta_store_GBps': _positive(dma['fitted_store_bandwidth_GBps'], 'DMA STORE bandwidth'),
              'provenance': provenance,
              'calibration_scope': 'historical_component_parameters_not_current_calibration',
              'formal_calibration_ready': bool(dma.get('formal_calibration_ready', False)),
              'hardware_equivalence': 'inherited_component_model_assumption_not_new_board_verification'}
    common.write(bundle / 'plans/ddr_hardware.json', result, replace=True)
    return result


def tensor_bytes(tensor):
    dtype = tensor['dtype']
    match = re.fullmatch(r'(?:u?int|float|bfloat)(\d+)(?:x(\d+))?', dtype)
    if dtype == 'bool':
        bits, lanes = 8, 1
    elif match:
        bits, lanes = int(match[1]), int(match[2] or 1)
    else:
        raise ValueError('unsupported tensor dtype: ' + dtype)
    count = 1
    for dim in tensor['shape']:
        if type(dim) is not int or dim < 0:
            raise ValueError('DDR bound requires static nonnegative tensor shapes')
        count *= dim
    return (count * bits * lanes + 7) // 8


def build_ddr(programs, traffic_records, bundle):
    common = api(bundle)
    bundle = Path(bundle)
    hardware = common.read(bundle / 'plans/ddr_hardware.json')
    common.verify_inventory(bundle, {r['path']: {'sha256': r['sha256'],
             'bytes': (bundle / r['path']).stat().st_size} for r in hardware['provenance'].values()})
    expected = {p for p, spec in programs.items() if spec['device'] == 'vta'}
    if set(traffic_records) != expected:
        raise ValueError('VTA DDR traffic coverage differs from complete program domain')
    table, traffic_sources = {}, {}
    for name, p in programs.items():
        if p['device'] == 'cpu':
            inputs = sum(tensor_bytes(t) for t in p['input_types'])
            outputs = sum(tensor_bytes(t) for t in p['output_types'])
            table[name] = {str(n): inputs / (hardware['cpu_GBps'][f'read_t{n}'] * 1e6)
                          + outputs / (hardware['cpu_GBps'][f'write_t{n}'] * 1e6) for n in range(1, 5)}
        else:
            record = traffic_records[name]
            s = record.get('measurement', {}).get('summary', record.get('summary', record))
            if s.get('hooks_present') is not True or s.get('count') != 1:
                raise ValueError('unverified VTA single-run traffic')
            load = _positive(s['load_bytes'], 'VTA LOAD payload')
            store = _positive(s['store_bytes'], 'VTA STORE payload')
            table[name] = {'1': load / (hardware['vta_load_GBps'] * 1e6)
                          + store / (hardware['vta_store_GBps'] * 1e6)}
            traffic_sources[name] = {'load_bytes': load, 'store_bytes': store,
                                    'measurement': record.get('measurement', {})}
    return {'stage_service_ms': table, 'provenance': hardware,
            'traffic_sources': traffic_sources, 'formula': 'max_with_sum_stage_DDR_service_not_added_to_wall_service',
            'assumptions': ['CPU logical input/read and output/write lower bound, not complete cache/weight traffic',
                            'VTA counters describe requested DMA payload, not measured AXI bus bytes',
                            'historical bandwidth parameters are provisional, not freshly calibrated',
                            'set/get wall and core work already included; no second fitted boundary charge',
                            'no old-graph GOP-to-traffic regression and no pipeline FPS fitting']}
