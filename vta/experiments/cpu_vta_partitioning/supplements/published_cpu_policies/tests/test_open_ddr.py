import json
from pathlib import Path

import pytest

from cpu_policies.open_resources import common, ddr


def test_tensor_size_and_unknown_dynamic_shape():
    assert ddr.tensor_bytes({'shape': [1, 2, 3], 'dtype': 'int32'}) == 24
    assert ddr.tensor_bytes({'shape': [2], 'dtype': 'float16x4'}) == 16
    with pytest.raises(ValueError):
        ddr.tensor_bytes({'shape': ['n'], 'dtype': 'int8'})


def test_historical_rates_are_copied_labelled_and_payload_is_current(tmp_path, monkeypatch):
    api = common.api()
    monkeypatch.setattr(ddr, 'api', lambda unused=None: api)
    hardware = ddr.freeze_hardware_parameters(tmp_path)
    assert hardware['formal_calibration_ready'] is False
    assert all(not r['remeasured_this_campaign'] for r in hardware['provenance'].values())
    programs = {'cpu': {'device': 'cpu', 'input_types': [{'dtype': 'int8', 'shape': [100]}],
                        'output_types': [{'dtype': 'int8', 'shape': [20]}]},
                'vta': {'device': 'vta'}}
    traffic = {'vta': {'measurement': {'summary': {'hooks_present': True, 'count': 1,
                                                  'load_bytes': 1000, 'store_bytes': 200}}}}
    x = ddr.build_ddr(programs, traffic, tmp_path)
    assert set(x['stage_service_ms']['cpu']) == {'1', '2', '3', '4'}
    assert x['stage_service_ms']['vta']['1'] == pytest.approx(
        1000 / (hardware['vta_load_GBps'] * 1e6) + 200 / (hardware['vta_store_GBps'] * 1e6))
    assert x['provenance']['calibration_scope'].startswith('historical')
    with pytest.raises(ValueError, match='coverage'):
        ddr.build_ddr(programs, {}, tmp_path)
    (tmp_path / 'provenance/ddr/vta_bandwidth.json').write_text('{}')
    with pytest.raises(ValueError):
        ddr.build_ddr(programs, traffic, tmp_path)
