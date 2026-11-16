#!/usr/bin/env python3
"""Stage01 local bitstream association and isolated current-source build."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import zipfile
import xml.etree.ElementTree as ET

from thesis_supplement_run import RunStore, sha256, write_new_json

ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / '3rdparty/vta-hw'
PROJECT = HW / 'build/hardware/xilinx/hls/axu5evb_1x16_i8w8a32_15_15_18_17'
HEADER = PROJECT / 'vta_compute/soln/impl/ip/drivers/compute_v1_0/src/xcompute_hw.h'


def parse_bit(data):
    offset = 0

    def take(size):
        nonlocal offset
        if offset + size > len(data):
            raise ValueError('Truncated bit container')
        part = data[offset:offset + size]
        offset += size
        return part

    def integer(size):
        return int.from_bytes(take(size), 'big')

    size = integer(2)
    if take(size) != bytes.fromhex('0ff00ff00ff00ff000'):
        raise ValueError('Unexpected Xilinx container magic')
    if integer(2) != 1:
        raise ValueError('Unexpected first key length')
    metadata = {}
    for key in b'abcd':
        if take(1) != bytes([key]):
            raise ValueError('Unexpected TLV key order')
        value = take(integer(2))
        if not value.endswith(b'\0'):
            raise ValueError('Unterminated TLV metadata')
        metadata[chr(key)] = value[:-1].decode('ascii')
    if take(1) != b'e':
        raise ValueError('Missing configuration payload')
    size = integer(4)
    payload_offset = offset
    payload = take(size)
    if offset != len(data) or size == 0:
        raise ValueError('Trailing bytes or empty payload')
    return {'metadata': metadata, 'payload_offset': payload_offset,
            'payload_bytes': size, 'payload_sha256': hashlib.sha256(payload).hexdigest()}, payload


def associate(run):
    paths = [ROOT / 'apps/vta_rpc/firmware/vta_hpc.bit', PROJECT / 'export/vta.bit',
             PROJECT / 'vta.runs/impl_1/vta_wrapper.bit']
    rows, payloads = [], []
    for path in paths:
        row, payload = parse_bit(path.read_bytes())
        rows.append({'path': str(path), 'file_sha256': sha256(path), **row})
        payloads.append(payload)
    if not all(value == payloads[0] for value in payloads):
        raise ValueError('Configuration payload differs')
    xsa = PROJECT / 'vta_wrapper.xsa'
    hwh = PROJECT / 'vta.srcs/sources_1/bd/vta/hw_handoff/vta.hwh'
    with zipfile.ZipFile(xsa) as archive:
        entries = archive.namelist()
        checks = []
        for path, basename in [(paths[2], 'vta_wrapper.bit'), (hwh, 'vta.hwh'), (HEADER, 'xcompute_hw.h')]:
            matched = [name for name in entries if Path(name).name == basename]
            if len(matched) != 1 or archive.read(matched[0]) != path.read_bytes():
                raise ValueError('XSA association failed: ' + basename)
            checks.append({'xsa_entry': matched[0], 'local_path': str(path), 'sha256': sha256(path)})
    rtl = []
    for name in ('compute.v', 'compute_CONTROL_BUS_s_axi.v'):
        original = PROJECT / 'vta_compute/soln/impl/ip/hdl/verilog' / name
        actual = PROJECT / 'vta.srcs/sources_1/bd/vta/ipshared/a8f1/hdl/verilog' / name
        if original.read_bytes() != actual.read_bytes():
            raise ValueError('Actual RTL differs from HLS export')
        rtl.append({'export_path': str(original), 'actual_path': str(actual), 'sha256': sha256(actual)})
    root = ET.parse(hwh).getroot()
    modules = {m.attrib['INSTANCE']: m for m in root.iter('MODULE')}
    bindings = {}
    for instance in ('fetch_0', 'load_0', 'compute_0', 'store_0', 'pll_clk'):
        module = modules[instance]
        bindings[instance] = {
            'parameters': [p.attrib for p in module.find('PARAMETERS')
                           if any(k in p.attrib.get('NAME', '') for k in
                                  ('CACHE_VALUE', 'PROT_VALUE', 'MMCM_CLKFBOUT_MULT_F',
                                   'MMCM_CLKOUT0_DIVIDE_F', 'DIVCLK_DIVIDE', 'PRIM_IN_FREQ'))],
            'clocks': [ET.tostring(p, encoding='unicode') for p in module.iter('PORT')
                       if p.attrib.get('SIGIS') == 'clk'],
            'memory_ranges': [p.attrib for p in module.iter('MEMRANGE')],
        }
    result = {'status': 'LOCAL_ASSOCIATION_PASS', 'bit_containers': rows,
              'xsa_sha256': sha256(xsa), 'xsa_bindings': checks, 'rtl_bindings': rtl,
              'hwh_connections': bindings, 'board_configuration_readback': 'unavailable; loading evidence separately required',
              'clock_note': 'HWH MMCM 12/12 and PL0→pll_clk→four operators; kernel PL0 report is not an independent physical frequency measurement.'}
    write_new_json(run.directory / 'analysis/bitstream_association.json', result)
    shutil.copyfile(HEADER, run.directory / 'raw/xcompute_hw.h')
    for row in rtl:
        shutil.copyfile(row['actual_path'], run.directory / 'raw' / Path(row['actual_path']).name)
    return result


def build(run):
    from build_vta_c3_completion_handshake import source_variant
    derived = run.directory / 'raw/derived_vta_hw'
    derived.mkdir()
    shutil.copytree(HW / 'config', derived / 'config', ignore=shutil.ignore_patterns('__pycache__'))
    (derived / 'include').symlink_to(HW / 'include', target_is_directory=True)
    (derived / 'src/axu5evb').mkdir(parents=True)
    for path in (HW / 'src').iterdir():
        if path.name != 'axu5evb':
            (derived / 'src' / path.name).symlink_to(path, target_is_directory=path.is_dir())
    header = run.directory / 'raw/certified_fresh_completion.h'
    source_header = Path(__file__).with_name('c3_fresh_completion.h')
    text = source_header.read_text()
    text = text[text.index('#ifndef'):]
    text = '// Stage01 generated-register binding is in bitstream_association.json.\n' + text
    header.write_text(text)
    shutil.copyfile(Path(__file__).with_name('c3_completion_probe.h'), run.directory / 'raw/c3_completion_probe.h')
    original = HW / 'src/axu5evb/axu5evb_driver.cc'
    text = source_variant(original.read_text(), header, True)
    text = text.replace('#include <vta/driver.h>', '#include <vta/driver.h>\n#include "' + str(run.directory / 'raw/xcompute_hw.h') + '"\nstatic_assert(XCOMPUTE_CONTROL_BUS_ADDR_DONE_O_DATA == VTA_COMPUTE_DONE_RD_OFFSET, "done register binding changed");\nstatic_assert(XCOMPUTE_CONTROL_BUS_ADDR_DONE_O_CTRL == 0x1c, "valid register binding changed");')
    text = text.replace('VTA_COMPUTE_DONE_RD_OFFSET + 4', 'XCOMPUTE_CONTROL_BUS_ADDR_DONE_O_CTRL')
    # One acquire/release barrier after the accepted MMIO event, before CPU use.
    text = text.replace('      return 0;\n', '      __sync_synchronize();\n      return 0;\n')
    destination = derived / 'src/axu5evb/axu5evb_driver.cc'
    destination.write_text(text)
    output = run.directory / 'raw/aarch64_build'
    output.mkdir()
    shutil.copyfile(ROOT / 'cmake/config.cmake', output / 'config.cmake')
    with (output / 'config.cmake').open('a') as stream:
        stream.write('\nset(USE_LLVM OFF)\nset(USE_RPC ON)\nset(USE_CPP_RPC ON)\nset(USE_VTA_FPGA ON)\nset(USE_VTA_FSIM OFF)\nset(USE_LIBBACKTRACE OFF)\nset(HIDE_PRIVATE_SYMBOLS OFF)\nset(USE_ALTERNATIVE_LINKER OFF)\n')
    shutil.copyfile(ROOT / 'build_axu_aarch64/vta_config.json', output / 'vta_config.json')
    script = run.directory / 'raw/build_current_runtime.sh'
    script.write_text('''#!/bin/bash
set -e
unset LD_LIBRARY_PATH
source /home/orange/alinx_plsdk/install/environment-setup-aarch64-xilinx-linux
export VTA_HW_PATH="$1"
export TVM_PATH="$2"
/usr/bin/cmake -S "$2" -B "$3" -G Ninja -DCMAKE_TOOLCHAIN_FILE="$OE_CMAKE_TOOLCHAIN_FILE" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
/usr/bin/cmake --build "$3" --target tvm_runtime vta tvm_rpc -j4
''')
    command = ['bash', str(script), str(derived), str(ROOT), str(output)]
    result = run.execute(command, 'current_runtime_bootstrap_build', 'one_time_preparation', 1200)
    write_new_json(run.directory / 'analysis/driver_derivation.json', {
        'original_source': str(original), 'original_sha256': sha256(original),
        'derived_source': str(destination), 'derived_sha256': sha256(destination),
        'generated_header_sha256': sha256(run.directory / 'raw/xcompute_hw.h'),
        'source_variant_helper_sha256': sha256(Path(__file__).with_name('build_vta_c3_completion_handshake.py')),
        'build_result': result, 'original_driver_or_prebuilt_files_modified': False,
        'added_sleep_for_completion': False,
    })
    if result['status'] != 'ok':
        raise RuntimeError('Current-source bootstrap failed; keep evidence')
    write_new_json(run.directory / 'analysis/runtime_artifacts.json', {
        name: {'path': str(output / name), 'sha256': sha256(output / name), 'bytes': (output / name).stat().st_size}
        for name in ('libvta.so', 'libtvm_runtime.so', 'tvm_rpc')
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('associate', 'build'))
    parser.add_argument('run_dir')
    args = parser.parse_args()
    run = RunStore(args.run_dir)
    if args.operation == 'associate':
        print(json.dumps(associate(run)['status']))
    else:
        build(run)
