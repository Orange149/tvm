"""Derive a wake granule from the matched boot Image, never from DNN FPS."""
from pathlib import Path
import json
import re
import struct

from run_three_thread_worker_gap import ROOT,POOL
from finite_service.common import freeze
from cpu_vta_partitioning.common import read,sha256


def audit():
    p=POOL/'kernel_parameter_sources';identity=read(p/'IMAGE_IDENTITY.json');mapping=read(p/'ADDRESS_MAP.json')
    image=(p/'Image').read_bytes();base=int(mapping['base'],16)
    if sha256(p/'Image')!=identity['Image_sha256'] or not identity['matched_running_image']:
        raise ValueError('boot Image identity differs')
    for item in mapping['functions'].values():
        if sha256(item['path'])!=item['sha256']:raise ValueError('compiled function audit changed')
    wake=(p/'wakeup_preempt_entity.isra.0.asm').read_text()
    update=(p/'update_sysctl.asm').read_text()
    page=re.search(r'\tadrp\tx0, (0x[0-9a-f]+)',wake)
    load=re.search(r'\tldr\tw0, \[x0, #(\d+)\]',wake)
    if not page or not load or '\tcmp\tx6, x0' not in wake:raise ValueError('wakeup comparison is not the audited pattern')
    address=int(page[1],16)+int(load[1]);offset=address-base
    initial=struct.unpack_from('<I',image,offset)[0]
    # The exact initializer stores min, latency, wake at +4,+8,+12 relative
    # to the scaling variable. Check both the emitted stores and constants.
    if ('\tstp\tw2, w3, [x0, #4]' not in update or '\tstr\tw1, [x0, #12]' not in update or
            '\tmul\tw1, w4, w1' not in update or '\tclz\tw1, w1' not in update):
        raise ValueError('CPU scaling initializer differs; do not guess policy')
    scale,min_gran,latency,wake_initial=struct.unpack_from('<IIII',image,offset-12)
    if (scale,min_gran,latency,wake_initial)!=(1,750000,6000000,1000000) or initial!=wake_initial:
        raise ValueError('compiled data does not match audited initializer immediates')
    platform=ROOT/'artifacts/static_revision_01/platform'/identity['boot_id']/'AUDIT.json';a=read(platform)
    if a['boot_id']!=identity['boot_id'] or a['values']['online_cores']!='0-3':raise ValueError('four-core scaling provenance differs')
    factor=1+min(4,8).bit_length()-1
    value=initial*factor
    raw=read(ROOT/'artifacts/current_model_gap_01/INPUTS.json')['quanta_ns']
    out=dict(wakeup_granularity_ns=value,normalized_compiled_ns=initial,online_cores=4,scaling='logarithmic',factor=factor,
        derivation='matched running Image initializer plus recorded online CPU count; not a kernel-memory read',
        runtime_value_directly_read=False,assumption_no_unrecorded_kernel_memory_modification=True,
        wake_load_file_offset=offset,Image_sha256=identity['Image_sha256'],build_id=identity['build_id'],
        running_boot_id=identity['boot_id'],old_wake_threshold_ns={k:int(v)/2 for k,v in raw.items()},
        mainline_source='https://raw.githubusercontent.com/torvalds/linux/v5.4/kernel/sched/fair.c',
        source_functions=['wakeup_preempt_entity','wakeup_gran','update_sysctl'],
        sleeper_credit_not_changed_or_claimed_verified=True,DNN_FPS_used=False,new_DNN=0,
        source_sha256=sha256(__file__),
        provenance={str(p/'IMAGE_IDENTITY.json'):sha256(p/'IMAGE_IDENTITY.json'),str(p/'ADDRESS_MAP.json'):sha256(p/'ADDRESS_MAP.json'),str(platform):sha256(platform)})
    freeze(p/'COMPILED_WAKEUP.json',out);return out


if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False,indent=2))
