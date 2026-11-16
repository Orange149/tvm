#!/usr/bin/env python3
"""Check range isolation and demonstrate the explicit live-input transport rule."""
import argparse
import json
from pathlib import Path
import numpy as np
import tvm
from tvm.contrib import graph_executor
from thesis_canonical_graph_split import UnitGraph
from thesis_supplement_run import sha256, write_new_json


def check(canonical, qualification, output):
    output.mkdir(parents=True, exist_ok=False)
    graph = UnitGraph(tvm.ir.load_json(canonical.read_text()))
    ranges = [(i, i+1) for i in range(21)] + [(0,3),(3,18),(18,21)]
    for begin,end in ranges:
        graph.stage(begin,end)
    original_frontier = graph.frontier
    faults = []
    for boundary,end in [(3,18),(7,8),(17,18)]:
        complete = original_frontier(boundary)
        def incomplete(value, target=boundary, edges=complete):
            return edges[1:] if value == target else original_frontier(value)
        graph.frontier = incomplete
        try:
            graph.stage(boundary,end)
        except ValueError as error:
            if 'recompute an operation outside' not in str(error):
                raise
            faults.append({'boundary':boundary,'removed_node_id':graph.ids[complete[0]],
                           'status':'REJECTED_BEFORE_COMPILATION','reason':str(error)})
        else:
            raise ValueError('An incomplete cut silently recomputed earlier units')
        finally:
            graph.frontier = original_frontier
    transport = []
    frozen = json.loads((qualification/'artifact_hashes.json').read_text())
    for begin in [3,6]:
        _,inputs,outputs = graph.stage(begin,begin+1)
        library = qualification/f'compiled/units_{begin:02}_{begin:02}/module.so'
        if frozen[str(library.relative_to(qualification))]['sha256']!=sha256(library):
            raise ValueError('Qualified CPU operator library changed')
        compiled = library.parent
        for case in ['zeros','uniform','normal']:
            with np.load(qualification/case/'canonical_boundaries.npz',allow_pickle=False) as saved:
                values=[saved[f'node_{graph.ids[n]}'] for n in inputs]
                references=[saved[f'node_{graph.ids[n]}'] for n in outputs]
            exe=graph_executor.create((compiled/'graph.json').read_text(),tvm.runtime.load_module(str(library)),tvm.cpu())
            exe.load_params((compiled/'params.bin').read_bytes())
            external=[tvm.nd.array(value,tvm.cpu()) for value in values]
            for i,array in enumerate(external):
                poison=np.bitwise_not(values[i].view('uint8')).view(values[i].dtype).reshape(values[i].shape)
                exe.get_input(f'edge{begin}_{i}').copyfrom(poison)
                exe.set_input_zero_copy(f'edge{begin}_{i}',array)
            exe.run()
            folder=output/f'unit{begin:02}_{case}'
            folder.mkdir()
            for i,node in enumerate(outputs):
                if node not in inputs:
                    if not np.array_equal(exe.get_output(i).numpy(),references[i]):
                        raise ValueError('Computed output affected by original pool poison')
                    continue
                routed=external[inputs.index(node)].numpy()
                getter=exe.get_output(i).numpy()
                np.save(folder/f'output{i}_live_input.npy',routed,allow_pickle=False)
                np.save(folder/f'output{i}_getter.npy',getter,allow_pickle=False)
                np.save(folder/f'output{i}_reference.npy',references[i],allow_pickle=False)
                if not np.array_equal(routed,references[i]):
                    raise ValueError('Explicit live-input route failed')
                transport.append({'unit':begin,'input':case,'output_index':i,'node_id':graph.ids[node],
                                  'explicit_live_input_correct':True,
                                  'naive_getter_mismatching_elements':int(np.count_nonzero(getter!=references[i])),
                                  'naive_getter_bytes_exact':getter.tobytes()==references[i].tobytes()})
    manifest=[]
    for begin,end in ranges:
        _,inputs,outputs=graph.stage(begin,end)
        manifest.append({'begin':begin,'end_exclusive':end,
                         'input_node_ids':[graph.ids[n] for n in inputs],
                         'output_sources':[{'node_id':graph.ids[n],
                             'kind':'live_input' if n in inputs else 'executor_output',
                             'live_input_index':inputs.index(n) if n in inputs else None,
                             'graph_output_index':i if n not in inputs else None}
                             for i,n in enumerate(outputs)]})
    write_new_json(output/'native_transport_interface.json',{
        'canonical_qir_sha256':sha256(canonical),'stage_interfaces':manifest,
        'carry_rule':'Keep the actual incoming buffer and ownership until the final consumer finishes; output API does not rebind a Var-backed passthrough data_entry after SetInputZeroCopy.',
        'native_runner_implemented_or_qualified':False,
    })
    result={'status':'HOST_INTERFACE_CHECK_PASS','isolated_stage_ranges':len(ranges),
            'incomplete_cut_negative_controls':faults,'passthrough_negative_controls':transport,
            'naive_getter_wrong_cases':sum(not x['naive_getter_bytes_exact'] for x in transport),
            'scope':'Two CPU unit libraries, three frozen synthetic inputs. No native/VTA lifetime/accuracy or performance conclusion.',
            'stage02_qualified':False}
    write_new_json(output/'summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--canonical-qir',type=Path,required=True)
    p.add_argument('--host-qualification',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(check(args.canonical_qir,args.host_qualification,args.output),indent=2))
