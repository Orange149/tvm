"""Conservative geometry reservation from legacy sources and later contracts.

A reserved geometry is not necessarily measured: failed and unmeasured contracts
are retained to prevent inadvertently reusing a declared holdout.
"""
import hashlib
import json
from pathlib import Path


def signature(workload):
    data,weight=workload[1][1],workload[2][1]
    return (int(data[0]*data[4]),int(data[1]*data[5]),int(data[2]),int(data[3]),
            int(weight[0]*weight[4]),int(weight[2]),int(weight[3]),
            *map(int,workload[3]),*map(int,workload[4]))


def geometries(value):
    found=set()
    if isinstance(value,dict):
        for key,item in value.items():
            if key=='geometry_signature' and isinstance(item,list) and len(item)==13 and all(type(x) is int for x in item):
                found.add(tuple(item))
            if key=='workload' and isinstance(item,list) and item and 'conv2d' in str(item[0]):
                found.add(signature(item))
            found.update(geometries(item))
    elif isinstance(value,list):
        for item in value:
            found.update(geometries(item))
    return found


def audit_sources(c3_root, legacy_sources):
    paths={Path(p).resolve() for p in legacy_sources}
    # Include nested board_collection_contract.json and contracts outside 07.
    paths.update(p.resolve() for p in Path(c3_root).rglob('*contract*.json') if p.is_file())
    rows=[]
    for path in sorted(paths):
        raw=path.read_bytes()
        if path.suffix=='.jsonl':
            values=[json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            values=[json.loads(raw)]
        found=set().union(*(geometries(value) for value in values))
        rows.append(dict(source=str(path),source_sha256=hashlib.sha256(raw).hexdigest(),
                         geometry_signatures=[list(s) for s in sorted(found)],
                         unique_geometry_count=len(found),
                         interpretation='reserved by source; not proof of measured latency'))
    return rows


def assert_unreserved(target, rows):
    collisions=[row['source'] for row in rows if list(target) in row['geometry_signatures']]
    if collisions:
        raise ValueError('geometry already reserved in historical source(s): '+', '.join(collisions[:8]))


def main():
    import argparse
    import prepare_vta_p7r115_yolo_confirmation as base
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    rows=audit_sources(base.C3,base.EXPOSURE_SOURCES.values())
    reserved=sorted({tuple(s) for row in rows for s in row['geometry_signatures']})
    document=dict(schema='c3_geometry_reservation_catalog_v1',sources=rows,
        source_count=len(rows),reserved_geometry_count=len(reserved),reserved_geometries=reserved,
        scope='legacy fixed evidence plus recursive C3 *contract*.json; no exhaustive proof about unindexed evidence',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        board_contacted=False,performance_values_used=False)
    args.output.mkdir(parents=True)
    dest=args.output/'catalog.json'
    dest.write_text(json.dumps(document,indent=2)+'\n')
    (args.output/'artifact_hashes.json').write_text(json.dumps({'artifacts':{
        'catalog.json':hashlib.sha256(dest.read_bytes()).hexdigest()}},indent=2)+'\n')
    print(json.dumps({k:document[k] for k in ('source_count','reserved_geometry_count','scope')}))


if __name__=='__main__':main()
