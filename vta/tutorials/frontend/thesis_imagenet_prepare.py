#!/usr/bin/env python3
"""Prepare official devkit labels; select the fixed subset only from full validation data."""
import argparse
import ast
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import numpy as np
from scipy.io import loadmat
from thesis_supplement_run import sha256, write_new_json


def labels(devkit, class_names, output):
    with tarfile.open(devkit, 'r:gz') as archive:
        meta = archive.extractfile('ILSVRC2012_devkit_t12/data/meta.mat').read()
        truth = archive.extractfile('ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt').read()
    synsets = loadmat(io.BytesIO(meta), squeeze_me=True, struct_as_record=False)['synsets']
    leaf = [item for item in synsets if int(item.num_children) == 0]
    by_id = {int(item.ILSVRC2012_ID): item for item in leaf}
    if set(by_id) != set(range(1, 1001)):
        raise ValueError('Official devkit leaf IDs differ from 1..1000')
    ordered = sorted(leaf, key=lambda item: str(item.WNID))
    names = ast.literal_eval(Path(class_names).read_text())
    if not isinstance(names, dict) or set(names) != set(range(1000)):
        raise ValueError('Expected the cached 1000-label class-name dictionary')
    if any(names[i] != str(item.words) for i, item in enumerate(ordered)):
        mismatch = [(i, names[i], str(item.words)) for i, item in enumerate(ordered) if names[i] != str(item.words)]
        raise ValueError('Class ordering is not confirmed by all cached names: ' + str(mismatch[:3]))
    index = {int(item.ILSVRC2012_ID): i for i, item in enumerate(ordered)}
    ids = [int(value) for value in truth.decode('ascii').splitlines()]
    counts = Counter(ids)
    if len(ids) != 50000 or set(counts) != set(range(1, 1001)) or set(counts.values()) != {50}:
        raise ValueError('Expected 50,000 labels and exactly 50 validation images per class')
    mapping = {
        'source': 'https://image-net.org/challenges/LSVRC/2012/2012-downloads',
        'devkit_sha256': sha256(devkit), 'cached_class_names_sha256': sha256(class_names),
        'ordering': 'lexicographically sorted WNID; all 1000 descriptions match the existing model class-name cache',
        'ordering_evidence_limit': 'Description match and canonical WNID order; weight file itself does not encode WNIDs. Preserve this assumption and cache provenance in the data contract.',
        'classes': [{'model_output_index': i, 'ilsvrc_id': int(item.ILSVRC2012_ID),
                     'wnid': str(item.WNID), 'words': str(item.words)} for i, item in enumerate(ordered)],
        'validation_ilsvrc_ids': ids,
        'validation_model_output_indices': [index[item] for item in ids],
        'raw_ground_truth_sha256': hashlib.sha256(truth).hexdigest(),
        'real_validation_images_present': False,
    }
    write_new_json(output, mapping)
    return {'labels': len(ids), 'classes': len(ordered), 'all_class_descriptions_match': True,
            'image_subset_selected': False}


def subset(images, mapping_path, output):
    mapping = json.loads(mapping_path.read_text())
    paths = {}
    for path in images.rglob('ILSVRC2012_val_*.JPEG'):
        match = re.fullmatch(r'ILSVRC2012_val_(\d{8})\.JPEG', path.name)
        if match is None or not path.is_file():
            continue
        identity = int(match.group(1))
        if identity in paths:
            raise ValueError('Duplicate validation ID')
        paths[identity] = path.resolve()
    if set(paths) != set(range(1, 50001)):
        raise ValueError('Subset selection requires the complete 50,000 original validation images')
    selected, inventory = {}, []
    for identity in range(1, 50001):
        path = paths[identity]
        item = {'validation_id': identity, 'path': str(path), 'sha256': sha256(path),
                'bytes': path.stat().st_size,
                'model_output_index': mapping['validation_model_output_indices'][identity - 1]}
        inventory.append(item)
        old = selected.get(item['model_output_index'])
        if old is None or (item['sha256'], identity) < (old['sha256'], old['validation_id']):
            selected[item['model_output_index']] = item
    if len(selected) != 1000:
        raise ValueError('Incomplete class coverage')
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / 'full_validation_inventory.json', inventory)
    write_new_json(output / 'subset_1000.json', {
        'rule': 'Minimum content SHA256 per model output class, tie by original validation ID',
        'label_map_sha256': sha256(mapping_path),
        'inventory_sha256': sha256(output / 'full_validation_inventory.json'),
        'images': [selected[i] for i in range(1000)], 'performance_input_indices': list(range(64)),
    })
    return {'selected': 1000, 'all_original_images_hashed': 50000}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='operation', required=True)
    p = sub.add_parser('labels')
    p.add_argument('--devkit', type=Path, required=True)
    p.add_argument('--class-names', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('subset')
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--mapping', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.operation == 'labels':
        print(json.dumps(labels(args.devkit, args.class_names, args.output)))
    else:
        print(json.dumps(subset(args.images, args.mapping, args.output)))
