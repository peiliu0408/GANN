"""Align SCAN bottom-up feature banks with the Karpathy split.

Reads the SCAN ids files (train/dev/test(_all)_ids.txt, one cocoid per line,
row i of the matching ims.npy = i-th id) and the Karpathy json, then writes:

  data/processed/feat_index.json : {str(cocoid): [bank_id, row]}
  and prints coverage statistics + feature shape.

Also writes feature_banks.json listing the .npy paths in bank order.
"""
import argparse
import json
import os

import numpy as np


def npy_shape(path):
    with open(path, 'rb') as f:
        version = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format._read_array_header(f, version)
    return shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--karpathy_json', default='data/captions/dataset_coco.json')
    ap.add_argument('--scan_dir', required=True,
                    help='dir containing coco_precomp/{train,dev,test,testall}_{ids.txt,ims.npy}')
    ap.add_argument('--out_dir', default='data/processed')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pre = os.path.join(args.scan_dir, 'coco_precomp')
    banks = [
        ('train', os.path.join(pre, 'train_ims.npy'), os.path.join(pre, 'train_ids.txt')),
        ('dev', os.path.join(pre, 'dev_ims.npy'), os.path.join(pre, 'dev_ids.txt')),
        ('test', os.path.join(pre, 'test_ims.npy'), os.path.join(pre, 'test_ids.txt')),
    ]
    bank_paths = []
    index = {}
    for name, npy, ids_f in banks:
        shape = npy_shape(npy)
        with open(ids_f, encoding='utf-8') as f:
            ids = [int(x) for x in f.read().split()]
        print(f'bank {name}: {npy} shape={shape} ids={len(ids)}')
        assert shape[0] == len(ids), f'{name}: rows {shape[0]} != ids {len(ids)}'
        dup = len(ids) - len(set(ids))
        assert dup == 0, f'{name}: {dup} duplicate ids'
        b = len(bank_paths)
        bank_paths.append(os.path.abspath(npy))
        for row, cid in enumerate(ids):
            index[str(cid)] = [b, row]

    with open(args.karpathy_json, encoding='utf-8') as f:
        kj = json.load(f)

    miss = {'train': [], 'restval': [], 'val': [], 'test': []}
    for im in kj['images']:
        if str(im['imgid']) not in index:
            miss[im['split']].append(im['imgid'])
    total = sum(len(v) for v in miss.values())
    print(f'karpathy images missing from banks: {total}')
    for k, v in miss.items():
        print(f'  {k}: {len(v)} missing', v[:5])

    with open(os.path.join(args.out_dir, 'feat_index.json'), 'w') as f:
        json.dump(index, f)
    with open(os.path.join(args.out_dir, 'feature_banks.json'), 'w') as f:
        json.dump({'banks': bank_paths,
                   'note': 'row i of bank b = cocoid whose id sits at line i of '
                           'the matching SCAN *_ids.txt'}, f, indent=1)
    print(f'feat_index.json: {len(index)} entries; banks: {bank_paths}')


if __name__ == '__main__':
    main()
