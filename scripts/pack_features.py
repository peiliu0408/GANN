"""One-time conversion: per-image npz -> single memmap-able float32 binary
+ offset index. Removes per-access zlib decompression (A800 feed speed).

  python scripts/pack_features.py            # ~15-30 min, ~41 GB output

Output:
  data/features/cocobu_att_packed/data.f32   concatenated (n,2048) feats
  data/processed/feat_index_packed.json      cocoid -> [row_start, row_end]
Then train/eval with:  --feature_mode packed \
                       --feature_index data/processed/feat_index_packed.json
"""
import glob
import json
import os
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATT = os.path.join(BASE, 'data', 'features', 'cocobu_att')
OUT_DIR = os.path.join(BASE, 'data', 'features', 'cocobu_att_packed')
IDX_OUT = os.path.join(BASE, 'data', 'processed', 'feat_index_packed.json')
DIM = 2048


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = glob.glob(os.path.join(ATT, '*.npz')) + \
        glob.glob(os.path.join(ATT, '*.npy'))
    files.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    print(f'{len(files)} feature files')
    data_path = os.path.join(OUT_DIR, 'data.f32')
    if os.path.exists(data_path) and os.path.exists(IDX_OUT):
        print('packed files already exist; delete them to re-pack')
        return

    index = {}
    t0 = time.time()
    with open(data_path, 'wb') as out:
        start = 0
        for i, p in enumerate(files):
            cid = int(os.path.splitext(os.path.basename(p))[0])
            d = np.load(p)
            a = d['feat'] if 'feat' in getattr(d, 'files', []) else d
            a = np.asarray(a, dtype=np.float32).reshape(-1, DIM)
            a.tofile(out)
            index[str(cid)] = [start, start + a.shape[0]]
            start += a.shape[0]
            if (i + 1) % 10000 == 0:
                el = time.time() - t0
                print(f'{i+1}/{len(files)} ({(i+1)/el:.0f} file/s)', flush=True)
    with open(IDX_OUT, 'w') as f:
        json.dump(index, f)
    with open(os.path.join(OUT_DIR, '..', '..', 'processed',
                           'packed_meta.json'), 'w') as f:
        json.dump({'data': data_path, 'index': IDX_OUT, 'dim': DIM}, f, indent=1)
    gb = os.path.getsize(data_path) / 1e9
    print(f'done: {data_path} ({gb:.1f} GB), index {IDX_OUT}, '
          f'{len(index)} images in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    sys.exit(main())
