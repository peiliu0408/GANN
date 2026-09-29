"""Finalize bottom-up features for training. Accepts ANY of these sources in
data/raw/ (or Downloads, moved there by the bat):

  cocobu_att.tar             (Google Drive, gzip tar of per-image npys)
  cocobu_att_train.zip + cocobu_att_val.zip   (Baidu AIStudio mirror)
  cocobu_box.zip             (optional: box coordinates)
  an already-extracted data/features/cocobu_att/ folder

All *.npy are flattened into data/features/cocobu_att/<cocoid>.npy and a
files-mode index (cocoid -> path) is written, with a hard 123287-image
coverage check against the Karpathy split.

Run:  python scripts/finalize_features.py
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, 'data', 'raw')
FEAT_DIR = os.path.join(BASE, 'data', 'features')
PROC = os.path.join(BASE, 'data', 'processed')
ATT_OUT = os.path.join(FEAT_DIR, 'cocobu_att')
BOX_OUT = os.path.join(FEAT_DIR, 'cocobu_box')


def extract_tar(path, tmp):
    print(f'extracting tar {path} ...')
    rc = subprocess.call(['tar', '-xf', path, '-C', tmp])
    if rc != 0:
        print('bsdtar failed, falling back to python tarfile')
        with tarfile.open(path) as tf:
            tf.extractall(tmp)


def extract_zip(path, tmp):
    print(f'extracting zip {path} ...')
    with zipfile.ZipFile(path) as zf:
        zf.extractall(tmp)


def flatten_npys(tmp, out_dir):
    """Move every .npy/.npz under tmp into out_dir (flat, original names).
    Windows globs are case-insensitive: *.npy and *.NPY return the same
    files, so dedupe by lowercased path and skip vanished entries."""
    os.makedirs(out_dir, exist_ok=True)
    seen = set()
    n = 0
    all_paths = glob.glob(os.path.join(tmp, '**', '*.npz'), recursive=True) + \
        glob.glob(os.path.join(tmp, '**', '*.npy'), recursive=True)
    for p in all_paths:
        k = p.lower()
        if k in seen:
            continue
        seen.add(k)
        if not os.path.exists(p):
            continue
        dst = os.path.join(out_dir, os.path.basename(p))
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(p):
            os.remove(p)
        else:
            shutil.move(p, dst)
        n += 1
    print(f'  -> {n} feature files collected into {out_dir}')


def build_files_index(att_dir):
    files = glob.glob(os.path.join(att_dir, '*.npz')) + \
        glob.glob(os.path.join(att_dir, '*.npy'))
    assert files, f'no npz/npy files under {att_dir}'
    index = {}
    for p in files:
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            cid = int(stem)
        except ValueError:
            continue                    # skip non-id files if any
        index[str(cid)] = p
    os.makedirs(PROC, exist_ok=True)
    with open(os.path.join(PROC, 'feat_index.json'), 'w') as f:
        json.dump(index, f)
    with open(os.path.join(PROC, 'feature_files.json'), 'w') as f:
        json.dump({'mode': 'files', 'att_dir': att_dir,
                   'count': len(index)}, f, indent=1)
    print(f'files-mode index: {len(index)} images -> {att_dir}')
    # coverage check vs karpathy
    with open(os.path.join(BASE, 'data', 'captions', 'dataset_coco.json'),
              encoding='utf-8') as f:
        kj = json.load(f)
    missing = {}
    for im in kj['images']:
        cid = str(im.get('cocoid', im['imgid']))   # cocoid = real COCO id
        if cid not in index:
            missing.setdefault(im['split'], []).append(cid)
    total_missing = sum(len(v) for v in missing.values())
    print(f'karpathy coverage: {123287 - total_missing}/123287 '
          f'(missing {total_missing})')
    for k, v in missing.items():
        print(f'  split {k}: {len(v)} missing, e.g. {v[:5]}')
    if total_missing:
        print('[!] 覆盖不全：可能还有 cocobu_att_*.zip 没有下载/放置，'
              '补齐后重跑本脚本即可（已有文件不会重复处理）。')
        return False
    return True


def main():
    os.makedirs(RAW, exist_ok=True)
    tmp = os.path.join(FEAT_DIR, '_finalize_tmp')
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)

    sources = []
    tar = os.path.join(RAW, 'cocobu_att.tar')
    if os.path.exists(tar):
        sources.append(('tar', tar))
    for z in sorted(glob.glob(os.path.join(RAW, 'cocobu_att*.zip'))):
        sources.append(('zip', z))

    if sources:
        for kind, p in sources:
            (extract_tar if kind == 'tar' else extract_zip)(p, tmp)
        flatten_npys(tmp, ATT_OUT)
        shutil.rmtree(tmp, ignore_errors=True)
        # only delete sources that were actually extracted above; a file that
        # lands in data/raw mid-run is picked up by the next run instead
        for _, p in sources:
            if os.path.exists(p):
                os.remove(p)
                print(f'removed extracted source {p}')
    elif os.path.isdir(ATT_OUT):
        print('using existing extracted folder', ATT_OUT)
    else:
        print(f'[!] 在 {RAW} 下没有找到 cocobu_att.tar 或 cocobu_att_*.zip')
        print('    请先把特征包放进去（或从"下载"文件夹运行 完成数据准备.bat）。')
        return 1

    # optional box coordinates (skip if already extracted)
    boxzip = os.path.join(RAW, 'cocobu_box.zip')
    already = os.path.isdir(BOX_OUT) and \
        len(glob.glob(os.path.join(BOX_OUT, '*'))) >= 123287
    if os.path.exists(boxzip) and not already:
        boxtmp = os.path.join(FEAT_DIR, '_box_tmp')
        shutil.rmtree(boxtmp, ignore_errors=True)
        os.makedirs(boxtmp, exist_ok=True)
        extract_zip(boxzip, boxtmp)
        flatten_npys(boxtmp, BOX_OUT)
        shutil.rmtree(boxtmp, ignore_errors=True)
        os.remove(boxzip)
        print('box coordinates extracted to', BOX_OUT)
    elif already:
        print('box coordinates already present:', BOX_OUT)

    ok = build_files_index(ATT_OUT)
    if ok:
        # cleanup obsolete partial parts if any
        parts = os.path.join(RAW, 'cocobu_parts')
        if os.path.isdir(parts):
            shutil.rmtree(parts, ignore_errors=True)
            print('cleaned obsolete cocobu_parts/')
    return 0 if ok else 2


if __name__ == '__main__':
    sys.exit(main())
