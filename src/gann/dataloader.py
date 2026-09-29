"""Dataset over pre-extracted bottom-up features + Karpathy captions.

Design goals:
  - No image loading, no dynamic feature extraction: everything comes from
    the SCAN bottom-up feature banks (36 x 2048 per image, float32, mmap'd).
  - Feature/caption alignment via a precomputed index (cocoid -> bank,row).
  - Caption-object label vectors are precomputed offline (no NLTK needed at
    train time).

Vocab convention: 0=<pad>, 1=<bos>, 2=<eos>, 3=<unk>.
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

PAD, BOS, EOS, UNK = 0, 1, 2, 3


class CaptionDataset(Dataset):
    """
    Inputs (all produced by scripts/build_vocab_labels.py and
    scripts/build_feature_index.py):
      input_json    : vocab.json  {ix_to_word / word_to_ix, splits: {split: [ids]}}
      feature_banks : list of .npy paths; index gives (bank, row) per cocoid
      labels_vec    : labels_{split}.npz with key=str(cocoid) -> 0/1[K]
    """
    def __init__(self, opt, split):
        super().__init__()
        self.opt = opt
        self.split = split

        with open(opt.input_json, 'r', encoding='utf-8') as f:
            self.info = json.load(f)
        self.ix_to_word = {int(k): v for k, v in self.info['ix_to_word'].items()}
        self.word_to_ix = {w: i for i, w in self.ix_to_word.items()}
        opt.itow = self.ix_to_word
        self.vocab_size = len(self.ix_to_word)

        # karpathy raw json (captions + split assignment)
        with open(opt.karpathy_json, 'r', encoding='utf-8') as f:
            kj = json.load(f)
        # NOTE: karpathy json has BOTH imgid (sequential 0..N-1) and cocoid
        # (real COCO id). Feature files are named by COCO id -> use cocoid.
        wanted = set(self.info['splits'][split])
        self.images = [im for im in kj['images']
                       if self._id(im) in wanted or str(self._id(im)) in wanted]
        by_id = {}
        for im in self.images:
            by_id[self._id(im)] = im
            by_id[str(self._id(im))] = im
        self.by_id = by_id

        self.ids = [self._id(im) for im in self.images]

        # feature banks (mmap) / per-image files / single packed binary
        self.banks = []
        self.packed = None
        if getattr(opt, 'feature_files', None) and \
                opt.feature_mode == 'banks':
            for p in opt.feature_files.split(','):
                arr = np.load(p.strip(), mmap_mode='r')
                self.banks.append(arr)
        with open(opt.feature_index, 'r', encoding='utf-8') as f:
            self.feat_index = json.load(f)
        self.feature_mode = opt.feature_mode
        if opt.feature_mode == 'packed':
            bin_path = None
            meta = os.path.join(os.path.dirname(opt.feature_index),
                                'packed_meta.json')
            if os.path.exists(meta):
                bin_path = json.load(open(meta, encoding='utf-8'))['data']
            else:                                   # default location
                bin_path = os.path.join(os.path.dirname(
                    os.path.dirname(opt.feature_index)),
                    'features', 'cocobu_att_packed', 'data.f32')
            self.packed = np.memmap(bin_path, dtype=np.float32, mode='r')

        # caption object labels (train uses them for L_label; others optional)
        self.label_vecs = None
        p = opt.labels_vec.format(split=split)
        if os.path.exists(p):
            z = np.load(p)
            self.label_vecs = {int(k): z[k] for k in z.files}

        # max ref count per image (5 for COCO)
        self.n_ref = max(len(im['sentences']) for im in self.images)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _id(im):
        """Real COCO image id (feature files / labels are keyed by it)."""
        return im.get('cocoid', im['imgid'])

    # ------------------------------------------------------------------ #
    def load_feats(self, cocoid):
        if self.feature_mode == 'packed':
            s, e = self.feat_index[str(cocoid)]      # row offsets, see pack_features.py
            d = self.opt.feat_dim
            arr = np.asarray(self.packed[s * d:e * d], dtype=np.float32)
            return arr.reshape(e - s, d)
        path = self.feat_index[str(cocoid)]
        d = np.load(path)
        if isinstance(d, np.lib.npyio.NpzFile):      # AIStudio layout
            arr = d['feat'] if 'feat' in d.files else d[d.files[0]]
            d.close()
        else:
            arr = d                                  # plain .npy
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:                            # flattened bank layout
            arr = arr.reshape(-1, self.opt.feat_dim)
        return arr

    def encode_caption(self, tokens):
        ids = [BOS]
        for t in tokens[:self.opt.max_seq_len - 1]:
            ids.append(self.word_to_ix.get(t, UNK))
        ids.append(EOS)
        return ids

    def refs_tensor(self, image, L):
        """(n_ref, L) long tensor of reference token ids (PAD-padded)."""
        out = torch.full((self.n_ref, L), PAD, dtype=torch.long)
        for i, s in enumerate(image['sentences']):
            ids = [self.word_to_ix.get(t, UNK) for t in s['tokens']]
            ids = ids[:L]
            out[i, :len(ids)] = torch.as_tensor(ids, dtype=torch.long)
        return out

    def __getitem__(self, idx):
        im = self.images[idx]
        cocoid = self._id(im)
        feats = self.load_feats(cocoid)

        # pick one caption at random (ruotianluo convention), clamp length
        si = np.random.randint(len(im['sentences']))
        cap = self.encode_caption(im['sentences'][si]['tokens'])

        item = {
            'id': cocoid,
            'feats': torch.from_numpy(feats),
            'keys_valid': feats.shape[0],
        }
        if self.label_vecs is not None and int(cocoid) in self.label_vecs:
            item['label_vec'] = torch.from_numpy(
                self.label_vecs[int(cocoid)].astype(np.float32))
        return item


class GannCollator:
    """Batches samples into model-ready tensors."""

    def __init__(self, dataset, mode):
        self.ds = dataset
        self.mode = mode          # 'enc_pre' | 'xe' | 'scst'

    def __call__(self, raw):
        B = len(raw)
        ids = [r['id'] for r in raw]
        feats = [r['feats'] for r in raw]
        n = max(f.shape[0] for f in feats)
        F_ = torch.zeros(B, n, feats[0].shape[1])
        valid = torch.as_tensor([f.shape[0] for f in feats], dtype=torch.long)
        for i, f in enumerate(feats):
            F_[i, :f.shape[0]] = f

        out = {'ids': ids, 'feats': F_, 'keys_valid': valid}
        # Always emit labels/refs: --phase all builds loaders once with
        # mode='enc_pre', then trains xe/scst with the same loader — a
        # mode-gated collator crashes the later phases (KeyError 'labels').
        # Harmless for enc_pre-only runs (run_epoch ignores them).
        caps = [self.ds.encode_caption(
            self.ds.by_id[i]['sentences'][np.random.randint(
                len(self.ds.by_id[i]['sentences']))]['tokens']) for i in ids]
        T = max(len(c) for c in caps)
        labels = torch.full((B, T), PAD, dtype=torch.long)
        for i, c in enumerate(caps):
            labels[i, :len(c)] = torch.as_tensor(c)
        out['labels'] = labels
        if self.mode in ('scst', 'all'):
            # 'all' included: phase=all builds the loader in enc_pre mode, then
            # trains scst with the same loader -> refs must always be present.
            L = min(16, max(max(len(s['tokens']) for s in self.ds.by_id[i]['sentences'])
                            for i in ids))
            out['refs'] = torch.stack([self.ds.refs_tensor(self.ds.by_id[i], L)
                                       for i in ids])
        if 'label_vec' in raw[0]:
            out['label_vec'] = torch.stack([r['label_vec'] for r in raw])
        else:
            K = self.ds.opt.n_labels
            out['label_vec'] = torch.zeros(B, K)
        return out


class EvalCollator:
    """Batches for decoding: features + all GT refs for language eval."""

    def __init__(self, dataset):
        self.ds = dataset

    def __call__(self, raw):
        B = len(raw)
        ids = [r['id'] for r in raw]
        feats = [r['feats'] for r in raw]
        n = max(f.shape[0] for f in feats)
        F_ = torch.zeros(B, n, feats[0].shape[1])
        valid = torch.as_tensor([f.shape[0] for f in feats], dtype=torch.long)
        for i, f in enumerate(feats):
            F_[i, :f.shape[0]] = f
        gts = [[s.get('raw') or ' '.join(s['tokens'])
                for s in self.ds.by_id[i]['sentences']] for i in ids]
        return {'ids': ids, 'feats': F_, 'keys_valid': valid, 'gts': gts}
