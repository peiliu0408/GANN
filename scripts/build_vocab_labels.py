"""Build vocabulary, the ~220 caption-object labels, and per-image label vectors.

Paper (Sec. III-A):
  1. vocab from training captions (freq threshold 5, ruotianluo convention)
  2. noun words with occurrence > 100              -> 359 candidates
  3. merge singular/plural, drop gender words
     (he/him/she/her -> one class "people")        -> 220 unique labels
The original 220-word list is lost; this script deterministically rebuilds it
with WordNet. The rebuilt size is recorded and may deviate slightly from 220.

Runs LOCALLY (needs nltk+wordnet). Outputs:
  data/processed/vocab.json              ix_to_word/word_to_ix/splits
  data/processed/labels.json             label list + counts
  data/processed/labels_{split}.npz      cocoid -> 0/1[K]
"""
import argparse
import json
import os
from collections import Counter

import numpy as np

GENDER_MERGE = {'he': 'people', 'him': 'people', 'she': 'people', 'her': 'people',
                'his': 'people', 'hers': 'people'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--karpathy_json', default='data/captions/dataset_coco.json')
    ap.add_argument('--out_dir', default='data/processed')
    ap.add_argument('--word_freq_threshold', type=int, default=5)
    ap.add_argument('--label_freq_threshold', type=int, default=100)
    ap.add_argument('--label_target', type=int, default=220,
                    help='paper reports 220 caption-object labels; the '
                         'threshold is auto-tuned to hit this scale')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import nltk
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets('dog')                       # trigger
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        from nltk.corpus import wordnet as wn

    def is_noun(w):
        return len(wn.synsets(w, pos=wn.NOUN)) > 0

    def singular(w):
        m = wn.morphy(w, wn.NOUN)
        return m or w

    print('loading karpathy json ...')
    with open(args.karpathy_json, 'r', encoding='utf-8') as f:
        kj = json.load(f)

    train_imgs = [im for im in kj['images'] if im['split'] in ('train', 'restval')]

    # ---------------- 1. vocabulary ----------------
    cnt = Counter()
    for im in train_imgs:
        for s in im['sentences']:
            cnt.update(s['tokens'])
    vocab = [w for w, c in cnt.items() if c > args.word_freq_threshold]
    vocab.sort(key=lambda w: (-cnt[w], w))
    itow = {0: '<pad>', 1: '<bos>', 2: '<eos>', 3: '<unk>'}
    wtoi = {'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}
    for i, w in enumerate(vocab):
        itow[i + 4] = w
        wtoi[w] = i + 4
    print(f'vocab size: {len(itow)} (threshold {args.word_freq_threshold})')

    splits = {'train': [], 'val': [], 'test': []}
    for im in kj['images']:
        key = im['split'] if im['split'] in splits else 'train'   # restval->train
        # cocoid = real COCO id (features/labels are keyed by it);
        # imgid in karpathy json is just a sequential index
        splits[key].append(im.get('cocoid', im['imgid']))

    vocab_json = {
        'ix_to_word': {str(k): v for k, v in itow.items()},
        'word_to_ix': wtoi,
        'splits': splits,
        'word_counts': {w: cnt[w] for w in vocab},
    }
    with open(os.path.join(args.out_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump(vocab_json, f)

    # ---------------- 2. nouns with freq > 100 ----------------
    candidates = []
    for w in vocab:
        if cnt[w] > args.label_freq_threshold and is_noun(w):
            candidates.append(w)
    print(f'noun candidates with count > {args.label_freq_threshold}: '
          f'{len(candidates)} (paper: 359)')

    # ---------------- 3. merge forms -> labels ----------------
    def merge_at(threshold):
        merged = Counter()
        for w in vocab:
            if cnt[w] <= threshold or not is_noun(w):
                continue
            if w in GENDER_MERGE:
                merged[GENDER_MERGE[w]] += cnt[w]
            else:
                merged[singular(w)] += cnt[w]
        return merged

    merged_100 = merge_at(args.label_freq_threshold)
    labels_100 = sorted(merged_100, key=lambda w: (-merged_100[w], w))
    print(f'labels at threshold {args.label_freq_threshold}: {len(labels_100)}')

    # the exact paper procedure is lost; its RESULT was 220 labels. Auto-tune
    # the threshold so the rebuilt set matches the paper's scale (~220).
    target = args.label_target
    lo, hi = int(cnt[vocab[-1]]), int(cnt[vocab[0]])     # thresholds to try
    distinct = sorted({int(cnt[w]) for w in vocab if is_noun(w)}, reverse=True)
    best_t, best_d = distinct[0], abs(target)
    for t in distinct:
        d = abs(len({w if w not in GENDER_MERGE else GENDER_MERGE[w]
                     for w in vocab if cnt[w] > t and is_noun(w)}) - target)
        if d < best_d:
            best_t, best_d = t, d
    merged = merge_at(best_t)
    labels = sorted(merged, key=lambda w: (-merged[w], w))
    print(f'auto-tuned threshold {best_t} -> {len(labels)} labels '
          f'(target {target}; raw freq>100 rule gave {len(labels_100)})')
    with open(os.path.join(args.out_dir, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump({'labels': labels,
                   'counts': {w: merged[w] for w in labels},
                   'threshold_used': best_t,
                   'threshold_100_labels': labels_100,
                   'note': 'rebuilt via WordNet; original paper list lost. '
                           'Threshold auto-tuned to match the paper-reported '
                           '220-label scale.'}, f, ensure_ascii=False, indent=1)

    # token -> label index mapping (exact, singular, gender-merged)
    tok2lab = {}
    for i, lab in enumerate(labels):
        tok2lab.setdefault(lab, i)
    for w in vocab:
        if w in GENDER_MERGE:
            tok2lab[w] = tok2lab.get(GENDER_MERGE[w])
        elif is_noun(w):
            sg = singular(w)
            if sg in tok2lab and w not in tok2lab:
                tok2lab[w] = tok2lab[sg]

    # ---------------- 4. per-image label vectors ----------------
    K = len(labels)
    for split, ids in splits.items():
        idset = set(ids)
        out = {}
        for im in kj['images']:
            if im.get('cocoid', im['imgid']) not in idset:
                continue
            vec = np.zeros(K, dtype=np.int8)
            for s in im['sentences']:
                for t in s['tokens']:
                    i = tok2lab.get(t)
                    if i is not None:
                        vec[i] = 1
            out[str(im.get('cocoid', im['imgid']))] = vec
        np.savez(os.path.join(args.out_dir, f'labels_{split}.npz'), **out)
        pos = float(np.mean([v.sum() for v in out.values()]))
        print(f'labels_{split}.npz: {len(out)} images, K={K}, '
              f'avg #pos/img={pos:.1f}')
    print('done.')


if __name__ == '__main__':
    main()
