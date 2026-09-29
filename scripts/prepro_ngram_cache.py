"""Precompute the CIDEr document-frequency cache used by SCST rewards.

Faithful to ruotianluo's scripts/prepro_ngrams.py: reference tokens are
mapped through the VOCABULARY (OOV -> '<unk>') and each sentence gets a
terminator ('<eos>') appended BEFORE counting n-grams, so that the df cache
word surface matches the hypothesis side at SCST time (our _seq_to_words
appends '<eos>' as well).

Runs locally or on the intranet (no GPU needed). Output:
  data/processed/coco-train-idxs.pkl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))

from gann.cider_reward import CiderReward                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--karpathy_json', default='data/captions/dataset_coco.json')
    ap.add_argument('--vocab_json', default='data/processed/vocab.json',
                    help='needed for OOV-><unk> mapping (original behavior)')
    ap.add_argument('--output_pkl', default='data/processed/coco-train-idxs.pkl')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.output_pkl), exist_ok=True)

    with open(args.karpathy_json, encoding='utf-8') as f:
        kj = json.load(f)
    with open(args.vocab_json, encoding='utf-8') as f:
        vj = json.load(f)
    wtoi = vj['word_to_ix']
    itow = {int(k): w for k, w in vj['ix_to_word'].items()}

    def to_vocab(tokens):
        """token list -> vocab-id strings + terminator, as prepro_ngrams.py"""
        out = [str(wtoi.get(t, wtoi['<unk>'])) for t in tokens]
        out.append(str(2))                     # our EOS id, as terminator
        return out

    scorer = CiderReward()
    n = 0
    for im in kj['images']:
        if im['split'] not in ('train', 'restval'):
            continue
        scorer.add_image_refs([to_vocab(s['tokens'])
                               for s in im['sentences']])
        n += 1
    scorer.save(args.output_pkl)
    print(f'df cache saved: {args.output_pkl} ({n} train images, '
          f'ref_len={scorer.ref_len})')


if __name__ == '__main__':
    main()
