"""CIDEr-D reward for SCST training (paper Eq.16-17).

Faithful line-by-line port of ruotianluo/cider's ciderD_scorer.py (the SCST
implementation used by objRel and thus by the GANN paper), including its
quirks (idf = log(N) - log(df), length counted as bigram totals).
The df cache format {'ref_len': int, 'document_frequency': dict} is
compatible with ruotianluo's prepro_ngrams.py output.
"""
import math
import pickle
from collections import defaultdict

import numpy as np
import torch


class CiderReward:
    def __init__(self, df=None, ref_len=None, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        # df cache over the TRAIN split (precomputed; see
        # scripts/prepro_ngram_cache.py)
        self.document_frequency = defaultdict(float, df or {})
        # ref_len is kept as the raw IMAGE COUNT; log applied at scoring time
        self.ref_len = float(ref_len) if ref_len else None

    # ---------------- df precompute (offline) ---------------- #
    def add_image_refs(self, refs_words):
        """refs_words: list of token-lists (one image's GT captions)."""
        if self.ref_len is None:
            self.ref_len = 0.0
        self.ref_len += 1
        for ngram in self._all_ngrams(refs_words):
            self.document_frequency[ngram] += 1

    def _all_ngrams(self, refs_words):
        s = set()
        for tokens in refs_words:
            for k in range(1, self.n + 1):
                for i in range(len(tokens) - k + 1):
                    s.add(tuple(tokens[i:i + k]))
        return s

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'ref_len': int(self.ref_len),
                         'document_frequency': dict(self.document_frequency)}, f)

    @classmethod
    def load(cls, path, n=4, sigma=6.0):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        return cls(df=d['document_frequency'], ref_len=d['ref_len'],
                   n=n, sigma=sigma)

    # ---------------- scoring (SCST time) ---------------- #
    def _counts(self, tokens):
        counts = defaultdict(int)
        for k in range(1, self.n + 1):
            for i in range(len(tokens) - k + 1):
                counts[tuple(tokens[i:i + k])] += 1
        return counts

    def _counts2vec(self, cnts):
        logN = math.log(self.ref_len)
        vec = [defaultdict(float) for _ in range(self.n)]
        norm = [0.0] * self.n
        length = 0
        for ngram, tf in cnts.items():
            df = math.log(max(1.0, self.document_frequency[ngram]))
            k = len(ngram) - 1
            vec[k][ngram] = float(tf) * (logN - df)           # tf * idf
            norm[k] += vec[k][ngram] ** 2
            if k == 1:                                        # official quirk
                length += tf
        norm = [math.sqrt(x) if x > 0 else 1.0 for x in norm]
        return vec, norm, length

    def _sim(self, vh, vr, nh, nr, lh, lr):
        delta = float(lh - lr)
        val = np.array([0.0] * self.n)
        for k in range(self.n):
            for ngram, w in vh[k].items():
                val[k] += min(w, vr[k][ngram]) * vr[k][ngram]
            val[k] /= (nh[k] * nr[k])
            val[k] *= math.exp(-(delta ** 2) / (2 * self.sigma ** 2))
        return val

    def score_one(self, hyp_words, refs_words):
        """CIDEr-D of one hypothesis vs one image's refs (x10, official)."""
        if self.ref_len is None or len(hyp_words) == 0:
            return 0.0
        vec, norm, length = self._counts2vec(self._counts(hyp_words))
        score = np.array([0.0] * self.n)
        for ref in refs_words:
            vr, nr, lr = self._counts2vec(self._counts(ref))
            score += self._sim(vec, vr, norm, nr, length, lr)
        return 10.0 * float(np.mean(score)) / len(refs_words)

    def batch_reward(self, words_sampled, words_greedy, refs_list):
        """Return (B, 2) float tensor: [sampled_reward, greedy_reward]."""
        out = torch.zeros(len(words_sampled), 2)
        for i, (hs, hg, refs) in enumerate(zip(words_sampled, words_greedy,
                                               refs_list)):
            out[i, 0] = self.score_one(hs, refs)
            out[i, 1] = self.score_one(hg, refs)
        return out
