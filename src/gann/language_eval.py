"""COCO captioning evaluation wrapper (BLEU-1/4, METEOR, ROUGE-L, CIDEr-D, SPICE).

Uses the vendored pycocoevalcap in third_party/ (ruotianluo's py3 fork of
tylin/coco-caption). Java (8+) is required for PTBTokenizer/METEOR/SPICE;
without Java we fall back to simple tokenization for BLEU/ROUGE/CIDEr and
skip METEOR/SPICE (metrics are near-identical since Karpathy GT is already
PTB-tokenized).
"""
import json
import os
import re
import shutil
import sys

THIRD_PARTY = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'third_party')


def _add_path():
    if THIRD_PARTY not in sys.path:
        sys.path.insert(0, THIRD_PARTY)


def _java_ok():
    p = os.path.join(os.environ.get('JAVA_HOME', ''), 'bin',
                     'java.exe' if os.name == 'nt' else 'java')
    if os.path.exists(p):
        return True
    return shutil.which('java') is not None


def _strip(texts):
    """{id: [strings]} -> {id: [cleaned strings]} (java-free fallback)."""
    out = {}
    for i, ss in texts.items():
        out[i] = [re.sub(r'\s+', ' ', re.sub(r"[^a-z0-9' ]", ' ', s.lower())).strip()
                  for s in ss]
    return out


def evaluate_predictions(gts, preds):
    """gts: {id: [5 refs]}; preds: {id: [1 hyp]}; ids are COCO image ids.
    Returns dict metric -> score (percentage)."""
    _add_path()
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    have_java = _java_ok()
    if have_java:
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        fmt = lambda d: {i: [{'caption': s} for s in ss] for i, ss in d.items()}
        gts_tok = PTBTokenizer().tokenize(fmt(gts))
        preds_tok = PTBTokenizer().tokenize(fmt(preds))
    else:
        print('[language_eval] Java not found -> simple tokenization; '
              'METEOR/SPICE skipped (install Java 8+ on the GPU machine to '
              'enable them)')
        gts_tok = _strip(gts)
        preds_tok = _strip(preds)

    out = {}
    bleu, _ = Bleu(4).compute_score(gts_tok, preds_tok)
    for i, s in enumerate(bleu, 1):
        out[f'BLEU-{i}'] = s * 100
    rouge, _ = Rouge().compute_score(gts_tok, preds_tok)
    out['ROUGE_L'] = rouge * 100
    cider, _ = Cider().compute_score(gts_tok, preds_tok)
    out['CIDEr'] = cider * 100

    if have_java:
        try:
            from pycocoevalcap.meteor.meteor import Meteor
            meteor, _ = Meteor().compute_score(gts_tok, preds_tok)
            out['METEOR'] = meteor * 100
        except Exception as e:                                    # noqa
            print(f'[language_eval] METEOR failed: {e}')
        try:
            from pycocoevalcap.spice.spice import Spice
            spice, _ = Spice().compute_score(gts_tok, preds_tok)
            out['SPICE'] = spice * 100
        except Exception as e:                                    # noqa
            print(f'[language_eval] SPICE failed (needs '
                  f'stanford-corenlp jars in spice/lib, see docs): {e}')
    return out


def save_predictions(path, gts, preds, info=None):
    payload = {'gts': {str(k): v for k, v in gts.items()},
               'preds': {str(k): v for k, v in preds.items()}}
    if info:
        payload['info'] = info
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f)
