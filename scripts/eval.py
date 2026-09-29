"""Evaluate a trained checkpoint on Karpathy splits.

  python scripts/eval.py --checkpoint log_gann_base --eval_splits test \
      --sample_method greedy [--beam_size 5]

Loads opt from <checkpoint>/infos.json / opt.json so the model matches
training; command line can override eval-related knobs.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))

import torch                                                        # noqa: E402

from gann.models.gann import GANNCaptioner                          # noqa: E402
from gann.eval_utils import eval_split                              # noqa: E402
from gann.language_eval import save_predictions                     # noqa: E402
from gann.utils import set_seed                                     # noqa: E402


class Opt:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--weights', default='model-best.pth')
    ap.add_argument('--eval_splits', default='test')
    ap.add_argument('--sample_method', default='greedy')
    ap.add_argument('--beam_size', type=int, default=5)
    ap.add_argument('--length_penalty', type=float, default=-1.0,
                    help='-1 = use saved opt value; else override')
    ap.add_argument('--batch_size', type=int, default=0, help='0 = keep opt')
    ap.add_argument('--num_images', type=int, default=-1)
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    set_seed(42)
    ck = args.checkpoint
    with open(os.path.join(ck, 'opt.json'), encoding='utf-8') as f:
        opt_d = json.load(f)
    opt = Opt()
    for k, v in opt_d.items():
        setattr(opt, k, v)
    # force proper types after json round-trip
    for k in ('batch_size', 'num_workers', 'beam_size', 'language_eval'):
        setattr(opt, k, int(getattr(opt, k)))
    opt.sample_method = args.sample_method
    opt.beam_size = args.beam_size
    if args.length_penalty >= 0:
        opt.length_penalty = args.length_penalty
    if args.batch_size:
        opt.batch_size = args.batch_size
    opt.eval_splits = args.eval_splits
    opt.feature_files = opt_d.get('feature_files', '')
    # banks mode: recover bank list if saved paths died; files mode needs only
    # feat_index.json
    if opt.feature_mode == 'banks':
        if opt.feature_files and not all(
                os.path.exists(p) for p in opt.feature_files.split(',')):
            opt.feature_files = ''
        if not opt.feature_files:
            fb = os.path.join(os.path.dirname(opt.feature_index),
                              'feature_banks.json')
            with open(fb, encoding='utf-8') as f:
                opt.feature_files = ','.join(json.load(f)['banks'])

    from gann.dataloader import CaptionDataset
    probe = CaptionDataset(opt, 'val')
    opt.vocab_size = probe.vocab_size
    opt.itow = probe.ix_to_word
    with open(opt.labels_json, encoding='utf-8') as f:
        opt.n_labels = len(json.load(f)['labels'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GANNCaptioner(opt).to(device)
    sd = torch.load(os.path.join(ck, args.weights), map_location='cpu')
    model.load_state_dict(sd)

    all_metrics = {}
    for split in args.eval_splits.split(','):
        preds, gts, metrics = eval_split(model, opt, split.strip(), device,
                                         max_images=None if args.num_images < 0
                                         else args.num_images)
        all_metrics[split] = metrics
        out = args.out or os.path.join(ck, f'preds_{split}'
                                           f'_{args.sample_method}.json')
        save_predictions(out, gts, preds,
                         info={'checkpoint': ck, 'split': split,
                               'method': args.sample_method,
                               'metrics': metrics})
        print(f'[{split}] saved predictions -> {out}')
    print(json.dumps(all_metrics, indent=2))


if __name__ == '__main__':
    main()
