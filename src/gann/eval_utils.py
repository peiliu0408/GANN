"""Greedy/beam decoding over a split + COCO language evaluation."""
import time

import torch

from .dataloader import CaptionDataset, EvalCollator
from .language_eval import evaluate_predictions
from torch.utils.data import DataLoader


@torch.no_grad()
def eval_split(model, opt, split, device, max_images=None, verbose=True):
    """Decode `split`, return (preds {id:[hyp]}, gts {id:[refs]}, metrics)."""
    model.eval()
    ds = CaptionDataset(opt, split)
    coll = EvalCollator(ds)
    dl = DataLoader(ds, batch_size=opt.batch_size, shuffle=False,
                    num_workers=opt.num_workers, collate_fn=coll)

    preds, gts = {}, {}
    t0 = time.time()
    n_done = 0
    for batch in dl:
        feats = batch['feats'].to(device)
        keys_valid = batch['keys_valid'].to(device)
        seqs = model.decode(feats, keys_valid, method=opt.sample_method,
                            beam_size=opt.beam_size,
                            length_penalty=getattr(opt, 'length_penalty', 0.0))
        itow = ds.ix_to_word
        for i, sid in enumerate(batch['ids']):
            toks = seqs[i].tolist() if torch.is_tensor(seqs[i]) else seqs[i]
            words = [itow.get(int(t), '<unk>') for t in toks
                     if int(t) not in (0, 1, 2)]
            preds[sid] = [' '.join(words)]
            gts[sid] = list(batch['gts'][i])
        n_done += len(batch['ids'])
        if verbose and n_done % (opt.batch_size * 50) == 0:
            print(f'  eval {split}: {n_done}/{len(ds)} '
                  f'({n_done / max(1e-6, time.time() - t0):.0f} img/s)', flush=True)
        if max_images and n_done >= max_images:
            break

    metrics = {}
    if opt.language_eval:
        metrics = evaluate_predictions(gts, preds)
        if verbose:
            pretty = '  '.join(f'{k}={v:.2f}' for k, v in sorted(metrics.items()))
            print(f'  [{split}] {pretty}', flush=True)
    return preds, gts, metrics
