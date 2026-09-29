"""Small helpers: seeding, LR schedules, checkpoints, CSV logger."""
import csv
import json
import os
import random

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def noam_lr(step, d_model, warmup, factor=1.0):
    """objRel/ruotianluo noam schedule (misc NoamOpt.rate):
    lr = factor * d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)
    NOTE: independent of --learning_rate (that flag only matters with
    --noamopt 0); peaks at ~4.4e-4 for d=512, warmup=10000."""
    step = max(1, step)
    return factor * (d_model ** -0.5) * min(step ** -0.5,
                                            step * (warmup ** -1.5))


def step_lr(epoch, base_lr, decay_start, decay_every, decay_rate):
    if epoch < decay_start:
        return base_lr
    return base_lr * (decay_rate ** ((epoch - decay_start) // decay_every))


def save_checkpoint(dirpath, model, infos, is_best):
    os.makedirs(dirpath, exist_ok=True)
    sd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save(sd, os.path.join(dirpath, 'model.pth'))
    with open(os.path.join(dirpath, 'infos.json'), 'w', encoding='utf-8') as f:
        json.dump(infos, f, indent=2, ensure_ascii=False, default=str)
    if is_best:
        torch.save(sd, os.path.join(dirpath, 'model-best.pth'))


def load_model_weights(model, path):
    sd = torch.load(path, map_location='cpu')
    tgt = model.module if hasattr(model, 'module') else model
    tgt.load_state_dict(sd)
    return model


class EMA:
    """Exponential moving average of floating weights (Izmailov et al. 2018
    style, single decay). Use: update() after every optimizer.step(); at eval
    time store() -> copy_to() -> eval/save -> restore()."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.backup = None

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    @torch.no_grad()
    def store(self, model):
        self.backup = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def copy_to(self, model):
        sd = model.state_dict()
        for k, s in self.shadow.items():
            sd[k].copy_(s.to(sd[k].dtype))

    @torch.no_grad()
    def restore(self, model):
        if self.backup:
            model.load_state_dict(self.backup)
            self.backup = None


class CsvLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._cols = None

    def log(self, row):
        if self._cols is None or set(row) - set(self._cols or []):
            self._cols = list(row.keys()) if self._cols is None else \
                self._cols + [k for k in row if k not in self._cols]
            new = not os.path.exists(self.path)
        else:
            new = False
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=self._cols, extrasaction='ignore')
            if new:
                w.writeheader()
            w.writerow(row)
