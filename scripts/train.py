"""Three-phase GANN training (paper IV-B):

  phase enc_pre : encoder + label head only, loss = L_label   (10 epochs, paper)
  phase xe      : joint XE  = lambda*L_label + (1-lambda)*L_XE  (early stop)
  phase scst    : joint RL  = lambda*L_label + (1-lambda)*L_R   (early stop)

  --phase all    : enc_pre -> xe -> scst (hand-off via best checkpoints)
  --phase xe     : single phase (weights via --start_from_model)

Example (paper-faithful defaults):
  python scripts/train.py --id gann_base --feature_files <train.npy>,<dev.npy>,<test.npy>
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'src'))

import torch                                                        # noqa: E402
from torch.utils.data import DataLoader                             # noqa: E402

from gann.models.gann import GANNCaptioner                          # noqa: E402
from gann.dataloader import CaptionDataset, GannCollator            # noqa: E402
from gann.cider_reward import CiderReward                           # noqa: E402
from gann.eval_utils import eval_split                              # noqa: E402
from gann.utils import (CsvLogger, EMA, load_model_weights, noam_lr,     # noqa: E402
                        save_checkpoint, set_seed, step_lr)
from gann.opts import parse_opt, save_opt                           # noqa: E402


def build_loaders(opt):
    train_ds = CaptionDataset(opt, 'train')
    if opt.max_train_images > 0:
        train_ds.ids = train_ds.ids[:opt.max_train_images]
        train_ds.images = train_ds.images[:opt.max_train_images]
    val_ds = CaptionDataset(opt, 'val')
    opt.vocab_size = train_ds.vocab_size
    opt.n_labels = len(json.load(open(opt.labels_json, encoding='utf-8'))['labels'])
    coll_train = GannCollator(train_ds, opt.phase_mode)
    coll_val = GannCollator(val_ds, opt.phase_mode)
    dl_train = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True,
                          num_workers=opt.num_workers, collate_fn=coll_train,
                          drop_last=True, pin_memory=True,
                          persistent_workers=opt.num_workers > 0)
    dl_val = DataLoader(val_ds, batch_size=opt.batch_size, shuffle=False,
                        num_workers=opt.num_workers, collate_fn=coll_val,
                        pin_memory=True,
                        persistent_workers=opt.num_workers > 0)
    return train_ds, val_ds, dl_train, dl_val


class Collapse(Exception):
    """Val CIDEr collapsed far below best on consecutive checks -> abort."""


def run_epoch(model, loader, opt, optimizer, device, epoch, logger, global_step,
              scheduler=None, ema=None, step_logger=None, val_step_fn=None,
              val_every_steps=0):
    model.train()
    sums, n = {}, 0
    t0 = time.time()
    last_check = (global_step // val_every_steps) if val_every_steps > 0 else -1
    for it, batch in enumerate(loader):
        feats = batch['feats'].to(device, non_blocking=True)
        keys_valid = batch['keys_valid'].to(device)
        label_vec = batch['label_vec'].to(device, non_blocking=True)
        kwargs = {}
        if opt.phase_mode == 'xe':
            kwargs['labels'] = batch['labels'].to(device)
        elif opt.phase_mode == 'scst':
            kwargs['labels'] = batch['labels'].to(device)
            kwargs['refs'] = batch['refs'].to(device)

        # bf16 autocast: ~2x on A800 (no GradScaler needed for bf16)
        use_amp = bool(getattr(opt, 'amp', 0)) and device.type == 'cuda'
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            losses = model(feats, kwargs.get('labels'), label_vec,
                           refs=kwargs.get('refs'), keys_valid=keys_valid,
                           mode=opt.phase_mode)
        loss = losses['loss'].mean() if losses['loss'].dim() else losses['loss']

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        if scheduler is not None:
            scheduler.step()
        optimizer.step()
        if ema is not None:
            ema.update(model)

        for k, v in losses.items():
            x = v.mean().item() if v.dim() else v.item()
            sums[k] = sums.get(k, 0.0) + x
        n += 1
        if it % 100 == 0:
            cur = {k: round(v / n, 4) for k, v in sums.items()}
            lr = optimizer.param_groups[0]['lr']
            print(f'  ep{epoch} it{it}/{len(loader)} loss={cur.get("loss"):.4f} '
                  f'lr={lr:.3e} {n * opt.batch_size / (time.time() - t0):.0f}it/s',
                  flush=True)
        # step-level loss log (kept forever, used for ablation analysis)
        if step_logger is not None and global_step % 50 == 0:
            lr = optimizer.param_groups[0]['lr']
            step_logger.log({'phase': opt.phase_mode, 'step': global_step,
                             'epoch': epoch,
                             **{k: round(v / n, 4) for k, v in sums.items()},
                             'lr': lr})
        # mid-epoch val for fast early-stop decisions
        if val_every_steps > 0 and val_step_fn is not None:
            k = (global_step + it + 1) // val_every_steps
            if k > last_check:
                last_check = k
                try:
                    val_step_fn(global_step + it + 1)
                except Collapse as e:
                    print(f'  [COLLAPSE] {e} -> aborting run early', flush=True)
                    return {k: v / max(1, n) for k, v in sums.items()}, \
                        global_step + it + 1, True
        global_step += 1
    return {k: v / max(1, n) for k, v in sums.items()}, global_step, False


def main():
    opt = parse_opt()
    set_seed(opt.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(opt.checkpoint_path, exist_ok=True)
    save_opt(opt, os.path.join(opt.checkpoint_path, 'opt.json'))
    logger = CsvLogger(os.path.join(opt.checkpoint_path, 'history.csv'))

    # banks mode needs the bank list; files mode just uses feat_index.json
    if opt.feature_mode == 'banks' and (not opt.feature_files or not all(
            os.path.exists(p) for p in opt.feature_files.split(','))):
        fb = os.path.join(os.path.dirname(opt.feature_index), 'feature_banks.json')
        with open(fb, encoding='utf-8') as f:
            opt.feature_files = ','.join(json.load(f)['banks'])
        print(f'feature_files (auto): {opt.feature_files}')

    # temporary phase for loader building (collators differ per mode)
    phases = (['enc_pre', 'xe', 'scst'] if opt.phase == 'all' else [opt.phase])
    opt.phase_mode = phases[0]
    train_ds, val_ds, dl_train, dl_val = build_loaders(opt)
    print(f'train {len(train_ds)} images / val {len(val_ds)} images; '
          f'vocab {opt.vocab_size}; labels {opt.n_labels}')

    model = GANNCaptioner(opt).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'model: arch={opt.arch} enc={getattr(opt, "num_encoder_layers", opt.num_layers)}L '
          f'dec={getattr(opt, "num_decoder_layers", opt.num_layers)}L '
          f'd={opt.input_encoding_size} h={opt.n_head} ff={opt.d_ff} '
          f'init={getattr(opt, "init_scheme", "xavier")} params={n_params/1e6:.1f}M')
    start_infos = {}
    if opt.start_from and os.path.exists(
            os.path.join(opt.start_from, 'infos.json')):
        with open(os.path.join(opt.start_from, 'infos.json'), encoding='utf-8') as f:
            start_infos = json.load(f)
        load_model_weights(model, os.path.join(opt.start_from, 'model.pth'))
        print(f'resumed from {opt.start_from}')
    elif opt.start_from_model:
        load_model_weights(model, opt.start_from_model)
        print(f'weights loaded from {opt.start_from_model}')

    if opt.n_gpu > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f'DataParallel over {torch.cuda.device_count()} GPUs')

    # SCST reward scorer
    if 'scst' in phases:
        model_tpl = model.module if hasattr(model, 'module') else model
        model_tpl.cider_scorer = CiderReward.load(opt.cider_cache)
        print(f'cider reward cache loaded (ref_len={model_tpl.cider_scorer.ref_len})')

    best_score = start_infos.get('best_score')
    for phase in phases:
        opt.phase_mode = phase
        if phase == 'enc_pre':
            print(f'===== PHASE enc_pre: label-loss-only encoder pretraining '
                  f'({opt.enc_pre_epochs} epochs, paper IV-B) =====')
            # optimize encoder + label head only
            root = model.module if hasattr(model, 'module') else model
            enc_ids = {id(p) for p in list(root.feat_embed.parameters()) +
                       list(root.encoder.parameters()) +
                       list(root.label_head.parameters())}
            params = [p for p in model.parameters() if id(p) in enc_ids]
        else:
            params = [p for p in model.parameters() if p.requires_grad]

        lr0 = opt.learning_rate if phase != 'scst' else opt.scst_learning_rate
        optimizer = torch.optim.Adam(params, lr=lr0,
                                     betas=(opt.optim_alpha, opt.optim_beta),
                                     eps=opt.optim_epsilon,
                                     weight_decay=opt.weight_decay)
        # CRITICAL (red-team finding): noam normalizes away the base lr, so a
        # noam SCST phase would run at ~4.4e-4 (9x the intended 5e-5) and
        # destabilize RL. SCST always runs on a fixed lr schedule.
        if phase == 'scst':
            scheduler = None
            for g in optimizer.param_groups:
                g['lr'] = lr0
            print(f'[scst] fixed lr={lr0} (noam ignored in this phase)')
        elif opt.noamopt:
            # multiplier = rate(step)/lr0 so Adam's base lr=lr0 yields noam rate
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda s: noam_lr(s + 1, opt.input_encoding_size,
                                  opt.noamopt_warmup) / lr0)
        elif phase == 'enc_pre':   # short linear warmup, then constant
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda s: min(1.0, (s + 1) / max(1, opt.noamopt_warmup // 8)))
        else:
            scheduler = None

        max_ep = (opt.enc_pre_epochs if phase == 'enc_pre' else
                  opt.max_epochs if phase == 'xe' else opt.scst_max_epochs)
        since_best = 0
        global_step = 0
        ema = EMA(model, opt.ema_decay) if getattr(opt, 'ema_decay', 0) > 0 \
            else None
        step_logger = CsvLogger(os.path.join(opt.checkpoint_path, 'steps.csv'))
        mid_state = {'last_below': False}

        def val_step_fn(step):
            """Mid-epoch quick val -> steps.csv; collapse circuit-breaker."""
            if ema is not None:
                ema.store(model)
                ema.copy_to(model)
            _, _, metrics = eval_split(model, opt, 'val', device,
                                       max_images=opt.val_steps_subset,
                                       verbose=False)
            score = metrics.get('CIDEr', 0.0)
            if ema is not None:
                ema.restore(model)
            step_logger.log({'phase': opt.phase_mode, 'tag': 'val_mid',
                             'step': step, 'val_score': round(score, 2)})
            below = best_score is not None and score < best_score - opt.collapse_drop
            if below and mid_state['last_below']:
                raise Collapse(f'step {step}: val {score:.1f} << best '
                               f'{best_score:.1f} twice in a row')
            mid_state['last_below'] = below
            return score

        for epoch in range(max_ep):
            if scheduler is None:
                if phase == 'scst' and not opt.scst_decay:
                    cur = lr0                      # constant unless --scst_decay 1
                else:
                    cur = step_lr(epoch, lr0, opt.learning_rate_decay_start,
                                  opt.learning_rate_decay_every,
                                  opt.learning_rate_decay_rate)
                for g in optimizer.param_groups:
                    g['lr'] = cur

            stats, global_step, collapsed = run_epoch(
                model, dl_train, opt, optimizer,
                device, epoch, logger, global_step,
                scheduler, ema, step_logger,
                val_step_fn, opt.val_every_steps)
            row = {'phase': phase, 'epoch': epoch,
                   'global_step': global_step, **{f'train_{k}': round(v, 4)
                                                  for k, v in stats.items()}}

            # ---------- validation (EMA weights if enabled) ----------
            if phase in ('xe', 'scst'):
                if ema is not None:
                    ema.store(model)
                    ema.copy_to(model)
                preds, gts, metrics = eval_split(model, opt, 'val', device,
                                                 max_images=opt.val_images_use)
                score = metrics.get('CIDEr', 0.0)
                row.update({f'val_{k}': round(v, 2) for k, v in metrics.items()})
                row['val_score'] = score
                improved = best_score is None or score > best_score
                if improved:
                    best_score = score
                    since_best = 0
                else:
                    since_best += 1
                save_checkpoint(opt.checkpoint_path, model,
                                {'phase': phase, 'epoch': epoch,
                                 'best_score': best_score,
                                 'vocab_json': opt.input_json,
                                 'karpathy_json': opt.karpathy_json,
                                 'labels_json': opt.labels_json,
                                 'ema': bool(ema)},
                                is_best=improved)
                if ema is not None:
                    ema.restore(model)
                print(f'  [{phase} ep{epoch}] val CIDEr={score:.2f} '
                      f'best={best_score:.2f} since_best={since_best}', flush=True)
                mid_state['last_below'] = (
                    best_score is not None and score < best_score - opt.collapse_drop)
                if since_best >= opt.early_stop_patience:
                    print(f'  early stop at epoch {epoch} (patience '
                          f'{opt.early_stop_patience})')
                    logger.log(row)
                    break
            else:
                save_checkpoint(opt.checkpoint_path, model,
                                {'phase': phase, 'epoch': epoch,
                                 'vocab_json': opt.input_json}, is_best=False)
            logger.log(row)
            if collapsed:
                print(f'  [{phase}] run aborted early (collapse detector); '
                      f'best weights are preserved in model-best.pth')
                break
        print(f'===== phase {phase} done =====')

        # hand-off: SCST starts from the best XE weights (paper: "Initializing
        # from the cross-entropy trained model")
        if phase == 'xe' and 'scst' in phases:
            best_p = os.path.join(opt.checkpoint_path, 'model-best.pth')
            if os.path.exists(best_p):
                load_model_weights(model, best_p)
                print(f'SCST init from {best_p}')

        if phase == 'scst':
            print(f'FINAL BEST val CIDEr: {best_score}')


if __name__ == '__main__':
    main()
