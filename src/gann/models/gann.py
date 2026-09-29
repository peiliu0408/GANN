"""GANN captioner: encoder + global caption feature + label head + GANN decoder.

Paper mapping (see docs/01_模型与Loss详解.md):
  - input  : R = [r0, r1..rn], r0 = zero vector (BERT [CLS] / <sos> role)
  - Eq.1   : Ybar = Encoder(R)                -> y0 (pos 0) + Y (rest)
  - Eq.2   : lhat = sigmoid(FFN(y0))          (caption object multi-label head)
  - Eq.3   : L_label  (BCEWithLogits on label head logits)
  - Eq.9-14: GlobalAttention in every decoder layer/head
  - Eq.15  : L_XE     (teacher forcing word CE)
  - Eq.16-17: L_R SCST (CIDEr reward, greedy self-critical baseline)
  - Eq.18-19: joint   = lambda * L_label + (1 - lambda) * task loss
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import (DecoderLayer, EncoderLayer, MultiHeadAttention,
                          PositionalEncoding, Stack)
from .global_attention import GlobalAttention

PAD, BOS, EOS, UNK = 0, 1, 2, 3


class LabelHead(nn.Module):
    """FFN + sigmoid on top of y0, predicting caption objects (Eq.2)."""

    def __init__(self, d_model=512, d_ff=2048, n_labels=220, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, n_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y0):
        return self.fc2(self.dropout(F.relu(self.fc1(y0))))   # logits


class GANNCaptioner(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        d = opt.input_encoding_size          # 512
        self.vocab_size = opt.vocab_size
        self.n_labels = opt.n_labels

        # ---------------- encoder ----------------
        # input projection Linear+ReLU+Dropout, faithful to objRel's att_embed
        self.feat_embed = nn.Sequential(
            nn.Linear(opt.feat_dim, d), nn.ReLU(), nn.Dropout(opt.dropout))
        n_enc = getattr(opt, 'num_encoder_layers', None) or opt.num_layers
        n_dec = getattr(opt, 'num_decoder_layers', None) or opt.num_layers
        self.n_enc_layers, self.n_dec_layers = n_enc, n_dec
        self.encoder = Stack(
            lambda: EncoderLayer(d, opt.n_head, d // opt.n_head, opt.d_ff,
                                 opt.dropout),
            n_enc, d)

        # ---------------- label head (Eq.2) ----------------
        self.label_head = LabelHead(d, opt.d_ff, opt.n_labels, opt.dropout)

        # ---------------- decoder ----------------
        self.embed = nn.Embedding(opt.vocab_size, d, padding_idx=PAD)
        self.pos_enc = PositionalEncoding(d, max_len=opt.max_seq_len + 2,
                                          dropout=opt.dropout, emb_sqrt=True)
        self.decoder = Stack(
            lambda: DecoderLayer(d, opt.n_head, d // opt.n_head, opt.d_ff,
                                 opt.dropout,
                                 cross_attention=self._make_cross(opt)),
            n_dec, d)
        self.logit = nn.Linear(d, opt.vocab_size)
        if getattr(opt, 'weight_tie', 0):
            # tie output projection to word embedding (Inan et al. 2016 /
            # Press & Wolf 2017); common in captioning/NMT, often +0.3~1 CIDEr
            self.logit.weight = self.embed.weight

        self.cider_scorer = None             # injected for SCST (scripts/train.py)
        self._reset_parameters()

    def _make_cross(self, opt):
        """Each decoder layer gets its own cross-attention module (no sharing)."""
        d = opt.input_encoding_size
        if opt.no_gann:
            return MultiHeadAttention(d, opt.n_head, d // opt.n_head, opt.dropout)
        return GlobalAttention(d, opt.n_head, d // opt.n_head, opt.dropout,
                               direct=opt.gann_direct,
                               share_heads=opt.gann_share_heads)

    # ------------------------------------------------------------------ #
    def _reset_parameters(self):
        scheme = getattr(self.opt, 'init_scheme', 'xavier')
        if scheme == 'bert':
            # Devlin et al. 2018: N(0, 0.02) everywhere, LN weight=1 bias=0
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        elif scheme == 'trunc_normal':
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.Embedding)):
                    nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            # xavier: objRel / original Transformer code base
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        with torch.no_grad():
            self.embed.weight[PAD].zero_()

    # ------------------------------------------------------------------ #
    def encode(self, feats, keys_valid=None):
        """Eq.1 with the r0 zero-vector slot injected at position 0.

        feats: (B, n, 2048); returns y0 (B, d), Y (B, n, d), mem_pad (B, n)
        """
        B, n, _ = feats.shape
        slot = feats.new_zeros(B, 1, feats.size(2))
        R = torch.cat([slot, feats], dim=1)                  # [r0, r1..rn]
        x = self.feat_embed(R)
        x = self.encoder(x, self_mask=None)
        y0, Y = x[:, 0], x[:, 1:]
        mem_pad = None
        if keys_valid is not None:                           # True = padded
            idx = torch.arange(n, device=feats.device)
            mem_pad = ~(idx.unsqueeze(0) < keys_valid.unsqueeze(1))
        return y0, Y, mem_pad

    # ------------------------------------------------------------------ #
    def _decode_logits(self, seq_in, Y, mem_pad, y0):
        """Run decoder over seq_in (B, T). Returns logits (B, T, V)."""
        x = self.pos_enc(self.embed(seq_in))
        x = self.decoder(x, mem=Y, mem_pad_mask=mem_pad, y0=y0)
        return self.logit(x)

    # ------------------------------------------------------------------ #
    def _xe_loss(self, labels, Y, mem_pad, y0):
        """Eq.15 with teacher forcing. labels: (B, T) [bos, w..., eos, pad...]."""
        seq_in = labels[:, :-1]
        target = labels[:, 1:]
        logits = self._decode_logits(seq_in, Y, mem_pad, y0)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size),
                               target.reshape(-1), ignore_index=PAD,
                               label_smoothing=self.opt.label_smoothing)
        return loss

    # ------------------------------------------------------------------ #
    def _greedy(self, Y, mem_pad, y0, max_len):
        B = Y.size(0)
        seq = Y.new_full((B, 1), BOS, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=Y.device)
        for _ in range(max_len):
            logits = self._decode_logits(seq, Y, mem_pad, y0)[:, -1]
            logits[:, PAD] = float('-inf')          # PAD is never generated
            nxt = logits.argmax(-1)
            nxt = nxt.masked_fill(done, PAD)
            seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
            done = done | (nxt == EOS)
            if done.all():
                break
        return seq[:, 1:]

    def _sample(self, Y, mem_pad, y0, max_len):
        """Multinomial sampling (train mode, T=1), as in objRel/SCST.
        Returns (seq, logprobs, mask): mask[b,t]=1 iff step t was actively
        sampled (up to & including the EOS step), matching the reference
        RewardCriterion mask convention."""
        B = Y.size(0)
        seq = Y.new_full((B, 1), BOS, dtype=torch.long)
        done = torch.zeros(B, dtype=torch.bool, device=Y.device)
        logprobs, active = [], []
        for _ in range(max_len):
            logits = self._decode_logits(seq, Y, mem_pad, y0)[:, -1]
            logits[:, PAD] = float('-inf')          # PAD is never sampled
            logp = F.log_softmax(logits, dim=-1)
            nxt_raw = torch.multinomial(logp.exp(), 1).squeeze(1)
            nxt = nxt_raw.masked_fill(done, PAD)
            logp_t = logp.gather(1, nxt_raw.unsqueeze(1)).squeeze(1)
            active.append(~done)                    # EOS step is included
            logprobs.append(logp_t)
            seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
            done = done | (nxt == EOS)
            if done.all():
                break
        mask = torch.stack(active, dim=1).float()
        return seq[:, 1:], torch.stack(logprobs, dim=1), mask

    # ------------------------------------------------------------------ #
    def _seq_to_words(self, seq):
        """Id sequence(s) -> word lists, with a single trailing '<eos>' so
        that SCST reward strings match the df-cache convention (the original
        pipeline includes the terminator in n-grams)."""
        itow = self.opt.itow
        out = []
        for s in seq:
            words = []
            for t in s:
                t = int(t)
                if t == EOS:
                    words.append('<eos>')
                    break
                if t in (PAD, BOS):
                    continue
                words.append(itow.get(t, '<unk>'))
            if not words or words[-1] != '<eos>':
                words.append('<eos>')            # hit max_len without EOS
            out.append(words)
        return out

    def _scst_loss(self, labels, Y, mem_pad, y0, refs):
        """SCST (Eq.16-17), line-faithful to the reference implementation
        (objRel misc/rewards.py + misc/utils.py RewardCriterion, which the
        GANN code base inherits):

          1. greedy baseline decoded under model.eval() (dropout OFF), then
             model.train() restored;
          2. reward = CIDEr-D(sample) - CIDEr-D(greedy)  [x10 scale, no len
             normalization], repeated over every time step;
          3. loss  = - sum(logp * reward * mask) / sum(mask)
             (GLOBAL token-count normalization of the sequence-sum logp --
             NOT a per-token average; the mask convention matches theirs:
             steps up to & including the terminator are trained);
          4. train_sample_n > 1 repeats the sampled branch (each with the
             same greedy baseline) and pools steps, equivalent to their
             sample_n concatenation.

        refs: (B, n_ref, L) long tensor of GT reference word ids (padded 0),
        so that DataParallel can scatter it along the batch dimension.
        """
        max_len = self.opt.max_seq_len

        # (1) greedy baseline in eval mode, like objRel rewards.py L40-46
        was_training = self.training
        if was_training:
            self.eval()
        try:
            with torch.no_grad():
                seq_g = self._greedy(Y, mem_pad, y0, max_len)
        finally:
            if was_training:
                self.train()

        words_g = self._seq_to_words(seq_g)
        refs_words = self._seq_to_words(
            refs.reshape(refs.size(0) * refs.size(1), -1))
        refs_words = [refs_words[i * refs.size(1):(i + 1) * refs.size(1)]
                      for i in range(refs.size(0))]

        # (2-4) sampled branch(es), pooled with global mask normalization
        n = max(1, int(getattr(self.opt, 'train_sample_n', 1)))
        lp_all, rw_all, mask_all = [], [], []
        adv_sum, r_sum = 0.0, 0.0
        for _ in range(n):
            seq_s, logp_s, mask = self._sample(Y, mem_pad, y0, max_len)
            words_s = self._seq_to_words(seq_s)
            rewards = self.cider_scorer.batch_reward(words_s, words_g,
                                                     refs_words)
            r = rewards.to(logp_s.device)
            adv = (r[:, 0] - r[:, 1]).detach()              # Eq.17
            lp_all.append(logp_s * mask)
            rw_all.append(adv.unsqueeze(1).expand_as(logp_s) * mask)
            mask_all.append(mask)
            adv_sum += adv.mean().item()
            r_sum += r[:, 0].mean().item()

        # sample_n>1: each branch may end at a different step (early break when
        # all done) -> pad every branch to the longest T before cat.
        if len(lp_all) > 1:
            T = max(x.size(1) for x in lp_all)
            pad_to = lambda x, v: F.pad(x, (0, T - x.size(1)), value=v)
            lp_all = [pad_to(x, 0.0) for x in lp_all]       # logprob pad = 0
            rw_all = [pad_to(x, 0.0) for x in rw_all]       # advantage pad = 0
            mask_all = [pad_to(x, 0.0) for x in mask_all]   # mask pad = 0

        lp_cat = torch.cat(lp_all)
        rw_cat = torch.cat(rw_all)
        mask_cat = torch.cat(mask_all)
        loss = -(lp_cat * rw_cat).sum() / mask_cat.sum().clamp_min(1.0)
        return loss, adv_sum / n, r_sum / n

    # ------------------------------------------------------------------ #
    def forward(self, feats, labels, label_vec, refs=None, keys_valid=None,
                mode='xe'):
        """Unified entry. Returns dict of scalar losses (DataParallel-friendly).

        mode='enc_pre' : encoder-only pretraining, loss = L_label (paper IV-B)
        mode='xe'      : joint XE  = lambda*L_label + (1-lambda)*L_XE   (Eq.18)
        mode='scst'    : joint RL  = lambda*L_label + (1-lambda)*L_R    (Eq.19)
        """
        y0, Y, mem_pad = self.encode(feats, keys_valid)
        label_logits = self.label_head(y0)                             # Eq.2
        lab_loss = F.binary_cross_entropy_with_logits(label_logits, label_vec)
        out = {'label_loss': lab_loss}

        lam = self.opt.label_loss_weight                               # lambda
        if mode == 'enc_pre':
            out['loss'] = lab_loss
            return out

        if mode == 'xe':
            xe = self._xe_loss(labels, Y, mem_pad, y0)                 # Eq.15
            out['xe_loss'] = xe
            out['loss'] = lam * lab_loss + (1.0 - lam) * xe            # Eq.18
        elif mode == 'scst':
            rl, adv, r_s = self._scst_loss(labels, Y, mem_pad, y0, refs)
            out['rl_loss'] = rl
            out['reward'] = torch.as_tensor(r_s, device=feats.device)
            out['advantage'] = torch.as_tensor(adv, device=feats.device)
            out['loss'] = lam * lab_loss + (1.0 - lam) * rl            # Eq.19
        else:
            raise ValueError(mode)
        return out

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def decode(self, feats, keys_valid=None, method='greedy', beam_size=5,
               max_len=None, length_penalty=0.0):
        """Greedy or beam decoding for evaluation."""
        y0, Y, mem_pad = self.encode(feats, keys_valid)
        max_len = max_len or self.opt.max_seq_len
        if method == 'greedy':
            return self._greedy(Y, mem_pad, y0, max_len)
        return self._beam(Y, mem_pad, y0, max_len, beam_size, length_penalty)

    @torch.no_grad()
    def _beam(self, Y, mem_pad, y0, max_len, k, alpha=0.0):
        """Beam search with optional length normalization (Wu et al. 2016):
        finished hypotheses are scored by sum_logp / len^alpha; alpha=0
        reproduces plain beam search. Eval-only, per-sample loop."""
        B = Y.size(0)
        results = []
        for b in range(B):
            Yb = Y[b:b + 1]
            pad_b = mem_pad[b:b + 1] if mem_pad is not None else None
            y0b = y0[b:b + 1]
            Yk = Yb.expand(k, -1, -1).contiguous()
            y0k = y0b.expand(k, -1).contiguous()
            padk = pad_b.expand(k, -1).contiguous() if pad_b is not None else None
            seqs = Yb.new_full((k, 1), BOS, dtype=torch.long)
            scores = torch.full((k,), float('-inf'), device=Y.device)
            scores[0] = 0.0                                   # only beam 0 active
            finished = []                                     # (norm_score, seq)
            for _ in range(max_len):
                logits = self._decode_logits(seqs, Yk, padk, y0k)[:, -1]
                logits[:, PAD] = float('-inf')      # PAD handled by done-beams
                logp = F.log_softmax(logits, dim=-1)
                v, idx = logp.topk(k, dim=-1)                 # (k, k)
                add = v
                done = (seqs[:, -1] == EOS)                   # (k,)
                if done.any():
                    # finished beams may only append PAD (id 0) with delta 0
                    add[done] = torch.where(idx[done] == PAD,
                                            add.new_zeros(add[done].shape),
                                            add.new_full(add[done].shape,
                                                         float('-inf')))
                cand_scores = (scores.unsqueeze(1) + add).view(-1)
                cand_seqs = torch.cat([seqs.repeat_interleave(k, 0),
                                       idx.reshape(-1, 1)], dim=1)
                # candidate length = tokens excluding BOS/PAD
                cand_lens = ((cand_seqs != PAD) &
                             (cand_seqs != BOS)).sum(1).clamp_min(1)
                norm_scores = cand_scores / cand_lens.float().pow(alpha)
                # lock in newly finished beams (top-k by raw sum that end EOS)
                just_done = (cand_seqs[:, -1] == EOS) & (cand_scores > float('-inf'))
                for ni in just_done.nonzero(as_tuple=True)[0].tolist():
                    finished.append((float(norm_scores[ni]), cand_seqs[ni]))
                if len(finished) >= k:
                    break
                top = cand_scores.topk(k).indices              # rank by raw sum
                scores = cand_scores[top]
                seqs = cand_seqs[top]
                if (seqs[:, -1] == EOS).all():
                    break
            if not finished:                                   # no EOS within max_len
                for i in range(k):
                    if scores[i] > float('-inf'):
                        finished.append((float(scores[i] /
                                               max(1, ((seqs[i] != PAD) &
                                                       (seqs[i] != BOS)).sum())
                                               ), seqs[i]))
            best = max(finished, key=lambda x: x[0])[1]
            toks = []
            for t in best[1:].tolist():
                if t == EOS:
                    break
                if t in (PAD, BOS):
                    continue
                toks.append(t)
            results.append(torch.as_tensor(toks, dtype=torch.long,
                                           device=Y.device))
        return results
