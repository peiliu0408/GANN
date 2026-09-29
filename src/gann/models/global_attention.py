"""Global-Attention module (the core of GANN, paper Sec. III-C, Eq.9-14).

Replaces the decoder's cross-attention. In every head of every decoder layer:

  local  (Eq.11):  W^l = Q K^T / sqrt(d_k)          Q,K from decoder hidden state
  global (Eq.9-10):Qg = y0 W_Q^g, Kg = Y W_K^g      y0 = global caption feature
                   W^g = softmax(Qg Kg^T / sqrt(d_k))   -> (B, h, n) per-region scores
  fused  (Eq.12):  W_ij = W^g_j exp(W^l_ij) / sum_j' W^g_j' exp(W^l_ij')
  output (Eq.13-14):Vhat = W V ;  head = Vhat

Equivalent numerically-stable form used by default:
  W = softmax_j( W^l_ij + log W^g_j )
because W_ij = exp(W^l_ij + log W^g_j) / Z. `--gann_direct` switches to the
literal Eq.12 computation (both are unit-tested to agree).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAttention(nn.Module):
    """Multi-head Global Attention.

    forward(x, mem, mask=None, y0=None):
      x  : (B, Nq, d_model) decoder hidden states (queries of local attention)
      mem: (B, Nk, d_model) encoded region features Y (keys/values)
      y0 : (B, d_model)     global caption feature (global query)
      mask: (B, Nq, Nk) bool True=masked  (region padding)
    """

    def __init__(self, d_model=512, n_head=8, d_k=64, dropout=0.1,
                 direct=False, share_heads=False):
        super().__init__()
        self.h = n_head
        self.d_k = d_k
        self.direct = direct
        self.share_heads = share_heads

        # local projections (Q from x; K,V from encoded regions)  -- Eq.4 style
        self.w_q = nn.Linear(d_model, n_head * d_k)
        self.w_k = nn.Linear(d_model, n_head * d_k)
        self.w_v = nn.Linear(d_model, n_head * d_k)
        self.w_o = nn.Linear(n_head * d_k, d_model)

        # global projections for y0 (query) and regions (key) -- Eq.9
        n_proj = 1 if share_heads else n_head
        self.w_qg = nn.Linear(d_model, n_proj * d_k)
        self.w_kg = nn.Linear(d_model, n_proj * d_k)

        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------ #
    def _split(self, t, B, h=None):
        h = self.h if h is None else h
        return t.view(B, -1, h, self.d_k).transpose(1, 2)   # (B, h, N, d_k)

    def _global_weights(self, y0, mem, keys_valid):
        """Eq.9-10 -> (B, h, Nk) global weights (softmaxed over regions)."""
        B, Nk, _ = mem.shape
        if self.share_heads:
            qg = self.w_qg(y0).unsqueeze(1)                       # (B,1,d_k)
            kg = self.w_kg(mem).transpose(1, 2).unsqueeze(1)      # (B,1,Nk,d_k)
        else:
            qg = self._split(self.w_qg(y0), B)                    # (B,h,1,d_k)
            kg = self._split(self.w_kg(mem), B)                   # (B,h,Nk,d_k)
        logits_g = (qg @ kg.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B,h,1,Nk)
        logits_g = logits_g.squeeze(-2)                               # (B,h,Nk)
        if keys_valid is not None:
            idx = torch.arange(Nk, device=mem.device)
            valid = idx.unsqueeze(0) < keys_valid.unsqueeze(1)        # (B,Nk)
            logits_g = logits_g.masked_fill(~valid.unsqueeze(1), float('-inf'))
        return F.softmax(logits_g, dim=-1)                            # Eq.10

    # ------------------------------------------------------------------ #
    def forward(self, x, mem, mask=None, y0=None):
        B = x.size(0)
        q = self._split(self.w_q(x), B)                    # (B,h,Nq,d_k)
        k = self._split(self.w_k(mem), B)                  # (B,h,Nk,d_k)
        v = self._split(self.w_v(mem), B)                  # (B,h,Nk,d_k)

        logits_l = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)   # Eq.11
        if mask is not None:
            logits_l = logits_l.masked_fill(mask.unsqueeze(1), float('-inf'))

        keys_valid = None
        if mask is not None:
            keys_valid = (~mask[:, 0]).sum(-1)   # (B,) valid regions per sample

        wg = self._global_weights(y0, mem, keys_valid)             # (B,h,Nk)

        if self.direct:
            # literal Eq.12 (safe: wg==0 rows turn the whole row into 0/0 ->
            # guarded by clamping exp(-inf)=0 and fixing denominator)
            exp_l = torch.exp(logits_l)                     # zeros where masked
            num = exp_l * wg.unsqueeze(-2)                  # broadcast over Nq
            den = num.sum(-1, keepdim=True)
            w = num / den.clamp_min(1e-8)
        else:
            # log-space equivalent of Eq.12
            bias = torch.log(wg.clamp_min(1e-20)).unsqueeze(-2)   # (B,h,1,Nk)
            w = F.softmax(logits_l + bias, dim=-1)

        out = self.dropout(w) @ v
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.w_o(out)
