"""Vanilla Transformer building blocks (paper Sec. III-B, Eq.4-8).

Faithful to Vaswani et al. 2017 / the paper's configuration:
- 8 heads, d_k = 64, d_model = 512, point-wise FFN with 2048 hidden units
- post-LayerNorm residual structure, dropout 0.1
Implemented from scratch (no HF dependency); used as reference for the
era-appropriate style found in google-research/bert and
huggingface/pytorch-pretrained-BERT (see ref_repos/).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Standard scaled dot-product multi-head attention (Eq.4-7).

    Query comes from `x`, Key/Value come from `a` (a = x for self-attention,
    a = encoded regions for decoder cross-attention).
    """

    def __init__(self, d_model=512, n_head=8, d_k=64, dropout=0.1):
        super().__init__()
        self.h = n_head
        self.d_k = d_k
        self.w_q = nn.Linear(d_model, n_head * d_k)
        self.w_k = nn.Linear(d_model, n_head * d_k)
        self.w_v = nn.Linear(d_model, n_head * d_k)
        self.w_o = nn.Linear(n_head * d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, t, B):
        # (B, N, h*d_k) -> (B, h, N, d_k)
        return t.view(B, -1, self.h, self.d_k).transpose(1, 2)

    def forward(self, x, a, mask=None, y0=None):
        """x: (B, Nq, d_model); a: (B, Nk, d_model); mask: (B, Nq, Nk) bool,
        True = masked out. `y0` is accepted (and ignored) so that vanilla
        attention and GlobalAttention share the same call signature."""
        B = x.size(0)
        q = self._split(self.w_q(x), B)
        k = self._split(self.w_k(a), B)
        v = self._split(self.w_v(a), B)
        logits = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)  # Eq.5
        if mask is not None:
            if mask.dim() == 2:               # (B, Nk) key padding
                mask = mask.unsqueeze(1)
            logits = logits.masked_fill(mask.unsqueeze(1), float('-inf'))
        w = F.softmax(logits, dim=-1)
        out = self.dropout(w) @ v
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.w_o(out)  # Eq.7


class PositionwiseFF(nn.Module):
    """Point-wise feed-forward, Eq.8: max(0, xW1+b1)W2 + b2."""

    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class EncoderLayer(nn.Module):
    """Post-LN encoder layer: self-attn -> add&norm -> FFN -> add&norm."""

    def __init__(self, d_model=512, n_head=8, d_k=64, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, d_k, dropout)
        self.ff = PositionwiseFF(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, self_mask=None):
        y = self.ln1(x + self.dropout(self.self_attn(x, x, self_mask)))
        return self.ln2(y + self.dropout(self.ff(y)))


class DecoderLayer(nn.Module):
    """Post-LN decoder layer.

    Cross-attention is pluggable: a standard MultiHeadAttention or the GANN
    GlobalAttention (paper Sec. III-C). The GANN module needs the global
    caption feature y0, so it is passed through kwargs.
    """

    def __init__(self, d_model=512, n_head=8, d_k=64, d_ff=2048, dropout=0.1,
                 cross_attention=None):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, d_k, dropout)
        self.cross_attn = cross_attention
        self.ff = PositionwiseFF(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mem, mem_pad_mask=None, dec_pad_mask=None, y0=None):
        causal = causal_mask(x.size(1), x.device)
        self_mask = causal if dec_pad_mask is None else (dec_pad_mask | causal)
        y = self.ln1(x + self.dropout(self.self_attn(x, x, self_mask)))
        # normalize (B, Nk) key-padding mask to (B, Nq, Nk)
        if mem_pad_mask is not None and mem_pad_mask.dim() == 2:
            mem_pad_mask = mem_pad_mask.unsqueeze(1).expand(-1, x.size(1), -1)
        # GANN / vanilla cross-attention
        z = self.cross_attn(y, mem, mask=mem_pad_mask, y0=y0)
        z = self.ln2(y + self.dropout(z))
        return self.ln3(z + self.dropout(self.ff(z)))


def causal_mask(n, device):
    """(1, n, n) upper-triangular bool mask; True = blocked."""
    return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1).unsqueeze(0)


class Stack(nn.Module):
    """Layer stack with the final LayerNorm of the original Transformer code
    (objRel / harvard-anmt style: layers + self.norm at the end)."""

    def __init__(self, layer_factory, n, d_model):
        super().__init__()
        self.layers = nn.ModuleList([layer_factory() for _ in range(n)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, **kw):
        for layer in self.layers:
            x = layer(x, **kw)
        return self.norm(x)


def pad_mask(keys_valid, n_q):
    """Build (B, n_q, n_k) mask from per-sample valid key counts.

    keys_valid: (B,) long, number of valid keys per sample.
    Returns bool mask, True = masked out.
    """
    B = keys_valid.size(0)
    n_k = int(keys_valid.max().item())
    idx = torch.arange(n_k, device=keys_valid.device)
    valid = idx.unsqueeze(0) < keys_valid.unsqueeze(1)          # (B, n_k)
    return (~valid).unsqueeze(1).expand(B, n_q, n_k)            # (B, n_q, n_k)


class PositionalEncoding(nn.Module):
    """Sinusoidal position encoding, used in the decoder (word side).
    Optionally scales embeddings by sqrt(d_model) first, as in the original
    Transformer code base that objRel (and thus the paper) builds upon."""

    def __init__(self, d_model=512, max_len=64, dropout=0.1, emb_sqrt=True):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
        self.d_model = d_model
        self.emb_sqrt = emb_sqrt
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.emb_sqrt:
            x = x * math.sqrt(self.d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])
