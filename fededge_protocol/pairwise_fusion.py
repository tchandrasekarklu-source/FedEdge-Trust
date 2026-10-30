"""Research-stage pairwise cross-modal attention for linked patient embeddings.

This is an UNTRAINED reference operator, not clinical inference. Quality and
clock status must be supplied from an independent pre-index quality review.
"""
import math
import torch
from torch import nn


class PairwiseQualityTimeAttention(nn.Module):
    """Attention across present modalities, with missing-query/key invariance.

    Inputs: embeddings [B,M,D], presence [B,M], quality [B,M] in [0,1],
    lag_hours [B,M] >=0, and clock_verified [B,M] flags.
    To use as the primary clinical experiment set require_clock_verified=True;
    there is NO way to infer a trustworthy ECG machine timestamp from embeddings.
    """
    def __init__(self, dim=128, modalities=3, time_decay=0.1, pair_decay=0.05):
        super().__init__()
        self.dim = dim
        self.modalities = modalities
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.risk = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.time_decay = nn.Parameter(torch.tensor(float(time_decay)))
        self.pair_decay = nn.Parameter(torch.tensor(float(pair_decay)))

    def forward(self, embeddings, presence, quality, lag_hours,
                clock_verified=None, require_clock_verified=False):
        z = embeddings
        if z.ndim != 3 or z.shape[1:] != (self.modalities, self.dim):
            raise ValueError('expected embeddings [B,M,D]')
        for name, tensor in [('presence', presence), ('quality', quality), ('lag_hours', lag_hours)]:
            if tensor.shape != z.shape[:2]:
                raise ValueError(f'{name} must be [B,M]')
        if not torch.isfinite(z).all() or not torch.isfinite(quality).all() or not torch.isfinite(lag_hours).all():
            raise ValueError('non-finite embeddings/quality/time input')
        if (quality < 0).any() or (quality > 1).any() or (lag_hours < 0).any():
            raise ValueError('quality outside [0,1] or future/invalid offsets')
        eligible = presence.bool() & (quality > 0)
        if require_clock_verified:
            if clock_verified is None or clock_verified.shape != eligible.shape:
                raise ValueError('Independent clock-verification mask required for strict use')
            eligible &= clock_verified.bool()
        any_valid = eligible.any(dim=-1)
        # Missing queries are masked AFTER attention; missing keys are never used.
        qq = self.query(z)
        kk = self.key(z)
        vv = self.value(z)
        logits = torch.matmul(qq, kk.transpose(-1,-2)) / math.sqrt(self.dim)
        key_quality = quality.clamp_min(1e-8).log().unsqueeze(-2)
        time_age = lag_hours.unsqueeze(-2)/24.0
        separation = (lag_hours.unsqueeze(-1) - lag_hours.unsqueeze(-2)).abs()/24.0
        logits = logits + key_quality - torch.nn.functional.softplus(self.time_decay)*time_age - torch.nn.functional.softplus(self.pair_decay)*separation
        pair_ok = eligible.unsqueeze(2) & eligible.unsqueeze(1)
        # Ensure empty query rows have a finite softmax, then zero them exactly.
        safe_logits = logits.masked_fill(~pair_ok, -1e9)
        attention = torch.softmax(safe_logits, dim=-1)*pair_ok
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        contexts = attention @ vv
        query_mass = quality*eligible
        result = (contexts*query_mass.unsqueeze(-1)).sum(dim=1) / query_mass.sum(dim=1,keepdim=True).clamp_min(1e-12)
        scores = self.risk(result).squeeze(-1).masked_fill(~any_valid, 0.0)
        return scores, attention, any_valid
