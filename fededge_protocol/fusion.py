"""Reference masked, quality- and time-gated attention module.

Expects already extracted, correctly linked per-patient modality embeddings.
Not a trained end-to-end ViT+CNN-BiLSTM+TGNN clinical model.
"""
import torch
from torch import nn

class QualityTimeAttention(nn.Module):
    """Fusion with no weight assigned to unavailable modalities.

    z: (batch, modalities, dim); mask: bool (B,M);
    quality: (B,M) in (0,1]; lag_hours: (B,M) >= 0.
    Abstains by returning valid=False when every modality is unavailable.
    """
    def __init__(self, dim=128, modalities=3, decay_init=0.1):
        super().__init__()
        self.dim=dim;self.modalities=modalities
        self.q=nn.Linear(dim,dim,bias=False)
        self.k=nn.Linear(dim,dim,bias=False)
        self.v=nn.Linear(dim,dim,bias=False)
        self.head=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,1))
        self.decay=nn.Parameter(torch.tensor(float(decay_init)))

    def forward(self,z,mask,quality,lag_hours):
        if z.ndim!=3 or z.shape[1:]!=(self.modalities,self.dim):
            raise ValueError('incorrect embedding tensor shape')
        if mask.shape!=z.shape[:2] or quality.shape!=mask.shape or lag_hours.shape!=mask.shape:
            raise ValueError('mask, quality and lag shapes must agree')
        if ((quality<0)|(quality>1)).any() or (lag_hours<0).any():
            raise ValueError('invalid quality or time offsets')
        mask=mask.bool()
        # A present but unusable (quality=0) stream must be excluded as well.
        effective=mask & (quality>0)
        valid=effective.any(dim=1)
        # Fallback safe key for empty patients, which MUST be abstained from.
        stable=effective.clone();stable[~valid,0]=True
        # The query MUST ignore absent/unusable streams; otherwise arbitrary
        # filler embeddings for missing inputs can change valid predictions.
        present=effective.unsqueeze(-1)
        query=(self.q(z)*present).sum(dim=1,keepdim=True) / present.sum(dim=1,keepdim=True).clamp(min=1)
        logits=(query*self.k(z)).sum(-1)/(self.dim**.5)
        logits=logits+torch.log(quality.clamp(min=1e-6)) \
               -torch.nn.functional.softplus(self.decay)*lag_hours/24.0
        logits=logits.masked_fill(~stable,-1e9)
        weights=torch.softmax(logits,dim=1)
        weights=weights*effective
        weights=weights/(weights.sum(dim=1,keepdim=True).clamp(min=1e-12))
        pooled=(weights.unsqueeze(-1)*self.v(z)).sum(dim=1)
        risk=self.head(pooled).flatten()
        risk=risk.masked_fill(~valid,0.0) # value only placeholder; must abstain
        return risk,weights,valid
