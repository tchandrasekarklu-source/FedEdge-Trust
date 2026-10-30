"""Trainable compact Echo ViT + ECG CNN-BiLSTM + temporal graph EHR fusion.

Engineering reference, not a reproduction of the authors' unreleased checkpoint.
"""
import math
import torch
from torch import nn
from fededge_protocol.pairwise_fusion import PairwiseQualityTimeAttention


class EchoVideoTransformer(nn.Module):
    def __init__(self, dim=32, patch=8, frames=4, height=32, width=32):
        super().__init__()
        if height % patch or width % patch:
            raise ValueError('Echo image width/height must be divisible by patch')
        self.patch = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        n_tokens = frames * (height // patch) * (width // patch)
        self.position = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=dim*2, dropout=0.,
            batch_first=True, norm_first=False), num_layers=1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if x.ndim != 5 or x.shape[2] != 1:
            raise ValueError('echo must be [B,T,1,H,W]')
        b, t, c, h, w = x.shape
        v = self.patch(x.reshape(b*t, c, h, w))
        v = v.flatten(2).transpose(1, 2).reshape(b, -1, v.shape[1])
        if v.shape[1] != self.position.shape[1]:
            raise ValueError('Echo shape must match configured frames/image resolution')
        return self.norm(self.encoder(v + self.position).mean(1))


class ECGCNNBiLSTM(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(12, dim, 7, stride=2, padding=3),
                                 nn.GELU(), nn.Conv1d(dim, dim, 5, stride=2, padding=2), nn.GELU())
        self.rnn = nn.LSTM(dim, dim // 2, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != 12:
            raise ValueError('ECG must be [B,12,L]')
        tokens = self.cnn(x).transpose(1, 2)
        states, _ = self.rnn(tokens)
        return self.norm(states.mean(1))


class TemporalGraphEHR(nn.Module):
    def __init__(self, features=16, dim=32):
        super().__init__()
        self.project = nn.Linear(features, dim)
        self.graph = nn.Linear(dim, dim)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, features, adjacency):
        if features.ndim != 3 or adjacency.shape != (*features.shape[:2], features.shape[1]):
            raise ValueError('EHR inputs must be [B,V,F] and adjacency [B,V,V]')
        if (adjacency < 0).any():
            raise ValueError('No negative EHR graph edge weights')
        z = torch.relu(self.project(features))
        row_sum = adjacency.sum(-1, keepdim=True).clamp_min(1.)
        graph_message = torch.bmm(adjacency / row_sum, z)
        out, _ = self.rnn(z + torch.relu(self.graph(graph_message)))
        return self.norm(out.mean(1))


class ITGFusion(nn.Module):
    """Strict verified-clock gating. absence/quality handled by fusion, no target-leakage data access."""
    def __init__(self, dim=32, frames=4, height=32, width=32,
                 patch=8, ehr_features=16, variant='full'):
        super().__init__()
        if variant not in {'full','late','echo_only','ecg_only','ehr_only',
                           'no_quality_time','no_mask'}:
            raise ValueError('Unsupported ablation/baseline variant')
        self.variant = variant
        self.echo = EchoVideoTransformer(dim,patch,frames,height,width)
        self.ecg = ECGCNNBiLSTM(dim)
        self.ehr = TemporalGraphEHR(ehr_features,dim)
        self.fusion = PairwiseQualityTimeAttention(dim=dim, modalities=3)
        self.late_head = nn.Sequential(nn.LayerNorm(dim*3), nn.Linear(dim*3,1))

    def forward(self, echo, ecg, ehr, adj, presence, quality, lag_hours, clock_verified):
        if self.variant in ('echo_only','ecg_only','ehr_only'):
            # Real unimodal baselines compute only their branch; other slots stay null.
            if self.variant == 'echo_only':
                v=self.echo(echo); z=torch.stack([v,torch.zeros_like(v),torch.zeros_like(v)],dim=1)
            elif self.variant == 'ecg_only':
                v=self.ecg(ecg); z=torch.stack([torch.zeros_like(v),v,torch.zeros_like(v)],dim=1)
            else:
                v=self.ehr(ehr,adj); z=torch.stack([torch.zeros_like(v),torch.zeros_like(v),v],dim=1)
        else:
            z = torch.stack([self.echo(echo),self.ecg(ecg),self.ehr(ehr,adj)],dim=1)
        valid = presence.bool() & clock_verified.bool() & (quality > 0)
        if self.variant == 'echo_only': valid[:,1:] = False
        elif self.variant == 'ecg_only': valid[:,0] = False; valid[:,2] = False
        elif self.variant == 'ehr_only': valid[:,:2] = False
        if self.variant == 'no_mask':
            # Counterfactual ablation: only use strictly permitted/verified records.
            # No-mask tests only difference in missing-modality conditioning;
            # a missing record must NEVER become an invented patient observation.
            gate = valid.to(z.dtype)
            logits = self.late_head((z*gate.unsqueeze(-1)).reshape(len(z),-1)).squeeze(-1)
            return logits
        if self.variant == 'late':
            logits = self.late_head((z*valid.to(z.dtype).unsqueeze(-1)).reshape(len(z),-1)).squeeze(-1)
            return logits
        q = quality if self.variant != 'no_quality_time' else torch.ones_like(quality)
        lag = lag_hours if self.variant != 'no_quality_time' else torch.zeros_like(lag_hours)
        logits, _, _ = self.fusion(z, valid, q, lag, clock_verified=clock_verified,
                                   require_clock_verified=True)
        return logits


DEFAULT_SHAPE = dict(dim=32,frames=4,height=32,width=32,patch=8,ehr_features=16)
