import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionPool(nn.Module):
    """
    Pool a set of per-fact embeddings K [B,R,d] into k_att [B,d] using r [B,d] as query.
    Returns k_att and attention weights attn [B,h,R] for debugging.
    """

    def __init__(self, d, num_heads=4, dropout=0.0):
        super().__init__()
        assert d % num_heads == 0
        self.d = d
        self.h = num_heads
        self.dk = d // num_heads
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d, d)

    def forward(self, r, K, mask=None):
        # r: [B,d], K: [B,R,d]
        B, R, d = K.shape
        q = self.Wq(r).view(B, self.h, self.dk)  # [B,h,dk]
        k = self.Wk(K).view(B, R, self.h, self.dk).transpose(1, 2)  # [B,h,R,dk]
        v = self.Wv(K).view(B, R, self.h, self.dk).transpose(1, 2)  # [B,h,R,dk]

        scores = (q.unsqueeze(2) * k).sum(-1) / math.sqrt(self.dk)  # [B,h,R]

        if mask is not None:
            # mask: [B,R] boolean True=valid
            scores = scores.masked_fill(~mask.unsqueeze(1), -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        pooled = (attn.unsqueeze(-1) * v).sum(dim=2)  # [B,h,dk]
        pooled = pooled.reshape(B, d)  # [B,d]
        pooled = self.out(pooled)
        return pooled, attn


class AttnGatedPoELatent(nn.Module):
    """
    Build q(z | C, K) by:
      - computing q_D(z|C) from r
      - pooling K using cross-attention conditioned on r -> k_att
      - computing q_K(z|K) from k_att
      - fusing via tempered PoE with trust tau in [0,1]
    """

    def __init__(self, d_rep, z_dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.attn = CrossAttentionPool(d_rep, num_heads=num_heads, dropout=dropout)

        # data-only Gaussian
        self.mu_d = nn.Linear(d_rep, z_dim)
        self.logvar_d = nn.Linear(d_rep, z_dim)

        # knowledge-only Gaussian (from pooled knowledge)
        self.mu_k = nn.Linear(d_rep, z_dim)
        self.logvar_k = nn.Linear(d_rep, z_dim)

        # trust gate tau = sigmoid(g([r,k_att,r*k_att]))
        self.gate = nn.Sequential(
            nn.Linear(3 * d_rep, d_rep),
            nn.GELU(),
            nn.Linear(d_rep, 1),
        )

        # for logging/debug
        self.last_tau = None
        self.last_attn = None

    @staticmethod
    def _fuse_poe(mu_d, logvar_d, mu_k, logvar_k, tau):
        # tau: [B,1] -> broadcast to [B,z_dim]
        tau = tau.expand_as(mu_d)

        var_d = torch.exp(logvar_d)
        var_k = torch.exp(logvar_k)
        prec_d = 1.0 / (var_d + 1e-8)
        prec_k = 1.0 / (var_k + 1e-8)

        prec = prec_d + tau * prec_k
        var = 1.0 / (prec + 1e-8)

        mu = var * (prec_d * mu_d + tau * prec_k * mu_k)
        logvar = torch.log(var + 1e-8)
        return mu, logvar

    def forward(self, r, K_emb=None, K_mask=None):
        # r: [B,d_rep]
        mu_d = self.mu_d(r)
        logvar_d = self.logvar_d(r)

        if K_emb is None or (K_emb.ndim == 3 and K_emb.shape[1] == 0):
            # no knowledge -> just data posterior
            self.last_tau = torch.zeros(r.shape[0], 1, device=r.device)
            self.last_attn = None
            return mu_d, logvar_d

        # K_emb: [B,R,d_rep]
        k_att, attn = self.attn(r, K_emb, mask=K_mask)

        mu_k = self.mu_k(k_att)
        logvar_k = self.logvar_k(k_att)

        gate_in = torch.cat([r, k_att, r * k_att], dim=-1)
        tau = torch.sigmoid(self.gate(gate_in))  # [B,1]

        mu, logvar = self._fuse_poe(mu_d, logvar_d, mu_k, logvar_k, tau)

        self.last_tau = tau.detach()
        self.last_attn = attn.detach()
        return mu, logvar
