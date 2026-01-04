import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from models.attn_poe_latent import AttnGatedPoELatent
from models.knowledge_encoders_per_fact import SetEncoderPerFact
from models.modules import Decoder, KnowledgeEncoder, XEncoder, XYEncoder
from models.utils import MultivariateNormalDiag


class INPAttnPoE(nn.Module):
    supports_gate_supervision = True

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.xy_encoder = XYEncoder(config)
        self.decoder = Decoder(config)
        self.x_encoder = XEncoder(config)
        self.train_num_z_samples = config.train_num_z_samples
        self.test_num_z_samples = config.test_num_z_samples

        self.latent_attn_poe = AttnGatedPoELatent(
            d_rep=config.hidden_dim,
            z_dim=config.hidden_dim,
            num_heads=getattr(config, "attn_heads", 4),
            dropout=getattr(config, "attn_dropout", 0.0),
        )

        if config.use_knowledge:
            if config.text_encoder == "set":
                self.per_fact_encoder = SetEncoderPerFact(config)
                self.knowledge_encoder = None
            else:
                self.per_fact_encoder = None
                self.knowledge_encoder = KnowledgeEncoder(config)
        else:
            self.per_fact_encoder = None
            self.knowledge_encoder = None

    def forward(
        self,
        x_context,
        y_context,
        x_target,
        y_target,
        knowledge=None,
        z_samples=None,
    ):
        x_context = self.x_encoder(x_context)  # [bs, num_context, x_transf_dim]
        x_target = self.x_encoder(x_target)  # [bs, num_context, x_transf_dim]

        R = self.encode_globally(x_context, y_context, x_target)
        q_z_Cc = self.infer_latent_dist_attn_poe(R, knowledge, x_context.shape[1])

        if z_samples is None:
            z_samples, _, q_zCct = self.sample_latent(
                R, x_context, x_target, y_target, knowledge, q_z_Cc
            )
        else:
            q_zCct = None
            if z_samples.dim() == 3:
                z_samples = z_samples.unsqueeze(2)

        R_target = self.target_dependent_representation(R, x_target, z_samples)
        p_yCc = self.decode_target(x_target, R_target)

        return p_yCc, z_samples, q_z_Cc, q_zCct

    def encode_globally(self, x_context, y_context, x_target):
        R = self.xy_encoder(x_context, y_context, x_target)

        if x_context.shape[1] == 0:
            R = torch.zeros((R.shape[0], 1, R.shape[-1])).to(R.device)

        return R

    def get_per_fact_knowledge(self, knowledge):
        if knowledge is None or not self.config.use_knowledge:
            return None, None

        if self.per_fact_encoder is not None:
            k_emb, mask = self.per_fact_encoder(knowledge)
        else:
            k = self.knowledge_encoder(knowledge)
            if k.dim() == 2:
                k = k.unsqueeze(1)
            k_emb = k
            mask = torch.ones(k_emb.shape[:2], dtype=torch.bool, device=k_emb.device)
        return k_emb, mask

    def infer_latent_dist_attn_poe(self, R, knowledge, n):
        del n  # unused but kept for API consistency
        drop_knowledge = torch.rand(1) < self.config.knowledge_dropout
        if drop_knowledge:
            K_emb, mask = None, None
        else:
            K_emb, mask = self.get_per_fact_knowledge(knowledge)

        r = R.mean(dim=1)
        mu, logvar = self.latent_attn_poe(r, K_emb=K_emb, K_mask=mask)
        mu = mu.unsqueeze(1)
        logvar = logvar.unsqueeze(1)

        q_z_scale = 0.01 + 0.99 * torch.exp(0.5 * logvar)
        q_zCc = MultivariateNormalDiag(mu, q_z_scale)
        return q_zCc

    def sample_latent(self, R, x_context, x_target, y_target, knowledge, q_zCc=None):
        if q_zCc is None:
            q_zCc = self.infer_latent_dist_attn_poe(R, knowledge, x_context.shape[1])

        if y_target is not None and self.training:
            R_from_target = self.encode_globally(x_target, y_target, x_target)
            q_zCct = self.infer_latent_dist_attn_poe(
                R_from_target, knowledge, x_target.shape[1]
            )
            sampling_dist = q_zCct
        else:
            q_zCct = None
            sampling_dist = q_zCc

        if self.training:
            z_samples = sampling_dist.rsample([self.train_num_z_samples])
        else:
            z_samples = sampling_dist.rsample([self.test_num_z_samples])
        return z_samples, q_zCc, q_zCct

    def target_dependent_representation(self, R, x_target, z_samples):
        R_target = z_samples
        R_target = R_target.expand(-1, -1, x_target.shape[1], -1)
        return R_target

    def decode_target(self, x_target, R_target):
        p_y_stats = self.decoder(x_target, R_target)

        p_y_loc, p_y_scale = p_y_stats.split(self.config.output_dim, dim=-1)
        p_y_scale = 0.1 + 0.9 * F.softplus(p_y_scale)

        p_yCc = MultivariateNormalDiag(p_y_loc, p_y_scale)

        return p_yCc

    def get_last_gate(self):
        return self.latent_attn_poe.last_tau

    def get_last_attention(self):
        return self.latent_attn_poe.last_attn
