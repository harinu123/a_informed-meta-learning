import torch
import torch.nn as nn

from models.modules import MLP


class SetEncoderPerFact(nn.Module):
    """Per-fact encoder that preserves the set structure for attention."""

    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.fact_encoder = MLP(
            input_size=config.knowledge_input_dim,
            hidden_size=config.knowledge_dim,
            num_hidden=1,
            output_size=config.knowledge_dim,
        )
        self.post_mlp = MLP(
            input_size=config.knowledge_dim,
            hidden_size=config.knowledge_dim,
            num_hidden=1,
            output_size=config.knowledge_dim,
        )

    def forward(self, knowledge):
        if knowledge is None:
            return None, None

        knowledge = knowledge.to(self.device)
        mask = torch.ones(knowledge.shape[:2], dtype=torch.bool, device=self.device)
        k = self.fact_encoder(knowledge)
        k = self.post_mlp(k)
        return k, mask
