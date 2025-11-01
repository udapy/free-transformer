"""Baseline standard Transformer for comparison."""

import torch
import torch.nn as nn

from .model import RMSNorm, TransformerBlock


class TransformerBaseline(nn.Module):
    """Standard autoregressive Transformer without latent planning."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=config.hidden_dim,
                    num_heads=config.num_heads,
                    num_kv_heads=config.num_kv_heads,
                    ffn_dim=config.ffn_hidden_dim,
                    dropout=config.dropout,
                    causal=True,
                    use_rope=config.use_rope,
                    max_seq_len=config.max_seq_len,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.norm = RMSNorm(config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.output(x)
        assert isinstance(logits, torch.Tensor)
        return logits
