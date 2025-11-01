"""Encoder module for latent plan inference."""

import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    """
    Non-causal encoder block for inferring latent plan Z from context.

    Uses a learned query embedding (zeta) to aggregate global information
    from the entire sequence via non-causal attention.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        latent_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.latent_dim = latent_dim

        # Learned query embedding (zeta in paper)
        self.zeta = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Non-causal attention (no masking)
        from .model import RMSNorm, TransformerBlock

        self.transformer_block = TransformerBlock(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            causal=False,  # Non-causal for global planning
            use_rope=False,  # No positional encoding in encoder
        )

        # Encoder readout: project to latent dimension
        self.norm = RMSNorm(dim)
        self.readout = nn.Linear(dim, latent_dim, bias=False)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Infer latent plan from context.

        Args:
            context: Context from first decoder half [batch, seq_len, dim]

        Returns:
            z_logits: Logits for latent plan [batch, seq_len, latent_dim]
        """
        batch_size, seq_len, _ = context.shape

        # Expand learned query for batch and sequence
        queries = self.zeta.expand(batch_size, seq_len, -1)

        # Non-causal attention over full context
        # queries attend to all of context (keys/values)
        encoder_output = self.transformer_block(queries, kv_input=context)

        # Project to latent dimension
        encoder_output = self.norm(encoder_output)
        z_logits = self.readout(encoder_output)

        assert isinstance(z_logits, torch.Tensor)
        return z_logits
