"""Unit tests for encoder module."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from free_transformer.encoder import EncoderBlock


def test_encoder_forward():
    """Test encoder forward pass."""
    batch_size = 2
    seq_len = 128
    dim = 512
    num_heads = 8
    ffn_dim = 2048
    latent_dim = 16

    encoder = EncoderBlock(dim, num_heads, ffn_dim, latent_dim)
    context = torch.randn(batch_size, seq_len, dim)

    z_logits = encoder(context)

    assert z_logits.shape == (batch_size, seq_len, latent_dim)


def test_encoder_learned_query():
    """Test that encoder uses learned query."""
    encoder = EncoderBlock(512, 8, 2048, 16)
    assert encoder.zeta.requires_grad
    assert encoder.zeta.shape == (1, 1, 512)
