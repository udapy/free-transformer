"""Unit tests for loss functions."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from free_transformer.losses import (compute_kl_divergence,
                                     compute_reconstruction_loss,
                                     compute_vae_loss)


def test_reconstruction_loss():
    """Test reconstruction loss calculation."""
    batch_size = 2
    seq_len = 128
    vocab_size = 10000

    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    loss = compute_reconstruction_loss(logits, targets)

    assert loss.dim() == 0  # Scalar
    assert loss.item() > 0


def test_kl_divergence():
    """Test KL divergence calculation."""
    batch_size = 2
    seq_len = 128
    latent_dim = 16

    z_logits = torch.randn(batch_size, seq_len, latent_dim)
    kl_loss = compute_kl_divergence(z_logits, latent_dim, free_bits=0.5)

    assert kl_loss.dim() == 0
    assert kl_loss.item() >= 0


def test_vae_loss():
    """Test complete VAE loss."""
    batch_size = 2
    seq_len = 128
    vocab_size = 10000
    latent_dim = 16

    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    z_logits = torch.randn(batch_size, seq_len, latent_dim)

    total_loss, metrics = compute_vae_loss(logits, targets, z_logits, latent_dim)

    assert "loss/total" in metrics
    assert "loss/reconstruction" in metrics
    assert "loss/kl" in metrics
    assert "metrics/perplexity" in metrics
