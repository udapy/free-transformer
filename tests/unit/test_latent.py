"""Unit tests for latent plan module."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from free_transformer.latent import BinaryMapper, LatentPlan


def test_binary_mapper():
    """Test binary mapper sampling."""
    batch_size = 2
    seq_len = 128
    latent_dim = 16

    mapper = BinaryMapper(latent_dim)
    logits = torch.randn(batch_size, seq_len, latent_dim)

    z_onehot = mapper(logits)

    # Check shape
    assert z_onehot.shape == (batch_size, seq_len, 2**latent_dim)

    # Check one-hot property (approximately, due to STE)
    assert torch.allclose(z_onehot.sum(dim=-1), torch.ones(batch_size, seq_len), atol=0.1)


def test_latent_plan_prior_sampling():
    """Test sampling from uniform prior."""
    batch_size = 2
    seq_len = 128
    latent_dim = 16
    hidden_dim = 512

    latent_plan = LatentPlan(latent_dim, hidden_dim)
    z_onehot = latent_plan.sample_from_prior(batch_size, seq_len, torch.device("cpu"))

    # Check shape
    assert z_onehot.shape == (batch_size, seq_len, 2**latent_dim)

    # Check one-hot property
    assert torch.allclose(z_onehot.sum(dim=-1), torch.ones(batch_size, seq_len))


def test_gradient_flow():
    """Test that gradients flow through binary mapper."""
    mapper = BinaryMapper(16)
    logits = torch.randn(2, 128, 16, requires_grad=True)

    z_onehot = mapper(logits)
    loss = z_onehot.sum()
    loss.backward()

    assert logits.grad is not None
