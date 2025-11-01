"""Integration tests for training pipeline."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from free_transformer import FreeTransformer, ModelConfig, TransformerBaseline
from free_transformer.losses import (compute_reconstruction_loss,
                                     compute_vae_loss)
from free_transformer.synthetic_data import SyntheticDataGenerator


@pytest.fixture
def small_config():
    return ModelConfig(
        vocab_size=1000,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        latent_dim=8,
        max_seq_len=64,
    )


@pytest.fixture
def synthetic_batch():
    generator = SyntheticDataGenerator(vocab_size=1000, seq_length=64)
    tokens = generator.generate_batch(4)
    return tokens[:, :-1], tokens[:, 1:]


def test_baseline_training_step(small_config, synthetic_batch):
    """Test single training step for baseline."""
    model = TransformerBaseline(small_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    inputs, targets = synthetic_batch

    # Forward
    logits = model(inputs)
    loss = compute_reconstruction_loss(logits, targets)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() > 0


def test_free_transformer_training_step(small_config, synthetic_batch):
    """Test single training step for Free Transformer."""
    model = FreeTransformer(small_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    inputs, targets = synthetic_batch

    # Forward
    logits, z_logits = model(inputs, mode="training")
    loss, metrics = compute_vae_loss(logits, targets, z_logits, small_config.latent_dim)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() > 0
    assert "loss/reconstruction" in metrics
    assert "loss/kl" in metrics


def test_overfitting_single_batch(small_config, synthetic_batch):
    """Test that model can overfit single batch."""
    model = FreeTransformer(small_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    inputs, targets = synthetic_batch

    initial_loss = None
    for step in range(50):
        logits, z_logits = model(inputs, mode="training")
        loss, _ = compute_vae_loss(logits, targets, z_logits, small_config.latent_dim)

        if initial_loss is None:
            initial_loss = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.5  # Should decrease significantly
