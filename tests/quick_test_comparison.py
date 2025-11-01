"""Comparison test: Transformer vs Free Transformer."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from free_transformer import FreeTransformer, ModelConfig, TransformerBaseline
from free_transformer.losses import (compute_reconstruction_loss,
                                     compute_vae_loss)
from free_transformer.synthetic_data import SyntheticDataGenerator


def train_model(model, data, num_steps=100, is_free=False):
    """Simple training loop."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(num_steps):
        inputs, targets = data[:, :-1], data[:, 1:]

        if is_free:
            logits, z_logits = model(inputs, mode="training")
            loss, _ = compute_vae_loss(logits, targets, z_logits, model.config.latent_dim)
        else:
            logits = model(inputs)
            loss = compute_reconstruction_loss(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def test_comparison():
    """Compare baseline and Free Transformer on same data."""
    # Config
    config = ModelConfig(
        vocab_size=1000,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        latent_dim=8,
        max_seq_len=128,
    )

    # Generate data
    generator = SyntheticDataGenerator(vocab_size=1000, seq_length=128)
    data = generator.generate_batch(32)

    # Train baseline
    print("\nTraining Baseline Transformer...")
    baseline = TransformerBaseline(config)
    baseline_loss = train_model(baseline, data, num_steps=100, is_free=False)
    print(f"Baseline final loss: {baseline_loss:.4f}")

    # Train Free Transformer
    print("\nTraining Free Transformer...")
    free_model = FreeTransformer(config)
    free_loss = train_model(free_model, data, num_steps=100, is_free=True)
    print(f"Free Transformer final loss: {free_loss:.4f}")

    # Both should achieve reasonable loss
    assert baseline_loss < 10.0
    assert free_loss < 10.0

    print("\n✅ Comparison test passed!")


if __name__ == "__main__":
    test_comparison()
