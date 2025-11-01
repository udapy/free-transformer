"""Loss functions for Free Transformer training."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_reconstruction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Standard cross-entropy loss for token prediction.

    Args:
        logits: Model predictions [batch, seq_len, vocab_size]
        targets: Target tokens [batch, seq_len]
        ignore_index: Index to ignore in loss computation

    Returns:
        loss: Scalar loss value
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Reshape for cross-entropy
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=ignore_index, reduction="mean")

    return loss


def compute_kl_divergence(
    z_logits: torch.Tensor,
    latent_dim: int = 16,
    free_bits: float = 0.3466,  # log(2)/2
) -> torch.Tensor:
    """
    Compute KL divergence with free bits for latent plan Z.

    Implements Equation (5) from paper:
    (1/T) * sum_t max(0, D_KL(Q(Z_t|S) || P(Z_t)) - kappa)

    Args:
        z_logits: Encoder logits [batch, seq_len, latent_dim]
        latent_dim: Dimension of latent space (H)
        free_bits: Free bits budget (kappa)

    Returns:
        kl_loss: Scalar KL divergence loss
    """
    batch_size, seq_len, H = z_logits.shape
    assert H == latent_dim, f"Expected latent_dim={latent_dim}, got {H}"

    # Convert logits to bit probabilities
    bit_probs = torch.sigmoid(z_logits)  # [batch, seq_len, H]

    # For each bit, compute KL between Bernoulli(p) and Uniform(0.5)
    # KL(Bernoulli(p) || Bernoulli(0.5)) = p*log(2p) + (1-p)*log(2(1-p))
    eps = 1e-10
    kl_per_bit = bit_probs * torch.log(2 * bit_probs + eps) + (1 - bit_probs) * torch.log(
        2 * (1 - bit_probs) + eps
    )

    # Sum over H bits to get KL for each Z_t
    kl_per_position = kl_per_bit.sum(dim=-1)  # [batch, seq_len]

    # Apply free bits: max(0, KL - kappa)
    kl_with_free_bits = torch.clamp(kl_per_position - free_bits, min=0.0)

    # Average over batch and sequence
    kl_loss = kl_with_free_bits.mean()

    return kl_loss


def compute_vae_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    z_logits: torch.Tensor,
    latent_dim: int = 16,
    beta_kl: float = 1.0,
    free_bits: float = 0.3466,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, dict]:
    """
    Complete VAE loss for Free Transformer.

    Loss = Reconstruction + beta * KL_with_free_bits

    Args:
        logits: Model predictions [batch, seq_len, vocab_size]
        targets: Target tokens [batch, seq_len]
        z_logits: Encoder logits [batch, seq_len, latent_dim]
        latent_dim: Latent dimension
        beta_kl: Weight for KL term
        free_bits: Free bits budget
        ignore_index: Index to ignore in reconstruction loss

    Returns:
        total_loss: Combined loss
        metrics: Dictionary of individual loss components
    """
    # Reconstruction loss
    recon_loss = compute_reconstruction_loss(logits, targets, ignore_index)

    # KL divergence loss with free bits
    kl_loss = compute_kl_divergence(z_logits, latent_dim, free_bits)

    # Combined loss
    total_loss = recon_loss + beta_kl * kl_loss

    # Metrics for logging
    metrics = {
        "loss/total": total_loss.item(),
        "loss/reconstruction": recon_loss.item(),
        "loss/kl": kl_loss.item(),
        "loss/kl_weighted": (beta_kl * kl_loss).item(),
    }

    # Compute perplexity
    with torch.no_grad():
        perplexity = torch.exp(recon_loss)
        metrics["metrics/perplexity"] = perplexity.item()

    return total_loss, metrics


class FreeTransformerLoss(nn.Module):
    """Wrapper class for Free Transformer loss."""

    def __init__(
        self,
        latent_dim: int = 16,
        beta_kl: float = 1.0,
        free_bits: float = 0.3466,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta_kl = beta_kl
        self.free_bits = free_bits
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        z_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        return compute_vae_loss(
            logits,
            targets,
            z_logits,
            self.latent_dim,
            self.beta_kl,
            self.free_bits,
            self.ignore_index,
        )
