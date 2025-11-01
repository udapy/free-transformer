"""Latent plan handling: Binary Mapper and sampling logic."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryMapper(nn.Module):
    """
    Differentiable discrete sampling using Straight-Through Estimator.

    Converts H logits into a 2^H dimensional one-hot vector by:
    1. Sampling H independent bits from sigmoid probabilities
    2. Converting bits to integer index d
    3. Creating one-hot vector with straight-through gradient
    """

    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.onehot_dim = 2**latent_dim

        # Pre-compute powers of 2 for bit-to-int conversion
        powers = torch.tensor([2**i for i in range(latent_dim)])
        self.register_buffer("powers", powers)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sample one-hot Z from logits using straight-through estimator.

        Args:
            logits: Binary logits [batch, seq_len, latent_dim]

        Returns:
            z_onehot: One-hot sampled Z [batch, seq_len, 2^latent_dim]
        """
        batch_size, seq_len, _ = logits.shape

        # 1. Convert logits to bit probabilities via sigmoid (Eq. 6)
        bit_probs = torch.sigmoid(logits)  # P(B_t,h = 1)

        # 2. Sample bits
        if self.training:
            # Stochastic sampling during training
            bits = torch.bernoulli(bit_probs)
        else:
            # Deterministic (threshold at 0.5) during eval
            bits = (bit_probs > 0.5).float()

        # 3. Convert H bits to integer index d (Eq. 7)
        # d = sum(B_t,h * 2^h) for h in 0..H-1
        powers_tensor = getattr(self, "powers")  # This is a registered buffer, so it's a Tensor
        assert isinstance(
            powers_tensor, torch.Tensor
        ), "powers should be a registered tensor buffer"
        indices = torch.sum(bits * powers_tensor, dim=-1).long()  # [batch, seq_len]

        # 4. Create one-hot vectors
        y_discrete = F.one_hot(indices, num_classes=self.onehot_dim).float()

        # 5. Compute continuous probability for straight-through (Eq. 8)
        # G_t,d = product of P(B_t,h = b_h) for each bit
        bit_log_probs = bits * torch.log(bit_probs + 1e-10) + (1 - bits) * torch.log(
            1 - bit_probs + 1e-10
        )
        g_continuous = torch.exp(bit_log_probs.sum(dim=-1))  # [batch, seq_len]

        # Convert to one-hot representation matching discrete output
        g_onehot = torch.zeros_like(y_discrete)
        g_onehot.scatter_(-1, indices.unsqueeze(-1), g_continuous.unsqueeze(-1))

        # 6. Straight-through estimator (Eq. 8)
        # Forward: use discrete Y, Backward: use continuous G
        z_onehot = y_discrete + (g_onehot - g_onehot.detach())

        return z_onehot


class LatentPlan(nn.Module):
    """
    Latent plan Z handler: sampling and projection.
    """

    def __init__(self, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.onehot_dim = 2**latent_dim

        # Binary mapper for differentiable sampling
        self.binary_mapper = BinaryMapper(latent_dim)

        # Post-sampler projection (Eq. in Algorithm 2)
        self.post_sampler_fc = nn.Linear(self.onehot_dim, hidden_dim, bias=False)

    def sample_from_logits(self, z_logits: torch.Tensor) -> torch.Tensor:
        """
        Sample Z from encoder logits (training mode).

        Args:
            z_logits: Encoder output [batch, seq_len, latent_dim]

        Returns:
            z_onehot: Sampled one-hot Z [batch, seq_len, 2^latent_dim]
        """
        result = self.binary_mapper(z_logits)
        assert isinstance(result, torch.Tensor)
        return result

    def sample_from_prior(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample Z from uniform prior (inference mode).

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            device: Target device

        Returns:
            z_onehot: Random one-hot Z [batch, seq_len, 2^latent_dim]
        """
        # Sample random indices from uniform distribution
        indices = torch.randint(0, self.onehot_dim, (batch_size, seq_len), device=device)

        # Convert to one-hot
        z_onehot = F.one_hot(indices, num_classes=self.onehot_dim).float()

        return z_onehot

    def project_to_hidden(self, z_onehot: torch.Tensor) -> torch.Tensor:
        """
        Project one-hot Z to hidden dimension for injection.

        Args:
            z_onehot: One-hot Z [batch, seq_len, 2^latent_dim]

        Returns:
            z_projected: Projected Z [batch, seq_len, hidden_dim]
        """
        result = self.post_sampler_fc(z_onehot)
        assert isinstance(result, torch.Tensor)
        return result
