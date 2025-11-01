"""Injection mechanism for integrating latent plan into decoder."""

import torch
import torch.nn as nn


class InjectionMechanism(nn.Module):
    """
    Injects latent plan Z into decoder by adding to context.

    Implements Algorithm 2 from paper:
    - Projects Z to hidden dimension
    - Adds to context representation
    - Used as keys/values for next decoder block
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # No additional parameters needed - projection is in LatentPlan
        self.hidden_dim = hidden_dim

    def forward(
        self,
        context: torch.Tensor,
        z_projected: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inject latent plan into context.

        Args:
            context: Context from first decoder half [batch, seq_len, hidden_dim]
            z_projected: Projected latent plan [batch, seq_len, hidden_dim]

        Returns:
            context_with_plan: Injected representation [batch, seq_len, hidden_dim]
        """
        # Import here to avoid circular dependency

        # Get the parent model's latent_plan module to access projection
        # In practice, this is called from the main model which handles projection
        # This is a simplified version - actual implementation gets projection from parent

        # Z should already be projected to hidden_dim by the model
        assert (
            z_projected.shape[-1] == self.hidden_dim
        ), "Z must be projected to hidden_dim before injection"

        # Add plan to context (Eq. in Algorithm 2)
        context_with_plan = context + z_projected

        return context_with_plan
