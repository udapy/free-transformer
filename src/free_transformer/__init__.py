"""
Free Transformer: A modular implementation of the Free Transformer architecture.
"""

__version__ = "0.1.2"

from .baseline import TransformerBaseline
from .config import ModelConfig, TrainingConfig
from .encoder import EncoderBlock
from .latent import BinaryMapper, LatentPlan
from .losses import compute_vae_loss
from .model import FreeTransformer
from .synthetic_data import SyntheticDataGenerator

__all__ = [
    "FreeTransformer",
    "TransformerBaseline",
    "EncoderBlock",
    "BinaryMapper",
    "LatentPlan",
    "compute_vae_loss",
    "ModelConfig",
    "TrainingConfig",
    "SyntheticDataGenerator",
]
