"""Configuration management for Free Transformer."""

from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    # Model dimensions
    vocab_size: int = 32000
    hidden_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: Optional[int] = None  # GQA, None = same as num_heads
    ffn_hidden_dim: int = 11008
    max_seq_len: int = 2048

    # Free Transformer specific
    latent_dim: int = 16  # H in paper, produces 2^16 = 65536 dimensional one-hot
    split_layer: Optional[int] = None  # Auto-computed as num_layers // 2 if None

    # Architecture details
    use_rmsnorm: bool = True
    use_rope: bool = True
    use_swiglu: bool = True
    dropout: float = 0.0

    # Attention
    attention_dropout: float = 0.0

    def __post_init__(self):
        if self.split_layer is None:
            self.split_layer = self.num_layers // 2
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 100000

    # Loss hyperparameters
    beta_kl: float = 1.0  # Weight for KL term
    kappa_free_bits: float = 0.3466  # log(2)/2 ≈ 0.3466 bits

    # Batch settings
    batch_size: int = 64
    gradient_accumulation_steps: int = 1

    # Distributed training
    use_fsdp: bool = False
    use_deepspeed: bool = False
    fsdp_config: dict = field(default_factory=dict)
    deepspeed_config: dict = field(default_factory=dict)

    # Checkpointing
    save_every: int = 5000
    eval_every: int = 1000
    checkpoint_dir: str = "./checkpoints"

    # Logging
    log_every: int = 100
    wandb_project: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str):
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f)


@dataclass
class DataConfig:
    """Configuration for data processing."""

    dataset_name: str = "synthetic"
    tokenizer_name: str = "gpt2"
    max_length: int = 2048

    # Synthetic data specific
    synthetic_vocab_size: int = 10000
    synthetic_seq_length: int = 512
    num_train_samples: int = 50000
    num_val_samples: int = 5000
    seed: int = 42
