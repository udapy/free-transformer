# Configuration Reference

This page provides detailed information about configuring Free Transformer models and training.

## Configuration Files

Free Transformer uses YAML configuration files to manage model and training parameters. The main configuration files are:

- `configs/free_transformer.yaml` - Free Transformer model configuration
- `configs/baseline.yaml` - Baseline Transformer configuration

## Model Configuration

### Basic Model Parameters

```yaml
model:
  # Architecture basics
  vocab_size: 50000           # Vocabulary size
  hidden_dim: 512             # Hidden dimension (d_model)
  num_layers: 12              # Number of transformer layers
  num_heads: 8                # Number of attention heads
  max_seq_len: 1024           # Maximum sequence length
  
  # Free Transformer specific
  latent_dim: 32              # Latent plan dimension
  encoder_layers: 2           # Number of encoder layers
  
  # Architecture details
  intermediate_size: 2048     # FFN intermediate size (usually 4 * hidden_dim)
  dropout: 0.1                # Dropout rate
  attention_dropout: 0.1      # Attention dropout rate
  
  # Position encoding
  rope_theta: 10000.0         # RoPE base frequency
  
  # Grouped-Query Attention
  num_key_value_heads: 2      # Number of KV heads (for GQA)
```

### Advanced Model Parameters

```yaml
model:
  # Normalization
  rms_norm_eps: 1e-6          # RMSNorm epsilon
  
  # Activation functions
  hidden_act: "silu"          # Activation function (silu, gelu, relu)
  
  # Initialization
  initializer_range: 0.02     # Weight initialization range
  
  # Binary mapping
  gumbel_temperature: 1.0     # Gumbel-Softmax temperature
  binary_threshold: 0.5       # Binary threshold for inference
  
  # Plan injection
  injection_method: "additive" # additive, gated, concat, cross_attention
  injection_layers: [6, 8, 10] # Which layers to inject plan (if multi-layer)
```

## Training Configuration

### Basic Training Parameters

```yaml
training:
  # Optimization
  batch_size: 32              # Training batch size
  eval_batch_size: 64         # Evaluation batch size
  learning_rate: 1e-4         # Peak learning rate
  num_epochs: 10              # Number of training epochs
  max_steps: null             # Maximum training steps (overrides epochs)
  
  # Regularization
  weight_decay: 0.01          # Weight decay coefficient
  dropout: 0.1                # Dropout rate
  gradient_clip_norm: 1.0     # Gradient clipping norm
  
  # Free Transformer specific
  kl_weight: 0.1              # KL divergence loss weight
  free_bits: 0.5              # Free bits threshold
  
  # Evaluation
  eval_steps: 500             # Steps between evaluations
  save_steps: 1000            # Steps between checkpoints
  logging_steps: 100          # Steps between log outputs
```

### Advanced Training Parameters

```yaml
training:
  # Learning rate scheduling
  warmup_steps: 1000          # Warmup steps
  warmup_ratio: 0.1           # Warmup ratio (alternative to warmup_steps)
  lr_scheduler_type: "cosine" # cosine, linear, constant
  min_lr_ratio: 0.1           # Minimum LR as ratio of peak LR
  
  # KL annealing
  kl_annealing: true          # Enable KL weight annealing
  kl_annealing_steps: 5000    # Steps to anneal KL weight
  initial_kl_weight: 1.0      # Initial KL weight
  final_kl_weight: 0.1        # Final KL weight
  
  # Free bits scheduling
  free_bits_annealing: true   # Enable free bits annealing
  initial_free_bits: 2.0      # Initial free bits
  final_free_bits: 0.5        # Final free bits
  
  # Mixed precision
  fp16: false                 # Use FP16 mixed precision
  bf16: true                  # Use BF16 mixed precision
  
  # Memory optimization
  gradient_checkpointing: true # Enable gradient checkpointing
  dataloader_num_workers: 4   # Number of data loading workers
  dataloader_pin_memory: true # Pin memory for data loading
```

## Optimizer Configuration

```yaml
optimizer:
  type: "adamw"               # Optimizer type
  betas: [0.9, 0.95]         # Adam beta parameters
  eps: 1e-8                  # Adam epsilon
  weight_decay: 0.01         # Weight decay
  
  # Alternative optimizers
  # type: "sgd"
  # momentum: 0.9
  # nesterov: true
```

## Data Configuration

```yaml
data:
  # Dataset
  dataset_name: "synthetic"   # Dataset name or path
  dataset_config: null        # Dataset configuration
  
  # Processing
  max_seq_len: 512           # Maximum sequence length for training
  tokenizer_name: null       # Tokenizer name (if using real data)
  
  # Synthetic data (if using synthetic dataset)
  num_train_samples: 10000   # Number of training samples
  num_val_samples: 1000      # Number of validation samples
  vocab_size: 50000          # Vocabulary size for synthetic data
  
  # Data loading
  shuffle: true              # Shuffle training data
  drop_last: true            # Drop last incomplete batch
```

## Distributed Training Configuration

```yaml
distributed:
  # FSDP (Fully Sharded Data Parallel)
  use_fsdp: false            # Enable FSDP
  fsdp_sharding_strategy: "full_shard" # full_shard, shard_grad_op, no_shard
  fsdp_backward_prefetch: "backward_pre" # backward_pre, backward_post
  fsdp_forward_prefetch: false # Enable forward prefetch
  
  # Model wrapping
  fsdp_auto_wrap_policy: "transformer_auto_wrap" # Auto-wrap policy
  fsdp_min_num_params: 1e6   # Minimum parameters for wrapping
  
  # Checkpointing
  fsdp_state_dict_type: "full_state_dict" # full_state_dict, local_state_dict, sharded_state_dict
```

## Logging and Monitoring

```yaml
logging:
  # Output directories
  output_dir: "./checkpoints" # Checkpoint output directory
  logging_dir: "./logs"      # Logging directory
  
  # Weights & Biases
  use_wandb: false           # Enable W&B logging
  wandb_project: "free-transformer" # W&B project name
  wandb_run_name: null       # W&B run name
  wandb_tags: []             # W&B tags
  
  # TensorBoard
  use_tensorboard: true      # Enable TensorBoard logging
  
  # Console logging
  log_level: "info"          # Logging level
  disable_tqdm: false        # Disable progress bars
```

## Environment Configuration

```yaml
environment:
  # Device
  device: "auto"             # Device (auto, cpu, cuda, cuda:0)
  
  # Random seeds
  seed: 42                   # Random seed
  
  # CUDA settings
  cuda_deterministic: false  # Enable CUDA deterministic operations
  cuda_benchmark: true       # Enable CUDA benchmark mode
  
  # Memory
  empty_cache_steps: 1000    # Steps between cache clearing
```

## Configuration Examples

### Small Model for Testing

```yaml
# configs/small.yaml
model:
  vocab_size: 1000
  hidden_dim: 128
  num_layers: 4
  num_heads: 4
  latent_dim: 8
  max_seq_len: 256

training:
  batch_size: 8
  learning_rate: 1e-3
  num_epochs: 5
  kl_weight: 0.1
  free_bits: 0.5

data:
  dataset_name: "synthetic"
  num_train_samples: 1000
  num_val_samples: 200
  max_seq_len: 128
```

### Large Model for Production

```yaml
# configs/large.yaml
model:
  vocab_size: 50000
  hidden_dim: 1024
  num_layers: 24
  num_heads: 16
  latent_dim: 64
  max_seq_len: 2048
  num_key_value_heads: 4

training:
  batch_size: 16
  learning_rate: 5e-5
  num_epochs: 3
  warmup_steps: 2000
  gradient_checkpointing: true
  bf16: true
  kl_weight: 0.05
  free_bits: 1.0

distributed:
  use_fsdp: true
  fsdp_sharding_strategy: "full_shard"

logging:
  use_wandb: true
  wandb_project: "free-transformer-large"
```

### Curriculum Learning Configuration

```yaml
# configs/curriculum.yaml
model:
  vocab_size: 32000
  hidden_dim: 512
  num_layers: 12
  num_heads: 8
  latent_dim: 32

training:
  # Phase 1: Short sequences, high KL weight
  phase1:
    max_seq_len: 128
    batch_size: 64
    kl_weight: 1.0
    free_bits: 2.0
    num_epochs: 3
  
  # Phase 2: Medium sequences, medium KL weight
  phase2:
    max_seq_len: 256
    batch_size: 32
    kl_weight: 0.5
    free_bits: 1.0
    num_epochs: 3
  
  # Phase 3: Long sequences, low KL weight
  phase3:
    max_seq_len: 512
    batch_size: 16
    kl_weight: 0.1
    free_bits: 0.5
    num_epochs: 4
```

## Using Configurations

### Command Line

```bash
# Use specific config file
python examples/train_free.py --config configs/small.yaml

# Override specific parameters
python examples/train_free.py \
  --config configs/free_transformer.yaml \
  --batch-size 16 \
  --learning-rate 5e-5 \
  --kl-weight 0.05
```

### Python API

```python
import yaml
from free_transformer import ModelConfig, TrainingConfig

# Load from YAML
with open('configs/free_transformer.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

model_config = ModelConfig(**config_dict['model'])
training_config = TrainingConfig(**config_dict['training'])

# Create model
model = FreeTransformer(model_config)
```

### Dynamic Configuration

```python
def create_config_for_size(model_size: str):
    """Create configuration based on model size."""
    if model_size == "small":
        return ModelConfig(
            vocab_size=1000,
            hidden_dim=128,
            num_layers=4,
            latent_dim=8
        )
    elif model_size == "medium":
        return ModelConfig(
            vocab_size=32000,
            hidden_dim=512,
            num_layers=12,
            latent_dim=32
        )
    elif model_size == "large":
        return ModelConfig(
            vocab_size=50000,
            hidden_dim=1024,
            num_layers=24,
            latent_dim=64
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}")

# Usage
config = create_config_for_size("medium")
model = FreeTransformer(config)
```

## Configuration Validation

The configuration system includes validation to catch common errors:

```python
from free_transformer.config import validate_config

# This will raise an error if configuration is invalid
validate_config(config_dict)

# Common validation checks:
# - hidden_dim must be divisible by num_heads
# - latent_dim must be positive
# - batch_size must be positive
# - learning_rate must be positive
# - kl_weight must be non-negative
```

## Best Practices

1. **Start small**: Begin with small configurations for testing
2. **Use templates**: Copy and modify existing configurations
3. **Version control**: Keep configuration files in version control
4. **Document changes**: Comment important configuration choices
5. **Validate early**: Check configurations before long training runs
6. **Monitor resources**: Ensure configurations fit your hardware

## Next Steps

- **[Training Guide](guide.md)**: Learn how to use these configurations
- **[Multi-GPU Training](multi-gpu.md)**: Distributed training setup
- **[Examples](../examples/basic.md)**: See configurations in action