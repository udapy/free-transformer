# Quick Start

Get up and running with Free Transformer in minutes.

## 1. Generate Synthetic Data

Start with small synthetic data for quick experimentation:

```bash
make generate-data-small
```

This creates training data in `data/synthetic/` with:
- 1000 training sequences
- 200 validation sequences
- Sequence length: 128 tokens
- Vocabulary size: 1000

## 2. Train Baseline Model

Train a standard Transformer for comparison:

```bash
make train-baseline
```

This will:
- Use configuration from `configs/baseline.yaml`
- Save checkpoints to `checkpoints/baseline/`
- Log training metrics to TensorBoard

## 3. Train Free Transformer

Train the Free Transformer with latent planning:

```bash
make train-free
```

This will:
- Use configuration from `configs/free_transformer.yaml`
- Save checkpoints to `checkpoints/free/`
- Include VAE loss components (reconstruction + KL)

## 4. Compare Models

Evaluate and compare both models:

```bash
make compare
```

Results saved to `results/comparison/results.json` with:
- Perplexity scores
- Generation quality metrics
- Training efficiency comparisons

## Full Demo Pipeline

Run everything at once:

```bash
make demo
```

## Python API Quick Start

### Basic Usage

```python
import torch
from free_transformer import FreeTransformer, ModelConfig

# Create model configuration
config = ModelConfig(
    vocab_size=1000,
    hidden_dim=256,
    num_layers=6,
    num_heads=8,
    latent_dim=16,
    max_seq_len=512
)

# Initialize model
model = FreeTransformer(config)

# Training mode - with latent encoding
tokens = torch.randint(0, 1000, (2, 128))  # batch_size=2, seq_len=128
logits, z_logits = model(tokens, mode='training')

print(f"Output logits shape: {logits.shape}")  # [2, 128, 1000]
print(f"Latent logits shape: {z_logits.shape}")  # [2, 16]
```

### Generation

```python
# Inference mode - generate new tokens
prompt = torch.randint(0, 1000, (1, 10))  # batch_size=1, prompt_len=10
generated = model.generate(
    prompt, 
    max_new_tokens=50,
    temperature=0.8,
    top_k=40
)

print(f"Generated sequence shape: {generated.shape}")  # [1, 60]
```

### Custom Training Loop

```python
import torch.nn.functional as F
from free_transformer.losses import free_transformer_loss

# Training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.train()

for batch in dataloader:
    tokens = batch['input_ids']
    
    # Forward pass
    logits, z_logits = model(tokens, mode='training')
    
    # Compute loss
    loss_dict = free_transformer_loss(
        logits=logits,
        z_logits=z_logits,
        targets=tokens,
        latent_dim=config.latent_dim,
        kl_weight=0.1,
        free_bits=0.5
    )
    
    total_loss = loss_dict['total_loss']
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print(f"Loss: {total_loss.item():.4f}")
```

## Configuration

### Model Configuration

Key parameters in `ModelConfig`:

```python
config = ModelConfig(
    # Architecture
    vocab_size=50000,          # Vocabulary size
    hidden_dim=512,            # Hidden dimension
    num_layers=12,             # Number of transformer layers
    num_heads=8,               # Number of attention heads
    
    # Free Transformer specific
    latent_dim=32,             # Latent plan dimension
    encoder_layers=2,          # Number of encoder layers
    
    # Training
    max_seq_len=1024,          # Maximum sequence length
    dropout=0.1,               # Dropout rate
)
```

### Training Configuration

Edit YAML config files:

```yaml
# configs/free_transformer.yaml
model:
  vocab_size: 50000
  hidden_dim: 512
  num_layers: 12
  latent_dim: 32

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 10
  kl_weight: 0.1
  free_bits: 0.5

data:
  max_seq_len: 512
  dataset_name: "synthetic"
```

## Next Steps

- **[Architecture Overview](../architecture/overview.md)**: Understand the model design
- **[Training Guide](../training/guide.md)**: Advanced training techniques
- **[API Reference](../api/model.md)**: Complete API documentation
- **[Examples](../examples/basic.md)**: More detailed examples