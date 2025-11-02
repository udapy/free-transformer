# Installation

## Requirements

- Python 3.12+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

## Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is the fastest way to install Free Transformer:

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install Free Transformer with development dependencies
uv pip install -e ".[dev]"
```

## Using pip

After the PyPI release:

```bash
pip install free-transformer
```

For development:

```bash
git clone https://github.com/udapy/free-transformer.git
cd free-transformer
pip install -e ".[dev]"
```

## Optional Dependencies

### Documentation
```bash
uv pip install -e ".[docs]"
```

### DeepSpeed (Future)
```bash
uv pip install -e ".[deepspeed]"
```

## Verify Installation

Test your installation:

```python
import torch
from free_transformer import FreeTransformer, ModelConfig

# Create a small model
config = ModelConfig(
    vocab_size=1000,
    hidden_dim=128,
    num_layers=4,
    num_heads=4,
    latent_dim=8,
)

model = FreeTransformer(config)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
```

## GPU Setup

For CUDA support, ensure you have:

1. NVIDIA drivers installed
2. CUDA toolkit (11.8+ or 12.0+)
3. PyTorch with CUDA support

Verify GPU availability:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'free_transformer'**
- Ensure you're in the correct virtual environment
- Try reinstalling with `uv pip install -e .`

**CUDA out of memory**
- Reduce batch size in config files
- Enable gradient checkpointing
- Use smaller model dimensions

**Slow training**
- Verify GPU is being used
- Enable mixed precision training
- Consider multi-GPU setup with FSDP

### Getting Help

- Check the [FAQ](../faq.md)
- Open an issue on [GitHub](https://github.com/udapy/free-transformer/issues)
- Review the [troubleshooting guide](../development/testing.md#troubleshooting)