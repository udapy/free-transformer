# Code Quality

This guide covers code quality tools, standards, and practices for the Free Transformer project.

## Quality Tools

The project uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **Ruff**: Fast Python linter
- **MyPy**: Static type checking
- **Pytest**: Testing framework

## Running Quality Checks

### Quick Commands

```bash
# Run all quality checks
make quality

# Individual tools
make format          # Format code with black and isort
make format-check    # Check formatting without changes
make lint           # Run ruff linter
make type-check     # Run mypy type checker
```

### Pre-commit Hooks

Install pre-commit hooks to run quality checks automatically:

```bash
# Install pre-commit
uv pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Code Formatting

### Black Configuration

Black is configured in `pyproject.toml`:

```toml
[tool.black]
line-length = 100
target-version = ['py312']
```

### Usage Examples

```bash
# Format all code
black src/ tests/ examples/

# Check formatting without changes
black --check src/ tests/ examples/

# Format specific file
black src/free_transformer/model.py
```

### Import Sorting with isort

isort configuration in `pyproject.toml`:

```toml
[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
```

Example usage:

```bash
# Sort imports
isort src/ tests/ examples/

# Check import sorting
isort --check-only src/ tests/ examples/
```

## Linting with Ruff

### Configuration

Ruff configuration in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]
ignore = [
    "E501",  # line too long (handled by black)
    "B008",  # do not perform function calls in argument defaults
    "C901",  # too complex
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # Allow unused imports in __init__.py
"tests/*" = ["B011"]      # Allow assert False in tests
```

### Common Lint Issues and Fixes

**Unused imports**:
```python
# Bad
import torch
import numpy as np  # Unused

def forward(x):
    return torch.relu(x)

# Good
import torch

def forward(x):
    return torch.relu(x)
```

**Long lines**:
```python
# Bad
def very_long_function_name(very_long_parameter_name, another_very_long_parameter_name, yet_another_parameter):
    pass

# Good
def very_long_function_name(
    very_long_parameter_name,
    another_very_long_parameter_name,
    yet_another_parameter
):
    pass
```

**Mutable default arguments**:
```python
# Bad
def process_data(data, config={}):
    config['processed'] = True
    return data

# Good
def process_data(data, config=None):
    if config is None:
        config = {}
    config['processed'] = True
    return data
```

## Type Checking with MyPy

### Configuration

MyPy configuration in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
```

### Type Annotation Examples

**Function annotations**:
```python
from typing import Optional, List, Dict, Tuple, Union
import torch

def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """Compute loss with optional weights."""
    loss = F.cross_entropy(logits, targets, reduction='none')
    
    if weights is not None:
        loss = loss * weights
    
    return {
        'loss': loss.mean(),
        'raw_loss': loss
    }
```

**Class annotations**:
```python
from typing import List, Optional
import torch.nn as nn

class FreeTransformer(nn.Module):
    """Free Transformer model with type annotations."""
    
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.layers: List[TransformerBlock] = []
        self._cache: Optional[Dict[str, torch.Tensor]] = None
    
    def forward(
        self, 
        tokens: torch.Tensor, 
        mode: str = 'training'
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with type hints."""
        # Implementation here
        pass
```

**Generic types**:
```python
from typing import TypeVar, Generic, List

T = TypeVar('T')

class DataLoader(Generic[T]):
    """Generic data loader."""
    
    def __init__(self, data: List[T]) -> None:
        self.data = data
    
    def __getitem__(self, index: int) -> T:
        return self.data[index]
```

### Common Type Issues

**Missing return type**:
```python
# Bad
def process_tokens(tokens):
    return tokens.long()

# Good
def process_tokens(tokens: torch.Tensor) -> torch.Tensor:
    return tokens.long()
```

**Any type usage**:
```python
# Bad
from typing import Any

def process_data(data: Any) -> Any:
    return data.process()

# Good
from typing import Protocol

class Processable(Protocol):
    def process(self) -> 'Processable':
        ...

def process_data(data: Processable) -> Processable:
    return data.process()
```

## Documentation Standards

### Docstring Format

Use Google-style docstrings:

```python
def free_transformer_loss(
    logits: torch.Tensor,
    z_logits: torch.Tensor,
    targets: torch.Tensor,
    latent_dim: int,
    kl_weight: float = 0.1,
    free_bits: float = 0.5
) -> Dict[str, torch.Tensor]:
    """Compute Free Transformer loss with VAE components.
    
    This function computes the total loss for the Free Transformer,
    combining reconstruction loss and KL divergence with free bits
    regularization.
    
    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size).
        z_logits: Latent variable logits of shape (batch_size, latent_dim).
        targets: Target token IDs of shape (batch_size, seq_len).
        latent_dim: Dimension of the latent space.
        kl_weight: Weight for KL divergence loss. Defaults to 0.1.
        free_bits: Free bits threshold for KL regularization. Defaults to 0.5.
    
    Returns:
        Dictionary containing:
            - total_loss: Combined reconstruction and KL loss
            - recon_loss: Cross-entropy reconstruction loss
            - kl_loss: KL divergence loss with free bits
    
    Raises:
        ValueError: If input tensors have incompatible shapes.
        
    Example:
        >>> logits = torch.randn(2, 10, 1000)
        >>> z_logits = torch.randn(2, 16)
        >>> targets = torch.randint(0, 1000, (2, 10))
        >>> loss_dict = free_transformer_loss(logits, z_logits, targets, 16)
        >>> print(f"Total loss: {loss_dict['total_loss']:.4f}")
    """
    # Implementation here
    pass
```

### Class Documentation

```python
class FreeTransformer(nn.Module):
    """Free Transformer with explicit latent planning.
    
    The Free Transformer extends standard autoregressive Transformers
    with a latent planning mechanism. It first creates an abstract plan
    for the entire sequence, then generates tokens conditioned on that plan.
    
    Attributes:
        config: Model configuration containing architecture parameters.
        token_embedding: Token embedding layer.
        encoder: Non-causal encoder for latent plan generation.
        decoder_blocks: List of transformer decoder blocks.
        
    Example:
        >>> config = ModelConfig(vocab_size=1000, hidden_dim=512)
        >>> model = FreeTransformer(config)
        >>> tokens = torch.randint(0, 1000, (2, 128))
        >>> logits, z_logits = model(tokens, mode='training')
    """
    
    def __init__(self, config: ModelConfig) -> None:
        """Initialize Free Transformer.
        
        Args:
            config: Model configuration with architecture parameters.
        """
        super().__init__()
        # Implementation here
```

## Code Organization

### Module Structure

```python
"""Module for Free Transformer model implementation.

This module contains the main Free Transformer model class and related
components for latent planning and conditional generation.
"""

# Standard library imports
import math
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
from .config import ModelConfig
from .encoder import NonCausalEncoder
from .latent import BinaryMapper, PlanInjection
from .losses import free_transformer_loss

# Module-level constants
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 40

# Public API
__all__ = [
    'FreeTransformer',
    'TransformerBlock',
    'RMSNorm',
]
```

### Function Organization

```python
class FreeTransformer(nn.Module):
    """Free Transformer implementation."""
    
    # Public methods first
    def __init__(self, config: ModelConfig) -> None:
        """Initialize model."""
        pass
    
    def forward(self, tokens: torch.Tensor, mode: str = 'training') -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass."""
        pass
    
    def generate(self, prompt: torch.Tensor, max_new_tokens: int = 100, **kwargs) -> torch.Tensor:
        """Generate text."""
        pass
    
    # Private methods last
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights."""
        pass
    
    def _compute_attention_mask(self, seq_len: int) -> torch.Tensor:
        """Compute causal attention mask."""
        pass
```

## Error Handling

### Exception Handling

```python
def validate_config(config: ModelConfig) -> None:
    """Validate model configuration.
    
    Args:
        config: Model configuration to validate.
        
    Raises:
        ValueError: If configuration is invalid.
    """
    if config.hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {config.hidden_dim}")
    
    if config.hidden_dim % config.num_heads != 0:
        raise ValueError(
            f"hidden_dim ({config.hidden_dim}) must be divisible by "
            f"num_heads ({config.num_heads})"
        )
    
    if config.latent_dim <= 0:
        raise ValueError(f"latent_dim must be positive, got {config.latent_dim}")

def safe_divide(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Safely divide tensors with epsilon for numerical stability.
    
    Args:
        a: Numerator tensor.
        b: Denominator tensor.
        eps: Small epsilon to prevent division by zero.
        
    Returns:
        Result of a / (b + eps).
    """
    return a / (b + eps)
```

### Logging

```python
import logging

logger = logging.getLogger(__name__)

def train_model(model: FreeTransformer, dataloader: DataLoader) -> None:
    """Train model with proper logging."""
    logger.info("Starting training")
    
    for epoch in range(num_epochs):
        logger.debug(f"Starting epoch {epoch}")
        
        try:
            epoch_loss = train_epoch(model, dataloader)
            logger.info(f"Epoch {epoch}: loss = {epoch_loss:.4f}")
        except Exception as e:
            logger.error(f"Training failed at epoch {epoch}: {e}")
            raise
    
    logger.info("Training completed successfully")
```

## Performance Guidelines

### Memory Efficiency

```python
# Use in-place operations when possible
def apply_activation_inplace(x: torch.Tensor) -> torch.Tensor:
    """Apply activation in-place to save memory."""
    return F.relu_(x)  # In-place ReLU

# Use context managers for temporary computations
def compute_attention_weights(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Compute attention weights efficiently."""
    with torch.cuda.amp.autocast():  # Mixed precision
        scores = torch.matmul(query, key.transpose(-2, -1))
        return F.softmax(scores, dim=-1)

# Clear intermediate variables
def forward_with_cleanup(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass with memory cleanup."""
    intermediate = self.layer1(x)
    x = self.layer2(intermediate)
    del intermediate  # Free memory
    return x
```

### Computational Efficiency

```python
# Vectorize operations
def compute_distances_vectorized(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute pairwise distances efficiently."""
    # Vectorized computation
    return torch.cdist(embeddings, embeddings, p=2)

# Use appropriate data types
def mixed_precision_forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass with mixed precision."""
    with torch.cuda.amp.autocast():
        return self.model(x.half())  # Use half precision
```

## Testing Quality

### Test Organization

```python
# tests/unit/test_model.py
class TestFreeTransformer:
    """Test suite for FreeTransformer class."""
    
    @pytest.fixture
    def config(self) -> ModelConfig:
        """Standard test configuration."""
        return ModelConfig(
            vocab_size=1000,
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            latent_dim=8
        )
    
    def test_initialization(self, config: ModelConfig) -> None:
        """Test model initializes correctly."""
        model = FreeTransformer(config)
        assert isinstance(model, FreeTransformer)
        assert model.config == config
    
    def test_forward_shapes(self, config: ModelConfig) -> None:
        """Test forward pass produces correct shapes."""
        model = FreeTransformer(config)
        tokens = torch.randint(0, config.vocab_size, (2, 16))
        
        logits, z_logits = model(tokens, mode='training')
        
        assert logits.shape == (2, 16, config.vocab_size)
        assert z_logits.shape == (2, config.latent_dim)
```

## Continuous Integration

### Quality Checks in CI

```yaml
# .github/workflows/quality.yml
name: Code Quality

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.12
    
    - name: Install dependencies
      run: |
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.cargo/env
        uv venv
        source .venv/bin/activate
        uv pip install -e ".[dev]"
    
    - name: Check formatting
      run: |
        source .venv/bin/activate
        black --check src/ tests/ examples/
        isort --check-only src/ tests/ examples/
    
    - name: Lint code
      run: |
        source .venv/bin/activate
        ruff check src/ tests/ examples/
    
    - name: Type check
      run: |
        source .venv/bin/activate
        mypy src/
    
    - name: Run tests
      run: |
        source .venv/bin/activate
        pytest tests/ --cov=src/free_transformer
```

## Best Practices Summary

1. **Formatting**: Use Black and isort for consistent code style
2. **Linting**: Fix all Ruff warnings and errors
3. **Type Hints**: Add type annotations to all public functions
4. **Documentation**: Write clear docstrings with examples
5. **Error Handling**: Use appropriate exceptions and logging
6. **Testing**: Maintain high test coverage with quality tests
7. **Performance**: Consider memory and computational efficiency
8. **Organization**: Structure code logically with clear imports

## Next Steps

- **[Testing Guide](testing.md)**: Comprehensive testing strategies
- **[Contributing](contributing.md)**: How to contribute to the project
- **[Training Guide](../training/guide.md)**: Training best practices