# Contributing to Free Transformer

We welcome contributions to the Free Transformer project! This guide will help you get started with contributing code, documentation, or other improvements.

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/free-transformer.git
cd free-transformer

# Add upstream remote
git remote add upstream https://github.com/udapy/free-transformer.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install development dependencies
uv pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

### 3. Verify Setup

```bash
# Run tests to ensure everything works
make test

# Run quality checks
make quality

# Generate synthetic data and run demo
make demo
```

## Development Workflow

### 1. Create a Feature Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow these guidelines when making changes:

- **Code Style**: Follow PEP 8 and use the provided formatters
- **Type Hints**: Add type hints to all new functions
- **Documentation**: Update docstrings and documentation
- **Tests**: Add tests for new functionality

### 3. Test Your Changes

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_model.py -v

# Run quality checks
make quality

# Test with different configurations
python examples/train_free.py --config configs/small.yaml
```

### 4. Commit and Push

```bash
# Stage your changes
git add .

# Commit with descriptive message
git commit -m "feat: add support for custom attention patterns"

# Push to your fork
git push origin feature/your-feature-name
```

### 5. Create Pull Request

1. Go to GitHub and create a pull request
2. Fill out the PR template
3. Link any related issues
4. Wait for review and address feedback

## Code Style Guidelines

### Python Code Style

We use several tools to maintain code quality:

```bash
# Format code
black src/ tests/ examples/
isort src/ tests/ examples/

# Lint code
flake8 src/ tests/ examples/
ruff check src/ tests/ examples/

# Type checking
mypy src/
```

### Code Organization

```
src/free_transformer/
├── __init__.py          # Public API exports
├── model.py             # Main model classes
├── baseline.py          # Baseline Transformer
├── encoder.py           # Non-causal encoder
├── latent.py           # Latent variable components
├── injection.py        # Plan injection mechanisms
├── losses.py           # Loss functions
├── config.py           # Configuration classes
├── train_utils.py      # Training utilities
└── synthetic_data.py   # Data generation
```

### Naming Conventions

- **Classes**: PascalCase (`FreeTransformer`, `ModelConfig`)
- **Functions/Variables**: snake_case (`compute_loss`, `hidden_dim`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_VOCAB_SIZE`)
- **Private methods**: Leading underscore (`_compute_attention`)

## Testing Guidelines

### Test Structure

```
tests/
├── unit/                # Unit tests for individual components
│   ├── test_model.py
│   ├── test_encoder.py
│   └── test_losses.py
├── integration/         # Integration tests
│   ├── test_training.py
│   └── test_generation.py
└── test_comparison.py   # Model comparison tests
```

### Writing Tests

```python
import pytest
import torch
from free_transformer import FreeTransformer, ModelConfig

class TestFreeTransformer:
    @pytest.fixture
    def config(self):
        return ModelConfig(
            vocab_size=1000,
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            latent_dim=8
        )
    
    @pytest.fixture
    def model(self, config):
        return FreeTransformer(config)
    
    def test_forward_training_mode(self, model, config):
        batch_size, seq_len = 2, 32
        tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        logits, z_logits = model(tokens, mode='training')
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert z_logits.shape == (batch_size, config.latent_dim)
    
    def test_generation(self, model, config):
        prompt = torch.randint(0, config.vocab_size, (1, 10))
        
        generated = model.generate(prompt, max_new_tokens=20)
        
        assert generated.shape == (1, 30)  # 10 + 20
        assert torch.all(generated >= 0)
        assert torch.all(generated < config.vocab_size)
```

### Test Coverage

Aim for high test coverage:

```bash
# Run tests with coverage
pytest --cov=src/free_transformer --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Documentation Guidelines

### Docstring Format

Use Google-style docstrings:

```python
def compute_loss(logits: torch.Tensor, targets: torch.Tensor, 
                 config: ModelConfig) -> Dict[str, torch.Tensor]:
    """Compute the Free Transformer loss.
    
    Args:
        logits: Model output logits of shape (batch_size, seq_len, vocab_size).
        targets: Target token IDs of shape (batch_size, seq_len).
        config: Model configuration containing loss hyperparameters.
    
    Returns:
        Dictionary containing:
            - total_loss: Combined reconstruction and KL loss
            - recon_loss: Cross-entropy reconstruction loss
            - kl_loss: KL divergence regularization loss
    
    Raises:
        ValueError: If logits and targets have incompatible shapes.
    
    Example:
        >>> logits = torch.randn(2, 10, 1000)
        >>> targets = torch.randint(0, 1000, (2, 10))
        >>> loss_dict = compute_loss(logits, targets, config)
        >>> print(loss_dict['total_loss'])
    """
```

### Documentation Updates

When adding new features:

1. **Update API docs**: Add docstrings to new classes/functions
2. **Update guides**: Add examples to relevant guides
3. **Update README**: If it affects installation or basic usage
4. **Add examples**: Create example scripts if appropriate

## Types of Contributions

### 1. Bug Fixes

- **Small fixes**: Can be submitted directly
- **Large fixes**: Please open an issue first to discuss

Example bug fix PR:
```
Title: Fix gradient flow in binary mapper
Description: The Gumbel-Softmax implementation was not properly 
handling gradients in training mode. This PR fixes the issue by...
```

### 2. New Features

Please open an issue first to discuss new features:

- **Architecture improvements**: New attention mechanisms, injection strategies
- **Training enhancements**: New loss functions, optimization techniques
- **Utility functions**: Data processing, evaluation metrics
- **Performance optimizations**: Memory usage, speed improvements

### 3. Documentation

- **API documentation**: Improve docstrings and type hints
- **Guides and tutorials**: Add new examples or improve existing ones
- **Architecture explanations**: Help explain complex concepts
- **FAQ updates**: Add common questions and solutions

### 4. Tests

- **Unit tests**: Test individual components
- **Integration tests**: Test component interactions
- **Performance tests**: Benchmark improvements
- **Regression tests**: Prevent known issues from reoccurring

## Review Process

### What We Look For

1. **Correctness**: Does the code work as intended?
2. **Style**: Does it follow our coding standards?
3. **Tests**: Are there adequate tests?
4. **Documentation**: Is it properly documented?
5. **Performance**: Does it maintain or improve performance?

### Review Timeline

- **Small fixes**: Usually reviewed within 1-2 days
- **Medium features**: Usually reviewed within 3-5 days
- **Large features**: May take 1-2 weeks depending on complexity

### Addressing Feedback

- **Be responsive**: Address feedback promptly
- **Ask questions**: If feedback is unclear, ask for clarification
- **Make incremental changes**: Small, focused commits are easier to review
- **Update tests**: Ensure tests pass after addressing feedback

## Release Process

### Version Numbering

We follow semantic versioning (SemVer):

- **Major** (1.0.0): Breaking changes
- **Minor** (0.1.0): New features, backward compatible
- **Patch** (0.0.1): Bug fixes, backward compatible

### Release Checklist

Before releasing:

1. **Update version**: In `pyproject.toml` and `__init__.py`
2. **Update CHANGELOG**: Document all changes
3. **Run full test suite**: Ensure everything passes
4. **Update documentation**: Reflect any changes
5. **Create release notes**: Summarize key changes

## Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Pull Request Comments**: Code-specific discussions

### Mentorship

New contributors are welcome! If you're new to the project:

1. **Start small**: Look for "good first issue" labels
2. **Ask questions**: Don't hesitate to ask for help
3. **Read the code**: Familiarize yourself with the codebase
4. **Join discussions**: Participate in issue discussions

## Recognition

Contributors are recognized in several ways:

- **CONTRIBUTORS.md**: All contributors are listed
- **Release notes**: Significant contributions are highlighted
- **GitHub**: Contributions show up on your GitHub profile

## Code of Conduct

We are committed to providing a welcoming and inclusive environment:

- **Be respectful**: Treat all contributors with respect
- **Be constructive**: Provide helpful feedback
- **Be patient**: Remember that everyone is learning
- **Be inclusive**: Welcome contributors from all backgrounds

## Common Tasks

### Adding a New Model Component

1. **Create the module**: Add to `src/free_transformer/`
2. **Add tests**: Create corresponding test file
3. **Update exports**: Add to `__init__.py`
4. **Add documentation**: Include docstrings and examples
5. **Update configs**: Add configuration options if needed

### Adding a New Loss Function

1. **Implement in `losses.py`**: Follow existing patterns
2. **Add unit tests**: Test edge cases and gradients
3. **Update training scripts**: Show how to use it
4. **Document parameters**: Explain hyperparameters
5. **Add examples**: Show typical usage

### Improving Performance

1. **Profile first**: Identify actual bottlenecks
2. **Benchmark changes**: Measure improvements
3. **Maintain correctness**: Ensure outputs don't change
4. **Update tests**: Add performance regression tests
5. **Document changes**: Explain the optimization

Thank you for contributing to Free Transformer! 🚀