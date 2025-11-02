# Testing Guide

This guide covers testing strategies and practices for the Free Transformer project.

## Test Structure

The test suite is organized into three main categories:

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_model.py       # Model architecture tests
│   ├── test_encoder.py     # Encoder component tests
│   ├── test_latent.py      # Latent variable tests
│   ├── test_losses.py      # Loss function tests
│   └── test_config.py      # Configuration tests
├── integration/            # Integration tests
│   ├── test_training.py    # Training pipeline tests
│   ├── test_generation.py  # Generation tests
│   └── test_data.py        # Data loading tests
└── test_comparison.py      # Model comparison tests
```

## Running Tests

### Basic Test Commands

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-comparison

# Run fast tests (no coverage)
make test-fast

# Run specific test file
pytest tests/unit/test_model.py -v

# Run specific test function
pytest tests/unit/test_model.py::TestFreeTransformer::test_forward_training_mode -v
```

### Test Configuration

Tests use pytest with the following configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "--cov=src/free_transformer --cov-report=html --cov-report=term"
```

## Unit Tests

### Model Architecture Tests

```python
# tests/unit/test_model.py
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
            latent_dim=8,
            max_seq_len=256
        )
    
    @pytest.fixture
    def model(self, config):
        return FreeTransformer(config)
    
    def test_model_initialization(self, model, config):
        """Test model initializes correctly."""
        assert model.config.vocab_size == config.vocab_size
        assert model.config.hidden_dim == config.hidden_dim
        assert model.config.latent_dim == config.latent_dim
        
        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
    
    def test_forward_training_mode(self, model, config):
        """Test forward pass in training mode."""
        batch_size, seq_len = 2, 32
        tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        logits, z_logits = model(tokens, mode='training')
        
        # Check output shapes
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert z_logits.shape == (batch_size, config.latent_dim)
        
        # Check output types
        assert isinstance(logits, torch.Tensor)
        assert isinstance(z_logits, torch.Tensor)
        
        # Check gradients can flow
        loss = logits.sum() + z_logits.sum()
        loss.backward()
        
        # Check some parameters have gradients
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients
    
    def test_forward_inference_mode(self, model, config):
        """Test forward pass in inference mode."""
        batch_size, seq_len = 2, 32
        tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        model.eval()
        with torch.no_grad():
            logits = model(tokens, mode='inference')
        
        # Check output shape
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert isinstance(logits, torch.Tensor)
    
    def test_generation(self, model, config):
        """Test text generation."""
        prompt = torch.randint(0, config.vocab_size, (1, 10))
        max_new_tokens = 20
        
        model.eval()
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=max_new_tokens)
        
        # Check output shape
        expected_length = prompt.size(1) + max_new_tokens
        assert generated.shape == (1, expected_length)
        
        # Check tokens are valid
        assert torch.all(generated >= 0)
        assert torch.all(generated < config.vocab_size)
        
        # Check prompt is preserved
        assert torch.equal(generated[:, :prompt.size(1)], prompt)
    
    def test_different_batch_sizes(self, model, config):
        """Test model works with different batch sizes."""
        for batch_size in [1, 2, 4, 8]:
            tokens = torch.randint(0, config.vocab_size, (batch_size, 16))
            
            logits, z_logits = model(tokens, mode='training')
            
            assert logits.shape[0] == batch_size
            assert z_logits.shape[0] == batch_size
    
    def test_different_sequence_lengths(self, model, config):
        """Test model works with different sequence lengths."""
        batch_size = 2
        
        for seq_len in [8, 16, 32, 64]:
            if seq_len <= config.max_seq_len:
                tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
                
                logits, z_logits = model(tokens, mode='training')
                
                assert logits.shape[1] == seq_len
                assert z_logits.shape == (batch_size, config.latent_dim)
    
    def test_model_device_compatibility(self, model, config):
        """Test model works on different devices."""
        tokens = torch.randint(0, config.vocab_size, (2, 16))
        
        # CPU test
        model_cpu = model.to('cpu')
        tokens_cpu = tokens.to('cpu')
        
        logits_cpu, z_logits_cpu = model_cpu(tokens_cpu, mode='training')
        assert logits_cpu.device.type == 'cpu'
        assert z_logits_cpu.device.type == 'cpu'
        
        # GPU test (if available)
        if torch.cuda.is_available():
            model_gpu = model.to('cuda')
            tokens_gpu = tokens.to('cuda')
            
            logits_gpu, z_logits_gpu = model_gpu(tokens_gpu, mode='training')
            assert logits_gpu.device.type == 'cuda'
            assert z_logits_gpu.device.type == 'cuda'
```

### Loss Function Tests

```python
# tests/unit/test_losses.py
import pytest
import torch
from free_transformer.losses import free_transformer_loss

class TestLosses:
    @pytest.fixture
    def sample_data(self):
        batch_size, seq_len, vocab_size, latent_dim = 2, 16, 1000, 8
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        z_logits = torch.randn(batch_size, latent_dim)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        return logits, z_logits, targets, latent_dim
    
    def test_free_transformer_loss_basic(self, sample_data):
        """Test basic loss computation."""
        logits, z_logits, targets, latent_dim = sample_data
        
        loss_dict = free_transformer_loss(
            logits=logits,
            z_logits=z_logits,
            targets=targets,
            latent_dim=latent_dim,
            kl_weight=0.1,
            free_bits=0.5
        )
        
        # Check all expected keys are present
        expected_keys = ['total_loss', 'recon_loss', 'kl_loss']
        for key in expected_keys:
            assert key in loss_dict
            assert isinstance(loss_dict[key], torch.Tensor)
            assert loss_dict[key].dim() == 0  # Scalar
        
        # Check loss values are reasonable
        assert loss_dict['total_loss'] > 0
        assert loss_dict['recon_loss'] > 0
        assert loss_dict['kl_loss'] >= 0  # Can be zero due to free bits
    
    def test_loss_gradients(self, sample_data):
        """Test that loss enables gradient computation."""
        logits, z_logits, targets, latent_dim = sample_data
        
        # Make tensors require gradients
        logits.requires_grad_(True)
        z_logits.requires_grad_(True)
        
        loss_dict = free_transformer_loss(
            logits=logits,
            z_logits=z_logits,
            targets=targets,
            latent_dim=latent_dim,
            kl_weight=0.1,
            free_bits=0.5
        )
        
        # Backward pass
        loss_dict['total_loss'].backward()
        
        # Check gradients exist
        assert logits.grad is not None
        assert z_logits.grad is not None
        assert not torch.all(logits.grad == 0)
        assert not torch.all(z_logits.grad == 0)
    
    def test_kl_weight_effect(self, sample_data):
        """Test that KL weight affects total loss."""
        logits, z_logits, targets, latent_dim = sample_data
        
        # Compute loss with different KL weights
        loss_low = free_transformer_loss(
            logits=logits, z_logits=z_logits, targets=targets,
            latent_dim=latent_dim, kl_weight=0.01, free_bits=0.0
        )
        
        loss_high = free_transformer_loss(
            logits=logits, z_logits=z_logits, targets=targets,
            latent_dim=latent_dim, kl_weight=1.0, free_bits=0.0
        )
        
        # Higher KL weight should increase total loss (if KL > 0)
        if loss_low['kl_loss'] > 0:
            assert loss_high['total_loss'] > loss_low['total_loss']
    
    def test_free_bits_effect(self, sample_data):
        """Test that free bits affects KL loss."""
        logits, z_logits, targets, latent_dim = sample_data
        
        # Compute loss with different free bits
        loss_no_free = free_transformer_loss(
            logits=logits, z_logits=z_logits, targets=targets,
            latent_dim=latent_dim, kl_weight=0.1, free_bits=0.0
        )
        
        loss_with_free = free_transformer_loss(
            logits=logits, z_logits=z_logits, targets=targets,
            latent_dim=latent_dim, kl_weight=0.1, free_bits=1.0
        )
        
        # Free bits should increase KL loss (clamping effect)
        assert loss_with_free['kl_loss'] >= loss_no_free['kl_loss']
```

## Integration Tests

### Training Pipeline Tests

```python
# tests/integration/test_training.py
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from free_transformer import FreeTransformer, ModelConfig
from free_transformer.losses import free_transformer_loss

class TestTrainingPipeline:
    @pytest.fixture
    def config(self):
        return ModelConfig(
            vocab_size=100,
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
            latent_dim=4,
            max_seq_len=32
        )
    
    @pytest.fixture
    def model(self, config):
        return FreeTransformer(config)
    
    @pytest.fixture
    def dataloader(self, config):
        # Create synthetic data
        num_samples = 50
        seq_len = 16
        
        data = torch.randint(0, config.vocab_size, (num_samples, seq_len))
        dataset = TensorDataset(data)
        
        return DataLoader(dataset, batch_size=8, shuffle=True)
    
    def test_training_step(self, model, config, dataloader):
        """Test a single training step."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        model.train()
        
        for batch in dataloader:
            tokens = batch[0]
            
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
            
            loss = loss_dict['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Check loss is finite
            assert torch.isfinite(loss)
            
            # Only test one batch
            break
    
    def test_training_convergence(self, model, config, dataloader):
        """Test that model can overfit to small dataset."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        model.train()
        
        initial_loss = None
        final_loss = None
        
        # Train for several epochs
        for epoch in range(10):
            epoch_losses = []
            
            for batch in dataloader:
                tokens = batch[0]
                
                logits, z_logits = model(tokens, mode='training')
                
                loss_dict = free_transformer_loss(
                    logits=logits,
                    z_logits=z_logits,
                    targets=tokens,
                    latent_dim=config.latent_dim,
                    kl_weight=0.01,  # Low KL weight for easier overfitting
                    free_bits=0.1
                )
                
                loss = loss_dict['total_loss']
                epoch_losses.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            
            if epoch == 0:
                initial_loss = avg_loss
            if epoch == 9:
                final_loss = avg_loss
        
        # Model should overfit (loss should decrease)
        assert final_loss < initial_loss
        assert final_loss < 5.0  # Reasonable threshold
    
    def test_evaluation_mode(self, model, config, dataloader):
        """Test evaluation mode doesn't update parameters."""
        # Get initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                tokens = batch[0]
                
                # Forward pass in training mode (for loss computation)
                logits, z_logits = model(tokens, mode='training')
                
                # Compute loss (but don't backward)
                loss_dict = free_transformer_loss(
                    logits=logits,
                    z_logits=z_logits,
                    targets=tokens,
                    latent_dim=config.latent_dim,
                    kl_weight=0.1,
                    free_bits=0.5
                )
                
                # Only test one batch
                break
        
        # Check parameters haven't changed
        for name, param in model.named_parameters():
            assert torch.equal(param, initial_params[name])
```

### Generation Tests

```python
# tests/integration/test_generation.py
import pytest
import torch
from free_transformer import FreeTransformer, ModelConfig

class TestGeneration:
    @pytest.fixture
    def config(self):
        return ModelConfig(
            vocab_size=100,
            hidden_dim=64,
            num_layers=2,
            num_heads=2,
            latent_dim=4,
            max_seq_len=64
        )
    
    @pytest.fixture
    def model(self, config):
        model = FreeTransformer(config)
        model.eval()
        return model
    
    def test_basic_generation(self, model, config):
        """Test basic generation functionality."""
        prompt = torch.randint(0, config.vocab_size, (1, 5))
        max_new_tokens = 10
        
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=max_new_tokens)
        
        # Check output properties
        assert generated.shape == (1, 5 + max_new_tokens)
        assert torch.all(generated >= 0)
        assert torch.all(generated < config.vocab_size)
        assert torch.equal(generated[:, :5], prompt)
    
    def test_generation_determinism(self, model, config):
        """Test generation determinism with same seed."""
        prompt = torch.randint(0, config.vocab_size, (1, 5))
        
        # Generate with same seed
        torch.manual_seed(42)
        gen1 = model.generate(prompt, max_new_tokens=10, temperature=0.0)
        
        torch.manual_seed(42)
        gen2 = model.generate(prompt, max_new_tokens=10, temperature=0.0)
        
        # Should be identical with temperature=0
        assert torch.equal(gen1, gen2)
    
    def test_generation_diversity(self, model, config):
        """Test generation produces diverse outputs."""
        prompt = torch.randint(0, config.vocab_size, (1, 5))
        
        generations = []
        for _ in range(5):
            with torch.no_grad():
                gen = model.generate(
                    prompt, 
                    max_new_tokens=20, 
                    temperature=1.0,
                    do_sample=True
                )
            generations.append(gen)
        
        # Check that not all generations are identical
        all_same = all(torch.equal(generations[0], gen) for gen in generations[1:])
        assert not all_same, "All generations are identical - no diversity"
    
    def test_batch_generation(self, model, config):
        """Test generation with batch input."""
        batch_size = 3
        prompt = torch.randint(0, config.vocab_size, (batch_size, 5))
        
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=10)
        
        assert generated.shape == (batch_size, 15)
        
        # Check each sequence in batch
        for i in range(batch_size):
            assert torch.equal(generated[i, :5], prompt[i])
    
    def test_generation_parameters(self, model, config):
        """Test different generation parameters."""
        prompt = torch.randint(0, config.vocab_size, (1, 5))
        
        # Test different temperatures
        for temp in [0.1, 0.5, 1.0, 1.5]:
            gen = model.generate(prompt, max_new_tokens=10, temperature=temp)
            assert gen.shape == (1, 15)
        
        # Test different top_k values
        for top_k in [1, 5, 10, 20]:
            gen = model.generate(prompt, max_new_tokens=10, top_k=top_k)
            assert gen.shape == (1, 15)
        
        # Test different top_p values
        for top_p in [0.1, 0.5, 0.9, 1.0]:
            gen = model.generate(prompt, max_new_tokens=10, top_p=top_p)
            assert gen.shape == (1, 15)
```

## Performance Tests

### Memory and Speed Tests

```python
# tests/test_performance.py
import pytest
import torch
import time
import psutil
import os
from free_transformer import FreeTransformer, ModelConfig

class TestPerformance:
    @pytest.fixture
    def small_config(self):
        return ModelConfig(
            vocab_size=1000,
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            latent_dim=8
        )
    
    @pytest.fixture
    def medium_config(self):
        return ModelConfig(
            vocab_size=10000,
            hidden_dim=512,
            num_layers=12,
            num_heads=8,
            latent_dim=32
        )
    
    def test_memory_usage(self, small_config):
        """Test memory usage is reasonable."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        model = FreeTransformer(small_config)
        
        after_model_memory = process.memory_info().rss / 1024 / 1024  # MB
        model_memory = after_model_memory - initial_memory
        
        # Model should use reasonable amount of memory (less than 500MB for small model)
        assert model_memory < 500, f"Model uses too much memory: {model_memory:.1f}MB"
    
    def test_forward_speed(self, small_config):
        """Test forward pass speed."""
        model = FreeTransformer(small_config)
        model.eval()
        
        batch_size, seq_len = 4, 128
        tokens = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                model(tokens, mode='training')
        
        # Time forward passes
        num_runs = 20
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                model(tokens, mode='training')
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        
        # Should be reasonably fast (less than 100ms for small model)
        assert avg_time < 0.1, f"Forward pass too slow: {avg_time:.3f}s"
    
    def test_generation_speed(self, small_config):
        """Test generation speed."""
        model = FreeTransformer(small_config)
        model.eval()
        
        prompt = torch.randint(0, small_config.vocab_size, (1, 10))
        max_new_tokens = 50
        
        # Warmup
        with torch.no_grad():
            model.generate(prompt, max_new_tokens=10)
        
        # Time generation
        start_time = time.time()
        
        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=max_new_tokens)
        
        end_time = time.time()
        generation_time = end_time - start_time
        tokens_per_second = max_new_tokens / generation_time
        
        # Should generate at reasonable speed (at least 10 tokens/sec)
        assert tokens_per_second > 10, f"Generation too slow: {tokens_per_second:.1f} tokens/sec"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_utilization(self, small_config):
        """Test GPU utilization."""
        device = torch.device('cuda')
        model = FreeTransformer(small_config).to(device)
        
        batch_size, seq_len = 8, 256
        tokens = torch.randint(0, small_config.vocab_size, (batch_size, seq_len)).to(device)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device)
        
        # Forward pass
        with torch.no_grad():
            logits, z_logits = model(tokens, mode='training')
        
        peak_memory = torch.cuda.max_memory_allocated(device)
        memory_used = (peak_memory - initial_memory) / 1024 / 1024  # MB
        
        # Should use reasonable GPU memory
        assert memory_used < 1000, f"Uses too much GPU memory: {memory_used:.1f}MB"
```

## Test Utilities

### Common Test Fixtures

```python
# tests/conftest.py
import pytest
import torch
from free_transformer import FreeTransformer, ModelConfig

@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def small_config():
    """Small model configuration for testing."""
    return ModelConfig(
        vocab_size=100,
        hidden_dim=64,
        num_layers=2,
        num_heads=2,
        latent_dim=4,
        max_seq_len=32
    )

@pytest.fixture
def medium_config():
    """Medium model configuration for testing."""
    return ModelConfig(
        vocab_size=1000,
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        latent_dim=16,
        max_seq_len=128
    )

@pytest.fixture
def sample_tokens(small_config):
    """Sample token sequences for testing."""
    return torch.randint(0, small_config.vocab_size, (4, 16))

@pytest.fixture
def trained_model(small_config, sample_tokens):
    """A model that has been trained for a few steps."""
    model = FreeTransformer(small_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    model.train()
    
    # Train for a few steps
    for _ in range(10):
        logits, z_logits = model(sample_tokens, mode='training')
        
        from free_transformer.losses import free_transformer_loss
        loss_dict = free_transformer_loss(
            logits=logits,
            z_logits=z_logits,
            targets=sample_tokens,
            latent_dim=small_config.latent_dim,
            kl_weight=0.1,
            free_bits=0.5
        )
        
        loss = loss_dict['total_loss']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    return model
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/test.yml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install UV
      run: |
        curl -LsSf https://astral.sh/uv/install.sh | sh
        echo "$HOME/.cargo/bin" >> $GITHUB_PATH
    
    - name: Install dependencies
      run: |
        uv venv --python ${{ matrix.python-version }}
        source .venv/bin/activate
        uv pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        source .venv/bin/activate
        make test
    
    - name: Run quality checks
      run: |
        source .venv/bin/activate
        make quality
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

## Best Practices

### Writing Good Tests

1. **Test one thing at a time**: Each test should focus on a single behavior
2. **Use descriptive names**: Test names should clearly describe what they test
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
4. **Use fixtures**: Share common setup code using pytest fixtures
5. **Test edge cases**: Include tests for boundary conditions and error cases
6. **Mock external dependencies**: Use mocks for external services or complex dependencies

### Test Coverage

Aim for high test coverage but focus on quality over quantity:

```bash
# Generate coverage report
pytest --cov=src/free_transformer --cov-report=html

# View coverage report
open htmlcov/index.html
```

Target coverage levels:
- **Unit tests**: 90%+ coverage of core components
- **Integration tests**: Cover main user workflows
- **End-to-end tests**: Test complete pipelines

### Performance Testing

Include performance tests to catch regressions:

```python
@pytest.mark.performance
def test_training_speed():
    """Test training speed doesn't regress."""
    # Implementation here
    pass

# Run performance tests separately
pytest -m performance
```

## Troubleshooting Tests

### Common Issues

**Tests fail on CI but pass locally**
- Check Python version compatibility
- Verify all dependencies are installed
- Check for platform-specific issues

**Flaky tests**
- Use fixed random seeds
- Mock time-dependent operations
- Increase timeouts for slow operations

**Memory issues in tests**
- Use smaller models for testing
- Clear GPU cache between tests
- Use pytest-xdist for parallel execution

**Slow test suite**
- Profile tests to find bottlenecks
- Use pytest-benchmark for performance tests
- Consider test parallelization

## Next Steps

- **[Code Quality](quality.md)**: Code quality tools and practices
- **[Contributing](contributing.md)**: How to contribute to the project
- **[Training Guide](../training/guide.md)**: Training best practices