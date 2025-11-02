# Basic Usage Examples

This page provides practical examples for using the Free Transformer in various scenarios.

## Model Creation and Basic Usage

### Creating a Model

```python
import torch
from free_transformer import FreeTransformer, ModelConfig

# Create configuration
config = ModelConfig(
    vocab_size=50000,
    hidden_dim=512,
    num_layers=12,
    num_heads=8,
    latent_dim=32,
    max_seq_len=1024
)

# Initialize model
model = FreeTransformer(config)
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
```

### Training Mode

```python
# Prepare training data
batch_size, seq_len = 4, 128
tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))

# Forward pass in training mode
model.train()
logits, z_logits = model(tokens, mode='training')

print(f"Output logits shape: {logits.shape}")  # [4, 128, 50000]
print(f"Latent logits shape: {z_logits.shape}")  # [4, 32]
```

### Inference Mode

```python
# Prepare prompt
prompt = torch.randint(0, config.vocab_size, (1, 20))

# Generate text
model.eval()
with torch.no_grad():
    generated = model.generate(
        prompt,
        max_new_tokens=100,
        temperature=0.8,
        top_k=40,
        do_sample=True
    )

print(f"Generated sequence length: {generated.shape[1]}")  # 120 (20 + 100)
```

## Text Generation Examples

### Basic Generation

```python
def generate_text(model, tokenizer, prompt_text, max_length=100):
    """Generate text from a prompt string."""
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt_text, return_tensors='pt')
    
    # Generate
    model.eval()
    with torch.no_grad():
        generated_tokens = model.generate(
            prompt_tokens,
            max_new_tokens=max_length,
            temperature=0.8,
            top_k=40,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = "The future of artificial intelligence"
generated = generate_text(model, tokenizer, prompt)
print(generated)
```

### Controlled Generation with Different Plans

```python
def generate_with_different_plans(model, prompt, num_plans=5):
    """Generate multiple texts with different latent plans."""
    generations = []
    
    model.eval()
    with torch.no_grad():
        for i in range(num_plans):
            # Each generation will sample a different latent plan
            generated = model.generate(
                prompt,
                max_new_tokens=50,
                temperature=0.8,
                top_k=40
            )
            generations.append(generated)
    
    return generations

# Example usage
prompt = torch.randint(0, config.vocab_size, (1, 10))
different_generations = generate_with_different_plans(model, prompt)
print(f"Generated {len(different_generations)} different continuations")
```

### Temperature and Sampling Control

```python
def compare_sampling_strategies(model, prompt):
    """Compare different sampling strategies."""
    strategies = [
        {"temperature": 0.1, "top_k": None, "name": "Low temperature"},
        {"temperature": 0.8, "top_k": 40, "name": "Balanced"},
        {"temperature": 1.2, "top_k": 100, "name": "High temperature"},
        {"temperature": 0.0, "top_k": None, "name": "Greedy (deterministic)"}
    ]
    
    results = {}
    model.eval()
    
    for strategy in strategies:
        with torch.no_grad():
            generated = model.generate(
                prompt,
                max_new_tokens=30,
                temperature=strategy["temperature"],
                top_k=strategy["top_k"],
                do_sample=strategy["temperature"] > 0
            )
        results[strategy["name"]] = generated
    
    return results

# Example usage
prompt = torch.randint(0, config.vocab_size, (1, 15))
sampling_results = compare_sampling_strategies(model, prompt)
```

## Training Examples

### Simple Training Loop

```python
import torch.nn.functional as F
from free_transformer.losses import free_transformer_loss

def train_epoch(model, dataloader, optimizer, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
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
        
        loss = loss_dict['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if num_batches % 100 == 0:
            print(f"Batch {num_batches}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches

# Example usage
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
avg_loss = train_epoch(model, train_dataloader, optimizer, config)
print(f"Average loss: {avg_loss:.4f}")
```

### Training with Validation

```python
def train_with_validation(model, train_loader, val_loader, num_epochs=5):
    """Training loop with validation."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, config)
        
        # Validation
        val_loss = evaluate_model(model, val_loader)
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print("Saved new best model!")
        
        print("-" * 50)

def evaluate_model(model, dataloader):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            tokens = batch['input_ids']
            logits, z_logits = model(tokens, mode='training')
            
            loss_dict = free_transformer_loss(
                logits=logits,
                z_logits=z_logits,
                targets=tokens,
                latent_dim=config.latent_dim,
                kl_weight=0.1,
                free_bits=0.5
            )
            
            total_loss += loss_dict['total_loss'].item()
            num_batches += 1
    
    return total_loss / num_batches
```

## Model Comparison

### Compare with Baseline

```python
from free_transformer import BaselineTransformer

def compare_models(free_model, baseline_model, test_data):
    """Compare Free Transformer with baseline."""
    results = {}
    
    for name, model in [("Free Transformer", free_model), ("Baseline", baseline_model)]:
        model.eval()
        total_loss = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in test_data:
                tokens = batch['input_ids']
                
                if name == "Free Transformer":
                    logits, _ = model(tokens, mode='training')
                else:
                    logits = model(tokens)
                
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    tokens.view(-1),
                    reduction='sum'
                )
                
                total_loss += loss.item()
                num_samples += tokens.numel()
        
        perplexity = torch.exp(torch.tensor(total_loss / num_samples))
        results[name] = {
            'loss': total_loss / num_samples,
            'perplexity': perplexity.item()
        }
    
    return results

# Example usage
baseline_config = ModelConfig(
    vocab_size=config.vocab_size,
    hidden_dim=config.hidden_dim,
    num_layers=config.num_layers,
    num_heads=config.num_heads,
    max_seq_len=config.max_seq_len
)
baseline_model = BaselineTransformer(baseline_config)

comparison_results = compare_models(model, baseline_model, test_dataloader)
print("Model Comparison Results:")
for model_name, metrics in comparison_results.items():
    print(f"{model_name}: Perplexity = {metrics['perplexity']:.2f}")
```

## Utility Functions

### Model Information

```python
def model_info(model):
    """Print detailed model information."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.1f} MB (float32)")
    
    # Layer breakdown
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            params = module.weight.numel()
            if hasattr(module, 'bias') and module.bias is not None:
                params += module.bias.numel()
            print(f"  {name}: {params:,} parameters")

model_info(model)
```

### Save and Load Models

```python
def save_model(model, config, path):
    """Save model and configuration."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'model_class': model.__class__.__name__
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")

def load_model(path):
    """Load model and configuration."""
    checkpoint = torch.load(path, map_location='cpu')
    
    # Recreate config
    config = ModelConfig(**checkpoint['config'])
    
    # Recreate model
    if checkpoint['model_class'] == 'FreeTransformer':
        model = FreeTransformer(config)
    else:
        model = BaselineTransformer(config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config

# Example usage
save_model(model, config, 'my_model.pt')
loaded_model, loaded_config = load_model('my_model.pt')
```

## Next Steps

- **[Custom Training](custom-training.md)**: Advanced training techniques
- **[Evaluation](evaluation.md)**: Comprehensive model evaluation
- **[API Reference](../api/model.md)**: Complete API documentation