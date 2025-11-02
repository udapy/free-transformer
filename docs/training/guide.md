# Training Guide

This guide covers training Free Transformer models from basic setups to advanced distributed training.

## Basic Training

### Single GPU Training

```bash
# Train with default config
python examples/train_free.py --config configs/free_transformer.yaml

# Train with custom parameters
python examples/train_free.py \
  --config configs/free_transformer.yaml \
  --batch-size 16 \
  --learning-rate 1e-4 \
  --num-epochs 5
```

### Configuration Files

Training configurations are defined in YAML files:

```yaml
# configs/free_transformer.yaml
model:
  vocab_size: 50000
  hidden_dim: 512
  num_layers: 12
  num_heads: 8
  latent_dim: 32
  max_seq_len: 1024

training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 10
  warmup_steps: 1000
  weight_decay: 0.01
  
  # Free Transformer specific
  kl_weight: 0.1
  free_bits: 0.5
  
optimizer:
  type: "adamw"
  betas: [0.9, 0.95]
  eps: 1e-8

scheduler:
  type: "cosine"
  min_lr: 1e-6

data:
  dataset_name: "synthetic"
  max_seq_len: 512
  num_workers: 4
```

## Loss Components

The Free Transformer uses a composite loss function:

### Reconstruction Loss
Standard cross-entropy loss for token prediction:
```python
recon_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
```

### KL Divergence Loss
Regularizes the latent space:
```python
kl_loss = kl_divergence(z_logits, uniform_prior)
```

### Free Bits Regularization
Prevents posterior collapse:
```python
kl_loss = torch.clamp(kl_loss, min=free_bits)
```

### Total Loss
```python
total_loss = recon_loss + kl_weight * kl_loss
```

## Training Strategies

### 1. Curriculum Learning

Start with simpler tasks and gradually increase complexity:

```python
# Phase 1: Small sequences, high KL weight
config.max_seq_len = 128
config.kl_weight = 1.0

# Phase 2: Medium sequences, medium KL weight  
config.max_seq_len = 256
config.kl_weight = 0.5

# Phase 3: Full sequences, low KL weight
config.max_seq_len = 512
config.kl_weight = 0.1
```

### 2. KL Annealing

Gradually reduce KL weight during training:

```python
def get_kl_weight(step, total_steps, initial_weight=1.0, final_weight=0.1):
    progress = step / total_steps
    return initial_weight * (1 - progress) + final_weight * progress
```

### 3. Free Bits Scheduling

Adjust free bits threshold over time:

```python
def get_free_bits(step, total_steps, initial_bits=2.0, final_bits=0.5):
    progress = step / total_steps
    return initial_bits * (1 - progress) + final_bits * progress
```

## Advanced Training

### Mixed Precision Training

Enable automatic mixed precision for faster training:

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        logits, z_logits = model(batch['input_ids'], mode='training')
        loss = compute_loss(logits, z_logits, batch['input_ids'])
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Gradient Checkpointing

Reduce memory usage at the cost of computation:

```python
model = FreeTransformer(config)
model.gradient_checkpointing_enable()
```

### Learning Rate Scheduling

Use cosine annealing with warmup:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=1000,
    num_training_steps=total_steps
)
```

## Monitoring and Logging

### TensorBoard Logging

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/free_transformer')

# Log losses
writer.add_scalar('Loss/Reconstruction', recon_loss, step)
writer.add_scalar('Loss/KL', kl_loss, step)
writer.add_scalar('Loss/Total', total_loss, step)

# Log learning rate
writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], step)
```

### Weights & Biases Integration

```python
import wandb

wandb.init(project="free-transformer")
wandb.config.update(config)

# Log metrics
wandb.log({
    'loss/reconstruction': recon_loss,
    'loss/kl': kl_loss,
    'loss/total': total_loss,
    'learning_rate': lr
})
```

## Evaluation During Training

### Perplexity Calculation

```python
def calculate_perplexity(model, dataloader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            logits, _ = model(batch['input_ids'], mode='training')
            loss = F.cross_entropy(
                logits.view(-1, vocab_size), 
                batch['input_ids'].view(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += batch['input_ids'].numel()
    
    return torch.exp(torch.tensor(total_loss / total_tokens))
```

### Generation Quality

```python
def evaluate_generation(model, prompts, max_length=100):
    model.eval()
    generations = []
    
    for prompt in prompts:
        with torch.no_grad():
            generated = model.generate(
                prompt,
                max_new_tokens=max_length,
                temperature=0.8,
                top_k=40
            )
            generations.append(generated)
    
    return generations
```

## Troubleshooting

### Common Issues

**Posterior Collapse**
- Symptoms: KL loss drops to zero, model ignores latent variable
- Solutions: Increase free bits, reduce KL weight, use KL annealing

**Training Instability**
- Symptoms: Loss spikes, gradient explosions
- Solutions: Gradient clipping, lower learning rate, warmup

**Poor Generation Quality**
- Symptoms: Repetitive or incoherent text
- Solutions: Adjust temperature, top-k sampling, increase model size

### Debugging Tips

1. **Monitor KL loss**: Should be positive and stable
2. **Check latent utilization**: Verify Z is being used
3. **Validate gradients**: Ensure gradients flow through all components
4. **Compare with baseline**: Train standard Transformer for comparison

## Best Practices

1. **Start small**: Begin with small models and datasets
2. **Use curriculum learning**: Gradually increase complexity
3. **Monitor closely**: Watch for posterior collapse
4. **Regular evaluation**: Check generation quality frequently
5. **Save checkpoints**: Regular saves for recovery
6. **Ablation studies**: Test different hyperparameters

## Next Steps

- **[Multi-GPU Training](multi-gpu.md)**: Scale to multiple GPUs
- **[Configuration](configuration.md)**: Detailed config options
- **[Synthetic Data](synthetic-data.md)**: Generate training data