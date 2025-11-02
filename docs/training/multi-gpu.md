# Multi-GPU Training

This guide covers distributed training of Free Transformer models using PyTorch's Fully Sharded Data Parallel (FSDP).

## Overview

Multi-GPU training enables:
- **Larger models**: Train models that don't fit on a single GPU
- **Faster training**: Parallel computation across multiple devices
- **Better throughput**: Higher effective batch sizes

Free Transformer supports FSDP for efficient distributed training.

## FSDP (Fully Sharded Data Parallel)

FSDP shards model parameters, gradients, and optimizer states across GPUs:

- **Parameter sharding**: Each GPU holds a subset of parameters
- **Gradient sharding**: Gradients are distributed across devices
- **Optimizer sharding**: Optimizer states are distributed
- **Communication**: All-gather for forward pass, reduce-scatter for backward pass

## Quick Start

### Using Makefile

```bash
# Train Free Transformer with FSDP (auto-detects GPUs)
make train-free-fsdp

# Train baseline with FSDP
make train-baseline-fsdp
```

### Using torchrun

```bash
# Auto-detect number of GPUs
torchrun --nproc_per_node=auto examples/train_free.py \
  --config configs/free_transformer.yaml \
  --use-fsdp

# Specify number of GPUs
torchrun --nproc_per_node=4 examples/train_free.py \
  --config configs/free_transformer.yaml \
  --use-fsdp
```

## Configuration

### FSDP Configuration

Add FSDP settings to your YAML config:

```yaml
distributed:
  # Enable FSDP
  use_fsdp: true
  
  # Sharding strategy
  fsdp_sharding_strategy: "full_shard"  # full_shard, shard_grad_op, no_shard
  
  # Backward prefetch
  fsdp_backward_prefetch: "backward_pre"  # backward_pre, backward_post
  
  # Forward prefetch
  fsdp_forward_prefetch: false
  
  # Auto-wrap policy
  fsdp_auto_wrap_policy: "transformer_auto_wrap"
  fsdp_min_num_params: 1000000  # Minimum parameters for wrapping
  
  # State dict type for checkpointing
  fsdp_state_dict_type: "full_state_dict"  # full_state_dict, local_state_dict, sharded_state_dict
  
  # Mixed precision
  fsdp_mixed_precision: true
  fsdp_param_dtype: "bfloat16"
  fsdp_reduce_dtype: "float32"
  fsdp_buffer_dtype: "float32"
```

### Training Configuration

Adjust training settings for multi-GPU:

```yaml
training:
  # Increase batch size for multiple GPUs
  batch_size: 64  # Per GPU batch size
  
  # Adjust learning rate for larger effective batch size
  learning_rate: 2e-4  # Scale with number of GPUs
  
  # Enable gradient checkpointing for memory efficiency
  gradient_checkpointing: true
  
  # Use mixed precision
  bf16: true
  
  # Gradient clipping
  gradient_clip_norm: 1.0
```

## Sharding Strategies

### 1. Full Shard (Recommended)

```yaml
fsdp_sharding_strategy: "full_shard"
```

- **Memory**: Lowest memory usage
- **Communication**: Highest communication overhead
- **Use case**: Large models, memory-constrained

### 2. Shard Grad Op

```yaml
fsdp_sharding_strategy: "shard_grad_op"
```

- **Memory**: Medium memory usage
- **Communication**: Medium communication overhead
- **Use case**: Balance between memory and communication

### 3. No Shard

```yaml
fsdp_sharding_strategy: "no_shard"
```

- **Memory**: Highest memory usage (like DDP)
- **Communication**: Lowest communication overhead
- **Use case**: Small models, communication-constrained

## Auto-Wrap Policies

### Transformer Auto-Wrap

```yaml
fsdp_auto_wrap_policy: "transformer_auto_wrap"
fsdp_min_num_params: 1000000
```

Automatically wraps transformer layers with sufficient parameters.

### Size-Based Auto-Wrap

```yaml
fsdp_auto_wrap_policy: "size_based_auto_wrap"
fsdp_min_num_params: 1000000
```

Wraps any module with more than `min_num_params` parameters.

### Custom Auto-Wrap

```python
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from free_transformer.model import DecoderBlock

def get_custom_wrap_policy():
    return ModuleWrapPolicy({DecoderBlock})
```

## Memory Optimization

### Mixed Precision

```yaml
fsdp_mixed_precision: true
fsdp_param_dtype: "bfloat16"    # Parameter dtype
fsdp_reduce_dtype: "float32"    # Gradient reduction dtype
fsdp_buffer_dtype: "float32"    # Buffer dtype
```

### Activation Checkpointing

```yaml
training:
  gradient_checkpointing: true
```

Trades computation for memory by recomputing activations during backward pass.

### CPU Offloading

```python
from torch.distributed.fsdp import CPUOffload

fsdp_config = {
    "cpu_offload": CPUOffload(offload_params=True)
}
```

Offloads parameters to CPU when not in use.

## Checkpointing

### Full State Dict (Recommended)

```yaml
fsdp_state_dict_type: "full_state_dict"
```

- **Pros**: Compatible with single-GPU loading, easy to use
- **Cons**: Requires gathering all parameters on rank 0

### Sharded State Dict

```yaml
fsdp_state_dict_type: "sharded_state_dict"
```

- **Pros**: Memory efficient, faster saving/loading
- **Cons**: Requires same number of GPUs to load

### Local State Dict

```yaml
fsdp_state_dict_type: "local_state_dict"
```

- **Pros**: Each rank saves its own shard
- **Cons**: Complex to manage, requires custom loading logic

## Example Training Script

```python
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from free_transformer import FreeTransformer, ModelConfig
from free_transformer.model import DecoderBlock

def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def get_fsdp_config():
    """Get FSDP configuration."""
    return {
        "auto_wrap_policy": ModuleWrapPolicy({DecoderBlock}),
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
        "mixed_precision": MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        "device_id": torch.cuda.current_device(),
    }

def main():
    setup_distributed()
    
    # Create model
    config = ModelConfig(
        vocab_size=50000,
        hidden_dim=1024,
        num_layers=24,
        num_heads=16,
        latent_dim=64
    )
    
    model = FreeTransformer(config)
    
    # Wrap with FSDP
    model = FSDP(model, **get_fsdp_config())
    
    # Training loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        logits, z_logits = model(batch['input_ids'], mode='training')
        loss = compute_loss(logits, z_logits, batch['input_ids'])
        
        loss.backward()
        optimizer.step()
        
        if dist.get_rank() == 0:
            print(f"Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
```

## Performance Tuning

### Batch Size Scaling

Scale batch size with number of GPUs:

```python
# Effective batch size = batch_size * num_gpus * gradient_accumulation_steps
num_gpus = torch.cuda.device_count()
effective_batch_size = 256
batch_size_per_gpu = effective_batch_size // num_gpus
```

### Learning Rate Scaling

Scale learning rate with effective batch size:

```python
# Linear scaling rule
base_lr = 1e-4
base_batch_size = 32
effective_batch_size = batch_size_per_gpu * num_gpus
scaled_lr = base_lr * (effective_batch_size / base_batch_size)
```

### Communication Optimization

```yaml
# Reduce communication frequency
fsdp_backward_prefetch: "backward_pre"  # Prefetch parameters
fsdp_forward_prefetch: true             # Prefetch for forward pass

# Use efficient data types
fsdp_param_dtype: "bfloat16"           # Reduce parameter size
fsdp_reduce_dtype: "float32"           # Maintain precision for gradients
```

## Monitoring and Debugging

### Memory Usage

```python
def print_memory_stats():
    """Print GPU memory statistics."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
```

### Communication Profiling

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    with_stack=True,
) as prof:
    # Training step
    pass

prof.export_chrome_trace("trace.json")
```

### FSDP Debug Mode

```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
torchrun --nproc_per_node=4 examples/train_free.py --config configs/free_transformer.yaml --use-fsdp
```

## Troubleshooting

### Common Issues

**Out of Memory**
- Reduce batch size per GPU
- Enable gradient checkpointing
- Use CPU offloading
- Increase sharding level

**Slow Training**
- Check network bandwidth between GPUs
- Reduce communication overhead
- Use mixed precision
- Optimize data loading

**Convergence Issues**
- Adjust learning rate for effective batch size
- Use gradient clipping
- Check for numerical instabilities
- Monitor gradient norms

### NCCL Issues

```bash
# Set NCCL debug level
export NCCL_DEBUG=INFO

# Set NCCL timeout
export NCCL_TIMEOUT=1800

# Use specific network interface
export NCCL_SOCKET_IFNAME=eth0
```

## Best Practices

1. **Start small**: Test with 2 GPUs before scaling up
2. **Monitor memory**: Use memory profiling to optimize usage
3. **Scale gradually**: Increase model size and GPU count incrementally
4. **Use checkpointing**: Save frequently with full state dict
5. **Profile communication**: Identify and optimize bottlenecks
6. **Test convergence**: Ensure multi-GPU results match single-GPU

## Hardware Recommendations

### GPU Configuration
- **Memory**: 24GB+ per GPU for large models
- **Interconnect**: NVLink or InfiniBand for best performance
- **Topology**: Avoid crossing CPU sockets when possible

### Network Requirements
- **Bandwidth**: 100Gbps+ for large-scale training
- **Latency**: Low latency interconnects (InfiniBand, NVLink)
- **Topology**: All-to-all connectivity preferred

## Next Steps

- **[Training Guide](guide.md)**: General training best practices
- **[Configuration](configuration.md)**: Detailed configuration options
- **[Examples](../examples/basic.md)**: See multi-GPU examples in action