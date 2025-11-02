# Frequently Asked Questions

## General Questions

### What is the Free Transformer?

The Free Transformer is a novel neural architecture that extends traditional autoregressive Transformers with explicit latent planning. Instead of generating tokens purely reactively based on previous tokens, it first creates an abstract "plan" (latent variable Z) and then generates tokens conditioned on both the history and this plan.

### How does it differ from standard Transformers?

| Aspect | Standard Transformer | Free Transformer |
|--------|---------------------|------------------|
| **Generation** | Reactive (token-by-token) | Plan-then-generate |
| **Training** | Language modeling loss | Conditional VAE loss |
| **Coherence** | Local | Global + Local |
| **Controllability** | Limited | High (via plan manipulation) |
| **Architecture** | Decoder-only | Decoder + Encoder + Latent |

### What are the main benefits?

1. **Better long-range coherence**: The latent plan helps maintain consistency across long sequences
2. **Controllable generation**: You can potentially manipulate the latent plan for controlled text generation
3. **Richer representations**: The model learns more structured internal representations
4. **Improved sample diversity**: Different plans lead to different generation styles

## Technical Questions

### What is the latent dimension and how do I choose it?

The latent dimension (`latent_dim`) determines the size of the binary plan vector Z. Typical values:

- **Small models (< 100M params)**: 8-16 dimensions
- **Medium models (100M-1B params)**: 16-32 dimensions  
- **Large models (> 1B params)**: 32-64 dimensions

Start with 16-32 and adjust based on your model size and task complexity.

### What is "free bits" and why is it important?

Free bits is a regularization technique that prevents posterior collapse in VAE training. It sets a minimum threshold for the KL divergence loss:

```python
kl_loss = torch.clamp(kl_loss, min=free_bits)
```

Typical values: 0.5-2.0. Higher values encourage more latent variable usage but may hurt reconstruction quality.

### How do I know if my model is working correctly?

Monitor these metrics during training:

1. **KL loss should be positive**: If it drops to zero, you have posterior collapse
2. **Reconstruction loss should decrease**: Standard language modeling progress
3. **Total loss should be stable**: No sudden spikes or instability
4. **Generation quality**: Manually inspect generated text

### What's the difference between training and inference modes?

**Training mode**:
- Uses the encoder to compute latent plan from the full sequence
- Optimizes both reconstruction and KL losses
- Plan is derived from the actual data

**Inference mode**:
- Samples latent plan from uniform prior (no encoder needed)
- Only uses reconstruction for generation
- Plan is randomly sampled

## Usage Questions

### Can I use this for real-world datasets?

Yes! While the examples use synthetic data for quick prototyping, the model works with any text dataset. You can:

1. Use HuggingFace datasets directly
2. Provide your own text files
3. Modify the data loading pipeline in `synthetic_data.py`

### How do I run on multiple GPUs?

Use FSDP (Fully Sharded Data Parallel):

```bash
# Automatic GPU detection
torchrun --nproc_per_node=auto examples/train_free.py --config configs/free_transformer.yaml --use-fsdp

# Or use the Makefile
make train-free-fsdp
```

### Can I run this without a GPU?

Yes, but it will be much slower. Use the CPU Docker image:

```bash
make docker-build-cpu
make docker-run-cpu
```

Or set device in your code:
```python
model = FreeTransformer(config)
model = model.to('cpu')
```

### How do I change the model size?

Edit the configuration file or create a new one:

```yaml
model:
  hidden_dim: 768      # Increase for larger model
  num_layers: 24       # More layers = more capacity
  num_heads: 12        # Usually hidden_dim // 64
  latent_dim: 32       # Scale with model size
```

## Training Questions

### My KL loss is zero. What's wrong?

This is posterior collapse. The model is ignoring the latent variable. Solutions:

1. **Increase free bits**: Try 1.0-2.0 instead of 0.5
2. **Reduce KL weight**: Start with 0.01-0.05 instead of 0.1
3. **Use KL annealing**: Gradually increase KL weight during training
4. **Check latent dimension**: Might be too large for your model

### Training is unstable. How do I fix it?

Common solutions:

1. **Gradient clipping**: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
2. **Lower learning rate**: Try 1e-5 instead of 1e-4
3. **Warmup**: Use learning rate warmup for first 1000 steps
4. **Mixed precision**: Can help with stability and speed

### How long should I train?

Depends on your dataset and model size:

- **Small synthetic data**: 5-10 epochs
- **Medium datasets (1M-10M tokens)**: 10-50 epochs
- **Large datasets (100M+ tokens)**: 1-5 epochs

Monitor validation loss and stop when it plateaus.

### Should I use curriculum learning?

Yes, it often helps! Start with:

1. **Short sequences** (128 tokens) → **Long sequences** (512+ tokens)
2. **High KL weight** (1.0) → **Low KL weight** (0.1)
3. **Simple data** → **Complex data**

## Comparison Questions

### How does it compare to other VAE-based language models?

The Free Transformer is specifically designed for autoregressive generation with:

- **Explicit binary plans**: More interpretable than continuous latents
- **Llama-style backbone**: Modern, efficient architecture
- **Flexible injection**: Plan can influence multiple layers
- **Training efficiency**: Competitive with standard Transformers

### When should I use Free Transformer vs standard Transformer?

**Use Free Transformer when**:
- You need better long-range coherence
- Controllable generation is important
- You're working with structured text (stories, articles)
- Sample diversity matters

**Use standard Transformer when**:
- You need maximum training efficiency
- Working with very short sequences
- Simplicity is preferred
- You have limited computational resources

## Deployment Questions

### Can I deploy this in production?

Yes, but consider:

1. **Inference mode is efficient**: No encoder overhead
2. **Model size**: Similar to equivalent standard Transformer
3. **Memory usage**: Slightly higher due to latent computations
4. **Latency**: Comparable to baseline models

### How do I optimize for inference?

1. **Use inference mode**: `model(tokens, mode='inference')`
2. **Enable eval mode**: `model.eval()`
3. **Disable gradients**: `torch.no_grad()`
4. **Consider quantization**: Standard PyTorch quantization works
5. **Batch inference**: Process multiple sequences together

### Can I convert to ONNX or TensorRT?

The model uses standard PyTorch operations, so conversion should work, but:

1. **Test thoroughly**: Some operations might not be supported
2. **Separate modes**: Export training and inference modes separately
3. **Dynamic shapes**: May need fixed input sizes

## Development Questions

### How do I contribute?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make test`
5. Run quality checks: `make quality`
6. Submit a pull request

### How do I add custom components?

The architecture is modular:

1. **Custom encoder**: Inherit from `nn.Module`, implement `forward()`
2. **Custom injection**: Modify `injection.py`
3. **Custom losses**: Add to `losses.py`
4. **Custom data**: Extend `synthetic_data.py`

### Where can I get help?

1. **Documentation**: This site covers most use cases
2. **GitHub Issues**: Report bugs or ask questions
3. **Code examples**: Check the `examples/` directory
4. **Tests**: Look at `tests/` for usage patterns

## Performance Questions

### How much slower is it than baseline Transformers?

**Training**: ~20-30% slower due to encoder and VAE loss
**Inference**: ~5-10% slower due to latent computations

The overhead is minimal and often worth it for the improved capabilities.

### How much memory does it use?

**Training**: ~30-40% more memory than baseline (due to encoder)
**Inference**: ~10-15% more memory than baseline

Use gradient checkpointing and mixed precision to reduce memory usage.

### Can I make it faster?

1. **Use mixed precision**: `torch.cuda.amp`
2. **Gradient checkpointing**: Trades compute for memory
3. **Efficient attention**: Flash Attention (planned feature)
4. **Model parallelism**: FSDP for large models
5. **Batch size tuning**: Find optimal batch size for your hardware

Still have questions? [Open an issue](https://github.com/udapy/free-transformer/issues) on GitHub!