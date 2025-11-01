# Docker Setup for Free Transformer Demo

This directory contains Docker configurations for running the Free Transformer demo with small synthetic data.

## Quick Start

### Option 1: Using Docker Compose (Recommended)

**For GPU systems:**
```bash
# Run the demo
docker-compose up free-transformer-demo

# For interactive development
docker-compose up -d free-transformer-interactive
docker-compose exec free-transformer-interactive bash
```

**For CPU-only systems:**
```bash
# Build CPU image
docker build -f Dockerfile.cpu -t free-transformer:cpu .

# Run demo
docker run --rm -v $(pwd)/data:/workspace/data -v $(pwd)/checkpoints:/workspace/checkpoints free-transformer:cpu
```

### Option 2: Using Docker directly

**Build and run GPU version:**
```bash
docker build -t free-transformer:demo .
docker run --gpus all --rm \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  free-transformer:demo
```

**Build and run CPU version:**
```bash
docker build -f Dockerfile.cpu -t free-transformer:cpu .
docker run --rm \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  free-transformer:cpu
```

## What the Demo Does

The demo script (`demo.sh`) performs the following steps:

1. **Data Generation**: Creates small synthetic dataset (1000 train, 200 val samples)
2. **Baseline Training**: Trains a standard Transformer for 100 steps
3. **Free Transformer Training**: Trains the Free Transformer for 100 steps
4. **Model Comparison**: Compares the performance of both models

## Configuration

The demo uses small configurations optimized for quick testing:

- **Vocabulary size**: 1000 tokens
- **Hidden dimension**: 128
- **Sequence length**: 128
- **Training steps**: 100
- **Batch size**: 8

These settings allow the demo to complete in a few minutes while demonstrating the key differences between the models.

## Customization

### Custom Configurations

To use different configurations, mount your config files:

```bash
docker run --gpus all --rm \
  -v $(pwd)/configs:/workspace/configs \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  free-transformer:demo
```

### Manual Training

For interactive training and experimentation:

```bash
# Start interactive container
docker-compose up -d free-transformer-interactive
docker-compose exec free-transformer-interactive bash

# Inside container, run individual commands
python examples/generate_data.py --vocab-size 2000 --num-train 5000
python examples/train_baseline.py --config configs/baseline.yaml
python examples/train_free.py --config configs/free_transformer.yaml
```

## Resource Requirements

### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 2GB free space

### Recommended for GPU
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **CUDA**: 11.8 or compatible
- **RAM**: 8GB
- **Storage**: 5GB free space

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in config files
2. **Permission denied**: Ensure Docker has access to mounted directories
3. **Slow training**: Use GPU version or reduce model size

### Logs and Debugging

Check container logs:
```bash
docker-compose logs free-transformer-demo
```

Access container for debugging:
```bash
docker-compose exec free-transformer-interactive bash
```

## Output

After successful completion, you'll find:

- **Data**: `./data/train.pt`, `./data/val.pt`
- **Baseline model**: `./checkpoints/demo/baseline/`
- **Free Transformer model**: `./checkpoints/demo/free/`
- **Training logs**: Console output with loss curves and metrics

## Next Steps

1. **Experiment with larger models**: Modify config files for full-scale training
2. **Try real datasets**: Replace synthetic data with actual text data
3. **Analyze results**: Use the evaluation scripts to compare model performance
4. **Scale up**: Use the FSDP options for multi-GPU training