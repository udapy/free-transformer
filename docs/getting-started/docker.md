# Docker Setup

Docker provides the fastest way to try Free Transformer without local installation.

## Quick Demo

Run the complete demo with one command:

```bash
git clone https://github.com/udapy/free-transformer.git
cd free-transformer
docker-compose up free-transformer-demo
```

This will:
1. Generate synthetic training data
2. Train baseline and Free Transformer models
3. Compare their performance
4. Display results

## Docker Images

### GPU Version (Recommended)

For NVIDIA GPUs with CUDA support:

```bash
# Build the image
make docker-build

# Run the demo
make docker-demo

# Interactive development
make docker-interactive
```

### CPU Version

For systems without GPU or CUDA:

```bash
# Build CPU-only image
make docker-build-cpu

# Run CPU demo
make docker-run-cpu
```

## Manual Docker Commands

### Build Images

```bash
# GPU version
docker build -t free-transformer:gpu .

# CPU version  
docker build -f Dockerfile.cpu -t free-transformer:cpu .
```

### Run Containers

```bash
# GPU demo
docker run --gpus all -v $(pwd)/results:/app/results free-transformer:gpu

# CPU demo
docker run -v $(pwd)/results:/app/results free-transformer:cpu

# Interactive shell
docker run --gpus all -it -v $(pwd):/app free-transformer:gpu bash
```

## Docker Compose

The `docker-compose.yml` provides several services:

### Demo Service

```yaml
services:
  free-transformer-demo:
    build: .
    volumes:
      - ./results:/app/results
      - ./checkpoints:/app/checkpoints
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Run with:
```bash
docker-compose up free-transformer-demo
```

### Development Service

```yaml
  free-transformer-dev:
    build: .
    volumes:
      - .:/app
    stdin_open: true
    tty: true
    command: bash
```

Run with:
```bash
docker-compose run free-transformer-dev
```

## Volume Mounts

Important directories to mount:

- `./results:/app/results` - Evaluation results
- `./checkpoints:/app/checkpoints` - Model checkpoints  
- `./data:/app/data` - Training data
- `.:/app` - Full source code (development)

## Environment Variables

Configure the container with environment variables:

```bash
docker run \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e WANDB_API_KEY=your_key \
  -e TORCH_DISTRIBUTED_DEBUG=INFO \
  free-transformer:gpu
```

Common variables:
- `CUDA_VISIBLE_DEVICES` - GPU selection
- `WANDB_API_KEY` - Weights & Biases logging
- `TORCH_DISTRIBUTED_DEBUG` - Distributed training debug
- `OMP_NUM_THREADS` - CPU thread count

## GPU Requirements

### NVIDIA Docker Setup

1. Install NVIDIA drivers
2. Install Docker
3. Install NVIDIA Container Toolkit:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

4. Test GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

## Customization

### Custom Dockerfile

Create your own Dockerfile for specific needs:

```dockerfile
FROM free-transformer:gpu

# Add custom dependencies
RUN pip install your-package

# Copy custom configs
COPY my-config.yaml /app/configs/

# Set custom entrypoint
ENTRYPOINT ["python", "my-script.py"]
```

### Custom Docker Compose

```yaml
version: '3.8'
services:
  my-training:
    build: .
    volumes:
      - ./my-data:/app/data
      - ./my-configs:/app/configs
    environment:
      - WANDB_PROJECT=my-project
    command: python examples/train_free.py --config configs/my-config.yaml
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA Docker setup
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Check container GPU access
docker run --rm --gpus all free-transformer:gpu python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory

Reduce batch size in configs or use CPU version:

```bash
# Use CPU version
docker-compose -f docker-compose.yml up free-transformer-demo-cpu

# Or modify config
docker run -v $(pwd)/configs:/app/configs free-transformer:gpu \
  python examples/train_free.py --config configs/small.yaml
```

### Permission Issues

Fix volume mount permissions:

```bash
# Create directories with correct permissions
mkdir -p results checkpoints data
chmod 777 results checkpoints data

# Or run with user ID
docker run --user $(id -u):$(id -g) -v $(pwd):/app free-transformer:gpu
```

## Performance Tips

1. **Use GPU version** for significant speedup
2. **Mount SSD storage** for data directories
3. **Allocate sufficient memory** (8GB+ recommended)
4. **Use multi-GPU** for large models:
   ```bash
   docker run --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3 free-transformer:gpu
   ```

## Next Steps

- **[Quick Start](quick-start.md)**: Run your first training
- **[Training Guide](../training/guide.md)**: Advanced training setup
- **[Multi-GPU](../training/multi-gpu.md)**: Distributed training