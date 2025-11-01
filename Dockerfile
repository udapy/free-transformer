FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python and UV
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Set working directory
WORKDIR /workspace

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv python install 3.12
RUN uv venv && . .venv/bin/activate && uv pip install -e ".[dev]"

# Copy project files
COPY . /workspace/

# Create data directory
RUN mkdir -p /workspace/data /workspace/checkpoints

# Set environment variables
ENV PYTHONPATH="/workspace/src:$PYTHONPATH"
ENV CUDA_VISIBLE_DEVICES=0

# Create demo script
COPY docker/demo.sh /workspace/demo.sh
RUN chmod +x /workspace/demo.sh

# Default command runs the demo
CMD ["./demo.sh"]
