#!/bin/bash
set -e

echo "🚀 Free Transformer Setup"
echo "========================="

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "✅ UV version: $(uv --version)"

# Create environment
echo "🐍 Creating Python environment..."
uv python install 3.12
uv venv --python 3.12

# Activate environment
source .venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
uv pip install -e ".[dev]"

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data checkpoints results logs

# Initialize git hooks (if in git repo)
if [ -d ".git" ]; then
    echo "🪝 Setting up pre-commit hooks..."
    uv run pre-commit install
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate environment: source .venv/bin/activate"
echo "  2. Generate data: make generate-data-small"
echo "  3. Run tests: make test"
echo "  4. Train model: make train-free"
echo ""
echo "Run 'make help' to see all available commands."
