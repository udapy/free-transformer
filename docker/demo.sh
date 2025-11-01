#!/bin/bash

# Free Transformer Demo Script
# This script runs a quick demo with small synthetic data

set -e

echo "🚀 Starting Free Transformer Demo"
echo "=================================="

# Activate virtual environment
source .venv/bin/activate

# Check if data exists, if not generate it
if [ ! -f "data/train.pt" ] || [ ! -f "data/val.pt" ]; then
    echo "📊 Generating small synthetic dataset..."
    python examples/generate_data.py \
        --output-dir ./data \
        --vocab-size 1000 \
        --seq-length 128 \
        --num-train 1000 \
        --num-val 200 \
        --seed 42
    echo "✅ Data generation complete!"
else
    echo "📊 Using existing dataset"
fi

echo ""
echo "🔧 Training Baseline Transformer (100 steps)..."
python examples/train_baseline.py \
    --config configs/baseline.yaml \
    --output-dir ./checkpoints/demo/baseline

echo ""
echo "🔧 Training Free Transformer (100 steps)..."
python examples/train_free.py \
    --config configs/free_transformer.yaml \
    --output-dir ./checkpoints/demo/free

echo ""
echo "📈 Comparing models..."
if [ -f "examples/eval_compare.py" ]; then
    python examples/eval_compare.py \
        --baseline-checkpoint ./checkpoints/demo/baseline/model_final.pt \
        --free-checkpoint ./checkpoints/demo/free/model_final.pt \
        --config-baseline configs/baseline.yaml \
        --config-free configs/free_transformer.yaml
else
    echo "⚠️  Evaluation script not found, skipping comparison"
fi

echo ""
echo "🎉 Demo complete! Check the checkpoints directory for trained models."
echo "📁 Baseline model: ./checkpoints/demo/baseline/"
echo "📁 Free Transformer model: ./checkpoints/demo/free/"