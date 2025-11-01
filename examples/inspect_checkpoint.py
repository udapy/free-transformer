#!/usr/bin/env python3
"""Utility to inspect checkpoint structure and infer model configuration."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def inspect_checkpoint(checkpoint_path: str):
    """Inspect a checkpoint and infer model configuration."""
    print(f"Inspecting checkpoint: {checkpoint_path}")
    print("=" * 60)

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return

    print(f"Checkpoint keys: {list(checkpoint.keys())}")

    if "model_state_dict" not in checkpoint:
        print("❌ No 'model_state_dict' found in checkpoint")
        return

    model_keys = list(checkpoint["model_state_dict"].keys())
    print(f"Model state dict has {len(model_keys)} keys")

    # Detect model type and configuration
    if any(k.startswith("first_half_blocks.") for k in model_keys):
        print("\n🔍 Detected: Free Transformer checkpoint")

        first_half_layers = len(
            [
                k
                for k in model_keys
                if k.startswith("first_half_blocks.") and k.endswith(".attn_norm.weight")
            ]
        )
        second_half_layers = len(
            [
                k
                for k in model_keys
                if k.startswith("second_half_blocks.") and k.endswith(".attn_norm.weight")
            ]
        )
        total_layers = first_half_layers + second_half_layers

        print(f"  Total layers: {total_layers}")
        print(f"  Split at layer: {first_half_layers}")
        print(f"  First half: {first_half_layers} layers")
        print(f"  Second half: {second_half_layers} layers")

        # Check for latent dimension
        encoder_keys = [k for k in model_keys if k.startswith("encoder.")]
        if encoder_keys:
            print(f"  Encoder keys: {len(encoder_keys)}")

        # Check for injection mechanism
        injection_keys = [k for k in model_keys if k.startswith("injection.")]
        if injection_keys:
            print(f"  Injection keys: {len(injection_keys)}")

    elif any(k.startswith("blocks.") for k in model_keys):
        print("\n🔍 Detected: Baseline Transformer checkpoint")

        num_layers = len(
            [k for k in model_keys if k.startswith("blocks.") and k.endswith(".attn_norm.weight")]
        )
        print(f"  Number of layers: {num_layers}")

        # Check embedding dimension
        if "token_embedding.weight" in model_keys:
            embedding_shape = checkpoint["model_state_dict"]["token_embedding.weight"].shape
            vocab_size, hidden_dim = embedding_shape
            print(f"  Vocabulary size: {vocab_size}")
            print(f"  Hidden dimension: {hidden_dim}")

        # Check attention heads
        if "blocks.0.wq.weight" in model_keys:
            wq_shape = checkpoint["model_state_dict"]["blocks.0.wq.weight"].shape
            total_q_dim = wq_shape[0]
            num_heads = total_q_dim // hidden_dim if "hidden_dim" in locals() else "unknown"
            print(f"  Estimated num_heads: {num_heads}")

    else:
        print("\n❓ Unknown checkpoint format")

    # Additional info
    if "step" in checkpoint:
        print(f"\nTraining step: {checkpoint['step']}")

    if "optimizer_state_dict" in checkpoint:
        print("✅ Optimizer state included")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Inspect checkpoint structure")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    args = parser.parse_args()

    inspect_checkpoint(args.checkpoint)


if __name__ == "__main__":
    main()
