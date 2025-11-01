"""Compare baseline and Free Transformer models."""

import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from free_transformer import FreeTransformer, ModelConfig, TransformerBaseline
from free_transformer.losses import (compute_reconstruction_loss,
                                     compute_vae_loss)
from free_transformer.synthetic_data import create_dataloaders
from free_transformer.train_utils import load_checkpoint


@torch.no_grad()
def evaluate_baseline(model, data_loader, device):
    """Evaluate baseline model."""
    model.eval()
    total_loss = 0
    total_samples = 0

    for inputs, targets in tqdm(data_loader, desc="Evaluating baseline"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = compute_reconstruction_loss(logits, targets)

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples
    perplexity = torch.exp(torch.tensor(avg_loss))

    return {
        "loss": avg_loss,
        "perplexity": perplexity.item(),
    }


@torch.no_grad()
def evaluate_free_transformer(model, data_loader, device, latent_dim=8):
    """Evaluate Free Transformer."""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_samples = 0

    for inputs, targets in tqdm(data_loader, desc="Evaluating Free Transformer"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits, z_logits = model(inputs, mode="training")
        loss, metrics = compute_vae_loss(logits, targets, z_logits, latent_dim=latent_dim)

        total_loss += metrics["loss/total"] * inputs.size(0)
        total_recon += metrics["loss/reconstruction"] * inputs.size(0)
        total_kl += metrics["loss/kl"] * inputs.size(0)
        total_samples += inputs.size(0)

    return {
        "loss": total_loss / total_samples,
        "reconstruction_loss": total_recon / total_samples,
        "kl_loss": total_kl / total_samples,
        "perplexity": torch.exp(torch.tensor(total_recon / total_samples)).item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare models")
    parser.add_argument("--baseline-checkpoint", type=str, help="Baseline checkpoint")
    parser.add_argument("--free-checkpoint", type=str, help="Free Transformer checkpoint")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results/comparison")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    # Better device detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Model Comparison: Baseline vs Free Transformer")
    print("=" * 60)

    # Load data
    _, val_loader = create_dataloaders(
        train_path=f"{args.data_dir}/train.pt",
        val_path=f"{args.data_dir}/val.pt",
        batch_size=args.batch_size,
        device=device,
    )

    results = {}

    # Evaluate baseline
    if args.baseline_checkpoint:
        print("\nEvaluating Baseline Transformer...")

        # Load checkpoint first to get the correct model size
        checkpoint = torch.load(args.baseline_checkpoint, map_location="cpu")

        # Infer model config from checkpoint structure
        model_keys = list(checkpoint["model_state_dict"].keys())
        state_dict = checkpoint["model_state_dict"]

        # Extract configuration from checkpoint shapes
        num_layers = len(
            [k for k in model_keys if k.startswith("blocks.") and k.endswith(".attn_norm.weight")]
        )

        # Get dimensions from token embedding
        vocab_size, hidden_dim = state_dict["token_embedding.weight"].shape

        # Get FFN dimension from first layer
        ffn_hidden_dim = state_dict["blocks.0.ffn.w1.weight"].shape[0]

        # Get max_seq_len from RoPE cache
        max_seq_len = state_dict["blocks.0.rope.cos_cached"].shape[1]

        # Get head dimension from RoPE
        head_dim = state_dict["blocks.0.rope.cos_cached"].shape[3]
        num_heads = hidden_dim // head_dim

        # Create config matching the checkpoint
        model_config = ModelConfig()
        model_config.num_layers = num_layers
        model_config.vocab_size = vocab_size
        model_config.hidden_dim = hidden_dim
        model_config.ffn_hidden_dim = ffn_hidden_dim
        model_config.max_seq_len = max_seq_len
        model_config.num_heads = num_heads
        model_config.num_kv_heads = num_heads  # Assume same for baseline

        print(f"Detected baseline config: {num_layers} layers, {hidden_dim}d, vocab={vocab_size}")

        model = TransformerBaseline(model_config).to(device)
        load_checkpoint(model, None, args.baseline_checkpoint)

        baseline_results = evaluate_baseline(model, val_loader, device)
        results["baseline"] = baseline_results

        print("Baseline Results:")
        print(f"  Loss: {baseline_results['loss']:.4f}")
        print(f"  Perplexity: {baseline_results['perplexity']:.4f}")

    # Evaluate Free Transformer
    if args.free_checkpoint:
        print("\nEvaluating Free Transformer...")

        # Load checkpoint first to get the correct model size
        checkpoint = torch.load(args.free_checkpoint, map_location="cpu")

        # Infer model config from checkpoint structure
        model_keys = list(checkpoint["model_state_dict"].keys())
        state_dict = checkpoint["model_state_dict"]

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

        # Get dimensions from token embedding
        vocab_size, hidden_dim = state_dict["token_embedding.weight"].shape

        # Get FFN dimension from first layer
        ffn_hidden_dim = state_dict["first_half_blocks.0.ffn.w1.weight"].shape[0]

        # Get max_seq_len from RoPE cache
        max_seq_len = state_dict["first_half_blocks.0.rope.cos_cached"].shape[1]

        # Get head dimension from RoPE
        head_dim = state_dict["first_half_blocks.0.rope.cos_cached"].shape[3]
        num_heads = hidden_dim // head_dim

        # Get latent dimension from encoder readout layer
        latent_dim = None
        if "encoder.readout.weight" in state_dict:
            latent_dim = state_dict["encoder.readout.weight"].shape[0]
        elif "latent_plan.binary_mapper.powers" in state_dict:
            latent_dim = state_dict["latent_plan.binary_mapper.powers"].shape[0]
        else:
            latent_dim = 16  # Default fallback

        # Create config matching the checkpoint
        model_config = ModelConfig()
        model_config.num_layers = total_layers
        model_config.split_layer = first_half_layers
        model_config.vocab_size = vocab_size
        model_config.hidden_dim = hidden_dim
        model_config.ffn_hidden_dim = ffn_hidden_dim
        model_config.max_seq_len = max_seq_len
        model_config.num_heads = num_heads
        model_config.num_kv_heads = num_heads
        model_config.latent_dim = latent_dim

        print(
            f"Detected free transformer config: {total_layers} layers ({first_half_layers}+{second_half_layers}), {hidden_dim}d, vocab={vocab_size}, latent_dim={latent_dim}"
        )

        model = FreeTransformer(model_config).to(device)
        load_checkpoint(model, None, args.free_checkpoint)

        free_results = evaluate_free_transformer(model, val_loader, device, model_config.latent_dim)
        results["free_transformer"] = free_results

        print("Free Transformer Results:")
        print(f"  Total Loss: {free_results['loss']:.4f}")
        print(f"  Reconstruction Loss: {free_results['reconstruction_loss']:.4f}")
        print(f"  KL Loss: {free_results['kl_loss']:.4f}")
        print(f"  Perplexity: {free_results['perplexity']:.4f}")

    # Comparison
    if "baseline" in results and "free_transformer" in results:
        print("\n" + "=" * 60)
        print("Comparison Summary")
        print("=" * 60)
        print(
            f"Perplexity Difference: "
            f"{results['free_transformer']['perplexity'] - results['baseline']['perplexity']:.4f}"
        )

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()
