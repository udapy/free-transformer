"""Generate synthetic training data."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from free_transformer.synthetic_data import SyntheticDataGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--output-dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length")
    parser.add_argument("--num-train", type=int, default=50000, help="Number of training samples")
    parser.add_argument("--num-val", type=int, default=5000, help="Number of validation samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic data:")
    print(f"  Vocabulary size: {args.vocab_size}")
    print(f"  Sequence length: {args.seq_length}")
    print(f"  Training samples: {args.num_train}")
    print(f"  Validation samples: {args.num_val}")
    print(f"  Random seed: {args.seed}")

    # Generate training data
    print("\nGenerating training data...")
    train_generator = SyntheticDataGenerator(
        vocab_size=args.vocab_size,
        seq_length=args.seq_length,
        seed=args.seed,
    )
    train_generator.save_dataset(
        num_samples=args.num_train, output_path=str(output_dir / "train.pt")
    )
    print(f"✓ Saved to {output_dir / 'train.pt'}")

    # Generate validation data
    print("\nGenerating validation data...")
    val_generator = SyntheticDataGenerator(
        vocab_size=args.vocab_size,
        seq_length=args.seq_length,
        seed=args.seed + 1,  # Different seed for validation
    )
    val_generator.save_dataset(num_samples=args.num_val, output_path=str(output_dir / "val.pt"))
    print(f"✓ Saved to {output_dir / 'val.pt'}")

    print("\n✅ Data generation complete!")


if __name__ == "__main__":
    main()
