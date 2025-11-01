"""Synthetic data generation for training and testing."""

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticDataGenerator:
    """Generate synthetic token sequences for training."""

    def __init__(
        self,
        vocab_size: int = 10000,
        seq_length: int = 512,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_sample(self) -> torch.Tensor:
        """Generate a single random sequence."""
        tokens = self.rng.randint(0, self.vocab_size, size=self.seq_length)
        return torch.tensor(tokens, dtype=torch.long)

    def generate_batch(self, batch_size: int) -> torch.Tensor:
        """Generate a batch of sequences."""
        tokens = self.rng.randint(0, self.vocab_size, size=(batch_size, self.seq_length))
        return torch.tensor(tokens, dtype=torch.long)

    def generate_dataset(self, num_samples: int) -> torch.Tensor:
        """Generate full dataset."""
        tokens = self.rng.randint(0, self.vocab_size, size=(num_samples, self.seq_length))
        return torch.tensor(tokens, dtype=torch.long)

    def save_dataset(self, num_samples: int, output_path: str):
        """Generate and save dataset to disk."""
        data = self.generate_dataset(num_samples)
        torch.save(data, output_path)

        # Save metadata
        metadata = {
            "vocab_size": self.vocab_size,
            "seq_length": self.seq_length,
            "num_samples": num_samples,
            "seed": self.seed,
        }
        metadata_path = Path(output_path).parent / f"{Path(output_path).stem}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)


class SyntheticDataset(Dataset):
    """PyTorch Dataset for synthetic data."""

    def __init__(self, data_path: str):
        self.data = torch.load(data_path)
        metadata_path = Path(data_path).parent / f"{Path(data_path).stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return input and target (shifted by 1)."""
        tokens = self.data[idx]
        # Input: all tokens except last
        # Target: all tokens except first (shifted by 1)
        return tokens[:-1], tokens[1:]


def create_dataloaders(
    train_path: str,
    val_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    device: Optional[torch.device] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    train_dataset = SyntheticDataset(train_path)
    val_dataset = SyntheticDataset(val_path)

    # Only use pin_memory for CUDA devices
    pin_memory = device is not None and device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader
