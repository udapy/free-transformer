"""Training utilities including FSDP and DeepSpeed support."""

import functools
import os
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

if TYPE_CHECKING:
    from torch.cuda.amp import GradScaler


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=rank
        )
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def wrap_model_fsdp(
    model: nn.Module,
    mixed_precision: bool = True,
    min_num_params: int = 1000000,
) -> FSDP:
    """Wrap model with FSDP for distributed training."""

    # Mixed precision policy
    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    # Auto wrap policy
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)

    # Wrap with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
    )

    return model


class Trainer:
    """Base trainer class."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        use_amp: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_amp = use_amp

        if use_amp:
            # Use the new torch.amp API with device detection
            if device.type == "cuda":
                from torch.cuda.amp import GradScaler

                self.scaler: Optional["GradScaler"] = GradScaler()
            elif device.type == "mps":
                # MPS doesn't support GradScaler, disable AMP
                self.use_amp = False
                self.scaler = None
            else:
                # CPU doesn't support GradScaler, disable AMP
                self.use_amp = False
                self.scaler = None
        else:
            self.scaler = None

        self.step = 0

    def train_step(
        self,
        batch: tuple,
        grad_clip: Optional[float] = None,
    ) -> dict:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass with automatic mixed precision
        if self.use_amp and self.scaler is not None:
            with torch.amp.autocast(device_type=self.device.type):
                loss, metrics = self._compute_loss(batch)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            if grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, metrics = self._compute_loss(batch)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

            self.optimizer.step()

        self.step += 1
        return metrics

    def _compute_loss(self, batch: tuple) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute loss - to be implemented by subclasses."""
        raise NotImplementedError

    @torch.no_grad()
    def eval_step(self, batch: tuple) -> dict[str, float]:
        """Single evaluation step."""
        self.model.eval()
        loss, metrics = self._compute_loss(batch)
        return metrics


class LRScheduler:
    """Learning rate scheduler with warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_lr: float,
        min_lr: float = 0.0,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.step_count = 0

    def step(self):
        """Update learning rate."""
        self.step_count += 1

        if self.step_count < self.warmup_steps:
            # Linear warmup
            lr = self.max_lr * self.step_count / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / self.warmup_steps
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                1 + torch.cos(torch.tensor(progress * 3.14159))
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: str,
    metadata: Optional[dict] = None,
):
    """Save training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }
    if metadata is not None:
        checkpoint.update(metadata)

    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
) -> int:
    """Load training checkpoint with better error handling."""
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {e}")

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"Checkpoint at {path} does not contain 'model_state_dict'")

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        # Provide more helpful error message
        model_keys = set(model.state_dict().keys())
        checkpoint_keys = set(checkpoint["model_state_dict"].keys())

        missing_in_checkpoint = model_keys - checkpoint_keys
        missing_in_model = checkpoint_keys - model_keys

        error_msg = f"State dict mismatch when loading {path}:\n"
        if missing_in_checkpoint:
            error_msg += f"  Missing in checkpoint: {sorted(list(missing_in_checkpoint))[:5]}...\n"
        if missing_in_model:
            error_msg += f"  Missing in model: {sorted(list(missing_in_model))[:5]}...\n"
        error_msg += f"  Original error: {e}"

        raise RuntimeError(error_msg)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return int(checkpoint.get("step", 0))
