"""Train Free Transformer."""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from free_transformer import FreeTransformer, ModelConfig, TrainingConfig
from free_transformer.losses import compute_vae_loss
from free_transformer.synthetic_data import create_dataloaders
from free_transformer.train_utils import (LRScheduler, Trainer,
                                          cleanup_distributed,
                                          count_parameters, save_checkpoint,
                                          setup_distributed)


class FreeTransformerTrainer(Trainer):
    def __init__(self, model, optimizer, device, train_config, use_amp=False):
        super().__init__(model, optimizer, device, use_amp)
        self.train_config = train_config

    def _compute_loss(self, batch):
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        logits, z_logits = self.model(inputs, mode="training")
        loss, metrics = compute_vae_loss(
            logits,
            targets,
            z_logits,
            latent_dim=self.model.config.latent_dim,
            beta_kl=self.train_config.beta_kl,
            free_bits=self.train_config.kappa_free_bits,
        )

        return loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train Free Transformer")
    parser.add_argument("--config", type=str, required=True, help="Config file")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/free")
    parser.add_argument("--use-fsdp", action="store_true", help="Use FSDP")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)

    model_config = ModelConfig(**config_dict["model"])
    train_config = TrainingConfig(**config_dict["training"])
    data_config = config_dict["data"]

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    # Better device detection
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if rank == 0:
        print("=" * 50)
        print("Training Free Transformer")
        print("=" * 50)
        print(f"Model: {count_parameters(FreeTransformer(model_config))/1e6:.2f}M parameters")
        print(
            f"Latent dim: {model_config.latent_dim} (2^{model_config.latent_dim} = {2**model_config.latent_dim} codes)"
        )
        print(f"Free bits: {train_config.kappa_free_bits:.4f}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_path=f"{data_config['data_dir']}/train.pt",
        val_path=f"{data_config['data_dir']}/val.pt",
        batch_size=train_config.batch_size,
        num_workers=data_config.get("num_workers", 4),
        device=device,
    )

    # Initialize model
    model = FreeTransformer(model_config).to(device)

    if args.use_fsdp and world_size > 1:
        from free_transformer.train_utils import wrap_model_fsdp

        model = wrap_model_fsdp(model)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
    )

    scheduler = LRScheduler(
        optimizer,
        warmup_steps=train_config.warmup_steps,
        max_lr=train_config.learning_rate,
    )

    # Trainer
    trainer = FreeTransformerTrainer(model, optimizer, device, train_config, use_amp=True)

    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    pbar = tqdm(total=train_config.max_steps, disable=rank != 0)

    while step < train_config.max_steps:
        for batch in train_loader:
            if step >= train_config.max_steps:
                break

            # Train step
            metrics = trainer.train_step(batch, grad_clip=train_config.grad_clip)
            lr = scheduler.step()
            metrics["lr"] = lr

            # Logging
            if step % train_config.log_every == 0 and rank == 0:
                pbar.set_postfix(metrics)

            # Validation
            if step % train_config.eval_every == 0 and rank == 0:
                val_metrics = []
                for val_batch in val_loader:
                    val_metrics.append(trainer.eval_step(val_batch))

                avg_metrics = {}
                for key in val_metrics[0].keys():
                    avg_metrics[key] = sum(m[key] for m in val_metrics) / len(val_metrics)

                print(
                    f"\nStep {step} | Val Loss: {avg_metrics['loss/total']:.4f} | "
                    f"KL: {avg_metrics['loss/kl']:.4f}"
                )

            # Checkpointing
            if step % train_config.save_every == 0 and rank == 0:
                save_checkpoint(model, optimizer, step, str(output_dir / f"checkpoint_{step}.pt"))

            step += 1
            pbar.update(1)

    # Save final model
    if rank == 0:
        save_checkpoint(model, optimizer, step, str(output_dir / "model_final.pt"))
        print("\n✅ Training complete!")

    cleanup_distributed()


if __name__ == "__main__":
    main()
