from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

try:
    from rl.vae import ConvVAE, vae_loss
    from rl.vae_dataset import VaeMemmapDataset
except ImportError:
    from vae import ConvVAE, vae_loss
    from vae_dataset import VaeMemmapDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Raffin-style ConvVAE on Donkey simulator images.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/vae/cache_raffin"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/vae_raffin_v1"))
    parser.add_argument("--z-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--kl-tolerance", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--save-samples", type=int, default=16)
    parser.add_argument("--log-interval", type=int, default=200)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def run_epoch(
    model: ConvVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    *,
    beta: float,
    kl_tolerance: float,
    epoch: int | None = None,
    log_interval: int = 0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "r_loss": 0.0, "kl_loss": 0.0, "n": 0}

    started_at = time.perf_counter()
    for batch_idx, batch in enumerate(loader, start=1):
        batch = batch.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            recon, mu, logvar = model(batch)
            loss, r_loss, kl_loss = vae_loss(
                recon,
                batch,
                mu,
                logvar,
                beta=beta,
                kl_tolerance=kl_tolerance,
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_n = batch.shape[0]
        totals["loss"] += float(loss.detach()) * batch_n
        totals["r_loss"] += float(r_loss.detach()) * batch_n
        totals["kl_loss"] += float(kl_loss.detach()) * batch_n
        totals["n"] += batch_n

        if training and log_interval > 0 and batch_idx % log_interval == 0:
            n = max(totals["n"], 1)
            elapsed = time.perf_counter() - started_at
            print(
                f"epoch={epoch:03d} batch={batch_idx:04d}/{len(loader):04d} "
                f"loss={totals['loss'] / n:.4f} r={totals['r_loss'] / n:.4f} "
                f"kl={totals['kl_loss'] / n:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    n = max(totals["n"], 1)
    return {
        "loss": totals["loss"] / n,
        "r_loss": totals["r_loss"] / n,
        "kl_loss": totals["kl_loss"] / n,
    }


@torch.no_grad()
def save_reconstructions(model: ConvVAE, loader: DataLoader, device: torch.device, path: Path, n: int) -> None:
    model.eval()
    batch = next(iter(loader)).to(device)
    batch = batch[:n]
    recon, _, _ = model(batch)
    grid = torch.cat([batch, recon], dim=0)
    save_image(grid, path, nrow=n, padding=2)


def save_checkpoint(
    path: Path,
    model: ConvVAE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: list[dict],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
            "metrics": metrics,
            "architecture": "raffin_conv_vae_pytorch",
        },
        path,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_dataset = VaeMemmapDataset(args.cache_dir, "train", args.max_train_samples)
    val_dataset = VaeMemmapDataset(args.cache_dir, "val", args.max_val_samples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = ConvVAE(z_size=args.z_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = 0
    metrics: list[dict] = []

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        metrics = list(checkpoint.get("metrics", []))
        print(f"Resumed VAE from {args.resume} at epoch {start_epoch}")

    print(
        json.dumps(
            {
                "device": str(device),
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "z_size": args.z_size,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "kl_tolerance": args.kl_tolerance,
                "beta": args.beta,
                "epochs": args.epochs,
                "log_interval": args.log_interval,
            },
            indent=2,
            sort_keys=True,
        )
    )

    best_val = float("inf")
    best_path = args.output_dir / "best.pt"
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            beta=args.beta,
            kl_tolerance=args.kl_tolerance,
            epoch=epoch,
            log_interval=args.log_interval,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            None,
            device,
            beta=args.beta,
            kl_tolerance=args.kl_tolerance,
        )

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        metrics.append(row)
        (args.output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        save_checkpoint(args.output_dir / "last.pt", model, optimizer, epoch, args, metrics)
        save_reconstructions(
            model,
            val_loader,
            device,
            args.output_dir / f"recon_epoch_{epoch:03d}.png",
            args.save_samples,
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(best_path, model, optimizer, epoch, args, metrics)
            save_reconstructions(
                model,
                val_loader,
                device,
                args.output_dir / "recon_best.png",
                args.save_samples,
            )

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} train_r={train_metrics['r_loss']:.4f} train_kl={train_metrics['kl_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_r={val_metrics['r_loss']:.4f} val_kl={val_metrics['kl_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
