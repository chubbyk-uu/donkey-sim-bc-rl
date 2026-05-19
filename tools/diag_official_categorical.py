import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bc"))

from train_bc import split_indices
from train_bc_official_categorical import (
    OfficialCategoricalDonkeyDataset,
    OfficialCategoricalModel,
    collect_predictions,
    make_bin_centers,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    cfg = ckpt["config"]
    steering_centers = make_bin_centers(cfg["steering_bins"], cfg["steering_min"], cfg["steering_max"])
    throttle_centers = make_bin_centers(cfg["throttle_bins"], cfg["throttle_min"], cfg["throttle_max"])

    val_dataset_full = OfficialCategoricalDonkeyDataset(
        args.data_dir,
        steering_centers=steering_centers,
        throttle_centers=throttle_centers,
        drop_head=cfg.get("drop_head", 0),
        drop_tail=cfg.get("drop_tail", 0),
        cache_images=False,
        min_throttle=cfg.get("min_throttle"),
        max_abs_angle=cfg.get("max_abs_angle"),
        flip_prob=0.0,
    )
    _, val_indices = split_indices(len(val_dataset_full), args.val_split, args.seed)
    val_dataset = Subset(val_dataset_full, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = OfficialCategoricalModel(cfg["steering_bins"], cfg["throttle_bins"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    diag = collect_predictions(model, val_loader, device, steering_centers, throttle_centers)
    diag["steering_centers"] = steering_centers.tolist()
    diag["throttle_centers"] = throttle_centers.tolist()
    diag["val_samples"] = len(val_dataset)

    print("=== diagnostics ===")
    print(json.dumps(diag, indent=2))

    if args.output:
        args.output.write_text(json.dumps(diag, indent=2))
        print(f"saved to {args.output}")

    # per-class accuracy
    print("\n=== per-class breakdown (steering) ===")
    pc = diag["pred_steer_class_counts"]
    tc = diag["true_steer_class_counts"]
    centers = diag["steering_centers"]
    print(f"{'idx':>3} {'center':>8} {'true':>7} {'pred':>7} {'diff':>7}")
    for i, (c, t, p) in enumerate(zip(centers, tc, pc)):
        diff = p - t
        marker = "  <-- " if abs(diff) > max(5, t * 0.3) else ""
        print(f"{i:>3} {c:>+8.3f} {t:>7d} {p:>7d} {diff:>+7d}{marker}")


if __name__ == "__main__":
    main()
