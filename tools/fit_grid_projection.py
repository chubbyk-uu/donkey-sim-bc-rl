"""Fit the fixed projection used by the dinov2_vits14_grid encoder.

Collects every captured frame (capture_*/step_*.png and initial_*.png) of the
given evaluation runs, computes DINOv2 patch tokens pooled to a 4x4 grid, and
keeps the leading principal components, whitened. The result is a small file
that training and evaluation load with --encoder-projection.

The frames only need to cover what the camera sees on this track under the
randomisation of interest; they are not labelled and need not come from the
policy that will be trained.

Usage:
    python tools/fit_grid_projection.py logs/<run> [logs/<run> ...] \
        --components 128 --output models/dinov2_grid_projection/pca128.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rl.train_vae_sac import FrozenDinoGridEncoder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--crop-top", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    paths = sorted(path for run in args.runs for pattern in ("capture_*/step_*.png", "initial_*.png")
                   for path in run.glob(pattern))
    if len(paths) <= args.components:
        raise SystemExit(f"Need more than {args.components} frames, found {len(paths)}")
    encoder = FrozenDinoGridEncoder(torch.device(args.device), crop_top=args.crop_top)
    features = []
    for start in range(0, len(paths), 128):
        images = np.stack([np.asarray(Image.open(path).convert("RGB")) for path in paths[start:start + 128]])
        features.append(encoder.grid_features(images).double())
        print(f"encoded {min(start + 128, len(paths))}/{len(paths)} frames", flush=True)
    features = torch.cat(features)
    mean = features.mean(0)
    centred = features - mean
    variance, vectors = torch.linalg.eigh(centred.T @ centred / (len(features) - 1))  # ascending
    variance, vectors = variance.flip(0)[:args.components], vectors.flip(1)[:, :args.components]
    explained = float(variance.sum() / centred.var(0, unbiased=True).sum())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mean": mean.float().cpu(), "components": vectors.T.float().contiguous().cpu(),
        "scale": variance.sqrt().float().cpu(),
        "frames": len(paths), "explained_variance": explained, "crop_top": args.crop_top,
        "runs": [str(run) for run in args.runs],
    }, args.output)
    print(f"{len(paths)} frames from {len(args.runs)} runs; {args.components} components keep "
          f"{explained:.1%} of the variance -> {args.output}")


if __name__ == "__main__":
    main()
