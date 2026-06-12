"""Probe: run off-the-shelf depth + road segmentation on sim frames.

Purpose: judge the sim->real domain gap before investing in a task-adapted
encoder (experiment-log future-work item). If a Cityscapes-pretrained
segmenter can isolate the road surface (and ignore shadows painted on it),
the depth/seg latent line is worth pursuing.

Usage:
    python tools/probe_depth_seg.py FRAME [FRAME ...] --out logs/probe_depth_seg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    SegformerForSemanticSegmentation,
)

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
SEG_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
CITYSCAPES_ROAD_ID = 0

# Cityscapes train-id palette (19 classes), enough to eyeball the seg map.
PALETTE = np.array(
    [
        [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
        [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
        [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
        [0, 80, 100], [0, 0, 230], [119, 11, 32],
    ],
    dtype=np.uint8,
)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    d = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    # simple turbo-ish ramp: near=warm, far=cool
    r = np.clip(1.5 - np.abs(2.0 * d - 0.5) * 2.0, 0, 1)
    g = np.clip(1.5 - np.abs(2.0 * d - 1.0) * 2.0, 0, 1)
    b = np.clip(1.5 - np.abs(2.0 * d - 1.5) * 2.0, 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("logs/probe_depth_seg"))
    parser.add_argument("--infer-width", type=int, default=518,
                        help="upscale frames to this width before inference")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    depth_proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(device).eval()
    seg_proc = AutoImageProcessor.from_pretrained(SEG_MODEL)
    seg_model = SegformerForSemanticSegmentation.from_pretrained(SEG_MODEL).to(device).eval()

    for frame_path in args.frames:
        img = Image.open(frame_path).convert("RGB")
        w, h = img.size
        scale = args.infer_width / w
        big = img.resize((args.infer_width, int(h * scale)), Image.BILINEAR)

        with torch.no_grad():
            d_in = depth_proc(images=big, return_tensors="pt").to(device)
            depth = depth_model(**d_in).predicted_depth[0].cpu().numpy()

            s_in = seg_proc(images=big, return_tensors="pt").to(device)
            logits = seg_model(**s_in).logits
            logits = torch.nn.functional.interpolate(
                logits, size=big.size[::-1], mode="bilinear", align_corners=False
            )
            seg = logits.argmax(dim=1)[0].cpu().numpy()

        depth_img = Image.fromarray(colorize_depth(depth)).resize(big.size, Image.BILINEAR)
        seg_img = Image.fromarray(PALETTE[seg % len(PALETTE)])

        road = (seg == CITYSCAPES_ROAD_ID)
        overlay = np.array(big).copy()
        overlay[road] = (0.4 * overlay[road] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
        road_img = Image.fromarray(overlay)

        panel = Image.new("RGB", (big.width * 4, big.height))
        for i, im in enumerate([big, depth_img, seg_img, road_img]):
            panel.paste(im, (i * big.width, 0))

        out_path = args.out / f"{frame_path.stem}_probe.png"
        panel.save(out_path)
        print(f"{frame_path.name}: road={road.mean():.1%} of pixels -> {out_path}")


if __name__ == "__main__":
    main()
