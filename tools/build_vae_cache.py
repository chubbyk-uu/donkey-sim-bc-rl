from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a uint8 memmap cache for Raffin-style Donkey VAE training.")
    parser.add_argument("--manifest", type=Path, default=Path("data/vae/manifest.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/vae/cache_raffin"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--camera-width", type=int, default=160)
    parser.add_argument("--camera-height", type=int, default=120)
    parser.add_argument(
        "--margin-top",
        type=int,
        default=None,
        help="Top crop. Defaults to camera_height // 3, matching Raffin config.py.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_crop(path: str, camera_width: int, camera_height: int, margin_top: int) -> tuple[int, np.ndarray]:
    image_path = Path(path)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if image.size != (camera_width, camera_height):
            raise ValueError(f"{image_path} has size {image.size}, expected {(camera_width, camera_height)}")
        array = np.asarray(image, dtype=np.uint8)
    return 0, array[margin_top:, :, :].copy()


def _load_crop_job(args: tuple[int, str, int, int, int]) -> tuple[int, np.ndarray]:
    idx, path, camera_width, camera_height, margin_top = args
    _, image = load_crop(path, camera_width, camera_height, margin_top)
    return idx, image


def main() -> None:
    args = parse_args()
    margin_top = args.camera_height // 3 if args.margin_top is None else args.margin_top
    image_height = args.camera_height - margin_top
    image_width = args.camera_width

    rows = read_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = args.output_dir / "images_uint8.dat"
    images = np.memmap(
        cache_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(rows), image_height, image_width, 3),
    )

    jobs = (
        (idx, row["path"], args.camera_width, args.camera_height, margin_top)
        for idx, row in enumerate(rows)
    )
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            iterator = executor.map(_load_crop_job, jobs, chunksize=256)
            for idx, image in iterator:
                images[idx] = image
    else:
        for job in jobs:
            idx, image = _load_crop_job(job)
            images[idx] = image

    images.flush()

    meta = {
        "manifest": args.manifest.as_posix(),
        "cache": cache_path.as_posix(),
        "num_images": len(rows),
        "dtype": "uint8",
        "shape": [len(rows), image_height, image_width, 3],
        "camera_width": args.camera_width,
        "camera_height": args.camera_height,
        "margin_top": margin_top,
        "roi": [0, margin_top, image_width, image_height],
        "raffin_config_match": True,
    }
    (args.output_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
