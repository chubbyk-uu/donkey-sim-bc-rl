from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an image manifest for Donkey VAE training.")
    parser.add_argument(
        "--source",
        type=Path,
        action="append",
        required=True,
        help="Image root to scan. Pass this multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/vae"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-width", type=int, default=160)
    parser.add_argument("--expected-height", type=int, default=120)
    parser.add_argument("--min-file-bytes", type=int, default=1000)
    parser.add_argument("--min-pixel-std", type=float, default=2.0)
    parser.add_argument("--dedupe", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def iter_image_paths(root: Path) -> list[Path]:
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(paths)


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(
    path: Path,
    expected_width: int,
    expected_height: int,
    min_file_bytes: int,
    min_pixel_std: float,
    dedupe: bool,
) -> tuple[str, bool, dict, str | None]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        return path.as_posix(), False, {}, f"stat_error:{exc}"

    if size < min_file_bytes:
        return path.as_posix(), False, {"bytes": size}, "too_small"

    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            if width != expected_width or height != expected_height:
                return path.as_posix(), False, {"width": width, "height": height, "bytes": size}, "bad_size"
            array = np.asarray(image)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return path.as_posix(), False, {"bytes": size}, f"bad_image:{exc}"

    pixel_mean = float(array.mean())
    pixel_std = float(array.std())
    if pixel_std < min_pixel_std:
        return path.as_posix(), False, {"width": width, "height": height, "bytes": size, "pixel_mean": pixel_mean, "pixel_std": pixel_std}, "low_std"

    stats = {
        "width": width,
        "height": height,
        "bytes": size,
        "pixel_mean": pixel_mean,
        "pixel_std": pixel_std,
    }
    if dedupe:
        stats["sha1"] = file_sha1(path)

    return path.as_posix(), True, stats, None


def source_name(path: Path) -> str:
    parts = path.parts
    if "slow_data" in parts:
        idx = parts.index("slow_data")
        if idx + 1 < len(parts):
            return f"slow_data/{parts[idx + 1]}"
        return "slow_data"
    if "vae_raw" in parts:
        idx = parts.index("vae_raw")
        if idx + 1 < len(parts):
            return f"vae_raw/{parts[idx + 1]}"
        return "vae_raw"
    return path.parent.name


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    accepted: list[dict] = []
    rejected = Counter()
    source_counts = Counter()
    seen_hashes: set[str] = set()
    duplicate_count = 0
    all_paths: list[Path] = []

    for root in args.source:
        root = root.resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        all_paths.extend(iter_image_paths(root))

    jobs = (
        (
            path,
            args.expected_width,
            args.expected_height,
            args.min_file_bytes,
            args.min_pixel_std,
            args.dedupe,
        )
        for path in all_paths
    )
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(_inspect_image_from_tuple, jobs, chunksize=256)
            rows = list(results)
    else:
        rows = [_inspect_image_from_tuple(job) for job in jobs]

    path_by_posix = {path.as_posix(): path for path in all_paths}
    for image_path_str, ok, stats, reason in rows:
        image_path = path_by_posix[image_path_str]
        if not ok:
            rejected[reason or "unknown"] += 1
            continue

        sha1 = stats.pop("sha1", None)
        if args.dedupe:
            if sha1 in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(sha1)

        try:
            rel_path = image_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            rel_path = image_path.resolve().as_posix()

        src = source_name(image_path)
        row = {
            "path": rel_path,
            "source": src,
            **stats,
        }
        if sha1 is not None:
            row["sha1"] = sha1
        accepted.append(row)
        source_counts[src] += 1

    rng.shuffle(accepted)
    val_count = int(round(len(accepted) * args.val_ratio))
    for idx, row in enumerate(accepted):
        row["split"] = "val" if idx < val_count else "train"

    accepted.sort(key=lambda row: (row["split"], row["source"], row["path"]))
    train_rows = [row for row in accepted if row["split"] == "train"]
    val_rows = [row for row in accepted if row["split"] == "val"]

    write_jsonl(args.output_dir / "manifest.jsonl", accepted)
    write_jsonl(args.output_dir / "train_manifest.jsonl", train_rows)
    write_jsonl(args.output_dir / "val_manifest.jsonl", val_rows)

    summary = {
        "total": len(accepted),
        "scanned": len(all_paths),
        "train": len(train_rows),
        "val": len(val_rows),
        "val_ratio": args.val_ratio,
        "sources": dict(sorted(source_counts.items())),
        "rejected": dict(sorted(rejected.items())),
        "duplicates": duplicate_count,
        "dedupe": args.dedupe,
        "workers": args.workers,
        "expected_size": [args.expected_width, args.expected_height],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _inspect_image_from_tuple(args: tuple) -> tuple[str, bool, dict, str | None]:
    return inspect_image(*args)


if __name__ == "__main__":
    main()
