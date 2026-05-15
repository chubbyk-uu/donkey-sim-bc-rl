import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def summarize_values(name: str, values: list[float]) -> None:
    if not values:
        print(f"{name}: no values")
        return

    arr = np.asarray(values, dtype=np.float32)
    print(
        f"{name}: count={arr.size} min={arr.min():.4f} p05={percentile(values, 5):.4f} "
        f"mean={arr.mean():.4f} p95={percentile(values, 95):.4f} max={arr.max():.4f}"
    )


def print_angle_histogram(angles: list[float], bins: int, min_angle: float, max_angle: float) -> None:
    if not angles:
        print("angle_histogram: no values")
        return

    arr = np.asarray(angles, dtype=np.float32)
    counts, edges = np.histogram(np.clip(arr, min_angle, max_angle), bins=np.linspace(min_angle, max_angle, bins + 1))
    print(f"angle_histogram bins={bins} range=[{min_angle:.3f}, {max_angle:.3f}]")
    for index, count in enumerate(counts):
        print(f"  {index:02d} [{edges[index]:+.3f}, {edges[index + 1]:+.3f}): {int(count)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path, nargs="+")
    parser.add_argument("--sample-images", type=int, default=5)
    parser.add_argument("--drop-head", type=int, default=0)
    parser.add_argument("--drop-tail", type=int, default=0)
    parser.add_argument("--hist-bins", type=int, default=21)
    parser.add_argument("--hist-min", type=float, default=-0.7)
    parser.add_argument("--hist-max", type=float, default=0.7)
    args = parser.parse_args()

    missing_images = []
    bad_records = []
    angles = []
    throttles = []
    modes = Counter()
    laps = Counter()
    locs = Counter()
    image_samples = []
    records_total = 0
    records_used = 0
    images_total = 0

    for data_dir in [path.expanduser() for path in args.data_dir]:
        if not data_dir.exists():
            raise SystemExit(f"data dir does not exist: {data_dir}")

        print(f"data_dir: {data_dir}")
        meta_path = data_dir / "meta.json"
        if meta_path.exists():
            print(f"meta: {meta_path}")
            print(meta_path.read_text())
        else:
            print("meta: missing")

        all_records = sorted(data_dir.glob("record_*.json"), key=lambda p: int(p.stem.split("_")[1]))
        images = sorted(data_dir.glob("*_cam-image_array_.jpg"), key=lambda p: int(p.name.split("_")[0]))
        end = len(all_records) - args.drop_tail if args.drop_tail else len(all_records)
        records = all_records[args.drop_head : end]
        records_total += len(all_records)
        records_used += len(records)
        images_total += len(images)
        print(f"  records_total: {len(all_records)}")
        print(f"  records_used:  {len(records)}")
        print(f"  images:        {len(images)}")

        for record_path in records:
            try:
                record = json.loads(record_path.read_text())
            except Exception as exc:
                bad_records.append((str(record_path), str(exc)))
                continue

            image_name = record.get("cam/image_array")
            if image_name and not (data_dir / image_name).exists():
                missing_images.append(str(data_dir / image_name))

            if "user/angle" in record:
                angles.append(float(record["user/angle"]))
            if "user/throttle" in record:
                throttles.append(float(record["user/throttle"]))
            if "user/mode" in record:
                modes[str(record["user/mode"])] += 1
            if "track/lap" in record:
                laps[int(record["track/lap"])] += 1
            if "track/loc" in record:
                locs[int(record["track/loc"])] += 1

        image_samples.extend(images[: max(0, args.sample_images - len(image_samples))])

    print(f"records_total_all: {records_total}")
    print(f"records_used_all:  {records_used}")
    print(f"drop_head: {args.drop_head}")
    print(f"drop_tail: {args.drop_tail}")
    print(f"images_all: {images_total}")

    print(f"bad_records: {len(bad_records)}")
    print(f"missing_images: {len(missing_images)}")
    summarize_values("angle", angles)
    print_angle_histogram(angles, args.hist_bins, args.hist_min, args.hist_max)
    summarize_values("throttle", throttles)
    print(f"modes: {dict(modes)}")
    print(f"laps: {dict(laps)}")
    print(f"track_locs_top10: {locs.most_common(10)}")

    for image_path in image_samples[: args.sample_images]:
        with Image.open(image_path) as image:
            print(f"image_sample: {image_path} size={image.size} mode={image.mode}")

    if bad_records:
        print("bad_record_examples:")
        for name, err in bad_records[:5]:
            print(f"  {name}: {err}")
    if missing_images:
        print("missing_image_examples:")
        for name in missing_images[:5]:
            print(f"  {name}")


if __name__ == "__main__":
    main()
