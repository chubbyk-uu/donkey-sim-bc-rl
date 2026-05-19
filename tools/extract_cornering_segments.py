import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def numeric_record_key(path: Path) -> int:
    return int(path.stem.split("_")[1])


def numeric_dir_key(path: Path) -> tuple[int, str]:
    return (int(path.name), path.name) if path.name.isdigit() else (10**9, path.name)


def load_records(tub_dir: Path) -> list[dict]:
    records = []
    for record_path in sorted(tub_dir.glob("record_*.json"), key=numeric_record_key):
        record = json.loads(record_path.read_text())
        records.append(
            {
                "path": record_path,
                "source_index": numeric_record_key(record_path),
                "record": record,
                "angle": float(record["user/angle"]),
                "throttle": float(record["user/throttle"]),
            }
        )
    return records


def find_intervals(mask: np.ndarray, margin: int, gap: int) -> list[tuple[int, int]]:
    positions = np.flatnonzero(mask)
    if len(positions) == 0:
        return []

    raw_intervals = []
    start = int(positions[0])
    previous = int(positions[0])
    for position in positions[1:]:
        position = int(position)
        if position - previous > gap:
            raw_intervals.append((start, previous))
            start = position
        previous = position
    raw_intervals.append((start, previous))

    expanded = []
    n = len(mask)
    for start, end in raw_intervals:
        expanded.append((max(0, start - margin), min(n - 1, end + margin)))

    merged: list[list[int]] = []
    for start, end in expanded:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def split_contiguous(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    runs = [[indices[0]]]
    for index in indices[1:]:
        if index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def copy_segment(
    records: list[dict],
    indices: list[int],
    source_tub: Path,
    output_tub: Path,
    min_segment_frames: int,
    trigger_abs_angle: float,
    min_trigger_frames: int,
) -> dict | None:
    if len(indices) < min_segment_frames:
        return None
    trigger_count = sum(abs(records[index]["angle"]) >= trigger_abs_angle for index in indices)
    if trigger_count < min_trigger_frames:
        return None

    output_tub.mkdir(parents=True, exist_ok=False)
    angles = []
    throttles = []
    source_indices = []

    for new_index, record_index in enumerate(indices):
        item = records[record_index]
        record = dict(item["record"])
        image_name = record["cam/image_array"]
        source_image = source_tub / image_name
        output_image = output_tub / f"{new_index}_cam-image_array_.jpg"
        output_record = output_tub / f"record_{new_index}.json"

        if not source_image.exists():
            raise FileNotFoundError(source_image)

        shutil.copy2(source_image, output_image)
        record["cam/image_array"] = output_image.name
        record["_source_tub"] = str(source_tub)
        record["_source_record"] = str(item["path"])
        record["_source_index"] = item["source_index"]
        output_record.write_text(json.dumps(record, indent=2))

        angles.append(item["angle"])
        throttles.append(item["throttle"])
        source_indices.append(item["source_index"])

    angle_arr = np.asarray(angles, dtype=np.float32)
    throttle_arr = np.asarray(throttles, dtype=np.float32)
    return {
        "output_tub": str(output_tub),
        "source_tub": str(source_tub),
        "frames": len(indices),
        "source_start": int(source_indices[0]),
        "source_end": int(source_indices[-1]),
        "angle_min": float(angle_arr.min()),
        "angle_max": float(angle_arr.max()),
        "angle_mean": float(angle_arr.mean()),
        "angle_std": float(angle_arr.std()),
        "throttle_min": float(throttle_arr.min()),
        "throttle_max": float(throttle_arr.max()),
        "throttle_mean": float(throttle_arr.mean()),
        "count_abs_angle_ge_03": int(np.sum(np.abs(angle_arr) >= 0.3)),
        "count_abs_angle_ge_04": int(np.sum(np.abs(angle_arr) >= 0.4)),
        "count_abs_angle_ge_05": int(np.sum(np.abs(angle_arr) >= 0.5)),
        "count_abs_angle_ge_06": int(np.sum(np.abs(angle_arr) >= 0.6)),
        "count_trigger_abs_angle": int(trigger_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--trigger-abs-angle", type=float, default=0.4)
    parser.add_argument("--max-throttle", type=float, default=0.25)
    parser.add_argument("--margin-frames", type=int, default=30)
    parser.add_argument("--merge-gap-frames", type=int, default=15)
    parser.add_argument("--min-segment-frames", type=int, default=20)
    parser.add_argument("--min-trigger-frames", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.input_root.exists():
        raise SystemExit(f"input root does not exist: {args.input_root}")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.dry_run:
        raise SystemExit(f"output root already exists and is not empty: {args.output_root}")

    tub_dirs = sorted([path for path in args.input_root.iterdir() if path.is_dir()], key=numeric_dir_key)
    if not tub_dirs:
        raise SystemExit(f"no tub directories in {args.input_root}")

    segment_summaries = []
    skipped_short_segments = 0
    skipped_no_trigger_segments = 0
    source_summary = []
    next_tub_index = 1

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)

    for source_tub in tub_dirs:
        records = load_records(source_tub)
        if not records:
            continue

        angles = np.asarray([record["angle"] for record in records], dtype=np.float32)
        throttles = np.asarray([record["throttle"] for record in records], dtype=np.float32)
        intervals = find_intervals(
            np.abs(angles) >= args.trigger_abs_angle,
            margin=args.margin_frames,
            gap=args.merge_gap_frames,
        )

        selected_indices: list[int] = []
        for start, end in intervals:
            for index in range(start, end + 1):
                if throttles[index] <= args.max_throttle:
                    selected_indices.append(index)

        runs = split_contiguous(selected_indices)
        source_kept = 0
        source_segments = 0
        for run in runs:
            if len(run) < args.min_segment_frames:
                skipped_short_segments += 1
                continue
            trigger_count = sum(abs(records[index]["angle"]) >= args.trigger_abs_angle for index in run)
            if trigger_count < args.min_trigger_frames:
                skipped_no_trigger_segments += 1
                continue
            source_segments += 1
            source_kept += len(run)
            if args.dry_run:
                continue
            output_tub = args.output_root / f"corner_{next_tub_index:03d}"
            summary = copy_segment(
                records,
                run,
                source_tub,
                output_tub,
                args.min_segment_frames,
                args.trigger_abs_angle,
                args.min_trigger_frames,
            )
            if summary is not None:
                segment_summaries.append(summary)
                next_tub_index += 1

        source_summary.append(
            {
                "source_tub": str(source_tub),
                "records": len(records),
                "trigger_intervals": len(intervals),
                "segments": source_segments,
                "kept_frames": source_kept,
                "source_count_abs_angle_ge_04": int(np.sum(np.abs(angles) >= 0.4)),
                "source_count_abs_angle_ge_05": int(np.sum(np.abs(angles) >= 0.5)),
                "source_count_abs_angle_ge_06": int(np.sum(np.abs(angles) >= 0.6)),
            }
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "source_summary": source_summary,
                    "skipped_short_segments": skipped_short_segments,
                    "skipped_no_trigger_segments": skipped_no_trigger_segments,
                },
                indent=2,
            )
        )
        return

    all_angles = []
    all_throttles = []
    for segment in segment_summaries:
        for record_path in Path(segment["output_tub"]).glob("record_*.json"):
            record = json.loads(record_path.read_text())
            all_angles.append(float(record["user/angle"]))
            all_throttles.append(float(record["user/throttle"]))

    angle_arr = np.asarray(all_angles, dtype=np.float32)
    throttle_arr = np.asarray(all_throttles, dtype=np.float32)
    summary = {
        "config": {
            "input_root": str(args.input_root),
            "output_root": str(args.output_root),
            "trigger_abs_angle": args.trigger_abs_angle,
            "max_throttle": args.max_throttle,
            "margin_frames": args.margin_frames,
            "merge_gap_frames": args.merge_gap_frames,
            "min_segment_frames": args.min_segment_frames,
            "min_trigger_frames": args.min_trigger_frames,
        },
        "source_summary": source_summary,
        "segments": segment_summaries,
        "skipped_short_segments": skipped_short_segments,
        "skipped_no_trigger_segments": skipped_no_trigger_segments,
        "total_segments": len(segment_summaries),
        "total_frames": int(len(angle_arr)),
        "angle_min": float(angle_arr.min()) if len(angle_arr) else None,
        "angle_max": float(angle_arr.max()) if len(angle_arr) else None,
        "angle_mean": float(angle_arr.mean()) if len(angle_arr) else None,
        "angle_std": float(angle_arr.std()) if len(angle_arr) else None,
        "throttle_min": float(throttle_arr.min()) if len(throttle_arr) else None,
        "throttle_max": float(throttle_arr.max()) if len(throttle_arr) else None,
        "throttle_mean": float(throttle_arr.mean()) if len(throttle_arr) else None,
        "count_abs_angle_ge_03": int(np.sum(np.abs(angle_arr) >= 0.3)),
        "count_abs_angle_ge_04": int(np.sum(np.abs(angle_arr) >= 0.4)),
        "count_abs_angle_ge_05": int(np.sum(np.abs(angle_arr) >= 0.5)),
        "count_abs_angle_ge_06": int(np.sum(np.abs(angle_arr) >= 0.6)),
        "count_abs_angle_ge_07": int(np.sum(np.abs(angle_arr) >= 0.7)),
    }
    (args.output_root / "curation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
