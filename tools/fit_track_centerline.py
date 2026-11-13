"""Recover the centre line of the generated track from evaluation telemetry.

The simulator reports the car's position and its signed cross-track error (CTE)
but not the track itself. Near any spot on the road CTE is a linear function of
position, so a local least-squares fit gives the line where CTE is zero, which
is the centre line. The result lets a policy be given the road ahead as numbers
(`--encoder sim_state`) instead of a camera image.

The track geometry is the same in every layout; only trees and light change.

Usage:
    python tools/fit_track_centerline.py logs/<run> [logs/<run> ...] \
        --output models/track_centerline/generated_track.npz
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SPACING = 0.05  # distance between centre-line points, in position units
RADIUS = 0.35  # neighbourhood of the local fit
SMOOTH = 0.3  # moving-average length along the line


def load(runs: list[Path]) -> dict[str, np.ndarray]:
    rows = defaultdict(list)
    for run in runs:
        with (run / "telemetry.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["speed"] < 0.5:
                    continue
                rows["episode"].append(hash((str(run), row.get("layout"), row.get("model"), row.get("episode"))))
                rows["step"].append(row["step"])
                rows["pos"].append((row["pos"][0], row["pos"][2]))
                rows["cte"].append(row["cte"])
                rows["yaw"].append(row["car"][2])
                rows["lap"].append(row["lap_count"])
    return {name: np.asarray(values) for name, values in rows.items()}


def resample(loop: np.ndarray, spacing: float) -> np.ndarray:
    """Points at equal distance along a closed line (first point not repeated)."""
    closed = np.vstack([loop, loop[:1]])
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    targets = np.arange(0.0, along[-1], along[-1] / round(along[-1] / spacing))
    return np.stack([np.interp(targets, along, closed[:, i]) for i in (0, 1)], axis=1)


def reference_lap(data: dict[str, np.ndarray]) -> np.ndarray:
    """One full driven lap, used only to order the fit along the track."""
    best = None
    for episode in np.unique(data["episode"]):
        mask = data["episode"] == episode
        order = np.argsort(data["step"][mask])
        laps, pos = data["lap"][mask][order], data["pos"][mask][order]
        if laps.max() >= 3:
            lap = pos[laps == 2]
            if best is None or len(lap) > len(best):
                best = lap
    if best is None:
        raise SystemExit("No episode with three completed laps in the given runs")
    return resample(best, SPACING)


def fit(data: dict[str, np.ndarray], reference: np.ndarray) -> tuple[np.ndarray, float]:
    tangent = np.roll(reference, -1, axis=0) - np.roll(reference, 1, axis=0)
    heading = np.degrees(np.arctan2(tangent[:, 0], tangent[:, 1]))
    centre, scales = np.empty_like(reference), []
    for i, point in enumerate(reference):
        offset = data["pos"] - point
        same_direction = np.abs((data["yaw"] - heading[i] + 180.0) % 360.0 - 180.0) < 45.0
        near = same_direction & (np.einsum("ij,ij->i", offset, offset) < RADIUS ** 2)
        if near.sum() < 30:
            raise SystemExit(f"Only {near.sum()} telemetry points near reference point {i}; give more runs")
        design = np.column_stack([offset[near], np.ones(near.sum())])
        (gx, gz, here), *_ = np.linalg.lstsq(design, data["cte"][near], rcond=None)
        gradient = np.array([gx, gz])
        centre[i] = point - here * gradient / gradient.dot(gradient)
        scales.append(np.linalg.norm(gradient))
    window = max(1, round(SMOOTH / SPACING)) | 1
    padded = np.vstack([centre[-(window // 2):], centre, centre[:window // 2]])
    kernel = np.ones(window) / window
    smooth = np.stack([np.convolve(padded[:, i], kernel, mode="valid") for i in (0, 1)], axis=1)
    return resample(smooth, SPACING), float(np.median(scales))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = load(args.runs)
    centre, cte_per_unit = fit(data, reference_lap(data))

    # Check against every telemetry point: the signed distance to the fitted line,
    # times the scale, should reproduce the simulator's CTE.
    from rl.track_state import TrackState  # noqa: PLC0415  (import after the fit so the tool has no cycle)
    track = TrackState(centre, cte_per_unit)
    sample = np.random.default_rng(0).choice(len(data["cte"]), size=min(20000, len(data["cte"])), replace=False)
    predicted = np.array([track.locate(data["pos"][i], data["yaw"][i])[1] * cte_per_unit for i in sample])
    truth = data["cte"][sample]
    sign = 1.0 if np.corrcoef(predicted, truth)[0, 1] > 0 else -1.0
    error = sign * predicted - truth
    report = {
        "runs": [str(run) for run in args.runs], "telemetry_points": int(len(data["cte"])),
        "centre_points": int(len(centre)), "track_length": float(len(centre) * SPACING),
        "cte_per_position_unit": cte_per_unit, "cte_sign_for_right_of_centre": sign,
        "cte_check_r2": float(1 - np.sum(error ** 2) / np.sum((truth - truth.mean()) ** 2)),
        "cte_check_mae": float(np.abs(error).mean()), "cte_check_p99_abs_error": float(np.percentile(np.abs(error), 99)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, centre=centre.astype(np.float32), cte_per_unit=np.float32(cte_per_unit),
             cte_sign=np.float32(sign), report=json.dumps(report))
    print(json.dumps(report, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
