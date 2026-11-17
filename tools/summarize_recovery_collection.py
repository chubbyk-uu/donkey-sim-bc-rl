"""Audit full-frame Round B captures and report geometry/recovery coverage offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rl.geometry_encoder import GEOM_IDX, MOTION_IDX, SPLIT_DAGGER_TRAIN_POOL  # noqa: E402


def summarize(run: Path) -> dict:
    metadata = json.loads((run / "metadata.json").read_text())
    if metadata["mode"] != "round_b_collect":
        raise ValueError("Expected a round_b_collect run")
    if metadata["condition_control_configs"] != {"teacher_recovery": {"sync_enabled": True, "reversal_unwind_scale": 1.0}}:
        raise ValueError("Collection does not use the frozen sync-only control baseline")
    manifest = json.loads((run / "split_manifest.json").read_text())["files"]
    if manifest.get("locked_eval_holdout"):
        raise ValueError("Recovery collection must contain training data only")
    episodes = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
    files = manifest[SPLIT_DAGGER_TRAIN_POOL]
    if len(files) != len(episodes) or len(episodes) != metadata["layouts"]:
        raise ValueError("Incomplete collection: capture/episode/layout counts differ")

    ctes, speeds, headings, phases = [], [], [], []
    per_episode = []
    max_action_alignment_error = 0.0
    for filename, ep in zip(files, episodes):
        with np.load(run / filename, allow_pickle=False) as cap:
            n = len(cap["step"])
            if str(cap["split"]) != SPLIT_DAGGER_TRAIN_POOL:
                raise ValueError(f"Holdout capture encountered: {filename}")
            if cap["images"].shape != (n, 120, 160, 3) or cap["images"].dtype != np.uint8:
                raise ValueError(f"Invalid full-frame RGB capture: {filename}")
            np.testing.assert_array_equal(cap["step"], np.arange(1, n + 1))
            np.testing.assert_allclose(cap["true_state"][:, 0] * 2, cap["cte"], atol=1e-6)
            np.testing.assert_array_equal(cap["true_geom"], cap["true_state"][:, GEOM_IDX])
            np.testing.assert_array_equal(cap["motion"], cap["true_state"][:, MOTION_IDX])
            np.testing.assert_array_equal(cap["history_post"][:, -2:], cap["student_exec_cmd"])
            np.testing.assert_array_equal(cap["history_pre"][1:], cap["history_post"][:-1])
            # The intervention may alter raw steering, but never throttle or the
            # frozen action mapping; recovery/cruise actions must be the teacher's.
            nominal = cap["recovery_phase"] != "kick"
            np.testing.assert_array_equal(cap["student_raw_action"][nominal], cap["teacher_raw_action_before_intervention"][nominal])
            np.testing.assert_array_equal(cap["student_raw_action"][:, 1], cap["teacher_raw_action_before_intervention"][:, 1])
            delta = cap["student_exec_cmd"][:, 0] - cap["history_pre"][:, -2]
            if np.max(np.abs(delta)) > 0.20001:
                raise ValueError(f"Frozen steering-rate limit exceeded: {filename}")
            if n > 1:
                err = float(np.max(np.abs(cap["teacher_true_raw_action_on_frame"][:-1] - cap["teacher_raw_action_before_intervention"][1:])))
                max_action_alignment_error = max(max_action_alignment_error, err)
                if err > 1e-3:
                    raise ValueError(f"Image/teacher-action timing mismatch in {filename}: {err}")
            cte = cap["cte"]
            ctes.append(cte)
            speeds.append(cap["speed"])
            headings.append(np.degrees(np.arctan2(cap["true_state"][:, 1], cap["true_state"][:, 2])))
            phases.append(cap["recovery_phase"])
            per_episode.append({
                "layout": ep["layout"], "frames": n,
                "max_abs_cte": float(np.max(np.abs(cte))),
                "frames_abs_cte_ge_1": int(np.sum(np.abs(cte) >= 1.0)),
                "startup_frames": min(n, 250), "capture": filename,
                "events": ep["recovery_events"],
            })
    cte, speed, heading, phase = map(np.concatenate, (ctes, speeds, headings, phases))
    bins = [0, 0.3, 0.5, 0.8, 1.0, 1.5, 1.9, float("inf")]
    events = [event for ep in episodes for event in ep["recovery_events"]]
    result = {
        "run": str(run), "layouts": len(episodes), "frames": len(cte),
        "startup_frames": sum(e["startup_frames"] for e in per_episode),
        "cte_units": "simulator CTE units, not metres",
        "abs_cte_bins": {f"[{lo},{hi})": int(np.sum((np.abs(cte) >= lo) & (np.abs(cte) < hi))) for lo, hi in zip(bins, bins[1:])},
        "positive_ge_1": int(np.sum(cte >= 1)), "negative_le_minus_1": int(np.sum(cte <= -1)),
        "speed_p10_p50_p90": np.percentile(speed, [10, 50, 90]).tolist(),
        "abs_heading_p50_p95_deg": np.percentile(np.abs(heading), [50, 95]).tolist(),
        "phase_frames": {str(p): int(np.sum(phase == p)) for p in np.unique(phase)},
        "interventions": len(events), "recovered": sum(e["recovered"] for e in events),
        "terminal_episodes": sum(not e["completed"] for e in episodes),
        "max_teacher_action_alignment_error": max_action_alignment_error,
        "audit_passed": True, "episodes": per_episode,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = summarize(args.run)
    (args.run / "coverage.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
