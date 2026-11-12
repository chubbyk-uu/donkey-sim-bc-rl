"""Summarize post-action telemetry and captured episode tails without the sim."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def key(row):
    return row.get("layout"), row.get("model"), row.get("episode")


def percentile(rows, field, q):
    values = [r[field] for r in rows if r.get(field) is not None]
    return float(np.percentile(values, q)) if values else None


def contact_sheet(run, result, rows, output):
    capture = result.get("capture")
    if not capture:
        return None
    folder = run / capture["directory"]
    frames = json.loads((folder / "index.json").read_text())["frames"][-60:]
    chosen = np.unique(np.linspace(0, len(frames) - 1, min(12, len(frames)), dtype=int))
    lookup = {r["step"]: r for r in rows}
    sheet = Image.new("RGB", (1280, ((len(chosen) + 3) // 4) * 278), "white")
    draw = ImageDraw.Draw(sheet)
    for cell, idx in enumerate(chosen):
        frame = frames[idx]
        row = lookup[frame["step"]]
        x, y = (cell % 4) * 320, (cell // 4) * 278
        with Image.open(folder / frame["file"]) as image:
            sheet.paste(image.convert("RGB").resize((320, 240)), (x, y))
        caption = f"step {row['step']}   CTE {row['cte']:+.2f}   v {row['speed']:.2f}"
        draw.text((x + 4, y + 241), caption, fill="black")
        draw.text((x + 4, y + 257), f"mapped steer {row['rl_steer']:+.2f} (preceding command)", fill="black")
    sheet.save(output)
    return output.name


def analyze(run: Path, output: Path, window_steps: int = 120):
    if window_steps < 2:
        raise ValueError("window_steps must be >= 2")
    metadata = json.loads((run / "metadata.json").read_text())
    if metadata["status"] != "completed":
        raise ValueError("Analyze a completed run; partial traces must be reviewed separately")
    results = json.loads((run / "results.json").read_text())["episodes"]
    grouped = defaultdict(list)
    for line in (run / "telemetry.jsonl").read_text().splitlines():
        row = json.loads(line)
        grouped[key(row)].append(row)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    args = metadata["args"]
    steer_max = args.get("max_steering", 1.)
    delta_max = (args.get("max_steering_diff", .2) - 1e-5) * 2 * steer_max
    cte_target = args.get("cte_target", 0.)
    cte_limit = args.get("max_cte_error", 2.)
    summaries = []
    for number, result in enumerate(results, 1):
        trace = grouped[key(result)]
        if len(trace) != result["steps"] or [r["step"] for r in trace] != list(range(1, len(trace) + 1)):
            raise ValueError(f"Incomplete trace: {key(result)}")
        rows = trace[-window_steps:]
        t = np.array([r["elapsed_s"] for r in rows]);t -= t[-1]
        dt = np.diff([r["elapsed_s"] for r in trace])
        cte = np.array([r["cte"] - cte_target for r in rows])
        executed = np.array([r["rl_steer"] for r in rows])
        requested = np.array([r["action"][0] for r in rows])
        success = bool(result.get("trunc", result.get("outcome", "").startswith("TRUNC")))
        name = f"{number:02d}_{'limit' if success else 'out'}"
        summary = {
            "layout": result.get("layout"), "model": result.get("model"), "episode": result.get("episode"),
            "steps": len(trace), "trunc": success, "reason": result["termination_reason"],
            "end_cte_error": float(cte[-1]), "tail_mean_speed": float(np.mean([r["speed"] for r in rows])),
            "tail_steering_saturation_fraction": float(np.mean(np.abs(executed) >= .95 * steer_max)),
            "tail_rate_limit_fraction": float(np.mean([abs(r["rl_steer_delta"]) >= delta_max - .0001 for r in rows])),
            "tail_requested_executed_gap": float(np.mean(np.abs(requested - executed))),
            "max_step_interval_ms": float(max(dt) * 1000) if len(dt) else None,
            "tail_step_interval_p95_ms": float(np.percentile(np.diff(t), 95) * 1000) if len(t) > 1 else None,
            "tail_policy_p95_ms": percentile(rows, "policy_seconds", 95),
            "tail_encoder_p95_ms": percentile(rows, "encoder_seconds", 95),
            "tail_env_step_p95_ms": percentile(rows, "env_step_seconds", 95),
            "plot": f"{name}.png", "contact_sheet": contact_sheet(run, result, trace, output / f"{name}_frames.png"),
        }
        for field in ("tail_policy_p95_ms", "tail_encoder_p95_ms", "tail_env_step_p95_ms"):
            if summary[field] is not None:
                summary[field] *= 1000
        fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
        axes[0].plot(t, cte);axes[0].axhline(cte_limit, color="r", ls="--");axes[0].axhline(-cte_limit, color="r", ls="--")
        axes[0].set_ylabel("CTE - target")
        axes[1].plot(t, requested, label="requested action k", alpha=.6)
        axes[1].plot(t, executed, label="mapped command k")
        axes[1].set_ylabel("Steering");axes[1].legend(loc="upper left")
        axes[2].plot(t, [r["speed"] for r in rows]);axes[2].set_ylabel("Speed")
        axes[3].plot(t[1:], np.diff(t) * 1000, label="record interval")
        for field, label in [("policy_seconds", "policy"), ("encoder_seconds", "encoder"), ("env_step_seconds", "env step incl. encoder")]:
            if all(r.get(field) is not None for r in rows):
                axes[3].plot(t, [r[field] * 1000 for r in rows], label=label)
        axes[3].set_ylabel("Client time (ms)");axes[3].set_xlabel("Seconds relative to episode end")
        axes[3].legend(loc="upper left")
        for ax in axes:ax.grid(alpha=.25)
        fig.suptitle(f"{key(result)}: {'time limit' if success else result['termination_reason']} at step {len(trace)}")
        fig.tight_layout();fig.savefig(output / summary["plot"], dpi=140);plt.close(fig)
        summaries.append(summary)
    artifact = {"run": str(run.resolve()), "window_steps": window_steps, "episodes": summaries,
                "notes": ["Frame/state k is returned by env.step(command k) and is input to the next decision; execution is not acknowledged.",
                          "Client step intervals do not measure network RTT or simulator actuation latency.",
                          "env_step_seconds includes waiting for telemetry and encoding, not only transport."]}
    (output / "summary.json").write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    with (output / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    print(json.dumps({"output": str(output.resolve()), "episodes": len(summaries),
                      "failures": sum(not row["trunc"] for row in summaries)}, indent=2))
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--window-steps", type=int, default=120)
    args = parser.parse_args()
    analyze(args.run, args.output_dir or args.run / "control_analysis", args.window_steps)


if __name__ == "__main__":
    main()
