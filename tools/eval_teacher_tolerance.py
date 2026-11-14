"""Stage 3.2: Empirical closed-loop tolerance screen for the frozen `sim_state` teacher.

Tests the frozen ground-truth teacher (`sac_sim_state_60000_steps.zip`) paired on
the same randomized layouts under:
  1. `baseline`            Exact TrackState.encode, 0 delay
  2. `delay1_only`         1-step geometry buffer delay (motion [3:6] & history current)
  3. `shared64_lags_1x_d1` 1-step geom delay + 1.0x held-out residual blocks from `shared64_lags_mlp`
  4. `raw6144_1f_1x_d1`    1-step geom delay + 1.0x held-out residual blocks from `raw6144_ridge (1f)`
  5. `pca128_1f_1x_d1`     1-step geom delay + 1.0x held-out residual blocks from `pca128_ridge (1f)`
  6. `pca128_1f_1.5x_d1`   1-step geom delay + 1.5x held-out residual blocks from `pca128_ridge (1f)`

Residuals are contiguous multi-step 15D segments sampled from the held-out OOF
predictions of Stage 3.1 (`oof_geometry_predictions_lags.npz`), preserving cross-dim
and temporal correlation. Heading error residuals are applied via circular angle
addition so (sin, cos) remains strictly on the unit circle.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import gym_donkeycar  # noqa: F401
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from rl.track_state import TrackState  # noqa: E402
from rl.train_vae_sac import DonkeyVaeSACEnv, RaffinRewardConfig, reload_scene  # noqa: E402


def load_residual_blocks(
    probe_dir: Path,
    pred_key: str,
    min_block_len: int = 80,
) -> list[dict[str, np.ndarray]]:
    """Extract contiguous held-out residual sequences for `pred_key`."""
    cache = np.load(probe_dir / "feature_cache.npz", allow_pickle=False)
    oof = np.load(probe_dir / "oof_geometry_predictions_lags.npz", allow_pickle=False)

    valid = oof["valid_lags_mask"].astype(bool)
    true_geom = oof["true_geom"]  # (N, 15): [0]=cte/2, [1:3]=sin/cos, [3:15]=lookahead
    pred_geom = oof[pred_key]     # (N, 15)

    run = cache["run"]
    layout = cache["layout"]
    model = cache["model"]
    step = cache["step"]

    # Precompute additive residuals for cte/2 & lookahead, and circular angle residual for heading
    d_cte_half = (pred_geom[:, 0] - true_geom[:, 0]).astype(np.float32)
    d_lookahead = (pred_geom[:, 3:15] - true_geom[:, 3:15]).astype(np.float32)

    sin_t, cos_t = true_geom[:, 1], true_geom[:, 2]
    norm_p = np.maximum(np.hypot(pred_geom[:, 1], pred_geom[:, 2]), 1e-6)
    sin_p, cos_p = pred_geom[:, 1] / norm_p, pred_geom[:, 2] / norm_p
    d_theta = np.arctan2(sin_p * cos_t - cos_p * sin_t, sin_p * sin_t + cos_p * cos_t).astype(np.float32)

    by_ep = defaultdict(list)
    for idx in np.flatnonzero(valid):
        by_ep[(run[idx], layout[idx], model[idx])].append(idx)

    blocks = []
    for indices in by_ep.values():
        indices.sort(key=lambda i: int(step[i]))
        # Split at any non-contiguous step gap
        start = 0
        for pos in range(1, len(indices) + 1):
            if pos == len(indices) or int(step[indices[pos]]) != int(step[indices[pos - 1]]) + 1:
                seg = indices[start:pos]
                if len(seg) >= min_block_len:
                    blocks.append({
                        "d_cte_half": d_cte_half[seg],
                        "d_theta": d_theta[seg],
                        "d_lookahead": d_lookahead[seg],
                    })
                start = pos
    if not blocks:
        raise RuntimeError(f"No contiguous residual blocks >= {min_block_len} found for {pred_key}")
    return blocks


class PerturbedTrackStateEncoder:
    """Wraps TrackState.encode with optional 15D residual block replay and 1-step geometry delay."""

    uses_telemetry = True
    z_size = 18

    def __init__(
        self,
        track: TrackState,
        blocks: list[dict[str, np.ndarray]] | None = None,
        scale: float = 0.0,
        delay_geom_steps: int = 0,
        block_len: int = 80,
        seed: int = 0,
    ) -> None:
        self.track = track
        self.blocks = blocks
        self.scale = float(scale)
        self.delay_geom_steps = int(delay_geom_steps)
        self.block_len = int(block_len)
        self.rng = np.random.default_rng(seed)
        self._prev_geom: tuple[float, float, float, np.ndarray] | None = None
        self._cur_block: dict[str, np.ndarray] | None = None
        self._cur_offset: int = 0
        self._cur_end: int = 0
        self.injected_cte_errs: list[float] = []
        self.injected_head_degs: list[float] = []

    def reset_episode(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self._prev_geom = None
        self._cur_block = None
        self._cur_offset = 0
        self._cur_end = 0
        self.injected_cte_errs.clear()
        self.injected_head_degs.clear()

    def _next_residual(self) -> tuple[float, float, np.ndarray]:
        if not self.blocks or self.scale == 0.0:
            return 0.0, 0.0, np.zeros(12, dtype=np.float32)
        if self._cur_block is None or self._cur_offset >= self._cur_end:
            blk = self.blocks[int(self.rng.integers(0, len(self.blocks)))]
            n = len(blk["d_cte_half"])
            length = min(self.block_len, n)
            start = int(self.rng.integers(0, n - length + 1))
            self._cur_block = blk
            self._cur_offset = start
            self._cur_end = start + length
        k = self._cur_offset
        self._cur_offset += 1
        return (
            float(self._cur_block["d_cte_half"][k]) * self.scale,
            float(self._cur_block["d_theta"][k]) * self.scale,
            self._cur_block["d_lookahead"][k] * self.scale,
        )

    def encode(self, image, info: dict) -> np.ndarray:
        raw = self.track.encode(image, info)  # (18,)
        d_cte_half, d_theta, d_la = self._next_residual()

        cte_half = float(raw[0]) + d_cte_half
        theta_true = float(np.arctan2(raw[1], raw[2]))
        theta_pert = theta_true + d_theta
        sin_p, cos_p = float(np.sin(theta_pert)), float(np.cos(theta_pert))
        la_pert = (raw[6:18] + d_la).astype(np.float32)

        cur_geom = (cte_half, sin_p, cos_p, la_pert)
        if self.delay_geom_steps > 0:
            if self._prev_geom is None:
                used_geom = cur_geom
            else:
                used_geom = self._prev_geom
            self._prev_geom = cur_geom
        else:
            used_geom = cur_geom

        # Track actual effective error seen by the controller (including delay + residual)
        eff_cte_err = (used_geom[0] - float(raw[0])) * 2.0
        eff_theta = float(np.arctan2(used_geom[1], used_geom[2]))
        eff_head_deg = float(np.degrees(np.arctan2(np.sin(eff_theta - theta_true), np.cos(eff_theta - theta_true))))
        self.injected_cte_errs.append(eff_cte_err)
        self.injected_head_degs.append(eff_head_deg)

        out = raw.copy()
        out[0] = used_geom[0]
        out[1] = used_geom[1]
        out[2] = used_geom[2]
        out[6:18] = used_geom[3]
        # Motion [3:6] stays at current step `raw[3:6]`!
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="172.25.240.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--teacher", type=Path, default=Path("models/rl_loop_sim_state_ts2_sdiff01_v1/sac_sim_state_60000_steps.zip"))
    parser.add_argument("--track-centerline", type=Path, default=Path("models/track_centerline/generated_track.npz"))
    parser.add_argument("--probe-dir", type=Path, default=Path("logs/probe_geometry_15d_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/teacher_tolerance_screen_v1"))
    parser.add_argument("--layouts", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--order-seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    track = TrackState.load(args.track_centerline)
    blocks_shared64 = load_residual_blocks(args.probe_dir, "pred_shared64_lags_mlp", min_block_len=80)
    blocks_raw6144_1f = load_residual_blocks(args.probe_dir, "pred_raw6144_ridge_1f", min_block_len=80)
    blocks_pca128_1f = load_residual_blocks(args.probe_dir, "pred_pca128_ridge_1f", min_block_len=80)

    conditions = [
        ("baseline", None, 0.0, 0),
        ("delay1_only", None, 0.0, 1),
        ("shared64_lags_1x_d1", blocks_shared64, 1.0, 1),
        ("raw6144_1f_1x_d1", blocks_raw6144_1f, 1.0, 1),
        ("pca128_1f_1x_d1", blocks_pca128_1f, 1.0, 1),
        ("pca128_1f_1.5x_d1", blocks_pca128_1f, 1.5, 1),
    ]

    teacher = SAC.load(str(args.teacher), device="cpu")
    dummy_enc = PerturbedTrackStateEncoder(track)

    base_env = gym.make(
        "donkey-generated-track-v0",
        conf={
            "host": args.host,
            "port": args.port,
            "body_style": "donkey",
            "body_rgb": (128, 128, 128),
            "car_name": "tol_screen",
            "font_size": 100,
            "cam_resolution": (120, 160, 3),
            "cam_offset": (0, 0, 0),
            "max_cte": 20.0,
        },
    )
    env = DonkeyVaeSACEnv(
        base_env,
        vae=dummy_enc,
        reward_config=RaffinRewardConfig(
            reward_crash=-20.0,
            crash_speed_weight=5.0,
            alive_reward=1.5,
            speed_reward_weight=0.15,
            cte_speed_penalty_weight=0.25,
            max_cte_error=2.0,
            cte_target=0.0,
        ),
        min_throttle=0.2,
        max_throttle=0.7,
        max_steering=1.0,
        max_steering_diff=0.1,
        n_command_history=20,
        scene_reload_every=0,
    )

    rng = np.random.default_rng(args.order_seed)
    all_episodes = []
    controller = env.env.unwrapped.viewer
    scene = controller.handler.SceneToLoad

    try:
        for layout_idx in range(1, args.layouts + 1):
            print(f"\n=== Reloading scene for Layout {layout_idx}/{args.layouts} ===", flush=True)
            if not reload_scene(controller, scene):
                raise RuntimeError(f"Scene reload timed out on layout {layout_idx}")

            order = list(range(len(conditions)))
            rng.shuffle(order)

            for cond_idx in order:
                cname, blks, scale, d_steps = conditions[cond_idx]
                enc = PerturbedTrackStateEncoder(
                    track=track,
                    blocks=blks,
                    scale=scale,
                    delay_geom_steps=d_steps,
                    block_len=80,
                    seed=layout_idx * 1000 + cond_idx,
                )
                enc.reset_episode(seed=layout_idx * 1000 + cond_idx)
                env.set_observation(enc, latent_lags=())

                obs, info = env.reset()
                steps = 0
                total_reward = 0.0
                abs_ctes = []
                speeds = []
                dsteers = []
                t0 = time.perf_counter()
                term_reason = "truncated"

                while steps < args.max_episode_steps:
                    action, _ = teacher.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    steps += 1
                    total_reward += float(reward)
                    abs_ctes.append(float(info["abs_cte"]))
                    speeds.append(float(info.get("speed", 0.0)))
                    dsteers.append(abs(float(info["rl_steer_delta"])))
                    if terminated:
                        term_reason = str(info.get("termination_reason") or "terminated")
                        break
                    if truncated:
                        break

                elapsed = time.perf_counter() - t0
                laps = int(info.get("lap_count", 0))
                completed = steps >= args.max_episode_steps and term_reason == "truncated"
                eff_cte_p95 = float(np.percentile(np.abs(enc.injected_cte_errs), 95)) if enc.injected_cte_errs else 0.0
                eff_head_p95 = float(np.percentile(np.abs(enc.injected_head_degs), 95)) if enc.injected_head_degs else 0.0

                ep_row = {
                    "layout": layout_idx,
                    "condition": cname,
                    "completed": completed,
                    "steps": steps,
                    "laps": laps,
                    "termination_reason": term_reason,
                    "mean_abs_cte": float(np.mean(abs_ctes)),
                    "max_abs_cte": float(np.max(abs_ctes)),
                    "mean_speed": float(np.mean(speeds)),
                    "mean_abs_dsteer": float(np.mean(dsteers)),
                    "reward": total_reward,
                    "eff_injected_cte_err_p95": eff_cte_p95,
                    "eff_injected_head_deg_p95": eff_head_p95,
                    "wall_seconds": elapsed,
                }
                all_episodes.append(ep_row)
                print(
                    f"  [L{layout_idx} {cname:<20}] {'PASS' if completed else 'FAIL':<4} "
                    f"steps={steps:>4} laps={laps:>2} mean|CTE|={ep_row['mean_abs_cte']:.3f} "
                    f"max|CTE|={ep_row['max_abs_cte']:.3f} speed={ep_row['mean_speed']:.2f} "
                    f"inj_P95(CTE={eff_cte_p95:.2f}, Head={eff_head_p95:.1f}°) ({elapsed:.1f}s)",
                    flush=True,
                )
    finally:
        env.close()

    # Aggregate summary by condition
    summary = {}
    print("\n=== Stage 3.2 Tolerance Screen Summary ===", flush=True)
    print(
        f"{'condition':<21} {'comp':>6} {'mean_steps':>10} {'mean_laps':>9} "
        f"{'mean|CTE|':>9} {'max|CTE|':>8} {'speed':>6} {'inj_CTE_P95':>11} {'inj_Head_P95':>12}",
        flush=True,
    )
    for cname, _, _, _ in conditions:
        rows = [r for r in all_episodes if r["condition"] == cname]
        comp_str = f"{sum(r['completed'] for r in rows)}/{len(rows)}"
        summary[cname] = {
            "completed": sum(r["completed"] for r in rows),
            "episodes": len(rows),
            "mean_steps": float(np.mean([r["steps"] for r in rows])),
            "mean_laps": float(np.mean([r["laps"] for r in rows])),
            "mean_abs_cte": float(np.mean([r["mean_abs_cte"] for r in rows])),
            "max_abs_cte": float(np.max([r["max_abs_cte"] for r in rows])),
            "mean_speed": float(np.mean([r["mean_speed"] for r in rows])),
            "mean_inj_cte_p95": float(np.mean([r["eff_injected_cte_err_p95"] for r in rows])),
            "mean_inj_head_p95": float(np.mean([r["eff_injected_head_deg_p95"] for r in rows])),
        }
        s = summary[cname]
        print(
            f"{cname:<21} {comp_str:>6} {s['mean_steps']:>10.1f} {s['mean_laps']:>9.2f} "
            f"{s['mean_abs_cte']:>9.3f} {s['max_abs_cte']:>8.3f} {s['mean_speed']:>6.2f} "
            f"{s['mean_inj_cte_p95']:>11.3f} {s['mean_inj_head_p95']:>11.2f}°",
            flush=True,
        )

    out_payload = {"layouts": args.layouts, "summary": summary, "episodes": all_episodes}
    (args.output_dir / "tolerance_results.json").write_text(json.dumps(out_payload, indent=2) + "\n")
    print(f"Saved tolerance results to {args.output_dir / 'tolerance_results.json'}", flush=True)


if __name__ == "__main__":
    main()
