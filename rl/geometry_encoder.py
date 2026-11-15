"""15-dim road geometry perception head and cache/residual utilities.

Bridges frozen DINOv2 4x4 patch-grid features (6144D) to the 15 geometric
inputs of `TrackState.encode` (rl/track_state.py:70-80):
  - `[0]`:    `cte / 2.0` (where `cte` is in simulator CTE units, limit +/-2.0)
  - `[1:3]`:  `sin(heading_error), cos(heading_error)` (projected to unit circle)
  - `[6:18]`: 6 lookahead points at `LOOKAHEAD = (0.3, 0.6, 1.0, 1.5, 2.0, 3.0)`
              position units, each `(right, forward) / distance`

Motion inputs `[3:6]` (`vel[2]/2, vel[0]*4, gyro[1]/2`) come from live telemetry
and are spliced untouched into the 18-dim teacher state vector.
"""
from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from rl.track_state import LOOKAHEAD, TrackState
from rl.train_vae_sac import FrozenDinoGridEncoder

GEOM_IDX = [0, 1, 2] + list(range(6, 18))  # 15 geometric dims in TrackState.encode
MOTION_IDX = [3, 4, 5]                     # 3 motion dims: vel[2]/2, vel[0]*4, gyro[1]/2


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_cache_fingerprint(
    runs: list[Path],
    centerline_path: Path,
    crop_top: int = 0,
    encoder_name: str = "dinov2_vits14",
) -> dict:
    """Build deterministic SHA256 fingerprint over all inputs of `feature_cache.npz`."""
    if not centerline_path.exists():
        raise FileNotFoundError(f"Track centerline not found: {centerline_path}")
    run_entries = []
    for run in runs:
        res_path = run / "results.json"
        tel_path = run / "telemetry.jsonl"
        if not res_path.exists() or not tel_path.exists():
            raise FileNotFoundError(f"Run {run} missing results.json or telemetry.jsonl")
        episodes = json.loads(res_path.read_text()).get("episodes", [])
        cap_hashes = []
        for ep in episodes:
            cap = ep.get("capture")
            if cap and cap.get("directory"):
                idx_file = run / cap["directory"] / "index.json"
                if idx_file.exists():
                    cap_hashes.append(f"{cap['directory']}:{sha256_file(idx_file)}")
        cap_digest = hashlib.sha256("\n".join(sorted(cap_hashes)).encode("utf-8")).hexdigest()
        run_entries.append({
            "run": run.name,
            "results_sha256": sha256_file(res_path),
            "telemetry_sha256": sha256_file(tel_path),
            "captures_index_sha256": cap_digest,
            "n_captured_episodes": len(cap_hashes),
        })

    spec = {
        "encoder_name": str(encoder_name),
        "crop_top": int(crop_top),
        "track_centerline": str(centerline_path),
        "track_centerline_sha256": sha256_file(centerline_path),
        "runs": run_entries,
    }
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    spec["fingerprint_sha256"] = hashlib.sha256(canonical).hexdigest()
    return spec


def verify_cache_fingerprint(
    cache_path: Path,
    runs: list[Path] | None = None,
    centerline_path: Path | None = None,
    crop_top: int = 0,
    encoder_name: str = "dinov2_vits14",
) -> dict:
    """Verify `feature_cache.meta.json` matches current on-disk inputs.

    Raises RuntimeError if metadata is missing or any input file hash changed.
    """
    meta_path = cache_path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise RuntimeError(
            f"Cache metadata {meta_path} is missing for {cache_path}. "
            "Pass --rebuild-cache to regenerate with verified provenance."
        )
    saved = json.loads(meta_path.read_text())
    if runs is not None and centerline_path is not None:
        expected = compute_cache_fingerprint(
            runs=runs,
            centerline_path=centerline_path,
            crop_top=crop_top,
            encoder_name=encoder_name,
        )
        if saved.get("fingerprint_sha256") != expected["fingerprint_sha256"]:
            raise RuntimeError(
                f"Feature cache {cache_path} fingerprint mismatch: "
                f"cached={saved.get('fingerprint_sha256')} vs current={expected['fingerprint_sha256']}. "
                "Input runs, centerline, or encoder settings changed; pass --rebuild-cache."
            )
    else:
        # Verify against paths recorded in saved metadata
        rec_centerline = Path(saved["track_centerline"])
        if not rec_centerline.exists() or sha256_file(rec_centerline) != saved["track_centerline_sha256"]:
            raise RuntimeError(f"Cached track_centerline {rec_centerline} changed since cache was built.")
    return saved


def normalize_sincos_np(pred_geom: np.ndarray) -> np.ndarray:
    """Return a copy of (..., 15) geometry predictions with [..., 1:3] on the unit circle."""
    out = np.array(pred_geom, dtype=np.float32, copy=True)
    norm = np.maximum(np.hypot(out[..., 1], out[..., 2]), 1e-6)
    out[..., 1] /= norm
    out[..., 2] /= norm
    return out


def splice_geometry_into_state(true_state: np.ndarray, pred_geom: np.ndarray, normalize_heading: bool = True) -> np.ndarray:
    """Replace geometric indices [0:3] and [6:18] of 18D `TrackState`, preserving motion [3:6]."""
    pg = normalize_sincos_np(pred_geom) if normalize_heading else np.asarray(pred_geom, dtype=np.float32)
    out = np.array(true_state, dtype=np.float32, copy=True)
    out[..., 0:3] = pg[..., 0:3]
    out[..., 6:18] = pg[..., 3:15]
    return out


class DinoGridLinearLag15Encoder:
    """Live closed-loop encoder: DINOv2 4x4 grid -> two-stage linear lag head (15D) + telemetry motion (3D).

    Exposes `uses_telemetry = True` and `z_size = 18` so `DonkeyVaeSACEnv` can pass
    its output directly into the frozen `sim_state` teacher policy (`58D = 18D + 40D history`).
    """

    uses_telemetry = True
    z_size = 18

    def __init__(
        self,
        head_path: Path,
        track: TrackState,
        device: torch.device,
        dino_encoder: FrozenDinoGridEncoder | None = None,
    ) -> None:
        saved = torch.load(head_path, map_location=device, weights_only=True)
        if saved.get("format") != "proj15_lags_ridge_v1":
            raise ValueError(f"Unsupported geometry head format in {head_path}: {saved.get('format')}")
        self.head_path = Path(head_path)
        self.head_sha256 = sha256_file(self.head_path)
        self.crop_top = int(saved.get("crop_top", 0))
        self.lags = tuple(int(x) for x in saved["lags"])
        self.max_lag = max(self.lags, default=0)

        self.w1 = saved["w1"].to(device=device, dtype=torch.float32)  # (6144, 15)
        self.b1 = saved["b1"].to(device=device, dtype=torch.float32)  # (15,)
        self.w2 = saved["w2"].to(device=device, dtype=torch.float32)  # (45, 15)
        self.b2 = saved["b2"].to(device=device, dtype=torch.float32)  # (15,)

        self.track = track
        self.device = device
        self.dino = dino_encoder or FrozenDinoGridEncoder(device=device, crop_top=self.crop_top, projection=None)
        if self.dino.crop_top != self.crop_top:
            raise ValueError(f"DINO crop_top mismatch: head expects {self.crop_top}, encoder has {self.dino.crop_top}")

        self._stage1_buf: deque[torch.Tensor] = deque(maxlen=self.max_lag + 1)
        self.last_vision_seconds: float = 0.0
        self.last_obs_rx_mono_s: float = 0.0
        self.last_encode_done_mono_s: float = 0.0
        self.last_eff_cte_err: float = 0.0
        self.last_eff_head_deg: float = 0.0
        self.last_eff_la1_err_pos: float = 0.0
        self.last_true_state: np.ndarray = np.zeros(18, dtype=np.float32)
        self.last_pred_state: np.ndarray = np.zeros(18, dtype=np.float32)
        self.last_true_geom: np.ndarray = np.zeros(15, dtype=np.float32)
        self.last_pred_geom: np.ndarray = np.zeros(15, dtype=np.float32)
        self.last_motion: np.ndarray = np.zeros(3, dtype=np.float32)
        self.injected_cte_errs: list[float] = []
        self.injected_head_degs: list[float] = []
        self.injected_la1_errs: list[float] = []
        self.vision_encode_seconds: list[float] = []

    def reset_episode(self, seed: int | None = None) -> None:
        self._stage1_buf.clear()
        self.last_vision_seconds = 0.0
        self.last_obs_rx_mono_s = 0.0
        self.last_encode_done_mono_s = 0.0
        self.last_eff_cte_err = 0.0
        self.last_eff_head_deg = 0.0
        self.last_eff_la1_err_pos = 0.0
        self.last_true_state = np.zeros(18, dtype=np.float32)
        self.last_pred_state = np.zeros(18, dtype=np.float32)
        self.last_true_geom = np.zeros(15, dtype=np.float32)
        self.last_pred_geom = np.zeros(15, dtype=np.float32)
        self.last_motion = np.zeros(3, dtype=np.float32)
        self.injected_cte_errs.clear()
        self.injected_head_degs.clear()
        self.injected_la1_errs.clear()
        self.vision_encode_seconds.clear()

    @torch.no_grad()
    def predict_geometry_from_grid_tensor(self, grid_1x6144: torch.Tensor) -> np.ndarray:
        """Step the stateful lag buffer with one (1, 6144) grid feature tensor and return (15,) numpy."""
        z1 = (grid_1x6144 @ self.w1 + self.b1).squeeze(0)  # (15,)
        self._stage1_buf.appendleft(z1)
        while len(self._stage1_buf) < self._stage1_buf.maxlen:
            self._stage1_buf.append(z1)
        parts = [self._stage1_buf[0]] + [self._stage1_buf[lag] for lag in self.lags]
        s45 = torch.cat(parts, dim=0)  # (45,)
        pred15 = (s45 @ self.w2 + self.b2).detach().cpu().numpy().astype(np.float32)
        return normalize_sincos_np(pred15)

    @torch.no_grad()
    def encode(self, image: np.ndarray, info: dict) -> np.ndarray:
        t0 = time.perf_counter()
        self.last_obs_rx_mono_s = t0
        img = np.asarray(image, dtype=np.uint8)
        grid_feat = self.dino.grid_features(img[None])  # (1, 6144) on self.device
        pred15 = self.predict_geometry_from_grid_tensor(grid_feat)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_vis = time.perf_counter()
        self.last_vision_seconds = t_vis - t0
        self.vision_encode_seconds.append(self.last_vision_seconds)

        raw = self.track.encode(img, info).astype(np.float32)  # (18,) ground-truth state
        out = splice_geometry_into_state(raw, pred15, normalize_heading=False)
        self.last_encode_done_mono_s = time.perf_counter()

        self.last_true_state = raw
        self.last_pred_state = out
        self.last_true_geom = raw[GEOM_IDX].astype(np.float32)
        self.last_pred_geom = pred15.astype(np.float32)
        self.last_motion = raw[MOTION_IDX].astype(np.float32)

        # Record live perception error against exact TrackState on this frame
        self.last_eff_cte_err = float((out[0] - raw[0]) * 2.0)
        theta_true = float(np.arctan2(raw[1], raw[2]))
        theta_pred = float(np.arctan2(out[1], out[2]))
        self.last_eff_head_deg = float(
            np.degrees(np.arctan2(np.sin(theta_pred - theta_true), np.cos(theta_pred - theta_true)))
        )
        # Lookahead 1.0 pos unit is index k=2 in LOOKAHEAD=(0.3, 0.6, 1.0, ...) -> raw[10:12]
        self.last_eff_la1_err_pos = float(np.hypot(out[10] - raw[10], out[11] - raw[11]) * 1.0)

        self.injected_cte_errs.append(self.last_eff_cte_err)
        self.injected_head_degs.append(self.last_eff_head_deg)
        self.injected_la1_errs.append(self.last_eff_la1_err_pos)
        return out


def classify_block_regime(state_seg: np.ndarray) -> str:
    """Classify a contiguous telemetry segment into `high_cte`, `corner`, or `cruising`."""
    abs_cte = np.abs(state_seg[:, 0] * 2.0)
    abs_yaw_rate = np.abs(state_seg[:, 5] * 2.0)
    if float(np.max(abs_cte)) >= 1.0:
        return "high_cte"
    if float(np.max(abs_yaw_rate)) >= 1.0:
        return "corner"
    return "cruising"


def load_aligned_residual_blocks(
    probe_dir: Path,
    pred_keys: list[str],
    min_block_len: int = 80,
    regime_filter: str = "all",
) -> list[dict]:
    """Load contiguous multi-step residual blocks aligned across all `pred_keys`.

    Because every model was evaluated on the exact same `valid_lags_mask`, block `b`
    at offset `k` corresponds to the exact same held-out `(run, layout, step)` frame
    across every model in `pred_keys`.
    """
    cache = np.load(probe_dir / "feature_cache.npz", allow_pickle=False)
    oof = np.load(probe_dir / "oof_geometry_predictions_lags.npz", allow_pickle=False)

    valid = oof["valid_lags_mask"].astype(bool)
    true_geom = oof["true_geom"]  # (N, 15)
    state = cache["state"]        # (N, 18)
    run = cache["run"]
    layout = cache["layout"]
    model = cache["model"]
    step = cache["step"]

    sin_t, cos_t = true_geom[:, 1], true_geom[:, 2]
    per_model_res = {}
    for key in pred_keys:
        if key not in oof.files:
            raise KeyError(f"Prediction key {key!r} not found in {probe_dir / 'oof_geometry_predictions_lags.npz'}")
        pg = oof[key]
        d_cte_half = (pg[:, 0] - true_geom[:, 0]).astype(np.float32)
        d_lookahead = (pg[:, 3:15] - true_geom[:, 3:15]).astype(np.float32)
        norm_p = np.maximum(np.hypot(pg[:, 1], pg[:, 2]), 1e-6)
        sin_p, cos_p = pg[:, 1] / norm_p, pg[:, 2] / norm_p
        d_theta = np.arctan2(sin_p * cos_t - cos_p * sin_t, sin_p * sin_t + cos_p * cos_t).astype(np.float32)
        per_model_res[key] = {
            "d_cte_half": d_cte_half,
            "d_theta": d_theta,
            "d_lookahead": d_lookahead,
        }

    by_ep = defaultdict(list)
    for idx in np.flatnonzero(valid):
        by_ep[(str(run[idx]), str(layout[idx]), str(model[idx]))].append(int(idx))

    blocks = []
    for (r_name, l_name, m_name), indices in sorted(by_ep.items()):
        indices.sort(key=lambda i: int(step[i]))
        start = 0
        for pos in range(1, len(indices) + 1):
            if pos == len(indices) or int(step[indices[pos]]) != int(step[indices[pos - 1]]) + 1:
                seg = indices[start:pos]
                if len(seg) >= min_block_len:
                    regime = classify_block_regime(state[seg])
                    if regime_filter in ("all", regime):
                        blk_id = len(blocks)
                        blocks.append({
                            "block_id": blk_id,
                            "source": f"{r_name}/{l_name}/{m_name}",
                            "start_step": int(step[seg[0]]),
                            "end_step": int(step[seg[-1]]),
                            "length": len(seg),
                            "regime": regime,
                            "models": {
                                key: {
                                    "d_cte_half": per_model_res[key]["d_cte_half"][seg],
                                    "d_theta": per_model_res[key]["d_theta"][seg],
                                    "d_lookahead": per_model_res[key]["d_lookahead"][seg],
                                }
                                for key in pred_keys
                            },
                        })
                start = pos
    if not blocks:
        raise RuntimeError(f"No contiguous residual blocks >= {min_block_len} for regime={regime_filter!r}")
    return blocks


def build_layout_block_schedule(
    blocks: list[dict],
    total_steps: int,
    block_len: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Pre-sample a deterministic step-by-step `(block_idx, offset)` schedule for one layout.

    Sharing this schedule across all conditions on the same layout guarantees that
    1.0x vs 1.5x, delay=0 vs delay=1, and model A vs model B all replay the exact
    same held-out trajectory segments at every step.
    """
    rng = np.random.default_rng(seed)
    schedule: list[tuple[int, int]] = []
    while len(schedule) < total_steps:
        b_idx = int(rng.integers(0, len(blocks)))
        n = int(blocks[b_idx]["length"])
        length = min(int(block_len), n)
        start = int(rng.integers(0, n - length + 1))
        for off in range(start, start + length):
            schedule.append((b_idx, off))
            if len(schedule) >= total_steps:
                break
    return schedule


SPLIT_DAGGER_TRAIN_POOL = "dagger_train_pool"
SPLIT_LOCKED_EVAL_HOLDOUT = "locked_eval_holdout"


def assign_layout_split(
    layout_idx: int,
    total_layouts: int = 24,
    train_pool_layouts: int | None = None,
) -> str:
    """Pre-registered layout partition separating retraining pool from locked holdout evaluation.

    By default, the first half of layouts (`1 .. total_layouts // 2`, e.g. `1..12` of 24)
    are assigned to `dagger_train_pool` (eligible to be combined with Round B recovery
    data for DAgger / perception retraining), while the second half (`13..24`) are
    permanently locked as `locked_eval_holdout` and must never be used for training.
    """
    if layout_idx < 1 or layout_idx > total_layouts:
        raise ValueError(f"layout_idx={layout_idx} out of range 1..{total_layouts}")
    cutoff = (total_layouts // 2) if train_pool_layouts is None else int(train_pool_layouts)
    if cutoff < 0 or cutoff > total_layouts:
        raise ValueError(f"train_pool_layouts={cutoff} out of range 0..{total_layouts}")
    return SPLIT_DAGGER_TRAIN_POOL if layout_idx <= cutoff else SPLIT_LOCKED_EVAL_HOLDOUT


def assert_not_locked_holdout(paths: Path | list[Path]) -> None:
    """Guard function for training scripts: raises PermissionError if any input is locked holdout."""
    path_list = [paths] if isinstance(paths, Path) else list(paths)
    for p in path_list:
        p = Path(p)
        if SPLIT_LOCKED_EVAL_HOLDOUT in p.name:
            raise PermissionError(
                f"Refusing to load locked evaluation holdout file for training: {p}"
            )
        if p.exists() and p.suffix == ".npz":
            with np.load(p, allow_pickle=False) as data:
                if "split" in data.files:
                    split_val = str(np.asarray(data["split"]).item())
                    if split_val == SPLIT_LOCKED_EVAL_HOLDOUT:
                        raise PermissionError(
                            f"Refusing to load capture with split={split_val!r} for training: {p}"
                        )


def extract_excursion_windows(
    cte: np.ndarray,
    eff_cte_err: np.ndarray,
    eff_head_deg: np.ndarray,
    rl_steer: np.ndarray,
    rl_steer_delta: np.ndarray,
    rl_throttle: np.ndarray,
    entry_threshold: float = 0.8,
    recovery_threshold: float = 0.5,
    rate_limit_threshold: float = 0.199,
) -> list[dict]:
    """Identify high-|CTE| excursion windows, pre-peak perception bias, and rate-limit saturation."""
    cte = np.asarray(cte, dtype=np.float32)
    eff_cte_err = np.asarray(eff_cte_err, dtype=np.float32)
    eff_head_deg = np.asarray(eff_head_deg, dtype=np.float32)
    rl_steer = np.asarray(rl_steer, dtype=np.float32)
    rl_steer_delta = np.asarray(rl_steer_delta, dtype=np.float32)
    rl_throttle = np.asarray(rl_throttle, dtype=np.float32)
    abs_cte = np.abs(cte)
    n = len(cte)
    windows: list[dict] = []
    i = 0
    while i < n:
        if abs_cte[i] < entry_threshold:
            i += 1
            continue
        start = i
        j = i
        while j < n and abs_cte[j] >= recovery_threshold:
            j += 1
        end = j - 1  # inclusive 0-based index
        seg_abs = abs_cte[start : end + 1]
        peak_rel = int(np.argmax(seg_abs))
        peak_idx = start + peak_rel

        pre_start = max(0, start - 15)
        pre_cte_err = eff_cte_err[pre_start:start] if start > pre_start else eff_cte_err[start : start + 1]
        pre_head_err = eff_head_deg[pre_start:start] if start > pre_start else eff_head_deg[start : start + 1]

        win_start = max(0, start - 10)
        win_dsteer = np.abs(rl_steer_delta[win_start : end + 1])
        sat_mask = win_dsteer >= rate_limit_threshold
        max_consec_sat = 0
        cur_sat = 0
        for flag in sat_mask:
            if flag:
                cur_sat += 1
                max_consec_sat = max(max_consec_sat, cur_sat)
            else:
                cur_sat = 0

        recovered = bool((end + 1) < n and abs_cte[end + 1] < recovery_threshold)
        windows.append({
            "start_step": int(start + 1),
            "peak_step": int(peak_idx + 1),
            "end_step": int(end + 1),
            "duration_steps": int(end - start + 1),
            "peak_cte_signed": float(cte[peak_idx]),
            "peak_abs_cte": float(abs_cte[peak_idx]),
            "eff_cte_err_at_peak": float(eff_cte_err[peak_idx]),
            "eff_head_deg_at_peak": float(eff_head_deg[peak_idx]),
            "rl_steer_at_peak": float(rl_steer[peak_idx]),
            "rl_throttle_at_peak": float(rl_throttle[peak_idx]),
            "pre15_mean_eff_cte_err": float(np.mean(pre_cte_err)),
            "pre15_mean_eff_head_deg": float(np.mean(pre_head_err)),
            "rate_limit_saturated_steps": int(np.sum(sat_mask)),
            "max_consecutive_rate_limit_steps": int(max_consec_sat),
            "recovered": recovered,
        })
        i = j
    return windows
