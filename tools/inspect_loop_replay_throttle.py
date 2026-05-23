"""Inspect throttle trajectories from a saved SAC replay buffer.

Decodes the raw policy action (stored in the buffer, in [-1, 1]) back to the
physical throttle that was sent to the sim, then segments the buffer by `done`
flags and reports per-episode throttle statistics. Useful for answering
"during the long 5000-step episode, did the agent actually vary throttle, or
did it pin to max?"
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


def decode_throttle(raw_a1: np.ndarray, min_throttle: float, max_throttle: float) -> np.ndarray:
    t = np.clip((raw_a1 + 1.0) / 2.0, 0.0, 1.0)
    return (1.0 - t) * min_throttle + max_throttle * t


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--buffer", type=Path, required=True)
    p.add_argument("--min-throttle", type=float, default=0.2)
    p.add_argument("--max-throttle", type=float, default=0.7)
    p.add_argument("--top-n", type=int, default=5, help="Inspect the N longest episodes")
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--dump-csv", type=Path, default=None,
                   help="If set, dump per-step throttle for the longest episode to this CSV")
    args = p.parse_args()

    with open(args.buffer, "rb") as f:
        buf = pickle.load(f)

    pos = int(buf.pos)
    full = bool(buf.full)
    size = buf.buffer_size if full else pos
    print(f"buffer size used: {size}  pos={pos}  full={full}")

    # SB3 ReplayBuffer: actions shape (buffer_size, n_envs, action_dim); dones (buffer_size, n_envs)
    actions = np.asarray(buf.actions)[:size, 0, :]
    dones = np.asarray(buf.dones)[:size, 0].astype(bool)
    timeouts = np.asarray(getattr(buf, "timeouts", np.zeros_like(dones)))[:size, 0].astype(bool) \
        if hasattr(buf, "timeouts") else np.zeros_like(dones)

    raw_a1 = actions[:, 1]
    throttle = decode_throttle(raw_a1, args.min_throttle, args.max_throttle)
    steer = actions[:, 0]

    # Walk buffer chronologically. If full, true start = pos (oldest); else start = 0.
    if full:
        order = np.concatenate([np.arange(pos, size), np.arange(0, pos)])
    else:
        order = np.arange(0, size)

    th_ord = throttle[order]
    st_ord = steer[order]
    done_ord = dones[order]
    to_ord = timeouts[order]

    # Build episodes by walking done flags.
    episodes = []
    start = 0
    for i in range(len(th_ord)):
        if done_ord[i]:
            episodes.append((start, i + 1, bool(to_ord[i])))
            start = i + 1
    if start < len(th_ord):
        episodes.append((start, len(th_ord), False))  # tail (no done yet)
    print(f"episodes found: {len(episodes)}")

    episodes_sorted = sorted(episodes, key=lambda e: e[1] - e[0], reverse=True)
    top = episodes_sorted[: args.top_n]
    print(f"\n--- top {len(top)} longest episodes ---")
    for rank, (s, e, was_to) in enumerate(top, 1):
        ep_th = th_ord[s:e]
        ep_st = st_ord[s:e]
        length = e - s
        tag = "TRUNC" if was_to else "TERM"
        print(
            f"#{rank:2d}  len={length:5d}  throttle "
            f"min={ep_th.min():.3f} mean={ep_th.mean():.3f} max={ep_th.max():.3f} "
            f"std={ep_th.std():.3f}  |steer| mean={np.abs(ep_st).mean():.3f}  {tag}"
        )

    # Recent episodes — buffer tail, sorted chronologically
    print(f"\n--- recent {args.top_n} episodes (buffer tail, chronological — true picture of CURRENT policy) ---")
    recent = episodes[-args.top_n:]
    for rank, (s, e, was_to) in enumerate(recent, 1):
        ep_th = th_ord[s:e]
        ep_st = st_ord[s:e]
        length = e - s
        tag = "TRUNC" if was_to else "TERM"
        print(
            f"#{rank:2d}  len={length:5d}  throttle "
            f"min={ep_th.min():.3f} mean={ep_th.mean():.3f} max={ep_th.max():.3f} "
            f"std={ep_th.std():.3f}  |steer| mean={np.abs(ep_st).mean():.3f}  {tag}"
        )

    # Aggregate stats on recent episodes
    recent_lens = np.asarray([e - s for s, e, _ in recent], dtype=np.int32)
    recent_truncs = sum(1 for _, _, t in recent if t)
    print(
        f"\nrecent {args.top_n} ep stats: len mean={recent_lens.mean():.0f} "
        f"median={int(np.median(recent_lens))} max={recent_lens.max()} min={recent_lens.min()}  "
        f"truncated={recent_truncs}/{len(recent)}"
    )

    # Detailed look at the longest episode
    s, e, was_to = episodes_sorted[0]
    ep_th = th_ord[s:e]
    print(f"\n--- longest episode (len={e - s}) throttle histogram ---")
    edges = np.linspace(args.min_throttle, args.max_throttle, args.bins + 1)
    hist, _ = np.histogram(ep_th, bins=edges)
    for i in range(args.bins):
        lo, hi = edges[i], edges[i + 1]
        bar = "#" * int(40 * hist[i] / max(1, hist.max()))
        pct = 100.0 * hist[i] / len(ep_th)
        print(f"  [{lo:.3f}, {hi:.3f})  {hist[i]:5d}  {pct:5.1f}%  {bar}")

    # Throttle trajectory in chunks (e.g., 50 chunks across the episode)
    print(f"\n--- longest episode throttle over time (50 bins) ---")
    n_bins = 50
    edges_t = np.linspace(0, len(ep_th), n_bins + 1, dtype=int)
    for i in range(n_bins):
        seg = ep_th[edges_t[i]:edges_t[i + 1]]
        if len(seg) == 0:
            continue
        m = seg.mean()
        # Bar from min_throttle to max_throttle
        rel = (m - args.min_throttle) / max(1e-6, args.max_throttle - args.min_throttle)
        bar = "#" * int(40 * rel)
        print(f"  step {edges_t[i]:5d}-{edges_t[i+1]:5d}  thr_mean={m:.3f}  {bar}")

    if args.dump_csv is not None:
        args.dump_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.dump_csv, "w") as f:
            f.write("step,throttle,steer\n")
            for i in range(len(ep_th)):
                f.write(f"{i},{ep_th[i]:.4f},{st_ord[s + i]:.4f}\n")
        print(f"\nDumped longest episode throttle to {args.dump_csv}")


if __name__ == "__main__":
    main()
