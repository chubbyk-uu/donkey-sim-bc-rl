#!/usr/bin/env python3
"""Throwaway check: does re-sending `load_scene` regenerate the random trees?

Connects to the running Donkey sim, then reloads the scene a few times with a
pause between each so you can watch the sim window and judge whether the tree
layout / shadows change on reload. If they do, per-N-episode scene reload is a
viable domain-randomization hook for training. If they stay identical, the sim
uses a fixed seed per scene and this approach won't help.

Usage (sim must be running, with random light + random trees enabled):

    export DONKEY_SIM_HOST="$(ip route | awk '/default/ {print $3}')"
    python tools/test_scene_reload.py --reloads 3 --pause 8
"""
from __future__ import annotations

import argparse
import os
import time

import gymnasium as gym
import gym_donkeycar  # noqa: F401 - registers donkey gym environments


def reload_scene(
    ctrl,
    scene: str,
    timeout: float = 30.0,
    exit_settle: float = 1.0,
    retry_every: float = 1.5,
) -> bool:
    """Reload the scene, gating success on the real `handler.loaded` state.

    The handshake is asymmetric:
    - load completion HAS a state signal: `handler.loaded`, set by
      on_car_loaded / on_need_car_config.
    - the exit→menu transition has NO reliable incoming message.

    So the two halves are handled differently. After exit_scene we wait a fixed
    `exit_settle` BEFORE the first load — sending load_scene too soon steps on
    the exit and the sim never leaves the current scene. Then we re-send
    load_scene every `retry_every` until `loaded` flips True, so a slow exit
    self-corrects (early loads land while still in-scene and are ignored).
    Returns True on confirmed reload, False on timeout.
    """
    ctrl.handler.loaded = False
    ctrl.handler.send_exit_scene()
    time.sleep(exit_settle)  # no exit signal exists — let the exit complete first
    deadline = time.time() + timeout
    while not ctrl.handler.loaded:
        if time.time() > deadline:
            return False
        ctrl.handler.send_load_scene(scene)
        time.sleep(retry_every)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", default="donkey-generated-track-v0")
    p.add_argument("--host", default=os.environ.get("DONKEY_SIM_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("DONKEY_SIM_PORT", "9091")))
    p.add_argument("--reloads", type=int, default=3, help="how many times to reload the scene")
    p.add_argument("--pause", type=float, default=8.0, help="seconds to pause so you can look")
    p.add_argument("--exit-settle", type=float, default=1.0,
                   help="seconds to wait after exit_scene before the first load")
    args = p.parse_args()

    conf = {"host": args.host, "port": args.port}
    print(f"connecting to {args.host}:{args.port}, scene={args.env_id}")
    env = gym.make(args.env_id, conf=conf)
    ctrl = env.unwrapped.viewer  # DonkeyUnitySimContoller
    scene = ctrl.handler.SceneToLoad

    print(f"\n=== INITIAL LOAD ({scene}) — look at the trees/shadows NOW ===")
    time.sleep(args.pause)

    for i in range(1, args.reloads + 1):
        print(f"\n--- reload (#{i}) ---")
        if not reload_scene(ctrl, scene, exit_settle=args.exit_settle):
            print("  TIMEOUT — sim never confirmed reload (handler.loaded stayed False).")
            break
        print(f"=== RELOAD #{i} confirmed (handler.loaded=True) — trees/shadows changed? ===")
        time.sleep(args.pause)

    print("\ndone, closing.")
    env.close()


if __name__ == "__main__":
    main()
