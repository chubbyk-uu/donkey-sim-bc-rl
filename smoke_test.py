import argparse
import logging
import os
import time

import gymnasium as gym
import gym_donkeycar  # noqa: F401 - registers donkey environments
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="donkey-warren-track-v0")
    parser.add_argument("--host", default=os.environ.get("DONKEY_SIM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DONKEY_SIM_PORT", "9091")))
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    conf = {
        "host": args.host,
        "port": args.port,
        "log_level": logging.INFO,
        "cam_resolution": (120, 160, 3),
    }

    print(f"Connecting to Donkey Simulator at {args.host}:{args.port}")
    env = gym.make(args.env_id, conf=conf)

    try:
        obs, info = env.reset()
        print(f"reset ok: obs_shape={getattr(obs, 'shape', None)} info_keys={sorted(info.keys())}")

        for i in range(args.steps):
            action = np.array([0.0, 0.35], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            if i % 10 == 0:
                print(
                    f"step={i:04d} obs_shape={getattr(obs, 'shape', None)} "
                    f"reward={reward:.4f} terminated={terminated} truncated={truncated} "
                    f"cte={info.get('cte')} speed={info.get('speed')}"
                )

            if terminated or truncated:
                print("episode ended; resetting")
                obs, info = env.reset()

            time.sleep(0.02)
    finally:
        env.close()


if __name__ == "__main__":
    main()
