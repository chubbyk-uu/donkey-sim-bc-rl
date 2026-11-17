"""Explicit camera-only student evaluation; ground truth is scoring/logging only."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import gymnasium as gym
import gym_donkeycar  # noqa: F401
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.donkey_sync import add_control_fix_args, configure_env_control_fixes
from rl.experiment import EvalRecord
from rl.train_vae_sac import DonkeyVaeSACEnv, RaffinRewardConfig
from rl.visual_student import OBSERVATION_CONTRACT, VisualGeometryEncoder, VisualStudentActor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, default=Path("models/visual_student_no_motion_v1/student.pt"))
    parser.add_argument("--geometry-head", type=Path, default=Path("models/dinov2_geometry_15d/proj15_lags_ridge.pt"))
    parser.add_argument("--host", default=os.environ.get("DONKEY_SIM_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/visual_student_closed_loop_v1"))
    parser.add_argument("--capture-tail-steps", type=int, default=120)
    parser.add_argument("--save-telemetry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_control_fix_args(parser)
    parser.set_defaults(low_jitter_sync=True, tcp_nodelay=True, observation_timeout=0.10)
    args = parser.parse_args()
    if args.episodes < 1 or args.max_episode_steps < 1 or args.capture_tail_steps < 0:
        parser.error("Invalid episode/capture limits")
    if args.reversal_unwind_scale != 1.0:
        parser.error("The student checkpoint uses fixed steering limits with unwind disabled")
    if args.observation_timeout and not args.low_jitter_sync:
        parser.error("Observation timeout requires low-jitter sync")
    encoder = VisualGeometryEncoder(args.geometry_head, args.device)
    policy = VisualStudentActor.load(args.student, geometry_head_sha256=encoder.head_sha256, device=args.device)
    with EvalRecord(args, {"student": args.student, "geometry_head": args.geometry_head}) as record:
        record.metadata["student_input_contract"] = OBSERVATION_CONTRACT
        record.encoder(encoder)
        base = gym.make("donkey-generated-track-v0", conf={"host":args.host,"port":args.port,
                        "cam_resolution":(120,160,3),"max_cte":20.0})
        wrapped = DonkeyVaeSACEnv(base, vae=encoder, min_throttle=.2, max_throttle=.7,
                    max_steering=1.0, max_steering_diff=.1, n_command_history=20, latent_lags=(2,4),
                    scene_reload_every=1, reward_config=RaffinRewardConfig(max_cte_error=2.0,
                    alive_reward=1.5,speed_reward_weight=.15,cte_speed_penalty_weight=.25))
        env = TimeLimit(wrapped, max_episode_steps=args.max_episode_steps)
        try:
            configure_env_control_fixes(wrapped,args)
            for episode in range(1,args.episodes+1):
                encoder.reset_episode()
                observation, info = env.reset()
                assert observation.shape == (85,)
                record.frame({"episode":episode},wrapped.last_raw_obs)
                total_reward, ctes, speeds = 0.0, [], []
                steps = 0
                decision_observations, raw_actions = [], []
                terminated = truncated = False
                while not (terminated or truncated):
                    started=time.perf_counter()
                    action,_=policy.predict(observation)
                    decision_observations.append(observation.copy())
                    raw_actions.append(action.copy())
                    predicted=time.perf_counter()
                    observation,reward,terminated,truncated,info=env.step(action)
                    info=dict(info,policy_seconds=predicted-started,env_step_seconds=time.perf_counter()-predicted)
                    steps+=1;total_reward+=reward
                    ctes.append(info['abs_cte']);speeds.append(info['speed'])
                    record.step({'episode':episode},steps,action,reward,terminated,truncated,info,
                                image=wrapped.last_raw_obs if args.capture_tail_steps else None)
                # Stop before image/JSON serialization; no further policy decision
                # or input-history mutation is made after the scored episode.
                handler=base.unwrapped.viewer.handler
                if callable(getattr(handler,'send_control',None)):
                    handler.send_control(0.,0.,1.)
                np.savez(record.path/f'policy_inputs_episode_{episode:02d}.npz',
                         observations=np.stack(decision_observations), raw_actions=np.stack(raw_actions))
                row={'episode':episode,'steps':steps,'completed':bool(truncated and not terminated),
                     'termination_reason':info.get('termination_reason') or 'time_limit','reward':total_reward,
                     'mean_abs_cte':float(np.mean(ctes)),'max_abs_cte':float(np.max(ctes)),
                     'mean_speed':float(np.mean(speeds)),'laps':int(info.get('lap_count',0))}
                record.episode(row);print(row,flush=True)
            record.finish({'episodes':len(record.results),'completed':sum(r['completed'] for r in record.results),
                           'student_input_contract':OBSERVATION_CONTRACT})
        finally:
            env.close()


if __name__ == "__main__":
    main()
