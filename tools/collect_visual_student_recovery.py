"""Collect fresh student trajectories; privileged state is separate offline-label data."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import time

import gymnasium as gym
import gym_donkeycar  # noqa: F401
from gymnasium.wrappers import TimeLimit
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.donkey_sync import add_control_fix_args, configure_env_control_fixes
from rl.experiment import EvalRecord, write_json
from rl.track_state import TrackState
from rl.train_vae_sac import DonkeyVaeSACEnv, RaffinRewardConfig
from rl.visual_student import OBSERVATION_CONTRACT, VisualGeometryEncoder, VisualStudentActor


def collect_episode(env, actor, track, record, episode):
    observation, info = env.reset()
    images = [env.env.last_raw_obs.copy()]
    observations, actions, states, indices = [], [], [], []
    rewards, ctes, speeds = [], [], []
    terminated = truncated = False
    while not (terminated or truncated):
        started = time.perf_counter()
        # The actor is called before privileged label-state construction and sees x only.
        action, _ = actor.predict(observation)
        predicted = time.perf_counter()
        observations.append(observation.copy()); actions.append(action.copy())
        states.append(track.encode(None, info))
        indices.append(track.locate([info['pos'][0], info['pos'][2]], info['car'][2])[0])
        observation, reward, terminated, truncated, info = env.step(action)
        info = dict(info, policy_seconds=predicted-started, env_step_seconds=time.perf_counter()-predicted)
        image = env.env.last_raw_obs
        images.append(image.copy())
        rewards.append(reward); ctes.append(info['abs_cte']); speeds.append(info['speed'])
        record.step({'episode':episode}, len(actions), action, reward, terminated, truncated, info, image=image)
    data = {'observations':np.stack(observations), 'raw_actions':np.stack(actions),
            'label_state':np.stack(states), 'track_index':np.asarray(indices,np.int32), 'images':np.stack(images)}
    result = {'episode':episode, 'steps':len(actions), 'completed':bool(truncated and not terminated),
              'termination_reason':info.get('termination_reason') or 'time_limit', 'laps':int(info.get('lap_count',0)),
              'reward':float(sum(rewards)), 'mean_abs_cte':float(np.mean(ctes)),
              'max_abs_cte':float(np.max(ctes)), 'mean_speed':float(np.mean(speeds))}
    return data, result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student',type=Path,default=Path('models/visual_student_no_motion_v1/student.pt'))
    parser.add_argument('--geometry-head',type=Path,default=Path('models/dinov2_geometry_15d/proj15_lags_ridge.pt'))
    parser.add_argument('--track-centerline',type=Path,default=Path('models/track_centerline/generated_track.npz'))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--host',default=os.environ.get('DONKEY_SIM_HOST','127.0.0.1'))
    parser.add_argument('--port',type=int,default=9091)
    parser.add_argument('--episodes',type=int,default=8)
    parser.add_argument('--train-layouts',type=int,default=6)
    parser.add_argument('--max-episode-steps',type=int,default=500)
    parser.add_argument('--device',default='cuda')
    add_control_fix_args(parser)
    parser.set_defaults(low_jitter_sync=True,tcp_nodelay=True,observation_timeout=.1,
                        capture_tail_steps=120,save_telemetry=True)
    args=parser.parse_args()
    if not 0<args.train_layouts<args.episodes or args.max_episode_steps<1:
        parser.error('Require distinct training/development layouts and a positive step cap')
    if args.reversal_unwind_scale!=1 or not args.low_jitter_sync:
        parser.error('Keep the frozen baseline control configuration')
    encoder=VisualGeometryEncoder(args.geometry_head,args.device)
    actor=VisualStudentActor.load(args.student,geometry_head_sha256=encoder.head_sha256,device=args.device)
    track=TrackState.load(args.track_centerline)
    with EvalRecord(args,{'student':args.student,'geometry_head':args.geometry_head,'label_centerline':args.track_centerline}) as record:
        record.metadata.update(student_input_contract=OBSERVATION_CONTRACT,teacher_loaded=False,
                               truth_usage='Separate offline-label state, scoring and termination only')
        record.encoder(encoder)
        shutil.copy2(__file__,record.path/'source'/Path(__file__).name)
        write_json(record.path/'split_manifest.json',{'train':list(range(1,args.train_layouts+1)),
                   'development':list(range(args.train_layouts+1,args.episodes+1)),
                   'rule':'Assigned before driving; all frames from a layout remain in the same split.'})
        base=gym.make('donkey-generated-track-v0',conf={'host':args.host,'port':args.port,
                      'cam_resolution':(120,160,3),'max_cte':20.})
        wrapped=DonkeyVaeSACEnv(base,vae=encoder,min_throttle=.2,max_throttle=.7,max_steering=1.,
                    max_steering_diff=.1,n_command_history=20,latent_lags=(2,4),scene_reload_every=1,
                    reward_config=RaffinRewardConfig(max_cte_error=2.,alive_reward=1.5,speed_reward_weight=.15,cte_speed_penalty_weight=.25))
        env=TimeLimit(wrapped,max_episode_steps=args.max_episode_steps)
        try:
            configure_env_control_fixes(wrapped,args)
            for episode in range(1,args.episodes+1):
                encoder.reset_episode()
                data,result=collect_episode(env,actor,track,record,episode)
                base.unwrapped.viewer.handler.send_control(0.,0.,1.)
                # Runtime input arrays and privileged label arrays are separate archives.
                np.savez(record.path/f'student_inputs_{episode:02d}.npz',
                         observations=data['observations'],raw_actions=data['raw_actions'])
                np.savez(record.path/f'label_state_{episode:02d}.npz',
                         state=data['label_state'],track_index=data['track_index'])
                np.savez_compressed(record.path/f'camera_{episode:02d}.npz',images=data['images'])
                result['split']='train' if episode<=args.train_layouts else 'development'
                record.episode(result);print(result,flush=True)
            record.finish({'episodes':len(record.results),'completed':sum(r['completed'] for r in record.results),
                           'purpose':'DAgger-style student-driven collection; no reliability claim',
                           'student_input_contract':OBSERVATION_CONTRACT})
        finally:
            try:base.unwrapped.viewer.handler.send_control(0.,0.,1.)
            finally:env.close()


if __name__=='__main__':main()
