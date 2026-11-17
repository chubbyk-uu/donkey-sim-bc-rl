"""Frozen camera-only students on shared fresh layouts with balanced run order."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import gymnasium as gym
import gym_donkeycar  # noqa: F401
from gymnasium.wrappers import TimeLimit
import numpy as np

if __package__ is None or __package__=='':sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rl.donkey_sync import add_control_fix_args,configure_env_control_fixes
from rl.experiment import EvalRecord,bootstrap_mean_interval,paired_statistics,sha256,write_json
from rl.train_vae_sac import DonkeyVaeSACEnv,RaffinRewardConfig,reload_scene
from rl.visual_student import VisualGeometryEncoder,VisualStudentActor,OBSERVATION_CONTRACT


def balanced_orders(layouts,seed):
    first=np.arange(layouts)%2
    np.random.default_rng(seed).shuffle(first)
    labels=('baseline','candidate')
    return [[labels[i],labels[1-i]] for i in first]


def paired_summary(rows):
    layouts=sorted({r['layout'] for r in rows})
    by_key={(r['layout'],r['model']):r for r in rows}
    if len(by_key)!=len(rows) or any((l,m) not in by_key for l in layouts for m in ['baseline','candidate']):
        raise ValueError('Need exactly one baseline/candidate episode per layout')
    models={}
    for model in ['baseline','candidate']:
        seq=[by_key[l,model] for l in layouts]
        models[model]={'layouts':len(seq),'completed':sum(r['completed'] for r in seq),
            'steps':sum(r['steps'] for r in seq),'laps':sum(r['laps'] for r in seq),
            'mean_abs_cte':float(np.mean([r['mean_abs_cte'] for r in seq])),
            'max_abs_cte':max(r['max_abs_cte'] for r in seq),
            'mean_speed':float(np.mean([r['mean_speed'] for r in seq])),
            'frames_abs_cte_ge_1':sum(r['frames_abs_cte_ge_1'] for r in seq)}
    differences={}
    for metric in ['mean_abs_cte','max_abs_cte','mean_speed','startup_mean_abs_cte','laps','frames_abs_cte_ge_1']:
        values=[by_key[l,'candidate'][metric]-by_key[l,'baseline'][metric] for l in layouts]
        differences[metric]={'per_layout':values,'mean':float(np.mean(values)),
                             'bootstrap_ci95':bootstrap_mean_interval(values)}
    return {'models':models,'candidate_minus_baseline':differences}


def run_episode(env,encoder,actor,record,context):
    encoder.reset_episode()
    observation,info=env.reset()
    initial_laps=int(info.get('lap_count',0))
    record.frame(context,env.env.last_raw_obs)
    images=[env.env.last_raw_obs.copy()];observations=[];actions=[]
    rewards=[];ctes=[];speeds=[];terminated=truncated=False
    while not (terminated or truncated):
        start=time.perf_counter();action,_=actor.predict(observation)
        observations.append(observation.copy());actions.append(action.copy());predicted=time.perf_counter()
        observation,reward,terminated,truncated,info=env.step(action)
        info=dict(info,policy_seconds=predicted-start,env_step_seconds=time.perf_counter()-predicted)
        images.append(env.env.last_raw_obs.copy())
        rewards.append(reward);ctes.append(info['abs_cte']);speeds.append(info['speed'])
        record.step(context,len(actions),action,reward,terminated,truncated,info,image=env.env.last_raw_obs)
    env.unwrapped.viewer.handler.send_control(0.,0.,1.)
    token=f"layout_{context['layout']:02d}_{context['model']}"
    np.savez(record.path/f'{token}_inputs.npz',observations=np.stack(observations),raw_actions=np.stack(actions))
    np.savez_compressed(record.path/f'{token}_camera.npz',images=np.stack(images))
    completed=bool(truncated and not terminated)
    return dict(context,steps=len(actions),completed=completed,trunc=completed,
                termination_reason=info.get('termination_reason') or 'time_limit',reward=float(sum(rewards)),
                laps=int(info.get('lap_count',0))-initial_laps,initial_lap_count=initial_laps,
                mean_abs_cte=float(np.mean(ctes)),max_abs_cte=float(np.max(ctes)),mean_speed=float(np.mean(speeds)),
                startup_mean_abs_cte=float(np.mean(ctes[:250])),frames_abs_cte_ge_1=int(np.sum(np.array(ctes)>=1)),
                inputs=f'{token}_inputs.npz',camera=f'{token}_camera.npz',evaluation_only=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=Path('models/visual_student_no_motion_v1/student.pt'))
    parser.add_argument('--candidate',type=Path,default=Path('models/visual_student_bend_recovery_v1/student.pt'))
    parser.add_argument('--geometry-head',type=Path,default=Path('models/dinov2_geometry_15d/proj15_lags_ridge.pt'))
    parser.add_argument('--host',default=os.environ.get('DONKEY_SIM_HOST','127.0.0.1'))
    parser.add_argument('--port',type=int,default=int(os.environ.get('DONKEY_SIM_PORT','9091')))
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--layouts',type=int,default=4)
    parser.add_argument('--max-episode-steps',type=int,default=2000)
    parser.add_argument('--order-seed',type=int,default=42)
    parser.add_argument('--device',default='cuda')
    add_control_fix_args(parser)
    parser.set_defaults(low_jitter_sync=True,tcp_nodelay=True,observation_timeout=.1,capture_tail_steps=120,save_telemetry=True)
    args=parser.parse_args()
    if args.layouts<1 or args.max_episode_steps<1 or args.reversal_unwind_scale!=1:
        parser.error('Positive limits and frozen steering mapping required')
    if not args.low_jitter_sync or not args.tcp_nodelay or args.observation_timeout!=.1:
        parser.error('Keep the common low-jitter/TCP_NODELAY/100ms settings')
    encoder=VisualGeometryEncoder(args.geometry_head,args.device)
    models={name:VisualStudentActor.load(path,geometry_head_sha256=encoder.head_sha256,device=args.device)
            for name,path in [('baseline',args.baseline),('candidate',args.candidate)]}
    # Stateless symmetric preflight only: no simulator action or temporal-cache mutation.
    warmup={}
    for name,actor in models.items():
        start=time.perf_counter();actor.predict(np.zeros(85,np.float32));warmup[name]=time.perf_counter()-start
    files={'baseline':args.baseline,'candidate':args.candidate,'geometry_head':args.geometry_head}
    with EvalRecord(args,files) as record:
        orders=balanced_orders(args.layouts,args.order_seed)
        record.metadata.update(student_input_contract=OBSERVATION_CONTRACT,teacher_loaded=False,
            actor_warmup_seconds=warmup,orders=orders,
            pairing_protocol='Reload once per layout; reset_car only between actors; balance first-run order.',
            runtime_protocol_note='Both stateless actors prewarmed before simulator connection. Historical single-student defaults are unchanged.')
        record.encoder(encoder)
        write_json(record.path/'order_plan.json',{'orders':orders,'seed_is_model_order_only':True,'evaluation_only':True})
        base=gym.make('donkey-generated-track-v0',conf={'host':args.host,'port':args.port,
                      'cam_resolution':(120,160,3),'max_cte':20.})
        wrapped=DonkeyVaeSACEnv(base,vae=encoder,min_throttle=.2,max_throttle=.7,max_steering=1.,
                    max_steering_diff=.1,n_command_history=20,latent_lags=(2,4),scene_reload_every=0,
                    reward_config=RaffinRewardConfig(max_cte_error=2.,alive_reward=1.5,speed_reward_weight=.15,cte_speed_penalty_weight=.25))
        env=TimeLimit(wrapped,max_episode_steps=args.max_episode_steps)
        controller=base.unwrapped.viewer;scene=controller.handler.SceneToLoad
        try:
            configure_env_control_fixes(wrapped,args)
            for layout,order in enumerate(orders,1):
                print(f'[paired-layout] loading {layout}/{args.layouts}, order={order}',flush=True)
                if not reload_scene(controller,scene):raise RuntimeError(f'Scene reload failed at layout {layout}')
                for place,model in enumerate(order,1):
                    context={'layout':layout,'model':model,'order':place,'episode':len(record.results)+1}
                    result=run_episode(env,encoder,models[model],record,context)
                    record.episode(result);print(result,flush=True)
            summary=paired_summary(record.results)
            assert all(sha256(path)==record.metadata['files'][name]['sha256'] for name,path in files.items())
            record.finish(summary,statistics=paired_statistics(record.results))
        finally:
            try:controller.handler.send_control(0.,0.,1.)
            finally:env.close()


if __name__=='__main__':main()
