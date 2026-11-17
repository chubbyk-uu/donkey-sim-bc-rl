"""One preregistered camera-only student fine-tune on fresh student trajectories."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from stable_baselines3 import SAC

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rl.visual_student import VisualStudentActor, map_actions, OBSERVATION_CONTRACT
from tools.train_visual_student import teacher_labels
from tools.compare_geometry_control import action_metrics


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def sample_weights(is_new, track_indices, unwind, train):
    """Truth-derived selection stays outside the actor input; validation has zero weight."""
    weights=np.ones(len(train),np.float32)
    weights[is_new]=2
    weights[is_new & (track_indices>=270) & (track_indices<=345)]=8
    weights[unwind]*=2
    weights[~train]=0
    return weights


def check_gates(before,after,minimum_bend_frames=50):
    checks={'development_bend_sample_count':after['fresh_bend']['frames']>=minimum_bend_frames}
    for name,ratio in [('fresh_bend',.9),('fresh_all',1.05),('old_all',1.05),('old_large',1.1)]:
        checks[name+'_mae']=after[name].get('steer_mae',float('inf'))<=ratio*before[name].get('steer_mae',0.)
    for name in ['old_all','fresh_all']:
        old,new=before[name].get('missed_unwind_rate'),after[name].get('missed_unwind_rate')
        checks[name+'_unwind']=old is not None and new is not None and new<=old+.02
    return checks


@torch.no_grad()
def evaluate(actor,x,y,state,masks):
    raw=actor.predict(x)[0]
    history=x[:,45:]
    commands=map_actions(torch.from_numpy(raw),torch.from_numpy(history)).numpy()
    target=map_actions(torch.from_numpy(y),torch.from_numpy(history)).numpy()
    return {name:action_metrics(raw,commands,y,target,history,mask,state) for name,mask in masks.items()},raw


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-dir',type=Path,default=Path('logs/visual_student_bend_recovery_v1'))
    parser.add_argument('--old-dir',type=Path,default=Path('logs/visual_student_no_motion_v1'))
    parser.add_argument('--old-cache',type=Path,default=Path('logs/round_b_negative_geometry_train_v1'))
    parser.add_argument('--export',type=Path,default=Path('models/visual_student_bend_recovery_v1/student.pt'))
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    root=args.experiment_dir;plan=json.loads((root/'plan.json').read_text());collection=root/'collection';out=root/'fit'
    if args.export.exists() or out.exists():parser.error('Preserve existing candidates; use a fresh explicit experiment/export')
    cm=json.loads((collection/'metadata.json').read_text())
    if cm['status']!='completed':raise ValueError('Collection did not complete; no automatic partial fit')
    episodes=json.loads((collection/'results.json').read_text())['episodes']
    assert len(episodes)==plan['collection']['episodes']==8
    split=json.loads((collection/'split_manifest.json').read_text())
    assert split['train']==plan['collection']['train_layouts'] and split['development']==plan['collection']['development_layouts']
    old_spec=json.loads((args.old_dir/'training_manifest.json').read_text())
    assert sha(args.old_cache/'training_manifest.json')==old_spec['source_cache_manifest_sha256']
    base_path=Path(plan['frozen_files']['student']['path']);head_path=Path(plan['frozen_files']['geometry_head']['path'])
    teacher_path=Path(old_spec['teacher_labels_only'])
    assert sha(teacher_path)==old_spec['teacher_sha256']
    frozen={str(p):sha(p) for p in [base_path,head_path,teacher_path]}
    for key,p in [('student',base_path),('geometry_head',head_path)]:
        assert sha(p)==plan['frozen_files'][key]['sha256']==cm['files'][key]['sha256']
    # Read all student inputs before opening the isolated state/label branch.
    with np.load(args.old_dir/'student_inputs.npz',allow_pickle=False) as data:
        old_x=data['observations'];old_train=data['train'];old_val=data['validation'];old_groups=data['groups'];old_steps=data['step']
    new_x=[];new_train=[];new_groups=[];new_steps=[];provenance=[]
    for e in episodes:
        ep=e['episode'];path=collection/f'student_inputs_{ep:02d}.npz'
        with np.load(path,allow_pickle=False) as data:
            assert set(data.files)=={'observations','raw_actions'}
            obs=data['observations'];assert obs.shape==(e['steps'],85)
        new_x.append(obs);new_train.extend([ep in split['train']]*len(obs))
        new_groups.extend([f'{collection.parent.name}/layout_{ep}']*len(obs));new_steps.extend(range(len(obs)))
        provenance.append({'episode':ep,'split':e['split'],'frames':len(obs),'completed':e['completed'],
                           'student_inputs_sha256':sha(path),'label_state_sha256':sha(collection/f'label_state_{ep:02d}.npz')})
    x=np.concatenate([old_x,*new_x]);assert x.shape[1]==85 and np.isfinite(x).all()
    new_train=np.asarray(new_train,bool);n_old=len(old_x)
    train=np.r_[old_train,new_train];old_val=np.r_[old_val,np.zeros(len(x)-n_old,bool)]
    fresh_val=np.r_[np.zeros(n_old,bool),~new_train]
    groups=np.r_[old_groups,np.asarray(new_groups)];steps=np.r_[old_steps,np.asarray(new_steps)]
    assert not np.any(train & (old_val|fresh_val))
    assert not set(groups[train]) & set(groups[old_val|fresh_val])
    # Separate teacher/scoring branch: the feature vector above is already complete.
    cache_spec=json.loads((args.old_cache/'training_manifest.json').read_text())
    with np.load(args.old_cache/'feature_cache.npz',allow_pickle=False) as cache:
        use=np.isin(cache['run'],sorted({e['run'] for e in cache_spec['captures']}))
        old_state=cache['state'][use];old_history=cache['history'][use]
    np.testing.assert_array_equal(old_history,old_x[:,45:])
    assert len(old_state)==n_old
    states=[old_state];indices=[np.full(n_old,-1,np.int32)]
    for e in episodes:
        with np.load(collection/f"label_state_{e['episode']:02d}.npz",allow_pickle=False) as data:
            states.append(data['state']);indices.append(data['track_index'])
    state=np.concatenate(states);indices=np.concatenate(indices)
    torch.set_num_threads(4);torch.manual_seed(plan['fit']['seed']);rng=np.random.default_rng(plan['fit']['seed'])
    teacher=SAC.load(str(teacher_path),device='cpu')
    y=teacher_labels(teacher,state,x[:,45:])
    with np.load(args.old_dir/'teacher_labels.npz',allow_pickle=False) as data:
        old_label_error=float(np.max(np.abs(y[:n_old]-data['raw_actions'])))
        assert old_label_error<1e-5
    target=map_actions(torch.from_numpy(y),torch.from_numpy(x[:,45:])).numpy()
    previous=x[:,-2];heading=np.degrees(np.arctan2(state[:,1],state[:,2]))
    unwind=(abs(previous)>=.3)&(abs(target[:,0])<abs(previous)-.02)&((abs(state[:,0]*2)>=.5)|(abs(heading)>=8))
    is_new=np.arange(len(x))>=n_old
    weights=sample_weights(is_new,indices,unwind,train)
    masks={'old_all':old_val,'old_large':old_val&(abs(state[:,0]*2)>=1),
           'fresh_all':fresh_val,'fresh_bend':fresh_val&(indices>=270)&(indices<=345)}
    if masks['fresh_bend'].sum()<plan['offline_gates']['new_development_bend_frames_min']:
        raise ValueError('Insufficient held development bend coverage; no fit or automatic new collection')
    out.mkdir()
    np.savez(out/'student_inputs.npz',observations=x)
    np.savez(out/'teacher_labels.npz',raw_actions=y,state=state)
    np.savez(out/'training_selection.npz',train=train,old_validation=old_val,fresh_validation=fresh_val,
             groups=groups,steps=steps,weights=weights,track_index=indices)
    spec={'plan_sha256':sha(root/'plan.json'),'collection':provenance,'old_inputs_sha256':sha(args.old_dir/'student_inputs.npz'),
          'old_labels_sha256':sha(args.old_dir/'teacher_labels.npz'),'old_teacher_label_replay_max_error':old_label_error,
          'frozen_files':frozen,'observation_contract':OBSERVATION_CONTRACT,'training_frames':int(train.sum()),
          'old_training_frames':int(old_train.sum()),'new_training_frames':int(new_train.sum()),
          'old_validation_frames':int(old_val.sum()),'fresh_validation_frames':int(fresh_val.sum()),
          'fresh_validation_bend_frames':int(masks['fresh_bend'].sum()),'train_and_validation_groups_disjoint':True,
          'failed_fresh_trajectories_retained':True,'prior_24_layout_evaluation_used_for_fit':False}
    write(out/'training_manifest.json',spec)
    actor=VisualStudentActor.load(base_path,geometry_head_sha256=sha(head_path),device=args.device)
    before,_=evaluate(actor,x,y,state,masks)
    actor.to(args.device);best=copy.deepcopy(actor.state_dict());best_score=1.;best_epoch=0
    optim=torch.optim.AdamW(actor.parameters(),lr=plan['fit']['lr'],weight_decay=plan['fit']['weight_decay'])
    tx=torch.from_numpy(x).to(args.device);ty=torch.from_numpy(y).to(args.device);tw=torch.from_numpy(weights).to(args.device)
    train_indices=np.flatnonzero(train);epochs=[];started=time.monotonic()
    for epoch in range(1,plan['fit']['epochs_cap']+1):
        actor.train();losses=[]
        for batch in np.array_split(rng.permutation(train_indices),int(np.ceil(len(train_indices)/plan['fit']['batch_size']))):
            if time.monotonic()-started>plan['fit']['seconds_cap']:break
            obs=tx[batch];raw=actor(obs);cmd=map_actions(raw,obs[:,45:]);target_cmd=map_actions(ty[batch],obs[:,45:])
            error=(cmd[:,0]-target_cmd[:,0]).square()+.1*(raw[:,0]-ty[batch,0]).square()+.5*(cmd[:,1]-target_cmd[:,1]).square()
            loss=(error*tw[batch]).sum()/tw[batch].sum()
            optim.zero_grad();loss.backward();optim.step();losses.append(float(loss.detach()))
        actor.eval();current,_=evaluate(actor,x,y,state,masks)
        score=.5*current['old_all']['steer_mae']/before['old_all']['steer_mae']+.5*current['fresh_bend']['steer_mae']/before['fresh_bend']['steer_mae']
        epochs.append({'epoch':epoch,'loss':float(np.mean(losses)) if losses else None,'selection_score':score,
                       'old_mae':current['old_all']['steer_mae'],'fresh_bend_mae':current['fresh_bend']['steer_mae']})
        print(epochs[-1],flush=True)
        if score<best_score:best_score=score;best=copy.deepcopy(actor.state_dict());best_epoch=epoch
        if time.monotonic()-started>plan['fit']['seconds_cap']:break
    actor.load_state_dict(best);actor.eval();after,raw=evaluate(actor,x,y,state,masks)
    gates=check_gates(before,after,plan['offline_gates']['new_development_bend_frames_min'])
    passed=all(gates.values())
    actor.save(args.export,geometry_head_sha256=sha(head_path),metadata={'baseline_sha256':sha(base_path),'offline_gate_passed':passed,'training_plan_sha256':sha(root/'plan.json')})
    loaded=VisualStudentActor.load(args.export,geometry_head_sha256=sha(head_path),device=args.device)
    np.testing.assert_allclose(loaded.predict(x[:512])[0],raw[:512],atol=1e-6,rtol=0)
    assert all(sha(p)==h for p,h in frozen.items())
    np.savez(out/'development_predictions.npz',raw_actions=raw[old_val|fresh_val],teacher_actions=y[old_val|fresh_val])
    report={'baseline':before,'candidate':after,'gates':gates,'offline_gate_passed':passed,'best_epoch':best_epoch,
            'epochs':epochs,'fit_seconds':time.monotonic()-started,'fit_runs':1,'baseline_overwritten':False,
            'candidate_checkpoint':str(args.export),'candidate_sha256':sha(args.export),
            'runtime_inputs':['camera','self_commands'],'candidate_live_runs':0,
            'next':'Eligible for a separately scoped fresh-layout paired screen; no superiority claim.' if passed else 'Stop; no automatic refit or candidate live evaluation.'}
    write(out/'results.json',report);write(root/'decision.json',report)
    source=out/'source';source.mkdir()
    for path in [Path(__file__),Path('tools/collect_visual_student_recovery.py'),Path('rl/visual_student.py'),Path('tools/compare_geometry_control.py')]:shutil.copy2(path,source/path.name)
    print('OFFLINE GATES',gates,'PASSED',passed,flush=True)


if __name__=='__main__':main()
