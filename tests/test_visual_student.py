from collections import deque
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from rl.train_vae_sac import DonkeyVaeSACEnv
from rl.visual_student import (GeometryProjection, OBSERVATION_CONTRACT, TEACHER_GEOMETRY,
                               VisualGeometryEncoder, VisualStudentActor, map_actions,
                               padded_lag_indices, prepare_visual_inputs, student_observation)


class ImageFeatures:
    crop_top = 0

    def grid_features(self, images):
        result = torch.zeros((len(images), 6144))
        result[:, 0] = torch.tensor(images.mean(axis=(1, 2, 3)) / 255, dtype=torch.float32)
        result[:, 1] = 1
        return result


@pytest.fixture
def geometry_head(tmp_path):
    rng = np.random.default_rng(10)
    path = tmp_path / 'geometry.pt'
    torch.save({'format':'proj15_lags_ridge_v1','lags':[2,4],'crop_top':0,
                'w1':torch.tensor(rng.normal(size=(6144,15)).astype(np.float32)*.01),
                'b1':torch.zeros(15),'w2':torch.tensor(rng.normal(size=(45,15)).astype(np.float32)*.1),
                'b2':torch.zeros(15)},path)
    return path


def test_cached_inputs_match_two_level_live_padding_without_crossing_resets(geometry_head):
    images=np.stack([np.full((8,8,3),v,np.uint8) for v in range(22)])
    features=ImageFeatures().grid_features(images).numpy()
    groups=np.array(['a']*12+['b']*10);steps=np.r_[np.arange(12),np.arange(10)]
    history=np.random.default_rng(0).normal(size=(22,40)).astype(np.float32)
    projection=GeometryProjection(geometry_head)
    cached,eligible,full,_=prepare_visual_inputs(features,history,groups,steps,projection)
    encoder=VisualGeometryEncoder(geometry_head,dino_encoder=ImageFeatures())
    online=[];past=deque(maxlen=5)
    for i,image in enumerate(images):
        if steps[i]==0:
            encoder.reset_episode();past.clear()
        geom=encoder.encode(image);past.appendleft(geom)
        while len(past)<5:past.append(geom)
        online.append(student_observation(np.stack([past[j] for j in [0,2,4]]),history[i]))
    np.testing.assert_allclose(cached,online,atol=1e-6,rtol=1e-5)
    assert eligible.all()
    np.testing.assert_array_equal(full,steps>=8)
    # Unreset tail sequences must discard the full effective eight-step prefix.
    _,eligible_tail,_,_=prepare_visual_inputs(features[:12],history[:12],np.array(['tail']*12),np.arange(100,112),projection)
    np.testing.assert_array_equal(eligible_tail,np.arange(12)>=8)


def test_missing_steps_start_new_context_and_duplicate_steps_fail():
    idx,full,reset=padded_lag_indices(['a']*5,[0,1,2,8,9])
    assert idx[3].tolist()==[3,3,3]
    assert reset.tolist()==[True,True,True,False,False]
    with pytest.raises(ValueError,match='Duplicate'):
        padded_lag_indices(['a','a'],[0,0])


def test_student_encoding_never_reads_any_telemetry(geometry_head):
    class Forbidden(dict):
        def get(self,*args):raise AssertionError('Telemetry read by student')
        def __getitem__(self,key):raise AssertionError('Telemetry read by student')
    encoder=VisualGeometryEncoder(geometry_head,dino_encoder=ImageFeatures())
    base=gym.Env()
    base.observation_space=gym.spaces.Box(0,255,(8,8,3),np.uint8)
    base.action_space=gym.spaces.Box(-1,1,(2,),np.float32)
    env=DonkeyVaeSACEnv(base,vae=encoder,latent_lags=(2,4),n_command_history=20)
    assert env.observation_space.shape==(85,)
    actor=VisualStudentActor().eval()
    image=np.full((8,8,3),30,np.uint8)
    outputs=[]
    for telemetry in [Forbidden(), {'vel':(999,999,999),'cte':999,'pos':(1,2,3),'gyro':(999,999,999)}]:
        encoder.reset_episode();env._latents.clear()
        observations=env._build_obs(image,telemetry)
        outputs.append((observations,actor.predict(observations)[0]))
    np.testing.assert_array_equal(outputs[0][0],outputs[1][0])
    np.testing.assert_array_equal(outputs[0][1],outputs[1][1])
    assert not encoder.uses_telemetry and not hasattr(encoder,'track')


def test_teacher_initialization_drops_motion_and_retains_geometry_history():
    torch.manual_seed(1)
    teacher=SimpleNamespace(actor=SimpleNamespace(latent_pi=nn.Sequential(nn.Linear(58,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU()),mu=nn.Linear(256,2)))
    student=VisualStudentActor();student.initialize_from_teacher(teacher)
    x=torch.randn(10,85);truth=torch.zeros(10,58)
    truth[:,TEACHER_GEOMETRY]=x[:,:15];truth[:,18:]=x[:,45:]
    expected=torch.tanh(teacher.actor.mu(teacher.actor.latent_pi(truth)))
    torch.testing.assert_close(student(x),expected)
    assert torch.count_nonzero(student.net[0].weight[:,15:45])==0


def test_checkpoint_enforces_contract_hash_and_predict_shape(tmp_path):
    student=VisualStudentActor().eval();path=tmp_path/'student.pt'
    student.save(path,geometry_head_sha256='head')
    loaded=VisualStudentActor.load(path,geometry_head_sha256='head')
    x=np.zeros((2,85),np.float32)
    np.testing.assert_array_equal(student.predict(x)[0],loaded.predict(x)[0])
    with pytest.raises(ValueError,match='hash'):
        VisualStudentActor.load(path,geometry_head_sha256='wrong')
    with pytest.raises(ValueError):loaded.predict(np.zeros(58))
    saved=torch.load(path,weights_only=True);saved['observation_contract']['motion_truth']=True
    torch.save(saved,tmp_path/'invalid.pt')
    with pytest.raises(ValueError,match='85D'):
        VisualStudentActor.load(tmp_path/'invalid.pt',geometry_head_sha256='head')


def test_torch_mapper_matches_production_and_allows_cloning_gradients():
    raw=torch.tensor([[1.,0.],[-1.,1.]],requires_grad=True)
    history=torch.zeros((2,40));history[:,-2]=torch.tensor([-.5,.5])
    command=map_actions(raw,history)
    np.testing.assert_allclose(command.detach(),[[-.30002,.45],[.30002,.7]],atol=1e-6)
    (command[:,1].sum()+raw.square().sum()*.1).backward()
    assert torch.isfinite(raw.grad).all()


def test_evaluation_entry_runs_with_no_motion_or_pose_fields(geometry_head, tmp_path, monkeypatch):
    import rl.eval_visual_student as evaluation

    class CameraOnlyEnvironment(gym.Env):
        observation_space=gym.spaces.Box(0,255,(8,8,3),np.uint8)
        action_space=gym.spaces.Box(-1,1,(2,),np.float32)

        def __init__(self):
            self.viewer=SimpleNamespace(handler=SimpleNamespace(SceneToLoad='test'))

        def reset(self,**kwargs):
            return np.zeros((8,8,3),np.uint8),{'cte':0.,'speed':0.,'lap_count':0}

        def step(self,action):
            return np.ones((8,8,3),np.uint8),0.,False,False,{'cte':.1,'speed':1.,'lap_count':0}

    encoder=VisualGeometryEncoder(geometry_head,dino_encoder=ImageFeatures())
    checkpoint=tmp_path/'student.pt'
    VisualStudentActor().save(checkpoint,geometry_head_sha256=encoder.head_sha256)
    monkeypatch.setattr(evaluation,'VisualGeometryEncoder',lambda *a:encoder)
    monkeypatch.setattr(evaluation.gym,'make',lambda *a,**kw:CameraOnlyEnvironment())
    monkeypatch.setattr(evaluation,'configure_env_control_fixes',lambda *a:None)
    monkeypatch.setattr('rl.train_vae_sac.reload_scene',lambda *a:True)
    monkeypatch.setattr(evaluation.sys,'argv',['eval','--student',str(checkpoint),'--geometry-head',str(geometry_head),
        '--episodes','1','--max-episode-steps','2','--capture-tail-steps','0','--device','cpu','--output-dir',str(tmp_path/'eval')])
    evaluation.main()
    metadata=json.loads((tmp_path/'eval/metadata.json').read_text())
    assert metadata['status']=='completed'
    assert metadata['student_input_contract']['sources']==['camera','self_commands']
    result=json.loads((tmp_path/'eval/results.json').read_text())
    assert result['episodes'][0]['steps']==2 and result['episodes'][0]['completed']
    with np.load(tmp_path/'eval/policy_inputs_episode_01.npz') as inputs:
        assert set(inputs.files)=={'observations','raw_actions'}
        assert inputs['observations'].shape==(2,85)
        np.testing.assert_array_equal(inputs['observations'][0,45:],np.zeros(40))
