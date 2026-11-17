from types import SimpleNamespace
import json

import gymnasium as gym
import numpy as np
import pytest

from rl.eval_visual_students_paired import balanced_orders,paired_summary


def test_paired_order_is_balanced_and_summary_keeps_layout_pairing():
    orders=balanced_orders(4,42)
    assert sum(order[0]=='baseline' for order in orders)==2
    assert orders==balanced_orders(4,42)
    rows=[]
    for layout in range(1,5):
        for name,cte in [('candidate',layout+.1),('baseline',layout)]:
            rows.append(dict(layout=layout,model=name,completed=True,steps=2,laps=0,mean_abs_cte=cte,
                             max_abs_cte=cte,mean_speed=1.,startup_mean_abs_cte=cte,frames_abs_cte_ge_1=0))
    assert np.isclose(paired_summary(rows)['candidate_minus_baseline']['mean_abs_cte']['mean'],.1)
    with pytest.raises(ValueError):paired_summary(rows[:-1])
    with pytest.raises(ValueError):paired_summary(rows+[rows[0]])


def test_paired_eval_reloads_once_per_pair_and_resets_all_history(tmp_path,monkeypatch):
    import rl.eval_visual_students_paired as evaluation
    counts={'reload':0,'reset':0,'encoder_reset':0}
    class Encoder:
        uses_telemetry=False;z_size=15;crop_top=0;head_sha256='head';model_name='image_only'
        def reset_episode(self):counts['encoder_reset']+=1
        def encode(self,image):return np.full(15,float(image[0,0,0]),np.float32)
        def state_dict(self):return {}
    class Actor:
        def predict(self,x):
            assert x.shape==(85,)
            return np.array([.1,0.],np.float32),None
    class Base(gym.Env):
        observation_space=gym.spaces.Box(0,255,(2,2,3),np.uint8)
        action_space=gym.spaces.Box(-1,1,(2,),np.float32)
        def __init__(self):self.viewer=SimpleNamespace(handler=SimpleNamespace(SceneToLoad='test',send_control=lambda *a:None))
        def reset(self,**kwargs):
            counts['reset']+=1
            return np.zeros((2,2,3),np.uint8),{'cte':0.,'speed':0.,'lap_count':0}
        def step(self,action):return np.ones((2,2,3),np.uint8),0.,False,False,{'cte':.1,'speed':1.,'lap_count':0}
    for name in ['base.pt','candidate.pt','head.pt']:(tmp_path/name).write_text('mock')
    monkeypatch.setattr(evaluation,'VisualGeometryEncoder',lambda *a:Encoder())
    monkeypatch.setattr(evaluation.VisualStudentActor,'load',lambda *a,**kw:Actor())
    monkeypatch.setattr(evaluation.gym,'make',lambda *a,**kw:Base())
    monkeypatch.setattr(evaluation,'configure_env_control_fixes',lambda *a:None)
    def reload(*args):counts['reload']+=1;return True
    monkeypatch.setattr(evaluation,'reload_scene',reload)
    monkeypatch.setattr(evaluation.sys,'argv',['paired','--baseline',str(tmp_path/'base.pt'),'--candidate',str(tmp_path/'candidate.pt'),
        '--geometry-head',str(tmp_path/'head.pt'),'--output-dir',str(tmp_path/'run'),'--layouts','4','--max-episode-steps','2','--device','cpu'])
    evaluation.main()
    assert counts=={'reload':4,'reset':8,'encoder_reset':8}
    results=json.loads((tmp_path/'run/results.json').read_text())
    assert len(results['episodes'])==8
    for result in results['episodes']:
        with np.load(tmp_path/'run'/result['inputs']) as data:
            assert set(data.files)=={'observations','raw_actions'}
            np.testing.assert_array_equal(data['observations'][0,45:],np.zeros(40))
            np.testing.assert_array_equal(data['observations'][0,:45],np.zeros(45))
