from types import SimpleNamespace

import numpy as np

from tools.collect_visual_student_recovery import collect_episode
from tools.finetune_visual_student_recovery import sample_weights, check_gates


def test_collection_pairs_current_inputs_with_pre_action_labels_and_reset_image():
    order=[]
    class Environment:
        def __init__(self):self.env=SimpleNamespace(last_raw_obs=np.zeros((2,2,3),np.uint8))
        def reset(self):self.i=0;return np.zeros(85,np.float32),self.info()
        def info(self):return {'pos':[self.i,0,0],'car':[0,0,0],'abs_cte':0.,'speed':1.,'lap_count':0}
        def step(self,action):
            self.i+=1;self.env.last_raw_obs.fill(self.i)
            return np.full(85,self.i,np.float32),0.,False,self.i==3,self.info()
    class Actor:
        def predict(self,observation):
            assert observation.shape==(85,)
            order.append(('actor',int(observation[0])))
            return np.zeros(2,np.float32),None
    class LabelTrack:
        def encode(self,image,info):
            order.append(('label',info['pos'][0]))
            return np.full(18,info['pos'][0],np.float32)
        def locate(self,*args):return 310,0,0
    record=SimpleNamespace(step=lambda *args,**kwargs:None)
    data,result=collect_episode(Environment(),Actor(),LabelTrack(),record,1)
    assert order==[('actor',0),('label',0),('actor',1),('label',1),('actor',2),('label',2)]
    np.testing.assert_array_equal(data['observations'][:,0],[0,1,2])
    np.testing.assert_array_equal(data['label_state'][:,0],[0,1,2])
    np.testing.assert_array_equal(data['images'][:,0,0,0],[0,1,2,3])
    assert result['completed'] and result['steps']==3


def test_privileged_weights_exclude_validation_and_emphasize_bend_unwind():
    new=np.array([False,True,True,True,True])
    weights=sample_weights(new,np.array([-1,100,310,310,310]),
                           np.array([False,False,False,True,True]),np.array([True,True,True,True,False]))
    np.testing.assert_array_equal(weights,[1,2,8,16,0])


def test_recovery_gate_requires_both_old_retention_and_new_improvement():
    before={name:{'frames':100,'steer_mae':.1,'missed_unwind_rate':.1}
            for name in ['old_all','old_large','fresh_all','fresh_bend']}
    after={name:dict(value) for name,value in before.items()}
    after['fresh_bend']['steer_mae']=.08
    assert all(check_gates(before,after).values())
    after['old_all']['steer_mae']=.106
    assert not check_gates(before,after)['old_all_mae']
    after['old_all']['steer_mae']=.1;after['fresh_all']['missed_unwind_rate']=None
    assert not check_gates(before,after)['fresh_all_unwind']
    after['fresh_bend']['frames']=20
    assert not check_gates(before,after)['development_bend_sample_count']
