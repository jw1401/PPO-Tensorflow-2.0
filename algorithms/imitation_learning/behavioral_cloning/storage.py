import numpy as np
from environment import EnvInfo
import os

PATH = './__WORKING_DIRS__/__Imitation_Data__/'


class StorageBehavioralCloning:
    """
        Storage for DEMONSTRATIONS from Experts
        ---------------------------------------
        ###
    """
    def __init__(self, size, env_info= EnvInfo):
        
        self.vec_obs_buf = np.array([])
        self.vis_obs_buf = np.array([])
        self.act_buf = None

        os.makedirs(os.path.dirname(PATH), exist_ok=True)

    def store_rollout(self, rollouts):

        idxs = np.where(rollouts['actions'] != 0)
        acts = rollouts['actions'][idxs]
        vec_obs = rollouts['vec_obses'][idxs] if rollouts['vec_obses'] is not None else None
        vis_obs = rollouts['vis_obses'][idxs] if rollouts['vis_obses'] is not None else None

        if acts.size == 0:
            return

        if self.act_buf is None:
            self.vec_obs_buf = vec_obs
            self.vis_obs_buf = vis_obs
            self.act_buf = acts
        else:
            self.vec_obs_buf = np.append(self.vec_obs_buf, vec_obs, axis= 0) if vec_obs is not None else None
            self.vis_obs_buf = np.append(self.vis_obs_buf, vis_obs, axis= 0) if vis_obs is not None else None
            self.act_buf = np.append(self.act_buf, acts, axis = 0)
        
    def save(self):

        with open(PATH + 'vec_obs', 'ab') as fo:
            np.save(fo, self.vec_obs_buf)
        with open(PATH + 'vis_obs', 'ab') as fo:
            np.save(fo, self.vis_obs_buf)
        with open(PATH + 'act', 'ab') as fa:
            np.save(fa, self.act_buf)
        
    def load(self):

        self.vec_obs_buf = np.load(PATH + 'vec_obs')
        self.vis_obs_buf = np.load(PATH + 'vis_obs')
        self.act_buf = np.load(PATH + 'act')

    def sample(self, batch_size):

        if self.act_buf.size == 0:
            raise Exception("No DEMONSTRATIONS stored...")

        if self.act_buf.size < batch_size:
            batch_size = self.act_buf.size
        
        idxs = np.random.choice(range(self.act_buf.size), size= batch_size, replace= False)
    
        return self._reformat(idxs)

    def _reformat(self, idxs):
        
        vec_obs = self.vec_obs_buf[idxs]
        vis_obs = self.vis_obs_buf[idxs] if self.vis_obs_buf.size > 1 else None
        act = self.act_buf[idxs]

        return vec_obs, vis_obs, act
