import os
import cloudpickle
import numpy as np
from . import Env, EnvInfo, BatchedEnvBase


def create_editor_env(env_params):
    env = Env(env_params= env_params, worker_id=0)
    return BatchedEditorEnv(env)


class BatchedEditorEnv(BatchedEnvBase):

    '''
    Batched editor env with multiple agents connected to single brain --> copy paste env in ml-agents
    '''

    def __init__(self, env= Env):
        super().__init__()

        self.env = env
        self.env_info = self.env.env_info

        self.ep_rew = [0] * self.num_envs
        self.ep_len = [0] * self.num_envs
        self.info = [None] * self.num_envs

    @property
    def num_envs(self):
        return self.env_info.num_agents

    def reset(self):
        """
            Reset all envs returns initial obs, rews, dones
        """
        vec_obs, vis_obs, rews, dones = self.env.reset()

        vec_obs = np.array(vec_obs, dtype=np.float32) if self.env_info.is_vector else None
        vis_obs = np.array(vis_obs, dtype=np.float32) if self.env_info.is_visual else None

        return vec_obs, vis_obs, np.array(rews, dtype=np.float32), np.array(dones, dtype=np.bool)

    def step(self, actions):
        """
            Step all envs with a list of actions for each env
            step_result contains obs, rews, dones in array format with data of each agent
        """
        vec_obs, vis_obs, rews, dones = self.env.step(actions)

        for i, rew in enumerate(rews):

            self.info[i] = None
            self.ep_rew[i] += rew
            self.ep_len[i] += 1

            if dones[i]:                                                                # Env is reset on done (if checked in Unity Agent) in Unity Editor and returns reset obs
                self.info[i] = {'ep_rew': self.ep_rew[i], 'ep_len': self.ep_len[i]}
                self.ep_rew[i] = 0
                self.ep_len[i] = 0

        vec_obs = np.array(vec_obs, dtype=np.float32) if self.env_info.is_vector else None
        vis_obs = np.array(vis_obs, dtype=np.float32) if self.env_info.is_visual else None

        return vec_obs, vis_obs, np.array(rews, dtype=np.float32), np.array(dones, dtype=np.bool), self.info

    def close(self):
        """
            Close all envs
        """
        self.env.close()
