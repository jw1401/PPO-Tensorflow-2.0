import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from .env_base import EnvBase


class Env(EnvBase):

    def __init__(self, env_params= None, worker_id= 0):
        super().__init__(env_params= env_params, worker_id= worker_id)

    def reset(self):

        vec_obs, vis_obs = None, None
        
        info = self._env.reset()[self._default_brain_name]

        if self._is_vector:
            vec_obs = info.vector_observations

        if self._is_visual:
            vis_obs = info.visual_observations[0]

        if self._is_frame_stacking:
            vec_obs, vis_obs = self._stack_agents_obs(vec_obs, vis_obs, True)   # Reset = True

        return vec_obs, vis_obs, info.rewards, info.local_done

    def step(self, a):

        vec_obs, vis_obs = None, None

        if self._num_agents is 1 and self._is_discrete: a = [int(a)]            # Little hack for discrete case and one agent to avoid error
        info = self._env.step(a)[self._default_brain_name]

        if self._is_vector:
            vec_obs = info.vector_observations

        if self._is_visual:
            vis_obs = info.visual_observations[0]                               # Take index 0 because we only allow one camera per agent

        if self._is_frame_stacking:
            vec_obs, vis_obs = self._stack_agents_obs(vec_obs, vis_obs, False)  # Reset = False

        return vec_obs, vis_obs, info.rewards, info.local_done

    def close(self):
        self._env.close()

    def _stack_agents_obs(self, vec_obs, vis_obs, reset= False):
        """
            Stacks the obs-frames for all agents in the Environment that are linked to default brain\n
            This is the case if there are multiple agents in Editor mode with frame stacking enabled

                Returns:    stacked vec-obs and vis-obs for number of n agents in the environemnt
        """
        vec, vis = [], []

        for i in range(self._num_agents):
            vec.append(self.vec_stackers[i].frame_stack(vec_obs[i], reset)) if self._is_vector else None
            vis.append(self.vis_stackers[i].frame_stack(vis_obs[i], reset)) if self._is_visual else None

        vec_obs = np.array(vec) if self._is_vector else None
        vis_obs = np.array(vis) if self._is_visual else None

        # self.show_stacked_images(vis_obs[0])

        return vec_obs, vis_obs
