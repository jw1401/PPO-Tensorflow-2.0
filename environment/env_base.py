import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass
from mlagents.envs.environment import UnityEnvironment
from utils import log2logger


@dataclass
class EnvInfo():
    env_name: str
    academy_name: str
    num_agents: int
    is_discrete: bool
    is_visual: bool
    is_vector: bool
    is_frame_stacking: bool
    stack_size: int
    shapes: dict
    act_size: int
    

def _start_env(file_name="", seed=0, worker_id=0):

    log2logger("Starting Environment: " + str(worker_id))

    if file_name == "":
        env = UnityEnvironment(file_name=None, seed=seed)
    else:
        env = UnityEnvironment(file_name=file_name, seed=seed, worker_id=worker_id)

    return env


class Stacker():
    """
        Frame Stacker
        -------------

            Stacks observations frames to the desired stack-size defined in env-params
            Input:      (84, 84, 3) or (, 5)
            Example:    Stack-Size = 3
                        Returns something of shape like (None, 84, 84, 9) or (None, 15) 
    """
    def __init__(self, stack_size= 3):

        self._stack_size = stack_size
        self._frames = deque([], maxlen=self._stack_size)

    def _stack(self):

        assert len(self._frames) == self._stack_size
        _out = np.concatenate(self._frames, axis=-1)        # Stacks last axis (84, 84, 3) -> 3 is stacked n times
        _out = _out.astype('float32')
        return _out

    def frame_stack(self, o, reset):

        if reset:
            [self._frames.append(o) for _ in range(self._stack_size)]
        else:
            self._frames.append(o)

        return self._stack()                                # Reshaping [None, : , :, :] is not necassary

    
class EnvBase():
    """
        Environment Base Class
        ----------------------

            Takes Env-Parameters and worker-id as input

            Starts the Environment and checks what kind of environment ist is
                - Vector
                - Visual
                - Frame Stacked
            
            Get Brains, Shapes etc.
            
            Sets the Env-Info Object for later use
    """
    def __init__(self, env_params= None, worker_id= 0):

        self._env_name = env_params.env_name
        self._seed = env_params.seed
        self._is_frame_stacking = env_params.frame_stacking
        self._stack_size = env_params.frames_stack_size

        self._is_discrete = False
        self._is_visual = False
        self._is_vector = False
        self._is_grayscale = False
        self._shapes = dict()
        self._env = _start_env(self._env_name, self._seed, worker_id)

        self._default_brain_name = self._env.brain_names[0]
        self._default_brain = self._env.brains[self._default_brain_name]
        self._info = self._env.reset()[self._default_brain_name]

        if self._default_brain.number_visual_observations is not 0:         # Check if there are Visual Observations and set _is_visual
            
            _vis_shape = self._info.visual_observations[0][0].shape

            self._is_visual = True
            if _vis_shape[2] is not 3: self._is_grayscale = True

            self._shapes['vis'] = _vis_shape
            
            if self._is_frame_stacking:
                self._shapes['vis'] = (_vis_shape[0], _vis_shape[1], _vis_shape[2] * self._stack_size)
            
            self.show_images_on_start()
        else:
            self._shapes['vis'] = 0

        if self._default_brain.vector_observation_space_size is not 0:      # Check for Vector Observations

            self._is_vector = True
            self._shapes['vec'] = (self._info.vector_observations.shape[1], ) 
            
            if self._is_frame_stacking:
                self._shapes['vec'] = (self._info.vector_observations.shape[1] * self._stack_size, )
        else:
            self._shapes['vec'] = 0
        
        self._is_discrete = True if self._default_brain.vector_action_space_type == 'discrete' else False
        self._num_agents = len(self._info.agents)                           # How much agents linked to one brain in editor env
        self._act_size = self._default_brain.vector_action_space_size[0]

        self.vec_stackers = []
        self.vis_stackers = []

        if self._is_frame_stacking:
            for i in range(self._num_agents):
                self.vec_stackers.append(Stacker(self._stack_size))
                self.vis_stackers.append(Stacker(self._stack_size))

        self._env_info = EnvInfo(   self._env_name, 
                                    self._env.academy_name, 
                                    self._num_agents,
                                    self._is_discrete,
                                    self._is_visual,
                                    self._is_vector,
                                    self._is_frame_stacking, 
                                    self._stack_size, 
                                    self._shapes, 
                                    self._act_size)

    @property
    def env_info(self):
        return self._env_info

    def show_images_on_start(self):

        if self._is_visual:  
            if self._is_grayscale:
                plt.imshow(self._info.visual_observations[0][0][None, :, :, 0][0], cmap='gray')
            else:
                plt.imshow(self._info.visual_observations[0][0][None, :, :, :][0])  # RGB
    
        plt.show(block=False)
        plt.pause(1)
        plt.close()

    def show_stacked_images(self, o):

        images = np.dsplit(o[0], 3)
        fig, axs = plt.subplots(1, self._stack_size)
        fig.suptitle('Images')
        for j in range(self._stack_size):
            axs[j].imshow(images[j])
        plt.show()

    def reset(self):
        pass

    def step(self):
        pass
