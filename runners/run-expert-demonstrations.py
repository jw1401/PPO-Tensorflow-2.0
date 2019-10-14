import numpy as np
import tensorflow as tf
from algorithms.ppo import Roller, Params
from algorithms.imitation_learning.behavioral_cloning import ImitationModel, StorageBehavioralCloning
from environment import create_editor_env
from utils.logger import Logger, log


def main(working_dir, config):

    params = Params(working_dir= working_dir, config_file= config)
    
    tf.random.set_seed(params.env.seed)
    np.random.seed(params.env.seed)
    
    env = create_editor_env(params.env)
    LOGGER = Logger(env.env_info.academy_name, working_dir, config) 
    
    expert_imitation = ImitationModel(env.env_info)

    roller = Roller(env, expert_imitation, params.trainer.steps_per_epoch, params.trainer.gamma, params.trainer.lam)

    storage = StorageBehavioralCloning(0, env.env_info)
   
    for epoch in range (params.trainer.epochs):

        rollouts, infos = roller.rollout()
        storage.store_rollout(rollouts)
        storage.save()

        for ep_rew, ep_len in zip(infos['ep_rews'], infos['ep_lens']):
                LOGGER.store('EP REW', ep_rew)
                LOGGER.store('EP LEN', ep_len)
                
        LOGGER.log_metrics(epoch)

    print("Closing Environment")
    env.close()
