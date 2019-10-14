import numpy as np
import tensorflow as tf
from algorithms.ppo import CategoricalModel, GaussianModel, Roller, network_builder, Params
from algorithms.imitation_learning.behavioral_cloning import BehavioralCloning, StorageBehavioralCloning, BehavioralCloningParams
from environment import create_editor_env
from utils.logger import Logger, log


def main(working_dir, config):

    params = Params(working_dir= working_dir, config_file= config)
    
    tf.random.set_seed(params.env.seed)
    np.random.seed(params.env.seed)
    
    env = create_editor_env(params.env)
    LOGGER = Logger(env.env_info.academy_name, working_dir, config) 

    network = network_builder(params.trainer.nn_architecure) \
        (hidden_sizes= params.policy.hidden_sizes, env_info = env.env_info)
        
    model = CategoricalModel(network = network, env_info = env.env_info)
    
    if params.trainer.load_model: 
        log('Loading Model ...')
        model.load_weights(LOGGER.tf_model_path('model_weights'))

    roller = Roller(env, model, params.trainer.steps_per_epoch, params.trainer.gamma, params.trainer.lam)
    bc = BehavioralCloning(model, env.env_info)

    storage = StorageBehavioralCloning(0, env.env_info)
    storage.load()
   
    for epoch in range (params.trainer.epochs):

        vec_obs, vis_obs, acts = storage.sample(BehavioralCloningParams.batch_size_bc)
        loss = bc.update(vec_obs, vis_obs, acts)
        LOGGER.store('Loss Imitation', loss)

        rollouts, infos = roller.rollout()

        for ep_rew, ep_len in zip(infos['ep_rews'], infos['ep_lens']):
                LOGGER.store('EP REW', ep_rew)
                LOGGER.store('EP LEN', ep_len)

        if (epoch % params.trainer.save_freq == 0) or (epoch == params.trainer.epochs - 1):     # Saving every n steps
                log('Saving Model ...')
                model.save_weights(LOGGER.tf_model_path('model_weights'), save_format='tf')
                
        LOGGER.log_metrics(epoch)

    print("Closing Environment")
    env.close()
