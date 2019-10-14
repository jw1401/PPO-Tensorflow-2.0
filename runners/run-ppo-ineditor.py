import numpy as np
import tensorflow as tf
from environment import create_editor_env
from algorithms.ppo import CategoricalModel, GaussianModel, Roller, network_builder, PolicyCombinedLoss, Params

from utils import Logger, log


def main(working_dir, config):

    params = Params(working_dir= working_dir, config_file= config)

    tf.random.set_seed(params.env.seed)
    np.random.seed(params.env.seed)
    
    env = create_editor_env(params.env)
    LOGGER = Logger(env.env_info.academy_name, working_dir, config)

    network = network_builder(params.trainer.nn_architecure) \
        (hidden_sizes=params.policy.hidden_sizes, env_info=env.env_info)
    
    model = CategoricalModel if env.env_info.is_discrete else GaussianModel
    model = model(network=network, env_info=env.env_info)                                    

    if params.trainer.load_model:
        log('Loading Model ...')
        model.load_weights(LOGGER.tf_model_path('model_weights'))

    roller = Roller(env, model, params.trainer.steps_per_epoch, params.trainer.gamma, params.trainer.lam)
    ppo = PolicyCombinedLoss(model = model, num_envs = env.num_envs)

    for epoch in range (params.trainer.epochs):

        rollouts, infos = roller.rollout()
        outs = ppo.update(rollouts)

        LOGGER.store('Loss Pi', outs['pi_loss'])
        LOGGER.store('Loss V', outs['v_loss'])
        LOGGER.store('Loss Ent', outs['entropy_loss'])
        LOGGER.store('Appr Ent', outs['approx_ent'])
        LOGGER.store('KL DIV', outs['approx_kl'])

        for ep_rew, ep_len in zip(infos['ep_rews'], infos['ep_lens']):
                LOGGER.store('EP REW', ep_rew)
                LOGGER.store('EP LEN', ep_len)

        if (epoch % params.trainer.save_freq == 0) or (epoch == params.trainer.epochs - 1):
                log('Saving Model ...')
                model.save_weights(LOGGER.tf_model_path('model_weights'), save_format='tf')
                
        LOGGER.log_metrics(epoch)

    env.close()
