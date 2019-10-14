
import numpy as np
import tensorflow as tf
from algorithms.ppo import CategoricalModel, GaussianModel, Roller, PolicyCombinedLoss, network_builder, Params
from environment import create_batched_env
from utils import Logger, log


def main(working_dir, config):

    params = Params(working_dir= working_dir, config_file= config)          # Get Configuration | HORIZON = Steps per epoch

    tf.random.set_seed(params.env.seed)                                     # Set Random Seeds for np and tf
    np.random.seed(params.env.seed)

    env = create_batched_env(params.env.num_envs, params.env)               # Create Environment in multiprocessing mode
    LOGGER = Logger(env.env_info.academy_name, working_dir, config)         # Set Logger

    network = network_builder(params.trainer.nn_architecure) \
        (hidden_sizes=params.policy.hidden_sizes, env_info=env.env_info)    # Build Neural Network with Forward Pass

    model = CategoricalModel if env.env_info.is_discrete else GaussianModel
    model = model(network=network, env_info=env.env_info)                   # Build Model for Discrete or Continuous Spaces
    
    if params.trainer.load_model:
        log('Loading Model ...')
        model.load_weights(LOGGER.tf_model_path('model_weights'))           # Load model if load_model flag set to true

    roller = Roller(env, model, params.trainer.steps_per_epoch,
                    params.trainer.gamma, params.trainer.lam)               # Define Roller for genrating rollouts for training

    ppo = PolicyCombinedLoss(model=model, num_envs=env.num_envs)            # Define PPO Policy with combined loss

    for epoch in range(params.trainer.epochs):                              # Main training loop for n epochs

        rollouts, infos = roller.rollout()                                  # Get Rollout and infos
        outs = ppo.update(rollouts)                                         # Push rollout in ppo and update policy accordingly

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
            model.save_weights( LOGGER.tf_model_path('model_weights'), 
                                save_format='tf')                           # Saving model-weights every n steps

        LOGGER.log_metrics(epoch)                                           # Push metrics to screen

    env.close()                                                             # Dont forget closing the environment
