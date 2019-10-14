import tensorflow as tf
import numpy as np
from .models import CategoricalModel
from utils import logger, log
from .cfg_ppo import Params


class PolicyBase():

    def __init__(self, env_info= None):

        self.env_info = env_info
        self.params = Params()
        self.lr = self.params.policy.lr
        self.train_iters = self.params.policy.train_iters
        self.clip_ratio = self.params.policy.clip_ratio
        self.target_kl = self.params.policy.target_kl
        self.ent_coef = self.params.policy.ent_coef
        self.v_coef = self.params.policy.v_coef
        self.clip_grads = self.params.policy.clip_grads

    
class PolicyCombinedLoss(PolicyBase):
    
    """
        Proximal Policy Optimization
        ----------------------------

        https://arxiv.org/pdf/1707.06347.pdf
        https://openai.com/blog/openai-baselines-ppo/
    """

    def __init__(self, model= CategoricalModel, env_info= None, num_envs= 1):
        super().__init__(env_info = env_info)

        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate= self.lr)

        self.num_envs = num_envs
        self.nbatch = self.num_envs * self.params.trainer.steps_per_epoch               # ==> HORIZON = steps per epoch
        self.nbatch_train = self.nbatch // self.params.trainer.num_mini_batches
        assert self.nbatch % self.params.trainer.num_mini_batches == 0

    def update(self, rollouts):
        """
            Update Policy and the Value Network
            -----------------------------------
                Inputs: obs, act, advantages, returns, logp-t
                Returns: loss-pi, loss-entropy, approx-ent, kl, loss-v, loss-total
        """
        inds = np.arange(self.nbatch)

        for i in range(self.train_iters):

            losses = self._inner_update_loop(   rollouts['vec_obses'], 
                                                rollouts['vis_obses'], 
                                                rollouts['actions'], 
                                                rollouts['advs'], 
                                                rollouts['returns'], 
                                                rollouts['logp'], 
                                                inds) 

            if losses['approx_kl'] > 1.5 * self.target_kl:
                log("Early stopping at step %d due to reaching max kl." %i)
                break
        
        return losses                                                                   # Return Metrics
        
    def _inner_update_loop(self, vec_obses, vis_obses, actions, advs, returns, logp_t, inds):
        """
            Make updates with random sampled minibatches and 
            return mean kl-div for early breaking
        """
        np.random.shuffle(inds)
        means = []

        for start in range(0, self.nbatch, self.nbatch_train):

            end = start + self.nbatch_train
            slices = inds[start:end]
            losses = self._train_one_step(  vec_obses[slices] if vec_obses is not None else None, 
                                            vis_obses[slices] if vis_obses is not None else None, 
                                            actions[slices], 
                                            advs[slices], 
                                            logp_t[slices], 
                                            returns[slices])

            means.append([  losses['pi_loss'], 
                            losses['v_loss'], 
                            losses['entropy_loss'], 
                            losses['approx_ent'], 
                            losses['approx_kl']])                                       # keep order in list for return later
        
        means = np.asarray(means)
        means = np.mean(means, axis= 0)

        return {    'pi_loss': means[0], 
                    'v_loss': means[1],
                    'entropy_loss': means[2], 
                    'approx_ent': means[3], 
                    'approx_kl': means[4]}

    def _loss(self, vec_obs, vis_obs, logp_old, act, adv, returns):

        pi, values = self.model({"vec_obs": vec_obs, "vis_obs": vis_obs})               # output from neural network are logits or mu 
                                                                                        # dependent on the policy(gaussian, categorical)

        logp = self.model.logp(pi, act)                                                 # PPO Objective 
        ratio = tf.exp(logp - logp_old)
        min_adv = tf.where(adv > 0, (1 + self.clip_ratio) * adv, (1 - self.clip_ratio) * adv)

        clipped_loss = -tf.reduce_mean(tf.math.minimum(ratio * adv, min_adv))           # Policy Loss = loss_clipped + entropy_loss * entropy_coef
                                                                                        # losses have negative sign for maximizing via backprop

        entropy = self.model.entropy(pi)                                                # Entropy loss - Categorical Policy --> returns entropy based on logits
        entropy_loss = -tf.reduce_mean(entropy)

        pi_loss = clipped_loss + entropy_loss * self.ent_coef                           # Policy Loss 

        approx_kl = tf.reduce_mean(logp_old - logp)                                     # Approximated  Kullback Leibler Divergence from OLD and NEW Policy
        approx_ent = tf.reduce_mean(-logp) 

        v_loss = 0.5 * tf.reduce_mean(tf.square(returns - values))                      # Value Network Loss

        total_loss = pi_loss + v_loss * self.v_coef                                     # Total Loss = pi_loss + value loss * v_coef
        
        return {    'pi_loss': pi_loss, 
                    'entropy_loss': entropy_loss, 
                    'approx_ent': approx_ent, 
                    'approx_kl': approx_kl, 
                    'v_loss': v_loss, 
                    'total_loss': total_loss}

    # @tf.function
    def _train_one_step(self, vec_obs, vis_obs, act, adv, logp_old, returns):

        with tf.GradientTape() as tape:
            _losses = self._loss(vec_obs, vis_obs, logp_old, act, adv, returns)
            
        trainable_variables = self.model.trainable_variables                            # take all trainable variables into account

        grads = tape.gradient(_losses['total_loss'], trainable_variables)               # differentiate --> dError/dVariables --> 
                                                                                        # dTotal_Loss/dTrainable_Variables used 
                                                                                        # for calculating gradients for backprop

        grads, grad_norm = tf.clip_by_global_norm(grads, self.clip_grads)               # clip gradients for slight updates

        self.optimizer.apply_gradients(zip(grads, trainable_variables))                 # Backprop gradients through network

        return _losses
