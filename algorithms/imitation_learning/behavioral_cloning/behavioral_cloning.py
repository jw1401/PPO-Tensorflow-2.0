import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from algorithms.ppo import CategoricalModel, PolicyBase, Params
from utils.logger import log
from .config import BehavioralCloningParams


class BehavioralCloning(PolicyBase):

    """
    ##  Behavioral Cloning

        Behavioral Cloning Implementation --> Works only for CATEGORICAL models

        Maps Observations to actions in a supervised learning setting with Categorical Cross Entropy
    ##
    """

    def __init__(self, model, env_info, **kwargs):
        super().__init__(env_info= env_info)

        self.env_info = env_info
        self.pi = model                                                         # pi = _actor network from nn_architectures

        self.batch_size_bc = BehavioralCloningParams.batch_size_bc              # Behavioral Cloning Arguments
        self.iters_bc = BehavioralCloningParams.iters_bc
        self.lr = BehavioralCloningParams.lr

        self.optimizer_pi = tf.keras.optimizers.Adam(learning_rate= self.lr)

    def update(self, vec_obs, vis_obs, acts): 
        
        for _ in range(self.iters_bc):
            loss = self._train_one_step(vec_obs, vis_obs, acts)
                
        return loss
    
    def _train_one_step(self, vec_obs, vis_obs, acts):

        with tf.GradientTape() as tape:
            logits, values = self.pi({'vec_obs': vec_obs, 'vis_obs': vis_obs})
            loss = self._loss(logits, acts)
            
        trainable_variables = self.pi.all_networks[0].trainable_variables       # index 0 is the actor network
        grads = tape.gradient(loss, trainable_variables)
        grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)

        self.optimizer_pi.apply_gradients(zip(grads, trainable_variables))

        return loss
    
    def _loss(self, logits, acts):
        """
            Behavioral Cloning Loss with Cross Entropy
        """
        acts = tf.cast(acts, tf.int32)
        labels_one_hot = tf.one_hot(acts, self.env_info.act_size)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels_one_hot, logits, from_logits=True))
        return loss
