import tensorflow as tf
import numpy as np
from environment import EnvInfo
from utils import log

EPS = 1e-8      # Const to avoid Machine Precission Error


class CategoricalModel(tf.keras.Model):
    """
        Categorical Model for Discrete Action Spaces
        --------------------------------------------

            Input:

                Network with foward pass and EnvInfo Object

            Returns:   

                call:                   logits, values from Neural Network 
                                        with defined forward pass

                get-action-lop-value:   logp, action, value 
                                        (action drawn from Random Categ. Dist)

                logp:                   Log probability for action x

                entropy:                Entropy Term from logits

    """
    def __init__(self, network= None, env_info= EnvInfo):
        super().__init__('CategoricalModel')

        self.env_info = env_info
        self.act_size = self.env_info.act_size                              # Number of possible actions
        self.forward = network['forward']                                   # Get feed forward chain
        self.all_networks = network['trainable_networks']                   # Get all trainable networks
  
    def pd(self, logits):
        return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)        # Draw from Random Categorical Distribution

    # @tf.function
    def call(self, inputs):
        return self.forward(inp = inputs) 
        
    def get_action_logp_value(self, obs):
        """
            Returns:

            logits --> Last layer of Neural Network without activation function \n
            logp --> SOFTMAX of logits which squashes logits between 0 .. 1 and returns log probabilities \n
            actions --> drawn from normal distribution
        """
        logits, values = self.predict(obs)                                  # Returns np arrays on predict | Input: np array or tensor or list
        actions = self.pd(logits)
        logp_t = self.logp(logits, actions) 
        return np.squeeze(actions), np.squeeze(logp_t), np.squeeze(values) 

    def logp(self, logits, action):
        """
            Returns:
            
            logp based on the action drawn from prob-distribution \n
            indexes in the logp_all with one_hot
        """
        logp_all = tf.nn.log_softmax(logits)
        one_hot = tf.one_hot(action, depth= self.act_size)
        logp = tf.reduce_sum( one_hot * logp_all, axis= -1)
        return logp
        
    def entropy(self, logits= None):
        """
            Entropy term for more randomness which means more exploration \n
            Based on OpenAI Baselines implementation
        """
        a0 = logits - tf.reduce_max(logits, axis= -1, keepdims=True)
        exp_a0 = tf.exp(a0)
        z0 = tf.reduce_sum(exp_a0, axis= -1, keepdims=True)
        p0 = exp_a0 / z0
        entropy = tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis= -1)
        return entropy


class GaussianModel(tf.keras.Model):
    """
        Gaussian Model for Continuous Action Spaces
        -------------------------------------------
            
            Input:

                Network and Forward Pass defintion

            Returns:

                call:                   MU, Values, 
                get-action-logp-value:  Log probs and Entropy
                entropy:                Entropy term
            
    """
    def __init__(self, network= None, env_info= EnvInfo):
        super().__init__('GaussianModel')

        self.env_info = env_info
        self.act_size = self.env_info.act_size                                  # Number of possible Continuous actions
        self.forward = network['forward']
        self.all_networks = network['trainable_networks']

        self.log_std = tf.Variable( name= 'LOG_STD', initial_value= -0.5 * 
                                    np.ones(self.act_size, dtype= np.float32))  # Standard Deviation in Normal Distribution is a trainable variable and is updated by the optimizer
        
    # @tf.function
    def call(self, inputs):
        return self.forward(inp = inputs) 
        
    def get_action_logp_value(self, obs):
        """
            Get Action and logarithmic probability on action at 
            Environment-Step t. Approximate mu from a Neural Network 
            
            Model a Gaussian distribution with mu and standard deviation (std) 
            where action is sampled from a Normal distribution  
            
            ## mu + random_normal * std

            Last calculate log probs at step t which is logp_old for PPO Update 
        """
        mu, values = self.predict(obs)                                          # Get mu from NN
        std = tf.exp(self.log_std)                                              # Take exp. of Std deviation
        action = mu + tf.random.normal(tf.shape(mu)) * std                      # Sample action from Gaussian Dist
        action = tf.clip_by_value(action, -1, 1)                                # Clip Continuous actions in range of [-1, 1] 
        logp_t = self.logp(mu, action)                                          # Calculate logp at timestep t for actions

        return np.squeeze(action), np.squeeze(logp_t), np.squeeze(values)

    def logp(self, mu, action):
        return self.gaussian_likelihood(action, mu, self.log_std)

    def gaussian_likelihood(self, x, mu, log_std):
        """
            calculate the liklihood logp of a gaussian distribution 
            for parameters x given the variables mu and log_std
        """
        pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS))**2 + 2 * log_std + np.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis= -1)

    def entropy(self, empty= None):
        """
            Entropy term for more randomness 
            which means more exploration in ppo 
        """
        entropy = tf.reduce_sum(self.log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)
        return entropy
