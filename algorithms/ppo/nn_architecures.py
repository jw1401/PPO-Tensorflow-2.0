"""
    NN-Architectures for PPO
    ------------------------

    Define neural network architectures in this file:

    1. Build CNN, MLPs, Layers wathever you want
    2. Define the forward pass for the Architecture from Input to Output
    3. Use the Architecture in the models.py file (is handed over in the runner files)

    Hints and Tricks:
    -----------------

        - Tanh -->  Should uses GLOROT UNIFORM (XAVIER UNIFORM) initialization 
        - RELU -->  Should uses HE NORMAL initalization 
        - RELU -->  avoids vanishing gradients but can produce dead neurons and bias 
                    LeakyRELU --> trys to avoid dead neurons
"""

import tensorflow as tf
import numpy as np
from environment import EnvInfo
from utils import log

mapping = {}


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk


def mlp(hidden_sizes= (32, 32), output_size= 1, activation= 'relu', activation_output= None, kernel_initalizer= 'glorot_uniform', name= 'MLP'):
    """
        MLP - Multilayer Perceptron
        ---------------------------

            Hidden Sizes = [32,32] Size of HIDDEN Layers 
            Output Size = (1) Size of OUTPUT Layer
            Activation = RELU
            Output Activation  = NONE
            Kernel Initializer = glorot uniform
            bias inintializer =  ZEROS

    """
    model = tf.keras.Sequential(name= name)
    
    for h in hidden_sizes:
        model.add(tf.keras.layers.Dense(units= h, activation= activation, name= name, kernel_initializer= kernel_initalizer, bias_initializer= 'zeros'))
    
    model.add(tf.keras.layers.Dense(units= output_size, activation= activation_output, name= name + '_output'))

    return model


def cnn_test(activation='elu', kernel_initializer = 'glorot_uniform'):
    """
        Convolutinal Test Network 
        -------------------------

            With 2 Convolutions and Max Pool - Without BatchNorm and Dropout
            Output is a flattened layer 
            Activation: ELU
    
    """
    model = tf.keras.Sequential(name= 'cnn_test')
    model.add(tf.keras.layers.Conv2D(filters= 32, kernel_size= (5, 5), strides= 1, padding= 'valid', activation= activation, kernel_initializer = kernel_initializer, name= 'cnn_test'))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(filters= 64, kernel_size= (3, 3), strides= 2, padding= 'valid', activation= activation, kernel_initializer = kernel_initializer))
    model.add(tf.keras.layers.MaxPool2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    return model


def cnn_simple(activation='relu', init= 'glorot_uniform', out_units= 256):
    """
        Simple Convolutional Network
        ----------------------------

            Use activation =    tf.keras.layers.LeakyReLU() as activation if getting weird behavior 
                                allows negative values for backprop

            out-units :         Output Units
                                Note that the last layer has a output size of [256] with Tanh activation
    
    """
    model = tf.keras.Sequential(name= 'simple_cnn')
    model.add(tf.keras.layers.Conv2D(filters= 16, kernel_size= (3, 3), strides= 1, padding= 'valid', activation= activation, kernel_initializer = init, name= 'simple_cnn'))
    model.add(tf.keras.layers.MaxPool2D(3, 2))
    model.add(tf.keras.layers.Conv2D(filters= 32, kernel_size= (3, 3), strides= 1, padding= 'valid', activation= activation, kernel_initializer = init))
    model.add(tf.keras.layers.MaxPool2D(3, 2))
    model.add(tf.keras.layers.Conv2D(filters= 32, kernel_size= (3, 3), strides= 1, padding= 'valid', activation= activation, kernel_initializer = init))
    model.add(tf.keras.layers.MaxPool2D(3, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units= out_units, activation = 'tanh', kernel_initializer = 'glorot_uniform'))
    return model, out_units


@register("cnn_simple_actor_critic")
def cnn_simple_actor_critic(    hidden_sizes= (32, 32), activation= 'relu', activation_output= None, 
                                kernel_initalizer= 'glorot_uniform', name= 'cnn_simple_actor_critic', env_info= EnvInfo):

    cnn, _ = cnn_simple()

    _actor = tf.keras.Sequential(name= 'actor')
    _critic = tf.keras.Sequential(name= 'critic')
    _actor.add(cnn)
    _critic.add(cnn)

    _mlp_actor = mlp(   hidden_sizes= hidden_sizes, output_size= env_info.act_size, activation= activation, 
                        activation_output= activation_output, name= name, kernel_initalizer= kernel_initalizer)

    _actor.add(_mlp_actor)

    _mlp_critic = mlp(  hidden_sizes= hidden_sizes, output_size= 1, activation= activation, 
                        activation_output= activation_output, name= name, kernel_initalizer= kernel_initalizer)

    _critic.add(_mlp_critic)

    log('Model Summary: ' + name)
    _actor.build(input_shape = (None,) + env_info.shapes['vis'])
    _actor.summary()
    _critic.build(input_shape = (None,) + env_info.shapes['vis'])
    _critic.summary()

    def forward(inp= None):
        logits = _actor(inp['vis_obs'])
        values = _critic(inp['vis_obs'])
        return logits, values

    return {"forward": forward, "trainable_networks": [_actor, _critic]}


@register("simple_actor_critic")
def simple_actor_critic( hidden_sizes= (32, 32), activation= 'relu', activation_output= None, 
                         kernel_initalizer= 'glorot_uniform', name= 'simple_actor_critic', env_info= EnvInfo):

    _actor = mlp(   hidden_sizes= hidden_sizes, output_size= env_info.act_size, activation= activation, 
                    activation_output= activation_output, name= name, kernel_initalizer= kernel_initalizer)
    
    _critic = mlp(  hidden_sizes= hidden_sizes, output_size= 1, activation= activation, 
                    activation_output= activation_output, name= name, kernel_initalizer= kernel_initalizer)

    log('Model Summary: ' + name)

    _actor.build(input_shape = (None,) + env_info.shapes['vec'])
    _actor.summary()

    _critic.build(input_shape = (None,) + env_info.shapes['vec'])
    _critic.summary()

    def forward(inp= None):
        logits = _actor(inp['vec_obs'])
        values = _critic(inp['vec_obs'])
        return logits, values

    return {"forward": forward, "trainable_networks": [_actor, _critic]}


@register("vis_vec_actor_critic")
def vis_vec_actor_critic (  hidden_sizes= (32, 32), activation= 'relu', activation_output= None, 
                            kernel_initalizer= 'glorot_uniform', name= 'vis_vec_actor_critic', env_info= EnvInfo):

    cnn, out_units = cnn_simple()

    _mlp_actor = mlp(   hidden_sizes= hidden_sizes, output_size= env_info.act_size, activation= activation, 
                        activation_output= activation_output, name= name, kernel_initalizer= kernel_initalizer)

    _mlp_critic = mlp(  hidden_sizes= hidden_sizes, output_size= 1, activation= activation, 
                        activation_output= activation_output, name= name, kernel_initalizer= kernel_initalizer)
    
    log('Model Summary: ' + name)

    cnn.build(input_shape = (None,) + env_info.shapes['vis'])
    cnn.summary()
    _mlp_actor.build(input_shape= (None, env_info.shapes['vec'][0] + out_units))
    _mlp_actor.summary()
    _mlp_critic.build(input_shape= (None, env_info.shapes['vec'][0] + out_units))
    _mlp_critic.summary()

    def forward(inp= None):
        out_cnn = cnn(inp['vis_obs'])                               # Convolutional Network for visuals 
        # out_vec_mlp = _mlp_vec(inp['vec_obs])                     # Put vec_obs thorugh Neural Network if to much features 
        mixed = tf.concat([out_cnn, inp['vec_obs']], -1)            # Concatenate cnn and vec_obs or out_vec_mlp
        # out_mixer_mlp = _mlp_mixer(mixed)                         # state mixer with Neural Network if needed
        logits = _mlp_actor(mixed)                                  # Feed with raw mixed or with out_mixer_mlp
        values = _mlp_critic(mixed)
        return logits, values

    return {"forward": forward, "trainable_networks": [cnn, _mlp_actor, _mlp_critic]}


def network_builder(name):
    """
        If you want to register your own network outside models.py.

        Usage Example:
        -------------

        import register

        @register("your_network_name")

        def your_network_define(**net_kwargs): return network_fn
    """
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Unknown network type: {}'.format(name))
