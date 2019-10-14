from dataclasses import dataclass, field
import yaml
import os
import datetime


def _(obj):
    return field(default_factory=lambda: obj)


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


@dataclass
class ParamsBase:
    
    working_dir: str = ""                                           # Working Dir Path
    config_file: str = ""                                           # Config.yaml to load
    
    def __post_init__(self):

        from utils import log2logger

        if self.config_file is not "":                              # If no config file --> use parameters defined in this class

            log2logger("Loading " + self.config_file + " from " + self.working_dir)

            try:
                with open(self.working_dir + self.config_file) as file:
                    data = yaml.safe_load(file)
                
                log2logger("LOADED PARAMETERS FROM " + self.config_file)
                return True, data

            except Exception as ex:
                log2logger("USE STANDARD VARIABLES")
                log2logger("Error >> " + str(ex))
                return False, None
        else:
            log2logger("USE STANDARD VARIABLES")
            return False, None


@dataclass
class EnvParams:

    num_envs: int = 2
    env_name: str = ""                                              # or "./Path/to/Environment.exe"
    seed: int = 0                                                   # must be same in trainer configs
    frame_stacking: bool = False
    frames_stack_size: int = 3


@dataclass
class TrainParams:

    trainer: str = "Proximal Policy Optimization"
    nn_architecure: str = "vis_vec_actor_critic"                    # choose from architecture defined in nn_architectures
    epochs: int = 1000                                              # Number of epochs to run
    steps_per_epoch: int = 250                                      # Steps per epoch
    num_mini_batches: int = 4                                       # Num mini batches per grad. descent step
    gamma: float = 0.99                                             # discount factor
    lam: float = 0.97                                               # lambda factor for GAE
    seed: int = EnvParams.seed                                      # starting seed for tensorflow and Env
    training: bool = True                                           # Training mode
    load_model: bool = False                                        # Load the Model yes/no
    save_freq: int = 5                                              # Save frequemcy


@dataclass
class PolicyParams:

    lr: float = 0.001                                               # learning rate
    train_iters: int = 5                                            # pi and v update iterations for backprop
    hidden_sizes: list = _([32, 32])                                # network sizes
    clip_ratio: float = 0.2                                         # clip for ppo
    target_kl: float = 0.01                                         # kullback leibler divergence
    ent_coef: float = 0.1                                           # entropy for exploration in Mult Env divide by n_envs
    v_coef: float = 0.5                                             # Value coef for backprop
    clip_grads: float = 0.5                                         # value for gradient clipping


@singleton
@dataclass
class Params(ParamsBase):

    trainer: TrainParams = TrainParams()
    policy: PolicyParams = PolicyParams()
    env: EnvParams = EnvParams()

    def __post_init__(self):
        
        _has_config_file, data = super().__post_init__()

        if _has_config_file:
            self.trainer = TrainParams(**data['train_params'])
            self.policy = PolicyParams(**data['policy_params'])
            self.env = EnvParams(**data['env_params'])
